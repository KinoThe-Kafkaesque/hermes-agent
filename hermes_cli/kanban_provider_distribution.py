"""Dispatch-time provider/model distribution for Kanban cards.

The dispatcher used to spawn every card on its assignee profile's configured
model, whatever the provider's quota state. A single quota wall on one provider
therefore serialized a whole lane: every requeue re-hit the same wall, and the
board could not even say which provider had failed, because ``task_runs``
recorded only the exit code.

Two mechanisms live here:

1. **Attribution.** ``resolve_provider_route`` picks the route a card starts on
   (its explicit pin, else the profile default plus a distribution chain), and
   ``parse_exit_trailer`` reads the provider/model a worker actually reported at
   exit. Both sides use this one vocabulary so a wall is attributable to a named
   provider/model.

2. **Demotion.** ``record_rate_limit`` / ``walled_routes`` keep a bounded,
   process-local tally of recent rate-limit walls per ``(provider, model)``. A
   route over the threshold is DEMOTED, never blocked: ``select_route`` walks
   the chain past a walled route and falls back to the primary when every entry
   is walled, so a lane can never deadlock on an exhausted provider.

The tally is deliberately in-process. The dispatcher that reaps a worker is the
one that requeues it, so the tick that observes a wall is the one that needs to
know; persisting a windowed aggregate would add a table for a signal whose
useful lifetime is minutes. Consequence, stated plainly: the tally is per
dispatcher process, so a restart clears it and a lane returns to its primary
until the next wall is observed. That is the fail-open direction.
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

# A route the dispatcher can place a card on. ``base_url`` is carried so a chain
# entry that targets a specific endpoint keeps it; ``reasoning`` is the per-tier
# effort resolved for that model (see ``resolve_per_model_reasoning_effort``).
@dataclass(frozen=True)
class ProviderRoute:
    provider: str
    model: str
    base_url: str = ""
    reasoning: str = ""

    @property
    def identity(self) -> tuple[str, str]:
        """Tally key. Provider+model, not base_url: a quota wall is per account/model."""
        return (self.provider.strip().lower(), self.model.strip().lower())

    def describe(self) -> str:
        return f"{self.provider}/{self.model}" if self.provider else (self.model or "unknown")


# Default distribution order for a profile with no ``fallback_providers`` of its
# own. Mirrors the gomu pattern that the live data shows is the only profile
# never hitting a wall (0/14 runs): opencode-go first, then nous, then workbuddy.
# Ordered by measured health on this fleet, not by preference.
DEFAULT_DISTRIBUTION_CHAIN: tuple[ProviderRoute, ...] = (
    ProviderRoute("opencode-go", "step-5-preview-free"),
    ProviderRoute("nous", "stepfun/step-5-preview:free"),
    ProviderRoute("workbuddy", "deepseek-v4.1-flash-1m"),
)

# Per-tier reasoning effort, reused rather than reinvented: the spelling-tolerant
# resolver owns precedence (exact, dots<->dashes, prefix-stripped).
DEFAULT_REASONING_OVERRIDES: dict[str, str] = {
    "step-5-preview-free": "xhigh",
    "stepfun/step-5-preview:free": "xhigh",
    "deepseek-v4.1-flash-1m": "high",
}

# Demotion threshold: this many rate-limit walls for one route inside
# ``DEMOTION_WINDOW_SECONDS`` demotes it for the next spawns.
DEFAULT_DEMOTION_THRESHOLD = 2
DEFAULT_DEMOTION_WINDOW_SECONDS = 900

_WALL_LOCK = threading.Lock()
# ``(provider, model) -> [wall epoch, ...]``, newest last, pruned on write.
_wall_events: dict[tuple[str, str], list[float]] = {}
_WALL_EVENTS_MAX_ROUTES = 256


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def demotion_threshold() -> int:
    """Walls-per-window that demote a route. 0 disables demotion entirely."""
    return _env_int("HERMES_KANBAN_PROVIDER_DEMOTION_THRESHOLD", DEFAULT_DEMOTION_THRESHOLD)


def demotion_window_seconds() -> int:
    return _env_int("HERMES_KANBAN_PROVIDER_DEMOTION_WINDOW", DEFAULT_DEMOTION_WINDOW_SECONDS)


def _per_model_reasoning(model: str, overrides: Optional[dict]) -> str:
    """Reasoning effort for *model*, reusing the shared spelling-tolerant resolver.

    Imported lazily and defensively: this module is on the dispatcher's import
    path, and a reasoning default must never be the reason a card fails to spawn.
    """
    if not model:
        return ""
    try:
        from hermes_constants import resolve_per_model_reasoning_effort

        resolved = resolve_per_model_reasoning_effort(model, overrides or DEFAULT_REASONING_OVERRIDES)
    except Exception:
        return ""
    if isinstance(resolved, dict):
        return str(resolved.get("effort") or "").strip()
    if isinstance(resolved, str):
        return resolved.strip()
    return ""


def normalize_route(provider: Any, model: Any, base_url: Any = "") -> Optional[ProviderRoute]:
    """``(provider, model)`` -> ``ProviderRoute``; None when either side is missing."""
    prov = str(provider or "").strip()
    mod = str(model or "").strip()
    if not prov or not mod:
        return None
    return ProviderRoute(prov, mod, str(base_url or "").strip())


def profile_default_route(profile: Optional[str]) -> Optional[ProviderRoute]:
    """The assignee profile's configured ``model.default`` / ``model.provider``.

    Reads the profile's own config under its home, not the dispatcher's ambient
    config: the dispatcher runs detached from any profile, so an unbound
    ``load_config`` would hand every card the launch profile's model.
    """
    if not profile:
        return None
    try:
        from hermes_cli.profiles import normalize_profile_name, resolve_profile_env

        home = resolve_profile_env(normalize_profile_name(profile))
    except Exception:
        return None
    if not home:
        return None
    try:
        from hermes_cli.config import load_config
        from hermes_constants import get_process_hermes_home, reset_hermes_home_override, set_hermes_home_override

        token = set_hermes_home_override(str(home))
        try:
            cfg = load_config() or {}
        finally:
            reset_hermes_home_override(token)
    except Exception:
        return None
    model_cfg = (cfg.get("model") or {}) if isinstance(cfg, dict) else {}
    if not isinstance(model_cfg, dict):
        return None
    return normalize_route(model_cfg.get("provider"), model_cfg.get("default"), model_cfg.get("base_url"))


def profile_fallback_chain(profile: Optional[str], primary: Optional[ProviderRoute]) -> list[ProviderRoute]:
    """The profile's own ``fallback_providers``, then the default distribution tail.

    An empty chain is the common case (5 of 6 worker profiles ship ``[]``) and is
    exactly what this feature fixes: the default tail is appended so a profile
    with no chain of its own still distributes. Entries duplicating the primary
    are dropped, and the chain is capped so a pathological config cannot grow the
    spawn argv without bound.
    """
    entries: list[ProviderRoute] = []
    if profile:
        try:
            from hermes_cli.fallback_config import get_fallback_chain

            home = None
            try:
                from hermes_cli.profiles import normalize_profile_name, resolve_profile_env

                home = resolve_profile_env(normalize_profile_name(profile))
            except Exception:
                home = None
            cfg: dict[str, Any] = {}
            if home:
                from hermes_cli.config import load_config
                from hermes_constants import reset_hermes_home_override, set_hermes_home_override

                token = set_hermes_home_override(str(home))
                try:
                    cfg = load_config() or {}
                finally:
                    reset_hermes_home_override(token)
            for entry in get_fallback_chain(cfg):
                route = normalize_route(entry.get("provider"), entry.get("model"), entry.get("base_url"))
                if route is not None:
                    entries.append(route)
        except Exception:
            entries = []
    # A profile whose own chain already covers the default tail needs no more.
    seen = {r.identity for r in entries}
    if primary is not None:
        seen.add(primary.identity)
    for route in DEFAULT_DISTRIBUTION_CHAIN:
        if route.identity not in seen:
            entries.append(route)
    return entries[:_MAX_CHAIN_ENTRIES]


_MAX_CHAIN_ENTRIES = 8


def build_chain(profile: Optional[str], pinned: Optional[ProviderRoute]) -> list[ProviderRoute]:
    """Ordered chain for a card: the pinned/profile primary first, then the tail.

    The primary is the route the card WOULD have run on before this feature; it
    stays first so a healthy fleet spawns exactly as today. Demotion happens at
    selection time, not here — this is the preference order, not the decision.
    """
    primary = pinned or profile_default_route(profile)
    chain: list[ProviderRoute] = []
    if primary is not None:
        primary = ProviderRoute(
            primary.provider, primary.model, primary.base_url,
            _per_model_reasoning(primary.model, DEFAULT_REASONING_OVERRIDES))
        chain.append(primary)
    seen = {primary.identity} if primary is not None else set()
    for route in profile_fallback_chain(profile, primary):
        if route.identity in seen:
            continue
        seen.add(route.identity)
        chain.append(ProviderRoute(
            route.provider, route.model, route.base_url,
            _per_model_reasoning(route.model, DEFAULT_REASONING_OVERRIDES)))
    return chain


def _prune_walls(now: float, window: float) -> None:
    """Drop expired walls; drop whole routes once their history is empty. Caller holds the lock."""
    if not _wall_events:
        return
    cutoff = now - max(window, 1)
    for identity in list(_wall_events):
        kept = [ts for ts in _wall_events[identity] if ts >= cutoff]
        if kept:
            _wall_events[identity] = kept
        else:
            _wall_events.pop(identity, None)
    # Bounded: a long-lived dispatcher must not grow this map without limit.
    while len(_wall_events) > _WALL_EVENTS_MAX_ROUTES:
        oldest = min(_wall_events, key=lambda k: _wall_events[k][-1])
        _wall_events.pop(oldest, None)


def record_rate_limit(provider: Any, model: Any) -> int:
    """Note one rate-limit wall for ``(provider, model)``; returns its count in-window."""
    route = normalize_route(provider, model)
    if route is None:
        return 0
    now = time.time()
    window = demotion_window_seconds()
    with _WALL_LOCK:
        _prune_walls(now, window)
        stamps = _wall_events.setdefault(route.identity, [])
        stamps.append(now)
        return len(stamps)


def walled_routes(now: Optional[float] = None) -> set[tuple[str, str]]:
    """Routes at or over the demotion threshold inside the window."""
    threshold = demotion_threshold()
    if threshold <= 0:
        return set()
    now = time.time() if now is None else now
    with _WALL_LOCK:
        _prune_walls(now, demotion_window_seconds())
        return {ident for ident, stamps in _wall_events.items() if len(stamps) >= threshold}


def walled_route_details(now: Optional[float] = None) -> dict[tuple[str, str], int]:
    """``{identity: wall count}`` for demoted routes, for run metadata / logs."""
    threshold = demotion_threshold()
    if threshold <= 0:
        return {}
    now = time.time() if now is None else now
    with _WALL_LOCK:
        _prune_walls(now, demotion_window_seconds())
        return {ident: len(s) for ident, s in _wall_events.items() if len(s) >= threshold}


def reset_demotion_state() -> None:
    """Clear the tally. Test seam and the documented restart behaviour."""
    with _WALL_LOCK:
        _wall_events.clear()


def select_route(chain: Iterable[ProviderRoute], *, walled: Optional[set] = None,
                 skip: Optional[set] = None) -> tuple[Optional[ProviderRoute], list[str]]:
    """Pick the first healthy route in *chain*; ``(route, reasons)``.

    ``walled`` routes are demoted (skip-forward) and ``skip`` identities are
    excluded outright (failover on requeue). Preference, never a block: when
    every entry is unhealthy the PRIMARY is returned so the lane still spawns
    rather than deadlocking. Reasons explain what was skipped and why.
    """
    walled = walled if walled is not None else walled_routes()
    skip = skip or set()
    routes = [r for r in chain if r.provider and r.model]
    if not routes:
        return None, ["no provider route configured"]
    reasons: list[str] = []
    fallback: Optional[ProviderRoute] = None
    for index, route in enumerate(routes):
        if route.identity in skip:
            reasons.append(f"skip {route.describe()} (failed over away from it)")
            continue
        if route.identity in walled:
            reasons.append(f"demote {route.describe()} (rate-limit wall)")
            if fallback is None:
                fallback = routes[0]
            continue
        if reasons:
            reasons.insert(0, f"start on {route.describe()}")
        else:
            reasons.append(f"start on {route.describe()}")
        return route, reasons
    # Everything unhealthy: spawn on the primary. Demotion is a preference.
    primary = routes[0]
    reasons.append(f"all routes unhealthy; fail-open on primary {primary.describe()}")
    return primary, reasons


# --- Exit-trailer parsing (worker side vocabulary, dispatcher side reading) ---

# ``[kanban-worker-exit] rc=75 provider=opencode-go model=step-5-preview-free``
# ``provider=`` / ``model=`` are optional so an OLD worker binary (pre-attribution)
# still parses: the rc is the load-bearing field and the provider reads as None
# rather than the run failing to classify.
_TRAILER_RE = re.compile(
    r"^\[kanban-worker-exit\]\s+rc=(\d+)(?P<fields>(?:\s+\w+=\S+)*)\s*$", re.MULTILINE)
_TRAILER_FIELD_RE = re.compile(r"(\w+)=(\S+)")


@dataclass(frozen=True)
class ExitTrailer:
    """What a worker's exit trailer says: the code plus the route that served it."""
    exit_code: int
    provider: Optional[str] = None
    model: Optional[str] = None

    @property
    def route(self) -> Optional[ProviderRoute]:
        return normalize_route(self.provider, self.model)


def parse_exit_trailer(raw: Optional[str]) -> Optional[ExitTrailer]:
    """Last trailer in *raw* -> ``ExitTrailer``; None when absent.

    The log is append-mode across re-runs, so the LAST trailer is this run's.
    """
    if not raw:
        return None
    matches = list(_TRAILER_RE.finditer(raw))
    if not matches:
        return None
    last = matches[-1]
    fields = dict(_TRAILER_FIELD_RE.findall(last.group("fields") or ""))
    return ExitTrailer(
        exit_code=int(last.group(1)),
        provider=(fields.get("provider") or "").strip() or None,
        model=(fields.get("model") or "").strip() or None,
    )


def strip_exit_trailer(raw: Optional[str]) -> str:
    """Remove every trailer from *raw* so log-summary readers never see it."""
    if not raw:
        return raw or ""
    return _TRAILER_RE.sub("", raw)


def trailer_fields(provider: Any, model: Any) -> str:
    """The `` provider=<p> model=<m>`` suffix a worker appends after its rc.

    Empty when either side is unknown: a half-named route is worse than none,
    because ``provider`` without ``model`` would tally a wall against a provider
    the run may not have used.
    """
    route = normalize_route(provider, model)
    if route is None:
        return ""
    return f" provider={route.provider} model={route.model}"
