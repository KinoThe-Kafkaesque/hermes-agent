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

2. **Demotion.** ``record_rate_limit`` / ``walled_routes`` keep a bounded tally
   of recent rate-limit walls per ``(provider, model)``. A route over the
   threshold is DEMOTED, never blocked: ``select_route`` walks the chain past a
   walled route and falls back to the primary when every entry is walled, so a
   lane can never deadlock on an exhausted provider.

The tally is SHARED across dispatcher processes. A wall's durable record is the
``rate_limited`` run row the reclaim transaction already writes, attributed to
the ``provider``/``model`` that served the run, so every dispatcher process
reads the same windowed tally from the board DB — no extra state table, and no
window in which a wall seen by one dispatcher is invisible to another. The
in-process dict stays as the fast path, as the no-database fallback, and as the
source of the run-metadata ``wall_count``.

One wall observed by one process counts ONCE: an in-process stamp and the
persisted row for the same wall are deduplicated by event key
(``<task_id>:<pid>``). Without that, the union would count the recording
process's own wall twice and silently halve the threshold.

The read is fail-open. A tally read that fails degrades to exactly the
in-process tally, and a route-resolution failure leaves ``route=None`` so the
worker uses the profile config — a broken tally must never stop a card from
spawning.
"""

from __future__ import annotations

import json
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

# How far back the SHARED (persisted) tally is read. Deliberately wider than the
# demotion window: ``HERMES_KANBAN_PROVIDER_DEMOTION_WINDOW`` is an operator knob
# and a sibling process may run with a longer window than this one, so the read
# must not clip a wall the reader's own window still covers. The count is always
# taken with THIS process's window; the wider read only bounds the scan.
_WALL_DB_READ_WINDOW_SECONDS = 86400

# ``task_runs.outcome`` for a quota wall (``_DeadWorker.run_outcome``) — the
# durable record of a wall, and the only row shape the shared tally reads.
_RATE_LIMITED_OUTCOME = "rate_limited"

# The shared-tally read. A module constant so the query itself is a seam: the
# fail-open arm is proven by retargeting it at a table that does not exist.
_WALL_QUERY = (
    "SELECT task_id, ended_at, worker_pid, metadata FROM task_runs "
    "WHERE outcome = ? AND ended_at IS NOT NULL AND ended_at >= ? "
    "  AND metadata IS NOT NULL"
)

_WALL_LOCK = threading.Lock()
# ``(provider, model) -> [(wall epoch, event key), ...]``, oldest first, pruned
# on write. The key (``<task_id>:<pid>``) identifies the wall itself, so a stamp
# and the persisted run row for the SAME wall are never counted twice.
_wall_events: dict[tuple[str, str], list[tuple[float, str]]] = {}
_WALL_EVENTS_MAX_ROUTES = 256

# Last shared-tally read failure, for the run metadata and the log. The read is
# fail-open, so this is the only trace a broken tally leaves.
_last_shared_read_error: Optional[str] = None


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _json_dict(raw: Any) -> dict:
    """Decoded JSON object from a metadata column; ``{}`` when absent/unparseable.

    Local rather than imported from ``kanban_db``: this module is on the
    dispatcher's import path and is imported by it, so reaching back into the
    facade here would be a cycle.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:  # health: allow BLE001 -- a corrupt metadata column is not a wall
        return {}
    return parsed if isinstance(parsed, dict) else {}


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


def wall_event_key(task_id: Optional[str], pid: Optional[Any]) -> str:
    """Identity of ONE observed wall: the run that hit it, not the route.

    The in-process stamp and the persisted ``rate_limited`` run row describe the
    same event, so they must carry the same key or the union counts it twice and
    the threshold halves. Both sides derive it from the same two fields the run
    row records (``task_id`` and the worker PID); a missing PID degrades to the
    task id alone, which still deduplicates the common one-run-per-task case.
    """
    return f"{task_id or ''}:{'' if pid is None else pid}"


def _prune_event_walls(now: float, window: float) -> None:
    """In-process prune of the event-keyed tally. Caller holds the lock."""
    if not _wall_events:
        return
    cutoff = now - max(window, 1)
    for identity in list(_wall_events):
        kept = [entry for entry in _wall_events[identity] if entry[0] >= cutoff]
        if kept:
            _wall_events[identity] = kept
        else:
            _wall_events.pop(identity, None)
    while len(_wall_events) > _WALL_EVENTS_MAX_ROUTES:
        oldest = min(_wall_events, key=lambda k: _wall_events[k][-1][0])
        _wall_events.pop(oldest, None)


def _shared_wall_rows(now: float, board: Optional[str] = None,
                      conn: Optional[Any] = None) -> list[Any]:
    """Raw ``rate_limited`` run rows inside the read window; ``[]`` on any failure.

    ``board`` names the board being dispatched. The gateway enumerates boards, so
    without it the read would consult the *current* board while the tick works on
    another one — the tally would silently be about the wrong fleet. ``conn`` is
    an ALREADY-OPEN connection to reuse (the reclaim transaction's own), so the
    write path never opens a second connection to the file it is writing.

    Fail-open by construction: any failure (unreadable DB, missing column, a
    board whose metadata predates the attribution) returns ``[]``, which degrades
    the tally to exactly the in-process one. It never raises: a broken tally must
    not be able to stop a card from spawning.
    """
    global _last_shared_read_error
    try:
        if conn is None:
            from hermes_cli import kanban_db_connect as _kbc

            with _kbc.connect(board=board) as owned:
                return _shared_wall_rows(now, board, owned)
        rows = conn.execute(
            _WALL_QUERY,
            (_RATE_LIMITED_OUTCOME, int(now) - _WALL_DB_READ_WINDOW_SECONDS),
        ).fetchall()
    except Exception as exc:  # health: allow BLE001 -- fail-open tally read boundary
        # Recorded (and surfaced in run metadata) rather than logged: the caller
        # is a spawn path where a broken tally must degrade silently but stay
        # attributable.
        _last_shared_read_error = f"{type(exc).__name__}: {exc}"
        return []
    _last_shared_read_error = None
    return rows


def _shared_wall_events(now: float, board: Optional[str] = None,
                        conn: Optional[Any] = None) -> list[tuple[tuple[str, str], float, str]]:
    """Persisted walls from the board DB: ``[(identity, epoch, event key), ...]``.

    The durable record of a quota wall is the ``rate_limited`` run row the
    reclaim transaction already writes, and ``t_6ed58421`` attributes it to the
    ``provider``/``model`` that served the run — so every dispatcher process can
    read ONE windowed tally without a new state table. Rows the reader cannot
    attribute are skipped, never charged to a guessed route.
    """
    events: list[tuple[tuple[str, str], float, str]] = []
    for row in _shared_wall_rows(now, board, conn):
        meta = _json_dict(row["metadata"])
        # A half-named route is not a route: ``provider`` without ``model`` would
        # tally a wall against a provider the run may not have used.
        route = normalize_route(meta.get("provider"), meta.get("model"))
        ended_at = row["ended_at"]
        if route is None or ended_at is None:
            continue  # an unnamed wall demotes nothing, in any process
        # The event key must match the one the recording process stamped. The
        # reclaim path writes the worker PID into the run metadata payload and
        # stamps the tally with the same value, so the metadata is the
        # authoritative source; ``task_runs.worker_pid`` is the fallback for a
        # run whose spawn never persisted it.
        pid = meta.get("pid")
        if pid is None:
            pid = row["worker_pid"]
        events.append((
            route.identity,
            float(ended_at),
            wall_event_key(row["task_id"], pid),
        ))
    return events


def _wall_counts(now: Optional[float] = None, *, board: Optional[str] = None,
                 shared: bool = True) -> dict[tuple[str, str], int]:
    """``{identity: wall count}`` inside the window, in-process plus shared.

    The two sources describe the same events, so a wall must not be counted
    twice: every in-process stamp counts once, and a persisted row whose event
    key this process already stamped is skipped. Everything else is a union,
    which is what makes a wall recorded in process A visible to process B inside
    the window. With no in-process stamps the result is purely the shared tally,
    so a restarted dispatcher starts demoting immediately instead of re-learning
    the wall.
    """
    now = time.time() if now is None else now
    window = demotion_window_seconds()
    cutoff = now - max(window, 1)
    with _WALL_LOCK:
        _prune_event_walls(now, window)
        local = {ident: list(entries) for ident, entries in _wall_events.items()}
    counts: dict[tuple[str, str], int] = {}
    local_keys: dict[tuple[str, str], set[str]] = {}
    for ident, entries in local.items():
        live = [(ts, key) for ts, key in entries if ts >= cutoff]
        if not live:
            continue
        counts[ident] = len(live)
        local_keys[ident] = {key for _, key in live}
    if shared:
        for ident, ts, key in _shared_wall_events(now, board):
            if ts < cutoff:
                continue
            if key in local_keys.get(ident, ()):
                continue  # this process already stamped this very wall
            counts[ident] = counts.get(ident, 0) + 1
    return counts


def record_rate_limit(provider: Any, model: Any, *, task_id: Optional[str] = None,
                      pid: Optional[Any] = None) -> int:
    """Note one rate-limit wall for ``(provider, model)``; returns its in-process count.

    ``task_id``/``pid`` name the wall itself (see :func:`wall_event_key`), which
    is what lets the READ path (:func:`walled_routes`) union this stamp with the
    persisted run row for the same wall without counting it twice.

    Deliberately no database read here: this is the WRITE path and it runs inside
    the reclaim transaction (``_reclaim_dead_workers`` holds an IMMEDIATE write
    txn when it classifies a dead worker), so a nested connection would contend
    with the writer it is called from. The shared view is read where it is
    needed — at route resolution, which runs outside any transaction.
    """
    route = normalize_route(provider, model)
    if route is None:
        return 0
    now = time.time()
    window = demotion_window_seconds()
    cutoff = now - max(window, 1)
    with _WALL_LOCK:
        _prune_event_walls(now, window)
        stamps = _wall_events.setdefault(route.identity, [])
        stamps.append((now, wall_event_key(task_id, pid)))
        return len([ts for ts, _ in stamps if ts >= cutoff])


def walled_routes(now: Optional[float] = None, *, board: Optional[str] = None) -> set[tuple[str, str]]:
    """Routes at or over the demotion threshold inside the window.

    The threshold applies to the SHARED count, so a wall observed by one
    dispatcher demotes the route for every other dispatcher process.
    """
    threshold = demotion_threshold()
    if threshold <= 0:
        return set()
    return {ident for ident, n in _wall_counts(now, board=board).items() if n >= threshold}


def walled_route_details(now: Optional[float] = None, *,
                         board: Optional[str] = None) -> dict[tuple[str, str], int]:
    """``{identity: wall count}`` for demoted routes, for run metadata / logs."""
    threshold = demotion_threshold()
    if threshold <= 0:
        return {}
    return {ident: n for ident, n in _wall_counts(now, board=board).items() if n >= threshold}


def shared_wall_read_error() -> Optional[str]:
    """Last shared-tally read failure, or None. Fail-open leaves no other trace."""
    return _last_shared_read_error


def reset_demotion_state() -> None:
    """Clear the in-process tally. Test seam and the documented restart behaviour.

    The SHARED tally lives in the board DB and is deliberately not cleared here:
    it is other dispatcher processes' evidence, and a test that needs a clean
    window isolates its board rather than erasing live fleet state.
    """
    with _WALL_LOCK:
        _wall_events.clear()


def select_route(chain: Iterable[ProviderRoute], *, walled: Optional[set] = None,
                 skip: Optional[set] = None, board: Optional[str] = None) -> tuple[Optional[ProviderRoute], list[str]]:
    """Pick the first healthy route in *chain*; ``(route, reasons)``.

    ``walled`` routes are demoted (skip-forward) and ``skip`` identities are
    excluded outright (failover on requeue). Preference, never a block: when
    every entry is unhealthy the PRIMARY is returned so the lane still spawns
    rather than deadlocking. Reasons explain what was skipped and why.

    ``board`` is the board whose shared tally ``walled`` is read from when the
    caller does not supply one (see :func:`walled_routes`).
    """
    walled = walled if walled is not None else walled_routes(board=board)
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


def requeue_failover_skip(task: Any, *, board: Optional[str] = None) -> set:
    """Routes a card should NOT restart on after a ``rate_limited`` requeue.

    Reads the latest closed run's recorded provider/model. Sticky-advance, not
    random: a genuinely-walled provider is skipped rather than retried into the
    same wall, and the advance is bounded (one route) so a card that keeps
    walling walks its chain forward rather than thrashing. ``board`` names the
    board being dispatched — the run row lives in that board's DB.

    Lives here with the rest of the run-route reads: this module owns the
    vocabulary a run's recorded route is expressed in.
    """
    if getattr(task, "current_run_id", None) is None:
        return set()
    meta = latest_run_metadata(task.id, board=board)
    if meta.get("outcome") != _RATE_LIMITED_OUTCOME:
        return set()
    route = normalize_route(meta.get("provider"), meta.get("model"))
    return {route.identity} if route is not None else set()


def latest_run_metadata(task_id: str, *, board: Optional[str] = None) -> dict:
    """``{outcome, provider, model, ...}`` of the latest CLOSED run; {} when none.

    ``provider``/``model`` fall back through both writers: the observed route the
    worker reported at exit, then the route the spawn requested. Fail-open: an
    unreadable board yields ``{}``, which callers read as "no prior route".
    """
    try:
        from hermes_cli import kanban_db_connect as _kbc

        with _kbc.connect(board=board) as conn:
            row = conn.execute(
                "SELECT outcome, metadata FROM task_runs WHERE task_id = ? "
                "AND ended_at IS NOT NULL ORDER BY ended_at DESC, id DESC LIMIT 1",
                (task_id,),
            ).fetchone()
    except Exception:  # health: allow BLE001 -- fail-open read of the latest run
        return {}
    if row is None:
        return {}
    meta = _json_dict(row["metadata"])
    meta["outcome"] = row["outcome"]
    if not meta.get("provider") and meta.get("requested_provider"):
        meta["provider"] = meta["requested_provider"]
        meta["model"] = meta.get("requested_model") or meta.get("model")
    return meta


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
