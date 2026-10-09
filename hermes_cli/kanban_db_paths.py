"""Kanban board path resolution: which root, which board, which file.

Split out of ``hermes_cli.kanban_db`` (whose facade re-exports every name here and reaches the
fence guard late-bound via ``_kb``) so this unit stops growing the facade's ``FILE_LINES`` cap.

The precedence here is load-bearing for production: an explicit ``board=`` argument — or the
scoped ``--board`` context (:func:`scoped_current_board`) — outranks the ``env_var`` pin ONLY
where no fence applies, and the dispatcher injects ``HERMES_KANBAN_DB`` /
``HERMES_KANBAN_WORKSPACES_ROOT`` / ``HERMES_KANBAN_BOARD`` into every worker so its workers
converge on its board.

:func:`isolated_board` is the supported one-call isolation helper for tests and ad-hoc probes. It
sets ``HERMES_KANBAN_HOME`` **and** clears every pin, because ``HERMES_KANBAN_DB`` outranks
``HERMES_KANBAN_HOME`` and the dispatcher injects it into every worker: a probe that set only
``HERMES_KANBAN_HOME`` looked isolated (board dirs, workspaces and logs all landed under its temp
root) while its task writes still went to the LIVE board.
"""

from __future__ import annotations

import contextlib
import os
import re
import tempfile
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Optional


class KanbanBoardPinConflict(RuntimeError):
    """``HERMES_KANBAN_DB`` pins a file that an explicitly-set ``HERMES_KANBAN_HOME`` does not
    contain. Raised instead of silently writing to the pinned board: the caller set
    ``HERMES_KANBAN_HOME`` to isolate its board, and the pin would ignore it."""


# Every env var :func:`_board_path` consults as a path pin. :func:`isolated_board` clears all of
# them: clearing only ``HERMES_KANBAN_HOME`` is the half-done isolation that wrote 5 phantom cards
# to the live board (2026-10-09).
BOARD_PIN_ENV_VARS: tuple[str, ...] = (
    "HERMES_KANBAN_DB",
    "HERMES_KANBAN_BOARD",
    "HERMES_KANBAN_WORKSPACES_ROOT",
    "HERMES_KANBAN_ATTACHMENTS_ROOT",
)


def _assert_pin_agrees_with_explicit_home(pin: str, env_var: Optional[str]) -> None:
    """Fail loud when a path pin contradicts an explicitly-set ``HERMES_KANBAN_HOME``.

    The pin is consulted first and wins, so a caller that set ``HERMES_KANBAN_HOME`` expecting
    isolation silently wrote to the pinned board instead — inside any dispatcher-spawned worker,
    which always carries the pin. A pin *under* the home agrees (the dispatcher's own injection,
    a deployment pinning both to one root) and stays silent.
    """
    home = os.environ.get("HERMES_KANBAN_HOME", "").strip()
    if not home:
        return
    try:
        pinned = Path(pin).expanduser().resolve()
        root = Path(home).expanduser().resolve()
    except OSError:  # unresolvable paths cannot be compared; the pin wins as before
        return
    if pinned == root or root in pinned.parents:
        return
    raise KanbanBoardPinConflict(
        f"{env_var} ({pinned}) is outside the explicitly-set HERMES_KANBAN_HOME ({root}), so the "
        "pin silently outranks the home and this write goes to the pinned board. Drop the pin, or "
        "use `with kanban_db.isolated_board():`, which sets the home and clears "
        f"{', '.join(BOARD_PIN_ENV_VARS)} together."
    )


def _outside_kanban_fence(root: Path) -> bool:
    """Whether a board at ``root`` is writable for this process.

    A ``delegate_task`` descendant's marker (``agent.delegation_context``) fences the pinned DB and
    everything under its own board root, so a temp root inside the lineage root would produce a
    silently read-only board — the failure mode ``scrub_kanban_env`` already documents.
    """
    from agent.delegation_context import kanban_path_is_fenced

    return not kanban_path_is_fenced(root / "kanban.db")


def _isolated_temp_root() -> Path:
    """A fresh temp root this process can actually write a board into.

    ``tempfile`` honours ``TMPDIR``, which every worker exports as its own scratch dir *inside* the
    fenced root, so the first candidate is usually fenced from a worker: retry under the system
    temp before giving up.
    """
    candidates: list[Optional[str]] = [None]  # None = tempfile's own choice (TMPDIR/TEMP/TMP)
    for parent in (tempfile.gettempdir(), "/var/tmp"):
        if parent and parent not in candidates:
            candidates.append(parent)
    for parent in candidates:
        try:
            root = Path(tempfile.mkdtemp(prefix="hermes-kanban-isolated-", dir=parent))
        except OSError:
            continue
        if _outside_kanban_fence(root):
            return root
    raise KanbanBoardPinConflict(
        "no temp root outside this process's fenced kanban board root, where a board would be "
        "read-only. Pass an explicit `root=` outside the root named by "
        "HERMES_DELEGATED_CHILD_CONTEXT."
    )


@contextlib.contextmanager
def isolated_board(root: "os.PathLike[str] | str | None" = None):
    """Point every Kanban path at a throwaway board for the duration of the ``with`` block.

    Sets ``HERMES_KANBAN_HOME`` and clears ``HERMES_KANBAN_DB``, ``HERMES_KANBAN_BOARD``,
    ``HERMES_KANBAN_WORKSPACES_ROOT`` and ``HERMES_KANBAN_ATTACHMENTS_ROOT``, so a board cannot be
    half-isolated; the active board is scoped to ``default`` so resolution is deterministic. Every
    one of those is restored on exit, including on an exception. Yields the isolated root.

    ``root`` defaults to a fresh temp dir chosen to sit *outside* the fence a ``delegate_task``
    descendant carries, so a worker's writes land in the temp board instead of being refused — or,
    with the pin left in place, landing on the live board.
    """
    if root is None:
        isolated_root = _isolated_temp_root()
    else:
        isolated_root = Path(root).expanduser()
        isolated_root.mkdir(parents=True, exist_ok=True)

    saved = {name: os.environ.get(name) for name in BOARD_PIN_ENV_VARS}
    saved_home = os.environ.get("HERMES_KANBAN_HOME")
    token: Token[str | None] = _CURRENT_BOARD_OVERRIDE.set(DEFAULT_BOARD)
    os.environ["HERMES_KANBAN_HOME"] = str(isolated_root)
    for name in BOARD_PIN_ENV_VARS:
        os.environ.pop(name, None)
    try:
        yield isolated_root
    finally:
        _CURRENT_BOARD_OVERRIDE.reset(token)
        if saved_home is None:
            os.environ.pop("HERMES_KANBAN_HOME", None)
        else:
            os.environ["HERMES_KANBAN_HOME"] = saved_home
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


# --- Paths ---

DEFAULT_BOARD = "default"
_CURRENT_BOARD_OVERRIDE: ContextVar[str | None] = ContextVar(
    "hermes_kanban_current_board_override", default=None,
)
# Machine flows that enumerate boards (gateway notifier / watcher / dispatcher
# ticks) resolve board paths env-pin-first — see pin_first_board_resolution().
_PIN_FIRST_BOARD_RESOLUTION: ContextVar[bool] = ContextVar(
    "hermes_kanban_pin_first_board_resolution", default=False,
)


@contextlib.contextmanager
def pin_first_board_resolution():
    """Resolve board paths env-pin-first for machine flows that enumerate boards.

    The gateway notifier, per-subscription cursor writes and the embedded
    dispatcher poll every board slug from ``list_boards()`` — the slug is not
    caller intent, it is an iteration variable. On a box whose environment pins
    ``HERMES_KANBAN_DB`` (the dispatcher default) every slug must map to that
    one pinned file: the notifier dedupes resolved DB paths, and per-slug
    paths read empty boards nobody writes, silently killing wake
    notifications. Explicit cross-board intent (CLI ``--board``, a model
    tool's ``board=``) is a USER property and must never run inside this
    context; with no pin set this context changes nothing.
    """
    token = _PIN_FIRST_BOARD_RESOLUTION.set(True)
    try:
        yield
    finally:
        _PIN_FIRST_BOARD_RESOLUTION.reset(token)


@contextlib.contextmanager
def scoped_current_board(slug: str):
    """Pin the active board for the current context only."""
    token: Token[str | None] = _CURRENT_BOARD_OVERRIDE.set(slug)
    try:
        yield
    finally:
        _CURRENT_BOARD_OVERRIDE.reset(token)


# Slug = directory name: strict enough to stop traversal / separators, loose
# enough for kebab-case. Display names (spaces, emoji) live in board.json.
_BOARD_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9\-_]{0,63}$")


def _normalize_board_slug(slug: Optional[str]) -> Optional[str]:
    """Lowercase + strip a slug; validate; return ``None`` for empty."""
    s = str(slug).strip().lower() if slug is not None else ""
    if not s:
        return None
    if not _BOARD_SLUG_RE.match(s):
        raise ValueError(
            f"invalid board slug {slug!r}: must be 1-64 chars, lowercase "
            f"alphanumerics / hyphens / underscores, not starting with '-' or '_'"
        )
    return s


def _slug_or_default(board: Optional[str]) -> str:
    return _normalize_board_slug(board) or DEFAULT_BOARD


def _require_slug(slug: str) -> str:
    normed = _normalize_board_slug(slug)
    if not normed:
        raise ValueError("board slug is required")
    return normed


def kanban_home() -> Path:
    """``HERMES_KANBAN_HOME`` else ``get_default_hermes_root()``. Shared across
    profiles BY DESIGN: resolving through the active profile's HERMES_HOME would
    fork the board per profile and break the dispatcher/worker handoff."""
    override = os.environ.get("HERMES_KANBAN_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root()


def boards_root() -> Path:
    """``<root>/kanban/boards`` — parent of the *additional* named boards.
    ``default`` is deliberately not here (its DB stays at ``<root>/kanban.db``)."""
    return kanban_home() / "kanban" / "boards"


def current_board_path() -> Path:
    """``<root>/kanban/current`` — one-line slug written by ``boards switch``; absent = ``default``."""
    return kanban_home() / "kanban" / "current"


def get_current_board() -> str:
    """Active slug: context override -> ``HERMES_KANBAN_BOARD`` -> ``<root>/kanban/current``
    (only while that board exists) -> ``DEFAULT_BOARD``. A malformed/stale slug
    falls through — the dispatcher must never crash on a hand-edited file."""
    def _existing(candidate: str) -> Optional[str]:
        if not candidate:
            return None
        try:
            normed = _normalize_board_slug(candidate)
        except ValueError:
            return None
        return normed if normed and board_exists(normed) else None

    for candidate in (
        (_CURRENT_BOARD_OVERRIDE.get() or "").strip(),
        os.environ.get("HERMES_KANBAN_BOARD", "").strip(),
    ):
        found = _existing(candidate)
        if found:
            return found
    try:
        f = current_board_path()
        if f.exists():
            # utf-8-sig read fix (ours): tolerate BOM-persisted current-board files.
            val = f.read_text(encoding="utf-8-sig").strip()
            if val:
                try:
                    normed = _normalize_board_slug(val)
                    if normed and board_exists(normed):
                        return normed
                except ValueError:
                    pass
    except OSError:
        pass
    return DEFAULT_BOARD


def set_current_board(slug: str) -> Path:
    """Persist ``slug`` as the active board; returns the file written. Does NOT
    check the board exists — callers do (so ``boards switch <typo>`` errors)."""
    _kb._assert_not_delegated_child_mutation()
    normed = _require_slug(slug)
    path = current_board_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(normed + "\n", encoding="utf-8")
    return path


def clear_current_board() -> None:
    """Remove ``<root>/kanban/current`` so the active board reverts to ``default``."""
    _kb._assert_not_delegated_child_mutation()
    with contextlib.suppress(FileNotFoundError):
        current_board_path().unlink()


def board_dir(board: Optional[str] = None) -> Path:
    """``<root>/kanban/boards/<slug>/``. For ``default`` this holds metadata
    only (board.json, workspaces/, logs/) — its DB stays at ``<root>/kanban.db``
    for back-compat (:func:`kanban_db_path`).
    """
    return boards_root() / _slug_or_default(board)


def board_exists(board: Optional[str] = None) -> bool:
    """Board has ``board.json`` or ``kanban.db`` on disk; ``default`` always exists."""
    slug = _slug_or_default(board)
    if slug == DEFAULT_BOARD:
        return True
    return _dir_holds_board(board_dir(slug))


def _dir_holds_board(d: Path) -> bool:
    return (d / "board.json").exists() or (d / "kanban.db").exists()


def _explicit_board_slug(board: Optional[str]) -> Optional[str]:
    """Explicit caller intent: a direct ``board=`` argument, else the scoped
    ``--board`` context (CLI ``hermes kanban --board``, dashboard plugin_api);
    ``None`` when the caller expressed neither."""
    if board is not None:
        return _normalize_board_slug(board)
    # A caller-scoped board (CLI `hermes kanban --board B ...`, dashboard
    # plugin_api) is explicit intent just like a direct board= argument —
    # without this a worker-pinned HERMES_KANBAN_DB silently outranks --board
    # (os-reviewer P1 on PR#107195 / t_11c4afd8).
    ctx = (_CURRENT_BOARD_OVERRIDE.get() or "").strip()
    if ctx:
        try:
            return _normalize_board_slug(ctx)
        except ValueError:
            return None
    return None


def _explicit_board_intent_pinned() -> bool:
    """Whether an explicit ``board=`` (or scoped ``--board``) must still resolve
    through the ``HERMES_KANBAN_DB``-style env pins instead of its own board dir.

    True for machine flows wrapped in :func:`pin_first_board_resolution` and
    for every execution the dispatcher fences: its own workers (they carry
    ``HERMES_KANBAN_TASK``) and delegated children / descendants (the
    ``HERMES_DELEGATED_CHILD_CONTEXT`` marker). The pins ARE the "workers
    physically cannot see other boards" isolation (5ec6baa), and
    ``agent.delegation_context.kanban_path_is_fenced`` checks the pinned path /
    fenced root — an explicit board that resolved elsewhere would also escape
    that fence."""
    if _PIN_FIRST_BOARD_RESOLUTION.get():
        return True
    from agent.delegation_context import explicit_board_intent_is_pinned
    return explicit_board_intent_is_pinned()


def _board_path(
    env_var: Optional[str], board: Optional[str], default_parts: tuple[str, ...], leaf: str,
) -> Path:
    """Shared resolver. An explicit ``board=`` argument — or the scoped
    ``--board`` context (:func:`scoped_current_board`) — outranks the ``env_var``
    pin ONLY where no fence applies (see :func:`_explicit_board_intent_pinned`):
    user-facing cross-board intent (CLI ``--board``, a model tool's ``board=``)
    is honored, but machine flows that enumerate boards (gateway notifier /
    watcher / dispatcher ticks) and dispatched or delegated workers keep
    resolving through the pin. Without explicit intent the ``env_var`` override
    pins the file, else legacy ``<root>/<default_parts>`` for the ``default``
    board, else ``board_dir(slug)/leaf``."""
    pin = os.environ.get(env_var, "").strip() if env_var else ""
    slug = _explicit_board_slug(board)
    if pin and (slug is None or _explicit_board_intent_pinned()):
        _assert_pin_agrees_with_explicit_home(pin, env_var)
        return Path(pin).expanduser()
    if slug is None:
        slug = get_current_board()
    if slug == DEFAULT_BOARD:
        return kanban_home().joinpath(*default_parts)
    return board_dir(slug) / leaf


def kanban_db_path(board: Optional[str] = None) -> Path:
    """``kanban.db`` path: ``HERMES_KANBAN_DB`` pins it (injected into workers);
    ``default`` -> ``<root>/kanban.db`` (back-compat), else the board dir."""
    return _board_path("HERMES_KANBAN_DB", board, ("kanban.db",), "kanban.db")


def workspaces_root(board: Optional[str] = None) -> Path:
    """Per-board scratch workspace root (``HERMES_KANBAN_WORKSPACES_ROOT`` wins);
    ``default`` keeps the legacy ``<root>/kanban/workspaces/``."""
    return _board_path("HERMES_KANBAN_WORKSPACES_ROOT", board, ("kanban", "workspaces"), "workspaces")


def attachments_root(board: Optional[str] = None) -> Path:
    """Per-board attachments root (``HERMES_KANBAN_ATTACHMENTS_ROOT`` wins). Workers
    read attachments by absolute path, so remote terminal backends must mount it."""
    return _board_path("HERMES_KANBAN_ATTACHMENTS_ROOT", board, ("kanban", "attachments"), "attachments")


def task_attachments_dir(task_id: str, board: Optional[str] = None) -> Path:
    """Return the per-task attachment directory ``<root>/<task_id>/``."""
    return attachments_root(board=board) / task_id


def worker_logs_dir(board: Optional[str] = None) -> Path:
    """Per-board worker log dir (logs follow the board so ``hermes kanban log``
    is unambiguous when two boards share a task id)."""
    return _board_path(None, board, ("kanban", "logs"), "logs")


# Late-bound facade namespace (the facade imports this module at its tail, so ``_kb`` is the
# partially-initialized ``hermes_cli.kanban_db`` module object; every use is at call time).
from hermes_cli import kanban_db as _kb  # noqa: E402
