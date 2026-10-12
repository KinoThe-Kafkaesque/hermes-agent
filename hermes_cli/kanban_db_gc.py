"""Retention windows for ``hermes kanban gc``: which old rows and log files a
sweep may delete, and the guards that keep a sweep away from rows the fleet still
reads.

Split out of ``hermes_cli.kanban_db``; origin-resident helpers (``write_txn``,
``worker_logs_dir``) are reached late-bound via ``_kb`` (import-cycle breaking) so
monkeypatching ``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import contextlib
import sqlite3
import time
from typing import Optional


def _retention_seconds(older_than_seconds: int) -> int:
    """Normalise a gc retention window, rejecting negatives.

    Shared by every gc sweep: a negative window puts the cutoff in the future,
    so "older than cutoff" would match every row / file instead of none —
    refuse before any sweep runs.
    """
    older_than_seconds = int(older_than_seconds)
    if older_than_seconds < 0:
        raise ValueError(
            f"older_than_seconds must be >= 0, got {older_than_seconds!r}: "
            "a negative retention selects everything."
        )
    return older_than_seconds


def gc_events(conn: sqlite3.Connection, *, older_than_seconds: int = 30 * 24 * 3600) -> int:
    """Prune old done/archived events, retaining decomposition identity until task deletion.

    ``older_than_seconds=0`` means everything older than now; the CLI maps
    ``--event-retention-days 0`` to "disabled" before calling this.
    """
    cutoff = int(time.time()) - _retention_seconds(older_than_seconds)
    with _kb.write_txn(conn):
        cur = conn.execute(
            "DELETE FROM task_events WHERE created_at < ? AND kind != 'decomposed' AND task_id IN "
            "(SELECT id FROM tasks WHERE status IN ('done', 'archived'))", (cutoff,),
        )
    return int(cur.rowcount or 0)


def gc_worker_logs(*, older_than_seconds: int = 30 * 24 * 3600, board: Optional[str] = None) -> int:
    """Delete worker log files older than the cutoff on one board; returns the count.

    ``older_than_seconds=0`` means everything older than now; the CLI maps
    ``--log-retention-days 0`` to "disabled" before calling this.
    """
    older_than_seconds = _retention_seconds(older_than_seconds)
    log_dir = _kb.worker_logs_dir(board=board)
    if not log_dir.exists():
        return 0
    cutoff = time.time() - older_than_seconds
    removed = 0
    for p in log_dir.iterdir():
        with contextlib.suppress(OSError):
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
    return removed


# A run row is the durable evidence of an attempt: it carries the handoff
# ``summary``/``metadata`` a card's declaration lives in, and it is the record the
# shared provider-demotion tally reads. Deleting it is therefore an OPERATOR
# decision, not a default: the CLI maps ``--run-retention-days 0`` to "disabled",
# and ``gc_runs`` is never reached unless a positive window is passed.
#
# Three guards make the sweep safe against the reads the fleet actually performs:
#
# 1. Only runs of TERMINAL tasks (``done``/``archived``) are eligible. A
#    ``rate_limited`` run on a card that is still retrying is the demotion tally's
#    evidence AND the respawn guard's input, so it is never eligible no matter how
#    old it is — an old wall on a live card is exactly the row that matters.
# 2. A run the task still points at (``tasks.current_run_id``) is never deleted.
#    The kernel's invariant is ``current_run_id IS NULL <=> run row terminal``, and
#    the reclaim/end-run paths write through that pointer; deleting the row would
#    orphan the pointer.
# 3. The window is FLOORED at the shared tally's read window
#    (``_WALL_DB_READ_WINDOW_SECONDS``, 1 day), so no ``rate_limited`` row the
#    shared read would still consult is deletable even if an operator passes a
#    smaller window. This is what makes the boundary a property of the code rather
#    than of operator discipline.
#
# The cut is on ``COALESCE(ended_at, started_at)`` so a run that never closed is
# measured from when it began rather than falling through as NULL.
def _run_retention_floor_seconds() -> int:
    """Shortest window ``gc_runs`` will honour: the shared tally's read window.

    Imported lazily (this module is loaded by the dispatcher's facade) and
    defensively: if the constant cannot be read, fall back to one day, which is
    its value today — the floor must never silently become zero.
    """
    try:
        from hermes_cli.kanban_provider_distribution import _WALL_DB_READ_WINDOW_SECONDS

        return max(int(_WALL_DB_READ_WINDOW_SECONDS), 1)
    except Exception:  # health: allow BLE001 -- floor must not break gc
        return 24 * 3600


def gc_runs(conn: sqlite3.Connection, *, older_than_seconds: int = 30 * 24 * 3600) -> int:
    """Prune old run rows for terminal tasks; returns the number deleted.

    ``older_than_seconds=0`` means everything older than now; the CLI maps
    ``--run-retention-days 0`` to "disabled" before calling this. The window is
    floored at the shared tally's read window (see the module note above).
    """
    window = max(_retention_seconds(older_than_seconds), _run_retention_floor_seconds())
    cutoff = int(time.time()) - window
    with _kb.write_txn(conn):
        cur = conn.execute(
            "DELETE FROM task_runs WHERE id NOT IN ("
            "    SELECT current_run_id FROM tasks WHERE current_run_id IS NOT NULL"
            ") AND COALESCE(ended_at, started_at) < ? AND task_id IN ("
            "    SELECT id FROM tasks WHERE status IN ('done', 'archived')"
            ")",
            (cutoff,),
        )
    return int(cur.rowcount or 0)


# Late-bound origin namespace (see module docstring); imported LAST so this
# module is fully populated before ``kanban_db`` imports from it.
from hermes_cli import kanban_db as _kb  # noqa: E402
