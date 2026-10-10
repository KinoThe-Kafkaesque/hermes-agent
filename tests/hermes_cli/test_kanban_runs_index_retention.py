"""The shared provider-demotion tally read must be served by an index, and
``kanban gc`` must be able to bound ``task_runs`` growth without eating a run the
fleet still reads.

Both halves are behaviour contracts, not change-detectors:

* the index arm runs the LITERAL read the dispatcher runs (``_WALL_QUERY``, via
  ``_shared_wall_rows``) and asserts the plan no longer full-scans ``task_runs`` —
  a regression that drops the index from the migration path fails here even though
  the query still returns the right rows;
* the retention arm asserts BOTH directions on the same fixture (a run inside the
  window survives, a run outside it is deleted), plus the two guards that make the
  sweep safe: a non-terminal card's wall is never eligible, and a run the task still
  points at is never deleted.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_ops
from hermes_cli import kanban_provider_distribution as kpd


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return tmp_path


def _plan(conn, sql: str, params) -> list[str]:
    return [row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + sql, params)]


def _runs_index_names(conn) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='task_runs'"
        )
    }


# --- The index serves the shared read ----------------------------------------


def test_shared_wall_read_is_served_by_an_index_not_a_full_scan(board):
    """The demotion tally read runs on EVERY spawn; it must not full-scan.

    Uses the production SQL and the production parameter tuple so a change to
    either is exercised here. The ``SCAN task_runs`` plan is the regression.
    """
    with kbc.connect_closing() as conn:
        assert "idx_runs_outcome_ended" in _runs_index_names(conn)
        plan = _plan(
            conn, kpd._WALL_QUERY, (kpd._RATE_LIMITED_OUTCOME, int(time.time()) - 86400)
        )
    joined = " ".join(plan)
    assert "idx_runs_outcome_ended" in joined, plan
    assert not any(p.startswith("SCAN task_runs") for p in plan), plan


def test_index_creation_is_idempotent_and_safe_on_a_populated_board(board):
    """Re-running the migration must be a no-op, and must not disturb rows."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        with kb.write_txn(conn):
            for i in range(50):
                conn.execute(
                    "INSERT INTO task_runs (task_id, status, started_at, ended_at,"
                    " outcome, metadata) VALUES (?, 'done', ?, ?, 'rate_limited', '{}')",
                    (tid, 1000 + i, 1000 + i),
                )
        before = conn.execute("SELECT count(*) FROM task_runs").fetchone()[0]
    # Force the full migration path again (init_db clears the process cache).
    for _ in range(2):
        kb.init_db()
        with kbc.connect_closing() as conn:
            assert conn.execute("SELECT count(*) FROM task_runs").fetchone()[0] == before
            assert "idx_runs_outcome_ended" in _runs_index_names(conn)


def test_index_is_added_to_an_existing_board_without_the_index(board):
    """An existing board (rows present, index absent) gains the index on connect."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_runs (task_id, status, started_at, ended_at,"
                " outcome, metadata) VALUES (?, 'done', 1, 2, 'rate_limited', '{}')",
                (tid,),
            )
        # Simulate a board predating the index.
        conn.execute("DROP INDEX IF EXISTS idx_runs_outcome_ended")
        assert "idx_runs_outcome_ended" not in _runs_index_names(conn)
    kb.init_db()
    with kbc.connect_closing() as conn:
        assert "idx_runs_outcome_ended" in _runs_index_names(conn)
        assert conn.execute("SELECT count(*) FROM task_runs").fetchone()[0] == 1


def test_legacy_drifted_task_runs_board_still_initializes(board):
    """A pre-AUTOINCREMENT ``task_runs`` (no ``outcome``/``ended_at``) is rebuilt
    before the index is created, so init must not abort with "no such column"."""
    db_path = kb.kanban_db_path(board="legacy")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.executescript(kb.SCHEMA_SQL)
    conn.executescript(
        """
        DROP TABLE task_runs;
        CREATE TABLE task_runs (id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
            profile TEXT, status TEXT NOT NULL, started_at INTEGER NOT NULL);
        """
    )
    conn.execute("INSERT INTO task_runs VALUES ('r-1', 'task-1', 'default', 'done', 1000)")
    conn.commit()
    conn.close()

    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with kbc.connect(db_path) as migrated:
        assert "idx_runs_outcome_ended" in _runs_index_names(migrated)
        assert migrated.execute("SELECT count(*) FROM task_runs").fetchone()[0] == 1


# --- The retention boundary --------------------------------------------------


def _terminal_task_with_wall(conn, *, ended_ago_seconds: int, status: str = "done") -> str:
    """A terminal task carrying one ``rate_limited`` run that ended *ago* seconds back."""
    tid = kb.create_task(conn, title="t", assignee="a")
    ended = int(time.time()) - ended_ago_seconds
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, tid))
        conn.execute(
            "INSERT INTO task_runs (task_id, status, started_at, ended_at, outcome, metadata)"
            " VALUES (?, 'done', ?, ?, 'rate_limited', '{\"provider\":\"p\",\"model\":\"m\"}')",
            (tid, ended - 10, ended),
        )
    return tid


def _run_count(conn, tid: str) -> int:
    return conn.execute("SELECT count(*) FROM task_runs WHERE task_id=?", (tid,)).fetchone()[0]


def test_gc_runs_deletes_outside_the_window_and_keeps_inside_it(board):
    """A discriminating arm: same fixture, both directions.

    The card's acceptance names exactly this — a row outside the window goes, a
    row inside it stays. One fixture with two rows rules out a sweep that deletes
    everything or nothing.
    """
    with kbc.connect_closing() as conn:
        old = _terminal_task_with_wall(conn, ended_ago_seconds=90 * 24 * 3600)
        recent = _terminal_task_with_wall(conn, ended_ago_seconds=2 * 3600)
        removed = kb.gc_runs(conn, older_than_seconds=30 * 24 * 3600)
        assert removed == 1
        assert _run_count(conn, old) == 0
        assert _run_count(conn, recent) == 1


def test_gc_runs_never_deletes_a_wall_on_a_non_terminal_task(board):
    """A ``rate_limited`` row on a still-retrying card is the demotion tally's own
    evidence and the respawn guard's input — never eligible, however old."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="retrying", assignee="a")
        ended = int(time.time()) - 90 * 24 * 3600
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
            conn.execute(
                "INSERT INTO task_runs (task_id, status, started_at, ended_at, outcome, metadata)"
                " VALUES (?, 'done', ?, ?, 'rate_limited', '{}')",
                (tid, ended - 10, ended),
            )
        assert kb.gc_runs(conn, older_than_seconds=30 * 24 * 3600) == 0
        assert _run_count(conn, tid) == 1


def test_gc_runs_never_deletes_the_run_the_task_points_at(board):
    """``tasks.current_run_id`` is a live pointer; deleting its row orphans it."""
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        ended = int(time.time()) - 90 * 24 * 3600
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_runs (id, task_id, status, started_at, ended_at, outcome)"
                " VALUES (999001, ?, 'done', ?, ?, 'completed')",
                (tid, ended - 10, ended),
            )
            conn.execute(
                "UPDATE tasks SET status='done', current_run_id=999001 WHERE id=?", (tid,)
            )
        assert kb.gc_runs(conn, older_than_seconds=30 * 24 * 3600) == 0
        assert conn.execute(
            "SELECT count(*) FROM task_runs WHERE id=999001"
        ).fetchone()[0] == 1


def test_gc_runs_window_is_floored_at_the_shared_tally_read_window(board):
    """A window shorter than the tally read window must not delete a wall the
    shared read would still consult — the floor is a property of the code."""
    floor = kb._run_retention_floor_seconds()
    assert floor == kpd._WALL_DB_READ_WINDOW_SECONDS
    with kbc.connect_closing() as conn:
        tid = _terminal_task_with_wall(conn, ended_ago_seconds=floor // 2)
        # Ask for a window far smaller than the floor: the floor wins.
        assert kb.gc_runs(conn, older_than_seconds=60) == 0
        assert _run_count(conn, tid) == 1


def test_gc_runs_rejects_negative_window(board):
    with kbc.connect_closing() as conn:
        tid = _terminal_task_with_wall(conn, ended_ago_seconds=90 * 24 * 3600)
        with pytest.raises(ValueError, match="older_than_seconds"):
            kb.gc_runs(conn, older_than_seconds=-86400)
        assert _run_count(conn, tid) == 1


# --- The CLI wiring ----------------------------------------------------------


def _gc_args(**overrides) -> argparse.Namespace:
    base = dict(event_retention_days=30, log_retention_days=30, run_retention_days=0)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_cmd_gc_defaults_to_keeping_runs(board):
    """The default is conservative: runs are kept unless explicitly asked for."""
    with kbc.connect_closing() as conn:
        tid = _terminal_task_with_wall(conn, ended_ago_seconds=90 * 24 * 3600)
    assert kanban_ops._cmd_gc(_gc_args()) == 0
    with kbc.connect_closing() as conn:
        assert _run_count(conn, tid) == 1


def test_cmd_gc_deletes_runs_when_a_window_is_passed(board):
    with kbc.connect_closing() as conn:
        tid = _terminal_task_with_wall(conn, ended_ago_seconds=90 * 24 * 3600)
    assert kanban_ops._cmd_gc(_gc_args(run_retention_days=30)) == 0
    with kbc.connect_closing() as conn:
        assert _run_count(conn, tid) == 0


def test_cmd_gc_accepts_a_namespace_without_run_retention_days(board):
    """A caller built before the flag existed (no attribute) keeps working, and
    keeps its runs."""
    with kbc.connect_closing() as conn:
        tid = _terminal_task_with_wall(conn, ended_ago_seconds=90 * 24 * 3600)
    legacy = argparse.Namespace(event_retention_days=30, log_retention_days=30)
    assert kanban_ops._cmd_gc(legacy) == 0
    with kbc.connect_closing() as conn:
        assert _run_count(conn, tid) == 1


def test_gc_parser_rejects_a_negative_run_window():
    from hermes_cli.kanban_parser import build_parser

    import argparse as _ap

    wrap = _ap.ArgumentParser(prog="wrap", add_help=False)
    parser = build_parser(wrap.add_subparsers(dest="_top"))
    with pytest.raises(SystemExit):
        parser.parse_args(["gc", "--run-retention-days", "-1"])
    ns = parser.parse_args(["gc"])
    assert ns.run_retention_days == 0
