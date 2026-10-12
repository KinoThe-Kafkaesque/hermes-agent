"""The partial-index statement-cache anomaly, pinned inside the repo's own tooling.

``tests/fixtures/partial_index_statement_cache_anomaly.py`` reproduces the
measured behaviour: with a partial index whose ``WHERE`` compares a column to a
literal, a REUSED parameterized statement bound to a different value returns
wrong rows, silently, on this fleet's SQLite builds (3.53.1 / 3.47.1 / 3.37.2).
The shipped ``idx_runs_outcome_ended`` is the composite form precisely to avoid
it (``hermes_cli/kanban_db_connect.py``).

These tests hold two contracts:

* the anomaly is REAL and REPRODUCIBLE here — if a future SQLite upgrade fixes
  it, ``test_partial_index_literal_reproduces_the_anomaly`` fails and tells us
  the workaround can be revisited, rather than the anomaly silently rotting into
  a stale comment;
* the composite form the fleet ships is UNAFFECTED by the same statement-reuse
  sequence — the regression that would land is replacing it with a partial form.

The static guard that refuses NEW dangerous-shaped sites lives in
``scripts/ci/check_partial_index_where.py``; this file is the behavioural arm, so
a shape-only pass can never be mistaken for a working one.
"""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import pytest

_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "partial_index_statement_cache_anomaly.py"
)


def _fixture_module():
    spec = importlib.util.spec_from_file_location("partial_index_anomaly", _FIXTURE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def anomaly():
    return _fixture_module()


def test_fixture_exists():
    assert _FIXTURE.is_file(), f"anomaly fixture missing: {_FIXTURE}"


def test_partial_index_literal_reproduces_the_anomaly(anomaly, tmp_path):
    """The dangerous form returns 0 where 200 is correct — the measured defect.

    This is the RED arm: it proves the anomaly is live on this host's SQLite, so
    the guard's refusal is grounded in behaviour rather than in a remembered
    anecdote. A SQLite build that fixes the bug makes this test fail on purpose.
    """
    result = anomaly.scenario(anomaly.DANGEROUS_PARTIAL, db_dir=str(tmp_path))
    assert result["primed"] == result["expected"]
    assert result["reused"] == result["expected"], "same bound value must stay correct"
    assert result["other"] == 0, (
        "expected the partial-index anomaly (0 rows for the other bound value); "
        f"got {result['other']} — if SQLite fixed this, the composite-index "
        "workaround and the guard's premise need revisiting"
    )


def test_shipped_composite_index_is_unaffected(anomaly, tmp_path):
    """The composite form the fleet actually ships returns the right rows."""
    result = anomaly.scenario(anomaly.SHIPPED_COMPOSITE, db_dir=str(tmp_path))
    assert result["primed"] == result["expected"]
    assert result["reused"] == result["expected"]
    assert result["other"] == result["expected"], (
        "the shipped composite index must be immune to statement-cache reuse"
    )


def test_statement_cache_is_the_vector(anomaly, tmp_path):
    """``cached_statements=0`` removes the anomaly — the reused statement is it."""
    result = anomaly.scenario(
        anomaly.DANGEROUS_PARTIAL, cached_statements=0, db_dir=str(tmp_path)
    )
    assert result["other"] == result["expected"]


def test_fresh_connection_is_always_correct(anomaly, tmp_path):
    """A connection opened AFTER the index exists is not poisoned."""
    result = anomaly.scenario(anomaly.DANGEROUS_PARTIAL, db_dir=str(tmp_path))
    assert result["other"] == 0  # the live connection is poisoned
    conn = sqlite3.connect(str(tmp_path / "repro.db"))
    fresh = len(conn.execute(anomaly.WALL_QUERY, (anomaly.COMPLETED, 0)).fetchall())
    conn.close()
    assert fresh == result["expected"]


def test_shipped_task_runs_index_is_composite_not_partial(tmp_path, monkeypatch):
    """The kanban board's real index is the composite form, exercised end to end.

    The sequence that reproduces the anomaly is the migration running on a LIVE
    cached connection whose read is already prepared — a board predating the
    index. Both bound values must return their correct row counts. A future card
    swapping the composite for a partial form reddens here even if it keeps the
    plan assertion green.
    """
    from pathlib import Path as _Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_provider_distribution as kpd

    kb.init_db()
    conn = sqlite3.connect(kb.kanban_db_path())
    conn.execute("DROP INDEX IF EXISTS idx_runs_outcome_ended")  # a board predating it
    conn.executemany(
        "INSERT INTO task_runs (task_id, status, started_at, ended_at, outcome, metadata)"
        " VALUES (?, 'done', ?, ?, ?, '{}')",
        [(f"t{i}", 1000 + i, 1000 + i, kpd._RATE_LIMITED_OUTCOME if i % 2 else "completed")
         for i in range(400)],
    )
    conn.commit()
    conn.close()

    with kbc.connect_closing() as live:
        # 1) the fleet's own read is prepared and cached while the board has no index
        primed = len(live.execute(kpd._WALL_QUERY, (kpd._RATE_LIMITED_OUTCOME, 0)).fetchall())
        # 2) the migration runs on THIS live connection and creates the index
        kbc._migrate_add_optional_columns(live)
        live.commit()
        sql = live.execute(
            "SELECT sql FROM sqlite_master WHERE name='idx_runs_outcome_ended'"
        ).fetchone()[0]
        assert "WHERE" not in sql.upper(), f"shipped index must be composite, got: {sql}"
        # 3) reuse the cached statement, then bind the other outcome
        reused = len(live.execute(kpd._WALL_QUERY, (kpd._RATE_LIMITED_OUTCOME, 0)).fetchall())
        other = len(live.execute(kpd._WALL_QUERY, ("completed", 0)).fetchall())

    assert primed == 200
    assert reused == 200
    assert other == 200, (
        "a partial idx_runs_outcome_ended mis-serves the reused tally read; the "
        "shipped index must stay composite"
    )
