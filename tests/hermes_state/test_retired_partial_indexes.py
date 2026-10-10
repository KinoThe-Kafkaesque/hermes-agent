"""The retired dangerous partial indexes must stay retired — behaviourally.

``tests/hermes_cli/test_partial_index_where_anomaly.py`` pins the ANOMALY itself
(a reused parameterized statement returns wrong rows against a partial index whose
``WHERE`` holds a scalar literal) using a synthetic table. This file pins the
PRODUCTION consumers: each of the six indexes the guard used to baseline is now
composite, and the real statement that reads it returns the right rows after the
index appears under a live cached connection.

Why this is a separate arm from the anomaly file: the anomaly file proves the
mechanism is real on this host; this file proves the specific shipped indexes are
no longer in the dangerous shape AND that a regression back to the partial form
would be caught by a behavioural test rather than only by the source-shape guard
(``scripts/ci/check_partial_index_where.py``, which is blind to queries).

Each test:
  1. builds a real state.db (``SessionDB``) / executions ledger with rows where
     the alternate bound value genuinely matches, so a mis-served plan reads 0;
  2. drops the target index and re-creates it in the DANGEROUS partial spelling on
     a live cached connection — this is the pre-fix revision, and the test asserts
     it returns the WRONG rows (the pin's RED arm, proving the test discriminates);
  3. re-creates the SHIPPED composite form on a fresh live cached connection and
     asserts the same statement returns the RIGHT rows.
"""

from __future__ import annotations

import sqlite3

import pytest

from hermes_state import SessionDB

# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

def _seed_messages(db: SessionDB, n: int = 400) -> None:
    """A session whose alternate bound values each match rows.

    * ``role`` alternates assistant/user, BOTH with ``tool_calls`` set;
    * ``active`` is 1 for the first half, 0 for the second;
    * ``display_order``/``display_identity`` are set on most rows, but every tenth
      row leaves them NULL (so the display-backfill predicate matches on both
      active values);
    * ``message_uid`` / ``platform_message_id`` are unique per row.
    """
    db.create_session("s1", source="cli")
    conn = db._conn
    rows = []
    for i in range(n):
        asst = (i % 2 == 1)
        active = 1 if i < n // 2 else 0
        backfill = (i % 10 == 0)  # this row is unindexed for display purposes
        rows.append((
            i + 1, "s1", "assistant" if asst else "user", "body",
            "call%d" % i, 1000.0 + i,
            active, 0,
            None if backfill else ("idn%d" % (i // 10)).encode(),
            None if backfill else i,
            "uid%d" % (i + 1), "pm%d" % i,
        ))
    conn.executemany(
        "INSERT INTO messages(id, session_id, role, content, tool_calls, timestamp,"
        " active, compacted, display_identity, display_order, message_uid, platform_message_id)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", rows,
    )
    conn.commit()


def _count(db: SessionDB, sql: str, params) -> int:
    return len(db._conn.execute(sql, params).fetchall())


# Every index on ``messages`` that could serve one of the consumer queries instead of the
# target index. Dropped for the RED arm so the dangerous spelling is provably the CHOSEN
# plan — otherwise the optimizer picks a safe sibling and the arm reads GREEN for the
# wrong reason (a vacuous pin).
_MESSAGES_INDEXES = (
    "idx_messages_session", "idx_messages_session_id", "idx_messages_session_active",
    "idx_messages_assistant_calls_by_session", "idx_messages_platform_msg_id",
    "idx_messages_display_page", "idx_messages_display_backfill",
    "idx_messages_display_identity", "idx_messages_active_null", "idx_messages_session_uid",
)


def _shipped_sql(db: SessionDB, index_name: str) -> str:
    row = db._conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND name=?", (index_name,)
    ).fetchone()
    assert row is not None, "index %s is missing from a freshly created state.db" % index_name
    return row[0] or ""


# --------------------------------------------------------------------------- #
# the six retired sites
# --------------------------------------------------------------------------- #

_MESSAGES_CASES = (
    # (index, dangerous partial DDL, shipped composite assertion fragment,
    #  consumer SQL, prime params, alternate params)
    (
        "idx_messages_assistant_calls_by_session",
        "CREATE INDEX idx_messages_assistant_calls_by_session ON messages(session_id)"
        " WHERE role = 'assistant' AND tool_calls IS NOT NULL",
        "(session_id, role, tool_calls)",
        "SELECT id FROM messages WHERE session_id = ? AND role = ? AND tool_calls IS NOT NULL"
        " ORDER BY id",
        ("s1", "assistant"), ("s1", "user"),
    ),
    (
        "idx_messages_display_page",
        "CREATE INDEX idx_messages_display_page ON messages(session_id, display_order,"
        " active DESC, id DESC) WHERE active = 1 OR compacted = 1",
        "(session_id, display_order, active DESC, id DESC)",
        "SELECT id FROM messages WHERE session_id = ? AND active = ? AND display_order IS NOT NULL"
        " ORDER BY id",
        ("s1", 1), ("s1", 0),
    ),
    (
        "idx_messages_display_backfill",
        "CREATE INDEX idx_messages_display_backfill ON messages(session_id)"
        " WHERE (display_order IS NULL OR display_identity IS NULL)"
        " AND (active = 1 OR compacted = 1)",
        "(session_id, display_order, display_identity)",
        "SELECT id FROM messages WHERE session_id = ? AND active = ?"
        " AND (display_order IS NULL OR display_identity IS NULL)",
        ("s1", 1), ("s1", 0),
    ),
    (
        "idx_messages_display_identity",
        "CREATE INDEX idx_messages_display_identity ON messages(session_id, display_identity,"
        " display_order) WHERE display_identity IS NOT NULL AND (active = 1 OR compacted = 1)",
        "(session_id, display_identity, display_order)",
        "SELECT id FROM messages WHERE session_id = ? AND active = ? AND display_identity IS NOT NULL",
        ("s1", 1), ("s1", 0),
    ),
    (
        "idx_messages_session_uid",
        "CREATE INDEX idx_messages_session_uid ON messages(session_id, message_uid) WHERE active = 1",
        "(session_id, message_uid, active)",
        "SELECT id FROM messages WHERE session_id = ? AND active = ? AND message_uid = ?",
        ("s1", 1, "uid2"), ("s1", 0, "uid201"),
    ),
)


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    _seed_messages(database)
    yield database
    database.close()


@pytest.mark.parametrize(
    "index_name,dangerous,composite_fragment,sql,prime,other", _MESSAGES_CASES,
    ids=[c[0] for c in _MESSAGES_CASES],
)
def test_retired_partial_index_is_composite_and_immune(
    db, index_name, dangerous, composite_fragment, sql, prime, other
):
    """The shipped index is composite, and the DANGEROUS spelling mis-serves.

    Two claims, one test: the shipped form is the composite (no ``WHERE``), and a
    regression back to the partial spelling reddens the very same statement — so
    the pin would fail on the pre-fix revision rather than only on a shape check.
    """
    shipped = _shipped_sql(db, index_name)
    assert "WHERE" not in shipped.upper(), (
        "%s must ship composite; found: %s" % (index_name, shipped)
    )
    assert composite_fragment in shipped, shipped

    # Pre-fix revision: the dangerous partial form, with every competing index dropped so
    # it is provably the CHOSEN plan. The statement is primed with the value the index's
    # WHERE describes BEFORE the index appears — the migration-on-a-live-board sequence.
    # The `== 0` assertion is the discrimination check: if the dangerous form were not the
    # chosen plan, the reused statement would still be correct and read non-zero.
    for name in _MESSAGES_INDEXES:
        db._conn.execute("DROP INDEX IF EXISTS %s" % name)
    db._conn.commit()
    primed = _count(db, sql, prime)
    assert primed > 0, "fixture must make the primed value match rows"
    db._conn.execute(dangerous)
    db._conn.commit()
    # The poison step: the cached statement runs again with the SAME bound value the
    # index's WHERE describes, which is what fixes the mis-served plan into the cache.
    assert _count(db, sql, prime) == primed
    assert _count(db, sql, other) == 0, (
        "the dangerous partial %s must mis-serve the reused statement (0 rows)" % index_name
    )

    # Shipped revision: composite on the same live connection stays correct.
    db._conn.execute("DROP INDEX IF EXISTS %s" % index_name)
    db._conn.execute(shipped if shipped.upper().startswith("CREATE INDEX")
                     else "CREATE INDEX " + shipped)
    db._conn.commit()
    assert _count(db, sql, prime) == primed
    restored = _count(db, sql, other)
    assert restored > 0, (
        "the shipped composite %s must return rows for the other bound value" % index_name
    )


def test_shipped_indexes_are_composite_and_not_dangerous():
    """A fresh state.db carries none of the six retired partial spellings.

    The shape-level companion to the behavioural arms above: whatever their queries do,
    the shipped DDL must not be a ``WHERE <col> = <literal>`` partial index.
    """
    import tempfile
    from hermes_state_common import _partial_where_has_literal_equality

    tmp = tempfile.mkdtemp()
    database = SessionDB(__import__("pathlib").Path(tmp) / "state.db")
    try:
        rows = database._conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
        dangerous = {name for name, sql in rows if _partial_where_has_literal_equality(sql or "")}
        assert dangerous == set(), "fresh state.db carries dangerous partial indexes: %s" % dangerous
    finally:
        database.close()


def test_executions_occurrence_is_composite_and_immune(tmp_path):
    """``cron/executions.py``'s occurrence index: composite, and the partial mis-serves."""
    import cron.executions as executions

    executions.EXECUTIONS_FILE = tmp_path / "cron" / "executions.db"
    executions.EXECUTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with executions._transaction() as conn:
        conn.execute(
            "INSERT INTO executions(id, job_id, source, process_id, pid, status, claimed_at,"
            " scheduled_instant) VALUES('e1','j1','sched','p',1,'completed',"
            "'2026-01-01T00:00:00+00:00','2026-01-01T00:00:00+00:00')"
        )
        for i in range(2, 200):
            conn.execute(
                "INSERT INTO executions(id, job_id, source, process_id, pid, status, claimed_at,"
                " scheduled_instant) VALUES(?, 'j1', 'sched', 'p', 1, ?, ?,"
                " '2026-01-01T00:00:00+00:00')",
                ("e%d" % i, "completed" if i % 2 else "failed",
                 "2026-01-01T00:00:%02d+00:00" % (i % 60)),
            )

    sql = "SELECT id FROM executions WHERE job_id = ? AND status = ? AND scheduled_instant IS NOT NULL"
    live = sqlite3.connect(executions.EXECUTIONS_FILE, cached_statements=128)
    try:
        shipped = live.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_executions_occurrence'"
        ).fetchone()[0]
        assert "WHERE" not in shipped.upper(), shipped
        assert "(job_id, scheduled_instant, status)" in shipped, shipped

        correct = len(live.execute(sql, ("j1", "failed")).fetchall())
        assert correct > 0

        live.execute("DROP INDEX IF EXISTS idx_executions_occurrence")
        live.execute(
            "CREATE INDEX idx_executions_occurrence ON executions(job_id, scheduled_instant)"
            " WHERE status='completed'"
        )
        live.commit()
        assert len(live.execute(sql, ("j1", "completed")).fetchall()) > 0
        assert len(live.execute(sql, ("j1", "failed")).fetchall()) == 0, (
            "the dangerous partial idx_executions_occurrence must mis-serve the reused statement"
        )
    finally:
        live.close()
        executions.EXECUTIONS_FILE = None
