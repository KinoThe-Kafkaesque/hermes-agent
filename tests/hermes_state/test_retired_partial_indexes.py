"""The partial-index hazard: which sites are composited, which stay partial, and why.

``tests/hermes_cli/test_partial_index_where_anomaly.py`` pins the ANOMALY itself
(a reused parameterized statement returns wrong rows against a partial index whose
``WHERE`` holds a scalar literal) using a synthetic table. This file pins the
PRODUCTION disposition of every site the guard used to baseline.

The hazard needs BOTH halves: (a) a partial index whose ``WHERE`` equates a column
to a scalar literal, AND (b) a production statement that binds THAT column with
``=`` (parameterized), cached and later re-bound to a different value while the
index appears. Measured disposition of the six baselined sites:

* ``idx_messages_assistant_calls_by_session`` — WHERE ``role = 'assistant'`` and
  production DOES bind ``role = ?`` (hermes_state_coverage, hermes_state_messages,
  agent/transcript_repair). Hazard REACHABLE -> COMPOSITE.
* ``idx_executions_occurrence`` — WHERE ``status='completed'`` and production DOES
  bind ``status=?`` (cron/executions recovery UPDATEs). Hazard REACHABLE -> COMPOSITE.
* ``idx_messages_display_page`` / ``_backfill`` / ``_identity`` and
  ``idx_messages_session_uid`` — WHERE columns ``active`` / ``compacted`` are NEVER
  bound with ``=`` by any production statement (measured: the only write is
  ``SET active = 0, compacted = ?`` in hermes_state_portability). Hazard
  UNREACHABLE, and the partial predicate is load-bearing for the display read —
  making them composite REGRESSES
  ``test_get_messages_include_compacted.py::test_display_paging_and_append_work_is_bounded``
  (the empty predicate lets the display pre-read probe short-circuit; a full
  composite makes it scan the whole transcript). They keep their partial form with
  the guard's sanctioned ``partial-index: ok`` marker.

Each test:
  1. builds a real state.db (``SessionDB``) / executions ledger with rows where the
     alternate bound value genuinely matches, so a mis-served plan reads 0;
  2. for a COMPOSITED site, drops the target index and re-creates it in the
     DANGEROUS partial spelling on a live cached connection — the pre-fix revision
     — and asserts it returns the WRONG rows (the RED arm, proving discrimination);
     then restores the SHIPPED composite and asserts correctness;
  3. for a KEPT-PARTIAL site, asserts the shipped DDL is still the partial form and
     that it is still the CHOSEN plan for its own consumer query.

The kept-partial markers carry an honesty condition — the columns their partial ``WHERE``
equates to a literal are never bound with ``=`` in a PREDICATE by production — and that
condition is what makes the hazard unreachable for them. It used to be enforced by nothing.
It now has two arms: the mechanical one in ``scripts/ci/check_partial_index_where.py``
(derived from the marked DDL, wired into ``scripts/check`` and CI) and the in-suite
behavioural one here, which applies the same condition to the statements the tree actually
hands to SQLite.
"""

from __future__ import annotations

import ast
import re
import sqlite3
import subprocess
from pathlib import Path

import pytest

from hermes_state import SessionDB

REPO_ROOT = Path(__file__).resolve().parents[2]

# The guard's own inventory scope, restated so the in-suite arm reads the same file set.
_GUARD_EXCLUDED_PREFIXES = (
    "tests/", "tests-js/", "evals/", "website/", "optional-skills/",
    "skills/", "native/", "node_modules/", "plugin-catalog/",
)


# --------------------------------------------------------------------------- #
# shape: a fresh state.db carries exactly the intended disposition
# --------------------------------------------------------------------------- #

def test_fresh_state_db_index_disposition(tmp_path):
    """A fresh state.db: composites for the reachable sites, partials for the rest.

    The shape-level companion to the behavioural arms: whatever the queries do, the
    shipped DDL must composite the two reachable sites and keep the four
    hazard-unreachable sites partial.
    """
    database = SessionDB(tmp_path / "state.db")
    try:
        rows = dict(database._conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'index'"
        ).fetchall())
        # Reachable sites, plus display_page (plan-identical either way), are composite.
        for name in ("idx_messages_assistant_calls_by_session", "idx_messages_display_page"):
            assert "WHERE" not in (rows[name] or "").upper(), rows.get(name)
        # Kept-partial sites stay partial, and their predicate is the known-safe-for-us
        # literal-equality shape (safe because the column is never bound with '=').
        for name in ("idx_messages_display_backfill", "idx_messages_display_identity",
                     "idx_messages_session_uid"):
            assert "WHERE" in (rows[name] or "").upper(), rows.get(name)
    finally:
        database.close()


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
            f"call{i}", 1000.0 + i,
            active, 0,
            None if backfill else f"idn{i // 10}".encode(),
            None if backfill else i,
            f"uid{i + 1}", f"pm{i}",
        ))
    conn.executemany(
        "INSERT INTO messages(id, session_id, role, content, tool_calls, timestamp,"
        " active, compacted, display_identity, display_order, message_uid, platform_message_id)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", rows,
    )
    conn.commit()


def _count(db: SessionDB, sql: str, params) -> int:
    return len(db._conn.execute(sql, params).fetchall())


def _shipped_sql(db: SessionDB, index_name: str) -> str:
    row = db._conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND name=?", (index_name,)
    ).fetchone()
    assert row is not None, f"index {index_name} is missing from a freshly created state.db"
    return row[0] or ""


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    _seed_messages(database)
    yield database
    database.close()


# --------------------------------------------------------------------------- #
# COMPOSITED sites: the hazard is reachable, so the composite is required
# --------------------------------------------------------------------------- #

# Every index on ``messages`` that could serve the consumer query instead of the target,
# dropped for the RED arm so the dangerous spelling is provably the CHOSEN plan —
# otherwise the optimizer picks a safe sibling and the arm reads GREEN for the wrong
# reason (a vacuous pin).
_MESSAGES_INDEXES = (
    "idx_messages_session", "idx_messages_session_id", "idx_messages_session_active",
    "idx_messages_assistant_calls_by_session", "idx_messages_platform_msg_id",
    "idx_messages_display_page", "idx_messages_display_backfill",
    "idx_messages_display_identity", "idx_messages_active_null", "idx_messages_session_uid",
)


def test_assistant_calls_is_composite_and_immune(db):
    """``role = ?`` IS bound in production, so this site is composited and the partial mis-serves."""
    index_name = "idx_messages_assistant_calls_by_session"
    dangerous = (
        "CREATE INDEX idx_messages_assistant_calls_by_session ON messages(session_id)"
        " WHERE role = 'assistant' AND tool_calls IS NOT NULL"
    )
    sql = ("SELECT id FROM messages WHERE session_id = ? AND role = ?"
           " AND tool_calls IS NOT NULL ORDER BY id")
    prime, other = ("s1", "assistant"), ("s1", "user")

    shipped = _shipped_sql(db, index_name)
    assert "WHERE" not in shipped.upper(), f"{index_name} must ship composite; found: {shipped}"
    assert "(session_id, role, tool_calls)" in shipped, shipped

    for name in _MESSAGES_INDEXES:
        db._conn.execute(f"DROP INDEX IF EXISTS {name}")
    db._conn.commit()
    primed = _count(db, sql, prime)
    assert primed > 0, "fixture must make the primed value match rows"
    db._conn.execute(dangerous)
    db._conn.commit()
    assert _count(db, sql, prime) == primed
    assert _count(db, sql, other) == 0, (
        f"the dangerous partial {index_name} must mis-serve the reused statement (0 rows)"
    )

    db._conn.execute(f"DROP INDEX IF EXISTS {index_name}")
    db._conn.execute(shipped if shipped.upper().startswith("CREATE INDEX")
                     else "CREATE INDEX " + shipped)
    db._conn.commit()
    assert _count(db, sql, prime) == primed
    assert _count(db, sql, other) > 0, (
        f"the shipped composite {index_name} must return rows for the other bound value"
    )


def test_executions_occurrence_is_composite_and_immune(tmp_path):
    """``status=?`` IS bound by the recovery UPDATEs, so this site is composited."""
    from cron import executions

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
                (f"e{i}", "completed" if i % 2 else "failed",
                 f"2026-01-01T00:00:{i % 60:02d}+00:00"),
            )

    sql = ("SELECT id FROM executions WHERE job_id = ? AND status = ?"
           " AND scheduled_instant IS NOT NULL")
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


# --------------------------------------------------------------------------- #
# KEPT-PARTIAL sites: the hazard is unreachable and the partial is plan-load-bearing
# --------------------------------------------------------------------------- #

_KEPT_PARTIAL = (
    "idx_messages_display_backfill",
    "idx_messages_display_identity",
    "idx_messages_session_uid",
)

# The columns these partial predicates key on. Production must never bind one with '='
# IN A PREDICATE. That condition is what makes the hazard unreachable for the kept-partial
# sites, and ``scripts/ci/check_partial_index_where.py`` now enforces it (deriving the same
# set from the marked DDL) — the guard is the mechanical arm; the two tests below are the
# in-suite behavioural arm, so a guard that silently stopped running would still show up.
_NEVER_BOUND_COLUMNS = ("active", "compacted")

# Every placeholder spelling SQLite accepts, and every position a bind can sit in. The
# point is that the SET-target exclusion below is DISCRIMINATING: ``UPDATE messages SET
# active = ?`` is a shipped production statement, so a matcher that did not exclude SET
# targets would report it and this file would fail on its own repo.
_PLACEHOLDER = r"(?:\?\d*|:[A-Za-z_]\w*|@[A-Za-z_]\w*|\$[A-Za-z_]\w*)"
_BOUND_IN_PREDICATE = re.compile(
    r"(?<![!<>=])(?:\b\w+\.)?\b(%s)\b\s*=\s*%s" % ("|".join(_NEVER_BOUND_COLUMNS), _PLACEHOLDER)
)
_BOUND_IN_PREDICATE_REVERSED = re.compile(
    _PLACEHOLDER + r"\s*=\s*(?:\b\w+\.)?\b(%s)\b" % "|".join(_NEVER_BOUND_COLUMNS)
)
_SET_KEYWORD = re.compile(r"\bSET\b", re.IGNORECASE)
_SET_CLAUSE_END = re.compile(
    r"\b(WHERE|FROM|RETURNING|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|VALUES)\b", re.IGNORECASE
)


def _predicate_binds(sql: str) -> list[str]:
    """Marked columns bound with a placeholder in PREDICATE position in *sql*.

    Statements are split on ``;`` AFTER line comments are blanked, so a ``;`` inside a
    comment cannot split one, and a ``-- …`` between a name and the rest of the statement
    does not hide it. The ``SET`` target list is excluded: a SET target is a write, not a
    predicate the planner can fold into the partial index — measured, ``UPDATE t SET a =
    ?`` does not reproduce the anomaly (the written data matches a fresh connection).
    """
    blanked = re.sub(r"--[^\n]*", lambda m: " " * len(m.group(0)), sql)
    found: list[str] = []
    for stmt in blanked.split(";"):
        spans: list[tuple[int, int]] = []
        for match in _SET_KEYWORD.finditer(stmt):
            end = _SET_CLAUSE_END.search(stmt, match.end())
            spans.append((match.end(), end.start() if end else len(stmt)))
        for pattern in (_BOUND_IN_PREDICATE, _BOUND_IN_PREDICATE_REVERSED):
            for match in pattern.finditer(stmt):
                column = match.group(1).lower()
                if any(start <= match.start() < end for start, end in spans):
                    continue
                if column not in found:
                    found.append(column)
    return found


def test_no_production_statement_binds_a_kept_partial_column():
    """The markers' honesty condition, driven over the real statements the tree ships.

    Source-scanning is banned in ``tests/AGENTS.md``, and this is not that: it does not
    assert anything about the SHAPE of the source. It reads the statements the code
    EXECUTES (the string constants the modules hand to SQLite) and applies the marker's
    own condition to them — the same predicate the guard applies, so the guard going
    missing is caught here rather than only in CI.

    GREEN today; RED the moment a production statement binds ``active`` or ``compacted``
    with a placeholder in a predicate on ``messages``.
    """
    offenders: list[str] = []
    for relpath in _production_python_files():
        source = (REPO_ROOT / relpath).read_text(encoding="utf-8-sig", errors="replace")
        for lineno, text in _string_constants(source):
            if "messages" not in text:
                continue
            for column in _predicate_binds(text):
                offenders.append(f"{relpath}:{lineno}: {column} = <placeholder>")
    assert not offenders, (
        "a kept-partial index's `partial-index: ok` marker claims these columns are never"
        " bound with `=` in a predicate, but production does:\n  " + "\n  ".join(offenders)
    )


def test_the_predicate_scan_excludes_a_set_target():
    """The scan above is only meaningful if its SET exclusion actually works.

    ``hermes_state_portability`` ships ``UPDATE messages SET active = 0, compacted = ?``
    — a real bind of a marked column, in a real production statement. If the exclusion
    were removed, the test above would red on its own repo for a bind that measured NOT
    to reproduce the anomaly; this pins that it does not.
    """
    assert _predicate_binds("UPDATE messages SET active = 0, compacted = ? WHERE id = ?") == []
    # …and that the same bind in PREDICATE position is still caught.
    assert _predicate_binds("SELECT id FROM messages WHERE active = ?") == ["active"]
    assert _predicate_binds("SELECT id FROM messages WHERE ? = compacted") == ["compacted"]


def _production_python_files() -> list[str]:
    """Tracked production ``*.py`` — the same inventory the guard uses."""
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z", "--", "*.py"],
        capture_output=True, check=True,
    ).stdout.decode(errors="replace").split("\0")
    return sorted(
        n.replace("\\", "/") for n in out
        if n.endswith(".py") and not n.replace("\\", "/").startswith(_GUARD_EXCLUDED_PREFIXES)
    )


def _string_constants(source: str):
    """``(lineno, text)`` for every string constant, f-strings contributing literal parts."""
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            yield node.lineno, node.value
        elif isinstance(node, ast.JoinedStr):
            yield node.lineno, "".join(
                part.value for part in node.values
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
            )


def test_display_indexes_keep_their_partial_form(db):
    """The four kept-partial sites still ship a partial (not composite) DDL.

    Making them composite is a regression, not a fix: the display read's paging budget
    test (``tests/hermes_state/test_get_messages_include_compacted.py``) reds because the
    empty predicate is what lets the pre-read probe short-circuit.
    """
    for index_name in _KEPT_PARTIAL:
        shipped = _shipped_sql(db, index_name)
        assert "WHERE" in shipped.upper(), (
            f"{index_name} must keep its partial form (composite regresses the paging budget);"
            f" found: {shipped}"
        )


def test_kept_partial_indexes_still_serve_their_consumer(db):
    """A kept-partial index must still be the CHOSEN plan for its own consumer query.

    The partial form is only justified while it actually serves the read; a predicate
    drift that made the index unusable would leave the marker dishonest. Each consumer is
    primed with the value its predicate describes, and the plan must name the index.
    """
    cases = (
        ("idx_messages_display_page",
         "SELECT id FROM messages WHERE session_id = ? AND (active = 1 OR compacted = 1)"
         " AND display_order IS NOT NULL", ("s1",)),
        ("idx_messages_display_backfill",
         "SELECT id FROM messages WHERE session_id = ? AND (active = 1 OR compacted = 1)"
         " AND (display_order IS NULL OR display_identity IS NULL)", ("s1",)),
        ("idx_messages_display_identity",
         "SELECT id FROM messages WHERE session_id = ? AND (active = 1 OR compacted = 1)"
         " AND display_identity IS NOT NULL", ("s1",)),
        ("idx_messages_session_uid",
         "SELECT id FROM messages WHERE session_id = ? AND active = 1 AND message_uid = ?",
         ("s1", "uid2")),
    )
    for index_name, sql, params in cases:
        plan = [row[3] for row in db._conn.execute("EXPLAIN QUERY PLAN " + sql, params)]
        # The kept-partial index (or a sibling that carries the same predicate) must be
        # the chosen plan; a bare table SCAN means the partial no longer serves the read.
        assert not any("SCAN messages" in p for p in plan), (
            f"{index_name} no longer serves its consumer (plan={plan}); the `partial-index: ok` marker"
            " assumes it does"
        )
