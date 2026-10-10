"""Minimal repro fixture: a PARTIAL index mis-serves a reused parameterized statement.

Measured on this fleet's SQLite builds (3.53.1, 3.47.1, 3.37.2 across Python
3.10/3.11/3.12/3.14): a connection that prepares and runs the shared read, then
CREATES a partial index whose WHERE holds a literal, then RE-RUNS the same
cached statement with the same bound value, returns **0 rows** for every other
bound value of the compared column. It is silent — wrong rows, never an error.
A composite index does not do this; a fresh connection is always correct;
``PRAGMA integrity_check`` reads ``ok``; ``cached_statements=0`` removes it.

This module is imported by ``tests/hermes_cli/test_partial_index_where_anomaly.py``
so the characterization is part of the repo's own test tooling rather than a
scratch script, and so a future SQLite upgrade that fixes the anomaly is
detected here rather than by a production wrong-rows report.

Run it directly for the human-readable table::

    python tests/fixtures/partial_index_statement_cache_anomaly.py
"""

from __future__ import annotations

import os
import sqlite3
import tempfile

# The fleet's own demotion-tally read, verbatim in shape: it binds ``outcome``
# with ``=`` and filters on the column the dangerous partial index would key on.
WALL_QUERY = (
    "SELECT task_id, ended_at, worker_pid, metadata FROM task_runs "
    "WHERE outcome = ? AND ended_at IS NOT NULL AND ended_at >= ? AND metadata IS NOT NULL"
)
RATE_LIMITED = "rate_limited"
COMPLETED = "completed"
ROWS = 400  # 200 rate_limited, 200 completed

_DDL = (
    "CREATE TABLE task_runs(id INTEGER PRIMARY KEY, task_id TEXT, status TEXT,"
    " ended_at INT, outcome TEXT, metadata TEXT, worker_pid INT);"
)


def _build(path: str) -> None:
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(path + suffix)
        except FileNotFoundError:
            pass
    conn = sqlite3.connect(path)
    conn.executescript(_DDL)
    for k in range(ROWS):
        conn.execute(
            "INSERT INTO task_runs VALUES(?,?,?,?,?,?,?)",
            (k, f"t{k}", "done", 1000 + k,
             RATE_LIMITED if k % 2 else COMPLETED, "{}", 100 + k),
        )
    conn.commit()
    conn.close()


def scenario(index_sql: str, *, cached_statements: int = 128, db_dir: str) -> dict:
    """Prime the read, create *index_sql*, re-run, then bind the OTHER value.

    Returns ``{"primed": n, "reused": n, "other": n, "expected": 200}`` — the
    shape the caller asserts on. ``other != expected`` is the anomaly.
    """
    path = os.path.join(db_dir, "repro.db")
    _build(path)
    conn = sqlite3.connect(path, cached_statements=cached_statements)

    def count(value: str) -> int:
        return len(conn.execute(WALL_QUERY, (value, 0)).fetchall())

    primed = count(RATE_LIMITED)
    if index_sql:
        conn.execute(index_sql)
        conn.commit()
    # The poison step: the cached statement runs again with the SAME bound value
    # the index's literal describes.
    count(RATE_LIMITED)
    result = {
        "primed": primed,
        "reused": count(RATE_LIMITED),
        "other": count(COMPLETED),
        "expected": ROWS // 2,
    }
    conn.close()
    return result


DANGEROUS_PARTIAL = (
    "CREATE INDEX ix_dangerous ON task_runs(ended_at) WHERE outcome='rate_limited'"
)
SHIPPED_COMPOSITE = "CREATE INDEX ix ON task_runs(outcome, ended_at)"


def _main() -> int:
    print("python", __import__("sys").version.split()[0], "sqlite", sqlite3.sqlite_version)
    with tempfile.TemporaryDirectory(prefix="partial-index-anomaly-") as tmp:
        for label, sql, cached in (
            ("partial(ended_at) WHERE outcome='rate_limited'", DANGEROUS_PARTIAL, 128),
            ("composite(outcome, ended_at)  [shipped]", SHIPPED_COMPOSITE, 128),
            ("partial, cached_statements=0", DANGEROUS_PARTIAL, 0),
        ):
            r = scenario(sql, cached_statements=cached, db_dir=tmp)
            flag = "" if r["other"] == r["expected"] else "   <-- WRONG"
            print(f"  {label:46s} primed={r['primed']} reused={r['reused']} "
                  f"other={r['other']} (expected {r['expected']}){flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
