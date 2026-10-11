"""Arm 2 of the partial-index WHERE guard: the ``partial-index: ok`` markers are falsifiable.

Split out of ``scripts/ci/check_partial_index_where.py``, whose ``main`` had grown to ``CC`` 24
against the repo's target of 20 when this arm landed (16 before it). The facade keeps arm 1 (the
DDL shape), the inventory and the report; this sibling owns arm 2 **and the WHERE/DDL matcher
primitives both arms read** — the facade imports them back, so the dependency runs one way and the
two can never form an import cycle. (The primitives live here rather than in the facade because a
facade-run-as-``__main__`` that a sibling late-imports would load a second copy of the facade
module; keeping the shared code on the sibling side avoids that.)

A ``partial-index: ok`` marker asserts the indexed column is never bound with ``=`` by a
production statement, and that assertion is the ONLY reason the anomaly is unreachable for the
kept-partial sites. Nothing enforced it: the marker could stay present — arm 1 green — while a new
writer bound the column with ``=``, making every marker on that column silently FALSE. This arm
derives the column set FROM the marked DDL (so a new marker widens what is enforced automatically)
and refuses any tracked production string constant that binds one of those columns with a
**placeholder** (``?``, ``?1``, ``:name``, ``@name``, ``$name``) in a **predicate** position —
outside a ``SET`` target list, which is a write rather than a predicate the planner can fold into
the partial index — in a statement that references the marked index's TABLE.

Measured: the shipped tree binds neither ``active`` nor ``compacted`` with a placeholder in any
position, so it is GREEN; a synthetic ``WHERE ... active = ?`` on ``messages`` reds it and names
the site.

Measured detail that matters for the allow-list: the hazard is NOT restricted to the indexed
column (``ON t(b) WHERE a='x'`` mis-serves), and the safe spellings of the same predicate are
``IS NOT NULL`` / ``!=`` / ``>=`` / ``LIKE`` / ``GLOB`` / a function call / ``IN`` with two-or-more
literals. Only ``=`` and a single-literal ``IN`` reproduce it. The anomaly needs the index to
APPEAR while the statement is already cached, so a bind only matters where a partial index could
serve it; the ``cached_statements`` size and the planner's choice are what make it reachable, and
a fresh connection is always correct.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# The repo root on sys.path: this module is imported as ``scripts.ci.…`` by the facade (run both
# as a script and under pytest), and ``scripts/ci`` carries no ``__init__.py`` — the namespace
# package needs the root, which is not sys.path[0] when the entry point is the facade script.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MARKER = "partial-index: ok"

# --- shared matcher primitives (arm 1 and arm 2 both read these) -----------------------------

_CREATE_INDEX = re.compile(
    r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r"([A-Za-z_]\w*)\s+ON\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*WHERE\s+(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_SQL_COMMENT = re.compile(r"--[^\n]*")
# ``col = <scalar literal>``, excluding the comparison operators that do NOT
# reproduce the anomaly (``!=``, ``<=``, ``>=``, ``<>``). A function call or a
# column on the right-hand side is left alone: neither reproduces it.
_EQUALS_LITERAL = re.compile(
    r"(?<![!<>=])\b([A-Za-z_][\w.]*)\s*=\s*('(?:[^']|'')*'|-?\d+(?:\.\d+)?)(?![.\d])"
)
# ``col IN (<exactly one literal>)`` — two or more literals measured safe.
_IN_ONE_LITERAL = re.compile(
    r"\b([A-Za-z_][\w.]*)\s+IN\s*\(\s*('(?:[^']|'')*'|-?\d+(?:\.\d+)?)\s*\)",
    re.IGNORECASE,
)


def is_marked(lines: list[str], lineno: int) -> bool:
    """True when ``lineno`` (1-based) or the line above carries the opt-out marker."""
    for idx in (lineno - 1, lineno - 2):
        if 0 <= idx < len(lines) and MARKER in lines[idx]:
            return True
    return False


def where_clauses(sql: str):
    """Yield ``(index_name, table, where, offset)`` for every partial index in *sql*.

    ``sql`` may be a whole schema script or one implicitly-concatenated Python
    string; statements are split on ``;`` so a multi-index script reports each
    index separately. ``offset`` is the character index of the INDEX NAME inside
    *sql* (not of the enclosing statement), so the caller can report the index's own
    line rather than the blank line a ``;``-split boundary lands on.

    SQL line comments are BLANKED to spaces (newlines preserved, so offsets stay
    exact) BEFORE the ``;`` split. Both halves matter: a schema may place an inline
    ``-- …`` between the index name and ``ON`` (the ``partial-index: ok`` opt-out
    itself is written that way), and a matcher requiring only whitespace there
    silently skips the statement — which is how a marked index stops being seen by
    this arm at all. Blanking first also keeps a ``;`` INSIDE a comment from
    splitting a statement in half. The marker check reads the RAW file lines, so
    blanking here does not affect it.
    """
    cursor = 0
    blanked_sql = _SQL_COMMENT.sub(lambda m: " " * len(m.group(0)), sql)
    for stmt in blanked_sql.split(";"):
        match = _CREATE_INDEX.search(stmt)
        if match:
            name_pos = stmt.find(match.group(1), match.start())
            yield match.group(1), match.group(2), match.group(4).strip(), cursor + name_pos
        cursor += len(stmt) + 1


def dangerous_columns(where: str) -> list[tuple[str, str, str]]:
    """``(column, literal, form)`` for each literal comparison that reproduces it."""
    found = [(m.group(1), m.group(2), "=") for m in _EQUALS_LITERAL.finditer(where)]
    found += [(m.group(1), m.group(2), "IN") for m in _IN_ONE_LITERAL.finditer(where)]
    return found


# --- arm 2: the marker claim ----------------------------------------------------------------

# Placeholder spellings SQLite accepts in a parameterized statement.
_PLACEHOLDER = r"(?:\?\d*|:[A-Za-z_]\w*|@[A-Za-z_]\w*|\$[A-Za-z_]\w*)"

# ``col = <placeholder>`` in PREDICATE position — the bind that makes a marked
# partial index's opt-out false. ``(?<![!<>=])`` keeps the safe comparison
# operators (``!=``/``<=``/``>=``/``<>``) out, which do NOT reproduce the anomaly.
_BOUND_EQUALS = re.compile(
    r"(?<![!<>=])(?:\b\w+\.)?\b([A-Za-z_]\w*)\b\s*=\s*" + _PLACEHOLDER
)
# The reversed operand order, ``<placeholder> = col``, reproduces it identically
# (measured) and is a different spelling of the same bind, so it needs its own arm.
_BOUND_EQUALS_REVERSED = re.compile(
    _PLACEHOLDER + r"\s*=\s*(?:\b\w+\.)?\b([A-Za-z_]\w*)\b"
)

# A ``SET`` clause is a write target, not a predicate the planner can fold into a
# partial index — ``UPDATE t SET a = ?`` measured NOT to reproduce the anomaly
# (the written data matches a fresh connection's). Bound SET targets are therefore
# out of scope for the marker: the marker claims the column is never a PREDICATE.
_SET_KEYWORD = re.compile(r"\bSET\b", re.IGNORECASE)
_SET_CLAUSE_END = re.compile(
    r"\b(WHERE|FROM|RETURNING|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|VALUES)\b", re.IGNORECASE
)


def set_clause_spans(stmt: str) -> list[tuple[int, int]]:
    """Character spans of every ``SET`` target list in *stmt* (exclusive of ``WHERE`` …)."""
    spans: list[tuple[int, int]] = []
    for match in _SET_KEYWORD.finditer(stmt):
        end = _SET_CLAUSE_END.search(stmt, match.end())
        spans.append((match.end(), end.start() if end else len(stmt)))
    return spans


def bound_marked_columns(stmt: str, columns: frozenset[str]) -> list[str]:
    """Marked *columns* bound with ``=`` in PREDICATE position anywhere in *stmt*.

    Both operand orders are covered: ``col = ?`` and ``? = col`` measured to
    reproduce the anomaly identically. A ``SET`` target is not a predicate, so it is
    skipped — see the module note on why.
    """
    spans = set_clause_spans(stmt)
    found: list[str] = []
    for pattern in (_BOUND_EQUALS, _BOUND_EQUALS_REVERSED):
        for match in pattern.finditer(stmt):
            column = match.group(1).lower()
            if column not in columns:
                continue
            if any(start <= match.start() < end for start, end in spans):
                continue
            if column not in found:
                found.append(column)
    return found


def bind_candidates(constants: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """The ``(lineno, text)`` subset that binds SOME column with ``=`` at all.

    A cheap pre-filter, not the decision: the marker's column set is only known once
    the whole tree has been read, so the per-file pass records every constant
    carrying the shape and :func:`scan_query_side` narrows it to the marked columns.
    """
    return [
        (lineno, text)
        for lineno, text in constants
        if _BOUND_EQUALS.search(text) or _BOUND_EQUALS_REVERSED.search(text)
    ]


def marked_partial_columns(constants: list[tuple[int, str]], lines_by_file: dict[str, list[str]],
                           relpath: str) -> tuple[frozenset[str], frozenset[str]]:
    """``(columns, tables)`` the ``partial-index: ok`` markers actually promise about.

    Derived FROM the marked DDL rather than hard-coded, so a new marked partial
    index (on a new column, a new table) widens what arm 2 enforces automatically —
    a hard-coded ``("active", "compacted")`` would silently stop covering it. The
    columns are the ones the marker's honesty claim is about: those the partial
    ``WHERE`` equates to a scalar literal.
    """
    columns: set[str] = set()
    tables: set[str] = set()
    lines = lines_by_file.get(relpath, [])
    for lineno, text in constants:
        for _name, table, where, offset in where_clauses(text):
            stmt_lineno = lineno + text[:offset].count("\n")
            if not is_marked(lines, stmt_lineno):
                continue
            tables.add(table.lower())
            columns.update(column.lower() for column, _literal, _form in dangerous_columns(where))
    return frozenset(columns), frozenset(tables)


def scan_query_side(relpath: str, constants: list[tuple[int, str]], columns: frozenset[str],
                    tables: frozenset[str]) -> list[str]:
    """``path:line: ...`` for each production bind that makes a marker false.

    Only statements that reference a MARKED index's table are candidates: a bind of
    the same column name on some other table cannot be served by that partial index,
    so flagging it would be a false positive the marker never claimed.
    """
    sites: list[str] = []
    for lineno, text in constants:
        if not any(table in text.lower() for table in tables):
            continue
        for stmt in text.split(";"):
            if not any(re.search(rf"\b{table}\b", stmt, re.IGNORECASE) for table in tables):
                continue
            bound = bound_marked_columns(stmt, columns)
            if not bound:
                continue
            flat = " ".join(stmt.split())
            for column in bound:
                sites.append(
                    f"{relpath}:{lineno}: `{column} = ?` in a production statement — a kept-partial"
                    f" index whose `{MARKER}` marker claims `{column}` is never bound with `=`"
                    f" (a reused parameterized statement bound to a different value then returns"
                    f" WRONG rows). Statement: {flat[:160]}"
                )
    return sites
