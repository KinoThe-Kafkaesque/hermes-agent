#!/usr/bin/env python3
"""Fail when a partial index's ``WHERE`` compares a column to a scalar literal,
and when a ``partial-index: ok`` opt-out stops being true.

Two arms, one file. Both are needed; either alone is a green light for a
different hole.

**Arm 1 (DDL shape).** Refuses a partial index whose ``WHERE`` compares a column
to a scalar literal. A ``CREATE INDEX ... WHERE <col> = <literal>`` (or
``<col> IN (<one literal>)``) is not just a planner hint: on this fleet's SQLite
builds it makes a **reused parameterized statement** return WRONG rows. Measured
2026-10-09/10 on SQLite 3.53.1 and 3.37.2 (Python 3.10/3.11/3.12/3.14), silent —
wrong rows, never an error:

    prepare  SELECT ... WHERE outcome = ? AND ended_at >= ?
    run it with outcome='rate_limited'      -> 200 rows
    CREATE INDEX ix ON task_runs(ended_at) WHERE outcome='rate_limited'
    run it again with outcome='rate_limited'  (same cached statement)
    run it with outcome='completed'         -> 0 rows   (200 is correct)

The trigger is the *combination*: the statement is already in the connection's
statement cache when the partial index appears (a migration on a live board, a
sibling connection, or an ``ANALYZE``), and the newly-bound value differs from
the index's literal. ``cached_statements=0`` removes it; a fresh connection is
always correct; ``PRAGMA integrity_check`` reads ``ok``; the composite
``(outcome, ended_at)`` form is exact and gives the same plan. See
``hermes_cli/kanban_db_connect.py`` (the ``idx_runs_outcome_ended`` comment) and
``website/docs/user-guide/features/kanban.md``.

What arm 1 refuses is the SHAPE, not a measured instance. The live tree no
longer carries any UNMARKED such site: each was either replaced with its composite
form (``hermes_state_common.py``'s ``idx_messages_assistant_calls_by_session``,
``cron/executions.py``'s ``idx_executions_occurrence``, and the
``_heal_dangerous_partial_indexes`` migration that retires the old spelling on an
already-open state.db) or, where the partial predicate is load-bearing for a
production read AND the column is never bound with ``=``, kept partial with the
sanctioned ``partial-index: ok`` marker (the display indexes and
``idx_messages_session_uid`` — see the note in ``DEFERRED_INDEX_SQL``). ``_BASELINE``
stays as the mechanism — ``scripts/check_no_tmp_literals.py``'s shape — so a future
file that GAINS a hit fails, and a stale/overstated entry is reported as an advisory
rather than turning CI red for a parallel clean-up.

**Arm 2 (the markers are falsifiable).** A ``partial-index: ok`` marker asserts the
indexed column is never bound with ``=`` by a production statement, and that
assertion is the ONLY reason the anomaly is unreachable for the kept-partial sites.
Nothing enforced it: the marker could stay present — arm 1 green — while a new
writer bound the column with ``=``, making every marker on that column silently
FALSE. Arm 2 derives the column set FROM the marked DDL (so a new marker widens
what is enforced automatically) and refuses any tracked production string constant
that binds one of those columns with a **placeholder** (``?``, ``?1``, ``:name``,
``@name``, ``$name``) in a **predicate** position — outside a ``SET`` target list,
which is a write rather than a predicate the planner can fold into the partial
index — in a statement that references the marked index's TABLE. Measured: the
shipped tree binds neither ``active`` nor ``compacted`` with a placeholder in any
position, so it is GREEN; a synthetic ``WHERE ... active = ?`` on ``messages`` reds
it and names the site.

Measured detail that matters for the allow-list: the hazard is NOT restricted to
the indexed column (``ON t(b) WHERE a='x'`` mis-serves), and the safe spellings
of the same predicate are ``IS NOT NULL`` / ``!=`` / ``>=`` / ``LIKE`` / ``GLOB``
/ a function call / ``IN`` with two-or-more literals. Only ``=`` and a
single-literal ``IN`` reproduce it. The anomaly needs the index to APPEAR while the
statement is already cached, so a bind only matters where a partial index could
serve it; the ``cached_statements`` size and the planner's choice are what make it
reachable, and a fresh connection is always correct.

Opt a deliberate site out with ``partial-index: ok — <why>`` on the offending
line or the line directly above it (Python comment). Legitimate reasons: the
column is never bound with ``=`` by any production statement, or the index is
retired by a follow-up card. "It has not bitten yet" is not one.

Exit codes: 0 = clean, 1 = offenders found (exact ``path:line`` sites on stdout),
2 = inventory/analysis error.

Usage:
    python scripts/ci/check_partial_index_where.py [repo_root]
    python scripts/ci/check_partial_index_where.py --all            # burn-down view
    python scripts/ci/check_partial_index_where.py --print-baseline
    python scripts/ci/check_partial_index_where.py --strict-baseline

Wired into ``scripts/check`` (the ``partial-index-where`` guard) and into
``.github/workflows/partial-index-where-guard.yml`` (blocking, called from
``ci.yaml``). Contract tests: ``tests/scripts/test_partial_index_where_guard.py``.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

MARKER = "partial-index: ok"

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


def _set_clause_spans(stmt: str) -> list[tuple[int, int]]:
    """Character spans of every ``SET`` target list in *stmt* (exclusive of ``WHERE`` …)."""
    spans: list[tuple[int, int]] = []
    for match in _SET_KEYWORD.finditer(stmt):
        end = _SET_CLAUSE_END.search(stmt, match.end())
        spans.append((match.end(), end.start() if end else len(stmt)))
    return spans


def _bound_marked_columns(stmt: str, columns: frozenset[str]) -> list[str]:
    """Marked *columns* bound with ``=`` in PREDICATE position anywhere in *stmt*.

    Both operand orders are covered: ``col = ?`` and ``? = col`` measured to
    reproduce the anomaly identically. A ``SET`` target is not a predicate, so it is
    skipped — see the module note on why.
    """
    spans = _set_clause_spans(stmt)
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

# Burn-down list: pre-existing dangerous-shaped sites, keyed by repo-relative
# path -> hit count at the time this guard landed. Now EMPTY — the six sites the
# guard shipped with were retired by replacing each partial index with its
# composite form (see the module docstring). The mechanism stays so a file that
# GAINS a hit fails and a stale entry is an advisory.
_BASELINE: dict[str, int] = {}

# Directories whose ``CREATE INDEX`` statements are not production DDL: tests may
# build hostile fixtures on purpose, docs quote the shape, evals are data.
_EXCLUDED_PREFIXES = (
    "tests/",
    "tests-js/",
    "evals/",
    "website/",
    "optional-skills/",
    "skills/",
    "native/",
    "node_modules/",
    "plugin-catalog/",
)

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


def _is_marked(lines: list[str], lineno: int) -> bool:
    """True when ``lineno`` (1-based) or the line above carries the opt-out marker."""
    for idx in (lineno - 1, lineno - 2):
        if 0 <= idx < len(lines) and MARKER in lines[idx]:
            return True
    return False


def _where_clauses(sql: str):
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


def _dangerous_columns(where: str) -> list[tuple[str, str, str]]:
    """``(column, literal, form)`` for each literal comparison that reproduces it."""
    found = [(m.group(1), m.group(2), "=") for m in _EQUALS_LITERAL.finditer(where)]
    found += [(m.group(1), m.group(2), "IN") for m in _IN_ONE_LITERAL.finditer(where)]
    return found


def _python_string_constants(path: Path):
    """``(lineno, text)`` for every string constant, or an error string.

    AST-based, so prose in comments and docstrings is not SQL by construction
    (a docstring that quotes the dangerous shape is documentation, not a schema).
    f-strings contribute their literal parts, so an interpolated statement is still
    seen. Returns ``(constants, error)`` where exactly one is populated; a file that
    will not parse is an inventory error, never a silent pass.
    """
    try:
        source = path.read_text(encoding="utf-8-sig", errors="replace")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError, ValueError) as exc:
        return [], f"parse failed: {exc}"
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))
    constants: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            constants.append((node.lineno, node.value))
        elif isinstance(node, ast.JoinedStr):
            text = "".join(
                part.value for part in node.values
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
            )
            constants.append((node.lineno, text))
    return constants, None


def _marked_partial_columns(constants: list[tuple[int, str]], lines_by_file: dict[str, list[str]],
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
        for _name, table, where, offset in _where_clauses(text):
            stmt_lineno = lineno + text[:offset].count("\n")
            if not _is_marked(lines, stmt_lineno):
                continue
            tables.add(table.lower())
            columns.update(column.lower() for column, _literal, _form in _dangerous_columns(where))
    return frozenset(columns), frozenset(tables)


def _scan_query_side(relpath: str, constants: list[tuple[int, str]], columns: frozenset[str],
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
            bound = _bound_marked_columns(stmt, columns)
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


def _tracked_python_files(root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", "*.py"],
        capture_output=True,
        timeout=300,
    )
    if proc.returncode != 0:
        sys.stderr.write(
            f"inventory failed: git ls-files exited {proc.returncode}: "
            f"{proc.stderr.decode(errors='replace').strip()}\n"
        )
        raise SystemExit(2)
    names = proc.stdout.decode(errors="replace").split("\0")
    files = sorted(
        n.replace("\\", "/")
        for n in names
        if n.endswith(".py") and not n.replace("\\", "/").startswith(_EXCLUDED_PREFIXES)
    )
    if not files:
        sys.stderr.write(
            f"inventory failed: no tracked production *.py files under {root} — "
            "refusing to pass an empty inventory\n"
        )
        raise SystemExit(2)
    return files


def _scan_file(root: Path, relpath: str) -> tuple[list[str], list[tuple[int, str]], frozenset[str], frozenset[str], list[str]]:
    """``(ddl_sites, constants, columns, tables, errors)`` for one file.

    ``ddl_sites`` is arm 1 (a partial index whose WHERE equates a column to a
    literal). ``columns``/``tables`` are what the ``partial-index: ok`` markers in
    this file promise about, so the caller can enforce them TREE-WIDE — the marker
    lives in the schema module while the statement that would falsify it can be in
    any file. ``constants`` is returned so the query-side pass does not re-parse.
    """
    path = root / relpath
    ddl_sites: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError as exc:
        return [], [], frozenset(), frozenset(), [f"{relpath}: read failed: {exc}"]
    constants, error = _python_string_constants(path)
    if error is not None:
        return [], [], frozenset(), frozenset(), [f"{relpath}: {error}"]
    ddl_constants = [(lineno, text) for lineno, text in constants if "CREATE INDEX" in text.upper()]
    for lineno, text in ddl_constants:
        for name, table, where, offset in _where_clauses(text):
            stmt_lineno = lineno + text[:offset].count("\n")
            for column, literal, form in _dangerous_columns(where):
                if _is_marked(lines, stmt_lineno):
                    continue
                ddl_sites.append(
                    f"{relpath}:{stmt_lineno}: partial index {name} ON {table} — WHERE "
                    f"compares {column} to the literal {literal} ({form}); a reused "
                    "parameterized statement bound to a different value returns WRONG "
                    "rows on this fleet's SQLite (see the guard's docstring)"
                )
    columns, tables = _marked_partial_columns(ddl_constants, {relpath: lines}, relpath)
    return ddl_sites, constants, columns, tables, []


def _print_baseline(counts: dict[str, int]) -> None:
    body = "".join(f'    "{path}": {count},\n' for path, count in sorted(counts.items()))
    print("_BASELINE: dict[str, int] = {\n" + body + "}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("root", nargs="?", default=".", help="git checkout to scan")
    parser.add_argument("--all", action="store_true",
                        help="ignore _BASELINE; report every dangerous site")
    parser.add_argument("--print-baseline", action="store_true",
                        help="print a _BASELINE literal matching the current tree")
    parser.add_argument("--strict-baseline", action="store_true",
                        help="fail on stale/overstated _BASELINE entries too")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"no such directory: {root}", file=sys.stderr)
        return 2

    ddl_per_file: dict[str, list[str]] = {}
    bind_candidates: list[tuple[str, int, str]] = []
    marked_columns: set[str] = set()
    marked_tables: set[str] = set()
    errors: list[str] = []
    for relpath in _tracked_python_files(root):
        ddl_sites, constants, columns, tables, file_errors = _scan_file(root, relpath)
        errors.extend(file_errors)
        if ddl_sites:
            ddl_per_file[relpath] = ddl_sites
        marked_columns.update(columns)
        marked_tables.update(tables)
        for lineno, text in constants:
            if _BOUND_EQUALS.search(text) or _BOUND_EQUALS_REVERSED.search(text):
                bind_candidates.append((relpath, lineno, text))

    if errors:
        for err in errors:
            print(f"inventory error: {err}", file=sys.stderr)
        print(
            f"{len(errors)} tracked file(s) could not be analyzed — failing the check "
            "rather than scanning a partial inventory",
            file=sys.stderr,
        )
        return 2

    # Arm 2. The marked columns/tables come from the marked DDL, so the set is
    # never hard-coded; a tree with no markers enforces nothing (and says so).
    frozen_columns = frozenset(marked_columns)
    frozen_tables = frozenset(marked_tables)
    bind_per_file: dict[str, list[str]] = {}
    for relpath, lineno, text in bind_candidates:
        sites = _scan_query_side(relpath, [(lineno, text)], frozen_columns, frozen_tables)
        if sites:
            bind_per_file.setdefault(relpath, []).extend(sites)

    counts = {path: len(sites) for path, sites in ddl_per_file.items()}
    if args.print_baseline:
        _print_baseline(counts)
        return 0

    baseline = {} if args.all else _BASELINE
    offenders: list[str] = []
    advisories: list[str] = []
    for path, sites in sorted(ddl_per_file.items()):
        allowed = baseline.get(path, 0)
        if len(sites) > allowed:
            offenders.extend(sites[allowed:])
    for path, allowed in baseline.items():
        if counts.get(path, 0) < allowed:
            advisories.append(
                f"{path}: {counts.get(path, 0)} dangerous site(s) left, _BASELINE says {allowed}"
            )

    if advisories:
        for note in advisories:
            print(f"advisory — {note}", file=sys.stderr)
        print(
            "advisory — _BASELINE in scripts/ci/check_partial_index_where.py is stale; "
            "regenerate it with --print-baseline",
            file=sys.stderr,
        )

    if not frozen_columns:
        print(
            f"advisory — no `{MARKER}` marker found in tracked production *.py, so the "
            "query-side arm enforced nothing; if the kept-partial sites were retired "
            "deliberately, drop this line's expectation",
            file=sys.stderr,
        )

    bind_offenders = [site for sites in bind_per_file.values() for site in sites]
    if offenders:
        for site in offenders:
            print(site)
        print(
            f"{len(offenders)} partial index(es) compare a column to a literal. Replace the "
            "partial form with the composite index on those columns (exact, same plan), or "
            f"mark the line `{MARKER} — <why>` when the column is never bound with `=`.",
            file=sys.stderr,
        )
    if bind_offenders:
        for site in bind_offenders:
            print(site)
        print(
            f"{len(bind_offenders)} production statement(s) bind a kept-partial column with `=`. "
            f"Either stop binding {'/'.join(sorted(frozen_columns))} with `=` (a literal or a "
            "non-equality predicate keeps the marker true), or composite the marked index whose "
            "marker that statement falsifies.",
            file=sys.stderr,
        )
    if offenders or bind_offenders:
        return 1
    if advisories and args.strict_baseline:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
