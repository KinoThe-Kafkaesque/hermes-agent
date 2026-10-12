#!/usr/bin/env python3
"""Fail when a partial index's ``WHERE`` compares a column to a scalar literal,
and when a ``partial-index: ok`` opt-out stops being true.

Two arms, one guard. Both are needed; either alone is a green light for a
different hole. Arm 1 lives here; arm 2 and the WHERE/DDL matcher primitives both
arms read live in the topical sibling
``scripts/ci/check_partial_index_where_markers.py``, which this facade imports —
so the dependency runs one way and the two can never form a cycle. (The split is
what brought this file's ``main`` back under the repo's ``CC`` cap: it read 16
before arm 2 landed and 24 after, against a target of 20.)

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
FALSE. Arm 2 (in the sibling) derives the column set FROM the marked DDL and refuses
any tracked production string constant that binds one of those columns with a
placeholder in a PREDICATE position on the marked table. Measured: the shipped tree
binds neither ``active`` nor ``compacted`` with a placeholder in any position, so it
is GREEN; a synthetic ``WHERE ... active = ?`` on ``messages`` reds it and names the
site.

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
import subprocess
import sys
from pathlib import Path

# The sibling is imported as ``scripts.ci.…``; ``scripts/ci`` carries no ``__init__.py``, so the
# repo root must be on sys.path — it is not sys.path[0] when this facade is the entry point.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ci import check_partial_index_where_markers as markers  # noqa: E402

MARKER = markers.MARKER
# The DDL/WHERE matchers arm 1 reads live in the sibling with arm 2 (one-way import).
_dangerous_columns = markers.dangerous_columns
_is_marked = markers.is_marked
_where_clauses = markers.where_clauses

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


def _tracked_python_files(root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", "*.py"],
        capture_output=True,
        timeout=300,
        check=False,
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
    columns, tables = markers.marked_partial_columns(ddl_constants, {relpath: lines}, relpath)
    return ddl_sites, constants, columns, tables, []


def _scan_tree(root: Path) -> tuple[dict[str, list[str]], list[tuple[str, int, str]],
                                    frozenset[str], frozenset[str], list[str]]:
    """Walk the tracked tree: ``(ddl_per_file, bind_candidates, columns, tables, errors)``.

    The marked columns/tables are unioned across every file, because arm 2's claim is
    enforced TREE-WIDE: the marker lives in the schema module while the statement that
    would falsify it can be in any file. ``bind_candidates`` keeps only the constants
    that bind some column with ``=`` — the cheap pre-filter for the (rarer) marked ones.
    """
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
        bind_candidates.extend(
            (relpath, lineno, text) for lineno, text in markers.bind_candidates(constants)
        )
    return ddl_per_file, bind_candidates, frozenset(marked_columns), frozenset(marked_tables), errors


def _bind_per_file(bind_candidates: list[tuple[str, int, str]], columns: frozenset[str],
                   tables: frozenset[str]) -> dict[str, list[str]]:
    """Arm 2's findings, grouped by file, for the candidates :func:`_scan_tree` kept."""
    bind_per_file: dict[str, list[str]] = {}
    for relpath, lineno, text in bind_candidates:
        sites = markers.scan_query_side(relpath, [(lineno, text)], columns, tables)
        if sites:
            bind_per_file.setdefault(relpath, []).extend(sites)
    return bind_per_file


def _baseline_findings(ddl_per_file: dict[str, list[str]], counts: dict[str, int],
                       baseline: dict[str, int]) -> tuple[list[str], list[str]]:
    """``(offenders, advisories)`` for arm 1 against the burn-down *baseline*."""
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
    return offenders, advisories


def _print_baseline(counts: dict[str, int]) -> None:
    body = "".join(f'    "{path}": {count},\n' for path, count in sorted(counts.items()))
    print("_BASELINE: dict[str, int] = {\n" + body + "}")


def _emit(offenders: list[str], bind_offenders: list[str], advisories: list[str],
          frozen_columns: frozenset[str]) -> None:
    """Print both arms' findings and the repo-specific remedy for each."""
    for note in advisories:
        print(f"advisory — {note}", file=sys.stderr)
    if advisories:
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

    ddl_per_file, bind_candidates, frozen_columns, frozen_tables, errors = _scan_tree(root)
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
    counts = {path: len(sites) for path, sites in ddl_per_file.items()}
    if args.print_baseline:
        _print_baseline(counts)
        return 0

    offenders, advisories = _baseline_findings(
        ddl_per_file, counts, {} if args.all else _BASELINE
    )
    bind_offenders = [
        site for sites in _bind_per_file(bind_candidates, frozen_columns, frozen_tables).values()
        for site in sites
    ]
    _emit(offenders, bind_offenders, advisories, frozen_columns)
    if offenders or bind_offenders:
        return 1
    if advisories and args.strict_baseline:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
