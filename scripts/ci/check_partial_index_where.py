#!/usr/bin/env python3
"""Fail when a partial index's ``WHERE`` compares a column to a scalar literal.

A ``CREATE INDEX ... WHERE <col> = <literal>`` (or ``<col> IN (<one literal>)``)
is not just a planner hint: on this fleet's SQLite builds it makes a **reused
parameterized statement** return WRONG rows. Measured 2026-10-09/10 on SQLite
3.53.1 and 3.37.2 (Python 3.10/3.11/3.12/3.14), silent — wrong rows, never an
error:

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

What this check refuses is the SHAPE, not a measured instance. The live tree no
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

Measured detail that matters for the allow-list: the hazard is NOT restricted to
the indexed column (``ON t(b) WHERE a='x'`` mis-serves), and the safe spellings
of the same predicate are ``IS NOT NULL`` / ``!=`` / ``>=`` / ``LIKE`` / ``GLOB``
/ a function call / ``IN`` with two-or-more literals. Only ``=`` and a
single-literal ``IN`` reproduce it.

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
    index separately. ``offset`` is the character index of the statement inside
    *sql*, so the caller can report the statement's own line rather than the
    enclosing string literal's.
    """
    cursor = 0
    for stmt in sql.split(";"):
        flat = " ".join(stmt.split())
        match = _CREATE_INDEX.search(flat)
        if match:
            yield match.group(1), match.group(2), match.group(4).strip(), cursor
        cursor += len(stmt) + 1


def _dangerous_columns(where: str) -> list[tuple[str, str, str]]:
    """``(column, literal, form)`` for each literal comparison that reproduces it."""
    found = [(m.group(1), m.group(2), "=") for m in _EQUALS_LITERAL.finditer(where)]
    found += [(m.group(1), m.group(2), "IN") for m in _IN_ONE_LITERAL.finditer(where)]
    return found


def _python_string_constants(path: Path):
    """``(lineno, text)`` for string constants that may hold DDL, or an error string.

    AST-based, so prose in comments and docstrings is not DDL by construction
    (a docstring that quotes the dangerous shape is documentation, not a schema).
    Returns ``(constants, error)`` where exactly one is populated; a file that
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
    constants = [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and "CREATE INDEX" in node.value.upper() and id(node) not in docstrings
    ]
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


def _scan_file(root: Path, relpath: str) -> tuple[list[str], list[str]]:
    """``(sites, errors)`` — ``path:line: ...`` for each dangerous site."""
    path = root / relpath
    sites: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError as exc:
        return [], [f"{relpath}: read failed: {exc}"]
    constants, error = _python_string_constants(path)
    if error is not None:
        return [], [f"{relpath}: {error}"]
    for lineno, text in constants:
        for name, table, where, offset in _where_clauses(text):
            stmt_lineno = lineno + text[:offset].count("\n")
            for column, literal, form in _dangerous_columns(where):
                if _is_marked(lines, stmt_lineno):
                    continue
                sites.append(
                    f"{relpath}:{stmt_lineno}: partial index {name} ON {table} — WHERE "
                    f"compares {column} to the literal {literal} ({form}); a reused "
                    "parameterized statement bound to a different value returns WRONG "
                    "rows on this fleet's SQLite (see the guard's docstring)"
                )
    return sites, []


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

    per_file: dict[str, list[str]] = {}
    errors: list[str] = []
    for relpath in _tracked_python_files(root):
        sites, file_errors = _scan_file(root, relpath)
        errors.extend(file_errors)
        if sites:
            per_file[relpath] = sites

    if errors:
        for err in errors:
            print(f"inventory error: {err}", file=sys.stderr)
        print(
            f"{len(errors)} tracked file(s) could not be analyzed — failing the check "
            "rather than scanning a partial inventory",
            file=sys.stderr,
        )
        return 2

    counts = {path: len(sites) for path, sites in per_file.items()}
    if args.print_baseline:
        _print_baseline(counts)
        return 0

    baseline = {} if args.all else _BASELINE
    offenders: list[str] = []
    advisories: list[str] = []
    for path, sites in sorted(per_file.items()):
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

    if offenders:
        for site in offenders:
            print(site)
        print(
            f"{len(offenders)} partial index(es) compare a column to a literal. Replace the "
            "partial form with the composite index on those columns (exact, same plan), or "
            f"mark the line `{MARKER} — <why>` when the column is never bound with `=`.",
            file=sys.stderr,
        )
        return 1
    if advisories and args.strict_baseline:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
