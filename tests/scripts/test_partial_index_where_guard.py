"""Contract tests for scripts/ci/check_partial_index_where.py.

The guard's whole value is that it can FAIL on a dangerous-shaped partial index
and PASS on the safe spellings, so each arm drives the real checker CLI against a
fresh tracked git fixture repo (the same shape ``tests/pm/test_no_lazy_deps.py``
uses: ``git add`` is enough, because the inventory is ``git ls-files``).

Both directions are pinned:

* a partial index whose WHERE equates a column to a scalar literal is RED — the
  form measured to make a reused parameterized statement return wrong rows;
* every safe spelling of the same intent (composite, ``IS NOT NULL``, ``!=``,
  ``>=``, ``LIKE``, a function call, ``IN`` with two-or-more literals) is GREEN.

The checker must also never pass by scanning nothing: a repo it cannot inventory
(not a git repo, no tracked production ``*.py``) and a tracked file that will not
parse both FAIL (exit 2) rather than reporting a clean tree.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "check_partial_index_where.py"

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    """A fresh minimal tracked git repo with one production Python file."""
    env = {**os.environ, **_GIT_ENV}

    def _git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, env=env, check=True, capture_output=True)

    _git("init", "-q")
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("def test_x():\n    pass\n", encoding="utf-8")
    _git("add", "-A")
    return tmp_path


def _run_checker(root: Path, *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(root), *flags],
        capture_output=True,
        text=True,
        timeout=120,
    )


def _track(root: Path, relpath: str, body: str) -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    subprocess.run(["git", "add", relpath], cwd=root, check=True, capture_output=True)


def _ddl(statement: str) -> str:
    return f'SCHEMA = "{statement}"\n'


# --- RED: the dangerous shape -------------------------------------------------

@pytest.mark.parametrize("statement", [
    # text literal on a non-indexed column (the fleet's measured task_runs shape)
    "CREATE INDEX ix ON task_runs(ended_at) WHERE outcome='rate_limited'",
    # text literal on the INDEXED column
    "CREATE INDEX ix ON task_runs(outcome) WHERE outcome = 'rate_limited'",
    # integer literal
    "CREATE INDEX ix ON messages(session_id) WHERE active = 1",
    # a single-literal IN reproduces it too
    "CREATE INDEX ix ON messages(session_id) WHERE role IN ('assistant')",
    # any literal inside a compound WHERE reproduces it
    "CREATE INDEX ix ON messages(session_id) WHERE display_identity IS NOT NULL AND active = 1",
    # inside parentheses / OR
    "CREATE INDEX ix ON messages(session_id, display_order) WHERE (active = 1 OR compacted = 1)",
])
def test_literal_equality_partial_index_is_refused(fixture_repo, statement):
    _track(fixture_repo, "app.py", _ddl(statement))
    result = _run_checker(fixture_repo)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "app.py:" in result.stdout


def test_gaining_a_hit_over_the_baseline_is_refused(fixture_repo):
    """A file already on the burn-down list may not gain another dangerous site."""
    _track(fixture_repo, "app.py", _ddl(
        "CREATE INDEX ix ON messages(session_id) WHERE active = 1"
    ))
    # The guard ships no _BASELINE entry for app.py, so one site is already a
    # failure; two must be too (the count only ever ratchets up).
    _track(fixture_repo, "app.py", _ddl(
        "CREATE INDEX ix ON messages(session_id) WHERE active = 1"
    ) + _ddl("CREATE INDEX iy ON messages(session_id) WHERE compacted = 1"))
    result = _run_checker(fixture_repo)
    assert result.returncode == 1
    assert result.stdout.count("app.py:") == 2, result.stdout


# --- GREEN: the safe spellings ------------------------------------------------

@pytest.mark.parametrize("statement", [
    # composite index — the fleet's shipped replacement, exact and same plan
    "CREATE INDEX ix ON task_runs(outcome, ended_at)",
    # IS NOT NULL, not a scalar equality
    "CREATE UNIQUE INDEX ix ON sessions(title) WHERE title IS NOT NULL",
    "CREATE INDEX ix ON messages(session_id, platform_message_id) "
    "WHERE platform_message_id IS NOT NULL",
    # inequality / ordering / pattern comparisons measured safe
    "CREATE INDEX ix ON task_runs(ended_at) WHERE outcome != 'rate_limited'",
    "CREATE INDEX ix ON task_runs(ended_at) WHERE outcome <> 'rate_limited'",
    "CREATE INDEX ix ON task_runs(ended_at) WHERE ended_at >= 0",
    "CREATE INDEX ix ON task_runs(ended_at) WHERE outcome LIKE 'rate%'",
    "CREATE INDEX ix ON task_runs(ended_at) WHERE outcome GLOB 'rate*'",
    # IN with two or more literals measured safe
    "CREATE INDEX ix ON messages(session_id) WHERE role IN ('assistant', 'tool')",
    # a function call is not a scalar literal
    "CREATE INDEX ix ON messages(session_id) WHERE length(role) > 0",
    # no WHERE at all
    "CREATE INDEX ix ON messages(session_id)",
])
def test_safe_partial_index_forms_are_accepted(fixture_repo, statement):
    _track(fixture_repo, "app.py", _ddl(statement))
    result = _run_checker(fixture_repo)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == ""


def test_opt_out_marker_silences_one_site(fixture_repo):
    _track(fixture_repo, "app.py", (
        "# partial-index: ok — column is never bound with `=`\n"
        'SCHEMA = "CREATE INDEX ix ON messages(session_id) WHERE active = 1"\n'
    ))
    result = _run_checker(fixture_repo)
    assert result.returncode == 0, result.stdout + result.stderr


def test_opt_out_marker_does_not_silence_an_unmarked_sibling(fixture_repo):
    """The marker is per-line, so a second unmarked site still fails."""
    _track(fixture_repo, "app.py", (
        'SCHEMA = "CREATE INDEX ix ON messages(session_id) WHERE active = 1"\n'
        'OTHER = "CREATE INDEX iy ON messages(session_id) WHERE compacted = 1"\n'
    ))
    result = _run_checker(fixture_repo)
    assert result.returncode == 1
    assert result.stdout.count("app.py:") == 2, result.stdout


def test_prose_mentioning_the_shape_is_not_ddl(fixture_repo):
    """A comment/docstring quoting the dangerous form is documentation, not a schema."""
    _track(fixture_repo, "app.py", (
        '"""Never write CREATE INDEX ix ON t(b) WHERE a=\'x\' — it mis-serves."""\n'
        "# CREATE INDEX ix ON t(b) WHERE a='x'\n"
        "x = 1\n"
    ))
    result = _run_checker(fixture_repo)
    assert result.returncode == 0, result.stdout + result.stderr


def test_multiple_indexes_in_one_script_are_each_judged(fixture_repo):
    _track(fixture_repo, "app.py", _ddl(
        "CREATE INDEX ix ON messages(session_id) WHERE active = 1; "
        "CREATE INDEX iy ON messages(session_id, platform_message_id) "
        "WHERE platform_message_id IS NOT NULL"
    ))
    result = _run_checker(fixture_repo)
    assert result.returncode == 1
    assert result.stdout.count("app.py:") == 1, result.stdout
    assert "ix ON messages" in result.stdout


# --- tests/ fixtures are out of scope ----------------------------------------

def test_tests_tree_is_not_scanned(fixture_repo):
    """A hostile fixture inside tests/ is the test's own business."""
    _track(fixture_repo, "tests/test_x.py",
           _ddl("CREATE INDEX ix ON messages(session_id) WHERE active = 1"))
    result = _run_checker(fixture_repo)
    assert result.returncode == 0, result.stdout + result.stderr


def test_untracked_files_are_not_scanned(fixture_repo):
    (fixture_repo / "debris.py").write_text(
        _ddl("CREATE INDEX ix ON messages(session_id) WHERE active = 1"), encoding="utf-8"
    )
    result = _run_checker(fixture_repo)
    assert result.returncode == 0, result.stdout + result.stderr


# --- never green by scanning nothing ------------------------------------------

def test_not_a_git_repo_fails_the_check(tmp_path):
    result = _run_checker(tmp_path)
    assert result.returncode == 2
    assert "inventory" in (result.stdout + result.stderr).lower()


def test_repo_without_tracked_python_fails_the_check(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "README.md").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True, capture_output=True)
    result = _run_checker(tmp_path)
    assert result.returncode == 2
    assert "inventory" in (result.stdout + result.stderr).lower()


def test_unparseable_tracked_file_fails_the_check(fixture_repo):
    _track(fixture_repo, "app.py", "def f(:\n")
    result = _run_checker(fixture_repo)
    assert result.returncode == 2
    assert "parse" in (result.stdout + result.stderr).lower()


def test_print_baseline_reports_current_counts(fixture_repo):
    _track(fixture_repo, "app.py", _ddl(
        "CREATE INDEX ix ON messages(session_id) WHERE active = 1"
    ))
    result = _run_checker(fixture_repo, "--print-baseline")
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"app.py": 1' in result.stdout


def test_shipped_tree_is_clean_under_its_own_baseline():
    """The live repo must pass its own guard (the burn-down list is exact)."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(REPO_ROOT)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
