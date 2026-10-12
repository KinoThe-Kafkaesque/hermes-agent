"""scripts/check runner, git hooks, CLI output modes and regex-rule scope, on real git repos.

Fixture repos carry a copy of this checkout's engine (scripts/check, scripts/code_health, the
guard scripts, pyproject's ruff pin) so the runner and the hooks execute exactly as they would
in a clone, against tiny trees that keep every check fast.
"""

from __future__ import annotations

import ast
import importlib.machinery
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from scripts.code_health import cli
from scripts.code_health.config import ENFORCEMENT

REPO = Path(__file__).resolve().parents[2]
# One entry per executed guard in scripts/check (`_GUARDS` plus `_DIFF_GUARDS`, which judge a
# diff rather than a tree): these are the scripts the runner executes.
# `test_guard_scripts_parity_with_scripts_check` parses scripts/check and enforces that claim, so
# a newly registered guard reddens here by name instead of surfacing as an opaque `Errno 2`.
_GUARD_SCRIPTS = (
    "scripts/check-windows-footguns.py", "scripts/check_bash_shebangs.py",
    "scripts/check_no_tmp_literals.py", "scripts/check_config_yaml_writers.py",
    "scripts/ci/check_os_marker_fakes.py", "scripts/check-case-collisions.py",
    "scripts/ci/check_lazy_deps_imports.py", "scripts/ci/check_profile_archive_boundary.py",
    "scripts/ci/check_agents_md_size.py", "scripts/ci/check_blob_size.py",
    "scripts/ci/check_partial_index_where.py",
)
# Read by the guard scripts rather than executed as one: a policy file, a module the
# agents-md guard measures, and the topical sibling the partial-index facade imports (its arm 2
# and the WHERE/DDL matchers both arms read), which would otherwise fail to import in the fixture.
# `test_support_tier_covers_every_staged_dependency` DERIVES this tier from what the staged
# scripts actually read, and `test_fixture_tree_guard_names_a_missing_support_file` proves the
# failure it prevents is a named one — so a new support file cannot be forgotten here.
_SUPPORT = (
    "scripts/ci/profile_scope_patterns.json", "agent/subdirectory_hints.py",
    "scripts/ci/check_partial_index_where_markers.py", *_GUARD_SCRIPTS,
)
_LEGACY = "def legacy(x):\n" + "".join(f"    if x == {i}:\n        return {i}\n" for i in range(21))
_GROWN = _LEGACY + "    if x == 99:\n        return 99\n"
_ENV_COPY = "import os\n\n\ndef child_env():\n    env = os.environ.copy()\n    return env\n"
# A violation of a rule that always blocks (PS-P05 is warning-only until replayed).
_SWALLOW = "def swallow():\n    try:\n        pass\n    except Exception:\n        pass\n"
_SWITCH = "scripts/code_health/config.py"


def _executed_guard_scripts() -> frozenset[str]:
    """The script each ``scripts/check`` guard entry runs, read off the module itself.

    Imported, not text-scanned: the claim under test is about the object the runner iterates,
    so deriving from the module makes a renamed or reshaped guard table a named failure here
    instead of a silently empty comparison. ``argv[0]`` is the script (``footguns`` appends
    ``--all``). BOTH executed tables count: ``_GUARDS`` judge a tree and ``_DIFF_GUARDS`` judge
    a diff against the ratchet's base/head, but either way the fixture must carry the script —
    which is the property ``_GUARD_SCRIPTS`` pins. ``_CHECKS`` also carries the two non-guard
    checks, so the two guard tables are the keys.
    """
    loader = importlib.machinery.SourceFileLoader(
        "_hermes_scripts_check", str(REPO / "scripts/check"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    # Registered for the duration: the module's dataclass resolves its annotations through
    # `sys.modules[cls.__module__]`, which is None for an unregistered module.
    sys.modules[loader.name] = module
    try:
        loader.exec_module(module)
    finally:
        sys.modules.pop(loader.name, None)
    return frozenset(
        argv[0]
        for table in (module._GUARDS, module._DIFF_GUARDS)
        for argv in table.values()
    )


def _engine_scripts() -> tuple[str, ...]:
    """The runner and every module of its package: the fixture stages the engine wholesale."""
    package = REPO / "scripts" / "code_health"
    return ("scripts/check", *sorted(
        str(p.relative_to(REPO)) for p in package.rglob("*.py") if "__pycache__" not in p.parts))


# The interpreter's own wording for "something the fixture did not stage". A guard that dies
# this way is the FIXTURE being incomplete, not the guard finding a problem, so the two are told
# apart by the error text rather than by the exit code (a guard exits non-zero for findings too).
_MISSING_DEPENDENCY = (
    re.compile(r"FileNotFoundError: \[Errno 2\] No such file or directory: '(?P<path>[^']+)'"),
    re.compile(r"ModuleNotFoundError: No module named '(?P<name>[^']+)'"),
    re.compile(r"ImportError: cannot import name '(?P<name>[^']+)' from '(?P<module>[^']+)'"),
)


def _missing_dependency(proc: subprocess.CompletedProcess) -> str | None:
    """The path or module a staged script could not reach, read off its own error output."""
    output = proc.stdout + proc.stderr
    for pattern in _MISSING_DEPENDENCY:
        match = pattern.search(output)
        if match:
            return match.groupdict().get("path") or match.groupdict().get("name")
    return None


def test_guard_scripts_parity_with_scripts_check():
    """``_GUARD_SCRIPTS`` is exactly the set of scripts ``scripts/check``'s ``_GUARDS`` executes.

    The fixture hand-copies this list under a claim of parity that nothing enforced, so
    ``t_fc247363`` registered a guard and never added it here; the fixture's own runner then
    died with an opaque ``can't open file ... [Errno 2]`` inside an unrelated assertion.
    """
    executed = _executed_guard_scripts()
    missing = sorted(executed - set(_GUARD_SCRIPTS))
    assert not missing, (
        f"{missing} are registered in scripts/check's _GUARDS but absent from _GUARD_SCRIPTS: "
        "add them (a guard the fixture does not carry fails as `can't open file ... [Errno 2]`)"
    )
    stale = sorted(set(_GUARD_SCRIPTS) - executed)
    assert not stale, (
        f"_GUARD_SCRIPTS lists {stale}, which scripts/check's _GUARDS does not execute: "
        "register them there or drop them here"
    )


def test_support_tier_is_not_demanded_to_be_an_executed_guard():
    """``_SUPPORT``'s other entries are read by guards, never run as one.

    The parity arm above keys on ``_GUARD_SCRIPTS`` alone, so it can never demand that the
    policy file, the measured module, or the partial-index facade's topical sibling be a
    ``_GUARDS`` entry. This pins the converse: the support tier is real (each entry exists) and
    disjoint from the executed set, so a support file that BECOMES an executed guard must be
    promoted into ``_GUARD_SCRIPTS`` — where the parity arm can see it — rather than left in the
    tier that arm cannot.
    """
    support_only = tuple(rel for rel in _SUPPORT if rel not in _GUARD_SCRIPTS)
    assert support_only, "the support tier is empty: _SUPPORT is just _GUARD_SCRIPTS again"
    for rel in support_only:
        assert (REPO / rel).is_file(), f"support entry {rel} does not exist"
    leaked = sorted(set(support_only) & _executed_guard_scripts())
    assert not leaked, (
        f"{leaked} are read as support but executed as guards: move them into _GUARD_SCRIPTS "
        "so the parity arm covers them"
    )


def _imported_siblings(source: str) -> frozenset[str]:
    """Repo modules a staged script imports, resolved from its import statements.

    ``from scripts.ci import markers`` may name either a module or an attribute, so a candidate
    path counts when it exists on disk: an attribute import of a real sibling module is exactly
    the case the fixture has to stage. Stdlib and third-party names resolve to nothing in this
    repo, so they fall out without a hardcoded allowlist.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            base = node.module.replace(".", "/")
            candidates = [f"{base}.py", f"{base}/__init__.py",
                          *(f"{base}/{alias.name}.py" for alias in node.names)]
        elif isinstance(node, ast.Import):
            candidates = [f"{alias.name.replace('.', '/')}.py" for alias in node.names]
        else:
            continue
        found.update(rel for rel in candidates if (REPO / rel).is_file())
    return frozenset(found)


def _staged_sibling_imports() -> frozenset[str]:
    """Sibling modules the staged scripts import that the fixture does not stage itself.

    Derived, never hand-listed: ``t_fc247363`` added the partial-index sibling and the fixture
    kept failing as an opaque ``ImportError`` because no arm derived this set. The engine
    (``scripts/check`` + ``scripts/code_health/*``) is staged wholesale by ``_add_engine``, so
    its own internal imports are already satisfied and are excluded here.
    """
    staged = {*_GUARD_SCRIPTS, *_engine_scripts()}
    imported: set[str] = set()
    for rel in sorted(staged):
        imported |= _imported_siblings((REPO / rel).read_text(encoding="utf-8-sig"))
    return frozenset(imported - staged)


def test_support_tier_covers_every_staged_sibling_import():
    """Every sibling module a staged script imports is in ``_SUPPORT`` — derived, not hand-listed.

    The support tier's hand-list is the second half of the drift this lane has now paid for three
    times: a new sibling module imported by a guard is absent from the fixture and dies as an
    ``ImportError`` inside whichever hook test runs first, not as a named failure. This derives
    the required set from the staged scripts' own import statements, so adding an import without
    staging it reddens here by name.

    Only the missing direction is asserted here. The reverse (every ``_SUPPORT`` entry is
    required) is NOT an import question — two of the three support-only entries are reached by a
    path literal, not an import, so a set-equality arm would demand ``AGENTS.md``, ``utils.py``
    and ``package.json`` too. That direction is pinned behaviourally by
    ``test_fixture_tree_guard_names_a_missing_support_file``, which proves each entry is
    load-bearing by dropping it and requiring a named failure.
    """
    required = _staged_sibling_imports()
    assert required, "derived no sibling imports at all: the derivation is broken, not the tier"
    missing = sorted(required - set(_SUPPORT))
    assert not missing, (
        f"{missing} are imported by staged scripts but absent from _SUPPORT: add them (a sibling "
        "the fixture does not carry fails as `ImportError: cannot import name ...`)"
    )


def test_fixture_tree_guard_names_a_missing_executed_guard(tmp_path, monkeypatch):
    """The fixture builder fails BY NAME when an executed guard was not copied in.

    A direct pin on the widened failure surface: dropping one guard from the copied set must
    raise an assertion naming the script, not let the fixture's own scripts/check report the
    opaque ``Errno 2`` from deep inside whichever test happens to run first.
    """
    dropped = "scripts/ci/check_partial_index_where.py"
    monkeypatch.setattr(sys.modules[__name__], "_SUPPORT",
                        tuple(rel for rel in _SUPPORT if rel != dropped))
    repo = _init(tmp_path / "repo")
    with pytest.raises(AssertionError, match=f"missing executed guard.*{re.escape(dropped)}"):
        _add_engine(repo)


def test_fixture_tree_guard_names_a_missing_support_file(tmp_path):
    """Each support entry is load-bearing, and losing it reddens ``scripts/check`` BY NAME.

    The support tier's half of the same surface, and the reverse of the import arm above: every
    entry is dropped from the staged set in turn and the fixture's own runner is executed. The
    run must fail, and the failure must name the file it could not reach — an unnamed traceback
    is the opaque ``Errno 2`` shape this lane keeps paying for. The complete tier must run clean
    first, so the arm cannot pass by everything being red.

    The tier is passed to the staging helper explicitly rather than read from ``_SUPPORT`` at
    call time: the dropped-variant loop mutates the module global, so a control that re-read it
    would stage the MUTATED tier and red on the control too (measured — the first version of this
    arm did exactly that under a dropped entry).
    """
    complete_tier = _SUPPORT
    support_only = tuple(rel for rel in complete_tier if rel not in _GUARD_SCRIPTS)
    assert support_only, "the support tier is empty: nothing to drop"

    def staged_repo(name: str, tier: tuple[str, ...]) -> Path:
        """An engine repo whose first commit carries the engine plus ``tier``, verbatim."""
        repo = _init(tmp_path / name)
        _write(repo, {"pkg/a.py": "def a():\n    return 1\n"})
        for rel in tier:
            (repo / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / rel, repo / rel)
        _write(repo, {"tests/README.md": "os-marker-fakes scans tests/\n"})
        shutil.copy2(REPO / "scripts/check", repo / "scripts/check")
        shutil.copytree(REPO / "scripts/code_health", repo / "scripts/code_health",
                        ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
        trees = [tree for tree in ("agent", "scripts", "tests") if (repo / tree).exists()]
        _git(repo, "add", "--all", "--", ".gitignore", "pyproject.toml", "pkg", *trees)
        _git(repo, "commit", "-q", "-m", "base")
        return repo

    def staged_run(repo: Path) -> subprocess.CompletedProcess:
        _write(repo, {"pkg/c.py": "def c():\n    return 3\n"})
        _git(repo, "add", "--", "pkg/c.py")
        return _check(repo, "--staged", "--base", "HEAD")

    complete = staged_run(staged_repo("full", complete_tier))
    absent = _missing_dependency(complete)
    assert complete.returncode == 0 and absent is None, (
        f"the declared support tier does not run clean: {absent!r} is missing "
        f"({complete.stdout + complete.stderr})"
    )

    for rel in support_only:
        proc = staged_run(staged_repo(rel.replace("/", "_").replace(".", "-"),
                                      tuple(r for r in complete_tier if r != rel)))
        assert proc.returncode != 0, f"dropping {rel} left the fixture green: {proc.stdout}"
        missing = _missing_dependency(proc)
        assert missing is not None, (
            f"dropping {rel} failed without naming the file it could not reach (the opaque "
            f"failure shape this arm exists to prevent): {proc.stdout + proc.stderr}"
        )
        assert Path(missing).name == Path(rel).name or Path(rel).stem in missing, (
            f"dropping {rel} named {missing!r} instead"
        )


def _env() -> dict[str, str]:
    """The caller's env minus anything that would point git at another repo or config."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    return env


def _sh(cwd: Path, *argv: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(list(argv), cwd=cwd, env=env or _env(), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
                          timeout=600, check=False)


def _git(repo: Path, *args: str) -> str:
    proc = _sh(repo, "git", *args)
    assert proc.returncode == 0, (args, proc.stdout, proc.stderr)
    return proc.stdout.strip()


def _write(repo: Path, files: Mapping[str, str | None]) -> None:
    for rel, text in files.items():
        path = repo / rel
        if text is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")


def _commit(repo: Path, files: Mapping[str, str | None]) -> str:
    _write(repo, files)
    _git(repo, "add", "--all", "--", *files)
    _git(repo, "commit", "-q", "-m", "step")
    return _git(repo, "rev-parse", "HEAD")


def _init(path: Path) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "t")
    (path / ".gitignore").write_text(".venv\n", encoding="utf-8")
    shutil.copy(REPO / "pyproject.toml", path / "pyproject.toml")
    return path


def _add_engine(repo: Path, checker: bool = True) -> list[str]:
    """What the checker's jobs read (guard scripts, policy, a tests/ dir) and, with ``checker``,
    scripts/check + scripts/code_health themselves; returns the top-level paths to stage."""
    for rel in _SUPPORT:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, repo / rel)
    # Widened failure surface: a guard the copied set forgot is a NAMED error here, not the
    # opaque `can't open file ... [Errno 2]` the fixture's own scripts/check would emit later.
    absent = sorted(rel for rel in _GUARD_SCRIPTS if not (repo / rel).is_file())
    assert not absent, (
        f"the fixture tree is missing executed guard(s) {absent}: they are in scripts/check's "
        "_GUARDS but not in the copied set (_SUPPORT)"
    )
    _write(repo, {"tests/README.md": "os-marker-fakes scans tests/\n"})
    if checker:
        shutil.copy2(REPO / "scripts/check", repo / "scripts/check")
        shutil.copytree(REPO / "scripts/code_health", repo / "scripts/code_health",
                        ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
    return ["agent", "scripts", "tests"]


def _engine_repo(tmp_path: Path) -> Path:
    """A repo whose first commit already carries the checker, with one clean module."""
    repo = _init(tmp_path / "repo")
    _write(repo, {"pkg/a.py": "def a():\n    return 1\n"})
    staged = _add_engine(repo)
    _git(repo, "add", "--all", "--", ".gitignore", "pyproject.toml", "pkg", *staged)
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def _check(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return _sh(repo, sys.executable, str(repo / "scripts/check"), *args)


def _timeless(report: str) -> str:
    """A report minus its elapsed time (`(0.1s)`), the one part two identical runs may differ in."""
    return re.sub(r"\(\d+\.\ds\)", "", report)


def _ratchet_repo(tmp_path: Path, switch: str | None = None) -> tuple[Path, str]:
    repo = _init(tmp_path / "repo")
    (repo / "scripts/ci").mkdir(parents=True)
    shutil.copy2(REPO / "scripts/ci/profile_scope_patterns.json", repo / "scripts/ci/")
    files: dict[str, str | None] = {"pkg/a.py": _LEGACY, **({_SWITCH: switch} if switch else {})}
    _write(repo, files)
    _git(repo, "add", "--all", "--", ".gitignore", "pyproject.toml", "scripts", "pkg")
    _git(repo, "commit", "-q", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


# --- F20: the staged/commit health verdict comes from the judged artifact only -----------------


@pytest.mark.parametrize("unstaged", [
    # the policy file read by the PS-P05 rule, narrowed so it no longer covers pkg/
    ("scripts/ci/profile_scope_patterns.json", lambda t: t.replace(
        '"id": "P05",', '"id": "P05",\n      "path_regex": "^nowhere/",')),
    # the engine itself, edited to exempt pkg/ from PS-P05
    ("scripts/code_health/config.py", lambda t: t.replace(
        'exclude=SINGLE_PROFILE, blocking=WARN_UNTIL_REPLAYED, pattern_id="P05"',
        'exclude=SINGLE_PROFILE + ("pkg/*",), blocking=WARN_UNTIL_REPLAYED, pattern_id="P05"')),
])
def test_staged_health_ignores_unstaged_policy_and_engine(tmp_path, unstaged):
    repo = _engine_repo(tmp_path)
    _write(repo, {"pkg/c.py": _ENV_COPY})
    _git(repo, "add", "--", "pkg/c.py")
    tree = _git(repo, "write-tree")
    first = _check(repo, "--staged", "--only", "health", "--base", "HEAD")
    assert first.returncode == 0 and "PS-P05" in first.stdout, first.stdout + first.stderr

    rel, edit = unstaged
    original = (repo / rel).read_text(encoding="utf-8")
    edited = edit(original)
    assert edited != original
    (repo / rel).write_text(edited, encoding="utf-8")
    again = _check(repo, "--staged", "--only", "health", "--base", "HEAD")
    assert _timeless(again.stdout) == _timeless(first.stdout), again.stdout + again.stderr
    # the judged artifact and the user's unstaged work are both untouched
    assert _git(repo, "write-tree") == tree
    assert (repo / rel).read_text(encoding="utf-8") == edited
    assert _git(repo, "diff", "--name-only") == rel


# --- F22: selectors are validated before anything runs ----------------------------------------


@pytest.mark.parametrize("selector", ["heath", ",", "", "heath,rof", "health,rof"])
def test_unknown_or_empty_selector_is_a_usage_error(selector):
    proc = _sh(REPO, sys.executable, str(REPO / "scripts/check"), "--only", selector)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "health" in proc.stderr and "shebangs" in proc.stderr  # lists the valid names
    assert "checks, ok" not in proc.stdout


def test_valid_selector_runs_its_check_and_keeps_its_status(tmp_path):
    repo = _engine_repo(tmp_path)
    _write(repo, {"pkg/c.py": _SWALLOW})
    _git(repo, "add", "--", "pkg/c.py")
    proc = _check(repo, "--staged", "--only", "health,shebangs", "--base", "HEAD")
    assert proc.returncode == 1 and "2 checks, FAILED: health" in proc.stdout, proc.stdout


# --- JSON output on every return path ----------------------------------------------------------


@pytest.mark.parametrize("case, switch, files, code, findings", [
    ("no measured change", None, {"README.md": "docs\n"}, 0, 0),
    ("enforcement off", 'ENFORCEMENT = "off"\n', {"pkg/a.py": _GROWN}, 0, 0),
    ("findings", None, {"pkg/a.py": _GROWN}, 1, 1),
    ("clean measured diff", None, {"pkg/b.py": "def b():\n    return 2\n"}, 0, 0),
])
def test_json_stdout_is_json_on_every_path(tmp_path, capsys, case, switch, files, code, findings):
    repo, base = _ratchet_repo(tmp_path, switch)
    head = _commit(repo, files)
    assert cli.run(repo, base, head, as_json=True) == code, case
    out, err = capsys.readouterr()
    data = json.loads(out)
    assert isinstance(data, list) and len(data) == findings, (case, out)
    if case in ("no measured change", "enforcement off"):
        assert "code health:" in err  # the human explanation moves to stderr


# --- pre-push: the pushed candidate decides, never the checked-out branch ----------------------


def _pushable(tmp_path: Path) -> tuple[Path, Path]:
    """`main` has the guards but predates scripts/check, and is published: (repo, bare remote)."""
    repo = _init(tmp_path / "repo")
    _write(repo, {"pkg/a.py": "def a():\n    return 1\n"})
    staged = _add_engine(repo, checker=False)
    _git(repo, "add", "--all", "--", ".gitignore", "pyproject.toml", "pkg", *staged)
    _git(repo, "commit", "-q", "-m", "main before scripts/check")
    (repo / ".venv").symlink_to(Path(sys.prefix), target_is_directory=True)
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "-q", "--bare", str(remote))
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "origin", "main")
    _git(repo, "fetch", "-q", "origin")
    return repo, remote


def _branch_with_checker(repo: Path, name: str, extra: dict[str, str]) -> str:
    _git(repo, "checkout", "-q", "-b", name, "main")
    _write(repo, extra)
    _git(repo, "add", "--all", "--", *_add_engine(repo), *extra)
    _git(repo, "commit", "-q", "-m", name)
    return _git(repo, "rev-parse", "HEAD")


def _remote_ref(remote: Path, branch: str) -> str:
    return _git(remote, "for-each-ref", "--format=%(objectname)", f"refs/heads/{branch}")


def test_pre_push_judges_a_checker_branch_pushed_from_an_older_checkout(tmp_path):
    repo, remote = _pushable(tmp_path)
    _branch_with_checker(repo, "feature", {"pkg/c.py": _SWALLOW})
    good = _branch_with_checker(repo, "clean", {"pkg/d.py": "def d():\n    return 4\n"})
    assert _check(repo, "--install-hook", "pre-push").returncode == 0
    _git(repo, "checkout", "-q", "main")
    assert not (repo / "scripts/check").exists()

    push = _sh(repo, "git", "push", "origin", "feature")
    assert push.returncode != 0, push.stdout + push.stderr
    assert "BLE001" in push.stdout + push.stderr
    assert _remote_ref(remote, "feature") == ""  # destination untouched

    # controls: a clean checker branch passes, a branch predating the checker is not judged,
    # a deletion is not judged
    push = _sh(repo, "git", "push", "origin", "clean")
    assert push.returncode == 0 and _remote_ref(remote, "clean") == good, push.stderr
    _git(repo, "checkout", "-q", "-b", "old", "main")
    old = _commit(repo, {"pkg/c.py": _SWALLOW})
    assert _sh(repo, "git", "push", "origin", "old").returncode == 0
    assert _remote_ref(remote, "old") == old
    assert _sh(repo, "git", "push", "origin", "--delete", "clean").returncode == 0
    assert _remote_ref(remote, "clean") == ""


# --- M5: interpreter choice and the version guard ---------------------------------------------


def test_hooks_prefer_the_repo_virtualenv_over_path_python(tmp_path):
    repo = _engine_repo(tmp_path)
    assert _check(repo, "--install-hook", "pre-commit").returncode == 0
    fake = tmp_path / "fakebin"
    fake.mkdir()
    for name in ("python3", "python"):
        (fake / name).write_text("#!/bin/sh\necho 'wrong interpreter' >&2\nexit 97\n",
                                 encoding="utf-8")
        (fake / name).chmod(0o755)
    env = _env()
    env["PATH"] = f"{fake}{os.pathsep}{env['PATH']}"
    (repo / ".venv").symlink_to(Path(sys.prefix), target_is_directory=True)
    _write(repo, {"pkg/b.py": "def b():\n    return 2\n"})
    _git(repo, "add", "--", "pkg/b.py")
    proc = _sh(repo, "git", "commit", "-q", "-m", "venv", env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # control: without a virtualenv the hook falls back to PATH's python3
    (repo / ".venv").unlink()
    _write(repo, {"pkg/b.py": "def b():\n    return 3\n"})
    _git(repo, "add", "--", "pkg/b.py")
    proc = _sh(repo, "git", "commit", "-q", "-m", "path", env=env)
    assert proc.returncode != 0 and "wrong interpreter" in proc.stderr


def test_check_refuses_old_python_before_importing_anything():
    # The guard reads sys.version_info; a 3.10 interpreter is emulated by overriding it before
    # scripts/check runs (tomllib, imported via ruff_runner, would otherwise crash it).
    code = ("import runpy, sys; sys.version_info = (3, 10, 12, 'final', 0); "
            f"sys.argv = ['scripts/check', '--only', 'shebangs']; "
            f"runpy.run_path({str(REPO / 'scripts/check')!r}, run_name='__main__')")
    proc = _sh(REPO, sys.executable, "-c", code)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "3.11" in proc.stderr and "Traceback" not in proc.stderr


# --- m3: the ENFORCEMENT switch ---------------------------------------------------------------


@pytest.mark.parametrize("switch, code", [
    ('ENFORCEMENT: str = "off"\n', 0),
    ('ENFORCEMENT: str = "advisory"\n', 0),
    ('ENFORCEMENT = "warn"\n', 1),  # an unknown value is not a way to relax the check
    ('ENFORCEMENT = "off"  # until the burn-down lands\n', 0),
])
def test_enforcement_switch_parses_annotated_and_unknown_values(tmp_path, capsys, switch, code):
    repo, base = _ratchet_repo(tmp_path, switch)
    head = _commit(repo, {"pkg/a.py": _GROWN})
    assert cli.run(repo, base, head) == code, capsys.readouterr().out


def test_enforcement_reads_this_trees_own_switch():
    assert cli.enforcement(REPO, "HEAD") == ENFORCEMENT
    text = (REPO / _SWITCH).read_text(encoding="utf-8")
    assert cli.parse_switch(text) == ENFORCEMENT


def _branch_behind_main(tmp_path: Path, main_switch: str, branch_files: dict[str, str]) -> Path:
    repo, base = _ratchet_repo(tmp_path, 'ENFORCEMENT = "blocking"\n')
    _git(repo, "checkout", "-q", "-b", "feature")
    _commit(repo, branch_files)
    _git(repo, "checkout", "-q", "main")
    tip = _commit(repo, {_SWITCH: main_switch, "NEWS.md": "main moved on\n"})
    _git(repo, "update-ref", "refs/remotes/origin/main", tip)
    _git(repo, "reset", "-q", "--hard", base)  # local main is stale; only origin/main moved
    _git(repo, "checkout", "-q", "feature")
    return repo


def test_local_runs_take_the_switch_from_the_main_tip(tmp_path, monkeypatch, capsys):
    repo = _branch_behind_main(tmp_path, 'ENFORCEMENT = "advisory"\n', {"pkg/a.py": _GROWN})
    monkeypatch.chdir(repo)
    assert cli.main([]) == 0, capsys.readouterr().out  # the flip on main reaches the branch
    head = _git(repo, "rev-parse", "HEAD")
    assert cli.main(["--head", head]) == 0
    out = capsys.readouterr().out
    assert "CC 23 > 22" in out and "advisory mode" in out


def test_a_branch_cannot_relax_its_own_switch(tmp_path, monkeypatch, capsys):
    grown_and_off = {"pkg/a.py": _GROWN, _SWITCH: 'ENFORCEMENT = "off"\n'}
    repo = _branch_behind_main(tmp_path, 'ENFORCEMENT = "blocking"\n', grown_and_off)
    monkeypatch.chdir(repo)
    assert cli.main([]) == 1, capsys.readouterr().out


# --- F23: the hooks under Git for Windows with MSYS path conversion disabled -----------------


def test_hooks_run_with_msys_path_conversion_disabled(tmp_path):
    """Hermes' Windows terminal exports both variables, so native git and python see the hook's
    paths verbatim: a POSIX `/c/...` temp path then reaches them unconverted. Inert elsewhere."""
    repo, remote = _pushable(tmp_path)
    _branch_with_checker(repo, "feature", {"pkg/d.py": "def d():\n    return 4\n"})
    for kind in ("pre-commit", "pre-push"):
        assert _check(repo, "--install-hook", kind).returncode == 0
    env = {**_env(), "MSYS_NO_PATHCONV": "1", "MSYS2_ARG_CONV_EXCL": "*"}

    _write(repo, {"pkg/e.py": _SWALLOW})
    _git(repo, "add", "--", "pkg/e.py")
    commit = _sh(repo, "git", "commit", "-q", "-m", "swallow", env=env)
    assert commit.returncode != 0 and "BLE001" in commit.stdout + commit.stderr, commit.stdout + commit.stderr
    _write(repo, {"pkg/e.py": "def e():\n    return 5\n"})
    _git(repo, "add", "--", "pkg/e.py")
    commit = _sh(repo, "git", "commit", "-q", "-m", "clean", env=env)
    assert commit.returncode == 0, commit.stdout + commit.stderr

    push = _sh(repo, "git", "push", "-q", "origin", "feature", env=env)
    assert push.returncode == 0, push.stdout + push.stderr
    assert _remote_ref(remote, "feature") == _git(repo, "rev-parse", "HEAD")
    _write(repo, {"pkg/f.py": _SWALLOW})
    _git(repo, "add", "--", "pkg/f.py")
    _git(repo, "commit", "-q", "--no-verify", "-m", "swallow")
    push = _sh(repo, "git", "push", "-q", "origin", "feature", env=env)
    assert push.returncode != 0 and "BLE001" in push.stdout + push.stderr, push.stdout + push.stderr


# --- m4: the CI guards that ran outside the lint workflow --------------------------------------


@pytest.mark.parametrize("job, files", [
    ("case-collisions", {"pkg/Notes.md": "a\n", "pkg/NOTES.md": "b\n"}),
    ("lazy-deps", {"pkg/uses.py": "import tools.lazy_deps\n"}),
    ("profile-archives", {"pkg/export.tar.gz": "not really\n"}),
    ("blob-size", {"pkg/shot.png": "\0" * (600 * 1024)}),
])
def test_ci_only_guards_run_in_scripts_check(tmp_path, job, files):
    repo = _engine_repo(tmp_path)
    clean = _check(repo, "--staged", "--only", job, "--base", "HEAD")
    assert clean.returncode == 0 and "1 checks, ok" in clean.stdout, clean.stdout + clean.stderr
    # Staged as blobs, never written to disk: a case-insensitive filesystem holds one of
    # Notes.md / NOTES.md, but the index (what --staged judges) holds both.
    for rel, text in files.items():
        blob = subprocess.run(["git", "hash-object", "-w", "--stdin"], cwd=repo, env=_env(),
                              input=text.encode("utf-8"), capture_output=True, timeout=60, check=True)
        _git(repo, "update-index", "--add", "--cacheinfo", f"100644,{blob.stdout.decode().strip()},{rel}")
    proc = _check(repo, "--staged", "--only", job, "--base", "HEAD")
    assert proc.returncode == 1 and f"FAILED: {job}" in proc.stdout, proc.stdout + proc.stderr


# --- F21: profile regex rules see executable code, not prose -----------------------------------


_FUNC = "import os\n\n\ndef build(cmd):\n{}    return cmd\n"


@pytest.mark.parametrize("body, flagged", [
    ("    # Avoid env = os.environ.copy(); use the scoped builder.\n", False),
    ('    """Avoid env = os.environ.copy(); use the scoped builder."""\n', False),
    ('    """Builder.\n\n    Never env = os.environ.copy() here.\n    """\n', False),
    ("    # don't read os.getenv(\"DISCORD_TOKEN\") directly\n", False),
    ("    cmd = [cmd]  # not os.environ.get('SLACK_TOKEN')\n", False),
    ("    'inert example: env = os.environ.copy()'\n", False),
    # positive controls: the real operations, including a literal env-key argument
    ("    env = os.environ.copy()\n    cmd = (cmd, env)\n", True),
    ("    token = os.getenv(\"DISCORD_TOKEN\")\n    cmd = (cmd, token)\n", True),
    ("    env = dict(os.environ)  # copy it\n    cmd = (cmd, env)\n", True),
])
def test_profile_regex_rules_ignore_comments_and_docstrings(tmp_path, capsys, body, flagged):
    repo, base = _ratchet_repo(tmp_path)
    head = _commit(repo, {"pkg/c.py": _FUNC.format(body)})
    code = cli.run(repo, base, head)
    out = capsys.readouterr().out
    assert code == 0, out  # PS-P05/P06 are warnings until replayed
    assert ("PS-P05" in out or "PS-P06" in out) == flagged, out


# --- allow comments may list several rules ----------------------------------------------------


@pytest.mark.parametrize("rules", ["BLE001 S110", "BLE001, S110", "BLE001,S110"])
def test_allow_comment_lists_rules_by_comma_or_space(tmp_path, capsys, rules):
    repo, base = _ratchet_repo(tmp_path)
    swallow = ("def other():\n    try:\n        pass\n"
               f"    except Exception:  # health: allow {rules} -- boundary\n        pass\n")
    head = _commit(repo, {"pkg/b.py": swallow})
    assert cli.run(repo, base, head) == 0, capsys.readouterr().out


# --- measurement tooling failures are errors, not findings ------------------------------------


def test_tooling_failure_exits_2_without_a_traceback(tmp_path, monkeypatch, capsys):
    repo, base = _ratchet_repo(tmp_path)
    lock = {"packages": {"node_modules/typescript": {"version": "0.0.0-not-published"}}}
    _commit(repo, {"package-lock.json": json.dumps(lock), "web/a.ts": "export const a = 1;\n"})
    if not shutil.which("npm") or not shutil.which("node"):
        pytest.skip("needs node + npm to reach the npm install path")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("npm_config_offline", "true")  # a real npm failure, without the network
    monkeypatch.chdir(repo)
    assert cli.main(["--base", base, "--head", "HEAD"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("code health:") and "npm" in err
