"""PM's own runtime must never import ``hermes_cli.config``.

``import hermes_cli.config`` runs bundled model-provider plugin discovery at import
time, which imports every ``plugins/model-providers/<name>/``. PM's runtime venv
declares four dependencies (``pm/pyproject.toml``); a plugin reaching anything else
— ``solstice`` → ``agent.gemini_native_adapter`` → ``httpx`` — dies at import and
the runtime prints, on every generation::

    Failed to load bundled provider plugin solstice: No module named 'httpx'

The failure mode is "a minimal runtime imports an app module with a heavy import
side effect", so these tests drive the REAL interpreter with the REAL import
machinery: a child ``python -I -B -c`` against the real checkout, with the runtime's
dependency CLOSURE enforced by an import blocker (every third-party top-level name
outside ``pm/pyproject.toml``'s set is refused). Nothing here reads source text —
the child imports production modules and the assertions are on its behaviour and
stderr.
"""

from __future__ import annotations

import importlib.metadata
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PM_PYPROJECT = REPO / "pm" / "pyproject.toml"

# The live instance of the defect: ``solstice`` → ``agent.gemini_native_adapter`` →
# ``httpx``, which PM's runtime does not declare.
_LIVE_COUPLING_MODULE = "httpx"


def _declared_import_names() -> set[str]:
    """Import names PM's runtime declares, derived from its pyproject.

    Derived, not hardcoded: adding a dependency to ``pm/pyproject.toml`` widens the
    allowed set automatically, so the pin never becomes a stale snapshot.
    ``packages_distributions()`` is inverted because a distribution name is not an
    import name (``ruamel.yaml`` imports as ``ruamel``).
    """
    data = tomllib.loads(PM_PYPROJECT.read_text(encoding="utf-8"))
    import_to_dists: dict[str, list[str]] = {}
    for import_name, dists in importlib.metadata.packages_distributions().items():
        for dist in dists:
            import_to_dists.setdefault(dist, []).append(import_name)
    names: set[str] = set()
    for spec in data["project"]["dependencies"]:
        dist = re.split(r"[<>=!\[; ]", spec.strip(), maxsplit=1)[0].strip()
        names.update(import_to_dists.get(dist, [dist.replace("-", "_")]))
    return names


_DECLARED = _declared_import_names()


def _child_script(body: str, *, enforce_closure: bool) -> str:
    """Wrap ``body`` so the child resolves imports like a PM runtime generation.

    ``enforce_closure`` installs a meta-path finder that refuses every third-party
    top-level module outside PM's declared set — the runtime's actual dependency
    closure. Without it the child sees the whole checkout environment, which is
    exactly the blindness that let the coupling ship.
    """
    prelude = [
        "import sys",
        f"sys.path.insert(0, {str(REPO)!r})",
    ]
    if enforce_closure:
        prelude += [
            "import importlib.machinery, pathlib",
            f"_DECLARED = {sorted(_DECLARED)!r}",
            f"_REPO = pathlib.Path({str(REPO)!r})",
            "_STDLIB = set(sys.stdlib_module_names)",
            "_REPO_TOPS = set()",
            "for _p in _REPO.iterdir():",
            "    if _p.name.startswith('.'):",
            "        continue",
            "    if _p.is_dir() and (_p / '__init__.py').is_file():",
            "        _REPO_TOPS.add(_p.name)",
            "    elif _p.suffix == '.py':",
            "        _REPO_TOPS.add(_p.stem)",
            "",
            "class _RuntimeClosure:",
            "    def find_spec(self, fullname, path=None, target=None):",
            "        _top = fullname.split('.')[0]",
            "        if _top in _STDLIB or _top in _REPO_TOPS or _top in _DECLARED:",
            "            return None",
            "        if importlib.machinery.PathFinder.find_spec(fullname, path) is None:",
            "            return None",
            "        raise ModuleNotFoundError(",
            "            f\"No module named {fullname!r}\", name=fullname)",
            "",
            "sys.meta_path.insert(0, _RuntimeClosure())",
        ]
    return "\n".join(prelude) + "\n" + body


def _run(script: str, *, home: Path, enforce_closure: bool = True,
         extra_env: dict | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess:
    # PYTHONPATH carries the checkout: ``-I`` drops the CWD entry, and a PM runtime
    # generation reaches ``pm`` the same way (its own interpreter, the tree on the path).
    env = dict(os.environ, HERMES_HOME=str(home), PYTHONPATH=str(REPO))
    env.pop("HERMES_DISABLE_LAZY_INSTALLS", None)
    env.pop("HERMES_MANAGED_DIR", None)
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-I", "-B", "-c", _child_script(script, enforce_closure=enforce_closure)],
        cwd=str(cwd or REPO), env=env, capture_output=True, text=True, timeout=120,
    )


def test_runtime_dependency_closure_is_enforced_by_this_pin(tmp_path):
    """Guard the guard: the blocker must actually refuse a non-declared module.

    Without this the pin could pass by blocking nothing at all.
    """
    assert _LIVE_COUPLING_MODULE not in _DECLARED, (
        f"{_LIVE_COUPLING_MODULE} is now declared by pm/pyproject.toml; the closure "
        "pin needs a different non-declared module to stay discriminating")
    script = f"""
try:
    import {_LIVE_COUPLING_MODULE}
except ModuleNotFoundError as exc:
    print("refused:", exc.name)
else:
    print("LEAKED: the closure blocker let {_LIVE_COUPLING_MODULE} through")
"""
    result = _run(script, home=tmp_path / "home")
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"refused: {_LIVE_COUPLING_MODULE}" in result.stdout, result.stdout + result.stderr


def test_policy_read_is_clean_on_a_minimal_runtime(tmp_path):
    """THE PIN. The policy read must not trigger app-level provider discovery.

    Drives the real ``pm.install.lazy_installs_allowed()`` in a child whose
    dependency closure is PM's declared set. Under the coupling
    (``from hermes_cli.config import ...``) the import of that module runs provider
    discovery, ``solstice`` reaches ``httpx``, and the runtime prints
    ``Failed to load bundled provider plugin solstice``. With the read routed
    through ``pm.config_policy`` nothing outside the runtime's own graph is touched
    and stderr stays empty.
    """
    home = tmp_path / "home"
    home.mkdir()
    script = """
from pm.install import lazy_installs_allowed
print("policy:", lazy_installs_allowed())
"""
    result = _run(script, home=home)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "policy: True" in result.stdout, result.stdout + result.stderr
    assert "Failed to load bundled provider plugin" not in result.stderr, result.stderr
    assert _LIVE_COUPLING_MODULE not in result.stderr, result.stderr


def test_pm_runtime_graph_never_imports_hermes_cli_config(tmp_path):
    """The coupling itself: nothing PM's runtime imports may pull ``hermes_cli.config``.

    Fails the import outright rather than degrading to a log line, so a new
    ``hermes_cli.config`` import anywhere in the runtime graph reds here even when
    the module happens to import cleanly in the developer's environment.

    The probe raises from the module's LOADER, not from ``find_spec``: a lookup is
    not an import, and ``pm.config_policy.config_system_present`` legitimately asks
    whether the module EXISTS without importing it. Only an actual ``exec_module``
    is the coupling.
    """
    home = tmp_path / "home"
    home.mkdir()
    script = """
import importlib.util
import sys


class _RefuseLoad:
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise AssertionError("pm's runtime graph imported hermes_cli.config")


class _NoConfigImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "hermes_cli.config":
            return None
        return importlib.util.spec_from_loader(fullname, _RefuseLoad())


sys.meta_path.insert(0, _NoConfigImport())
import pm.install
from pm import client, operations, workspace, _uv, registry, runtime, config_policy
assert config_policy.config_system_present() is True, "the module is present; only its load is refused"
print("policy:", pm.install.lazy_installs_allowed())
"""
    result = _run(script, home=home)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "policy: True" in result.stdout, result.stdout + result.stderr
    assert "AssertionError" not in result.stderr, result.stderr


def test_default_matches_the_config_system_default(tmp_path):
    """``pm.config_policy``'s literal default must equal the config system's.

    The runtime cannot import ``hermes_cli.config`` to read ``DEFAULT_CONFIG``, so the
    default is a literal here; this pin is what keeps the two from drifting. The
    import happens in THIS test process (which has the full application), never in the
    runtime.
    """
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    from pm import config_policy

    assert (config_policy.DEFAULT_ALLOW_LAZY_INSTALLS
            == DEFAULT_CONFIG["security"]["allow_lazy_installs"])


@pytest.mark.parametrize("content,expected", [
    ("{}\n", True),
    ("security:\n  allow_lazy_installs: true\n", True),
    ("security:\n  allow_lazy_installs: false\n", False),
    ("security:\n  allow_lazy_installs: 'false'\n", False),
    ("security: [\n", False),
])
def test_real_policy_is_boolean_and_fails_closed(tmp_path, content, expected):
    """The same contract ``tests/pm/test_lazy_install_policy.py`` pins, through the new read.

    An unparseable config must FAIL CLOSED: a broken security policy never grants
    permission to acquire packages.
    """
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(content, encoding="utf-8")
    script = """
from pm.install import lazy_installs_allowed
print("policy:", lazy_installs_allowed())
"""
    result = _run(script, home=home)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"policy: {expected}" in result.stdout, result.stdout + result.stderr


def test_internal_disable_overrides_enabled_config(tmp_path):
    """``HERMES_DISABLE_LAZY_INSTALLS`` short-circuits before any config read."""
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("security:\n  allow_lazy_installs: true\n", encoding="utf-8")
    script = """
from pm.install import lazy_installs_allowed
print("policy:", lazy_installs_allowed())
"""
    result = _run(script, home=home, extra_env={"HERMES_DISABLE_LAZY_INSTALLS": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "policy: False" in result.stdout, result.stdout + result.stderr


def test_managed_scope_pin_wins_over_user_config(tmp_path):
    """An administrator-pinned value beats the user's, matching the config overlay."""
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("security:\n  allow_lazy_installs: true\n", encoding="utf-8")
    managed = tmp_path / "etc-hermes"
    managed.mkdir()
    (managed / "config.yaml").write_text("security:\n  allow_lazy_installs: false\n", encoding="utf-8")
    script = """
from pm.install import lazy_installs_allowed
print("policy:", lazy_installs_allowed())
"""
    result = _run(script, home=home, extra_env={"HERMES_MANAGED_DIR": str(managed)})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "policy: False" in result.stdout, result.stdout + result.stderr


def test_bootstrap_stage_without_hermes_cli_counts_as_allowed(tmp_path):
    """The tool-only build stage has no config system; the policy read must not need it.

    Mirrors ``tests/pm/test_bootstrap_import_closure.py``: only ``pm/``,
    ``hermes_constants.py``, ``utils.py`` and ``hermes_yaml.py`` are staged. The read
    must answer (bootstrap ⇒ allowed) rather than raising out of the policy.
    """
    stage = tmp_path / "stage"
    shutil.copytree(REPO / "pm", stage / "pm", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("hermes_constants.py", "utils.py", "hermes_yaml.py"):
        shutil.copy2(REPO / name, stage / name)
    (stage / "hermes_cli").mkdir()
    for name in ("__init__.py", "runtime_state.py"):
        shutil.copy2(REPO / "hermes_cli" / name, stage / "hermes_cli" / name)
    assert not (stage / "hermes_cli" / "config.py").exists()

    script = """
import sys
sys.path.insert(0, %r)
from pm import config_policy
print("present:", config_policy.config_system_present())
try:
    print("policy:", config_policy.allow_lazy_installs())
except config_policy.BootstrapEnvironment as exc:
    print("bootstrap:", exc.name)
from pm.install import lazy_installs_allowed
print("effective:", lazy_installs_allowed())
""" % str(stage)
    env = dict(os.environ, HERMES_HOME=str(tmp_path / "home"))
    env.pop("HERMES_DISABLE_LAZY_INSTALLS", None)
    env.pop("HERMES_MANAGED_DIR", None)
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        cwd=str(stage), env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "present: False" in result.stdout
    assert "bootstrap: hermes_cli.config" in result.stdout, result.stdout
    assert "effective: True" in result.stdout, result.stdout + result.stderr
