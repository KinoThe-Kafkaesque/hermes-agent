"""The one config.yaml read PM's runtime may do, without importing ``hermes_cli.config``.

PM's own runtime is a deliberately minimal venv (``pm/pyproject.toml``: packaging,
tomli-w, ruamel.yaml, truststore). Importing ``hermes_cli.config`` is NOT a config
read: that module runs bundled model-provider plugin discovery at import time
(``hermes_cli/config.py`` ``_inject_profile_env_vars()``), which imports every
``plugins/model-providers/<name>/`` — and any plugin that reaches a third-party
dependency the runtime does not carry (``solstice`` → ``agent.gemini_native_adapter``
→ ``httpx``) fails at import. The runtime then prints, on every generation:

    Failed to load bundled provider plugin solstice: No module named 'httpx'

Measured: ``import hermes_cli.config`` pulls 357 modules, 62 of them application
modules (provider discovery, ``agent.transports``, 37 model-provider plugins).
Repairing the app cannot require resolving it, so the single boolean the runtime
needs is read here instead: stdlib + ``hermes_constants`` + ``utils`` (→
``hermes_yaml`` → ruamel, a declared dependency) + ``hermes_cli.managed_scope``
(which imports no config module). Nothing here may import ``hermes_cli.config``.

Semantics preserved from the previous ``hermes_cli.config`` read, which callers
depend on (``pm/install.py::lazy_installs_allowed``, ``pm/client.py``,
``pm/operations.py``, ``pm/_uv.py``, ``pm/workspace.py``):

* bootstrap — the config system is genuinely absent (the Docker toolchain stage
  copies only ``pm/``; the hermetic stage copies ``pm/``, ``hermes_constants.py``
  and two ``hermes_cli`` modules) — counts as ALLOWED, raised as
  :class:`BootstrapEnvironment` so the caller keeps its own answer;
* an unreadable, unparseable, or non-mapping config.yaml FAILS CLOSED (the caller
  returns ``False``), because a broken security policy must not grant permission
  to acquire packages;
* a missing config.yaml uses the documented default (``True``);
* a managed-scope (``/etc/hermes``) pin WINS over the user's value, matching
  ``hermes_cli.config._merge_managed_overlay``;
* the value must be exactly ``True`` (``'false'``/``0``/``yes`` strings are not).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Optional

POLICY_KEYS = ("security", "allow_lazy_installs")

#: ``security.allow_lazy_installs``'s documented default, for a config that does not
#: set it (and for a home with no config.yaml yet). The authority is
#: ``hermes_cli/config_defaults.py`` ``DEFAULT_CONFIG["security"]`` — the same value
#: ``hermes_cli.config.load_config_readonly()`` merges under a user config. Kept as a
#: literal because the runtime may not import that module's package; ``tests/pm/
#: test_runtime_config_policy.py`` pins the two together so they cannot drift.
DEFAULT_ALLOW_LAZY_INSTALLS = True


class BootstrapEnvironment(ModuleNotFoundError):
    """The config system is genuinely absent: there is no policy to consult.

    Raised (not returned) so the caller keeps its own bootstrap answer instead of
    this module inventing one. ``ModuleNotFoundError`` with ``name`` set to
    ``hermes_cli.config`` so a caller that does not know this type still lands in
    the same fail-closed clause the previous ``from hermes_cli.config import ...``
    produced.
    """

    def __init__(self, message: str, name: str = "hermes_cli.config") -> None:
        super().__init__(message, name=name)


def config_system_present() -> bool:
    """True when ``hermes_cli.config`` exists, WITHOUT importing it.

    ``find_spec`` imports only the parent package, and ``hermes_cli/__init__`` is a
    lazy facade (no plugin discovery), so this stays import-light. Resolution can
    fail for a missing parent (``ModuleNotFoundError``, an ``ImportError``), a
    namespace package with no ``__path__`` (``ValueError``) or a finder that returns
    a malformed spec (``AttributeError``); all of them mean the same thing here —
    there is no config system to consult.
    """
    try:
        return importlib.util.find_spec("hermes_cli.config") is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _parse_yaml(text: str) -> Any:
    """The runtime's own YAML reader; a parse failure is the caller's to handle."""
    import utils

    return utils.fast_safe_load(text)


def _leaf(config: Any, keys: tuple[str, ...]) -> Any:
    """Walk ``keys`` through nested mappings; a non-mapping hop is a miss."""
    node = config
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def read_config_file(path: Path) -> dict:
    """Parse ``path`` as a mapping, or raise.

    Missing file → ``{}`` (first run; the CLI seeds it later). Empty document →
    ``{}``. A parse error, an unreadable file, or a non-mapping top level raises:
    the caller turns any of those into the fail-closed answer.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    config = _parse_yaml(text)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"top-level YAML must be a mapping, got {type(config).__name__}")
    return config


def _managed_config() -> Optional[dict]:
    """The administrator-pinned config, or None when no managed scope is present.

    ``hermes_cli.managed_scope`` is import-safe for this runtime (stdlib + utils
    only) and is the ONE seam for managed locations, so the runtime asks it rather
    than re-deriving ``/etc/hermes``.
    """
    from hermes_cli import managed_scope

    config = managed_scope.load_managed_config()
    return config if isinstance(config, dict) and config else None


def allow_lazy_installs() -> Any:
    """``security.allow_lazy_installs`` from the effective config, or the documented default.

    Returns the effective leaf so the caller keeps its exact-``True`` test. Raises
    :class:`BootstrapEnvironment` when there is no config system at all, and the
    reader's own error when a config exists but cannot be read or parsed.
    """
    if not config_system_present():
        raise BootstrapEnvironment("hermes_cli.config is not installed")

    from hermes_constants import get_hermes_home

    user = _leaf(read_config_file(Path(get_hermes_home()) / "config.yaml"), POLICY_KEYS)
    managed = _managed_config()
    pinned = None if managed is None else _leaf(managed, POLICY_KEYS)
    if pinned is not None:
        return pinned
    return DEFAULT_ALLOW_LAZY_INSTALLS if user is None else user
