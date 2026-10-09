"""Worker-shaped board isolation: ``HERMES_KANBAN_HOME`` alone cannot isolate a board.

The defect this covers: ``_board_path`` consults the ``env_var`` pin (``HERMES_KANBAN_DB``)
FIRST and short-circuits before ``kanban_home()``, and the dispatcher injects that pin into every
worker subprocess (``kanban_db_dispatch``). A probe that set only ``HERMES_KANBAN_HOME`` therefore
looked isolated — ``kanban_home()``, board dirs, workspaces and logs all landed under its temp
root — while ``kanban_db_path()`` still resolved the LIVE board, and its writes went there. Five
phantom cards reached the live board that way on 2026-10-09 and the real dispatcher claimed and
spawned workers on them.

The suite's autouse ``_kanban_write_guard`` (tests/conftest.py) cannot see this class: it refuses
the write instead of reproducing it, so these tests resolve and write paths directly and assert on
where the bytes landed.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_paths as kbp
from hermes_cli.kanban_db_connect import connect

_PINS = kbp.BOARD_PIN_ENV_VARS


@pytest.fixture(autouse=True)
def _clean_board_env(monkeypatch):
    """Start from the ambient env minus every kanban pin, like a bare shell."""
    for name in (*_PINS, "HERMES_KANBAN_HOME", "HERMES_KANBAN_TASK",
                 "HERMES_DELEGATED_CHILD_CONTEXT"):
        monkeypatch.delenv(name, raising=False)


def _worker_shaped(monkeypatch, tmp_path, live_db: Path) -> None:
    """The exact env every dispatcher-spawned worker runs with."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(live_db))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(tmp_path / "live-workspaces"))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")


def test_home_alone_cannot_silently_split_root_from_db(monkeypatch, tmp_path):
    """The defect class, pinned: with the dispatcher's pin inherited, setting
    ``HERMES_KANBAN_HOME`` used to move the *root* while the DB stayed pinned — a silent split
    that made the isolation look complete. It must now be loud, never silent."""
    live_db = tmp_path / "live" / "kanban.db"
    _worker_shaped(monkeypatch, tmp_path, live_db)

    # Production precedence is untouched: no explicit home, so the dispatcher's pin wins.
    assert kb.kanban_db_path() == live_db

    isolated = tmp_path / "isolated-home"
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(isolated))
    assert kb.kanban_home() == isolated, "the root still follows the home"
    with pytest.raises(kb.KanbanBoardPinConflict):
        kb.kanban_db_path()  # …and the DB refuses to follow it silently


def test_isolated_board_clears_every_pin_and_moves_the_db(monkeypatch, tmp_path):
    """``isolated_board()`` cannot be half-done: root and DB both land in the temp dir."""
    live_db = tmp_path / "live" / "kanban.db"
    _worker_shaped(monkeypatch, tmp_path, live_db)

    with kb.isolated_board(root=tmp_path / "throwaway") as root:
        assert kb.kanban_home() == root
        assert kb.kanban_db_path() == root / "kanban.db"
        assert kb.workspaces_root() == root / "kanban" / "workspaces"
        assert kb.attachments_root() == root / "kanban" / "attachments"
        for name in _PINS:
            assert name not in os.environ, f"{name} survived isolated_board()"

    # And every pin is restored on exit — the worker's own board must not be re-pointed.
    assert os.environ["HERMES_KANBAN_DB"] == str(live_db)
    assert os.environ["HERMES_KANBAN_TASK"] == "t_worker"


def test_isolated_board_write_lands_there_and_not_on_the_live_board(monkeypatch, tmp_path):
    """The acceptance proof: a real write inside a worker-shaped env lands on the throwaway board
    and the live board gains nothing."""
    live_db = tmp_path / "live" / "kanban.db"
    _worker_shaped(monkeypatch, tmp_path, live_db)
    connect(live_db).close()  # the live board exists and is initialized

    with kb.isolated_board(root=tmp_path / "throwaway"):
        assert not kb.kanban_db_path().exists()
        with connect() as conn:
            kb.create_task(conn, title="throwaway card", assignee="nobody")
        isolated_db = kb.kanban_db_path()
        assert isolated_db.is_file(), "the write did not create the isolated board"

    import sqlite3
    with sqlite3.connect(live_db) as conn:
        live_titles = [row[0] for row in conn.execute("SELECT title FROM tasks")]
    with sqlite3.connect(isolated_db) as conn:
        isolated_titles = [row[0] for row in conn.execute("SELECT title FROM tasks")]

    assert isolated_titles == ["throwaway card"]
    assert live_titles == [], f"the throwaway write reached the live board: {live_titles!r}"


def test_contradictory_pin_and_explicit_home_fails_loud(monkeypatch, tmp_path):
    """Acceptance 2: ``HERMES_KANBAN_DB`` disagreeing with an explicitly-set
    ``HERMES_KANBAN_HOME`` is a loud, actionable error instead of a silent live-board write."""
    live_db = tmp_path / "live" / "kanban.db"
    _worker_shaped(monkeypatch, tmp_path, live_db)
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "intended-home"))

    with pytest.raises(kb.KanbanBoardPinConflict) as excinfo:
        kb.kanban_db_path()
    message = str(excinfo.value)
    assert str(live_db) in message and "HERMES_KANBAN_HOME" in message
    assert "isolated_board" in message, "the error must name the supported remedy"


def test_pin_inside_the_home_keeps_working(monkeypatch, tmp_path):
    """A deployment that pins both to one root is not a contradiction: the dispatcher's own
    injection and the Docker layout put the DB under the home, and must stay silent."""
    root = tmp_path / "shared-root"
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(root))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(root / "kanban.db"))

    assert kb.kanban_db_path() == root / "kanban.db"
    # A named board resolves under the same home (the dispatcher enumerates slugs), and the pin
    # stays consistent with it.
    assert (root / "kanban" / "boards" / "other" / "kanban.db") == kb.kanban_db_path(board="other")


def test_board_pin_conflict_is_exported_from_the_facade(monkeypatch, tmp_path):
    """Callers import kanban paths from ``hermes_cli.kanban_db``; the guard must be reachable
    there too, or a caller cannot catch it by name."""
    assert kb.KanbanBoardPinConflict is kbp.KanbanBoardPinConflict
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "elsewhere.db"))
    with pytest.raises(kb.KanbanBoardPinConflict):
        kb.kanban_db_path()


def test_isolated_board_works_from_inside_a_fenced_descendant(tmp_path):
    """A ``delegate_task`` descendant's marker fences the pinned DB and its own board root, so a
    temp root under the lineage root yields a silently read-only board. The default root must be
    chosen outside that fence, and a real write must succeed there."""
    repo_root = Path(__file__).resolve().parents[2]
    script = tmp_path / "fenced_probe.py"
    script.write_text(
        "import json, os, sys\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        "from hermes_cli import kanban_db as kb\n"
        "from hermes_cli.kanban_db_connect import connect\n"
        "with kb.isolated_board() as root:\n"
        "    with connect() as conn:\n"
        "        kb.create_task(conn, title='from a descendant')\n"
        "    out = {'root': str(root), 'db': str(kb.kanban_db_path()),\n"
        "           'outside_fence': not str(kb.kanban_db_path()).startswith(os.environ['HERMES_DELEGATED_CHILD_CONTEXT'])}\n"
        "print('ISOLATED=' + json.dumps(out))\n",
        encoding="utf-8",
    )
    lineage_root = tmp_path / "lineage"
    lineage_root.mkdir()
    env = dict(os.environ)
    env["HOME"] = str(tmp_path)
    env["HERMES_HOME"] = str(lineage_root)
    env["TMPDIR"] = str(lineage_root / "scratch")  # a worker exports its scratch dir here
    (lineage_root / "scratch").mkdir()
    env["HERMES_DELEGATED_CHILD_CONTEXT"] = str(lineage_root)
    env["HERMES_KANBAN_DB"] = str(lineage_root / "kanban.db")
    env.pop("HERMES_KANBAN_TASK", None)

    proc = subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=90,
    )
    assert proc.returncode == 0, proc.stderr
    payload = [line for line in proc.stdout.splitlines() if line.startswith("ISOLATED=")][-1]
    import json
    result = json.loads(payload.split("=", 1)[1])
    assert result["outside_fence"], (
        f"the isolated board landed inside the descendant's fenced root: {result['db']}"
    )
