"""Dispatch-time provider distribution: demotion on a quota wall, failover on requeue,
per-run provider attribution, and a no-op on a healthy fleet.

These are behaviour contracts, not change-detectors: each test drives the real
dispatcher path (``_worker_argv`` / ``_classify_dead_worker_exit`` /
``detect_crashed_workers``) and asserts what a spawned worker would actually run
on, or what a run row records, so a regression in the wiring — not merely in the
arithmetic — fails here.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_provider_distribution as kpd
from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER, exit_single_query

PRIMARY = ("opencode-go", "step-5-preview-free")
SECONDARY = ("nous", "stepfun/step-5-preview:free")


@pytest.fixture(autouse=True)
def _clean_demotion_state():
    """The tally is process-local; every test starts from a healthy fleet."""
    kpd.reset_demotion_state()
    yield
    kpd.reset_demotion_state()


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    kbd._recent_worker_exits.clear()
    kb.init_db()
    return home


def _dead_worker_with_log(conn, tid: str, pid: int, rc: int,
                          provider: str | None = None, model: str | None = None) -> None:
    """Claim ``tid`` for a worker that already exited ``rc`` and wrote its own log.

    ``provider``/``model`` are what the worker's trailer reports as the route that
    served the run; omit both for the legacy (pre-attribution) trailer shape.
    """
    host = kb._claimer_id().split(":", 1)[0]
    kb.claim_task(conn, tid, claimer=f"{host}:w{pid}")
    conn.execute(
        "UPDATE tasks SET worker_pid=?, worker_started_at=NULL, started_at=? WHERE id=?",
        (pid, int(time.time()) - 120, tid),
    )
    conn.commit()
    log = kb.worker_log_path(tid)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"the model said something\n\n{KANBAN_WORKER_EXIT_TRAILER}{rc}"
                f"{kpd.trailer_fields(provider, model)}\n")


def _flag_value(argv: list[str], flag: str) -> str | None:
    """Value of *flag* in a worker argv, skipping the interpreter's own ``-m <module>``.

    The module invocation itself uses ``-m``, so a bare ``argv.index("-m")`` lands
    on the interpreter's module and reads the wrong token.
    """
    start = 0
    if "-m" in argv and argv[argv.index("-m") + 1] == "hermes_cli.main":
        start = argv.index("-m") + 2
    try:
        return argv[argv.index(flag, start) + 1]
    except ValueError:
        return None


def _argv_route(task) -> "tuple[list[str], str | None]":
    """``(argv, model)`` the worker would be launched with for *task*."""
    route, _reasons = kbd.resolve_task_route(task)
    argv = kbd._worker_argv(task, "scrapingcow", None, route)
    return argv, _flag_value(argv, "-m")


# --- Attribution (the observability prerequisite) ----------------------------


def test_rate_limited_run_records_the_provider_that_served_it(kanban_home):
    """A quota wall is attributed to a named provider/model, not just an exit code.

    This is the prerequisite for demotion: without it a wall cannot be told from
    another provider's, so demotion would be blind.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _dead_worker_with_log(conn, tid, 80001, kb.KANBAN_RATE_LIMIT_EXIT_CODE, *PRIMARY)

        kbd.detect_crashed_workers(conn)

        run = conn.execute(
            "SELECT metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)
        ).fetchone()
        meta = kb._json_dict(run["metadata"])
        assert meta["provider"] == PRIMARY[0]
        assert meta["model"] == PRIMARY[1]
        assert meta["exit_code"] == kb.KANBAN_RATE_LIMIT_EXIT_CODE
        assert meta["wall_count"] == 1


def test_completed_and_crashed_runs_also_record_their_provider(kanban_home):
    """Attribution is not rate-limit-only: any outcome names its serving route."""
    with kbc.connect() as conn:
        crashed = kb.create_task(conn, title="crash", assignee="a")
        _dead_worker_with_log(conn, crashed, 80002, 1, *SECONDARY)

        kbd.detect_crashed_workers(conn)

        run = conn.execute(
            "SELECT metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (crashed,)
        ).fetchone()
        meta = kb._json_dict(run["metadata"])
        assert meta["provider"] == SECONDARY[0]
        assert meta["model"] == SECONDARY[1]


def test_legacy_worker_without_provider_still_classifies(kanban_home):
    """An older binary that names no provider parses fine and demotes nothing.

    Backward compatibility is load-bearing: a fleet mid-upgrade has both shapes
    running, and an unnamed wall must never be charged to a provider.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _dead_worker_with_log(conn, tid, 80003, kb.KANBAN_RATE_LIMIT_EXIT_CODE)

        kbd.detect_crashed_workers(conn)

        run = conn.execute(
            "SELECT outcome, metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)
        ).fetchone()
        meta = kb._json_dict(run["metadata"])
        assert run["outcome"] == "rate_limited"
        assert meta["provider"] is None
        assert "wall_count" not in meta
        assert kpd.walled_routes() == set()
        # The requeue still did not count a failure.
        assert kb.get_task(conn, tid).consecutive_failures == 0


def test_rate_limit_requeue_never_counts_a_failure(kanban_home):
    """The distribution layer must not weaken the existing failover accounting."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _dead_worker_with_log(conn, tid, 80004, kb.KANBAN_RATE_LIMIT_EXIT_CODE, *PRIMARY)

        kbd.detect_crashed_workers(conn)

        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.consecutive_failures == 0


# --- Demotion on a quota wall ------------------------------------------------


def test_card_starts_on_next_provider_when_primary_is_walled(kanban_home):
    """Over-threshold provider is demoted: the spawned worker runs on the next route."""
    with kbc.connect() as conn:
        task = kb.get_task(conn, kb.create_task(conn, title="t", assignee="scrapingcow"))

        # Healthy fleet: the primary is chosen, exactly as before this feature.
        _argv, healthy_model = _argv_route(task)
        assert healthy_model == PRIMARY[1]

        # Wall the primary past the threshold, then re-resolve.
        for _ in range(kpd.demotion_threshold()):
            kpd.record_rate_limit(*PRIMARY)
        _argv, demoted_model = _argv_route(task)
        assert demoted_model == SECONDARY[1]


def test_worker_argv_carries_the_chosen_provider(kanban_home):
    """The decision reaches the spawn argv — an unpassed route would be inert."""
    with kbc.connect() as conn:
        task = kb.get_task(conn, kb.create_task(conn, title="t", assignee="scrapingcow"))
        for _ in range(kpd.demotion_threshold()):
            kpd.record_rate_limit(*PRIMARY)
        route, _reasons = kbd.resolve_task_route(task)
        argv = kbd._worker_argv(task, "scrapingcow", None, route)
        assert "-m" in argv and "--provider" in argv
        assert _flag_value(argv, "--provider") == SECONDARY[0]
        assert _flag_value(argv, "-m") == SECONDARY[1]


def test_explicit_pin_is_never_demoted(kanban_home):
    """A per-card ``model_override``/``provider_override`` always wins."""
    with kbc.connect() as conn:
        task = kb.get_task(conn, kb.create_task(
            conn, title="pinned", assignee="scrapingcow",
            model_override="some-model", provider_override="some-provider"))
        # Wall the pinned route past the threshold; the pin must still hold.
        for _ in range(kpd.demotion_threshold() * 3):
            kpd.record_rate_limit("some-provider", "some-model")
        route, reasons = kbd.resolve_task_route(task)
        assert (route.provider, route.model) == ("some-provider", "some-model")
        assert any("pinned by card" in r for r in reasons)
        argv = kbd._worker_argv(task, "scrapingcow", None, route)
        assert _flag_value(argv, "-m") == "some-model"


def test_every_route_walled_fails_open_on_the_primary(kanban_home):
    """Demotion is a preference, never a block: the lane still spawns."""
    chain = kpd.build_chain("scrapingcow", None)
    for route in chain:
        for _ in range(kpd.demotion_threshold()):
            kpd.record_rate_limit(route.provider, route.model)
    selected, reasons = kpd.select_route(chain)
    assert selected is not None
    assert (selected.provider, selected.model) == PRIMARY
    assert any("fail-open" in r for r in reasons)


# --- Failover on requeue -----------------------------------------------------


def test_rate_limited_requeue_advances_to_the_next_provider(kanban_home):
    """A card that just walled does not restart on the same provider."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="scrapingcow")
        _dead_worker_with_log(conn, tid, 80005, kb.KANBAN_RATE_LIMIT_EXIT_CODE, *PRIMARY)
        kbd.detect_crashed_workers(conn)

        # Re-claim so the spawn path can read the requeued run, then resolve.
        claimed = kb.claim_task(conn, tid, claimer=f"{kb._claimer_id()}")
        assert claimed is not None
        route, reasons = kbd.resolve_task_route(claimed, skip=kbd._requeue_failover_skip(claimed))
        assert (route.provider, route.model) == SECONDARY
        assert any("skip" in r for r in reasons)


def test_clean_requeue_does_not_skip_any_provider(kanban_home):
    """Failover is driven by a rate-limit requeue, not by every retry."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="scrapingcow")
        _dead_worker_with_log(conn, tid, 80006, 1, *PRIMARY)
        kbd.detect_crashed_workers(conn)
        claimed = kb.claim_task(conn, tid, claimer=f"{kb._claimer_id()}")
        assert kbd._requeue_failover_skip(claimed) == set()


# --- Chain construction ------------------------------------------------------


def test_profile_fallback_chain_is_respected_and_defaulted(kanban_home):
    """gomu's own chain leads; a profile with ``[]`` still gets the default tail."""
    gomu = kpd.build_chain("gomu", None)
    assert [r.describe() for r in gomu][0] == "opencode-go/step-5-preview-free"
    # gomu declares nous + workbuddy itself, so no duplicate is appended.
    assert len({r.identity for r in gomu}) == len(gomu)

    bare = kpd.build_chain("scrapingcow", None)
    assert len(bare) >= 2
    assert bare[0].identity == PRIMARY


def test_per_model_reasoning_effort_is_resolved_per_tier():
    """Each tier carries its own effort via the shared spelling-tolerant resolver."""
    chain = kpd.build_chain("scrapingcow", None)
    by_model = {r.model: r.reasoning for r in chain}
    assert by_model[PRIMARY[1]] == "xhigh"
    assert by_model[SECONDARY[1]] == "xhigh"
    assert by_model["deepseek-v4.1-flash-1m"] == "high"


# --- Trailer round-trip ------------------------------------------------------


def test_exit_trailer_round_trips_the_serving_route(monkeypatch, capsys):
    """What the worker writes is what the dispatcher parses back."""
    from hermes_cli.kanban_provider_distribution import parse_exit_trailer

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1")
    # A CLI stub carrying the route the agent resolved onto.
    class _Agent:
        provider, model = PRIMARY

    class _Cli:
        agent = _Agent()

    from hermes_cli import quiet_single_query as qsq

    qsq.bind_exit_trailer_cli(_Cli())
    with pytest.raises(SystemExit) as exc:
        exit_single_query(kb.KANBAN_RATE_LIMIT_EXIT_CODE)
    assert exc.value.code == kb.KANBAN_RATE_LIMIT_EXIT_CODE

    trailer = parse_exit_trailer(capsys.readouterr().err)
    assert trailer is not None
    assert trailer.exit_code == kb.KANBAN_RATE_LIMIT_EXIT_CODE
    assert (trailer.provider, trailer.model) == PRIMARY


def test_exit_trailer_omits_provider_when_the_route_is_unknown(monkeypatch, capsys):
    """An unresolvable route writes the legacy trailer rather than a half-named one."""
    from hermes_cli import quiet_single_query as qsq
    from hermes_cli.kanban_provider_distribution import parse_exit_trailer

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1")
    qsq.bind_exit_trailer_cli(None)
    with pytest.raises(SystemExit):
        exit_single_query(1)
    err = capsys.readouterr().err
    trailer = parse_exit_trailer(err)
    assert trailer is not None
    assert trailer.exit_code == 1
    assert trailer.provider is None
    assert trailer.model is None


# --- End-to-end through the spawn path ---------------------------------------
# The argv tests above prove what ``_worker_argv`` builds; these prove the
# dispatcher actually reaches it with the route, which is the leg that makes the
# decision real rather than inert.


def _capture_spawn(monkeypatch) -> dict:
    """Intercept ``subprocess.Popen`` and record the worker argv/env instead of spawning."""
    captured: dict = {}

    class _FakeProc:
        pid = 99999

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env", {})
        return _FakeProc()

    monkeypatch.setattr("subprocess.Popen", _fake_popen)
    return captured


def _spawn(task, captured) -> None:
    kbd._default_spawn(task, "/tmp", board=None)


def test_default_spawn_places_the_worker_on_the_demoted_provider(kanban_home, monkeypatch):
    """The full spawn path: a walled primary means the real worker runs on the fallback.

    This is the acceptance criterion — "prove by forcing the threshold and
    observing the spawned provider" — so it drives ``_default_spawn`` itself
    rather than only the argv builder.
    """
    captured = _capture_spawn(monkeypatch)
    with kbc.connect() as conn:
        task = kb.get_task(conn, kb.create_task(conn, title="t", assignee="scrapingcow"))
        kb.claim_task(conn, task.id)

        # Healthy: the primary is what gets spawned.
        _spawn(task, captured)
        assert _flag_value(captured["cmd"], "-m") == PRIMARY[1]

        # Wall the primary, re-claim (fresh run row), and spawn again.
        for _ in range(kpd.demotion_threshold()):
            kpd.record_rate_limit(*PRIMARY)
        kb.recompute_ready(conn, failure_limit=99)
        reclaimed = kb.get_task(conn, task.id)
        kb.claim_task(conn, task.id)
        _spawn(reclaimed, captured)
        assert _flag_value(captured["cmd"], "-m") == SECONDARY[1]
        assert _flag_value(captured["cmd"], "--provider") == SECONDARY[0]


def test_default_spawn_records_the_route_on_the_run_row(kanban_home, monkeypatch):
    """"Which provider served this card" is answerable from the run metadata."""
    captured = _capture_spawn(monkeypatch)
    with kbc.connect() as conn:
        task = kb.get_task(conn, kb.create_task(conn, title="t", assignee="scrapingcow"))
        kb.claim_task(conn, task.id)
        for _ in range(kpd.demotion_threshold()):
            kpd.record_rate_limit(*PRIMARY)
        _spawn(task, captured)

        row = conn.execute(
            "SELECT metadata FROM task_runs WHERE id = "
            "(SELECT current_run_id FROM tasks WHERE id = ?) AND ended_at IS NULL",
            (task.id,),
        ).fetchone()
        meta = kb._json_dict(row["metadata"])
        assert meta["provider"] == SECONDARY[0]
        assert meta["model"] == SECONDARY[1]
        assert f"{PRIMARY[0]}/{PRIMARY[1]}" in meta["demoted_routes"]


def test_default_spawn_fails_open_when_route_resolution_breaks(kanban_home, monkeypatch):
    """A resolution failure must not stop a card from spawning."""
    captured = _capture_spawn(monkeypatch)

    def _boom(*_a, **_kw):
        raise RuntimeError("resolution exploded")

    monkeypatch.setattr(kbd, "resolve_task_route", _boom)
    with kbc.connect() as conn:
        task = kb.get_task(conn, kb.create_task(conn, title="t", assignee="scrapingcow"))
        kb.claim_task(conn, task.id)
        _spawn(task, captured)
    # Still spawned; the worker falls back to the profile's own model.
    assert captured["cmd"]
    assert _flag_value(captured["cmd"], "-m") is None
