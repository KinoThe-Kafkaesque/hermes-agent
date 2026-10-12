"""Dispatch-time provider distribution: demotion on a quota wall, failover on requeue,
per-run provider attribution, and a no-op on a healthy fleet.

These are behaviour contracts, not change-detectors: each test drives the real
dispatcher path (``_worker_argv`` / ``_classify_dead_worker_exit`` /
``detect_crashed_workers``) and asserts what a spawned worker would actually run
on, or what a run row records, so a regression in the wiring — not merely in the
arithmetic — fails here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
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
    """The in-process tally starts empty; the SHARED tally lives in each test's
    own temp board, so a fresh board is a healthy fleet without clearing state
    that belongs to a live dispatcher process."""
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


def _reap_this_processes_worker(pid: int, rc: int) -> None:
    """Feed the reap registry the way ``os.waitpid`` does for a child of THIS process.

    This is the gateway-embedded dispatcher: it reaps its own workers, so
    ``_classify_worker_exit`` returns a real kind and (before the fix) the trailer
    read was skipped entirely.
    """
    kbd._record_worker_exit(pid, (int(rc) & 0xFF) << 8)


def test_a_locally_reaped_wall_still_names_the_route_that_served_it(kanban_home):
    """The gateway-embedded dispatcher reaps its own children — and must still attribute.

    This is the production path. The trailer read used to be gated behind
    ``kind == "unknown"``, which is never true here, so every wall the gateway
    booked recorded ``provider: null`` and the shared demotion tally saw nothing:
    measured on the live board, 203 of 203 gateway-reaped ``rate_limited`` rows
    were null. The trailer names the route even when the rc came from the reap
    registry, so attribution must not depend on which process noticed the death.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _dead_worker_with_log(conn, tid, 80011, kb.KANBAN_RATE_LIMIT_EXIT_CODE, *PRIMARY)

        _reap_this_processes_worker(80011, kb.KANBAN_RATE_LIMIT_EXIT_CODE)
        kbd.detect_crashed_workers(conn)

        run = conn.execute(
            "SELECT outcome, metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)
        ).fetchone()
        meta = kb._json_dict(run["metadata"])
        assert run["outcome"] == "rate_limited"
        assert meta["provider"] == PRIMARY[0], "a locally reaped wall lost its route"
        assert meta["model"] == PRIMARY[1]
        assert meta["wall_count"] == 1


def test_a_locally_reaped_wall_is_visible_to_the_shared_tally(kanban_home):
    """The end-to-end property: a gateway-reaped wall DEMOTES the route.

    Attribution is only worth anything if ``walled_routes()`` — the shared,
    cross-process read — acts on it. Before the fix the row was written with a
    null provider and ``_shared_wall_events`` skipped it, so the whole
    cross-process failover feature was deployed and inert.
    """
    with kbc.connect() as conn:
        for n, pid in enumerate((80021, 80022)):
            tid = kb.create_task(conn, title=f"t{n}", assignee="a")
            _dead_worker_with_log(conn, tid, pid, kb.KANBAN_RATE_LIMIT_EXIT_CODE, *PRIMARY)
            _reap_this_processes_worker(pid, kb.KANBAN_RATE_LIMIT_EXIT_CODE)
            kbd.detect_crashed_workers(conn)

        assert kpd.walled_routes() == {PRIMARY}, "the shared tally never saw the walls"


def test_a_stale_log_tail_from_a_later_run_is_not_charged_to_its_route(kanban_home):
    """The log is append-mode: a tail trailer from a LATER run must be discarded whole.

    Measured on the live board, 37 of 50 latest ``rate_limited`` runs (rc=75) tail
    a trailer carrying a DIFFERENT rc — typically rc=0 from the run that later
    succeeded, on the fallback provider. Trusting that trailer would stamp the
    wrong rc on the wall and charge it to a route that never served it. The
    observed rc is the discriminator, so the mismatched trailer is ignored — and
    the wall stays unnamed (demoting nothing) rather than guessed.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        # The log's last trailer is the LATER successful run on the fallback.
        _dead_worker_with_log(conn, tid, 80031, 0, *SECONDARY)
        # …but THIS run was reaped rate-limited on the primary.
        _reap_this_processes_worker(80031, kb.KANBAN_RATE_LIMIT_EXIT_CODE)
        kbd.detect_crashed_workers(conn)

        run = conn.execute(
            "SELECT outcome, metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)
        ).fetchone()
        meta = kb._json_dict(run["metadata"])
        assert run["outcome"] == "rate_limited"
        assert meta["provider"] is None, "a later run's route was charged to this wall"
        assert meta["model"] is None
        assert "wall_count" not in meta
        assert kpd.walled_routes() == set()


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


# --- Shared (cross-process) tally --------------------------------------------
# The tally used to be process-local: a wall observed by one dispatcher did not
# demote for another until each re-observed its own wall. The durable record of a
# wall is the ``rate_limited`` run row the reclaim transaction writes, attributed
# to the provider/model that served the run, so every dispatcher reads ONE tally
# from the board DB. These tests drive that path end to end.


def _persisted_wall(conn, tid: str, pid: int, provider: str | None, model: str | None,
                    *, ended_at: float | None = None, outcome: str = "rate_limited",
                    pid_in_metadata: bool = True) -> None:
    """Write the run row a wall leaves behind — the SHARED tally's only input.

    Shaped exactly like the reclaim transaction's: ``outcome=rate_limited``, the
    worker PID on the run row (``_set_worker_pid``) and in the exit payload, and
    the observed route in the metadata (``t_6ed58421``'s attribution).
    ``pid_in_metadata=False`` covers a run whose metadata lost the pid, which
    must still deduplicate through the run row's own column.
    """
    kb.claim_task(conn, tid, claimer=f"{kb._claimer_id().split(':', 1)[0]}:w{pid}")
    conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (pid, tid))
    run_id = kb.get_task(conn, tid).current_run_id
    conn.execute("UPDATE task_runs SET worker_pid=? WHERE id=?", (pid, run_id))
    meta: dict[str, object] = {"provider": provider, "model": model}
    if pid_in_metadata:
        meta["pid"] = pid
    conn.execute(
        "UPDATE task_runs SET outcome=?, status=?, ended_at=?, metadata=? WHERE id=?",
        (outcome, outcome, ended_at if ended_at is not None else int(time.time()),
         kb._json_or_null(meta), run_id),
    )
    conn.commit()


def test_a_persisted_wall_demotes_for_a_process_that_never_saw_it(kanban_home):
    """A wall another dispatcher recorded demotes HERE, with no local observation.

    This is the coverage gap the card closes: process B never reaps the worker,
    so it has no in-process stamp — it must still see the wall.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _persisted_wall(conn, tid, 70001, *PRIMARY)
        task = kb.get_task(conn, kb.create_task(conn, title="t2", assignee="scrapingcow"))

        # No in-process stamp at all: the shared tally alone must carry it.
        assert kpd.walled_routes() == set()
        _persisted_wall(conn, kb.create_task(conn, title="t3", assignee="a"),
                        70002, *PRIMARY)
        assert kpd.walled_routes() == {PRIMARY}

        # …and the spawn path acts on it.
        _argv, model = _argv_route(task)
        assert model == SECONDARY[1]


def test_a_wall_this_process_observed_is_not_counted_twice(kanban_home):
    """The stamp and the run row for the SAME wall are one wall, not two.

    Without the event-key dedupe the recording process's own wall would arrive
    from both sources, halving the effective threshold. The key comes from the
    pid in the run metadata (the reclaim payload).
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _persisted_wall(conn, tid, 70003, *PRIMARY)

        count = kpd.record_rate_limit(*PRIMARY, task_id=tid, pid=70003)
        assert count == 1, "the same wall was counted twice"
        assert kpd.walled_routes() == set()  # threshold 2; one wall is not enough

        # A genuinely DIFFERENT wall on the same route is a second one.
        count = kpd.record_rate_limit(*PRIMARY, task_id="t_other", pid=70004)
        assert count == 2
        assert kpd.walled_routes() == {PRIMARY}


def test_dedupe_also_works_without_a_pid_in_the_run_metadata(kanban_home):
    """A run whose metadata lost the pid still dedupes, via ``task_runs.worker_pid``.

    Both key sources must agree, or a wall recorded by this process would be
    counted twice and the threshold would halve.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _persisted_wall(conn, tid, 70006, *PRIMARY, pid_in_metadata=False)

        count = kpd.record_rate_limit(*PRIMARY, task_id=tid, pid=70006)
        assert count == 1, "the same wall was counted twice (run-row pid fallback)"
        assert kpd.walled_routes() == set()


def test_persisted_wall_outside_the_window_does_not_demote(kanban_home):
    """The window still expires: a wall older than 900 s is not a demotion."""
    with kbc.connect() as conn:
        for n, pid in enumerate((70010, 70011)):
            _persisted_wall(
                conn, kb.create_task(conn, title=f"old{n}", assignee="a"), pid, *PRIMARY,
                ended_at=int(time.time()) - (kpd.demotion_window_seconds() + 60),
            )
        assert kpd.walled_routes() == set()
        # A wall INSIDE the window is the control: the same read path demotes.
        _persisted_wall(conn, kb.create_task(conn, title="fresh", assignee="a"),
                        70012, *PRIMARY)
        _persisted_wall(conn, kb.create_task(conn, title="fresh2", assignee="a"),
                        70013, *PRIMARY)
        assert kpd.walled_routes() == {PRIMARY}


def test_unnamed_persisted_wall_demotes_nothing(kanban_home):
    """A legacy row with no attributed route is not charged to any provider.

    The arm writes TWO unnamed rows, not one: ``demotion_threshold()`` is 2, so a
    single unnamed row leaves ``walled_routes()`` empty whether or not the reader
    skips it — the assertion would hold for a reader that CHARGED the wall to a
    guessed route. Two rows make the property discriminating: a reader that
    substituted a provider here would reach the threshold and return a non-empty
    set, so this pin REDs under exactly that mutation (proven by
    ``mut_dir1.sh``: mutating the reader's ``route is None`` skip into a guess
    leaves the one-row form green and reds this two-row form).
    """
    with kbc.connect() as conn:
        for n, pid in enumerate((70020, 70021)):
            tid = kb.create_task(conn, title=f"legacy{n}", assignee="a")
            # The canonical fixture helper with no route: metadata carries
            # ``provider: null``, exactly the legacy row shape production wrote.
            _persisted_wall(conn, tid, pid, None, None)
        # Two unnamed walls, both inside the window: still no demotion, because
        # neither row names a route. The count assertion is the discriminating
        # half — the reader must not manufacture a route to reach the threshold.
        assert kpd.walled_routes() == set()
        assert kpd._wall_counts() == {}


def test_shared_tally_read_failure_is_fail_open(kanban_home, monkeypatch):
    """Mutation: point the tally read at a table that does not exist.

    The card must STILL spawn on the primary. A broken tally is a demotion
    signal lost, never a spawn blocked.
    """
    captured = _capture_spawn(monkeypatch)
    monkeypatch.setattr(
        kpd, "_WALL_QUERY",
        "SELECT task_id, ended_at, worker_pid, metadata FROM no_such_table_here",
    )
    with kbc.connect() as conn:
        task = kb.get_task(conn, kb.create_task(conn, title="t", assignee="scrapingcow"))
        kb.claim_task(conn, task.id)
        assert kpd.walled_routes() == set()
        assert kpd.shared_wall_read_error(), "the failure must be recorded, not swallowed"
        _spawn(task, captured)
    assert captured["cmd"], "the card must still spawn"
    assert _flag_value(captured["cmd"], "-m") == PRIMARY[1]
    # And a genuinely walled route is not rescued by the broken read either:
    # the in-process tally still demotes.
    for _ in range(kpd.demotion_threshold()):
        kpd.record_rate_limit(*PRIMARY, task_id="t_local", pid=1)
    assert kpd.walled_routes() == {PRIMARY}


def test_explicit_pin_is_never_demoted_by_a_persisted_wall(kanban_home):
    """The shared tally must not reach a pinned card either."""
    with kbc.connect() as conn:
        task = kb.get_task(conn, kb.create_task(
            conn, title="pinned", assignee="scrapingcow",
            model_override="some-model", provider_override="some-provider"))
        for pid in (70030, 70031, 70032):
            _persisted_wall(conn, kb.create_task(conn, title=f"w{pid}", assignee="a"),
                            pid, "some-provider", "some-model")
        route, reasons = kbd.resolve_task_route(task)
        assert (route.provider, route.model) == ("some-provider", "some-model")
        assert any("pinned by card" in r for r in reasons)


# --- The real two-process probe ---------------------------------------------
# Everything above is single-process. The card's acceptance is a REAL two-process
# proof: a wall recorded in process A is observed by process B inside the window.

_TWO_PROCESS_SCRIPT = r'''
import json, os, sys, time
sys.path.insert(0, sys.argv[1])
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_provider_distribution as kpd

role = sys.argv[2]
tid = sys.argv[3]
pid = int(sys.argv[4])
provider, model = sys.argv[5], sys.argv[6]

out = {"role": role, "pid": os.getpid()}
if role == "wall":
    # Record the wall exactly as the reclaim path does: stamp the in-process
    # tally AND close the run row (the shared record).
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
        count = kpd.record_rate_limit(provider, model, task_id=tid, pid=pid)
        out["stamp_count"] = count
        conn.execute(
            "UPDATE task_runs SET outcome='rate_limited', status='rate_limited', "
            "ended_at=?, metadata=? WHERE id=?",
            (int(time.time()),
             kb._json_or_null({"provider": provider, "model": model, "pid": pid}),
             task.current_run_id),
        )
        conn.execute("UPDATE task_runs SET worker_pid=? WHERE id=?",
                     (pid, task.current_run_id))
        conn.commit()
else:
    # A DIFFERENT process: fresh interpreter, empty in-process tally.
    out["walled_routes"] = sorted("{}|{}".format(p, m) for p, m in kpd.walled_routes())
    out["local_tally_size"] = len(kpd._wall_events)
    out["shared_read_error"] = kpd.shared_wall_read_error()
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
        route, reasons = kbd.resolve_task_route(task)
    out["chosen"] = "{}/{}".format(route.provider, route.model) if route else None
    out["reasons"] = reasons

print("PROBE=" + json.dumps(out))
'''


def test_two_processes_share_one_tally(kanban_home, tmp_path):
    """A wall recorded in process A is observed by process B inside the window.

    A real subprocess probe, not a single-process simulation: process B is a
    fresh interpreter with an empty in-process tally, so the only way it can see
    the wall is the shared (persisted) record.
    """
    repo_root = Path(__file__).resolve().parents[2]
    script = tmp_path / "two_process_wall_probe.py"
    script.write_text(_TWO_PROCESS_SCRIPT, encoding="utf-8")

    home = tmp_path / ".hermes"
    env = dict(os.environ)
    env["HERMES_HOME"] = str(home)
    env["HERMES_KANBAN_HOME"] = str(home)
    for pin in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD",
                "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_ATTACHMENTS_ROOT",
                "HERMES_KANBAN_TASK", "HERMES_DELEGATED_CHILD_CONTEXT"):
        env.pop(pin, None)

    with kbc.connect() as conn:
        wall_task = kb.create_task(conn, title="wall", assignee="scrapingcow")
        kb.claim_task(conn, wall_task, claimer=f"{kb._claimer_id().split(':', 1)[0]}:w71001")
        # The threshold is 2, so ONE persisted wall must not be enough…
        reader_task = kb.create_task(conn, title="reader", assignee="scrapingcow")

    def _run(role: str, tid: str, pid: int) -> dict:
        proc = subprocess.run(
            [sys.executable, str(script), str(repo_root), role, tid, str(pid), *PRIMARY],
            env=env, capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE=")][-1]
        return json.loads(line.split("=", 1)[1])

    first = _run("wall", wall_task, 71001)
    assert first["stamp_count"] == 1

    reader = _run("read", reader_task, 71001)
    assert reader["local_tally_size"] == 0, "the reader must have NO in-process tally"
    assert reader["pid"] != first["pid"], "the probe must be two processes"
    assert reader["shared_read_error"] is None
    assert reader["walled_routes"] == []          # one wall < threshold 2
    assert reader["chosen"] == f"{PRIMARY[0]}/{PRIMARY[1]}"

    # Process A records a SECOND wall; process B now demotes on the shared tally.
    with kbc.connect() as conn:
        second = kb.create_task(conn, title="wall2", assignee="scrapingcow")
        kb.claim_task(conn, second, claimer=f"{kb._claimer_id().split(':', 1)[0]}:w71002")
    _run("wall", second, 71002)

    reader2 = _run("read", reader_task, 71002)
    assert reader2["local_tally_size"] == 0
    assert reader2["walled_routes"] == [f"{PRIMARY[0]}|{PRIMARY[1]}"]
    assert reader2["chosen"] == f"{SECONDARY[0]}/{SECONDARY[1]}", (
        "process B did not demote on a wall it never observed"
    )

