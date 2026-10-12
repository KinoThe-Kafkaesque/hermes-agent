"""Live-owner reclaim must not fire on a healthy run (#131247, #115692).

A worker wedged on a lock passes ``_owner_is_live()`` forever, so its ``running`` row was never
reclaimed and every future fire of the job was skipped (#115692). The first fix bounded that with
``max(3 × HERMES_CRON_TIMEOUT, script timeout, 7200)`` — but ``7200`` is a bare wall-clock
constant, and a job whose own interval IS 120 minutes therefore had its reclaim bound sitting
exactly ON its period. Measured on ``bot-manager-cycle`` (``every 120m``): eight healthy runs of
122–125 minutes were reclaimed mid-flight, discarding their reports (#131247).

The bound is now derived from THIS job's own cadence and measured runtime envelope (each with
headroom), and a run whose own ``fire_claim`` heartbeat is still fresh is never reclaimed at all —
heartbeat is positive evidence of liveness, claim age is not evidence of death.
"""

from __future__ import annotations

import time
import uuid
from datetime import timedelta

import pytest

import cron.executions as executions_mod
from cron.executions import _hermes_now, _transaction, recover_interrupted_executions

# The real heartbeat reader, captured before the autouse fixture stubs it out (tests that exercise
# the reader's own logic re-install this). ``getattr`` keeps this module COLLECTABLE on a tree that
# predates the reader, so the regression pin below can be shown RED on the pre-fix revision rather
# than erroring at import.
_REAL_HEARTBEAT_READER = getattr(executions_mod, "_live_owner_heartbeat_is_fresh", None)


@pytest.fixture(autouse=True)
def _ledger(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(executions_mod, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    # The owner is alive for every row in this module; the bound and the heartbeat decide.
    monkeypatch.setattr(executions_mod, "_owner_is_live", lambda pid, started_at: True)
    # No job record by default: the cadence term contributes nothing and the heartbeat is absent.
    # ``raising=False`` keeps this fixture usable on a tree predating the heartbeat reader, so the
    # regression pin reports a real assertion failure there instead of a fixture ERROR.
    monkeypatch.setattr(
        executions_mod, "_live_owner_heartbeat_is_fresh", lambda job_id, at: False, raising=False
    )
    yield


def _job(monkeypatch, *, schedule: str, job_id: str = "job-x") -> str:
    """Create a real recurring job in the temp store and return its id."""
    from cron.jobs import create_job

    job = create_job(prompt="x", schedule=schedule, name=job_id)
    return job["id"]


def _seed_running(job_id: str, *, age_seconds: float, pid: int = 99999) -> str:
    claimed_at = (_hermes_now() - timedelta(seconds=age_seconds)).isoformat()
    with _transaction() as conn:
        eid = uuid.uuid4().hex
        conn.execute(
            """INSERT INTO executions
               (id, job_id, status, source, process_id, pid, process_started_at, claimed_at)
               VALUES (?, ?, 'running', 'cron', ?, ?, ?, ?)""",
            (eid, job_id, "ext-proc", pid, int(time.time()), claimed_at),
        )
    return eid


def _seed_terminal(job_id: str, *, duration_seconds: float, pid: int = 11111) -> str:
    """A completed attempt of *job_id* taking *duration_seconds* (the runtime-envelope sample)."""
    finished = _hermes_now()
    started = finished - timedelta(seconds=duration_seconds)
    with _transaction() as conn:
        eid = uuid.uuid4().hex
        conn.execute(
            """INSERT INTO executions
               (id, job_id, status, source, process_id, pid, process_started_at,
                claimed_at, started_at, finished_at)
               VALUES (?, ?, 'completed', 'cron', ?, ?, ?, ?, ?, ?)""",
            (eid, job_id, "old-proc", pid, int(time.time()), started.isoformat(),
             started.isoformat(), finished.isoformat()),
        )
    return eid


def _status(eid: str) -> str:
    with _transaction() as conn:
        return conn.execute("SELECT status FROM executions WHERE id=?", (eid,)).fetchone()["status"]


def _error(eid: str) -> str:
    with _transaction() as conn:
        return conn.execute("SELECT error FROM executions WHERE id=?", (eid,)).fetchone()["error"]


# ── The regression pin: a healthy run at its own interval is NOT reclaimed ──────────────────────

def test_healthy_run_at_its_own_interval_is_not_reclaimed(monkeypatch):
    """THE #131247 PIN — the CADENCE term is load-bearing here.

    A job that runs every 120 minutes with a SHORT runtime envelope (2 minutes: a job that does
    its work quickly) must still be bounded above its own period. On the pre-fix bound this row
    was reclaimed (7200 s == the job's own 120-minute period, so any run that legitimately
    exceeded one interval crossed it); on the runtime term alone it still would be (2 min p95 →
    only 4 min of envelope), so this pin can only pass while the cadence term exists.
    """
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "600")
    jid = _job(monkeypatch, schedule="every 120m")
    for _ in range(5):
        _seed_terminal(jid, duration_seconds=2 * 60)  # p95 = 2 min, far below the period

    with _transaction() as conn:
        bound = executions_mod._live_owner_stale_after_seconds(conn, jid)
    # 2 × the job's 120-minute period is the winning term; the derived ceilings are far below it.
    assert bound == pytest.approx(240 * 60.0), f"bound {bound}"

    # 122 minutes — the exact measured duration of the reclaimed 2026-10-11 19:44 run.
    eid = _seed_running(jid, age_seconds=122 * 60)

    assert recover_interrupted_executions() == 0
    assert _status(eid) == "running"

    # The property is stated on the PERIOD, not on 122 minutes: it holds for every run that
    # overruns its own interval, which is what the old bound got wrong.
    assert bound > 120 * 60


def test_short_interval_job_reclaims_only_past_its_own_scaled_bound(monkeypatch):
    """The cadence term is per JOB, not a global floor: a 5-minute job is still reclaimable.

    Guards against the fix degenerating into "never reclaim" — the #115692 property must survive.
    """
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "600")
    jid = _job(monkeypatch, schedule="every 5m")
    with _transaction() as conn:
        bound = executions_mod._live_owner_stale_after_seconds(conn, jid)
    # max(3×600=1800, script timeout 3600, 5m×2=600) — the derived ceilings dominate a short job.
    assert bound == pytest.approx(3600.0)

    fresh = _seed_running(jid, age_seconds=60, pid=88888)
    stale = _seed_running(jid, age_seconds=bound + 60, pid=77777)

    assert recover_interrupted_executions() == 1
    assert _status(fresh) == "running"
    assert _status(stale) == "unknown"
    assert "outlived the derived stale bound" in _error(stale)


def test_runtime_envelope_widens_the_bound(monkeypatch):
    """A job whose own runs are long is bounded above its REAL envelope, not at a fixed constant.

    ``runtime_p95 × 2`` must be the winning term when it exceeds the others, and a run inside it
    must survive.
    """
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "600")
    jid = _job(monkeypatch, schedule="every 10m")
    for _ in range(10):
        _seed_terminal(jid, duration_seconds=90 * 60)  # p95 = 90 min

    with _transaction() as conn:
        bound = executions_mod._live_owner_stale_after_seconds(conn, jid)
    assert bound == pytest.approx(180 * 60.0)  # 2 × p95, above every other term

    inside = _seed_running(jid, age_seconds=100 * 60, pid=66666)
    assert recover_interrupted_executions() == 0
    assert _status(inside) == "running"


# ── Heartbeat: positive liveness evidence beats claim age ──────────────────────────────────────

def test_fresh_heartbeat_is_never_reclaimed_however_old_the_claim(monkeypatch):
    """A run that is still heartbeating is ALIVE, so its claim must survive any age.

    This is the second half of the fix: claim age alone cannot distinguish "wedged" from "still
    working", and the heartbeat is the signal that can. Even far past the derived bound, a fresh
    stamp keeps the row owned.
    """
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "600")
    monkeypatch.setattr(
        executions_mod, "_live_owner_heartbeat_is_fresh", lambda job_id, at: True
    )
    jid = _job(monkeypatch, schedule="every 5m")
    eid = _seed_running(jid, age_seconds=30 * 24 * 3600)  # 30 days: past every derived term

    assert recover_interrupted_executions() == 0
    assert _status(eid) == "running"


def test_heartbeat_reader_credits_only_this_rows_run(monkeypatch):
    """A later fire's heartbeat must not rescue an OLDER wedged row.

    The real reader is exercised (not stubbed): the job's ``fire_claim`` stamp is compared against
    the row's own ``claimed_at``, so a stamp that predates the row cannot vouch for it.
    """
    from cron.jobs import update_job

    monkeypatch.setattr(
        executions_mod, "_live_owner_heartbeat_is_fresh", _REAL_HEARTBEAT_READER
    )
    jid = _job(monkeypatch, schedule="every 5m")
    now = _hermes_now()
    stale_row_claimed = (now - timedelta(hours=6)).isoformat()

    # A heartbeat stamped BEFORE the row was claimed (an earlier run's): must not be credited.
    update_job(jid, {"fire_claim": {"at": (now - timedelta(hours=8)).isoformat(), "by": "o"}})
    assert executions_mod._live_owner_heartbeat_is_fresh(jid, stale_row_claimed) is False

    # A heartbeat stamped AFTER the claim and fresh NOW: credited for this row.
    update_job(jid, {"fire_claim": {"at": now.isoformat(), "by": "o"}})
    assert executions_mod._live_owner_heartbeat_is_fresh(jid, stale_row_claimed) is True

    # The same fresh stamp is NOT credited to a row claimed later than it (a different run).
    later_row_claimed = (now + timedelta(minutes=5)).isoformat()
    assert executions_mod._live_owner_heartbeat_is_fresh(jid, later_row_claimed) is False


def test_absent_or_unreadable_heartbeat_falls_back_to_the_bound(monkeypatch):
    """No job record / no claim / unparseable stamp is NOT evidence of death.

    The reader must return False (fall through to the bound), never True — a missing heartbeat
    cannot be allowed to make every row look alive, nor to skip the #115692 reclaim.
    """
    from cron.jobs import update_job

    monkeypatch.setattr(
        executions_mod, "_live_owner_heartbeat_is_fresh", _REAL_HEARTBEAT_READER
    )
    # No such job at all.
    assert executions_mod._live_owner_heartbeat_is_fresh("no-such-job", _hermes_now().isoformat()) is False

    jid = _job(monkeypatch, schedule="every 5m")
    claimed = _hermes_now().isoformat()
    assert executions_mod._live_owner_heartbeat_is_fresh(jid, claimed) is False  # no fire_claim

    update_job(jid, {"fire_claim": {"at": "not-a-timestamp", "by": "o"}})
    assert executions_mod._live_owner_heartbeat_is_fresh(jid, claimed) is False

    update_job(jid, {"fire_claim": {"at": claimed, "by": "o"}})
    assert executions_mod._live_owner_heartbeat_is_fresh(jid, "not-a-timestamp") is False


# ── Fail-closed and the pre-existing reclaim contract ──────────────────────────────────────────

@pytest.mark.parametrize("timeout", ["0", "nan", "inf"])
def test_unlimited_inactivity_timeout_never_reclaims_live_owners(monkeypatch, timeout):
    """With no bound to derive from, live owners are left alone (fail closed)."""
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", timeout)
    jid = _job(monkeypatch, schedule="every 5m")
    ancient = _seed_running(jid, age_seconds=30 * 24 * 3600, pid=77777)

    assert recover_interrupted_executions() == 0
    assert _status(ancient) == "running"
    with _transaction() as conn:
        assert executions_mod._live_owner_stale_after_seconds(conn, jid) is None


def test_dead_owner_reason_is_unchanged_by_the_new_bound(monkeypatch):
    """The provably-dead path keeps its own reason: the fix must not blur the two classes."""
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "600")
    monkeypatch.setattr(executions_mod, "_owner_is_live", lambda pid, started_at: False)
    jid = _job(monkeypatch, schedule="every 120m")
    eid = _seed_running(jid, age_seconds=60, pid=55555)

    assert recover_interrupted_executions() == 1
    assert _status(eid) == "unknown"
    assert "Scheduler restarted" in _error(eid)


def test_progress_stamps_measure_silence_not_run_length(monkeypatch):
    """A live-owned claim past the bound is spared while its own ``progress_at`` is fresh.

    The derived bound answers "how long may a claim go unanswered", so the age it compares
    must be SILENCE, not run length: the run monitor keeps stamping ``progress_at`` while the
    agent is active, and only a run that stopped stamping is wedged (#115692). Without this
    arm a healthy multi-hour run is reclaimed the moment its claim crosses the bound.
    """
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "600")
    jid = _job(monkeypatch, schedule="every 120m")
    with _transaction() as conn:
        bound = executions_mod._live_owner_stale_after_seconds(conn, jid)
    assert bound is not None

    busy = _seed_running(jid, age_seconds=bound + 60, pid=66666)
    with _transaction() as conn:
        conn.execute(
            "UPDATE executions SET progress_at=? WHERE id=?",
            ((_hermes_now() - timedelta(seconds=120)).isoformat(), busy),
        )
    assert recover_interrupted_executions() == 0
    assert _status(busy) == "running"
    # Once the stamps stop for longer than the bound, the same row is reclaimed.
    with _transaction() as conn:
        conn.execute(
            "UPDATE executions SET progress_at=? WHERE id=?",
            ((_hermes_now() - timedelta(seconds=bound + 1)).isoformat(), busy),
        )
    assert recover_interrupted_executions() == 1
    assert _status(busy) == "unknown"


def test_reclaim_reason_is_its_own_class_not_a_bare_unknown(monkeypatch):
    """The loss is VISIBLE: the reclaim text is distinct and classifies on its own.

    A discarded live run used to read identically to a crash (a bare ``unknown``). The ledger
    reason now carries its own phrases, and both classifier ladders must map them to a distinct
    class rather than ``unknown``.
    """
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "600")
    jid = _job(monkeypatch, schedule="every 5m")
    with _transaction() as conn:
        bound = executions_mod._live_owner_stale_after_seconds(conn, jid)
    eid = _seed_running(jid, age_seconds=bound + 60, pid=44444)
    assert recover_interrupted_executions() == 1

    reason = _error(eid)
    assert executions_mod._OWNER_WEDGED_REASON == reason

    from cron.incidents import _classify_failure_type

    assert _classify_failure_type(reason) == "stale_live_claim"

    from agent.monitoring.cron_health import classify_cron_error

    assert classify_cron_error(reason) == "stale_live_claim"
    # …and a genuine crash still reads as itself, not as the reclaim class.
    assert classify_cron_error("Scheduler restarted after this execution's owner exited") != "stale_live_claim"
