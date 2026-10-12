"""A refused fire claim must RESOLVE its occurrence, and the scan must not dispatch an owned job.

Regression for the dominant cron failure class (job ``f97cff5541c4``, measured 2026-10-11):
142 rows reading ``Fire claim lost; execution was not started.``, of which 141 are a SINGLE
scheduled instant (``2026-10-07T07:43:52.182565Z``) written once per 60s tick from one pid —
07:44:52Z through 10:04:52Z — plus one recurrence on 2026-10-10T10:15:04Z.

Two independent holes let one refusal become 141 rows, and each has its own test here:

* ``TestOwnedJobIsNotDispatched`` — the due scan dispatched a recurring job whose ``fire_claim``
  was live. Its only possible outcome was a refusal, so every tick wrote one. (Before the fix the
  one-shot ``run_claim`` was guarded and the recurring ``fire_claim`` was not.)
* ``TestRefusedClaimResolvesOccurrence`` — a refused claim left the ``pending_slot`` stamp in
  place, so the next scan restored the SAME instant and refused it again, forever.
"""

from __future__ import annotations

import os
import socket
from datetime import timedelta

import pytest

import cron.executions as executions
import cron.jobs as jobs
import cron.scheduler as sched

_REFUSED_PREFIX = "Fire claim lost; execution was not started."


@pytest.fixture
def cron_home(tmp_path, monkeypatch):
    """Throwaway HERMES_HOME wired into the job store, the ledger and the scheduler."""
    home = tmp_path / ".hermes"
    (home / "cron" / "output").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MACHINE_ID", raising=False)

    monkeypatch.setattr(jobs, "HERMES_DIR", home)
    monkeypatch.setattr(jobs, "CRON_DIR", home / "cron")
    monkeypatch.setattr(jobs, "JOBS_FILE", home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", home / "cron" / "output")
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", home / "cron" / "executions.db")
    monkeypatch.setattr(sched, "_hermes_home", home)
    sched._running_job_ids.clear()
    sched._running_since.clear()
    sched._running_futures.clear()
    yield home
    sched._running_job_ids.clear()
    sched._running_since.clear()
    sched._running_futures.clear()
    sched._shutdown_parallel_pool()


def _live_claim() -> dict:
    """A claim owned by THIS host and a live pid, so ``_claim_is_live`` honours it."""
    return {"at": jobs._hermes_now().isoformat(), "by": f"{socket.gethostname()}:{os.getpid()}"}


def _seed_recurring(name: str = "cycle") -> str:
    job = jobs.create_job(prompt="x", schedule="every 2h", name=name, deliver="local")
    return job["id"]


def _seed_restorable_slot(job_id: str, minutes_ago: int, *, live_claim: bool) -> str:
    """Leave the store exactly as a tick that advanced-then-refused leaves it."""
    now = jobs._hermes_now()
    slot = (now - timedelta(minutes=minutes_ago)).replace(microsecond=0).isoformat()
    stored = jobs.load_jobs()
    for row in stored:
        if row["id"] == job_id:
            row["next_run_at"] = (now + timedelta(hours=2)).isoformat()
            row["pending_slot"] = {"scheduled_at": slot, "at": now.isoformat(), "by": "pop-os:999999"}
            row["fire_claim"] = _live_claim() if live_claim else None
    jobs.save_jobs(stored)
    return slot


def _refused_rows(job_id: str) -> list:
    return [r for r in executions.list_executions(job_id=job_id, limit=80)
            if (r.get("error") or "").startswith(_REFUSED_PREFIX)]


class TestOwnedJobIsNotDispatched:
    def test_live_fire_claim_keeps_the_job_out_of_the_due_scan(self, cron_home):
        """A recurring job owned by a live ``fire_claim`` is not due, so no refusal is written.

        Dispatching it can only produce the refusal, and the refusal is what accumulated rows.
        """
        job_id = _seed_recurring()
        _seed_restorable_slot(job_id, minutes_ago=4, live_claim=True)

        assert jobs.get_due_jobs() == [], "a job owned by a live fire must not be handed to a worker"

        sched.tick(verbose=False, sync=True)
        assert _refused_rows(job_id) == [], (
            "a live-owned job was dispatched and refused — the per-tick row storm")
        assert "pending_slot" in jobs.get_job(job_id), (
            "the occurrence must stay with its live owner, not be resolved by this tick")

    def test_live_run_claim_still_keeps_a_oneshot_out_of_the_due_scan(self, cron_home):
        """The pre-existing one-shot guard is unchanged (no regression in the other direction)."""
        job = jobs.create_job(prompt="x", schedule="in 2h", name="oneshot", deliver="local")
        stored = jobs.load_jobs()
        for row in stored:
            if row["id"] == job["id"]:
                row["run_claim"] = _live_claim()
        jobs.save_jobs(stored)

        assert jobs.get_due_jobs() == []


class TestRefusedClaimResolvesOccurrence:
    def test_refusal_drops_the_stamp_and_does_not_refabricate_the_occurrence(self, cron_home, monkeypatch):
        """One refusal, and only one, for an occurrence the scheduler refuses to run.

        The claim CAS is stubbed to refuse (the contended-fire-fence shape measured on 10-10: two
        ``Timed out waiting for fire fence`` ERRORs bracketing the restored occurrence). Before the
        fix every later tick restored the same instant from ``pending_slot`` and refused it again.
        """
        job_id = _seed_recurring()
        slot = _seed_restorable_slot(job_id, minutes_ago=4, live_claim=False)

        monkeypatch.setattr(sched, "claim_job_for_fire", lambda *_a, **_kw: False)

        sched.tick(verbose=False, sync=True)
        rows = _refused_rows(job_id)
        assert len(rows) == 1, f"expected exactly one refusal, got {len(rows)}"
        assert rows[0]["error"].startswith(_REFUSED_PREFIX), rows[0]["error"]
        assert rows[0]["scheduled_instant"] == slot, "the refusal must name the occurrence it refused"

        record = jobs.get_job(job_id)
        assert "pending_slot" not in record, (
            "the stamp must be dropped — it is the mechanism that re-fabricated the same instant")

        # The occurrence is accounted for, not silently dropped: the miss is recorded and the
        # schedule keeps its advance (at-most-once), so the next natural occurrence is the retry.
        assert record["next_run_at"] != slot, "the at-most-once advance must stand"

        for _ in range(3):
            sched.tick(verbose=False, sync=True)
        assert len(_refused_rows(job_id)) == 1, (
            "later ticks re-refused the same occurrence — the unbounded row storm is back")

    def test_refusal_names_the_contention_source(self, cron_home, monkeypatch):
        """The one error class stops conflating a live owner with a contended fence."""
        job_id = _seed_recurring()
        _seed_restorable_slot(job_id, minutes_ago=4, live_claim=False)
        monkeypatch.setattr(sched, "claim_job_for_fire", lambda *_a, **_kw: False)

        sched.tick(verbose=False, sync=True)
        error = _refused_rows(job_id)[0]["error"]
        assert error != _REFUSED_PREFIX, "the refusal must say which source refused it"
        assert "fire fence was contended" in error, error
