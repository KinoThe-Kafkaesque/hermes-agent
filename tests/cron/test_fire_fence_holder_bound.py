"""A wedged fire-fence holder must not cost an unbounded number of refused occurrences.

Regression for the 10-10 recurrence of job ``f97cff5541c4`` (``cron/fire_fence.py`` carries the
measured shape). The 10-07 burst and the 10-10 recurrence are DIFFERENT mechanisms and only the
second is this card's subject: 10-07 was a live foreign ``fire_claim`` (closed by
``cron/claim_refusal.py``), 10-10 was a CONTENDED FENCE — two ``Timed out waiting for fire fence``
ERRORs bracketing the restored occurrence, with no live claim involved.

Before this bound, the claim path waited out a flock held by a process with no live claim, failed
closed, and the occurrence was refused. The two tests here are the two directions of that bound:

* ``TestWedgedHolderIsBounded`` — a holder that keeps the flock with NO live claim is abandoned
  after the wait budget, so the claim succeeds and the occurrence is NOT refused.
* ``TestLiveOwnerStillHoldsTheFence`` — a holder WITH a live claim keeps the fence authoritative, so
  the claim is still refused. Without this leg the bound would be indistinguishable from "delete
  the fence".
"""

from __future__ import annotations

import os
import socket
import threading
import time
from datetime import timedelta

import pytest

from cron import executions, fire_fence, jobs
from cron import scheduler as sched

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


class _ForeignHolder:
    """Holds a job's cross-process fire fence from a SIBLING THREAD.

    A sibling thread is enough to contend the flock: ``fcntl.flock`` is per open file description,
    so the holder's own fd conflicts with the claimant's even inside one process. That mirrors the
    measured shape (a wedged peer holding ``.fire-<uuid5>.lock``) without needing a second
    interpreter.
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.path = fire_fence.fence_lock_path(job_id)
        self._fd = None
        self._held = threading.Event()
        self._release = threading.Event()
        self._thread = threading.Thread(target=self._hold, daemon=True)

    def _hold(self) -> None:
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        self._held.set()
        self._release.wait(timeout=30)
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)

    def __enter__(self):
        self._thread.start()
        assert self._held.wait(timeout=5), "holder never took the fence"
        return self

    def __exit__(self, *exc):
        self._release.set()
        self._thread.join(timeout=5)


class TestWedgedHolderIsBounded:
    def test_a_holder_with_no_live_claim_costs_at_most_one_wait_not_the_occurrence(
            self, cron_home, monkeypatch):
        """A wedged holder is abandoned after the budget, so the occurrence is not refused.

        The wait budget is 0.2 s here so the test is fast; the production value is 30 s
        (``_JOBS_LOCK_TIMEOUT_SECONDS``). The claim path (``_under_fire_fence``) is the path under
        test — it is the one the tick takes through ``claim_job_for_fire``.
        """
        monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.2)
        job = jobs.create_job(prompt="x", schedule="every 2h", name="wedged", deliver="local")
        job_id = job["id"]
        _seed_restorable_slot(job_id, minutes_ago=4, live_claim=False)

        with _ForeignHolder(job_id):
            started = time.monotonic()
            claimed = jobs.claim_job_for_fire(job_id)
            waited = time.monotonic() - started

        assert claimed is True, (
            "a flock held with no live fire claim refused the claim — the occurrence is lost")
        assert waited >= 0.2, "the bound must be the wait budget, not an instant bypass"

        stats = fire_fence.get_fire_fence_overrun_stats()
        assert stats["fire_fence_overruns"] >= 1, (
            "the abandoned wait must be recorded, or a wedged holder stays invisible")
        assert any(r["job_id"] == job_id for r in stats["recent"]), stats["recent"]

    def test_the_bound_does_not_reintroduce_a_refusal_row_storm(self, cron_home, monkeypatch):
        """The occurrence is claimed once; later ticks do not re-refuse it (no rewind, no storm)."""
        monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.2)
        job = jobs.create_job(prompt="x", schedule="every 2h", name="no-storm", deliver="local")
        job_id = job["id"]
        slot = _seed_restorable_slot(job_id, minutes_ago=4, live_claim=False)

        # A claim held by the store so the tick cannot win it, but with NO live claim behind it —
        # the contended-fence shape the 10-10 recurrence arrived by.
        with _ForeignHolder(job_id):
            sched.tick(verbose=False, sync=True)
            for _ in range(3):
                sched.tick(verbose=False, sync=True)

        rows = _refused_rows(job_id)
        assert len(rows) <= 1, f"more than one refusal for one occurrence: {len(rows)}"
        record = jobs.get_job(job_id)
        assert record["next_run_at"] != slot, "the at-most-once advance must stand"


class TestLiveOwnerStillHoldsTheFence:
    def test_a_holder_with_a_live_claim_still_refuses_the_claim(self, cron_home, monkeypatch):
        """The other direction: a live claim keeps the fence authoritative, so the bound cannot
        silently degrade into 'the fence never blocks anything'."""
        monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.2)
        job = jobs.create_job(prompt="x", schedule="every 2h", name="live-owner", deliver="local")
        job_id = job["id"]
        _seed_restorable_slot(job_id, minutes_ago=4, live_claim=True)

        before = fire_fence.get_fire_fence_overrun_stats()["fire_fence_overruns"]
        with _ForeignHolder(job_id):
            assert jobs.claim_job_for_fire(job_id) is False, (
                "a live fire claim must still keep the fence authoritative")

        # The counter is process-global (other tests in this file increment it), so the claim is
        # about the DELTA: a live owner's hold must not be recorded as an overrun.
        assert fire_fence.get_fire_fence_overrun_stats()["fire_fence_overruns"] == before, (
            "a live owner's hold is not an overrun")

    def test_a_side_effect_hold_still_fails_closed_on_a_lapsed_claim(
            self, cron_home, monkeypatch):
        """Side-effect holds are NOT bounded: a delivery must never run without exclusivity.

        The stored claim is LAPSED (past its lease) so this test discriminates: a lapsed claim is
        exactly what the claim-path bound abandons on, so if the side-effect path were ever wired
        to that bound this hold would bypass and report ownership. It must not — a delivery that
        ran without exclusivity could land after the terminal write it linearizes against.
        """
        monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.2)
        job = jobs.create_job(prompt="x", schedule="every 2h", name="side-effect", deliver="local")
        job_id = job["id"]
        assert jobs.claim_job_for_fire(job_id) is True
        owner = jobs.get_job(job_id)["fire_claim"]["by"]
        # Age the claim past FIRE_CLAIM_TTL_SECONDS (300 s) so it is no longer live.
        from cron.constants import FIRE_CLAIM_TTL_SECONDS

        stored = jobs.load_jobs()
        for row in stored:
            if row["id"] == job_id:
                stale_at = jobs._hermes_now() - timedelta(seconds=FIRE_CLAIM_TTL_SECONDS + 60)
                row["fire_claim"] = {"at": stale_at.isoformat(), "by": owner}
        jobs.save_jobs(stored)
        assert fire_fence.fence_is_unowned(job_id) is True, "precondition: the claim is lapsed"

        before = fire_fence.get_fire_fence_overrun_stats()["fire_fence_overruns"]
        with _ForeignHolder(job_id):
            with jobs.fire_claim_fence(job_id, expected_owner=owner) as owns:
                assert owns is False, "a contended side-effect fence must fail closed"

        assert fire_fence.get_fire_fence_overrun_stats()["fire_fence_overruns"] == before
