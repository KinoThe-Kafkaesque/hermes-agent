"""The per-job fire fence: cross-process exclusivity for one job's claim, terminal write and
external side effects.

The fence is justified by the fire claim it serves: ``claim_job_for_fire`` stamps ``fire_claim``
while holding it, ``mark_job_run`` writes the terminal status under it, and ``fire_claim_fence``
serializes a delivery against terminal revocation. A holder that keeps the flock while no live
claim justifies it protects no live owner, so waiting on it only costs the occurrence: the waiter
times out, the claim is refused, and the occurrence is resolved as a miss
(``cron/claim_refusal.py``).

Measured shape (job ``f97cff5541c4``, 2026-10-10): the restart-safe worker for the 08:11 run held
``.fire-690ef05a….lock`` across a 600 s bot-chat delivery, and the tick that restored the 10:11
occurrence waited its full 30 s and failed closed — ``Timed out waiting for fire fence …; failing
closed`` at 10:14:34.970 and 10:15:34.764 (``logs/errors.log.1:4593``/``:4595``). Nothing bounded
how long a holder could keep that flock.

The bound is the claim lease the fence serves, not a wall-clock cap on the hold. On the CLAIM path
(``_under_fire_fence``: the store CAS in ``claim_job_for_fire``, the terminal write in
``mark_job_run``) a CROSS-PROCESS wait that outlives its budget while no LIVE ``fire_claim``
justifies the fence is recorded and abandoned. That is sound because at that moment the fence is
unowned and the path's real authority is elsewhere: the CAS runs inside ``_with_job`` under the
``.jobs.lock`` flock, and ``mark_job_run`` is owner-fenced, so a stale owner still cannot overwrite
a replacement run. A live claim keeps the fence authoritative, so at-most-once is untouched.

Two narrower bounds were rejected. A wall-clock ceiling enforced on the HOLDER needs a watchdog per
hold, and releasing the flock mid-side-effect destroys the exclusivity the fence exists for; the
lease is a signal the system already maintains (``heartbeat_fire_claim`` every 60 s,
``FIRE_CLAIM_TTL_SECONDS`` 300 s) and it cannot preempt a legitimate long delivery. Bounding the
IN-PROCESS lock is unsound for the same reason in miniature: its waiter is a sibling THREAD of the
holder, so bypassing it would let a terminal write race the delivery the fence serializes — and no
measured wedge came from there (a same-process holder is reentrant, so it cannot time itself out).

SIDE-EFFECT holds (``fire_claim_fence``) always fail closed: a delivery that ran without
exclusivity could land after the terminal write it must linearize against.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import uuid
from typing import Any, Callable

logger = logging.getLogger(__name__)

# In-process lock per fence key. ``jobs.remove_job`` prunes this registry, so it is shared with the
# job store rather than private here.
_fire_fence_locks: dict[str, threading.RLock] = {}
_fire_fence_locks_guard = threading.Lock()
_fire_fence_lock_state = threading.local()

_FENCE_OVERRUN_COUNT: int = 0
_FENCE_OVERRUN_RECENT: list = []


def _jobs_lock_timeout() -> float:
    """The facade's wait budget, read at CALL time so a monkeypatched value still governs."""
    from cron.jobs import _JOBS_LOCK_TIMEOUT_SECONDS

    return float(_JOBS_LOCK_TIMEOUT_SECONDS)


def _current_cron_dir():
    from cron.jobs import _current_cron_store

    return _current_cron_store().cron_dir


def fence_lock_key(job_id: Any) -> str:
    """Registry key for one job's fence: the store it belongs to plus the job id."""
    return f"{_current_cron_dir().resolve()}::{job_id}"


def fence_lock_path(job_id: Any) -> Any:
    """The cross-process flock path for one job's fence (``<cron dir>/.fire-<uuid5>.lock``)."""
    digest = uuid.uuid5(uuid.NAMESPACE_URL, fence_lock_key(job_id)).hex
    return _current_cron_dir() / f".fire-{digest}.lock"


def fence_is_unowned(job_id: Any) -> bool:
    """True when no LIVE fire claim justifies this job's fence, so waiting on it costs a slot.

    Malformed, future-dated and over-lease claims are stale by ``_claim_is_live``, so none of them
    holds the fence legitimately, and a job carrying no claim at all holds it for nothing. An
    unreadable store is NOT unowned: an unknown state must keep the fence authoritative (fail
    closed), because the alternative is proceeding against a live owner.
    """
    from cron.constants import FIRE_CLAIM_TTL_SECONDS
    from cron.jobs import _claim_is_live, _hermes_now, get_job

    try:
        job = get_job(str(job_id))
        if not isinstance(job, dict):
            return True
        return not _claim_is_live(job.get("fire_claim"), _hermes_now(), FIRE_CLAIM_TTL_SECONDS)
    except Exception:
        logger.warning("Cannot decide fire-fence ownership for job %s", job_id, exc_info=True)
        return False


def record_fence_overrun(job_id: Any, waited_seconds: float, kind: str) -> None:
    """Make a fence wait a lapsed claim left unowned operator-visible.

    The wait is the cost this bound exists to stop paying, so log it loudly and append a
    probe-visible record: without it a wedged holder is only ever inferred from the refusal row it
    caused, which conflates it with every other contention source.
    """
    global _FENCE_OVERRUN_COUNT
    from cron.jobs import _append_telemetry_record, _hermes_now

    _FENCE_OVERRUN_COUNT += 1
    _append_telemetry_record(
        "fire_fence_overruns.jsonl",
        {
            "job_id": str(job_id),
            "fence": kind,
            "waited_seconds": round(float(waited_seconds), 3),
            "at": _hermes_now().isoformat(),
        },
        _FENCE_OVERRUN_RECENT,
    )
    logger.error(
        "Fire fence (%s) for job %s was held %.1fs with no live fire claim; proceeding so the "
        "occurrence is not lost — the holder is wedged, not working",
        kind, job_id, waited_seconds)


def get_fire_fence_overrun_stats() -> dict[str, Any]:
    """Probe-visible snapshot of fence overruns (lapsed-claim waits that had to be abandoned)."""
    return {
        "fire_fence_overruns": _FENCE_OVERRUN_COUNT,
        "recent": list(_FENCE_OVERRUN_RECENT),
    }


@contextlib.contextmanager
def _fire_job_lock(job_id: str, *, claim_path: bool = False):
    """Serialize one job's owner mutations and external side effects. Unlike the global jobs lock
    this may be held across network delivery; scoped to one profile + job so unrelated jobs keep
    progressing. Fails closed when cross-process locking is unavailable, on the in-process lock, and
    whenever a LIVE fire claim justifies the cross-process fence.

    ``claim_path``: this hold is the job's own claim / terminal write, whose authority is the claim
    CAS rather than the flock, so it abandons a CROSS-PROCESS fence no live claim justifies (see the
    module docstring). A side-effect hold never does.
    """
    from cron.jobs import _acquire_flock, _ensure_cron_dir, _release_flock

    timeout = _jobs_lock_timeout()
    lock_key = fence_lock_key(job_id)
    with _fire_fence_locks_guard:
        local_lock = _fire_fence_locks.setdefault(lock_key, threading.RLock())

    if not local_lock.acquire(timeout=timeout):
        logger.error("Timed out waiting for local fire fence %s; failing closed", lock_key)
        yield False
        return

    try:
        held_locks = _fire_fence_lock_state.__dict__.setdefault("held", {})
        if lock_key in held_locks:
            # Nested same-thread caller: reuse the held fence. The single release below (the
            # outer try's finally) balances this call's own local_lock.acquire.
            yield held_locks[lock_key]
            return

        lock_fd = None
        acquired = False
        bypassed = False
        _ensure_cron_dir(_current_cron_dir())
        lock_path = fence_lock_path(job_id)
        try:
            lock_fd = open(lock_path, "a+", encoding="utf-8")
            lock_fd.seek(0)
            result = _acquire_flock(lock_fd, timeout)
            if result is None:  # pragma: no cover - supported platforms provide one backend
                logger.error("No cross-process lock backend for cron fire fence")
            elif not result:
                # The holder outlived the wait budget. A live claim keeps the fence
                # authoritative (a delivery in flight must not be revoked); a lapsed one means
                # the fence protects no live owner, and waiting it out only costs the
                # occurrence.
                bypassed = claim_path and fence_is_unowned(job_id)
                if bypassed:
                    record_fence_overrun(job_id, timeout, "cross-process")
                else:
                    logger.error(
                        "Timed out waiting for fire fence %s; failing closed", lock_path)
            acquired = bool(result) or bypassed
        except OSError as exc:
            logger.error("Cron fire fence unavailable for %s: %s", job_id, exc)

        held_locks[lock_key] = acquired
        try:
            yield acquired
        finally:
            held_locks.pop(lock_key, None)
            if lock_fd is not None:
                if acquired and not bypassed:
                    _release_flock(lock_fd)
                else:
                    lock_fd.close()
    finally:
        local_lock.release()


def _under_fire_fence(job_id: str, fn: Callable[[], Any]) -> Any:
    """Run ``fn()`` holding the job's fire fence; False (fail closed) when it can't be acquired.

    This is the claim path (the store CAS and the terminal write), so a cross-process fence whose
    claim lease has lapsed is abandoned rather than waited out.
    """
    with _fire_job_lock(job_id, claim_path=True) as acquired:
        if not acquired:
            return False
        return fn()


@contextlib.contextmanager
def fire_claim_fence(job_id: str, *, expected_owner: str):
    """Hold a per-job fence while an owner performs an external side effect. A missing record
    is accepted only for the active run that removed this exact job (#111039).

    Side-effect holds always fail closed: the bound above is scoped to the claim path, because a
    delivery that ran without exclusivity could land after the terminal write it must linearize
    against.
    """
    from cron.jobs import _jobs_lock, load_jobs, self_removal_delivery_allowed

    with _fire_job_lock(job_id) as acquired:
        if not acquired:
            yield False
            return
        with _jobs_lock():
            job = next((item for item in load_jobs() if item.get("id") == job_id), None)
            claim = job.get("fire_claim") if isinstance(job, dict) else None
            owns_claim = isinstance(claim, dict) and claim.get("by") == expected_owner
            if job is None:
                owns_claim = self_removal_delivery_allowed(job_id)
        yield owns_claim
