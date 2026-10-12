"""Fire-claim ownership for the due scan, and repair for a refused claim.

The ``Fire claim lost; execution was not started.`` class is written by exactly one site —
``cron/scheduler.py::_process_due_job`` — when the tick's ``claim_job_for_fire`` returns False.
Measured on the live store (job ``f97cff5541c4``, 2026-10-11): 142 rows, and the 141-row burst is
ONE scheduled instant (``2026-10-07T07:43:52.182565Z``) written once per 60s tick from a single pid
(07:44:52Z → 10:04:52Z), plus one recurrence on ``2026-10-10T10:15:04Z``.

Two things have to be true for one refused claim to become 141 rows. Both are fixed here.

1. The tick advanced ``next_run_at`` and stamped ``pending_slot`` BEFORE dispatch (at-most-once
   across a crash), then the claim was refused. A refusal resolves nothing, so the next scan
   restored the SAME instant from the stamp and was refused again — one failed row per tick,
   forever. ``resolve_refused_claim`` hands the occurrence back to the schedule (the stamp is
   dropped, ``next_run_at`` returns to the attempted instant) so the ordinary late /
   fast-forward / ``cron.catch_up_missed`` policy accounts for it instead of the stamp
   re-fabricating it every tick.

2. A LIVE fire claim never reached the due scan's own guard. The one-shot path already refused to
   re-dispatch over a live ``run_claim``; a recurring job owned by a live ``fire_claim`` was still
   dispatched, so its only possible outcome was the refusal above. ``another_fire_owns`` closes
   that symmetry. The 10-07 burst is this shape: the wedged owner's ledger row was reclaimed as
   ``unknown`` at 07:44:52.147747 (its live-owner bound had passed) and the first refusal landed
   26ms later at 07:44:52.173921 — the reclaim releases the LEDGER row, not the job's
   ``fire_claim``, and because the wedged owner process stays alive ``_claim_is_live`` keeps
   honouring that claim past its TTL, so every later tick was refused.

The 10-10 recurrence is the second contention source reaching the same site: two ``Timed out
waiting for fire fence`` ERRORs (10:14:34, 10:15:34) bracket the restored occurrence (10:15:04)
and the refusal row (finished 10:15:34.764) — no live claim was involved. The parenthesised suffix
``resolve_refused_claim`` writes is what keeps those two apart in one error class.
"""

from __future__ import annotations

from typing import Any, Dict

REFUSED_CLAIM_ERROR = "Fire claim lost; execution was not started."

_RECURRING_KINDS = ("cron", "interval")


def another_fire_owns(job: Dict[str, Any], now: Any, run_claim_ttl: float) -> bool:
    """True when a live claim says another fire owns ``job`` right now, so the scan must not
    dispatch it.

    One-shots keep the caller's ``run_claim`` TTL (unchanged behaviour). Recurring jobs get the
    ``fire_claim`` lease — the same check ``claim_job_for_fire`` itself applies — so the due scan
    and the claim CAS agree on ownership instead of the scan handing a worker a job whose only
    possible outcome is a refusal. Malformed or future-dated claims count as stale (clock/TZ
    skew), never eternally fresh.
    """
    from cron.constants import FIRE_CLAIM_TTL_SECONDS
    from cron.jobs import _claim_is_live

    kind = (job.get("schedule") or {}).get("kind")
    if kind == "once":
        return _claim_is_live(job.get("run_claim"), now, run_claim_ttl)
    if kind in _RECURRING_KINDS:
        return _claim_is_live(job.get("fire_claim"), now, FIRE_CLAIM_TTL_SECONDS)
    return False


def _refusal_reason(job: Dict[str, Any]) -> str:
    """Which contention source refused the claim, read from the store the refusal left behind."""
    from cron.constants import FIRE_CLAIM_TTL_SECONDS
    from cron.jobs import _claim_is_live, _hermes_now, is_job_runnable, is_terminal_job

    if _claim_is_live(job.get("fire_claim"), _hermes_now(), FIRE_CLAIM_TTL_SECONDS):
        return "another fire owns this occurrence"
    if is_terminal_job(job):
        return "the job is in a terminal state"
    if not is_job_runnable(job):
        return "the job is paused or disabled"
    return "the fire fence was contended"


def resolve_refused_occurrence(job_id: str) -> bool:
    """Account for the occurrence a refused claim was handed, so it is not re-fabricated.

    The tick advances ``next_run_at`` and stamps ``pending_slot`` BEFORE dispatch so a PROCESS
    DEATH in that window restores the occurrence (``_restore_unclaimed_slot``). A refusal is not a
    death: the tick is alive and knows the outcome, and leaving the stamp in place made the next
    scan restore the SAME instant and refuse it again — one failed row per tick for as long as the
    contention lasted (141 rows for one scheduled instant on 2026-10-07). The stamp's job is done,
    so it is dropped and the occurrence is recorded as a miss, which is how this scheduler accounts
    for every occurrence it did not run (``cron.catch_up_missed`` does the same past its grace).
    The advance itself stands — retrying here would defeat the at-most-once semantics the advance
    exists for; the next natural occurrence is the retry.

    Declines when a live claim appeared since the refusal (that owner resolves the occurrence) or
    when the job is not recurring. Returns True when the store changed.
    """
    from cron.constants import FIRE_CLAIM_TTL_SECONDS
    from cron.jobs import _claim_is_live, _hermes_now, _with_job, save_jobs

    def apply(jobs, _i, job):
        if (job.get("schedule") or {}).get("kind") not in _RECURRING_KINDS:
            return False
        if "pending_slot" not in job:
            return False  # already resolved (or never stamped) — nothing to account for
        if _claim_is_live(job.get("fire_claim"), _hermes_now(), FIRE_CLAIM_TTL_SECONDS):
            return False
        job.pop("pending_slot", None)
        from hermes_cli.observability.shared_metrics_gateway import record_cron_missed

        record_cron_missed(job)
        save_jobs(jobs)
        return True

    return bool(_with_job(job_id, apply, False))


def resolve_refused_claim(job: Dict[str, Any]) -> str:
    """Repair a refused occurrence and return the honest, still-greppable error text.

    The ``Fire claim lost; execution was not started.`` prefix is preserved — the executions
    ledger and the incident digest key the class on it — and the parenthesised suffix names the
    contention source, so a live owner is no longer indistinguishable from a contended fence.
    """
    reason = _refusal_reason(job)
    resolve_refused_occurrence(str(job.get("id") or ""))
    return f"{REFUSED_CLAIM_ERROR} ({reason})"
