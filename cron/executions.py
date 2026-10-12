"""Profile-local durable audit ledger for cron execution attempts.

The ledger records what is known about each attempt; it is not a retry queue. Interrupted attempts
become ``unknown`` only after their owner process is proved gone — a start-time reading that fails
to match the claim-time fingerprint is not proof of death. Terminal states are immutable.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now
from cron.constants import CLAIM_TTL_INACTIVITY_HEADROOM
from hermes_cli.observability.shared_metrics_gateway import record_cron_finish

logger = logging.getLogger(__name__)

# Optional test override. Production resolves the path at transaction time so dashboard operations
# that temporarily enter another profile cannot leak that profile's records into the import-time
# home.
EXECUTIONS_FILE: Optional[Path] = None
MAX_TERMINAL_EXECUTIONS = 1000
HANDOFF_ADOPTION_GRACE_SECONDS = 30.0
# Sample sizes for the live-owner bound's two measured terms (#131247); see
# _live_owner_stale_after_seconds.
LIVE_OWNER_CADENCE_SAMPLE = 20
LIVE_OWNER_RUNTIME_SAMPLE = 200
# Headroom multipliers over the two MEASURED terms of the live-owner bound: the job's own
# schedule period and the p95 wall-clock duration of its recent terminal attempts. Both exist so
# the bound cannot land on (or below) the job's own cadence — the shape that reclaimed a healthy
# 122-minute run of a 120-minute job mid-flight (#131247).
LIVE_OWNER_CADENCE_HEADROOM = 2.0
LIVE_OWNER_RUNTIME_HEADROOM = 2.0
_TERMINAL_STATES = ("completed", "failed", "unknown")
_lock = threading.RLock()
_PROCESS_ID = uuid.uuid4().hex


# --- executions ledger --------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    # Late imports: a scheduler daemon that outlives an on-disk upgrade already has the OLD
    # ``hermes_cli.sqlite_util`` / ``cron.jobs`` cached, so new names must be resolved at call time,
    # not at import time (the guarantee cron/ledger.py used to carry, see e24c8499).
    from cron.jobs import _ensure_cron_dir
    from hermes_cli.sqlite_util import open_db

    path = EXECUTIONS_FILE or (get_hermes_home().resolve() / "cron" / "executions.db")
    _ensure_cron_dir(path.parent)
    return open_db(path, db_label="cron/executions.db", synchronous_full=True, initialize=_initialize_schema)


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_cli.sqlite_util import add_column_if_missing

    conn.execute(
        """CREATE TABLE IF NOT EXISTS executions (
             id TEXT PRIMARY KEY,
             job_id TEXT NOT NULL,
             source TEXT NOT NULL,
             process_id TEXT NOT NULL,
             pid INTEGER NOT NULL,
             process_started_at INTEGER,
             status TEXT NOT NULL CHECK(status IN
               ('claimed','running','completed','failed','unknown')),
             handoff_pending INTEGER NOT NULL DEFAULT 0,
             handoff_started_at REAL,
             claimed_at TEXT NOT NULL,
             started_at TEXT,
             finished_at TEXT,
             error TEXT
           )"""
    )
    add_column_if_missing(
        conn, "executions", "handoff_pending",
        "handoff_pending INTEGER NOT NULL DEFAULT 0",
    )
    add_column_if_missing(
        conn, "executions", "handoff_started_at", "handoff_started_at REAL"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_job_claimed "
        "ON executions(job_id, claimed_at DESC, id DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_status_claimed "
        "ON executions(status, claimed_at DESC, id DESC)"
    )
    add_column_if_missing(conn, "executions", "delivery_outcome", "delivery_outcome TEXT")
    add_column_if_missing(conn, "executions", "scheduled_instant", "scheduled_instant TEXT")
    add_column_if_missing(conn, "executions", "progress_at", "progress_at TEXT")
    # Replace the retired partial ``(job_id, scheduled_instant) WHERE status='completed'``
    # on an already-created ledger. ``CREATE INDEX IF NOT EXISTS`` below is a no-op when the
    # name exists, so a name-only create would leave every existing executions.db carrying
    # the dangerous index. The stored DDL is TESTED for the dangerous shape rather than
    # matched against a quoted copy of it, so this heals any equivalent spelling and cannot
    # itself go stale.
    from hermes_state_common import _partial_where_has_literal_equality
    try:
        stored = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_executions_occurrence'"
        ).fetchone()
        if stored is not None and _partial_where_has_literal_equality(stored[0] or ""):
            conn.execute("DROP INDEX IF EXISTS idx_executions_occurrence")
    except sqlite3.OperationalError:
        pass
    # COMPOSITE, not the retired partial: a partial index whose WHERE equates a column to
    # a scalar literal makes a REUSED parameterized statement return WRONG rows once the
    # index appears under a live connection (measured; the vector is the cached statement
    # — see scripts/ci/check_partial_index_where.py). ``status`` is a production bind
    # parameter (the recovery UPDATEs, ``status IN (...)``), so the hazard is reachable
    # here. The composite is exact and serves both consumers (``completed_occurrence``:
    # job_id + scheduled_instant + status; and ``list_executions``: job_id).
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_executions_occurrence "
        "ON executions(job_id, scheduled_instant, status)"
    )


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    from hermes_cli.sqlite_util import transaction

    with _lock, transaction(_connect()) as conn:
        yield conn


def _fetch(conn: sqlite3.Connection, execution_id: str) -> Optional[dict[str, Any]]:
    row = conn.execute("SELECT * FROM executions WHERE id=?", (execution_id,)).fetchone()
    return dict(row) if row is not None else None


def _emit_execution_state(
    record: Optional[dict[str, Any]], *, delivery_outcome: Optional[str] = None
) -> None:
    """Project durable state to monitoring without affecting ledger behavior."""
    try:
        from agent.monitoring.cron_health import emit_execution_state

        emit_execution_state(record, delivery_outcome=delivery_outcome)
    except Exception:
        pass


def _process_start_time(pid: int) -> Optional[int]:
    try:
        from gateway.status import get_process_start_time
        return get_process_start_time(pid)
    except Exception:
        return None


def _owner_is_live(pid: int, started_at: Optional[int]) -> bool:
    try:
        from gateway.status import _pid_exists
        if not _pid_exists(pid):
            return False
    except Exception:
        return True  # fail safe: inability to prove death must not rewrite state
    if started_at is None:
        return pid == os.getpid()
    current = _process_start_time(pid)
    if current is None:
        return True  # cannot compare -> cannot prove death; a misread must not rewrite state
    # Drifted same-host readings (#117505) are not proof of death; a live misread is still
    # bounded by the stale-claim sweep below.
    from gateway.status import start_time_fingerprints_match
    return start_time_fingerprints_match(started_at, current)


def _live_owner_heartbeat_is_fresh(job_id: str, row_claimed_at: str) -> bool:
    """True when *job_id*'s durable ``fire_claim`` heartbeat proves a live run RIGHT NOW.

    Claim age alone cannot distinguish "wedged since the claim" from "still working", and it is
    the only signal the bound had — so a job whose period equals the bound was reclaimed
    mid-flight (#131247). The run's own heartbeat thread refreshes ``fire_claim.at`` every
    ``_RUN_CLAIM_HEARTBEAT_SECONDS`` (60 s) for as long as the run is alive, and
    ``mark_job_run`` clears the claim at terminal completion; a wedged worker stops refreshing.
    A fresh stamp is therefore positive evidence of liveness, and its ABSENCE (or an unreadable
    store) is not evidence of death: return False and let the derived bound decide.

    The stamp is per-job and per-run: it is only credited while the claim is younger than the
    row being judged, so a later fire's heartbeat can never rescue an older wedged row.
    """
    try:
        from cron.jobs import get_job
        from cron.occurrences import scheduled_instant

        job = get_job(job_id)
    except Exception:  # health: allow BLE001 -- an unreadable job store is not proof of death
        return False
    if not job:
        return False
    claim = job.get("fire_claim")
    if not isinstance(claim, dict):
        return False
    stamped = scheduled_instant(claim.get("at"))
    claimed = scheduled_instant(row_claimed_at)
    # Only credit a stamp belonging to THIS row's run: a later fire's heartbeat must never
    # rescue an older wedged row.
    if stamped is None or claimed is None or stamped < claimed:
        return False
    age = (_hermes_now() - datetime.fromisoformat(stamped)).total_seconds()
    return age < _live_owner_heartbeat_window_seconds()


def _live_owner_heartbeat_window_seconds() -> float:
    """How fresh ``fire_claim.at`` must be to count as a live heartbeat.

    Derived from the run's own refresh cadence (``_RUN_CLAIM_HEARTBEAT_SECONDS``, 60 s) times the
    same 3× headroom the claim TTLs use, so a heartbeat that slips a tick or two still counts.
    """
    try:
        from cron.scheduler import _RUN_CLAIM_HEARTBEAT_SECONDS as cadence
    except ImportError:  # scheduler is imported lazily; its absence is not a fatal condition
        cadence = 60.0
    return float(cadence) * CLAIM_TTL_INACTIVITY_HEADROOM


def _job_schedule_period_seconds(job_id: str) -> Optional[float]:
    """The job's own schedule period in seconds, or None when it is not a recurring job.

    Measured, not guessed: ``cron.jobs._schedule_cadence_seconds`` is the same reader the
    catch-up grace window uses, so the bound and the grace window cannot disagree about how long
    one period of this job is.
    """
    try:
        from cron.jobs import _schedule_cadence_seconds, get_job

        job = get_job(job_id)
        if not job:
            return None
        period = _schedule_cadence_seconds(job.get("schedule") or {})
    except Exception:  # health: allow BLE001 -- an unreadable job store has no period to derive
        return None
    if period is None or not math.isfinite(period) or period <= 0:
        return None
    return float(period)


def _job_runtime_p95_seconds(conn: sqlite3.Connection, job_id: str) -> Optional[float]:
    """p95 wall-clock duration (seconds) of this job's recent terminal attempts, or None.

    The measured runtime envelope: a healthy run of this job routinely lands here, so a bound at
    twice p95 cannot fire on normal work. ``MAX(started_at, claimed_at)`` is the start so a row
    adopted by an external worker (``claimed_at`` stamped by the gateway, ``started_at`` by the
    worker) measures the run, not the handoff.
    """
    rows = conn.execute(
        """SELECT (julianday(finished_at) - julianday(COALESCE(started_at, claimed_at))) * 86400.0 AS s
           FROM executions
           WHERE job_id=? AND status IN ('completed','failed')
             AND finished_at IS NOT NULL
           ORDER BY julianday(claimed_at) DESC LIMIT ?""",
        (str(job_id), LIVE_OWNER_RUNTIME_SAMPLE),
    ).fetchall()
    durations = sorted(
        float(r["s"]) for r in rows if r["s"] is not None and float(r["s"]) > 0
    )
    if not durations:
        return None
    # Nearest-rank p95 (no interpolation): a real observed run, never a synthesized value.
    rank = max(1, min(len(durations), math.ceil(0.95 * len(durations))))
    return durations[rank - 1]


def _live_owner_stale_after_seconds(
    conn: sqlite3.Connection, job_id: str,
) -> Optional[float]:
    """Age past which a claimed/running row with a LIVE owner is treated as wedged.

    Two MEASURED terms, never a bare wall-clock constant (#131247):

    * the job's own schedule period (``cron.jobs._schedule_cadence_seconds``, the same reader the
      catch-up grace window uses), and
    * the p95 wall-clock duration of the job's recent terminal attempts.

    plus the existing derived ceilings ``3 × HERMES_CRON_TIMEOUT`` and the cron script timeout.
    Each measured term carries 2× headroom. Because the cadence term is included, the bound is
    strictly greater than one full period of the job by construction, so a healthy run that
    legitimately exceeds its own interval — the exact shape #131247 reclaimed mid-flight — can
    never be reclaimed by age alone; and because the p95 term is included, a job whose runs
    normally take most of a period is bounded above its real envelope rather than at a fixed
    2 h.

    Returns ``None`` (never reclaim live owners) when the inactivity timeout is 0/unlimited or
    not a finite positive number: with no bound to derive from, fail closed.
    """
    from cron.scheduler import _cron_inactivity_seconds
    from cron.scheduler_script import _get_script_timeout

    inactivity = float(_cron_inactivity_seconds())
    if not math.isfinite(inactivity) or inactivity <= 0:
        return None
    terms = [
        inactivity * CLAIM_TTL_INACTIVITY_HEADROOM,
        float(_get_script_timeout()),
    ]
    period = _job_schedule_period_seconds(job_id)
    if period is not None:
        terms.append(period * LIVE_OWNER_CADENCE_HEADROOM)
    runtime_p95 = _job_runtime_p95_seconds(conn, job_id)
    if runtime_p95 is not None:
        terms.append(runtime_p95 * LIVE_OWNER_RUNTIME_HEADROOM)
    return max(terms)


def _claim_age_seconds(claimed_at: str) -> float:
    """Seconds since ``claimed_at`` (NOT NULL, always the aware ISO string from hermes_time.now)."""
    return (_hermes_now() - datetime.fromisoformat(claimed_at)).total_seconds()


def _stale_age_seconds(claimed_at: str, progress_at: Optional[str]) -> float:
    """Seconds since the owner last proved it was making progress.

    A wedged worker (#115692) stops stamping ``progress_at``; a long but healthy run keeps
    stamping it, so the derived stale bound measures silence, not run length. Rows written
    before the column existed have no stamp and fall back to claim age.
    """
    return _claim_age_seconds(progress_at or claimed_at)


def touch_execution_progress(execution_id: str) -> bool:
    """Stamp ``progress_at`` on a running attempt this process owns. Returns False when the row
    is no longer ours or no longer running (the caller treats that as informational only)."""
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions SET progress_at=?
               WHERE id=? AND status IN ('claimed','running') AND process_id=? AND pid=?""",
            (now, execution_id, _PROCESS_ID, os.getpid()),
        )
        return cur.rowcount == 1


def _prune_unlocked(conn: sqlite3.Connection) -> None:
    conn.execute(
        """DELETE FROM executions WHERE id IN (
             SELECT id FROM executions
             WHERE status IN ('completed','failed','unknown')
             ORDER BY julianday(finished_at) DESC, finished_at DESC,
                      julianday(claimed_at) DESC, claimed_at DESC, id DESC LIMIT -1 OFFSET ?
           )""",
        (max(0, int(MAX_TERMINAL_EXECUTIONS)),),
    )


def create_execution(
    job_id: str, *, source: str, scheduled_instant: Optional[str] = None,
) -> dict[str, Any]:
    """Persist a claimed attempt before executor/provider dispatch."""
    from cron.occurrences import scheduled_instant as canonical_instant

    now = _hermes_now().isoformat()
    execution_id = uuid.uuid4().hex
    pid = os.getpid()
    with _transaction() as conn:
        conn.execute(
            """INSERT INTO executions
               (id, job_id, source, process_id, pid, process_started_at,
                status, claimed_at, scheduled_instant)
               VALUES (?, ?, ?, ?, ?, ?, 'claimed', ?, ?)""",
            (execution_id, str(job_id), str(source), _PROCESS_ID, pid,
             _process_start_time(pid), now, canonical_instant(scheduled_instant)),
        )
        record = _fetch(conn, execution_id)
    _emit_execution_state(record)
    return record  # type: ignore[return-value]


def set_execution_occurrence(execution_id: str, instant: Optional[str]) -> None:
    """Bind the store-claimed snapshot before a provider hands it to a worker."""
    from cron.occurrences import scheduled_instant

    with _transaction() as conn:
        cur = conn.execute(
            "UPDATE executions SET scheduled_instant=? WHERE id=? AND status='claimed' "
            "AND handoff_pending=0 AND process_id=? AND pid=?",
            (scheduled_instant(instant), execution_id, _PROCESS_ID, os.getpid()),
        )
        if cur.rowcount != 1:
            raise RuntimeError("Cron occurrence could not be bound before dispatch")


def mark_execution_handoff_pending(execution_id: str) -> Optional[dict[str, Any]]:
    """Fence restart recovery while an external worker is adopting a claim."""
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET handoff_pending=1, handoff_started_at=?
               WHERE id=? AND status='claimed'
                 AND process_id=? AND pid=?""",
            (time.time(), execution_id, _PROCESS_ID, os.getpid()),
        )
        if cur.rowcount != 1:
            return None
        record = _fetch(conn, execution_id)
    _emit_execution_state(record)
    return record


def adopt_claimed_execution(execution_id: str) -> Optional[dict[str, Any]]:
    """Atomically transfer and start an attempt in its worker process.

    The dispatching gateway creates the row before spawning a restart-safe
    worker.  Adoption is the single ``claimed`` → ``running`` gate: only the
    winner may acknowledge ownership or run side effects.
    """
    pid = os.getpid()
    process_started_at = _process_start_time(pid)
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET process_id=?, pid=?, process_started_at=?,
                   status='running', started_at=?, handoff_pending=0,
                   handoff_started_at=NULL
               WHERE id=? AND status='claimed' AND handoff_pending=1""",
            (_PROCESS_ID, pid, process_started_at, now, execution_id),
        )
        if cur.rowcount != 1:
            return None
        record = _fetch(conn, execution_id)
    _emit_execution_state(record)
    return record


def mark_execution_running(execution_id: str) -> Optional[dict[str, Any]]:
    """Transition one claimed attempt to running exactly once."""
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET status='running', started_at=?, handoff_pending=0,
                   handoff_started_at=NULL
               WHERE id=? AND status='claimed' AND handoff_pending=0
                 AND process_id=? AND pid=?""",
            (now, execution_id, _PROCESS_ID, os.getpid()),
        )
        if cur.rowcount != 1:
            return None
        record = _fetch(conn, execution_id)
    _emit_execution_state(record)
    return record


def finish_execution(
    execution_id: str, *, success: bool, error: Optional[str] = None,
    delivery_outcome: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Write a terminal result once; terminal attempts cannot be rewritten."""
    now = _hermes_now().isoformat()
    status = "completed" if success else "failed"
    detail = None if success else (str(error) if error else "unknown failure")
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE executions
               SET status=?, finished_at=?, error=?, handoff_pending=0,
                   handoff_started_at=NULL, delivery_outcome=?
               WHERE id=? AND status IN ('claimed','running')
                 AND process_id=? AND pid=?""",
            (status, now, detail, delivery_outcome, execution_id, _PROCESS_ID, os.getpid()),
        )
        if cur.rowcount != 1:
            return None
        _prune_unlocked(conn)
        record = _fetch(conn, execution_id)
    _emit_execution_state(record, delivery_outcome=delivery_outcome)
    record_cron_finish(record, delivery_outcome)
    return record


_OWNER_GONE_REASON = (
    "Scheduler restarted after this execution's owner exited before a durable "
    "terminal state; whether side effects ran is unknown."
)
# A live owner whose claim outlived the bound for THIS job, while its own heartbeat had stopped.
# Distinct from the crash class above and from the stale-claim class below: this one names a
# genuinely stopped run, so the operator can tell "wedged worker" from "the scheduler was
# restarted" (a report-discarding reclaim that used to be filed as a bare ``unknown``, #131247).
_OWNER_WEDGED_REASON = (
    "Owner process is still alive but the claim outlived the derived stale bound and its "
    "heartbeat stopped; treated as wedged (#115692). The process was not terminated; whether "
    "side effects ran is unknown."
)


def settle_unstarted_execution(execution_id: str, job_id: str, error: str) -> None:
    """Close the receipt of a run that never started: a ``claimed`` row never resolves. Best-effort
    so a ledger write cannot mask the failure the caller logs or re-raises."""
    try:
        finish_execution(execution_id, success=False, error=error)
    except (sqlite3.Error, OSError) as record_err:
        logger.error("Job '%s': failed to close execution receipt %s (%s): %s",
                     job_id, execution_id, error, record_err)


def _live_owner_reclaim_plan(
    rows: list[sqlite3.Row],
) -> dict[tuple[str, str], str]:
    """Decide, per live-owned open row, which reclaim reason (if any) applies.

    Deliberately runs OUTSIDE the ledger transaction: the bound reads the job store
    (``cron.jobs.get_job`` → ``_jobs_lock``), and ``mark_job_run`` already calls into this module
    while holding that lock, so a job-store read inside ``_transaction()`` would invert the lock
    order and deadlock. Returns ``{(job_id, claimed_at): reason}``; absent means "leave owned".
    """
    plan: dict[tuple[str, str], str] = {}
    bounds: dict[str, Optional[float]] = {}
    for row in rows:
        if row["process_id"] == _PROCESS_ID:
            continue
        if not _owner_is_live(int(row["pid"]), row["process_started_at"]):
            plan[(str(row["job_id"]), row["claimed_at"])] = _OWNER_GONE_REASON
            continue
        # A live owner is normally a legitimately running job. A worker permanently deadlocked
        # (e.g. futex_wait behind a route/proxy flip, #115692) also passes this check — but so
        # does a healthy long run, which is why age alone is not enough (#131247): the run's own
        # fire_claim heartbeat is consulted first, and a fresh stamp keeps the row owned however
        # old the claim is.
        if _live_owner_heartbeat_is_fresh(str(row["job_id"]), row["claimed_at"]):
            continue
        job_id = str(row["job_id"])
        if job_id not in bounds:
            with _transaction() as conn:
                bounds[job_id] = _live_owner_stale_after_seconds(conn, job_id)
        bound = bounds[job_id]
        if bound is None or _stale_age_seconds(row["claimed_at"], row["progress_at"]) <= bound:
            continue
        plan[(job_id, row["claimed_at"])] = _OWNER_WEDGED_REASON
    return plan


def recover_interrupted_executions() -> int:
    """Mark abandoned attempts unknown without scheduling retries: rows whose owner is provably
    dead, plus rows whose live owner holds a claim older than the bound derived from that job's
    own cadence and runtime envelope (the process is not killed). A live owner whose own
    heartbeat is still fresh is never reclaimed, whatever its claim age (#131247)."""
    now = _hermes_now().isoformat()
    changed = 0
    recovered: list[dict[str, Any]] = []
    with _transaction() as conn:
        rows = conn.execute(
            """SELECT id, job_id, status, process_id, pid, process_started_at,
                      handoff_pending, handoff_started_at, claimed_at, progress_at
               FROM executions
               WHERE status IN ('claimed','running')"""
        ).fetchall()
    if not rows:
        return 0
    # Job-store reads (heartbeat + cadence) happen here, outside the ledger transaction, so the
    # lock order stays ledger-after-jobs (mark_job_run holds _jobs_lock while calling in).
    plan = _live_owner_reclaim_plan(rows)
    with _transaction() as conn:
        for row in rows:
            reason = plan.get((str(row["job_id"]), row["claimed_at"]))
            if reason is None:
                continue
            handoff_started_at = row["handoff_started_at"]
            if (
                row["handoff_pending"]
                and handoff_started_at is not None
                and time.time() - float(handoff_started_at)
                < HANDOFF_ADOPTION_GRACE_SECONDS
            ):
                continue
            cur = conn.execute(
                """UPDATE executions
                   SET status='unknown', finished_at=?, error=?,
                       handoff_pending=0, handoff_started_at=NULL
                   WHERE id=? AND status=? AND process_id=? AND pid=?
                     AND handoff_pending=?
                     AND handoff_started_at IS ?""",
                (now, reason, row["id"], row["status"], row["process_id"], row["pid"],
                 row["handoff_pending"], row["handoff_started_at"]),
            )
            changed += cur.rowcount
            if cur.rowcount:
                record = _fetch(conn, row["id"])
                if record is not None:
                    recovered.append(record)
        if changed:
            _prune_unlocked(conn)
    for record in recovered:
        _emit_execution_state(record)
    return changed


def terminalize_dead_owner(execution_id: str, *, reason: str) -> bool:
    """Record one attempt as ``unknown`` with a cause this process actually observed.

    ``recover_interrupted_executions`` sweeps every attempt whose owner is provably
    dead, and knows nothing but that absence — so all it can write is
    ``_OWNER_GONE_REASON``, which asserts a scheduler restart. A waiter that held the
    worker's ``Popen`` knows more: the owner was that external worker, and it exited
    with a known status. Without this, a manual run whose worker dies is filed as
    "Scheduler restarted ..." (a restart that never happened) and, because the sweep
    leaves the row terminal, the waiter reports success and never records the run —
    the job's ``fire_claim`` then blocks the next manual fire for the whole lease
    (#128509).

    The attempt stays ``unknown``, not ``failed``: whether side effects ran is still
    unknown. Only the CAUSE becomes truthful. Returns False — leaving the caller to
    fall back to the generic sweep — when the row is absent, already terminal, owned by
    this process, inside the handoff adoption grace, or owned by a live process: a
    worker that is still running must never be terminalized out from under itself.
    """
    now = _hermes_now().isoformat()
    with _transaction() as conn:
        row = conn.execute(
            """SELECT id, status, process_id, pid, process_started_at,
                      handoff_pending, handoff_started_at
               FROM executions WHERE id=?""",
            (execution_id,),
        ).fetchone()
        if row is None or row["status"] not in ("claimed", "running"):
            return False
        if row["process_id"] == _PROCESS_ID:
            return False
        if _owner_is_live(int(row["pid"]), row["process_started_at"]):
            return False
        handoff_started_at = row["handoff_started_at"]
        if (
            row["handoff_pending"]
            and handoff_started_at is not None
            and time.time() - float(handoff_started_at) < HANDOFF_ADOPTION_GRACE_SECONDS
        ):
            return False
        cur = conn.execute(
            """UPDATE executions
               SET status='unknown', finished_at=?, error=?,
                   handoff_pending=0, handoff_started_at=NULL
               WHERE id=? AND status=? AND process_id=? AND pid=?""",
            (now, reason, row["id"], row["status"], row["process_id"], row["pid"]),
        )
        if cur.rowcount != 1:
            return False
        record = _fetch(conn, execution_id)
        _prune_unlocked(conn)
    _emit_execution_state(record)
    return True


def list_executions(
    *, job_id: Optional[str] = None, limit: int = 50, before_claimed_at: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Return indexed, newest-first execution history with cursor pagination."""
    clauses: list[str] = []
    params: list[Any] = []
    if job_id is not None:
        clauses.append("job_id=?")
        params.append(str(job_id))
    if before_claimed_at is not None:
        # Same (instant, text) key as the ORDER BY, so a page never skips or repeats a row.
        clauses.append("(julianday(claimed_at), claimed_at) < (julianday(?), ?)")
        params.extend([str(before_claimed_at)] * 2)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    params.append(max(1, min(int(limit), 500)))
    # Stamps carry the local offset, which changes at DST and on a timezone change, so text order
    # is not time order. julianday() compares instants (ms); the text breaks same-ms ties.
    with _transaction() as conn:
        rows = conn.execute(
            "SELECT * FROM executions" + where
            + " ORDER BY julianday(claimed_at) DESC, claimed_at DESC, id DESC LIMIT ?",
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def get_execution(execution_id: str) -> Optional[dict[str, Any]]:
    """Return one exact execution attempt, or ``None`` when it is absent."""
    with _transaction() as conn:
        row = conn.execute(
            "SELECT * FROM executions WHERE id=?",
            (str(execution_id),),
        ).fetchone()
    return dict(row) if row is not None else None


def latest_execution(job_id: str) -> Optional[dict[str, Any]]:
    rows = list_executions(job_id=job_id, limit=1)
    return rows[0] if rows else None


def live_inflight_execution(job_id: str) -> Optional[dict[str, Any]]:
    """The job's latest attempt while it is still claimed/running under a LIVE owner, else ``None``.

    This is scheduler OWNERSHIP, not recent activity: a run inside a long tool call writes no
    heartbeat yet stays owned, while a run whose process died (watchdog kill, crash) does not.
    Read-only — unlike ``recover_interrupted_executions`` it never rewrites a row.
    """
    record = latest_execution(job_id)
    if not record or record.get("status") not in ("claimed", "running"):
        return None
    if not _owner_is_live(int(record["pid"]), record.get("process_started_at")):
        return None
    return record


def latest_executions(job_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Load latest execution for many jobs in one query."""
    clean = [str(job_id) for job_id in dict.fromkeys(job_ids) if job_id]
    if not clean:
        return {}
    placeholders = ",".join("?" for _ in clean)
    # One windowed sort: a per-row correlated ORDER BY julianday() cannot use the index and
    # grows quadratically with history (~90 ms at 1000 rows).
    with _transaction() as conn:
        rows = conn.execute(
            f"""SELECT e.* FROM executions e WHERE e.id IN (
                  SELECT id FROM (
                    SELECT id, ROW_NUMBER() OVER (
                             PARTITION BY job_id
                             ORDER BY julianday(claimed_at) DESC, claimed_at DESC, id DESC
                           ) AS rn
                    FROM executions WHERE job_id IN ({placeholders}))
                  WHERE rn=1)""",
            clean,
        ).fetchall()
    return {row["job_id"]: dict(row) for row in rows}
