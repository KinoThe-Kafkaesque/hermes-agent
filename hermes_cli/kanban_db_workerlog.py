"""The worker log's own file lifecycle and its dead-worker readers.

Split out of ``hermes_cli.kanban_db_dispatch`` (the file sits at its FILE_LINES cap,
so each fix there must offset its own growth). Two concerns live here:

* the log-tail readers for a worker the dispatcher did not reap —
  :func:`_worker_log_exit_trailer` / :func:`_worker_log_exit_code`, the durable
  witness of how the worker exited, written by the worker itself in
  ``hermes_cli.quiet_single_query.exit_single_query``. This is where the route
  that SERVED the run is named (``provider=``/``model=``), so a quota wall can be
  attributed to a provider rather than only to an exit code. And
  :func:`_worker_final_output`, the board diagnostic for a worker that exited
  without a terminal kanban call.
* the log file's rotation (:func:`worker_log_rotation_config` /
  :func:`_rotate_worker_log`), which bounds an append-mode log that grows across
  re-runs.

Origin-resident helpers (``read_worker_log``) are reached late-bound via ``_kb``
(import-cycle breaking) so monkeypatching ``kanban_db.<name>`` keeps working, the
same convention the other ``kanban_db_*`` siblings use.
"""

from __future__ import annotations

import contextlib
import re
import time
from pathlib import Path
from typing import Any
from typing import Optional

from hermes_cli.kanban_provider_distribution import (
    parse_exit_trailer,
    strip_exit_trailer,
)
from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER

_EXIT_TRAILER_RE = re.compile(
    r"^" + re.escape(KANBAN_WORKER_EXIT_TRAILER) + r"(\d+)(?P<fields>(?:\s+\w+=\S+)*)\s*$",
    re.MULTILINE,
)

# The worker CLI's exit summary names the session it just ran, and the session id
# embeds that session's own START (``YYYYMMDD_HHMMSS_<hex>``)::
#
#     Resume this session with:
#       hermes --resume 20261011_185224_9c04ba -p masisim
#
# The ``hermes --resume`` line is a literal in every shipped locale (only the
# surrounding label is translated), so it is a locale-safe per-run stamp — and it
# is the ONLY in-log evidence of WHEN a trailer was written, because a worker log
# carries no per-line timestamps. Measured over the whole live board, the stamp sits
# 155–426 bytes above its trailer (n=1096), so this window always reaches it.
_TRAILER_SESSION_RE = re.compile(r"hermes --resume\s+(\d{8})_(\d{6})_\w+")
_TRAILER_SESSION_LOOKBACK = 3000
# The stamp is the worker's OWN clock, taken a few seconds AFTER the dispatcher wrote
# ``started_at`` (measured: +2..+21 s over 325 runs, never negative) and before its own
# ``ended_at``. Both bounds matter: a trailer from an EARLIER run of the same task
# carries a stamp below ``started_at``, and one from a LATER run a stamp above
# ``ended_at``, so a two-sided window rejects both while accepting the run's own.
# Measured over the whole live board, no run's own trailer falls outside [started_at,
# ended_at]; the nearest foreign stamp is +368 s above ``started_at``. The tolerance
# only absorbs clock skew between the two processes.
_TRAILER_SESSION_TOLERANCE_SECONDS = 60


def _session_stamp_epoch(compact: str) -> Optional[int]:
    """Epoch of a ``YYYYMMDD_HHMMSS`` session stamp; None when it does not parse.

    The CLI builds the stamp from ``datetime.now()``, so it is LOCAL time — the
    same clock ``task_runs.started_at`` is written from.
    """
    try:
        return int(time.mktime(time.strptime(compact, "%Y%m%d%H%M%S")))
    except ValueError:
        return None


def _last_trailer_session_start(raw: Optional[str]) -> Optional[int]:
    """Epoch of the session start belonging to the LAST trailer's own block; None if absent.

    Read from the ``hermes --resume <session_id>`` line in the exit summary that
    immediately precedes that trailer, so it describes the trailer's run rather
    than the file's first one. None means the log carries no stamp for that block —
    a worker killed before its exit epilogue — which the caller must treat as
    "this trailer is not provably this run's", never as a match.
    """
    if not raw:
        return None
    trailers = list(_EXIT_TRAILER_RE.finditer(raw))
    if not trailers:
        return None
    window = raw[max(0, trailers[-1].start() - _TRAILER_SESSION_LOOKBACK):trailers[-1].start()]
    stamp = None
    for match in _TRAILER_SESSION_RE.finditer(window):
        stamp = match
    if stamp is None:
        return None
    return _session_stamp_epoch(stamp.group(1) + stamp.group(2))


def _read_worker_log(task_id: str, board: Optional[str] = None):
    """Late-bound ``kanban_db.read_worker_log``.

    Imported at CALL time, not module load: this sibling is imported from the top
    of ``kanban_db_dispatch``, which ``kanban_db`` itself imports from later in its
    own body — a module-level ``_kb`` here would close that cycle early and fail
    with "cannot import name ... from partially initialized module". Call-time
    resolution also keeps ``monkeypatch.setattr(kanban_db, "read_worker_log", ...)``
    working, which is the convention every ``kanban_db_*`` sibling follows.
    """
    from hermes_cli import kanban_db as _kb
    return _kb.read_worker_log(task_id, tail_bytes=4000, board=board)


def _worker_log_exit_trailer(task_id: str, board: Optional[str] = None,
                             *, observed_code: Optional[int] = None,
                             run_started_at: Optional[int] = None,
                             read_at: Optional[int] = None):
    """``(exit_code, provider, model)`` from the trailer the worker CLI wrote to its own log;
    all None when absent.

    The durable twin of ``_recent_worker_exits``: written by the worker itself
    (``hermes_cli.quiet_single_query.exit_single_query``), so it is there whether
    or not the process running this sweep ever reaped the worker. The provider/model
    fields are optional and absent from an older worker binary, which parses fine:
    the rc is the load-bearing field and an unnamed route records as unknown rather
    than failing the classification.

    Two DIFFERENT questions are answered here, and they have different evidence:

    * ``observed_code`` decides whether the whole read belongs to this run at all. It
      is the rc the SWEEP already knows independently (the reaped exit status, or
      ``None`` for an unreaped worker). When given, only a trailer whose rc MATCHES it
      is accepted. Measured on the live board, 429 of 539 ``rate_limited`` rows tail a
      trailer with a different rc — e.g. rc=0 from the very run that later succeeded.
      A trailer that does not match proves the log tail is not this run's, so it is
      discarded WHOLE — never partially trusted for its route.

    * ``run_started_at`` / ``read_at`` decide whether the trailer's ROUTE may be
      charged, on the path where there is no observed rc at all (the unreaped path,
      where the pid is not in this process's reap registry). The trailer's own block
      carries the session start it ran (see ``_last_trailer_session_start``); the route
      is trusted only inside ``[run_started_at - tol, read_at + tol]``. BOTH bounds are
      load-bearing: an earlier run's trailer stamps below ``run_started_at`` and a later
      run's above ``read_at`` (the moment this sweep looked), so a one-sided lower bound
      accepts every later run and discriminates nothing. Measured over the whole live
      board: every run's own trailer stamps +2..+21 s after ``run_started_at`` (n=325,
      never before it) while every foreign one is at least +368 s out.

      The rc is deliberately NOT gated by this window. On the unreaped path the tail
      trailer is the ONLY witness of how the worker exited, so blanking it would flip
      the row from ``rate_limited`` to a failure-counting ``crashed`` — reclassifying a
      quota wall to fix an attribution bug. Only the route is withheld; the wall stays
      unnamed and demotes nothing, which is the safe direction.
    """
    try:
        raw = _read_worker_log(task_id, board=board)
    except Exception:  # health: allow BLE001 -- a missing/unreadable worker log is not a wall
        return None, None, None
    trailer = parse_exit_trailer(raw)
    if trailer is None:
        return None, None, None
    if observed_code is not None and trailer.exit_code != observed_code:
        return None, None, None
    if run_started_at is not None:
        session_start = _last_trailer_session_start(raw)
        upper = (time.time() if read_at is None else read_at) + _TRAILER_SESSION_TOLERANCE_SECONDS
        if session_start is None or not (run_started_at - _TRAILER_SESSION_TOLERANCE_SECONDS
                                         <= session_start <= upper):
            return trailer.exit_code, None, None
    return trailer.exit_code, trailer.provider, trailer.model


def _worker_log_exit_code(task_id: str, board: Optional[str] = None) -> Optional[int]:
    """Exit code from the trailer the worker CLI wrote to its own log; None when absent.

    The durable twin of ``_recent_worker_exits``: written by the worker itself
    (``hermes_cli.quiet_single_query.exit_single_query``), so it is there whether
    or not the process running this sweep ever reaped the worker. Last trailer
    wins — the log is append-mode across re-runs.
    """
    code, _provider, _model = _worker_log_exit_trailer(task_id, board=board)
    return code


# Rich panel/rule chrome around the rendered response, and the CLI's own preamble lines.
_LOG_CHROME = re.compile(r"[─━═╭╮╰╯│┃┌┐└┘]+|☤\s*Hermes")


def _exit_summary_marker() -> str:
    """The CLI exit-summary header (``cli_session_mixin.show_exit_summary``), in the active language."""
    from agent.i18n import t
    return t("cli.session.exit_resume_hint")


def _log_noise_prefixes() -> tuple[str, ...]:
    from agent.i18n import t
    return ("session_id:", "Query:", t("cli.chat.initializing_agent"))


def _worker_final_output(task_id: str, board: Optional[str] = None) -> str:
    """Best-effort read of a dead worker's last printed text, for the board diagnostic.

    A ``chat -q`` worker's stdout/stderr are redirected to its per-task log
    (``_default_spawn``), so when it exits without a terminal board call the
    reason is usually sitting there: the model's own explanation of why it could
    not comply (#88603), or the rendered provider error (#46593). The reap used to
    discard it in favour of a canned message on every retry. Trims the CLI exit
    summary, rule lines and the ``session_id:`` trailer; returns "" (never raises)
    on a missing/empty log.

    ``board`` must come from the dispatching tick: ambient current-board resolution
    is wrong for every board but the one the dispatcher thread happens to call
    "current", so the log would silently not be found.
    """
    try:
        raw = _read_worker_log(task_id, board=board)
    except Exception:
        return ""
    if not raw:
        return ""
    raw = strip_exit_trailer(raw)
    cut = raw.rfind(_exit_summary_marker())
    if cut != -1:
        raw = raw[:cut]
    lines = []
    for ln in raw.splitlines():
        ln = _LOG_CHROME.sub("", ln).strip()
        if ln and not ln.startswith(_log_noise_prefixes()):
            lines.append(ln)
    return " ".join(lines)[-400:]


# --- Log rotation (moved from the dispatch facade to offset its FILE_LINES cap) ---
# Worker logs larger than ``DEFAULT_LOG_ROTATE_BYTES`` at spawn time are rotated.

DEFAULT_LOG_ROTATE_BYTES = 2 * 1024 * 1024   # 2 MiB
DEFAULT_LOG_BACKUP_COUNT = 1


def _positive_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


def worker_log_rotation_config(kanban_cfg: Optional[dict] = None) -> tuple[int, int]:
    """Return ``(rotate_bytes, backup_count)`` for worker log rotation.
    Defaults: rotate at 2 MiB, keep one backup (``.log.1``); both overridable
    from ``config.yaml``.
    """
    if kanban_cfg is None:
        try:
            from hermes_cli.config import load_config

            kanban_cfg = (load_config().get("kanban") or {})
        except Exception:
            kanban_cfg = {}
    kanban_cfg = kanban_cfg or {}
    max_bytes = _positive_int(kanban_cfg.get("worker_log_rotate_bytes"), DEFAULT_LOG_ROTATE_BYTES, minimum=1)
    backup_count = _positive_int(kanban_cfg.get("worker_log_backup_count"), DEFAULT_LOG_BACKUP_COUNT, minimum=0)
    return max_bytes, backup_count


def _rotated_log_path(log_path: Path, generation: int) -> Path:
    return log_path.with_suffix(log_path.suffix + f".{generation}")


def _rotate_worker_log(
    log_path: Path,
    max_bytes: int,
    backup_count: int = DEFAULT_LOG_BACKUP_COUNT,
) -> None:
    """Rotate ``<log>`` when it exceeds ``max_bytes``: ``<log>`` → ``<log>.1``,
    older generations shift up to ``backup_count``.
    """
    try:
        if not log_path.exists() or log_path.stat().st_size <= max_bytes:
            return
        backup_count = _positive_int(backup_count, DEFAULT_LOG_BACKUP_COUNT, minimum=0)
        if backup_count == 0:
            log_path.unlink()
            return
        oldest = _rotated_log_path(log_path, backup_count)
        with contextlib.suppress(OSError):
            if oldest.exists():
                oldest.unlink()
        for generation in range(backup_count - 1, 0, -1):
            src = _rotated_log_path(log_path, generation)
            if not src.exists():
                continue
            with contextlib.suppress(OSError):
                src.rename(_rotated_log_path(log_path, generation + 1))
        log_path.rename(_rotated_log_path(log_path, 1))
    except OSError:
        pass
