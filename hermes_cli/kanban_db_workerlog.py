"""Reading a dead worker's own log: its exit trailer and its last printed output.

Split out of ``hermes_cli.kanban_db_dispatch`` (the file was at its FILE_LINES cap
and the provider-attribution fix had to offset its own growth). Two readers live
here, both of them log-tail readers for a worker the dispatcher did not reap:

* :func:`_worker_log_exit_trailer` / :func:`_worker_log_exit_code` — the durable
  witness of how the worker exited, written by the worker itself in
  ``hermes_cli.quiet_single_query.exit_single_query``. This is where the route
  that SERVED the run is named (``provider=``/``model=``), so a quota wall can be
  attributed to a provider rather than only to an exit code.
* :func:`_worker_final_output` — the board diagnostic for a worker that exited
  without a terminal kanban call.

Origin-resident helpers (``read_worker_log``) are reached late-bound via ``_kb``
(import-cycle breaking) so monkeypatching ``kanban_db.<name>`` keeps working, the
same convention the other ``kanban_db_*`` siblings use.
"""

from __future__ import annotations

import re
from typing import Optional

from hermes_cli.kanban_provider_distribution import (
    parse_exit_trailer,
    strip_exit_trailer,
)
from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER

_EXIT_TRAILER_RE = re.compile(
    r"^" + re.escape(KANBAN_WORKER_EXIT_TRAILER) + r"(\d+)\s*$", re.MULTILINE,
)


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
                             *, observed_code: Optional[int] = None):
    """``(exit_code, provider, model)`` from the trailer the worker CLI wrote to its own log;
    all None when absent.

    The durable twin of ``_recent_worker_exits``: written by the worker itself
    (``hermes_cli.quiet_single_query.exit_single_query``), so it is there whether
    or not the process running this sweep ever reaped the worker. The provider/model
    fields are optional and absent from an older worker binary, which parses fine:
    the rc is the load-bearing field and an unnamed route records as unknown rather
    than failing the classification.

    *observed_code* is the rc the SWEEP already knows independently (the reaped
    exit status, or ``None`` for an unreaped worker). When given, only a trailer
    whose rc MATCHES it is accepted. The log is append-mode across re-runs, so the
    tail trailer can belong to a LATER run of the same task: measured on the live
    board, 429 of 539 ``rate_limited`` rows tail a trailer with a different rc —
    e.g. rc=0 from the very run that later succeeded. Accepting that trailer would
    stamp the wrong rc onto the wall AND charge the wall to whatever route the
    later run happened to use, which is exactly the guessed attribution the
    unnamed-wall rule exists to prevent. A trailer that does not match proves only
    that the log tail is not this run's, so it is discarded whole — never
    partially trusted for its route.

    Passing ``None`` keeps the legacy "last trailer wins" read for callers that
    have no independent rc (none in production today; the trailer-only contract
    predates the attribution fields).
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
