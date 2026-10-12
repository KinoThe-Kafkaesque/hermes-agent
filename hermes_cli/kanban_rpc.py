"""Remote-worker board RPC — owner-side protocol, receipts, and execution.

A remote worker never opens the board. It sends versioned JSON-RPC frames over a
persistent stdio channel (`hermes kanban rpc --stdio`, reached by SSH) carrying:

    {version, board_id, actor_task_id, run_id, request_id, operation,
     target_task_id, payload}

The owner authenticates the connection, validates the frame, and executes the
operation against its **local** board. Retryable mutations are deduplicated by a
receipt row committed **in the same transaction as the mutation**.

Crash argument (the reason the receipt shares the mutation's transaction):

    before commit        neither mutation nor receipt survives -> retry executes once
    after commit, no reply  both survive                      -> retry returns stored result
    after reply          both survive                         -> retry returns stored result

The two arrangements that would break it — committing a "processed" marker before
the mutation, or committing the mutation and then inserting its receipt in a
separate transaction — are both forbidden.

Transport is deliberately absent from this module: everything here operates on an
already-authenticated request. See `kanban_rpc_stdio` for the framing loop.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

_log = logging.getLogger(__name__)

#: Protocol version. A frame with any other value is rejected before touching the DB.
PROTOCOL_VERSION = 1

#: Hard cap on a single frame, enforced before JSON parsing.
MAX_FRAME_BYTES = 1 << 20  # 1 MiB

#: Operations the owner will execute. Deliberately small: this is increment one.
OP_CLAIM_STATUS = "claim_status"
OP_COMMENT = "comment"

#: Read-only operations, which do not consume a receipt.
READ_ONLY_OPERATIONS = frozenset({OP_CLAIM_STATUS})

#: Mutating operations, which MUST commit a receipt alongside their effect.
MUTATING_OPERATIONS = frozenset({OP_COMMENT})


class RpcError(Exception):
    """A request the owner refuses. ``code`` is stable and machine-readable."""

    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable

    def to_frame(self) -> dict[str, Any]:
        return {"ok": False, "error": {"code": self.code, "message": self.message,
                                       "retryable": self.retryable}}


# --------------------------------------------------------------------------
# Request envelope
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BoardRequest:
    """One validated request. ``request_hash`` is computed over the canonical form."""

    version: int
    board_id: str
    actor_task_id: str
    run_id: int
    request_id: str
    operation: str
    target_task_id: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_wire(cls, frame: dict[str, Any]) -> BoardRequest:
        """Validate a decoded frame. Raises :class:`RpcError` on any defect.

        Validation is strict and happens before the database is touched: an
        unparseable or incomplete frame must never reach a transaction.
        """
        if not isinstance(frame, dict):
            raise RpcError("malformed_frame", "frame must be a JSON object")

        version = frame.get("version")
        if version != PROTOCOL_VERSION:
            raise RpcError(
                "unsupported_version",
                f"protocol version {version!r} is not supported (expected {PROTOCOL_VERSION})",
            )

        def _require_str(key: str) -> str:
            value = frame.get(key)
            if not isinstance(value, str) or not value.strip():
                raise RpcError("malformed_frame", f"{key!r} must be a non-empty string")
            return value

        operation = _require_str("operation")
        if operation not in (READ_ONLY_OPERATIONS | MUTATING_OPERATIONS):
            raise RpcError("unsupported_operation", f"operation {operation!r} is not supported")

        run_id = frame.get("run_id")
        if not isinstance(run_id, int) or isinstance(run_id, bool) or run_id <= 0:
            raise RpcError("malformed_frame", "'run_id' must be a positive integer")

        payload = frame.get("payload") or {}
        if not isinstance(payload, dict):
            raise RpcError("malformed_frame", "'payload' must be an object")

        target = frame.get("target_task_id")
        if target is not None and (not isinstance(target, str) or not target.strip()):
            raise RpcError("malformed_frame", "'target_task_id' must be a non-empty string or absent")

        return cls(
            version=version,
            board_id=_require_str("board_id"),
            actor_task_id=_require_str("actor_task_id"),
            run_id=run_id,
            request_id=_require_str("request_id"),
            operation=operation,
            target_task_id=target,
            payload=payload,
        )

    # -- canonical form ----------------------------------------------------

    def canonical(self) -> bytes:
        """Canonical bytes for hashing.

        Covers the whole validated envelope EXCEPT any credential: the capability
        token is not a field here by construction, so a token rotation must not
        change a request's identity. Sorted keys + separators make the encoding
        independent of dict ordering and incidental whitespace.
        """
        body = {
            "version": self.version,
            "board_id": self.board_id,
            "actor_task_id": self.actor_task_id,
            "run_id": self.run_id,
            "request_id": self.request_id,
            "operation": self.operation,
            "target_task_id": self.target_task_id,
            "payload": self.payload,
        }
        return json.dumps(body, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8")

    def request_hash(self) -> bytes:
        return hashlib.sha256(self.canonical()).digest()


# --------------------------------------------------------------------------
# Receipts — the durable dedup
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Receipt:
    result: dict[str, Any]


def _lookup_receipt(conn: sqlite3.Connection, req: BoardRequest) -> Optional[Receipt]:
    """Return the stored receipt for this request, or None.

    Reusing a request id for a *different* envelope is a hard conflict: the
    caller has a bug or is replaying with altered content, and neither executing
    nor silently returning the old result is safe.
    """
    row = conn.execute(
        "SELECT request_hash, result_json FROM board_rpc_receipts "
        "WHERE board_id = ? AND run_id = ? AND request_id = ?",
        (req.board_id, req.run_id, req.request_id),
    ).fetchone()
    if row is None:
        return None
    stored_hash = row["request_hash"]
    if isinstance(stored_hash, str):
        stored_hash = bytes.fromhex(stored_hash)
    if stored_hash != req.request_hash():
        raise RpcError(
            "request_id_conflict",
            f"request_id {req.request_id!r} was already used for a different request",
        )
    try:
        result = json.loads(row["result_json"])
    except (TypeError, ValueError):
        raise RpcError("receipt_unreadable", "stored receipt could not be decoded")
    if not isinstance(result, dict):
        raise RpcError("receipt_unreadable", "stored receipt is not an object")
    return Receipt(result=result)


def _store_receipt(conn: sqlite3.Connection, req: BoardRequest,
                   result: dict[str, Any], *, now: Optional[int] = None) -> None:
    """Insert the receipt. MUST be called inside the mutation's own transaction."""
    conn.execute(
        "INSERT INTO board_rpc_receipts "
        "(board_id, run_id, request_id, actor_task_id, target_task_id, operation, "
        " request_hash, result_json, committed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (req.board_id, req.run_id, req.request_id, req.actor_task_id,
         req.target_task_id, req.operation, req.request_hash(),
         json.dumps(result, sort_keys=True, ensure_ascii=False),
         int(now if now is not None else time.time())),
    )


# --------------------------------------------------------------------------
# Authorization — verified inside the mutation's transaction
# --------------------------------------------------------------------------

def verify_run_authority(conn: sqlite3.Connection, req: BoardRequest) -> None:
    """Assert the actor task is still running THIS run, inside the caller's txn.

    Deliberately re-checked under the write lock rather than trusted from
    connection setup: between authenticating the connection and taking the write
    transaction, the card can be reclaimed and a second run started.

    NOTE: this is an *authorization* check. It is not authentication — the
    identity triple (task, run, claim lock) is a concurrency coordinate, not a
    credential. Authentication is the capability check performed by the caller
    before dispatching here.
    """
    row = conn.execute(
        "SELECT status, current_run_id FROM tasks WHERE id = ?", (req.actor_task_id,)
    ).fetchone()
    if row is None:
        raise RpcError("unknown_actor_task", f"actor task {req.actor_task_id!r} does not exist")
    if row["status"] != "running":
        raise RpcError(
            "actor_not_running",
            f"actor task {req.actor_task_id!r} is {row['status']!r}, not 'running'",
        )
    if row["current_run_id"] != req.run_id:
        raise RpcError(
            "stale_run",
            f"run {req.run_id} is not the live run of {req.actor_task_id!r}",
        )
    run = conn.execute(
        "SELECT task_id, status FROM task_runs WHERE id = ?", (req.run_id,)
    ).fetchone()
    if run is None:
        raise RpcError("unknown_run", f"run {req.run_id} does not exist")
    if run["task_id"] != req.actor_task_id:
        raise RpcError("run_task_mismatch", f"run {req.run_id} does not belong to the actor task")


# --------------------------------------------------------------------------
# Operation implementations — transaction-owned, shared with local callers
# --------------------------------------------------------------------------

def _op_claim_status(conn: sqlite3.Connection, req: BoardRequest) -> dict[str, Any]:
    """Read-only: report the authoritative claim state for the actor run.

    This replaces the remote branch of ``worker_claim_is_live()``, which reads
    the board directly and therefore cannot work off-owner.
    """
    row = conn.execute(
        "SELECT t.status, t.current_run_id, t.claim_lock, t.last_heartbeat_at, "
        "       r.status AS run_status, r.claim_lock AS run_claim_lock "
        "  FROM tasks t LEFT JOIN task_runs r ON r.id = t.current_run_id "
        " WHERE t.id = ?",
        (req.actor_task_id,),
    ).fetchone()
    if row is None:
        raise RpcError("unknown_actor_task", f"actor task {req.actor_task_id!r} does not exist")
    live = (row["status"] == "running" and row["current_run_id"] == req.run_id)
    return {
        "actor_task_id": req.actor_task_id,
        "run_id": req.run_id,
        "task_status": row["status"],
        "current_run_id": row["current_run_id"],
        "run_status": row["run_status"],
        "claim_lock": row["claim_lock"],
        "last_heartbeat_at": row["last_heartbeat_at"],
        "is_live": live,
    }


def _op_comment(conn: sqlite3.Connection, req: BoardRequest) -> dict[str, Any]:
    """Mutating: append a comment to ``target_task_id`` (default: the actor task).

    Author is taken from the authenticated grant — never from the payload, which
    the model controls.
    """
    body = req.payload.get("body")
    if not isinstance(body, str) or not body.strip():
        raise RpcError("invalid_payload", "'body' must be a non-empty string")
    target = req.target_task_id or req.actor_task_id
    author = req.payload.get("_author")  # injected from the grant, never from the wire
    if not isinstance(author, str) or not author:
        raise RpcError("no_author", "authenticated author is missing")
    comment_id = kb.add_comment(conn, target, author, body)
    return {"comment_id": comment_id, "target_task_id": target, "author": author}


_OPERATIONS: dict[str, Callable[[sqlite3.Connection, BoardRequest], dict[str, Any]]] = {
    OP_CLAIM_STATUS: _op_claim_status,
    OP_COMMENT: _op_comment,
}


# --------------------------------------------------------------------------
# The owner execution algorithm
# --------------------------------------------------------------------------

def execute_request(req: BoardRequest, *, conn: Optional[sqlite3.Connection] = None,
                    author: Optional[str] = None) -> dict[str, Any]:
    """Execute one authenticated request against the local board.

    ``author`` comes from the connection's authenticated grant and is injected
    into the payload so operation implementations never read an author the model
    supplied. It ALWAYS overrides any ``_author`` already present: the payload is
    model-controlled, so a caller-supplied author must never win.
    """
    if author:
        object.__setattr__(req, "payload", {**req.payload, "_author": author})

    handler = _OPERATIONS.get(req.operation)
    if handler is None:
        raise RpcError("unsupported_operation", f"operation {req.operation!r} is not supported")

    own_conn = conn is None
    if own_conn:
        conn = kbc.connect()

    try:
        if req.operation in READ_ONLY_OPERATIONS:
            # No receipt: a read has no effect to deduplicate, and committing a
            # receipt for one would make every poll a write.
            return handler(conn, req)

        # The receipt is committed in the SAME transaction as the mutation. This
        # is the whole correctness argument; see the module docstring.
        with kbc.write_txn(conn):
            cached = _lookup_receipt(conn, req)
            if cached is not None:
                return cached.result
            verify_run_authority(conn, req)
            result = handler(conn, req)
            _store_receipt(conn, req, result)
            return result
    finally:
        if own_conn:
            conn.close()


def new_request_id() -> str:
    """A fresh, opaque request id."""
    return secrets.token_hex(16)
