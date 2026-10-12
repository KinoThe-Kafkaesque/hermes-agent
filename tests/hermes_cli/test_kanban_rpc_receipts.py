"""Remote-worker board RPC — the durable-receipt contract.

The property under test is the real transaction boundary, not a mocked one: a
mutation and its receipt must commit together, so that a lost reply is harmless
and a retry can never apply the effect twice.

Deliberately uses real imports and a real on-disk board. Mocking ``connect()``
here would test the mock, not the boundary.
"""

from __future__ import annotations

import pathlib

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_rpc as rpc


@pytest.fixture()
def board(tmp_path):
    """A disposable board with one running task + run, created by real domain code."""
    conn = kbc.connect(db_path=pathlib.Path(tmp_path) / "kanban.db")
    task_id = kb.create_task(conn, title="remote fixture", assignee="masisim", body="x")
    if hasattr(task_id, "id"):
        task_id = task_id.id
    with kbc.write_txn(conn):
        conn.execute("UPDATE tasks SET status='running' WHERE id=?", (task_id,))
        conn.execute(
            "INSERT INTO task_runs (task_id, status, started_at, claim_lock) VALUES (?,?,?,?)",
            (task_id, "running", 0, "owner:1"),
        )
    run_id = conn.execute(
        "SELECT id FROM task_runs WHERE task_id=?", (task_id,)
    ).fetchone()[0]
    with kbc.write_txn(conn):
        conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (run_id, task_id))
    yield conn, task_id, run_id
    conn.close()


def _frame(task_id, run_id, body, request_id, operation="comment"):
    return {
        "version": rpc.PROTOCOL_VERSION,
        "board_id": "board-uuid-test",
        "actor_task_id": task_id,
        "run_id": run_id,
        "request_id": request_id,
        "operation": operation,
        "payload": {"body": body},
    }


def _send(conn, frame):
    return rpc.execute_request(rpc.BoardRequest.from_wire(frame), conn=conn, author="masisim")


def _counts(conn, task_id):
    return (
        conn.execute("SELECT COUNT(*) FROM task_comments WHERE task_id=?", (task_id,)).fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM board_rpc_receipts").fetchone()[0],
    )


def test_first_send_executes_once(board):
    conn, task_id, run_id = board
    result = _send(conn, _frame(task_id, run_id, "hello", "req-1"))
    assert result["comment_id"] >= 1
    assert result["author"] == "masisim"
    assert _counts(conn, task_id) == (1, 1)


def test_replay_returns_stored_result_without_reapplying(board):
    """The lost-reply retry: same request id, same payload."""
    conn, task_id, run_id = board
    first = _send(conn, _frame(task_id, run_id, "hello", "req-1"))
    second = _send(conn, _frame(task_id, run_id, "hello", "req-1"))
    assert first == second
    assert _counts(conn, task_id) == (1, 1)


def test_reused_request_id_with_different_payload_is_a_conflict(board):
    """A request id is bound to its envelope; reusing it for other content must not execute."""
    conn, task_id, run_id = board
    _send(conn, _frame(task_id, run_id, "hello", "req-1"))
    with pytest.raises(rpc.RpcError) as excinfo:
        _send(conn, _frame(task_id, run_id, "DIFFERENT", "req-1"))
    assert excinfo.value.code == "request_id_conflict"
    assert _counts(conn, task_id) == (1, 1)


def test_stale_run_cannot_mutate(board):
    """Once the card is no longer running THIS run, a new mutation is refused."""
    conn, task_id, run_id = board
    with kbc.write_txn(conn):
        conn.execute("UPDATE tasks SET status='done', current_run_id=NULL WHERE id=?", (task_id,))
    with pytest.raises(rpc.RpcError) as excinfo:
        _send(conn, _frame(task_id, run_id, "late", "req-2"))
    assert excinfo.value.code == "actor_not_running"
    assert _counts(conn, task_id) == (0, 0)


def test_receipt_replay_precedes_the_liveness_rejection(board):
    """A successful completion closes its run; if the reply was lost, the retry must
    still be served from the receipt rather than rejected as a stale run."""
    conn, task_id, run_id = board
    first = _send(conn, _frame(task_id, run_id, "hello", "req-1"))
    with kbc.write_txn(conn):
        conn.execute("UPDATE tasks SET status='done', current_run_id=NULL WHERE id=?", (task_id,))
    replayed = _send(conn, _frame(task_id, run_id, "hello", "req-1"))
    assert replayed == first
    assert _counts(conn, task_id) == (1, 1)


def test_read_only_operation_writes_no_receipt(board):
    """Polling must not turn into a write."""
    conn, task_id, run_id = board
    result = _send(conn, _frame(task_id, run_id, "", "poll-1", operation=rpc.OP_CLAIM_STATUS))
    assert result["is_live"] is True
    assert result["current_run_id"] == run_id
    assert _counts(conn, task_id) == (0, 0)


def test_protocol_version_is_enforced(board):
    conn, task_id, run_id = board
    frame = _frame(task_id, run_id, "hi", "req-1")
    frame["version"] = rpc.PROTOCOL_VERSION + 1
    with pytest.raises(rpc.RpcError) as excinfo:
        _send(conn, frame)
    assert excinfo.value.code == "unsupported_version"


def test_author_comes_from_the_grant_not_the_payload(board):
    """The model controls the payload; it must not be able to set the author."""
    conn, task_id, run_id = board
    frame = _frame(task_id, run_id, "hello", "req-1")
    frame["payload"]["_author"] = "somebody-else"
    result = _send(conn, frame)
    assert result["author"] == "masisim"
