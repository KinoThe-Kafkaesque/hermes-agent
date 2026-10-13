import threading

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method

# onboarding.state and onboarding.set_run are unscoped on purpose: onboarding.run and the first-chat
# flag live in the root profile's config.yaml whichever profile this backend was launched under.
@method("onboarding.state")
def _(rid, params: dict) -> dict:
    return _ok(rid, _run_state())


@method("onboarding.set_run")
def _(rid, params: dict) -> dict:
    from hermes_cli.onboarding_run import set_run
    if not isinstance(params.get("run"), bool):
        return _err(rid, 4002, "onboarding.set_run requires a boolean 'run'")
    try:
        set_run(params["run"], mark_profile_offered=bool(params.get("mark_profile_offered")))
    except Exception as e:
        logger.exception("onboarding.set_run failed")
        return _err(rid, 5077, str(e))
    return _ok(rid, _run_state())


def _run_state() -> dict:
    from hermes_cli.anon_auth import guest_enabled
    from hermes_cli.onboarding_run import should_run
    # The lookup ends a run whose quiet spell ran out and closes the flag, so it goes first.
    session_id = _active_onboarding_session()
    return {"run": should_run(), "eligible": guest_enabled(), "session_id": session_id}


# ── first-run onboarding: one chat in the default profile, driven by setup_beat ───────────────

# Onboarding always runs in the default profile, whichever profile this backend was launched under.
_ONBOARDING_PROFILE = {"profile": "default"}
# Two windows pressing start at once must not both find no run and open two onboarding chats.
_onboarding_start_lock = threading.Lock()


def _active_onboarding_session() -> str | None:
    from agent.onboarding_record import active_session_id
    with _profile_db(_ONBOARDING_PROFILE, writer=True) as db:
        return None if db is None else active_session_id(db)


@method("onboarding.start")
def _(rid, params: dict) -> dict:
    from agent.onboarding_facts import read_facts
    from agent.onboarding_flow import Facts, SkipAll
    from agent.onboarding_record import advance, load, open_session_ids, start
    from agent.onboarding_text import CHAT_TITLE, opening_prompt
    # Held through the opening submit: two windows pressing start at once get one chat and one opening.
    with _onboarding_start_lock:
        try:
            with _profile_db(_ONBOARDING_PROFILE, writer=True) as db:
                if db is None:
                    return _db_unavailable_error(rid, code=5000)
                open_ids = open_session_ids(db)
                stored = None if params.get("rerun") else next(iter(open_ids), None)
                created = stored is None
                if created:
                    # The tours setting is the default profile's, whichever profile launched the backend.
                    home = _profile_home("default")
                    with _session_profile_runtime_scope({"profile_home": str(home) if home else None},
                                                        hydrate_secrets=False):
                        facts = read_facts(facts_source=params.get("facts_source") or "local", locale=params.get("locale"))
                    stored = db.create_session(new_session_id(), "desktop")
                    db.set_session_title(stored, db.get_next_title_in_lineage(CHAT_TITLE))
                    # The record exists before the agent is built, so the build folds the onboarding toolset in.
                    start(db, stored, facts)
                else:
                    facts = Facts.from_stored(load(db, stored)["facts"])
                # One open run at a time. The kept run is open before the others end, so the install
                # flag never closes in between; finishing it closes the flag.
                replaced = [sid for sid in open_ids if sid != stored]
                for old in replaced:
                    advance(db, old, SkipAll())
                # A chat with no messages never got its opening (a restart between start and the first turn).
                opening = None if (db.get_session(stored) or {}).get("message_count") else opening_prompt(facts)
        except Exception as e:
            logger.exception("onboarding.start failed")
            return _err(rid, 5078, str(e))
        _withdraw_setup_cards(replaced)
        resumed = _methods["session.resume"](rid, {**_ONBOARDING_PROFILE, "session_id": stored, "source": "desktop"})
        if "error" in resumed:
            return resumed
        sid = resumed["result"]["session_id"]
        if opening is not None:
            # The opening rules ride a hidden user turn; its prelude plays the greeting and the name card.
            submitted = _methods["prompt.submit"](rid, {"session_id": sid, "text": opening, "display_kind": "hidden"})
            if "error" in submitted:
                return submitted
    return _ok(rid, {"session_id": sid, "stored_session_id": stored, "resumed": not created})


@method("onboarding.skip")
def _(rid, params: dict) -> dict:
    from agent.onboarding_flow import SkipAll
    from agent.onboarding_record import advance, load
    session, err = _sess_nowait(params, rid)
    if err:
        return err
    agent = session.get("agent")
    stored = getattr(agent, "session_id", None) or session["session_key"]
    with _session_db(session) as db:
        if load(db, stored) is None:
            return _ok(rid, {"skipped": False})
        step = advance(db, stored, SkipAll())
    from tui_gateway import server_requests
    # The card that waits returns to setup_beat, which reads the skipped record.
    server_requests.cancel(params["session_id"], "resolved", method="setup_beat")
    return _ok(rid, {"skipped": bool(step.result.get("skipped_all"))})


def _withdraw_setup_cards(stored_ids: list[str]) -> None:
    """Close the card still waiting in a chat whose run just ended; its answer would be refused."""
    from tui_gateway import server_requests
    if not stored_ids:
        return
    with _sessions_lock:
        live = [sid for sid, sess in _sessions.items()
                if (getattr(sess.get("agent"), "session_id", None) or sess.get("session_key")) in stored_ids]
    for sid in live:
        server_requests.cancel(sid, "resolved", method="setup_beat")


def _setup_beat_request(sid: str, payload: dict) -> dict | None:
    from agent.onboarding_flow import CARD_TIMEOUT_S
    from tui_gateway import server_requests
    return server_requests.send("setup_beat", sid, dict(payload), timeout=CARD_TIMEOUT_S)


def _builds_onboarding(agent) -> bool:
    return "setup_beat" in (getattr(agent, "valid_tool_names", None) or ())


def _onboarding_toolsets(enabled, db, session_id: str):
    """*enabled* plus ``onboarding`` for a chat whose run is in progress. Decided once, at agent build:
    the tool array is part of the cached prefix, so a finished run keeps setup_beat (as a refusal)
    until the session is rebuilt."""
    from agent.onboarding_flow import is_active
    from agent.onboarding_record import load
    if not enabled or "onboarding" in enabled:
        return enabled
    return [*enabled, "onboarding"] if is_active(load(db if db is not None else _get_db(), session_id)) else enabled


def _carry_onboarding_toolset(enabled, agent):
    """A tool refresh keeps the onboarding toolset the agent was built with."""
    if not enabled or "onboarding" in enabled or "onboarding" not in (getattr(agent, "enabled_toolsets", None) or ()):
        return enabled
    return [*enabled, "onboarding"]


def _onboarding_prelude(agent, history: list):
    """The opening's scripted step, greeting and name card, while the name beat was never shown. A
    greeting already in the chat (a relaunch) is not said twice."""
    from agent.onboarding_flow import PENDING, is_active
    from agent.onboarding_record import load
    from agent.onboarding_text import GREETING
    if not _builds_onboarding(agent):
        return None
    record = load(agent._session_db, agent.session_id)
    if not is_active(record) or record["beats"]["name"] != PENDING:
        return None
    said = any(m.get("role") == "assistant" and m.get("content") == GREETING for m in history)

    def opening():
        yield ("" if said else GREETING), "setup_beat", {"beat": "name"}
    return opening()


def _stage_onboarding_turn_note(sid: str, agent, history_empty: bool) -> None:
    """Before a user turn in the onboarding chat, while no card waits: which card is closed and how to
    bring it back. Rides the turn's API-only notes, like the first-contact note."""
    from agent.onboarding_flow import is_active
    from agent.onboarding_record import load
    from agent.onboarding_text import turn_note
    from tui_gateway import server_requests
    if history_empty or not _builds_onboarding(agent) or server_requests.pending_kind(sid) == "setup_beat":
        return
    record = load(agent._session_db, agent.session_id)
    if is_active(record) and (note := turn_note(record)):
        prior = getattr(agent, "_gateway_turn_context_notes", "") or ""
        agent._gateway_turn_context_notes = f"{prior}\n\n{note}" if prior else note


def register(server) -> None:
    bind_module(globals(), server, skip=("_",))
