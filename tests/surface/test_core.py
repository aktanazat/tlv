"""The index a call names is the database the agent opens, even with calls on other indexes in
flight; and a jockey follow-up sees the earlier turns of its session. The agent is replaced by a
stand-in that records what it was given, since the real one needs the models."""

import os
import sys
import threading
import types

import pytest

import tlv
from tlv import core


@pytest.fixture
def agent(monkeypatch, home):
    fake = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "tlv.agent", fake)
    monkeypatch.setattr(tlv, "agent", fake, raising=False)
    return fake


def test_a_call_on_another_index_waits_while_calls_on_the_same_index_run_together(agent):
    entered = {name: threading.Event() for name in "abc"}
    release = threading.Event()
    seen = {}

    def search(query, **_):
        seen[query] = os.environ["TLV_INDEX"]
        entered[query].set()
        release.wait(30)
        return {"query": query, "then": [], "results": []}

    agent.search = search
    calls = [("games", "a"), ("practice", "b"), ("games", "c")]
    threads = {query: threading.Thread(target=core.video_search, args=(index, query)) for index, query in calls}
    threads["a"].start()
    assert entered["a"].wait(30)
    threads["b"].start()
    threads["c"].start()
    assert entered["c"].wait(30), "a call on the running index was held back"
    assert not entered["b"].wait(0.5), "a call on another index ran beside the running ones"
    release.set()
    for thread in threads.values():
        thread.join()
    assert seen == {"a": "games", "b": "practice", "c": "games"}


def test_a_follow_up_sees_the_earlier_turns_of_its_session(agent):
    given = []

    def answer(question, video_ids=None, progress=None, history=None):
        given.append(list(history))
        return {"text": f"answer to {question}", "status": "completed", "citations": [{"marker": "[1]"}]}

    agent.answer = answer
    first = core.video_jockey("tlv", "who scored first?")
    core.video_jockey("tlv", "and after that?", session_id=first["session_id"])
    core.video_jockey("tlv", "unrelated")
    first_turn = {
        "question": "who scored first?",
        "text": "answer to who scored first?",
        "citations": [{"marker": "[1]"}],
    }
    assert given == [[], [first_turn], []]


@pytest.mark.parametrize("session_id", ["0123456789ab", "../../etc/passwd"])
def test_an_unknown_session_is_refused(agent, session_id):
    with pytest.raises(ValueError, match="No session"):
        core.video_jockey("tlv", "and then?", session_id=session_id)
