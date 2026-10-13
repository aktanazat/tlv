"""Jev answers each state in the order asked, and whenever it cannot answer the caller gets a
JudgeError, never a raw network error, so it can keep its own checks. A stand-in for TypeSafe's API
runs on localhost: no key or network is needed."""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tlv import judge

KEY = "test-key"
QUESTION = judge.Question("Is `n` large?", yes="`n` is large.", no="`n` is small.")
LAST = 2


class FakeJev(BaseHTTPRequestHandler):
    """Answers like TypeSafe's endpoint, a yes probability of n / 10 for the state {"n": n}, and
    refuses a request it would refuse. The request for n = 0 is held until the one for n = LAST has
    been answered, so with the requests in flight together the answers finish out of order. Every
    state that reaches it is kept in `asked`."""

    server: "Server"

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.asked.append(body["state"])
        question = body["questions"]["q"]
        if self.headers["Authorization"] != f"Bearer {KEY}" or body["model"] != judge.MODEL:
            self._reply(401, {"error": "invalid API key"})
            return
        if question["type"] != "noul" or not question["instructions"]:
            self._reply(400, {"error": "bad question"})
            return
        n = body["state"]["n"]
        if n == 0:
            self.server.last_answered.wait(timeout=10)
        self._reply(200, {"answers": {"q": {"type": "noul", "noul": n / 10}}, "usage": {}})
        if n == LAST:
            self.server.last_answered.set()

    def _reply(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        """Quiet: the tests read answers, not the request log."""


class Server(ThreadingHTTPServer):
    last_answered: threading.Event
    asked: list[dict]


@pytest.fixture
def jev(monkeypatch: pytest.MonkeyPatch) -> Iterator[Server]:
    server = Server(("127.0.0.1", 0), FakeJev)
    server.last_answered = threading.Event()
    server.asked = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(judge, "URL", f"http://127.0.0.1:{server.server_port}/v1/systemone")
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)
    yield server
    server.shutdown()
    server.server_close()


def test_answers_come_back_in_the_order_asked(jev: Server) -> None:
    assert judge.bools(QUESTION, [{"n": n} for n in range(LAST + 1)]) == [0.0, 0.1, 0.2]


def test_without_a_key_nothing_is_sent(jev: Server, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY")
    with pytest.raises(judge.JudgeError, match="TYPESAFE_API_KEY"):
        judge.bools(QUESTION, [{"n": 1}])
    assert jev.asked == []


def test_a_refusal_names_its_status_and_reason(jev: Server, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "revoked-key")
    with pytest.raises(judge.JudgeError, match=r"HTTP 401 .*invalid API key"):
        judge.bools(QUESTION, [{"n": 1}])


def test_an_unreachable_api_is_a_judge_error(jev: Server) -> None:
    jev.shutdown()
    jev.server_close()
    with pytest.raises(judge.JudgeError, match="could not be reached"):
        judge.bools(QUESTION, [{"n": 1}])
