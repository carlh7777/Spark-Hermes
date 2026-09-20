"""The proxy on one GPU: the pinned sampling is its own default, a refused call is held rather than failed, and an
episode's token budget ends it cleanly."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import sh.validator.proxy as P


def _engine(script):
    """A fake engine answering POSTs from `script`: a list of (status, body) consumed in order."""
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            status, body = script.pop(0) if script else (200, {"usage": {"prompt_tokens": 1, "completion_tokens": 1}})
            data = body if isinstance(body, bytes) else json.dumps(body).encode()  # bytes: a raw SSE stream
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream" if isinstance(body, bytes) else "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


def _proxy(tmp_path, upstream):
    tokens = P.Tokens(tmp_path / "tokens")
    srv = P.serve("127.0.0.1:0", upstream, tmp_path / "tokens", tmp_path / "usage", P.PINNED_SAMPLING)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, tokens


def _post(port, token, body=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body or {"messages": [], "temperature": 0.0}).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_the_pinned_sampling_is_the_proxys_own_default_and_overrides_the_request(tmp_path):
    """A proxy started without --sampling ran r0003–r0004 at the retired temperature 0.2."""
    pins = (__import__("pathlib").Path(__file__).parents[1] / "docs" / "pins.md").read_text()
    assert P.PINNED_SAMPLING == {"temperature": 0.7, "top_p": 0.95, "max_tokens": 8192}
    assert "`temperature 0.7`**, `top_p 0.95`, `max_tokens 8192`" in pins
    engine, seen = _engine([])
    proxy, tokens = _proxy(tmp_path, f"http://127.0.0.1:{engine.server_address[1]}")
    status, _ = _post(proxy.server_address[1], tokens.issue("ep-1", ttl=60))
    assert status == 200 and seen[0]["temperature"] == 0.7 and seen[0]["top_p"] == 0.95


def test_a_call_the_engine_refuses_for_capacity_is_held_and_retried_not_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    engine, seen = _engine(
        [
            (503, {"error": "server overloaded: no capacity for this request right now"}),
            (503, {"error": "server overloaded: no capacity for this request right now"}),
            (200, {"usage": {"prompt_tokens": 100, "completion_tokens": 5}}),
        ]
    )
    proxy, tokens = _proxy(tmp_path, f"http://127.0.0.1:{engine.server_address[1]}")
    status, body = _post(proxy.server_address[1], tokens.issue("ep-2", ttl=60))
    assert status == 200 and len(seen) == 3
    row = json.loads((tmp_path / "usage" / "ep-2.jsonl").read_text().splitlines()[0])
    assert row["usage"]["prompt_tokens"] == 100 and "queued_s" in row


def test_an_ordinary_client_error_is_passed_through_at_once(tmp_path, monkeypatch):
    monkeypatch.setattr(P.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("must not wait")))
    engine, seen = _engine([(400, {"error": "bad request: messages is empty"})])
    proxy, tokens = _proxy(tmp_path, f"http://127.0.0.1:{engine.server_address[1]}")
    status, _ = _post(proxy.server_address[1], tokens.issue("ep-3", ttl=60))
    assert status == 400 and len(seen) == 1


def test_a_spent_token_budget_ends_the_episode_with_a_402_and_a_marker(tmp_path):
    engine, seen = _engine([(200, {"usage": {"prompt_tokens": 900, "completion_tokens": 150}})])
    proxy, tokens = _proxy(tmp_path, f"http://127.0.0.1:{engine.server_address[1]}")
    tok = tokens.issue("ep-4", ttl=60, budget=1000)
    assert _post(proxy.server_address[1], tok)[0] == 200  # 0 spent: forwarded, 1050 recorded
    status, body = _post(proxy.server_address[1], tok)
    assert status == 402 and "budget spent" in body["error"]["message"] and len(seen) == 1
    marks = [json.loads(line) for line in (tmp_path / "usage" / "ep-4.jsonl").read_text().splitlines()]
    assert marks[-1]["budget_spent"] == {"spent": 1050, "budget": 1000}
    assert tokens.issue("ep-5", ttl=60) and _post(proxy.server_address[1], tokens.issue("ep-5", ttl=60))[0] == 200


def test_reasoning_tokens_do_not_spend_the_action_budget(tmp_path):
    """Deliberation draws on its own allowance. When the engine reports reasoning_tokens as a subset of
    completion, those tokens must not 402 the action budget — otherwise the harness penalises thinking, which
    is the same defect the runner's max_reasoning_steps exists to stop at the turn counter."""
    engine, seen = _engine(
        [
            (
                200,
                {
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 400,
                        "completion_tokens_details": {"reasoning_tokens": 350},
                    }
                },
            ),
            (200, {"usage": {"prompt_tokens": 10, "completion_tokens": 10}}),
        ]
    )
    proxy, tokens = _proxy(tmp_path, f"http://127.0.0.1:{engine.server_address[1]}")
    tok = tokens.issue("ep-r", ttl=60, budget=200)
    assert _post(proxy.server_address[1], tok)[0] == 200  # action 150, reasoning 350; action budget 200 still open
    status, _ = _post(proxy.server_address[1], tok)
    assert status == 200 and len(seen) == 2  # second call forwarded; reasoning did not trip the 402
    assert P.usage_split(
        {"prompt_tokens": 100, "completion_tokens": 400, "completion_tokens_details": {"reasoning_tokens": 350}}
    ) == (150, 350)


def test_a_reasoning_only_policy_still_has_to_stop(tmp_path):
    """The allowance is bounded, not free. 2× the action budget, then 402."""
    engine, seen = _engine(
        [
            (
                200,
                {
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 500,
                        "reasoning_tokens": 500,
                    }
                },
            )
        ]
    )
    proxy, tokens = _proxy(tmp_path, f"http://127.0.0.1:{engine.server_address[1]}")
    tok = tokens.issue("ep-think", ttl=60, budget=100)  # reasoning cap 200
    assert _post(proxy.server_address[1], tok)[0] == 200  # first call records 500 reasoning
    status, body = _post(proxy.server_address[1], tok)
    assert status == 402 and "reasoning budget spent" in body["error"]["message"] and len(seen) == 1


def test_a_spent_budget_is_an_ending_never_a_void(monkeypatch, tmp_path):
    import sh.validator.grade as g

    (tmp_path / "result.json").write_text(
        json.dumps({"messages": [], "failed": True, "failure_reason": "billing", "failure_retryable": False})
    )
    (tmp_path / "finish.json").write_text(
        json.dumps({"api_calls": 40, "wall_s": 900.0, "budget_spent": {"spent": 612000, "budget": 600000}})
    )
    monkeypatch.setattr(g, "grade_in_container", lambda *a, **k: {"published_pass": True, "protected_modified": []})
    rec = g.grade(tmp_path, {"task_id": "t-1", "published": {"predicates": []}}, None, "img")
    assert not rec["void"] and "token_budget_spent" in rec["signals"] and rec["budget_spent"]["budget"] == 600000


REFUSED = {
    "choices": [
        {"message": {"role": "assistant", "content": "server overloaded: no capacity for this request right now"}}
    ],
    "usage": {"prompt_tokens": 40, "completion_tokens": 0},
}


def test_a_refusal_wearing_a_200_is_held_and_retried_like_a_503(tmp_path):
    """The engine refuses with a 200 — its overload message where the answer should be, no completion tokens — and
    the 429/503 hold never saw it: a fifth of r0004–r0006's episodes were voided that way, each one a re-run."""
    engine, seen = _engine([(200, REFUSED), (200, {"usage": {"prompt_tokens": 40, "completion_tokens": 3}})])
    proxy, tokens = _proxy(tmp_path, f"http://127.0.0.1:{engine.server_port}")
    tok = tokens.issue("ep-r", 60)
    status, body = _post(proxy.server_port, tok)
    assert status == 200 and body["usage"]["completion_tokens"] == 3 and len(seen) == 2
    rows = [json.loads(line) for line in (tmp_path / "usage" / "ep-r.jsonl").read_text().splitlines()]
    assert [r["usage"]["completion_tokens"] for r in rows] == [3]  # the refusal spent none of the budget


def test_a_refused_stream_is_retried_before_a_byte_reaches_the_agent(tmp_path):
    refused = b'data: {"choices":[{"delta":{"content":"server overloaded: no capacity"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    answered = (
        b'data: {"choices":[{"delta":{"role":"assistant","content":null}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"fixed"}}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":40,"completion_tokens":2}}\n\ndata: [DONE]\n\n'
    )
    engine, seen = _engine([(200, refused), (200, answered)])
    proxy, tokens = _proxy(tmp_path, f"http://127.0.0.1:{engine.server_port}")
    tok = tokens.issue("ep-s", 60)
    req = urllib.request.Request(
        f"http://127.0.0.1:{proxy.server_port}/v1/chat/completions",
        data=json.dumps({"messages": [], "stream": True}).encode(),
        headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        out = r.read()
    assert b"fixed" in out and b"overloaded" not in out and len(seen) == 2
    rows = [json.loads(line) for line in (tmp_path / "usage" / "ep-s.jsonl").read_text().splitlines()]
    assert [r["usage"]["completion_tokens"] for r in rows] == [2]


def test_a_completion_that_mentions_capacity_is_not_a_refusal():
    from sh.validator.proxy import refusal

    assert refusal(json.dumps(REFUSED).encode())
    assert refusal(b'{"error": {"message": "server overloaded"}}')
    assert not refusal(
        json.dumps(
            {"choices": [{"message": {"content": "no capacity left"}}], "usage": {"completion_tokens": 4}}
        ).encode()
    )
    assert not refusal(b'{"choices":[{"delta":{"role":"assistant","content":null}}]}')  # an ordinary first chunk
