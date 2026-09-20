"""Inference proxy (spec §12.2): the only thing an episode can talk to.

  * single-use bearer tokens per episode — issued by the host, revoked when the boundary is destroyed
  * the era's pinned sampling parameters OVERRIDE whatever the request carries (a bundle cannot change them)
  * per-call `usage` recorded by episode, independent of the agent's self-report
  * SSE streaming passed through, with `stream_options.include_usage` injected so usage is still captured
  * a call the engine refuses for lack of capacity is **held and retried**, not failed: the engine serves two long
    contexts, and a refusal used to void the whole episode — every turn it had already paid for
  * an episode's **token budget** (prompt + completion, as the engine counts them) is enforced here: once spent,
    the call is answered `402` and the episode ends; the grader reads the marker, not the agent's wording

    python -m sh.validator.proxy --listen 0.0.0.0:8080 --upstream http://127.0.0.1:8080 --tokens /run/sh/tokens.json --usage-dir /var/sh/usage
The token store is a directory holding one file per live token; the host creates and removes them, and the proxy
reads only the one file a request names, so issuing many tokens at once cannot lose any of them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# The era's pinned sampling (docs/pins.md, "sampling"). The default IS the pin: a proxy started without flags must
# not fall back to anything else — one did, and ran a round at the retired temperature 0.2.
PINNED_SAMPLING = {"temperature": 0.7, "top_p": 0.95, "max_tokens": 8192}
OVERLOAD_WAIT_S = 240  # how long a refused call is held before the refusal is passed on
MAX_WAITING = 2  # calls of one episode allowed to queue behind its call in flight; more are refused at once
_UNPINNED = ("n", "best_of", "logprobs", "top_logprobs", "echo")  # knobs that multiply what one call costs the engine
_OVERLOAD_WORDS = (b"overloaded", b"no capacity", b"capacity")
_REFUSAL_PHRASES = (b"overloaded", b"no capacity")  # in a 200: the engine's words, where a completion should be


def refusal(payload: bytes) -> bool:
    """A 200 that is really a capacity refusal: the engine puts its overload message where the answer should be and
    generates nothing (in testing a fifth of a round's episodes were voided this way, unseen by the 429/503 hold). A
    completion that merely mentions capacity has completion tokens. For a stream, `payload` is the first data chunk."""
    if not any(p in payload.lower() for p in _REFUSAL_PHRASES):
        return False
    try:
        j = json.loads(payload)
    except ValueError:
        return True
    if not isinstance(j, dict):
        return False
    if j.get("error"):
        return True
    return int((j.get("usage") or {}).get("completion_tokens") or 0) == 0


def usage_split(usage: dict) -> tuple[int, int]:
    """`(action_tokens, reasoning_tokens)` from one engine usage blob.

    Reasoning is a subset of `completion_tokens` when the engine reports it (`reasoning_tokens` or
    `completion_tokens_details.reasoning_tokens`). Deliberation draws on its own allowance — the same argument
    the runner already makes for thinking-only turns — so those tokens do not consume the action budget. When
    the engine reports nothing (the current pin: no reasoning-token field), the split is `(prompt+completion, 0)`
    and the action budget behaves exactly as before.
    """
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    details = usage.get("completion_tokens_details")
    extra = details.get("reasoning_tokens") if isinstance(details, dict) else None
    reasoning = int(extra or usage.get("reasoning_tokens") or 0)
    reasoning = max(0, min(reasoning, completion))
    return prompt + completion - reasoning, reasoning


def spent_split(usage_file: Path) -> tuple[int, int]:
    """Action and reasoning tokens an episode has used so far, from the proxy's own record."""
    action = reasoning = 0
    try:
        with open(usage_file) as f:
            for line in f:
                u = json.loads(line).get("usage") if line.strip() else None
                if u:
                    a, r = usage_split(u)
                    action += a
                    reasoning += r
    except (OSError, ValueError):
        pass
    return action, reasoning


def spent(usage_file: Path) -> int:
    """Action tokens an episode has used so far. Reasoning is billed separately when the engine reports it."""
    return spent_split(usage_file)[0]


class Tokens:
    """A token store as a **directory with one file per token**.

    It began as a single JSON blob rewritten on every issue and revoke, which is a read-modify-write race: four
    episodes starting at once read the same blob and the last writer dropped the other three, so their agents met
    `401 no valid episode token` on their first call and the episodes died one turn in. Issue and revoke are now a
    single atomic filesystem operation each and there is nothing shared to lose.
    """

    def __init__(self, path: Path):
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, token: str) -> Path:
        # the filename is a digest, so a token never lands in a directory listing, a log line or an `ls`
        return self.dir / f"{hashlib.sha256(token.encode()).hexdigest()}.json"

    def record(self, token: str) -> dict | None:
        if not token:
            return None
        try:
            rec = json.loads(self._path(token).read_text())
        except (OSError, ValueError):
            return None
        if rec.get("expires", 0) < time.time():
            return None
        return rec

    def lookup(self, token: str) -> str | None:
        rec = self.record(token)
        return rec.get("episode") if rec else None

    def issue(self, episode: str, ttl: int, budget: int | None = None) -> str:
        tok = "sh_" + secrets.token_urlsafe(32)
        path = self._path(tok)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"episode": episode, "expires": time.time() + ttl, "budget": budget}))
        tmp.replace(path)  # atomic: a reader sees the whole record or no file at all
        return tok

    def revoke(self, token: str) -> None:
        self._path(token).unlink(missing_ok=True)

    def sweep(self) -> int:
        """Drop expired records. Not required for correctness — `lookup` checks the expiry itself."""
        n = 0
        for f in self.dir.glob("*.json"):
            try:
                if json.loads(f.read_text()).get("expires", 0) < time.time():
                    f.unlink(missing_ok=True)
                    n += 1
            except (OSError, ValueError):
                pass
        return n


class Gate:
    """One call at a time per episode. The budget is then exact — each call is checked against everything spent
    before it — and an agent firing requests in parallel with its own token cannot overshoot its budget or hold the
    proxy's threads and the engine's slots against other episodes: a few calls queue, the rest are refused."""

    def __init__(self):
        self._lock = threading.Lock()
        self._episodes: dict[str, tuple[threading.Lock, int]] = {}

    def enter(self, episode: str) -> threading.Lock | None:
        with self._lock:
            lock, waiting = self._episodes.get(episode, (threading.Lock(), 0))
            if waiting > MAX_WAITING:
                return None
            self._episodes[episode] = (lock, waiting + 1)
        lock.acquire()
        return lock

    def leave(self, episode: str, lock: threading.Lock) -> None:
        lock.release()
        with self._lock:
            held, waiting = self._episodes.get(episode, (lock, 1))
            if waiting <= 1:
                self._episodes.pop(episode, None)
            else:
                self._episodes[episode] = (held, waiting - 1)


def make_handler(upstream: str, tokens: Tokens, usage_dir: Path, sampling: dict):
    gate = Gate()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:  # quiet
            pass

        def _deny(self, code: int, msg: str):
            body = json.dumps({"error": msg}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            episode = tokens.lookup((self.headers.get("Authorization") or "").removeprefix("Bearer ").strip())
            if not episode:
                return self._deny(401, "no valid episode token")
            self._forward(b"", episode)

        def do_POST(self):
            rec = tokens.record((self.headers.get("Authorization") or "").removeprefix("Bearer ").strip())
            if not rec or not rec.get("episode"):
                return self._deny(401, "no valid episode token")
            episode = rec["episode"]
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n)
            lock = gate.enter(episode)
            if lock is None:
                return self._deny(429, "one call at a time per episode")
            try:
                self._post(raw, episode, rec.get("budget"))
            finally:
                gate.leave(episode, lock)

        def _post(self, raw: bytes, episode: str, budget):
            action, reasoning = spent_split(usage_dir / f"{episode}.jsonl")
            reasoning_budget = int(budget) * 2 if budget else 0
            if budget and (action >= budget or (reasoning_budget and reasoning >= reasoning_budget)):
                usage_dir.mkdir(parents=True, exist_ok=True)
                mark = {"spent": action, "budget": budget}
                if reasoning:
                    mark["reasoning_spent"] = reasoning
                    mark["reasoning_budget"] = reasoning_budget
                with open(usage_dir / f"{episode}.jsonl", "a") as f:
                    f.write(json.dumps({"t": time.time(), "budget_spent": mark}) + "\n")
                which = (
                    f"episode token budget spent: {action} of {budget} tokens"
                    if action >= budget
                    else f"episode reasoning budget spent: {reasoning} of {reasoning_budget} tokens"
                )
                body = json.dumps(
                    {
                        "error": {
                            "message": which,
                            "type": "insufficient_quota",
                            "code": "insufficient_quota",
                        }
                    }
                ).encode()
                self.send_response(402)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            try:
                body = json.loads(raw) if raw else {}
            except ValueError:
                return self._deny(400, "bad json")
            if isinstance(body, dict):
                body.update({k: v for k, v in sampling.items() if v is not None})  # pinned sampling wins
                for knob in _UNPINNED:
                    body.pop(knob, None)
                if body.get("stream"):
                    so = body.get("stream_options") or {}
                    so["include_usage"] = True
                    body["stream_options"] = so
                raw = json.dumps(body).encode()
            self._forward(raw, episode)

        def _forward(self, raw: bytes, episode: str):
            t0, pause, data = time.time(), 2.0, b""
            while True:
                req = urllib.request.Request(
                    upstream + self.path,
                    data=raw if self.command == "POST" else None,
                    method=self.command,
                    headers={"Content-Type": "application/json", "Accept": self.headers.get("Accept", "*/*")},
                )
                try:
                    resp = urllib.request.urlopen(req, timeout=600)
                except urllib.error.HTTPError as e:
                    data = e.read()
                    refused = e.code in (429, 503) or any(w in data.lower() for w in _OVERLOAD_WORDS)
                    if refused and self.command == "POST" and time.time() - t0 < OVERLOAD_WAIT_S:
                        time.sleep(pause)  # the other episode's call will finish and free the engine
                        pause = min(pause * 1.5, 10.0)
                        continue
                    self.send_response(e.code)
                    self.send_header("Content-Type", e.headers.get("Content-Type", "application/json"))
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                except Exception as e:
                    return self._deny(502, f"upstream: {e!r}"[:200])
                ctype = resp.headers.get("Content-Type", "application/json")
                streaming = "text/event-stream" in ctype
                # The engine also refuses with a 200. Look before a byte reaches the client — the first data chunk
                # of a stream, the whole body otherwise — so that refusal is held and retried like a 503.
                head: list[bytes] = []
                if streaming:
                    for line in resp:
                        head.append(line)
                        if line.startswith(b"data:"):
                            break
                    peek = head[-1][5:].strip() if head and head[-1].startswith(b"data:") else b""
                else:
                    data = resp.read()
                    peek = data
                if self.command == "POST" and refusal(peek) and time.time() - t0 < OVERLOAD_WAIT_S:
                    resp.close()
                    time.sleep(pause)
                    pause = min(pause * 1.5, 10.0)
                    continue
                break
            queued = round(time.time() - t0, 1)

            def record(u):  # written BEFORE the client sees the end of the response, so a reader never races it
                if self.command != "POST" or not u:
                    return
                usage_dir.mkdir(parents=True, exist_ok=True)
                with open(usage_dir / f"{episode}.jsonl", "a") as f:
                    f.write(json.dumps({"t": time.time(), "usage": u, "queued_s": queued}) + "\n")

            if streaming:
                self.send_response(resp.status)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def emit(line: bytes) -> None:
                    if line.startswith(b"data:") and b'"usage"' in line:
                        try:
                            record(json.loads(line[5:].strip()).get("usage"))
                        except ValueError:
                            pass
                    self.wfile.write(f"{len(line):X}\r\n".encode() + line + b"\r\n")
                    self.wfile.flush()

                for line in head:  # what the peek read ahead
                    emit(line)
                for line in resp:
                    emit(line)
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            else:
                try:
                    record(json.loads(data).get("usage"))
                except ValueError:
                    pass
                self.send_response(resp.status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

    return H


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64

    def handle_error(self, request, client_address):
        """A client hanging up mid-response is ordinary; a traceback per occurrence buried the real errors."""
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError, TimeoutError)):
            return
        super().handle_error(request, client_address)


def serve(listen: str, upstream: str, tokens_path: Path, usage_dir: Path, sampling: dict):
    host, port = listen.rsplit(":", 1)
    return Server((host, int(port)), make_handler(upstream.rstrip("/"), Tokens(tokens_path), usage_dir, sampling))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="0.0.0.0:8080")
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--tokens", required=True, help="token store DIRECTORY (one file per live token)")
    ap.add_argument("--usage-dir", required=True)
    ap.add_argument("--sampling", default=json.dumps(PINNED_SAMPLING), help="override the pin (experiments only)")
    a = ap.parse_args(argv)
    sampling = json.loads(a.sampling)
    srv = serve(a.listen, a.upstream, Path(a.tokens), Path(a.usage_dir), sampling)
    print(f"proxy on {a.listen} -> {a.upstream} sampling={json.dumps(sampling, sort_keys=True)}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
