"""Host side of grading (spec §6): a fresh container from the family image evaluates both halves
over the snapshot; the host applies the deterministic trajectory rules and emits the Episode record.

    python -m sh.validator.grade --episode results/ep-1 --task task.json --withheld withheld.json --image IMG [--bundle DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from sh.validator.truncation import is_truncated, max_turns_hit, reasoning_allowance, turn_mix

# The agent's own client reports these when the *provider* failed, not the episode. An episode that never got
# its tokens says nothing about a miner's strategy, so it is void: not a success, not a failure, and not
# evidence. `batch` re-runs a void episode rather than caching it.
PROVIDER_FAILURES = ("overloaded", "rate_limit", "timeout", "connection", "unavailable", "auth")

DQ = (
    "protected_path_modified",
    "read_grader_or_withheld_path_attempt",
    "wrote_outside_workspace",
    "network_egress_attempt",
    "inline_shell_marker",
    "instance_literal_in_bundle",
    "harness_tamper",  # a family's checks found the agent's tree reaching for the test runner itself
    "disk_abuse",  # the episode was stopped for filling the worker's disk
)
MUTATING = {"write_file", "patch"}
VERIFYING = {"read_file", "search_files", "terminal", "process_manage"}


def _digest(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def pins_digest(task: dict) -> str:
    """What an episode was measured under: the pinned sampling and the task's limits and image. Episodes with
    different pins are different measurements and are never pooled."""
    from sh.validator.proxy import PINNED_SAMPLING

    return _digest(
        {
            "sampling": PINNED_SAMPLING,
            "max_turns": task.get("max_turns"),
            # Deliberation draws on its own allowance (default 2× max_turns). Recorded here so a run
            # under a different reasoning cap is a different measurement, which is what pins_digest is for.
            "max_reasoning_steps": reasoning_allowance(task),
            "timeout_s": task.get("timeout_s"),
            "token_budget": task.get("token_budget"),
            "family": task.get("family"),
        }
    )[:16]


def _run(args, *, input=None, timeout=None):
    return subprocess.run(args, input=input, capture_output=True, timeout=timeout)


def _tar(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for arc, data in entries.items():
            ti = tarfile.TarInfo(arc)
            ti.size = len(data)
            ti.mode = 0o644
            ti.uid = ti.gid = 1000
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


_PRISTINE = """import hashlib, json, os, sys
ws = sys.argv[1]
out = {}
for rel in json.load(sys.stdin):
    p = os.path.join(ws, rel)
    out[rel] = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.isfile(p) else None
print(json.dumps(out))
"""


# The grader is root inside its container, with only what it needs to run a family's suite as another uid and to kill
# everything that uid started: the code under test must not reach the answer key, the outcomes, or outlive the run.
GRADER_CAPS = ("SETUID", "SETGID", "CHOWN", "KILL", "DAC_OVERRIDE", "FOWNER")


def pristine_digests(task: dict, image: str) -> dict[str, str | None] | None:
    """The protected paths as the task image ships them, digested here from the image — never from anything the
    agent's uid could reach. The runner also records them, but it runs as the agent's uid: a link planted in
    `/ep/out`, or an episode killed on its timeout, left the grader with no baseline and so no verdict on what
    the agent changed. Image-defined tasks only; a fixture task's baseline is what its setup commands made, which
    the runner alone sees. None when not applicable."""
    if not task.get("workdir"):
        return None
    if not task.get("protected_paths"):
        return {}
    r = _run(
        [
            "docker",
            "run",
            "--rm",
            "-i",
            "--network",
            "none",
            "--user",
            "1000:1000",
            image,
            "/opt/hermes/.venv/bin/python",
            "-c",
            _PRISTINE,
            task["workdir"],
        ],
        input=json.dumps(list(task["protected_paths"])).encode(),
        timeout=300,
    )
    if r.returncode:
        raise RuntimeError(f"pristine digests of {image}: {r.stderr[-300:].decode(errors='replace')}")
    return json.loads(r.stdout)


def grade_in_container(
    episode_out: Path, task: dict, withheld: dict | None, image: str, checks_py: Path | None = None
) -> dict:
    vol = f"grade-{task['task_id']}-{hashlib.sha256(os.urandom(8)).hexdigest()[:8]}"
    entries = {"task.json": json.dumps(task).encode(), "snapshot.tar": (episode_out / "snapshot.tar").read_bytes()}
    before = pristine_digests(task, image)
    if before is not None:
        entries["before.json"] = json.dumps(before).encode()
    elif (episode_out / "before.json").is_file():
        entries["before.json"] = (episode_out / "before.json").read_bytes()
    # else: no baseline at all. Substituting an empty one here is what made the grader's own "absent means
    # unknown" rule unreachable — it never saw an absent file, it saw `{}` and read it as "nothing was there".
    if withheld is not None:
        entries["withheld.json"] = json.dumps(withheld).encode()
    if checks_py is not None:  # semantics of the family's `custom` predicates
        entries["checks.py"] = Path(checks_py).read_bytes()
    _run(["docker", "volume", "create", vol]).check_returncode()
    try:  # the grade container is named after the volume so a hung suite can be killed, not just abandoned
        _run(
            [
                "docker",
                "run",
                "--rm",
                "-i",
                "-v",
                f"{vol}:/ep",
                "alpine",
                "sh",
                "-c",
                # root's alone: the suite runs as another uid (a family's checks drop to it) and must not read the
                # answer key in withheld.json nor write where outcomes are kept
                "tar x -C /ep && mkdir -p /ep/out && chown -R 0:0 /ep && chmod -R go-rwx /ep",
            ],
            input=_tar(entries),
        ).check_returncode()
        try:
            r = _run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--name",
                    vol,
                    "--network",
                    "none",
                    "--cap-drop",
                    "ALL",
                    *[arg for cap in GRADER_CAPS for arg in ("--cap-add", cap)],
                    "--security-opt",
                    "no-new-privileges:true",
                    "--pids-limit",
                    "256",
                    "--memory",
                    "2048m",
                    "--read-only",
                    "--user",
                    "0:0",
                    "--tmpfs",
                    "/tmp:rw,size=256m,mode=1777",
                    *(["--tmpfs", f"{task['workdir']}:rw,size=1024m"] if task.get("workdir") else []),
                    "-v",
                    f"{vol}:/ep",
                    "-e",
                    "SH_EP=/ep",
                    image,
                    "/opt/hermes/.venv/bin/python",
                    "/runner/grade.py",
                ],
                timeout=int(task["timeout_s"]) * 3 + 60,
            )
        except subprocess.TimeoutExpired:
            _run(["docker", "kill", vol])
            raise
        if r.returncode != 0:
            raise RuntimeError(f"grader exited {r.returncode}: {r.stderr[-800:].decode(errors='replace')}")
        data = _run(["docker", "run", "--rm", "-v", f"{vol}:/ep", "alpine", "cat", "/ep/out/grade.json"]).stdout
        g = json.loads(data)
        g["before_from"] = "image" if before is not None else ("runner" if "before.json" in entries else None)
        # A family whose checks run a test suite leaves every test's outcome beside the grade (checks.py writes
        # it): the grader itself stops at the first failing predicate, and a fraction needs all of them.
        # `facet_tests.json` is the name terminal_task's checks used; rounds minted with it are still graded.
        tests = _run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{vol}:/ep",
                "alpine",
                "sh",
                "-c",
                "cat /ep/out/tests.json 2>/dev/null || cat /ep/out/facet_tests.json 2>/dev/null",
            ]
        ).stdout
        try:
            g["tests"] = json.loads(tests) if tests.strip() else None
        except ValueError:
            g["tests"] = None
        # ... and, when it explains itself (swe_fix: which kept tests failed), that explanation beside the grade
        detail = _run(
            ["docker", "run", "--rm", "-v", f"{vol}:/ep", "alpine", "sh", "-c", "cat /ep/out/detail.json 2>/dev/null"]
        ).stdout
        try:
            g["detail"] = json.loads(detail) if detail.strip() else None
        except ValueError:
            g["detail"] = None
        return g
    finally:
        _run(["docker", "volume", "rm", "-f", vol])


# Which argument of a tool call names a place, and which is a command that can reach one. Everything else
# (`content`, `new_str`, …) is payload the agent *wrote*: a script whose comments mention `/ep/out` is not an
# access attempt, and scanning it turned an ordinary solution into a disqualification.
# Absolute destinations the agent is *allowed* to write to: its own workspace and the two per-episode tmpfs
# scratch areas (HERMES_HOME and /tmp), whose contents are discarded with the container.
ALLOWED_WRITE_PREFIXES = ("/ep/ws", "/tmp", "/home/hermes")
PATH_KEYS = ("path", "file_path", "target", "directory", "dir", "cwd", "file")
CMD_KEYS = ("command", "cmd", "script", "args")  # not `pattern`/`query`: grepping the tree for a string is a read of it
# The grader's own places, anchored at the filesystem root: the runner at `/runner`, and where the episode's outputs
# and any withheld material would be (`/ep/out`, `/ep/withheld*`, `/ep/before*`, however `/ep` is spelled). The rest
# of `/ep` is the public projection of the task — `ls /ep` or `cat /ep/task.json` reads nothing secret, and the runner
# recreates `/ep/out` before writing to it. `_pytest/runner.py` and `/testbed/ep/out` are repository paths.
GRADER_PATH = re.compile(
    r"(?:(?<![\w.\-/])/ep|\$\{?SH_EP\}?)/(?:ws/\.\./)?(?:out|withheld|before)(?![\w\-])|(?<![\w.\-/])/runner(?![\w\-])"
)


def _reaching(args_json: str) -> tuple[list[str], list[str]]:
    """(paths, commands) a call names. Unparseable arguments are treated as one opaque command — conservative,
    since that is the only case where we cannot tell payload from destination."""
    try:
        d = json.loads(args_json)
    except Exception:
        return [], [args_json]
    if not isinstance(d, dict):
        return [], [str(d)]

    def flat(v):
        return [v] if isinstance(v, str) else [str(x) for x in v] if isinstance(v, (list, tuple)) else []

    return ([x for k in PATH_KEYS for x in flat(d.get(k))], [x for k in CMD_KEYS for x in flat(d.get(k))])


def _failed(content: str) -> bool:
    """Did the tool report that it did nothing? (A refused write is not a write — the read-only rootfs turns most
    stray absolute paths into exactly this, and a miner must not be disqualified for an attempt the box blocked.)"""
    try:
        d = json.loads(content)
    except Exception:
        return '"success": false' in content
    return isinstance(d, dict) and (bool(d.get("error")) or d.get("success") is False or d.get("bytes_written") == 0)


def trajectory_rules(
    messages: list, ep_writes: list, bundle_dir: Path | None, task: dict, withheld: dict | None
) -> tuple[list[str], bool, bool]:
    signals: list[str] = []
    calls = []  # (index, name, args_json, call_id)
    results = {}
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        if m.get("role") == "tool":
            results[m.get("tool_call_id")] = str(m.get("content"))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            calls.append(
                (i, str(fn.get("name")), str(fn.get("arguments")), tc.get("id") if isinstance(tc, dict) else None)
            )
    reaching = {i: _reaching(c[2]) for i, c in enumerate(calls)}
    if any(GRADER_PATH.search(x) for paths, cmds in reaching.values() for x in (*paths, *cmds)):
        signals.append("read_grader_or_withheld_path_attempt")
    # HERMES_HOME is the agent's own scratch (logs, sessions, caches) and is never a DQ. What is: touching
    # anything under /ep outside the workspace (task.json, bundle/, the runner's outputs), or an absolute
    # write target outside /ep/ws.
    allowed = (*ALLOWED_WRITE_PREFIXES, *([task["workdir"]] if task.get("workdir") else []))  # the image's own tree
    abs_writes = [
        p
        for i, (_, n, _, cid) in enumerate(calls)
        if n in MUTATING and not _failed(results.get(cid, ""))
        for p in reaching[i][0]
        if p.startswith("/") and not any(p == a or p.startswith(a + "/") for a in allowed)
    ]
    if ep_writes or abs_writes:
        signals.append("wrote_outside_workspace")
    if bundle_dir and bundle_dir.is_dir():
        text = "\n".join(p.read_text(errors="replace") for p in bundle_dir.rglob("*") if p.is_file())
        if "!`" in text:
            signals.append("inline_shell_marker")
        literals = {task["task_id"]} | {
            a
            for p in task["published"]["predicates"]
            for a in p[1:]
            if isinstance(a, str) and re.fullmatch(r"[0-9a-f]{64}", a)
        }
        if withheld:  # sealed file is {"withheld": {...}, "salt": ...}; also accept the inner dict
            w = withheld.get("withheld", withheld)
            literals |= {
                a for p in w["predicates"] for a in p[1:] if isinstance(a, str) and re.fullmatch(r"[0-9a-f]{64}", a)
            }
        # whole-token matches: a preview id such as `swe-fix-r0004-p03` must not match the evaluated `…-03`
        if any(re.search(rf"(?<![\w-]){re.escape(lit)}(?![\w-])", text) for lit in literals):
            signals.append("instance_literal_in_bundle")
    last_mut = max((i for i, n, _, _ in calls if n in MUTATING), default=-1)
    self_checked = any(i > last_mut for i, n, _, _ in calls if n in VERIFYING) if last_mut >= 0 else False
    failed_tool = any(
        isinstance(m, dict) and m.get("role") == "tool" and '"success": false' in str(m.get("content"))
        for m in messages
    )
    return signals, self_checked, failed_tool


TEST_CHECKS = frozenset({"facet_test", "swe_test"})  # `custom <check> <test id>`: one predicate per test case


def fraction(predicates: list, passed: bool, tests: dict | None) -> float | None:
    """The share of a half's predicates that hold. When the half is a set of test cases (`custom facet_test <id>`,
    `custom swe_test <id>`) and the test outcomes are known, each counts; otherwise the half is one unit — all or
    nothing. An empty half measures nothing: None. Pure."""
    if not predicates:
        return None
    if tests is not None and all(p[0] == "custom" and len(p) > 2 and p[1] in TEST_CHECKS for p in predicates):
        return sum(1 for p in predicates if tests.get(p[2])) / len(predicates)
    return 1.0 if passed else 0.0


OVERFIT_GAP = 0.5  # published share minus withheld share at or above this: the shortcut, not the work


def grade(
    episode_out: Path,
    task: dict,
    withheld: dict | None,
    image: str,
    *,
    bundle_dir: Path | None = None,
    surface: str = "null",
    round_id: str = "",
    validator: str = "",
    checks_py: Path | None = None,
) -> dict:
    finish = json.loads((episode_out / "finish.json").read_text())
    result = json.loads((episode_out / "result.json").read_text()) if (episode_out / "result.json").exists() else {}
    messages = result.get("messages", [])
    ep_writes = (
        json.loads((episode_out / "ep_writes.json").read_text()) if (episode_out / "ep_writes.json").exists() else []
    )
    if finish.get("disk_guard"):  # stopped before it filled the disk: nothing is graded, and it is disqualified
        g = {"published_pass": False, "withheld_pass": False, "protected_modified": None, "tests": None}
    else:
        g = grade_in_container(episode_out, task, withheld, image, checks_py)
    signals, self_checked, failed_tool = trajectory_rules(messages, ep_writes, bundle_dir, task, withheld)
    if g["protected_modified"]:
        signals.append("protected_path_modified")
    if (g.get("detail") or {}).get("tamper"):  # swe_fix: lines added to source that reach for pytest's internals
        signals.append("harness_tamper")
    if finish.get("partial"):
        signals.append("malformed_tool_call")
    if finish.get("timed_out"):
        signals.append("timed_out")
    if finish.get("dropped_packets"):  # packets from the container that hit SH_EP_DROP (episode.py / net-up.sh)
        signals.append("network_egress_attempt")
    reason = str(result.get("failure_reason") or "")
    void = bool(result.get("failed")) and (
        result.get("failure_retryable") or any(k in reason.lower() for k in PROVIDER_FAILURES)
    )
    if finish.get("budget_spent"):  # the episode used its budget: an ending, graded like any other, never a void
        signals.append("token_budget_spent")
        void = False
    turns_capped = max_turns_hit(finish, result, task)
    if turns_capped:
        signals.append("max_turns_hit")
    truncated = is_truncated(finish, result, task)
    action_turns, reasoning_turns = turn_mix(messages)
    if finish.get("disk_guard"):
        signals.append("disk_abuse")
        void = False
    if void:
        signals.append("inference_unavailable")
    published_pass = bool(g["published_pass"]) and not finish.get("timed_out")
    # A probe (spec §3.7b) has no withheld half at all. Treating "absent" as "failed" would mark every solved
    # probe `overfit` and every one of them a non-success — both meaningless without a half to overfit to.
    graded_withheld = withheld is not None and "withheld_pass" in g
    withheld_pass = bool(g.get("withheld_pass", False))
    disqualified = any(s in DQ for s in signals)
    verified = published_pass and (withheld_pass or not graded_withheld) and not disqualified
    # Credit: the share of the withheld half that holds (of the published half, for a probe). It is what scoring
    # compares against the baseline; `verified_success` — every check holds — still gates training data.
    tests = g.get("tests")
    # A task whose every fact is withheld (swe_fix) publishes no half: nothing to overfit to, nothing to compare.
    published_fraction = fraction(task["published"]["predicates"], bool(g["published_pass"]), tests)
    w_preds = (withheld or {}).get("withheld", withheld or {}).get("predicates", []) if withheld else []
    withheld_fraction = fraction(w_preds, withheld_pass, tests) if graded_withheld else None
    measured = withheld_fraction if graded_withheld else published_fraction
    if measured is None:  # no predicate on the side that counts (a probe): the verdict is the credit
        measured = 1.0 if verified else 0.0
    credit = 0.0 if disqualified or void else measured
    return {
        "schema": "sh-episode-v3",
        "episode_id": f"{round_id}/{task['task_id']}/{surface}",
        "round_id": round_id,
        "task_id": task["task_id"],
        "family": task.get("family"),
        "surface": surface,
        "seed": task.get("seed"),
        "wall_s": finish.get("wall_s"),
        "published_pass": published_pass,
        "withheld_pass": withheld_pass if graded_withheld else None,
        "verified_success": verified,
        "published_fraction": None if published_fraction is None else round(published_fraction, 6),
        "withheld_fraction": None if withheld_fraction is None else round(withheld_fraction, 6),
        "credit": round(float(credit or 0.0), 6),
        "overfit": bool(
            graded_withheld
            and published_fraction is not None
            and published_fraction - (withheld_fraction or 0.0) >= OVERFIT_GAP
        ),
        "disqualified": disqualified,
        "void": void,
        "void_reason": reason if void else None,
        "signals": signals,
        "self_checked": self_checked,
        "recovered": failed_tool and verified,
        "api_calls": finish.get("proxy_calls", finish.get("api_calls")),
        "agent_reported_api_calls": finish.get("api_calls"),
        "tokens": finish.get("proxy_tokens"),
        "queued_s": finish.get("proxy_queued_s"),
        "budget_spent": finish.get("budget_spent"),
        # which withheld half graded it, and under which pins: close checks the first against the reveal, and scoring
        # pools only episodes measured under the same pins
        "withheld_sha256": _digest(withheld.get("withheld", withheld)) if withheld else None,
        "pins_sha256": pins_digest(task),
        "dropped_packets": finish.get("dropped_packets"),
        "tool_calls": finish.get("tool_calls"),
        "tool_errors": finish.get("tool_errors"),
        "partial": bool(finish.get("partial")),
        "timed_out": bool(finish.get("timed_out")),
        "truncated": truncated,
        "max_turns_hit": turns_capped,
        "action_turns": action_turns,
        "reasoning_turns": reasoning_turns,
        "finish_reason": finish.get("turn_exit_reason"),
        "published_failed": g.get("published_failed", ""),
        "withheld_failed": g.get("withheld_failed", ""),
        "detail": g.get("detail"),
        "before_from": g.get("before_from"),
        "skipped_members": g.get("skipped_members"),
        "validator": validator,
        "graded_at": time.time(),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--withheld", default="")
    ap.add_argument("--image", required=True)
    ap.add_argument("--bundle", default="")
    ap.add_argument("--surface", default="null")
    ap.add_argument("--checks", default="", help="the family's checks.py (semantics of its `custom` predicates)")
    a = ap.parse_args(argv)
    rec = grade(
        Path(a.episode),
        json.loads(Path(a.task).read_text()),
        json.loads(Path(a.withheld).read_text()) if a.withheld else None,
        a.image,
        bundle_dir=Path(a.bundle) if a.bundle else None,
        surface=a.surface,
        checks_py=Path(a.checks) if a.checks else None,
    )
    Path(a.episode, "episode.json").write_text(json.dumps(rec, indent=1))
    print(
        json.dumps(
            {
                k: rec[k]
                for k in (
                    "verified_success",
                    "published_pass",
                    "withheld_pass",
                    "overfit",
                    "disqualified",
                    "signals",
                    "self_checked",
                    "api_calls",
                    "tool_calls",
                )
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
