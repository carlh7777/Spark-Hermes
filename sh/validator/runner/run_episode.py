"""Runs INSIDE the episode boundary as uid 1000. Everything it needs is under /ep (the per-episode
volume): task.json (public projection + bundle_sha256), bundle/ (the surface; empty for NULL),
seed. Everything it produces goes to /ep/out. It never sees the withheld check.

Steps (spec §5.3): copy the surface into HERMES_HOME and verify its digest; write config.yaml and
.env; materialise the fixture and run the family's setup_commands; digest protected paths; run
unmodified Hermes; write the result, the ShareGPT trajectory, home_writes.json and a snapshot tar.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

EP = Path(os.environ.get("SH_EP", "/ep"))
OUT = EP / "out"
WS = EP / "ws"
HOME = Path(os.environ["HERMES_HOME"])
MODEL = os.environ.get("SH_MODEL", "custom/qwen38-nvfp4")
INFERENCE = os.environ["SH_INFERENCE"]
TOKEN = os.environ.get("SH_TOKEN", "none")
SKILLS_TOOLSET = "skills"  # required for the skills index to render (V3); skill_manage neutralised by write_approval


def bundle_digest(root: Path) -> str:
    files = sorted(p for p in root.rglob("*") if p.is_file())
    lines = [f"{p.relative_to(root).as_posix()}\0{hashlib.sha256(p.read_bytes()).hexdigest()}" for p in files]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def digests(root: Path, paths) -> dict:
    return {
        rel: (hashlib.sha256((root / rel).read_bytes()).hexdigest() if (root / rel).is_file() else None)
        for rel in paths
    }


def sh(cmd: str, cwd: Path, timeout: int) -> tuple[int, str]:
    env = {
        "PATH": f"{cwd / 'bin'}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}",
        "HOME": str(HOME),
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    try:
        d = subprocess.run(
            ["/bin/sh", "-c", cmd],
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return d.returncode, (d.stdout + d.stderr)[-2000:]
    except subprocess.TimeoutExpired:
        return 124, "timed out"


def _reset_out() -> list[str]:
    """OUT as an empty directory the runner alone has written to. The agent runs as this uid and can reach /ep:
    a link planted at `/ep/out/before.json`, or `/ep/out` itself replaced, would otherwise steer or swallow what
    is written next. Whatever was there is returned — it is something the agent wrote outside its workspace."""
    stray: list[str] = []
    if OUT.is_symlink() or OUT.is_file():
        stray.append(str(OUT))
        OUT.unlink()
    elif OUT.is_dir():
        stray += [str(p) for p in OUT.iterdir() if p.name != "system_prompt.txt"]

        def _writable(func, path, _exc):  # a directory the agent made unreadable is still ours to remove
            os.chmod(path, 0o700)
            func(path)

        shutil.rmtree(OUT, onerror=_writable)
    OUT.mkdir(parents=True)
    return stray


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    finish = {"started": time.time(), "stage": "init"}
    before, ep_before, home_before = {}, {}, {}
    system_prompt = ""
    result, traj = None, None  # the runner's own outputs; written to OUT only after the agent's stray writes are read
    try:
        task = json.loads((EP / "task.json").read_text())
        global WS
        WS = Path(task.get("workdir") or WS)  # an image-defined task (family terminal_task) works in its own tree
        # /ep/seed is written by the host and deliberately not applied here: sampling, the seed included, is
        # pinned by the proxy so that no bundle and no runner change can alter it (spec §5.6). Wiring the
        # per-episode seed through the proxy is still open — it pins one sampling set for all episodes today.

        # 1. surface → HERMES_HOME (writable copy), digest-verified
        HOME.mkdir(parents=True, exist_ok=True)
        bundle = EP / "bundle"
        if bundle.is_dir():
            shutil.copytree(bundle, HOME, dirs_exist_ok=True)
        got = bundle_digest(bundle) if bundle.is_dir() else bundle_digest(Path(tempfile_empty()))
        if task.get("bundle_sha256") and got != task["bundle_sha256"]:
            raise RuntimeError(f"bundle digest mismatch: {got} != {task['bundle_sha256']}")
        finish["bundle_sha256"] = got

        # 2. config + env
        max_turns = int(task["max_turns"])
        # Deliberation draws on its own allowance, for the same reason verification does: max_turns is an
        # ACTION budget. Unmodified Hermes counts every API call against max_iterations, including a turn that
        # only thought, so a reasoning-only call would otherwise steal an action. Default 2× max_turns when
        # the task does not set one — bounded rather than free. A policy that bundles thinking into the same
        # call as a tool (the pin: <think> inside the gpt turn) does not spend the extra.
        max_reasoning_steps = (
            int(task["max_reasoning_steps"]) if task.get("max_reasoning_steps") is not None else max_turns * 2
        )
        max_iterations = max_turns + max_reasoning_steps
        (HOME / "config.yaml").write_text(
            "model:\n  default: %s\nproviders:\n  custom:\n    base_url: %s\n    api_key_env: SH_TOKEN\n"
            "agent:\n  max_turns: %d\nterminal:\n  backend: local\n  cwd: %s\n"
            "skills:\n  inline_shell: false\n  template_vars: false\n  auto_load: []\n  write_approval: true\n  guard_agent_created: true\n"
            % (MODEL, INFERENCE, max_iterations, WS)
        )
        (HOME / ".env").write_text(f"SH_TOKEN={TOKEN}\n")

        # 3. fixture, setup commands, pre-agent digests
        WS.mkdir(parents=True, exist_ok=True)
        for rel, f in task["fixture"].items():
            p = WS / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(base64.b64decode(f["b64"]))
            os.chmod(p, int(f.get("mode", "0644"), 8))
        for cmd in task.get("setup_commands", []):
            code, out = sh(cmd, WS, int(task["timeout_s"]))
            if code != 0:
                finish.update(stage="setup_failed", setup_error=out)
                raise SystemExit(0)
        # Held in MEMORY until the end: the agent runs as the same uid and could otherwise edit /ep/out/before.json.
        before = digests(WS, task["protected_paths"])
        ep_before = {
            str(p): (p.stat().st_mtime_ns, p.stat().st_size)
            for p in EP.rglob("*")
            if p.is_file() and WS not in p.parents and p != WS
        }
        home_before = {str(p): p.stat().st_mtime for p in HOME.rglob("*") if p.is_file()}
        os.chdir(WS)

        # 4–6. unmodified Hermes
        finish["stage"] = "agent"
        sys.path.insert(0, "/opt/hermes")
        from run_agent import AIAgent

        agent = AIAgent(
            model=MODEL,
            base_url=INFERENCE,
            api_key=TOKEN,
            max_iterations=max_iterations,
            enabled_toolsets=[*task["tools"], SKILLS_TOOLSET],
            skip_memory=True,
            skip_context_files=False,
            load_soul_identity=True,
            quiet_mode=True,
            platform="batch",
            max_tokens=8192,
        )
        system_prompt = agent._build_system_prompt()
        (OUT / "system_prompt.txt").write_text(system_prompt)
        t0 = time.time()
        result = agent.run_conversation(user_message=task["prompt"], task_id=task["task_id"])
        finish["agent_wall_s"] = round(time.time() - t0, 1)
        messages = result.get("messages", [])
        try:  # held in memory; written to OUT in the finally, after _reset_out reads what the agent left there
            traj = agent._convert_to_trajectory_format(messages, task["prompt"], bool(result.get("completed", True)))
        except Exception as e:
            finish["trajectory_error"] = repr(e)[:300]
        finish.update(
            stage="done",
            api_calls=int(
                result.get("api_calls")
                or sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant")
            ),
            tool_calls=sum(
                len(m.get("tool_calls") or []) for m in messages if isinstance(m, dict) and m.get("role") == "assistant"
            ),
            tool_errors=sum(
                1
                for m in messages
                if isinstance(m, dict) and m.get("role") == "tool" and '"success": false' in str(m.get("content"))
            ),
            completed=bool(result.get("completed")),
            partial=bool(result.get("partial")),
            failed=bool(result.get("failed")),
            turn_exit_reason=str(result.get("turn_exit_reason")),
            max_turns=max_turns,
            max_reasoning_steps=max_reasoning_steps,
            final_head=str(result.get("final_response", ""))[:300],
        )
    except SystemExit:
        pass
    except Exception:
        finish["stage"] = finish.get("stage", "?") + "_error"
        finish["error"] = traceback.format_exc()[-2000:]
    finally:
        # 7. what the agent wrote outside the workspace, and the snapshot. First, nothing it left running may
        # race what follows, and nothing it left in /ep/out survives.
        try:
            os.kill(-1, signal.SIGKILL)  # every other process of this uid: the agent's shells and their children
        except (ProcessLookupError, PermissionError, OSError):
            pass
        stray: list[str] = []
        try:
            stray = _reset_out()  # what the AGENT left in /ep/out — the runner has written nothing there yet
            (OUT / "system_prompt.txt").write_text(system_prompt)
            if result is not None:  # the runner's outputs, into the clean OUT: never mistaken for the agent's writes
                json.dump(result, open(OUT / "result.json", "w"), default=str)
            if traj is not None:
                json.dump(traj, open(OUT / "trajectory.json", "w"), default=str)
        except Exception as e:  # the outputs go wherever OUT now is; the grader sees what it sees
            finish["reset_error"] = repr(e)[:300]
        try:
            (OUT / "before.json").write_text(json.dumps(before))  # from memory, after the agent is gone
            writes = [
                str(p)
                for p in HOME.rglob("*")
                if p.is_file() and (str(p) not in home_before or p.stat().st_mtime != home_before[str(p)])
            ]
            (OUT / "home_writes.json").write_text(
                json.dumps(writes)
            )  # informational: HERMES_HOME is the agent's scratch
            ep_now = {
                str(p): (p.stat().st_mtime_ns, p.stat().st_size)
                for p in EP.rglob("*")
                if p.is_file() and WS not in p.parents and p != WS and OUT not in p.parents
            }
            ep_writes = sorted(
                set(ep_now) - set(ep_before)
                | {k for k in ep_now if k in ep_before and ep_now[k] != ep_before[k]}
                | set(stray)
            )
            (OUT / "ep_writes.json").write_text(
                json.dumps(ep_writes)
            )  # anything under /ep outside ws touched by the agent → DQ
        except Exception:
            pass
        if WS.exists():
            with tarfile.open(OUT / "snapshot.tar", "w") as tf:
                tf.add(WS, arcname=".")
        finish["ended"] = time.time()
        finish["wall_s"] = round(finish["ended"] - finish["started"], 1)
        (OUT / "finish.json").write_text(json.dumps(finish, indent=1))
    return 0


def tempfile_empty() -> str:
    import tempfile

    return tempfile.mkdtemp()


if __name__ == "__main__":
    sys.exit(main())
