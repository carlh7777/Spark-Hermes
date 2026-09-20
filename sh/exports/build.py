"""Training data out of a closed round (spec §8).

The competition exists to produce this. Every row is an episode that a **withheld** half verified, so the data
says "this trajectory actually solved the task", not "this trajectory looked right".

  * **SFT** — one row per verified, non-disqualified, **non-truncated** episode of the crowned strategy, in the
    converter's `{from, value}` shape. The king's trajectories are the round's product; every other surface's are
    evidence. A trajectory the harness cut off (turn cap, token budget, or the host clock) is not a model to
    imitate — its last step is the harness saying so.
  * **DPO** — a chosen/rejected pair per instance. Prefer a correctness pair: the king's best untruncated episode,
    doing at least 80 % of the withheld checks, against another surface doing at least 50 points fewer. When every
    surface passes, fall back to an efficiency pair: the same king against a verified surface that cost at least
    1.5× as many tokens. Across instances a pair would encode difficulty. A round with no king exports nothing,
    and the manifest says so.

Two rules from V4 govern the system turn, and they matter more than they look. The converter emits Hermes'
*generic* function-calling prompt, which is not what the agent actually ran under — so the export replaces it
with the NULL-surface prompt captured at the pin, and carries tool schemas in the row instead. Training on the
converter's own system turn would teach the model a prompt no deployment ever shows it.

    python -m sh.exports.build --round DIR --episodes DIR --close FILE --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from sh.scoring.v2 import credit
from sh.validator.grade import TEST_CHECKS
from sh.validator.truncation import episode_truncated

SCHEMA_SFT = "sh-sft-v2"
SCHEMA_DPO = "sh-dpo-v2"
DPO_CHOSEN_MIN = 0.8  # the preferred side must do most of the task
DPO_MARGIN = 0.5  # and the other side at least this much less of it
# Fallback when every surface on an instance passes: same task, same king, far fewer tokens. A round at a high
# pass rate has no correctness loser, so the only preference left is the one the promotion discussion actually
# scores — efficiency. Compared against the king's own cost, not a pooled median: we have one king episode per
# instance, not a distribution of repeats.
EFFICIENCY_MARGIN = 0.5


def _leak_scan(text: str, secrets: set[str]) -> list[str]:
    """Nothing withheld may leave in a training row: the withheld half is what makes future rounds gradeable."""
    return sorted({s for s in secrets if s and s in text})


def _secrets(reveal: dict) -> set[str]:
    """Everything withheld, as strings, so a row carrying any of it can be refused rather than uploaded. The ids
    of a withheld test suite are not secrets: they are the family's public dataset, and any honest `pytest -v`
    names them."""
    secrets: set[str] = set()
    for entry in reveal.values():
        secrets.add(entry.get("salt", ""))
        for predicate in entry.get("withheld", {}).get("predicates", []):
            if predicate[0] == "custom" and predicate[1:2] and predicate[1] in TEST_CHECKS:
                continue
            secrets.update(str(a) for a in predicate[1:] if isinstance(a, str) and len(str(a)) >= 16)
    return secrets


def _cost(episode: dict) -> int:
    """Tokens billed to an episode: proxy prompt+completion, else 0. Unpriced episodes cannot rank by cost."""
    tokens = episode.get("tokens")
    if isinstance(tokens, dict):
        return int(tokens.get("prompt_tokens") or 0) + int(tokens.get("completion_tokens") or 0)
    if isinstance(tokens, (int, float)) and not isinstance(tokens, bool):
        return int(tokens)
    return 0


def _rows_for(episode_dir: Path, episode: dict, task: dict, system_prompt: str) -> dict | None:
    trajectory = episode_dir / "trajectory.json"
    if not trajectory.exists():
        return None
    captured = episode_dir / "system_prompt.txt"  # what the episode actually ran under: the runner records it
    if captured.is_file():
        system_prompt = captured.read_text()
    turns = json.loads(trajectory.read_text())
    if not isinstance(turns, list) or not turns:
        return None
    # Rule 1 (V4): the converter's system turn is Hermes' generic function-calling prompt, not the prompt the
    # episode ran under. Replace it; keep everything else the converter produced.
    body = [t for t in turns if t.get("from") != "system"]
    return {
        "schema": SCHEMA_SFT,
        "task_id": episode.get("task_id"),
        "family": episode.get("family"),
        "round_id": episode.get("round_id"),
        "surface": episode.get("surface"),
        "conversations": [{"from": "system", "value": system_prompt}, *body],
        "tools": task.get("tools", []),  # Rule 2 (V4): schemas travel as a field, not as prose
        "verified_success": bool(episode.get("verified_success")),
        "api_calls": episode.get("api_calls"),
        "tool_calls": episode.get("tool_calls"),
        "self_checked": episode.get("self_checked"),
        "pins": episode.get("pins"),
    }


def build(
    round_dir: Path,
    episodes_dir: Path,
    close_file: Path,
    out: Path,
    *,
    king: str | None = None,
    system_prompt: str | None = None,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    tasks = {p.stem: json.loads(p.read_text()) for p in sorted((round_dir / "tasks").glob("*.json"))}
    closed = json.loads(close_file.read_text())

    reveal = close_file.parent / "reveal.json"
    secrets = _secrets(json.loads(reveal.read_text())) if reveal.exists() else set()

    sft, by_task = [], {}
    gates = {
        "not_king": 0,
        "void": 0,
        "no_trajectory": 0,
        "not_verified": 0,
        "disqualified": 0,
        "leaked": 0,
        "truncated": 0,
    }
    for episode_json in sorted(episodes_dir.rglob("episode.json")):
        episode = json.loads(episode_json.read_text())
        task = tasks.get(str(episode.get("task_id")), {})
        by_task.setdefault(episode.get("task_id"), []).append((episode, episode_json.parent))
        if king is None or episode.get("surface") != king:
            gates["not_king"] += 1
            continue
        if episode.get("void"):  # a provider outage is not a solved task, whatever its recorded flags
            gates["void"] = gates.get("void", 0) + 1
            continue
        if episode.get("disqualified"):
            gates["disqualified"] += 1
            continue
        if not episode.get("verified_success"):
            gates["not_verified"] += 1
            continue
        if episode_truncated(episode):
            # SFT is imitation. A trajectory the harness cut off ends mid-work, and its last recorded step is
            # the harness saying so. Training on it teaches stopping short.
            gates["truncated"] += 1
            continue
        row = _rows_for(
            episode_json.parent,
            episode,
            task,
            system_prompt or "You are Hermes, an agent operating a terminal and a filesystem.",
        )
        if row is None:
            gates["no_trajectory"] += 1
            continue
        if _leak_scan(json.dumps(row), secrets):
            gates["leaked"] += 1
            continue
        sft.append(row)

    # DPO: same instance, the king solved it and another surface did not. Across instances a pair would encode
    # difficulty rather than strategy, so pairs never cross a task_id.
    dpo = []
    for task_id, entries in by_task.items() if king else []:
        task = tasks.get(str(task_id), {})
        prompt = system_prompt or ""
        # Take the first side of each pair that actually yields a row. An episode killed on its timeout has no
        # trajectory at all, and picking only the first loser silently dropped every pair whose first loser
        # happened to be one of those — half the training value of the round, lost to list order.
        kings = [
            (credit(e), e, d)
            for e, d in entries
            if e.get("surface") == king
            and not e.get("disqualified")
            and credit(e) >= DPO_CHOSEN_MIN
            and not episode_truncated(e)  # a truncated pass is not the example: it stopped mid-work
        ]
        chosen = rejected = chosen_ep = None
        c_credit = 0.0
        for c_credit, e, d in sorted(kings, key=lambda x: -x[0]):
            if (chosen := _rows_for(d, e, task, prompt)) is not None:
                chosen_ep = e
                break
        kind = "correctness"
        if chosen is not None:
            losers = sorted(
                (
                    (credit(e), e, d)
                    for e, d in entries
                    if e.get("surface") != king and not e.get("void") and credit(e) <= c_credit - DPO_MARGIN
                ),
                key=lambda x: x[0],
            )
            rejected = next((row for _, e, d in losers if (row := _rows_for(d, e, task, prompt)) is not None), None)
            if rejected is None and chosen_ep and _cost(chosen_ep) > 0:
                # No correctness loser: every other surface also passed. Pair on cost instead, when the
                # rejected side is clearly more expensive than the king — not on a 2% sampling wobble.
                floor = _cost(chosen_ep) * (1 + EFFICIENCY_MARGIN)
                expensive = sorted(
                    (
                        (_cost(e), e, d)
                        for e, d in entries
                        if e.get("surface") != king
                        and not e.get("void")
                        and e.get("verified_success")
                        and _cost(e) >= floor
                    ),
                    key=lambda x: -x[0],
                )
                for _, e, d in expensive:
                    if (rejected := _rows_for(d, e, task, prompt)) is not None:
                        kind = "efficiency"
                        break
        if chosen and rejected:
            # both sides carry the king's system turn and the same task, so the preference is over the trajectory
            # alone and not over which strategy prompt produced it
            rejected_turns = [
                chosen["conversations"][0],
                *[t for t in rejected["conversations"] if t.get("from") != "system"],
            ]
            dpo.append(
                {
                    "schema": SCHEMA_DPO,
                    "task_id": task_id,
                    "family": chosen["family"],
                    "round_id": chosen["round_id"],
                    "prompt": task.get("prompt"),
                    "chosen": chosen["conversations"],
                    "rejected": rejected_turns,
                    "chosen_surface": chosen["surface"],
                    "rejected_surface": rejected["surface"],
                    "kind": kind,
                }
            )

    (out / "sft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sft))
    (out / "dpo.jsonl").write_text("".join(json.dumps(r) + "\n" for r in dpo))
    manifest = {
        "schema": "sh-export-manifest-v2",
        "round_id": closed.get("round_id"),
        "king": king,
        "sft_rows": len(sft),
        "dpo_pairs": len(dpo),
        "gates": gates,
        "families": sorted({r["family"] for r in sft if r.get("family")}),
        "sft_sha256": hashlib.sha256((out / "sft.jsonl").read_bytes()).hexdigest(),
        "dpo_sha256": hashlib.sha256((out / "dpo.jsonl").read_bytes()).hexdigest(),
        "leak_scan": {"secrets_checked": len(secrets), "rows_refused": gates["leaked"]},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--close", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--king", help="the crowned hotkey; without one nothing is exported")
    ap.add_argument("--system-prompt")
    a = ap.parse_args(argv)
    prompt = Path(a.system_prompt).read_text() if a.system_prompt else None
    manifest = build(Path(a.round), Path(a.episodes), Path(a.close), Path(a.out), king=a.king, system_prompt=prompt)
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
