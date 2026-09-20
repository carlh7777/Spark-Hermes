"""One round, end to end (spec §9.1).

    publish(r)   the sealed instances go out: prompt, fixture, published half, withheld COMMITMENT
    evaluate(r)  NULL, CANON and every active bundle run through the same runner, boundary and grader
    score(r)     FamilyStats over the window -> scoring references -> a Score per hotkey -> weights
    close(r)     the withheld half and its salt are revealed, and every commitment is re-verified in public

`close` is the step that makes the whole design checkable by the people it judges: until it runs, a miner is
told only what was published; after it, anyone can recompute `HMAC(salt, canonical_json(withheld))` and confirm
the validator graded them against the half it committed to *before* they submitted, not one written afterwards.

    python -m sh.validator.round --round DIR --episodes DIR --out DIR [--reveal DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sys
from pathlib import Path

from sh.scoring.v2 import PARAMS_V2, FamilyReference, MinerWindow, credit, score, weights
from sh.validator.stats import family_stats, load_episodes
from sh.validator.truncation import suite_truncation

RESERVED_SURFACES = ("null", "canon")


def canonical(obj) -> bytes:
    """The bytes a commitment is over. Must match `supply/seal.py` exactly or nothing verifies."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def commitment_for(withheld: dict, salt: str) -> str:
    """Must reproduce `supply/seal.py::commitment_for` byte for byte — algorithm prefix included, so the scheme
    can be changed later without a silent mismatch here."""
    return "hmac-sha256:" + hmac.new(bytes.fromhex(salt), canonical(withheld), hashlib.sha256).hexdigest()


def verify_commitment(withheld: dict, salt: str, commitment: str) -> bool:
    return hmac.compare_digest(commitment_for(withheld, salt), commitment)


def reference_stats(episodes: list[dict], families: set[str], window: list[str], era: str) -> tuple[dict, dict]:
    """`FamilyStats` per family, plus the scoring references built from their NULL arms."""
    records, refs = {}, {}
    for family in sorted(families):
        st = family_stats(episodes, family, window, era)
        records[family] = st.record()
        samples = {
            m: [
                float(e[m])
                for e in st.null.episodes
                if e.get("verified_success") and isinstance(e.get(m), (int, float))
            ]
            for m in PARAMS_V2.metrics
        }
        refs[family] = FamilyReference(
            family=family,
            n=st.null.n,
            successes=st.null.successes,
            medians={m: v["median"] for m, v in st.null.efficiency().items()},
            samples=samples,
            requires_self_check=any(e.get("requires_self_check") for e in st.null.episodes),
            mean_credit=st.null.mean_credit,
            var_credit=st.null.var_credit,
            baseline={str(e["task_id"]): credit(e) for e in st.null.episodes if e.get("task_id") and not e.get("void")},
        )
    return records, refs


def close(
    round_dir: Path,
    episodes_dir: Path,
    out: Path,
    *,
    reveal_dir: Path | None = None,
    era: str = "e0",
    params=PARAMS_V2,
    window: list[str] | None = None,
) -> dict:
    """Score the round and publish everything needed to check it.

    `window` is the list of round ids whose episodes are pooled — for the reference arms *and* the miners. The
    two must see the same window: a miner scored over eight rounds against a baseline measured on one would be
    compared to the wrong denominator. The reveal and the commitment check cover only this round's tasks."""
    out.mkdir(parents=True, exist_ok=True)
    tasks = {p.stem: json.loads(p.read_text()) for p in sorted((round_dir / "tasks").glob("*.json"))}
    if not tasks:
        raise SystemExit(f"no tasks in {round_dir / 'tasks'}")
    round_id = next(iter(tasks.values())).get("round_id", "")
    eps = load_episodes(episodes_dir)
    families = {t.get("family") for t in tasks.values() if t.get("family")}

    window = window or [round_id]
    records, refs = reference_stats(eps, families, window, era)

    miners: dict[str, MinerWindow] = {}
    for e in eps:
        surface = e.get("surface")
        if (
            not isinstance(surface, str)
            or surface in RESERVED_SURFACES
            or e.get("void")
            or e.get("round_id") not in window
        ):
            continue
        miners.setdefault(surface, MinerWindow(surface)).episodes.append(e)
    scores = {h: score(m, refs, params) for h, m in miners.items()}
    w = weights({h: s["score"] for h, s in scores.items()})

    # The reveal: the withheld half, its salt, and a public re-verification of every commitment.
    reveal_dir = reveal_dir or (round_dir / "withheld")
    reveals, verified = {}, {}
    for task_id, task in tasks.items():
        path = reveal_dir / f"{task_id}.json"
        if not path.exists():
            # a committed task with no reveal is a broken close (the criteria cannot be checked), not a probe
            verified[task_id] = False if task.get("withheld_commitment") else None
            continue
        sealed = json.loads(path.read_text())
        withheld, salt = sealed["withheld"], sealed["salt"]
        reveals[task_id] = {"withheld": withheld, "salt": salt}
        if sealed.get("source"):  # where the task came from (swe_fix: the dataset's instance id), public at close
            reveals[task_id]["source"] = sealed["source"]
        commitment = task.get("withheld_commitment")
        verified[task_id] = bool(commitment) and verify_commitment(withheld, salt, commitment)

    record = {
        "schema": "sh-round-close-v3",
        "scoring": "sh-scoring-v3 (credit)",
        "round_id": round_id,
        "era": era,
        "tasks": sorted(tasks),
        "episodes": len(eps),
        **suite_truncation(eps),
        "family_stats": records,
        "scores": scores,
        "weights": w,
        "commitments_verified": verified,
        "commitments_ok": all(v for v in verified.values() if v is not None),
    }
    (out / "close.json").write_text(json.dumps(record, indent=1))
    (out / "reveal.json").write_text(json.dumps(reveals, indent=1))
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reveal")
    ap.add_argument("--era", default="e0")
    a = ap.parse_args(argv)
    r = close(Path(a.round), Path(a.episodes), Path(a.out), reveal_dir=Path(a.reveal) if a.reveal else None, era=a.era)
    print(
        json.dumps(
            {
                "round_id": r["round_id"],
                "episodes": r["episodes"],
                "commitments_ok": r["commitments_ok"],
                "weights": r["weights"],
                "scores": {h: round(s["score"], 6) for h, s in r["scores"].items()},
            },
            indent=1,
        )
    )
    return 0 if r["commitments_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
