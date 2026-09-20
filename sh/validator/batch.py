"""Run every (task, surface) of a round through episode + grade, N at a time (B6-lite: idempotent by
episode.json presence; no persistent queue yet).

    python -m sh.validator.batch --round DIR --surfaces null,canon=DIR,hk1=DIR --image IMG --inference URL --out OUT --concurrency 4
DIR/tasks/*.json (public projections) and DIR/withheld/*.json are read; OUT/<surface>/<task_id>/ holds each episode.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sh.validator.episode import run_episode
from sh.validator.grade import grade
from sh.validator.truncation import suite_truncation


def one(
    task: dict,
    withheld: dict | None,
    surface: str,
    bundle: Path | None,
    image: str,
    inference: str,
    out: Path,
    network: str = "host",
    tokens: Path | None = None,
    usage_dir: Path | None = None,
    checks_py: Path | None = None,
) -> dict:
    ep = out / surface / task["task_id"]
    image = task.get("image") or image  # an image-defined task names its own image; --image is the fallback
    if (ep / "episode.json").exists():
        return json.loads((ep / "episode.json").read_text())
    done = (ep / "finish.json").exists() and json.loads((ep / "finish.json").read_text()).get("stage") == "done"
    if not done:  # resume: a finished episode whose grading crashed is only re-graded
        run_episode(task, bundle, image, inference, ep, network=network, tokens_path=tokens, usage_dir=usage_dir)
    rec = grade(
        ep,
        task,
        withheld,
        image,
        bundle_dir=bundle,
        surface=surface,
        round_id=task.get("round_id", ""),
        checks_py=checks_py,
    )
    if rec.get("void"):
        # The provider failed, not the miner. Move the attempt aside — its finish.json says "done", and left in
        # place a resume would only re-grade the same failed trajectory, void again, forever.
        n = 1
        while (aside := ep.with_name(f"{ep.name}.void-{n}")).exists():
            n += 1
        shutil.move(str(ep), str(aside))
        return rec
    (ep / "episode.json").write_text(json.dumps(rec, indent=1))
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True)
    ap.add_argument("--surfaces", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--inference", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--network", default="host")
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--usage-dir", default=None)
    ap.add_argument("--void-retries", type=int, default=4, help="passes that re-run episodes the provider voided")
    a = ap.parse_args(argv)
    rd, out = Path(a.round), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(p.read_text()) for p in sorted((rd / "tasks").glob("*.json"))]
    withheld = {p.stem: json.loads(p.read_text()) for p in (rd / "withheld").glob("*.json")}
    # One family per round directory in practice; `checks/<family>.py` keeps it explicit and mirrors what
    # close(r) publishes (spec §9.2).
    checks_for = {p.stem: p for p in (rd / "checks").glob("*.py")} if (rd / "checks").is_dir() else {}
    surfaces: dict[str, Path | None] = {}
    for s in a.surfaces.split(","):
        name, _, d = s.partition("=")
        surfaces[name] = Path(d) if d else None
    jobs = [(t, withheld.get(t["task_id"]), name, b) for name, b in surfaces.items() for t in tasks]
    t0 = time.time()
    recs: list[dict] = []
    for attempt in range(1 + a.void_retries):
        voided: list[tuple] = []
        with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
            futs = {
                pool.submit(
                    one,
                    t,
                    w,
                    name,
                    b,
                    a.image,
                    a.inference,
                    out,
                    a.network,
                    Path(a.tokens) if a.tokens else None,
                    Path(a.usage_dir) if a.usage_dir else None,
                    checks_for.get(t.get("family", "")),
                ): (t, w, name, b)
                for t, w, name, b in jobs
            }
            for f in as_completed(futs):
                t, w, name, b = futs[f]
                try:
                    r = f.result()
                    if r.get("void"):
                        voided.append((t, w, name, b))
                        print(f"{name:>10} {t['task_id']}: void ({str(r.get('void_reason'))[:80]})", flush=True)
                        continue
                    recs.append(r)
                    print(
                        f"{name:>10} {t['task_id']}: verified={r['verified_success']} overfit={r['overfit']} "
                        f"dq={r['disqualified']} calls={r['api_calls']} wall={r['wall_s']}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"{name:>10} {t['task_id']}: ERROR {e!r}"[:300], flush=True)
        if not voided or attempt == a.void_retries:
            if voided:
                print(f"{len(voided)} episode(s) still void after {attempt + 1} passes", flush=True)
            break
        wait = 60 * (attempt + 1)  # the provider was out of capacity: give it room before the next pass
        print(f"{len(voided)} void episode(s); re-running them in {wait}s (pass {attempt + 2})", flush=True)
        time.sleep(wait)
        jobs = voided
    total = time.time() - t0
    (out / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    summary = {
        "episodes": len(recs),
        "total_wall_s": round(total, 1),
        "concurrency": a.concurrency,
        **suite_truncation(recs),
        "per_surface": {},
    }
    for name in surfaces:
        rs = [r for r in recs if r["surface"] == name]
        if rs:
            walls = [r["wall_s"] for r in rs if r.get("wall_s")]
            summary["per_surface"][name] = {
                "n": len(rs),
                "verified": sum(r["verified_success"] for r in rs),
                "overfit": sum(r["overfit"] for r in rs),
                "dq": sum(r["disqualified"] for r in rs),
                "partial": sum(r["partial"] for r in rs),
                "timed_out": sum(r["timed_out"] for r in rs),
                "truncated": sum(bool(r.get("truncated")) for r in rs),
                "truncated_failures": sum(1 for r in rs if r.get("truncated") and not r.get("verified_success")),
                "mean_wall_s": round(statistics.mean(walls), 1) if walls else None,
                "mean_api_calls": round(statistics.mean(r["api_calls"] or 0 for r in rs), 1),
                "self_checked": sum(bool(r["self_checked"]) for r in rs),
            }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
