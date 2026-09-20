"""`FamilyStats` — what the baseline can do on a family right now (spec §2.2, FR-BL-2/3/5, FR-TG-9).

Pooled over a rolling window of rounds and **never across model eras**: a promotion is a different baseline, so
its episodes describe a different family difficulty. Everything here is a pure function of archived Episode
records, so the same numbers recompute from the archive at any time (the D1 exit).
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from sh.validator.truncation import suite_truncation

Z_95 = 1.959963984540054


def wilson(successes: int, n: int, *, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, clamped to [0, 1].

    The normal approximation is wrong exactly where these rates live: at 8/8 it reports a width of zero,
    claiming certainty from eight observations, and a family label would inherit that certainty.
    """
    if n <= 0:
        raise ValueError("an interval over no observations describes nothing")
    if not 0 <= successes <= n:
        raise ValueError(f"{successes} successes out of {n} is not a proportion")
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


# Label bands over the NULL arm's Wilson interval. A family is only worth a round if the baseline sometimes
# fails it (otherwise miners have no headroom) and something can sometimes pass it (otherwise no one is measured).
TRIVIAL_LOW = 0.80  # baseline reliably solves it
TRIVIAL_EFFICIENCY_MARGIN = 1.15  # … but a trivial family stays while it still costs the baseline calls (FR-TG-9)
# "Impossible" is a statement that *nothing* passes, and a Wilson bound is the wrong instrument for it: at 0/16 the
# upper bound is still 0.19, so any threshold tight enough to mean "never" would need hundreds of episodes. The
# evidence that matters is a clean sweep of zeros in both arms over a window with enough episodes to notice.
IMPOSSIBLE_MIN_N = 16
MIN_POOLED = 8  # fewer observations than this describe nothing (FR-TG-9)
MAX_PARTIAL_RATE = 0.5  # harness/template fault, not a task property (FR-BL-3)

EFFICIENCY_METRICS = ("api_calls", "tool_calls")


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    s = sorted(values)

    def q(p):
        return s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))]

    return {"median": statistics.median(s), "p25": q(0.25), "n": len(s)}


@dataclass
class Arm:
    """One reference arm (NULL or CANON) over the window."""

    episodes: list[dict] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.episodes)

    @property
    def successes(self) -> int:
        return sum(1 for e in self.episodes if e.get("verified_success"))

    @property
    def p(self) -> float | None:
        return self.successes / self.n if self.n else None

    @property
    def mean_credit(self) -> float | None:
        """Mean credit — the share of withheld checks that hold — over the arm (sh-scoring-v3's reference)."""
        from sh.scoring.v2 import credit

        return statistics.mean(credit(e) for e in self.episodes) if self.episodes else None

    @property
    def var_credit(self) -> float | None:
        from sh.scoring.v2 import credit

        return statistics.pvariance([credit(e) for e in self.episodes]) if self.n > 1 else None

    def wilson(self) -> tuple[float, float] | None:
        if not self.n:
            return None
        low, high = wilson(self.successes, self.n)
        return (round(low, 4), round(high, 4))

    def efficiency(self) -> dict[str, dict[str, float]]:
        """Medians over **verified successes only** — the cost of a failure is not a cost of solving."""
        won = [e for e in self.episodes if e.get("verified_success")]
        out = {}
        for m in EFFICIENCY_METRICS:
            vals = [float(e[m]) for e in won if isinstance(e.get(m), (int, float))]
            if q := _quantiles(vals):
                out[m] = q
        return out

    def record(self) -> dict:
        r = {
            "n": self.n,
            "successes": self.successes,
            "p": self.p,
            "mean_credit": None if self.mean_credit is None else round(self.mean_credit, 4),
            "wilson": self.wilson(),
            "efficiency": self.efficiency(),
            **suite_truncation(self.episodes),
        }
        return r


@dataclass
class FamilyStats:
    family: str
    window: list[str]
    era: str
    null: Arm
    canon: Arm
    partial_rate: float
    efficiency_reference: dict[str, int]

    @property
    def pooled(self) -> int:
        return self.null.n

    def label(self) -> str:
        """`trivial | frontier | impossible` from the NULL arm; `unknown` until the window has evidence."""
        iv = self.null.wilson()
        if self.pooled < MIN_POOLED or iv is None:
            return "unknown"
        if iv[0] >= TRIVIAL_LOW:
            return "trivial"
        if (
            (self.null.mean_credit or 0) < 0.02
            and (self.canon.mean_credit or 0) < 0.02
            and self.null.n >= IMPOSSIBLE_MIN_N
        ):
            return "impossible"
        return "frontier"

    def delta_c(self) -> float | None:
        """How much prose alone moves the baseline — the miners' yardstick (FR-BL-5)."""
        if self.null.mean_credit is None or self.canon.mean_credit is None:
            return None
        return round(self.canon.mean_credit - self.null.mean_credit, 4)

    def delta_e(self) -> dict[str, float]:
        """Fractional efficiency gain of CANON over NULL, **paired by instance**.

        Comparing each arm's median over its own successes compares two different populations: an arm only
        contributes the instances it solved, so the weaker arm's successes are its easy ones. Family #2's first
        screen showed this plainly — CANON solved one harder instance NULL never did, and paid for it in calls,
        so an unpaired comparison reported CANON as 43 % *less* efficient while it was in fact strictly better.
        Only instances both arms verified can say anything about cost, so only those are compared.
        """
        out = {}
        for metric in EFFICIENCY_METRICS:
            pairs = [
                (n, c)
                for task_id, n in self._by_task(self.null, metric).items()
                if (c := self._by_task(self.canon, metric).get(task_id)) is not None
            ]
            if not pairs:
                continue
            null_total = sum(n for n, _ in pairs)
            if null_total:
                out[metric] = round((null_total - sum(c for _, c in pairs)) / null_total, 4)
        return out

    @staticmethod
    def _by_task(arm: "Arm", metric: str) -> dict[str, float]:
        """Mean of `metric` over the arm's verified successes, per instance (repeats of one instance average)."""
        rows: dict[str, list[float]] = {}
        for e in arm.episodes:
            if e.get("verified_success") and isinstance(e.get(metric), (int, float)):
                rows.setdefault(str(e.get("task_id")), []).append(float(e[metric]))
        return {k: sum(v) / len(v) for k, v in rows.items()}

    def paired_instances(self) -> int:
        """How many instances both arms solved — the sample `delta_e` actually rests on."""
        return len(
            {t for t in self._by_task(self.null, "api_calls")} & {t for t in self._by_task(self.canon, "api_calls")}
        )

    def retirement(self) -> str | None:
        """Why the family should leave rotation, or None. Order matters: a harness fault is not a task property."""
        if self.partial_rate >= MAX_PARTIAL_RATE:
            return f"partial_rate {self.partial_rate:.2f} ≥ {MAX_PARTIAL_RATE} — harness or template fault (FR-BL-3)"
        label = self.label()
        if label == "impossible":
            return "baseline and CANON both never pass — nothing is being measured"
        if label == "trivial":
            ref = self.efficiency_reference.get("api_calls")
            p25 = (self.null.efficiency().get("api_calls") or {}).get("p25")
            if ref and p25 is not None and p25 <= TRIVIAL_EFFICIENCY_MARGIN * ref:
                return (
                    f"trivial and cheap for the baseline (p25 api_calls {p25} ≤ "
                    f"{TRIVIAL_EFFICIENCY_MARGIN}×{ref}) — no headroom left (FR-TG-9)"
                )
        if (
            self.pooled >= MIN_POOLED
            and self.canon.n >= MIN_POOLED
            and (self.delta_c() or 0) <= 0
            and not self.delta_e()
        ):
            return "CANON no longer beats NULL — prose has stopped helping (FR-BL-5)"
        return None

    def record(self) -> dict:
        canon = self.canon.record() | {
            "delta_c": self.delta_c(),
            "delta_e": self.delta_e(),
            "delta_e_paired_instances": self.paired_instances(),
        }
        retire = self.retirement()
        cut = suite_truncation(self.null.episodes + self.canon.episodes)
        return {
            "schema": "sh-family-stats-v2",
            "family": self.family,
            "era": self.era,
            "window": self.window,
            "null": self.null.record(),
            "canon": canon,
            "label": self.label(),
            "partial_rate": round(self.partial_rate, 4),
            "truncated_episodes": cut["truncated_episodes"],
            "truncated_failures": cut["truncated_failures"],
            "in_rotation": retire is None,
            "retirement": retire,
        }


def family_stats(
    episodes: list[dict], family: str, window: list[str], era: str, efficiency_reference: dict[str, int] | None = None
) -> FamilyStats:
    """Pool the reference arms for one family over `window` in `era`. Miner surfaces are ignored here."""
    rounds = set(window)
    mine = [
        e
        for e in episodes
        if e.get("family") == family and e.get("round_id") in rounds and e.get("era", era) == era and not e.get("void")
    ]  # an episode the provider never served is not evidence about the family
    null = Arm([e for e in mine if e.get("surface") == "null"])
    canon = Arm([e for e in mine if e.get("surface") == "canon"])
    ref_eps = null.episodes + canon.episodes
    partials = sum(1 for e in ref_eps if e.get("partial") or e.get("timed_out"))
    return FamilyStats(
        family=family,
        window=list(window),
        era=era,
        null=null,
        canon=canon,
        partial_rate=partials / len(ref_eps) if ref_eps else 0.0,
        efficiency_reference=efficiency_reference or {},
    )


def load_episodes(*paths: Path) -> list[dict]:
    """Episode records from `episodes.jsonl` files or directories of `episode.json`."""
    out = []
    for p in paths:
        if p.is_dir():
            out += [json.loads(f.read_text()) for f in sorted(p.rglob("episode.json"))]
        elif p.suffix == ".jsonl":
            out += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        else:
            out.append(json.loads(p.read_text()))
    return out


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="recompute FamilyStats from archived episodes")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--family", required=True)
    ap.add_argument("--window", required=True, help="comma-separated round ids")
    ap.add_argument("--era", default="e0")
    ap.add_argument("--efficiency-reference", default="{}")
    a = ap.parse_args(argv)
    eps = load_episodes(*[Path(p) for p in a.paths])
    s = family_stats(eps, a.family, a.window.split(","), a.era, json.loads(a.efficiency_reference))
    print(json.dumps(s.record(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
