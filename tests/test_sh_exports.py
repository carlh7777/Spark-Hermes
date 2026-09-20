"""Exports: what leaves a closed round as training data, and what never does."""

from __future__ import annotations

import json

from sh.exports.build import _secrets


def test_test_ids_of_a_withheld_suite_are_not_secrets_for_the_leak_scan():
    """A verified swe_fix trajectory names the tests it ran; those ids are public at close and part of any honest
    `pytest -v`. Only salts and withheld *values* are secrets."""
    reveal = {
        "t": {
            "salt": "a" * 64,
            "withheld": {
                "predicates": [
                    ["custom", "swe_test", "tests/text/test_fonts.py::DescribeFontFiles::it_catalogs"],
                    ["custom", "facet_test", "test_state.py::test_report_has_the_total"],
                    ["digest_is", "out.txt", "b" * 64],
                ]
            },
        }
    }
    assert _secrets(reveal) == {"a" * 64, "b" * 64}


def test_a_row_carries_the_system_prompt_its_episode_ran_under(tmp_path):
    from sh.exports.build import _rows_for

    ep = tmp_path / "ep"
    ep.mkdir()
    (ep / "trajectory.json").write_text(
        json.dumps(
            [
                {"from": "system", "value": "generic"},
                {"from": "human", "value": "fix it"},
                {"from": "gpt", "value": "done"},
            ]
        )
    )
    (ep / "system_prompt.txt").write_text("the king's SOUL, as the agent saw it")
    row = _rows_for(ep, {"task_id": "t"}, {"prompt": "fix it"}, "fallback")
    assert row is not None and "the king's SOUL" in json.dumps(row) and "fallback" not in json.dumps(row)


def test_a_void_king_episode_is_never_exported(tmp_path):
    from sh.exports.build import build

    rd = tmp_path / "round"
    (rd / "tasks").mkdir(parents=True)
    (rd / "close.json").write_text(json.dumps({"tasks": []}))
    eps = rd / "episodes" / "5K" / "t0"
    eps.mkdir(parents=True)
    (eps / "episode.json").write_text(
        json.dumps({"task_id": "t0", "surface": "5K", "verified_success": True, "void": True})
    )
    (eps / "trajectory.json").write_text(
        json.dumps([{"from": "human", "value": "fix it"}, {"from": "gpt", "value": "done"}])
    )
    m = build(rd, rd / "episodes", rd / "close.json", tmp_path / "out", king="5K")
    assert m["sft_rows"] == 0 and m["gates"]["void"] == 1


def test_a_truncated_king_episode_is_never_sft(tmp_path):
    """SFT is imitation. A trajectory the harness cut off ends mid-work."""
    from sh.exports.build import build

    rd = tmp_path / "round"
    (rd / "tasks").mkdir(parents=True)
    (rd / "close.json").write_text(json.dumps({"tasks": []}))
    eps = rd / "episodes" / "5K" / "t0"
    eps.mkdir(parents=True)
    (eps / "episode.json").write_text(
        json.dumps({"task_id": "t0", "surface": "5K", "verified_success": True, "truncated": True})
    )
    (eps / "trajectory.json").write_text(
        json.dumps([{"from": "human", "value": "fix it"}, {"from": "gpt", "value": "done"}])
    )
    m = build(rd, rd / "episodes", rd / "close.json", tmp_path / "out", king="5K")
    assert m["sft_rows"] == 0 and m["gates"]["truncated"] == 1


def test_a_truncated_episode_is_never_the_chosen_side_of_a_pair(tmp_path):
    """It stopped at the cap, mid-work. Imitating it teaches an agent to stop before it finishes."""
    from sh.exports.build import build

    rd = tmp_path / "round"
    (rd / "tasks").mkdir(parents=True)
    (rd / "tasks" / "t0.json").write_text(json.dumps({"task_id": "t0", "prompt": "fix it", "tools": []}))
    (rd / "close.json").write_text(json.dumps({"tasks": ["t0"]}))
    for surface, credit, truncated in (("5K", 1.0, True), ("null", 0.0, False), ("5B", 1.0, False)):
        d = rd / "episodes" / surface / "t0"
        d.mkdir(parents=True)
        (d / "episode.json").write_text(
            json.dumps(
                {
                    "task_id": "t0",
                    "surface": surface,
                    "credit": credit,
                    "verified_success": credit == 1.0,
                    "truncated": truncated,
                }
            )
        )
        (d / "system_prompt.txt").write_text(surface)
        (d / "trajectory.json").write_text(
            json.dumps([{"from": "system", "value": surface}, {"from": "gpt", "value": f"by {surface}"}])
        )
    build(rd, rd / "episodes", rd / "close.json", tmp_path / "out", king="5K")
    pairs = [json.loads(x) for x in (tmp_path / "out" / "dpo.jsonl").read_text().splitlines()]
    assert pairs == []  # the only king episode was truncated; nothing to prefer


def test_a_task_every_surface_passes_still_yields_an_efficiency_pair(tmp_path):
    """At a high pass rate there is no correctness loser. What is left is the same task solved for far fewer tokens."""
    from sh.exports.build import build

    rd = tmp_path / "round"
    (rd / "tasks").mkdir(parents=True)
    (rd / "tasks" / "t0.json").write_text(json.dumps({"task_id": "t0", "prompt": "fix it", "tools": []}))
    (rd / "close.json").write_text(json.dumps({"tasks": ["t0"]}))
    for surface, tokens in (("5K", 20_000), ("null", 90_000)):
        d = rd / "episodes" / surface / "t0"
        d.mkdir(parents=True)
        (d / "episode.json").write_text(
            json.dumps(
                {
                    "task_id": "t0",
                    "surface": surface,
                    "credit": 1.0,
                    "verified_success": True,
                    "tokens": {"prompt_tokens": tokens, "completion_tokens": 0},
                }
            )
        )
        (d / "system_prompt.txt").write_text("KING SOUL" if surface == "5K" else "GENERIC")
        (d / "trajectory.json").write_text(
            json.dumps([{"from": "system", "value": surface}, {"from": "gpt", "value": f"by {surface}"}])
        )
    build(rd, rd / "episodes", rd / "close.json", tmp_path / "out", king="5K")
    pairs = [json.loads(x) for x in (tmp_path / "out" / "dpo.jsonl").read_text().splitlines()]
    assert len(pairs) == 1 and pairs[0]["kind"] == "efficiency"
    assert pairs[0]["chosen_surface"] == "5K" and pairs[0]["rejected_surface"] == "null"
    assert pairs[0]["rejected"][0]["value"] == "KING SOUL"


def test_attempts_within_noise_do_not_mint_an_efficiency_pair(tmp_path):
    """A 2% gap is sampling noise. Inventing a pair there trains the wobble."""
    from sh.exports.build import build

    rd = tmp_path / "round"
    (rd / "tasks").mkdir(parents=True)
    (rd / "tasks" / "t0.json").write_text(json.dumps({"task_id": "t0", "prompt": "fix it", "tools": []}))
    (rd / "close.json").write_text(json.dumps({"tasks": ["t0"]}))
    for surface, tokens in (("5K", 50_000), ("null", 51_000)):
        d = rd / "episodes" / surface / "t0"
        d.mkdir(parents=True)
        (d / "episode.json").write_text(
            json.dumps(
                {
                    "task_id": "t0",
                    "surface": surface,
                    "credit": 1.0,
                    "verified_success": True,
                    "tokens": {"prompt_tokens": tokens, "completion_tokens": 0},
                }
            )
        )
        (d / "trajectory.json").write_text(json.dumps([{"from": "gpt", "value": surface}]))
    build(rd, rd / "episodes", rd / "close.json", tmp_path / "out", king="5K")
    assert (tmp_path / "out" / "dpo.jsonl").read_text() == ""


def test_a_dpo_pair_shares_the_kings_system_turn(tmp_path):
    from sh.exports.build import build

    rd = tmp_path / "round"
    (rd / "tasks").mkdir(parents=True)
    (rd / "tasks" / "t0.json").write_text(json.dumps({"task_id": "t0", "prompt": "fix it", "tools": []}))
    (rd / "close.json").write_text(json.dumps({"tasks": ["t0"]}))
    for surface, credit, sp in (("5K", 1.0, "KING SOUL"), ("null", 0.0, "GENERIC")):
        d = rd / "episodes" / surface / "t0"
        d.mkdir(parents=True)
        (d / "episode.json").write_text(
            json.dumps({"task_id": "t0", "surface": surface, "credit": credit, "verified_success": credit == 1.0})
        )
        (d / "system_prompt.txt").write_text(sp)
        (d / "trajectory.json").write_text(
            json.dumps(
                [
                    {"from": "system", "value": sp},
                    {"from": "human", "value": "fix it"},
                    {"from": "gpt", "value": f"by {surface}"},
                ]
            )
        )
    build(rd, rd / "episodes", rd / "close.json", tmp_path / "out", king="5K")
    pairs = [json.loads(x) for x in (tmp_path / "out" / "dpo.jsonl").read_text().splitlines()]
    assert len(pairs) == 1
    p = pairs[0]
    assert p["chosen"][0] == {"from": "system", "value": "KING SOUL"}
    assert p["rejected"][0] == {"from": "system", "value": "KING SOUL"}  # not GENERIC: the confound is removed
    assert p["rejected"][-1]["value"] == "by null"  # but the rejected trajectory is still null's
    assert p["kind"] == "correctness"
