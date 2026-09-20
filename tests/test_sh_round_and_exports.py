"""Closing a round and turning it into training data — the two stages that make the round checkable and useful."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

from sh.exports.build import build
from sh.validator.round import canonical, close, commitment_for

WITHHELD = {"predicates": [["digest_is", "out.txt", "a" * 64]]}
SALT = "ab" * 32


def _round(tmp_path: Path, *, commitment=None) -> Path:
    rd = tmp_path / "round"
    (rd / "tasks").mkdir(parents=True)
    (rd / "withheld").mkdir()
    task = {
        "task_id": "t-1",
        "family": "f",
        "round_id": "r1",
        "prompt": "do the thing",
        "tools": ["terminal"],
        "published": {"predicates": []},
        "withheld_commitment": commitment or commitment_for(WITHHELD, SALT),
    }
    (rd / "tasks" / "t-1.json").write_text(json.dumps(task))
    (rd / "withheld" / "t-1.json").write_text(json.dumps({"withheld": WITHHELD, "salt": SALT}))
    return rd


def _episodes(tmp_path: Path, spec: list[tuple[str, bool]]) -> Path:
    root = tmp_path / "eps"
    for surface, won in spec:
        d = root / surface / "t-1"
        d.mkdir(parents=True)
        (d / "episode.json").write_text(
            json.dumps(
                {
                    "task_id": "t-1",
                    "family": "f",
                    "round_id": "r1",
                    "surface": surface,
                    "verified_success": won,
                    "published_pass": won,
                    "overfit": False,
                    "disqualified": False,
                    "api_calls": 5 if won else 9,
                    "tool_calls": 5,
                    "self_checked": True,
                }
            )
        )
        (d / "trajectory.json").write_text(
            json.dumps(
                [
                    {"from": "system", "value": "GENERIC HERMES FUNCTION-CALLING PROMPT"},
                    {"from": "human", "value": "do the thing"},
                    {"from": "gpt", "value": "<think>plan</think>doing it"},
                ]
            )
        )
    return root


def test_closing_a_round_verifies_every_commitment(tmp_path):
    """The point of the reveal: anyone can check the validator graded against the half it committed to first."""
    rd = _round(tmp_path)
    eps = _episodes(tmp_path, [("null", False), ("5Fminer", True)])
    record = close(rd, eps, tmp_path / "out")
    assert record["commitments_ok"] and record["commitments_verified"]["t-1"] is True
    assert record["truncated_episodes"] == 0 and record["truncated_failures"] == 0
    assert json.loads((tmp_path / "out" / "reveal.json").read_text())["t-1"]["salt"] == SALT


def test_a_commitment_that_does_not_match_the_revealed_half_is_caught(tmp_path):
    """A validator swapping the withheld half after submissions is exactly what this makes impossible."""
    rd = _round(tmp_path, commitment=commitment_for({"predicates": [["file_exists", "other.txt"]]}, SALT))
    eps = _episodes(tmp_path, [("null", False), ("5Fminer", True)])
    record = close(rd, eps, tmp_path / "out")
    assert not record["commitments_ok"] and record["commitments_verified"]["t-1"] is False


def test_the_reference_arms_are_never_scored_as_miners(tmp_path):
    rd = _round(tmp_path)
    eps = _episodes(tmp_path, [("null", False), ("canon", True), ("5Fminer", True)])
    record = close(rd, eps, tmp_path / "out")
    assert set(record["scores"]) == {"5Fminer"}
    assert set(record["weights"]) == {"5Fminer"}


def test_the_commitment_format_matches_the_private_sealer():
    """Both sides must agree byte for byte, prefix included, or nothing ever verifies."""
    expected = "hmac-sha256:" + hmac.new(bytes.fromhex(SALT), canonical(WITHHELD), hashlib.sha256).hexdigest()
    assert commitment_for(WITHHELD, SALT) == expected


def test_exports_replace_the_converter_s_system_turn(tmp_path):
    """V4's rule: the converter emits a generic prompt the agent never ran under; training on it teaches a
    prompt no deployment shows the model."""
    rd = _round(tmp_path)
    eps = _episodes(tmp_path, [("null", False), ("5Fminer", True)])
    close(rd, eps, tmp_path / "out")
    manifest = build(
        rd,
        eps,
        tmp_path / "out" / "close.json",
        tmp_path / "export",
        king="5Fminer",
        system_prompt="THE REAL PINNED PROMPT",
    )
    rows = [json.loads(line) for line in (tmp_path / "export" / "sft.jsonl").read_text().splitlines()]
    assert manifest["sft_rows"] == 1
    assert rows[0]["conversations"][0] == {"from": "system", "value": "THE REAL PINNED PROMPT"}
    assert all("GENERIC HERMES" not in t["value"] for t in rows[0]["conversations"])
    assert rows[0]["tools"] == ["terminal"]


def test_only_withheld_verified_episodes_become_training_data(tmp_path):
    """A row must mean "this actually solved it", not "this looked right"."""
    rd = _round(tmp_path)
    eps = _episodes(tmp_path, [("null", False), ("5Fa", True), ("5Fb", True)])
    close(rd, eps, tmp_path / "out")
    manifest = build(rd, eps, tmp_path / "out" / "close.json", tmp_path / "export", king="5Fa")
    assert manifest["sft_rows"] == 1 and manifest["gates"]["not_king"] == 2  # 5Fb solved it too; not the king
    assert manifest["king"] == "5Fa"


def test_a_round_without_a_king_exports_nothing(tmp_path):
    """The king's trajectories are the product; without a king there is no product, only evidence."""
    rd = _round(tmp_path)
    eps = _episodes(tmp_path, [("null", False), ("5Fa", True)])
    close(rd, eps, tmp_path / "out")
    manifest = build(rd, eps, tmp_path / "out" / "close.json", tmp_path / "export")
    assert manifest["king"] is None and manifest["sft_rows"] == 0 and manifest["dpo_pairs"] == 0


def test_dpo_pairs_never_cross_an_instance(tmp_path):
    """Across instances a pair encodes difficulty rather than strategy."""
    rd = _round(tmp_path)
    eps = _episodes(tmp_path, [("null", False), ("5Fa", True)])
    close(rd, eps, tmp_path / "out")
    build(rd, eps, tmp_path / "out" / "close.json", tmp_path / "export", king="5Fa")
    pairs = [json.loads(line) for line in (tmp_path / "export" / "dpo.jsonl").read_text().splitlines()]
    assert len(pairs) == 1
    assert pairs[0]["chosen_surface"] == "5Fa" and pairs[0]["rejected_surface"] == "null"
    assert pairs[0]["task_id"] == "t-1"


def test_a_row_carrying_withheld_material_is_refused(tmp_path):
    """The withheld half is what makes the next round gradeable; it must never leave in training data."""
    rd = _round(tmp_path)
    eps = _episodes(tmp_path, [("null", False), ("5Fa", True)])
    d = eps / "5Fa" / "t-1"
    (d / "trajectory.json").write_text(
        json.dumps([{"from": "system", "value": "x"}, {"from": "gpt", "value": f"the answer digest is {'a' * 64}"}])
    )
    close(rd, eps, tmp_path / "out")
    manifest = build(rd, eps, tmp_path / "out" / "close.json", tmp_path / "export", king="5Fa")
    assert manifest["sft_rows"] == 0 and manifest["gates"]["leaked"] == 1
    assert manifest["leak_scan"]["rows_refused"] == 1


def test_the_leaderboard_is_built_only_from_published_artefacts(tmp_path):
    """A leaderboard that knows something its readers cannot recompute is one they have to trust."""
    from sh.web.build import render

    rd = _round(tmp_path)
    eps = _episodes(tmp_path, [("null", False), ("5Fminer", True)])
    record = close(rd, eps, tmp_path / "out")
    page = render(record)
    assert "5Fminer" in page and "match" in page and "MISMATCH" not in page
    assert "<title>" in page and "site.css" in page  # themed by the shared stylesheet, like every other page
    assert SALT not in page  # the salt is in reveal.json, not on the page


def test_the_leaderboard_says_why_a_miner_earned_nothing(tmp_path):
    """Transparency to miners (FR-TRN): a zero must come with its reason."""
    from sh.web.build import render

    rd = _round(tmp_path)
    eps = _episodes(tmp_path, [("null", False), ("5Fminer", True)])
    record = close(rd, eps, tmp_path / "out")
    assert record["scores"]["5Fminer"]["reason"]  # one episode is far below the window minimum
    assert "window episodes" in render(record)


def test_a_pair_is_not_lost_because_the_first_loser_timed_out(tmp_path):
    """An episode killed on its timeout has no trajectory. Taking only the first loser dropped every pair whose
    first loser happened to be one of those — half a round's training value, lost to directory order."""
    rd = _round(tmp_path)
    eps = _episodes(tmp_path, [("5Fa", True), ("5Fb", False), ("5Fc", False)])
    (eps / "5Fb" / "t-1" / "trajectory.json").unlink()  # timed out: no trajectory was ever written
    close(rd, eps, tmp_path / "out")
    build(rd, eps, tmp_path / "out" / "close.json", tmp_path / "export", king="5Fa")
    pairs = [json.loads(line) for line in (tmp_path / "export" / "dpo.jsonl").read_text().splitlines()]
    assert len(pairs) == 1 and pairs[0]["rejected_surface"] == "5Fc"


def test_the_round_page_shows_the_github_author_beside_the_hotkey(tmp_path):
    """Identity on the page comes from the round's published `github` map (attested at seal): avatar, @login, the
    hotkey beneath, and the crown as a badge on the king's avatar. Without a map the page degrades to the hotkey."""
    from sh.web.build import render

    rd = _round(tmp_path)
    eps = _episodes(tmp_path, [("null", False), ("5Fminer", True)])
    record = close(rd, eps, tmp_path / "out")
    page = render(record, {"github": {"5Fminer": "octocat"}, "king": "5Fminer"})
    assert "github.com/octocat.png" in page and ">@octocat</a>" in page and "crown-badge" in page
    bare = render(record)
    assert "unlinked" in bare and "github.com/octocat.png" not in bare and "crown-badge" not in bare


def test_the_round_page_escapes_a_github_login(tmp_path):
    """A login is attacker-influenced text from GitHub; it is escaped wherever it lands (URL and label)."""
    from sh.web.build import render

    rd = _round(tmp_path)
    eps = _episodes(tmp_path, [("null", False), ("5Fminer", True)])
    record = close(rd, eps, tmp_path / "out")
    page = render(record, {"github": {"5Fminer": '"><b'}})
    assert '"><b' not in page and "&quot;&gt;&lt;b" in page
