"""Truncation is recorded per episode and must be aggregated, not left as a silent cap.

`max_turns_hit` has always been knowable from `turn_exit_reason` and was never summed, so the headline
read as a capability rate with no sign the run was censored. Same principle as `partial_rate`: report the
denominator rather than let a rate stand in for a measurement it does not describe.
"""

from __future__ import annotations

from sh.validator.truncation import (
    episode_truncated,
    is_truncated,
    max_turns_hit,
    reasoning_allowance,
    suite_truncation,
    turn_mix,
)


def test_thinking_does_not_count_as_an_action():
    """A THINKING-only assistant turn is deliberation, not a tool call.

    The old custom harness billed every non-tool_result step against max_steps, so a THINKING step cost
    exactly as much as a call. Unmodified Hermes bills API calls; a reasoning-only call still consumes
    max_iterations, which is why the runner gives deliberation its own allowance.
    """
    action, reasoning = turn_mix(
        [
            {"role": "assistant", "reasoning": "considering ls", "content": ""},
            {"role": "assistant", "tool_calls": [{"id": "c0", "function": {"name": "terminal"}}]},
            {"role": "tool", "content": "ok"},
            {"from": "gpt", "value": "<think>plan</think>done"},
        ]
    )
    assert action == 2 and reasoning == 1


def test_in_band_think_markup_with_a_tool_call_is_an_action():
    """At this pin, V4 showed <think> inside the same gpt turn as the call. That is one action, not a
    reasoning turn plus an action — charging it twice would recreate the bug this module exists to report."""
    action, reasoning = turn_mix(
        [{"from": "gpt", "value": '<think>plan</think><tool_call>{"name":"terminal"}</tool_call>'}]
    )
    assert action == 1 and reasoning == 0


def test_max_turns_hit_is_the_harness_cutting_in():
    task = {"max_turns": 100}
    assert max_turns_hit(
        {"turn_exit_reason": "max_iterations_reached(100/100)", "api_calls": 100, "completed": False},
        {},
        task,
    )
    assert max_turns_hit({"api_calls": 100, "completed": False}, {}, task)
    assert not max_turns_hit({"api_calls": 100, "completed": True, "turn_exit_reason": "stop"}, {}, task)
    assert not max_turns_hit(
        {"turn_exit_reason": "episode token budget spent", "api_calls": 40, "completed": False}, {}, task
    )


def test_a_pass_can_still_be_truncated():
    """The work was done and then the harness stopped the episode. Still censored, still not SFT."""
    finish = {"turn_exit_reason": "budget_exhausted", "api_calls": 100, "completed": True}
    assert max_turns_hit(finish, {}, {"max_turns": 100})
    assert is_truncated(finish, {}, {"max_turns": 100})


def test_timeout_and_token_budget_are_truncation_too():
    assert is_truncated({"timed_out": True}, {}, {})
    assert is_truncated({"budget_spent": {"spent": 1, "budget": 1}}, {}, {})
    assert not is_truncated({"api_calls": 3, "completed": True}, {}, {"max_turns": 100})


def test_truncation_is_reported_beside_the_success_rate():
    record = suite_truncation(
        [
            {"verified_success": True, "truncated": False},
            {"verified_success": True, "truncated": True},
            {"verified_success": False, "truncated": True},
            {"verified_success": False, "truncated": False},
        ]
    )
    assert record["truncated_episodes"] == 2
    assert record["truncated_failures"] == 1


def test_a_run_that_was_never_truncated_says_zero_rather_than_omitting_it():
    """The field has to be present at zero. An absent key and a zero are the same thing to a reader who does
    not know the key exists, which is how this went unnoticed in the first place."""
    record = suite_truncation([{"verified_success": True}])
    assert record["truncated_episodes"] == 0 and record["truncated_failures"] == 0


def test_old_archives_reconstruct_truncation_from_what_they_already_carried():
    """New fields reproduce the existing headline with the counts added — including on rounds graded before
    `truncated` existed, from timed_out / budget_spent / finish_reason / signals."""
    assert episode_truncated({"timed_out": True})
    assert episode_truncated({"signals": ["token_budget_spent"]})
    assert episode_truncated({"finish_reason": "max_iterations_reached(100/100)"})
    assert not episode_truncated({"verified_success": True, "api_calls": 9})
    assert not episode_truncated({"void": True})  # the field is absent; a void is not a truncated run
    cut = suite_truncation(
        [
            {"verified_success": True, "timed_out": True},
            {"verified_success": False, "signals": ["token_budget_spent"]},
            {"verified_success": True, "api_calls": 4},
            {"verified_success": False, "void": True, "finish_reason": "overloaded"},
        ]
    )
    assert cut["truncated_episodes"] == 2 and cut["truncated_failures"] == 1


def test_reasoning_allowance_defaults_to_twice_the_action_budget():
    assert reasoning_allowance({"max_turns": 100}) == 200
    assert reasoning_allowance({"max_turns": 100, "max_reasoning_steps": 10}) == 10
    assert reasoning_allowance({}) == 0
