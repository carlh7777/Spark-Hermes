"""When the harness cut an episode off, and how much of a run that describes.

`max_turns_hit` and `token_budget_spent` have always been knowable per episode — the first from Hermes'
`turn_exit_reason`, the second from the proxy marker — and aggregated nowhere. The headline then reads as a
capability rate with no sign the majority of a run was censored.

Two things break:

  * `api_calls` / `tool_calls` / token totals are *upper-censored* for those episodes, so comparing two runs'
    efficiency compares truncated distributions.
  * a budget-limited failure is indistinguishable from a capability failure in the only number anyone reads.

`truncated_episodes` and `truncated_failures` sit beside the rate. Same principle the suite already applies to
`partial_rate`: report the denominator rather than let a rate stand in for a measurement it does not describe.

A truncated episode is also not a trajectory to imitate — its last step is the harness saying so, not the model
finishing — so exports refuse it as SFT and as the chosen side of a preference pair.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

# Hermes names for "the iteration cap fired". Token-budget wording is excluded: that is a different cap, recorded
# as `budget_spent` on the finish, and must not be double-counted as a turn-limit hit.
_TURN_CAP = ("max_iterations", "budget_exhausted", "iteration budget")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def turn_mix(messages: Sequence[object] | None) -> tuple[int, int]:
    """`(action_turns, reasoning_turns)` over assistant messages.

    An action is a turn that called a tool (structured `tool_calls`, or Hermes-dialect `<tool_call>` markup).
    A reasoning turn deliberated and did not call: `reasoning` / `reasoning_content`, or a `<think>` block with
    no call. A final text answer counts as an action — the model finished.

    Measured because the pinned model's chat template sets reasoning high and returns deliberation on a separate
    channel. In the old custom harness that channel was a THINKING *step* billed against `max_steps` (49% of
    every budget). Unmodified Hermes bills an API call, and at this pin thinking is usually in-band in the same
    call as the tool (`<think>` inside the gpt turn). Reasoning-only calls still exist, and they still consume
    `max_iterations`. The split is how we tell those apart from actions.
    """
    action = reasoning = 0
    for raw in messages or ():
        if not isinstance(raw, Mapping):
            continue
        role = raw.get("role") or raw.get("from")
        if role not in ("assistant", "gpt"):
            continue
        if raw.get("tool_calls"):
            action += 1
            continue
        content = str(raw.get("content") or raw.get("value") or "")
        thought = raw.get("reasoning") or raw.get("reasoning_content") or ""
        called = "<tool_call>" in content
        remainder = _THINK.sub("", content).strip()
        thought_only = (bool(thought) or "<think>" in content) and not called and not remainder
        if called:
            action += 1
        elif thought_only:
            reasoning += 1
        else:
            action += 1
    return action, reasoning


def max_turns_hit(finish: Mapping, result: Mapping | None, task: Mapping) -> bool:
    """Did the *turn* cap fire — Hermes cutting in, not the model finishing.

    A pass can still be truncated: the work was done and then the harness stopped the episode. That is still
    censored, and it is still not a trajectory to imitate.
    """
    result = result or {}
    if finish.get("max_turns_hit") or result.get("max_turns_hit"):
        return True
    reason = f"{finish.get('turn_exit_reason') or ''} {result.get('turn_exit_reason') or ''}".lower()
    if "token budget" not in reason and "insufficient_quota" not in reason:
        if any(s in reason for s in _TURN_CAP):
            return True
    cap = int(task.get("max_turns") or finish.get("max_turns") or 0)
    api = int(finish.get("api_calls") or result.get("api_calls") or 0)
    completed = finish.get("completed")
    if completed is None:
        completed = result.get("completed")
    return bool(cap and api >= cap and completed is False)


def is_truncated(finish: Mapping, result: Mapping | None = None, task: Mapping | None = None) -> bool:
    """The harness ended the episode: turn cap, token budget, or the host clock."""
    if finish.get("timed_out") or finish.get("budget_spent"):
        return True
    return max_turns_hit(finish, result, task or {})


def episode_truncated(episode: Mapping) -> bool:
    """Truncation on an episode record, including archives that predate the `truncated` field.

    The field has to be readable at zero on new records, and reconstructable from what older records already
    carried — otherwise the count is `None` for every round before this change, which is how the gap stayed
    invisible in the first place.
    """
    if "truncated" in episode:
        return bool(episode["truncated"])
    if episode.get("timed_out") or episode.get("budget_spent") or episode.get("max_turns_hit"):
        return True
    if "token_budget_spent" in (episode.get("signals") or ()) or "max_turns_hit" in (episode.get("signals") or ()):
        return True
    reason = str(episode.get("finish_reason") or "").lower()
    return "token budget" not in reason and any(s in reason for s in _TURN_CAP)


def suite_truncation(episodes: Sequence[Mapping]) -> dict[str, int]:
    """Counts over episodes that ran. Voids are not evidence; they are not truncated either."""
    ran = [e for e in episodes if not e.get("void")]
    cut = [e for e in ran if episode_truncated(e)]
    return {
        "truncated_episodes": len(cut),
        "truncated_failures": sum(1 for e in cut if not e.get("verified_success")),
    }


def reasoning_allowance(task: Mapping) -> int:
    """Deliberation's own turn budget. Defaults to `2 * max_turns` when a task does not set one.

    Bounded rather than free: a policy that only thinks still has to terminate. No task-definition change —
    the default is a harness policy, and `pins_digest` records it.
    """
    if task.get("max_reasoning_steps") is not None:
        return int(task["max_reasoning_steps"])
    return int(task.get("max_turns") or 0) * 2
