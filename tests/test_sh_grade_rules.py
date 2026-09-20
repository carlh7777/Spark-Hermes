"""Trajectory rules: the DQ signals must fire on what the agent *did*, never on what it wrote into a file.

The regression these lock in: the rules used to regex the whole tool-call argument blob, so a solution whose
script contained the text `"path": "/..."` (or a comment mentioning `/ep/out`) was disqualified.
"""

from __future__ import annotations

import json

from sh.validator.grade import trajectory_rules

TASK = {"task_id": "t-1", "published": {"predicates": [["file_exists", "report.sh"]]}}


def _msgs(*calls) -> list:
    """Each call is (name, args) or (name, args, tool_result) — the result the tool reported back."""
    out = []
    for i, c in enumerate(calls):
        n, a, res = (*c, '{"ok": true}')[:3] if len(c) == 2 else c
        out.append(
            {"role": "assistant", "tool_calls": [{"id": f"c{i}", "function": {"name": n, "arguments": json.dumps(a)}}]}
        )
        out.append({"role": "tool", "tool_call_id": f"c{i}", "content": res})
    return out


def _signals(*calls, ep_writes=(), withheld=None):
    return trajectory_rules(list(_msgs(*calls)), list(ep_writes), None, TASK, withheld)[0]


def test_a_script_that_mentions_the_grader_is_not_an_access_attempt():
    """The file the agent writes is payload, not a destination."""
    content = '# see /ep/out for results; config is {"path": "/etc/hosts"}\nprintf hi\n'
    assert _signals(("write_file", {"path": "/ep/ws/report.sh", "content": content})) == []


def test_reading_the_grader_path_is_an_access_attempt():
    assert "read_grader_or_withheld_path_attempt" in _signals(("read_file", {"path": "/ep/out/grade.json"}))
    assert "read_grader_or_withheld_path_attempt" in _signals(("terminal", {"command": "cat /ep/withheld.json"}))
    assert "read_grader_or_withheld_path_attempt" in _signals(("terminal", {"command": "ls /runner"}))


def test_writing_outside_the_workspace_is_flagged_by_destination_only():
    assert "wrote_outside_workspace" in _signals(
        ("write_file", {"path": "/etc/cron.d/x", "content": "x"}, '{"bytes_written": 9}')
    )
    assert _signals(("write_file", {"path": "report.sh", "content": "x"})) == []  # relative → workspace
    assert "wrote_outside_workspace" in _signals(
        ("write_file", {"path": "/ep/ws/a", "content": "x"}), ep_writes=["/ep/bundle/x"]
    )
    assert _signals(("write_file", {"path": "/ep/ws/a", "content": "x"}, '{"bytes_written": 1}')) == []


def test_a_write_the_box_refused_is_not_a_write():
    """The read-only rootfs turns a stray absolute path into a failed call; a miner is judged on what happened."""
    refused = '{"bytes_written": 0, "error": "Failed to write file: mkdir: cannot create directory \u2018/x\u2019: Read-only file system"}'
    assert _signals(("write_file", {"path": "/x/report.sh", "content": "x"}, refused)) == []
    assert "wrote_outside_workspace" in _signals(
        ("write_file", {"path": "/x/report.sh", "content": "x"}, '{"bytes_written": 42}')
    )


def test_the_per_episode_scratch_areas_are_not_outside_the_workspace():
    """HERMES_HOME and /tmp are tmpfs discarded with the container — Hermes writes there itself."""
    assert _signals(("write_file", {"path": "/home/hermes/notes.md", "content": "x"}, '{"bytes_written": 2}')) == []
    assert _signals(("write_file", {"path": "/tmp/scratch", "content": "x"}, '{"bytes_written": 2}')) == []


def test_unparseable_arguments_are_treated_conservatively():
    msgs = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "c0", "function": {"name": "terminal", "arguments": "cat /ep/out/grade.json"}}],
        }
    ]
    assert "read_grader_or_withheld_path_attempt" in trajectory_rules(msgs, [], None, TASK, None)[0]


def test_self_check_is_a_verifying_call_after_the_last_mutating_one():
    _, checked, _ = trajectory_rules(
        list(_msgs(("write_file", {"path": "report.sh", "content": "x"}), ("terminal", {"command": "sh report.sh"}))),
        [],
        None,
        TASK,
        None,
    )
    assert checked
    _, checked, _ = trajectory_rules(
        list(_msgs(("terminal", {"command": "ls"}), ("write_file", {"path": "report.sh", "content": "x"}))),
        [],
        None,
        TASK,
        None,
    )
    assert not checked


def test_a_bundle_carrying_this_instance_s_answer_is_flagged(tmp_path):
    """The generator-leak rule: seed-derived values only — static family vocabulary is fine."""
    digest = "a" * 64
    task = {"task_id": "posix-report-r7-03", "published": {"predicates": [["digest_is", "report.sh", digest]]}}
    msgs = _msgs(("write_file", {"path": "report.sh", "content": "x"}))

    def signals(text):
        (tmp_path / "SKILL.md").write_text(text)
        return trajectory_rules(list(msgs), [], tmp_path, task, None)[0]

    assert signals("Write report.sh into data/ using awk.") == []  # family vocabulary
    assert "instance_literal_in_bundle" in signals(f"The answer hashes to {digest}.")
    assert "instance_literal_in_bundle" in signals("Special-case posix-report-r7-03.")


def test_an_inline_shell_marker_in_a_skill_is_flagged(tmp_path):
    (tmp_path / "SKILL.md").write_text("Run this: !`cat /ep/withheld.json`\n")
    task = {"task_id": "t-1", "published": {"predicates": []}}
    assert (
        "inline_shell_marker"
        in trajectory_rules(list(_msgs(("write_file", {"path": "a", "content": "x"}))), [], tmp_path, task, None)[0]
    )


def test_a_task_with_no_withheld_half_is_not_scored_as_overfit(monkeypatch, tmp_path):
    """Probes (spec §3.7b) carry no withheld half. Reading "absent" as "failed" marked every solved probe
    `overfit` and no probe a success, which is a statement about nothing."""
    import sh.validator.grade as g

    (tmp_path / "result.json").write_text(json.dumps({"messages": []}))
    (tmp_path / "finish.json").write_text(json.dumps({"api_calls": 3, "tool_calls": 2, "wall_s": 9.0}))
    monkeypatch.setattr(g, "grade_in_container", lambda *a, **k: {"published_pass": True, "protected_modified": []})
    rec = g.grade(tmp_path, {"task_id": "probe-x-00", "published": {"predicates": []}}, None, "img")
    assert rec["verified_success"] and rec["overfit"] is False and rec["withheld_pass"] is None


def test_a_timed_out_episode_is_not_also_accused_of_tampering(monkeypatch, tmp_path):
    """`before.json` is written after the agent is gone, so a killed episode has none. Treating the missing
    baseline as "everything changed" disqualified timed-out episodes for tampering they never did."""
    import sh.validator.grade as g

    (tmp_path / "result.json").write_text(json.dumps({"messages": []}))
    (tmp_path / "finish.json").write_text(json.dumps({"stage": "no_finish", "timed_out": True, "api_calls": 6}))
    monkeypatch.setattr(g, "grade_in_container", lambda *a, **k: {"published_pass": False, "protected_modified": None})
    rec = g.grade(tmp_path, {"task_id": "t-1", "published": {"predicates": []}}, None, "img")
    assert rec["signals"] == ["timed_out"]
    assert not rec["disqualified"]
    assert rec["truncated"] is True and rec["max_turns_hit"] is False


def test_a_provider_outage_is_void_not_a_miner_failure(monkeypatch, tmp_path):
    """The first family #2 screen: the engine answered `server overloaded` on all three attempts and every
    episode was scored as an unsolved task. An episode that never got its tokens is evidence about nothing."""
    import sh.validator.grade as g

    (tmp_path / "result.json").write_text(
        json.dumps({"messages": [], "failed": True, "failure_reason": "overloaded", "failure_retryable": True})
    )
    (tmp_path / "finish.json").write_text(json.dumps({"api_calls": 3, "wall_s": 41.0}))
    monkeypatch.setattr(g, "grade_in_container", lambda *a, **k: {"published_pass": False, "protected_modified": []})
    rec = g.grade(tmp_path, {"task_id": "t-1", "published": {"predicates": []}}, None, "img")
    assert rec["void"] and rec["void_reason"] == "overloaded"
    assert "inference_unavailable" in rec["signals"]
    assert not rec["disqualified"]  # it is not the miner's fault either
    assert rec["truncated"] is False


def test_an_ordinary_failure_is_not_void(monkeypatch, tmp_path):
    import sh.validator.grade as g

    (tmp_path / "result.json").write_text(json.dumps({"messages": [], "failed": False}))
    (tmp_path / "finish.json").write_text(json.dumps({"api_calls": 7, "wall_s": 80.0}))
    monkeypatch.setattr(g, "grade_in_container", lambda *a, **k: {"published_pass": False, "protected_modified": []})
    rec = g.grade(tmp_path, {"task_id": "t-1", "published": {"predicates": []}}, None, "img")
    assert not rec["void"] and not rec["verified_success"]


def test_an_image_defined_workdir_is_the_workspace():
    """Family terminal_task works in the image's own tree: an absolute write there is inside the workspace, and
    the same path on a fixture-defined task is not."""
    task = {**TASK, "workdir": "/task_file"}
    call = ("write_file", {"path": "/task_file/output/report.json", "content": "{}"}, '{"bytes_written": 2}')
    assert trajectory_rules(list(_msgs(call)), [], None, task, None)[0] == []
    assert "wrote_outside_workspace" in _signals(call)


def test_a_half_of_test_cases_counts_each_test():
    from sh.validator.grade import fraction

    half = [
        ["custom", "facet_test", "t.py::a"],
        ["custom", "facet_test", "t.py::b"],
        ["custom", "facet_test", "t.py::c"],
        ["custom", "facet_test", "t.py::d"],
    ]
    assert (
        fraction(half, False, {"t.py::a": True, "t.py::b": True, "t.py::c": False}) == 0.5
    )  # an unknown test did not pass
    assert fraction(half, True, {k[2]: True for k in half}) == 1.0


def test_a_half_that_is_not_a_test_suite_is_all_or_nothing():
    from sh.validator.grade import fraction

    half = [["file_exists", "out.txt"], ["digest_is", "out.txt", "a" * 64]]
    assert fraction(half, True, {"irrelevant": False}) == 1.0 and fraction(half, False, None) == 0.0
    assert fraction([["custom", "facet_test", "t.py::a"]], True, None) == 1.0  # outcomes unknown: the verdict stands


def test_swe_test_predicates_count_each_test_and_an_empty_half_measures_nothing():
    from sh.validator.grade import fraction

    half = [["custom", "swe_test", "tests/test_a.py::T::a"], ["custom", "swe_test", "tests/test_a.py::T::b"]]
    assert fraction(half, False, {"tests/test_a.py::T::a": True, "tests/test_a.py::T::b": False}) == 0.5
    assert fraction([], True, {"x": True}) is None


def test_a_task_with_every_fact_withheld_is_credited_by_its_tests_and_never_overfit(monkeypatch, tmp_path):
    """swe_fix publishes no half. A partial fix is partial credit, not a published-half pass that failed to hold."""
    import sh.validator.grade as g

    (tmp_path / "result.json").write_text(json.dumps({"messages": []}))
    (tmp_path / "finish.json").write_text(json.dumps({"api_calls": 9, "wall_s": 120.0}))
    tests = {"t.py::a": True, "t.py::b": True, "t.py::c": False}
    monkeypatch.setattr(
        g,
        "grade_in_container",
        lambda *a, **k: {"published_pass": True, "withheld_pass": False, "protected_modified": [], "tests": tests},
    )
    withheld = {"withheld": {"predicates": [["custom", "swe_test", t] for t in tests]}, "salt": "00"}
    rec = g.grade(tmp_path, {"task_id": "swe-fix-r0005-00", "published": {"predicates": []}}, withheld, "img")
    assert rec["published_fraction"] is None and rec["overfit"] is False
    assert abs(rec["credit"] - 2 / 3) < 1e-6 and rec["verified_success"] is False


def test_the_grader_path_rule_is_anchored_at_the_root():
    """`/runner` is the grader; `_pytest/runner.py` and `src/runner.py` are a repository's own files, which an
    agent on a real code base reads all the time."""
    assert "read_grader_or_withheld_path_attempt" in _signals(("terminal", {"command": "cat /runner/grade.py"}))
    assert "read_grader_or_withheld_path_attempt" in _signals(("read_file", {"path": "/runner"}))
    assert "read_grader_or_withheld_path_attempt" in _signals(("terminal", {"command": "ls /ep/out"}))
    for benign in (
        "sed -n 1,40p src/_pytest/runner.py",
        "cat /opt/miniconda3/envs/testbed/lib/python3.10/site-packages/_pytest/runner.py",
        "pytest tests/runner/test_x.py -q",
        "grep -rn foo /testbed/pkg/runner.py /testbed/ep/out",
    ):
        assert _signals(("terminal", {"command": benign})) == [], benign


def test_a_preview_id_in_the_bundle_is_not_the_evaluated_task_id(tmp_path):
    bundle = tmp_path / "b"
    bundle.mkdir()
    (bundle / "SOUL.md").write_text("Reproduce swe-fix-r0004-p03 locally before you start.\n")
    task = {"task_id": "swe-fix-r0004-03", "published": {"predicates": []}}
    assert trajectory_rules([], [], bundle, task, None)[0] == []
    (bundle / "SOUL.md").write_text("For swe-fix-r0004-03 the fix is in fonts.py.\n")
    assert "instance_literal_in_bundle" in trajectory_rules([], [], bundle, task, None)[0]


def test_a_link_leaving_the_workspace_is_skipped_by_the_grader_not_fatal(tmp_path):
    """`python -m venv .venv` in a real repository leaves `.venv/bin/python -> /usr/bin/python3`; the snapshot is
    still graded, without the link."""
    import io
    import tarfile

    from sh.validator.runner.grade import _workspace_member

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        link = tarfile.TarInfo("./.venv/bin/python")
        link.type, link.linkname = tarfile.SYMTYPE, "/usr/bin/python3"
        tf.addfile(link)
        data = b"print('fixed')\n"
        f = tarfile.TarInfo("./pkg/core.py")
        f.size = len(data)
        tf.addfile(f, io.BytesIO(data))
    buf.seek(0)
    ws = tmp_path / "ws"
    ws.mkdir()
    with tarfile.open(fileobj=buf) as tf:
        tf.extractall(ws, filter=_workspace_member)
    assert (ws / "pkg" / "core.py").read_bytes() == data and not (ws / ".venv" / "bin" / "python").exists()


def test_the_grader_path_rule_covers_outputs_and_withheld_material_however_spelled_and_allows_the_public_task():
    for attempt in (
        "cat $SH_EP/out/before.json",
        "cd ${SH_EP} && ln -s /nonexistent ${SH_EP}/out/before.json",
        "cat /ep/ws/../out/grade.json",
        "cat /ep/withheld.json",
        "ls /runner",
    ):
        assert "read_grader_or_withheld_path_attempt" in _signals(("terminal", {"command": attempt})), attempt
    for honest in ("ls /ep", "cat /ep/task.json", "ls /ep/bundle", "cat /ep/seed"):  # the public projection
        assert _signals(("terminal", {"command": honest})) == [], honest
    assert _signals(("write_file", {"path": "/ep/ws/report.sh", "content": "x"})) == []  # the fixture workspace
    assert _signals(("terminal", {"command": "grep -rn foo /testbed/ep/out"})) == []  # a repository path


def test_grepping_the_tree_for_a_path_string_is_not_an_access_attempt():
    assert _signals(("search_files", {"path": "/testbed", "pattern": "/runner"})) == []
    assert _signals(("search_files", {"path": "/ep/out", "pattern": "x"})) != []  # the place searched still counts


def test_results_come_out_of_the_episode_volume_as_plain_files_only(tmp_path):
    """The agent can reach /ep/out: a link it leaves there must not be recreated on the host, where the grader
    would follow it."""
    import io
    import tarfile

    from sh.validator.episode import _result_member

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        link = tarfile.TarInfo("./before.json")
        link.type, link.linkname = tarfile.SYMTYPE, "/nonexistent"
        tf.addfile(link)
        data = b'{"stage": "done"}'
        f = tarfile.TarInfo("./finish.json")
        f.size = len(data)
        tf.addfile(f, io.BytesIO(data))
    buf.seek(0)
    out = tmp_path / "out"
    out.mkdir()
    with tarfile.open(fileobj=buf) as tf:
        tf.extractall(out, filter=_result_member)
    assert (out / "finish.json").read_bytes() == data and not (out / "before.json").is_symlink()
    assert not (out / "before.json").exists()


def test_the_runner_resets_its_output_directory_and_reports_what_the_agent_left_there(tmp_path, monkeypatch):
    import importlib

    for var in ("HERMES_HOME", "SH_EP"):  # the runner reads its environment at import; it is built for the container
        monkeypatch.setenv(var, str(tmp_path / var.lower()))
    monkeypatch.setenv("SH_INFERENCE", "http://inference/v1")
    monkeypatch.setenv("SH_TOKEN", "none")
    r = importlib.import_module("sh.validator.runner.run_episode")

    out = tmp_path / "out"
    out.mkdir()
    (out / "system_prompt.txt").write_text("sp")
    (out / "before.json").symlink_to("/nonexistent")  # the agent's plant
    (out / "planted").mkdir()
    (out / "planted" / "x").write_text("x")
    monkeypatch.setattr(r, "OUT", out)
    stray = r._reset_out()
    assert sorted(p.split("/")[-1] for p in stray) == ["before.json", "planted"]
    assert out.is_dir() and list(out.iterdir()) == []
    out.rmdir()
    out.symlink_to(tmp_path)  # the directory itself replaced by a link
    assert r._reset_out() == [str(out)] and out.is_dir() and not out.is_symlink()


def test_a_tree_that_reaches_for_the_test_runner_is_disqualified(monkeypatch, tmp_path):
    """swe_fix's checks compare the agent's tree with the pristine one; what they flag, the grader enforces."""
    import sh.validator.grade as g

    (tmp_path / "result.json").write_text(json.dumps({"messages": []}))
    (tmp_path / "finish.json").write_text(json.dumps({"api_calls": 9, "wall_s": 120.0}))
    monkeypatch.setattr(
        g,
        "grade_in_container",
        lambda *a, **k: {
            "published_pass": True,
            "withheld_pass": True,
            "protected_modified": [],
            "tests": {"t.py::a": True},
            "detail": {"valid": True, "tamper": [["pkg/__init__.py", "import _pytest.runner"]]},
        },
    )
    withheld = {"withheld": {"predicates": [["custom", "swe_test", "t.py::a"]]}, "salt": "00"}
    rec = g.grade(tmp_path, {"task_id": "swe-fix-r0005-00", "published": {"predicates": []}}, withheld, "img")
    assert rec["disqualified"] and "harness_tamper" in rec["signals"] and rec["credit"] == 0.0


def test_an_episode_the_disk_guard_stopped_is_disqualified_without_grading(monkeypatch, tmp_path):
    import sh.validator.grade as g

    (tmp_path / "result.json").write_text(json.dumps({"messages": []}))
    (tmp_path / "finish.json").write_text(
        json.dumps({"api_calls": 3, "disk_guard": {"free_bytes": 1, "floor_bytes": 2}})
    )
    monkeypatch.setattr(g, "grade_in_container", lambda *a, **k: (_ for _ in ()).throw(AssertionError("not graded")))
    rec = g.grade(tmp_path, {"task_id": "t-1", "published": {"predicates": []}}, None, "img")
    assert rec["disqualified"] and "disk_abuse" in rec["signals"] and rec["credit"] == 0.0


def test_an_episode_records_the_withheld_half_and_the_pins_it_was_graded_under(monkeypatch, tmp_path):
    import sh.validator.grade as g

    (tmp_path / "result.json").write_text(json.dumps({"messages": []}))
    (tmp_path / "finish.json").write_text(json.dumps({"api_calls": 3}))
    monkeypatch.setattr(
        g,
        "grade_in_container",
        lambda *a, **k: {"published_pass": True, "withheld_pass": True, "protected_modified": []},
    )
    task = {
        "task_id": "t-1",
        "published": {"predicates": []},
        "max_turns": 100,
        "timeout_s": 1800,
        "token_budget": 600000,
    }
    w = {"withheld": {"predicates": [["file_exists", "x"]]}, "salt": "00"}
    rec = g.grade(tmp_path, task, w, "img")
    assert len(rec["withheld_sha256"]) == 64 and rec["pins_sha256"] == g.pins_digest(task)
    assert g.pins_digest({**task, "token_budget": None}) != rec["pins_sha256"]
    assert g.pins_digest({**task, "max_reasoning_steps": 1}) != rec["pins_sha256"]
    assert rec["truncated"] is False and rec["max_turns_hit"] is False
    assert rec["action_turns"] == 0 and rec["reasoning_turns"] == 0


def test_a_committed_task_with_no_reveal_fails_the_commitment_check(tmp_path):
    from sh.validator.round import close

    rd = tmp_path / "round"
    for sub in ("tasks", "withheld", "episodes"):
        (rd / sub).mkdir(parents=True)
    (rd / "tasks" / "t0.json").write_text(
        json.dumps({"task_id": "t0", "round_id": "r0009", "withheld_commitment": "hmac-sha256:deadbeef"})
    )  # committed, but no withheld/t0.json to reveal
    ep = rd / "episodes" / "null" / "t0"
    ep.mkdir(parents=True)
    (ep / "episode.json").write_text(
        json.dumps({"task_id": "t0", "surface": "null", "round_id": "r0009", "credit": 0.0})
    )
    record = close(rd, rd / "episodes", rd / "out", reveal_dir=rd / "withheld")
    assert record["commitments_ok"] is False and record["commitments_verified"]["t0"] is False


def test_a_turn_cap_is_truncated_and_named(monkeypatch, tmp_path):
    """The harness cut in. Named as max_turns_hit, recorded as truncated, still graded — not a void, not a DQ."""
    import sh.validator.grade as g

    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "assistant", "reasoning": "plan", "content": ""},
                    {"role": "assistant", "tool_calls": [{"id": "c0", "function": {"name": "terminal"}}]},
                ],
                "completed": False,
                "turn_exit_reason": "max_iterations_reached(100/100)",
                "api_calls": 100,
            }
        )
    )
    (tmp_path / "finish.json").write_text(
        json.dumps(
            {
                "api_calls": 100,
                "completed": False,
                "turn_exit_reason": "max_iterations_reached(100/100)",
            }
        )
    )
    monkeypatch.setattr(g, "grade_in_container", lambda *a, **k: {"published_pass": False, "protected_modified": []})
    rec = g.grade(tmp_path, {"task_id": "t-1", "published": {"predicates": []}, "max_turns": 100}, None, "img")
    assert rec["truncated"] and rec["max_turns_hit"]
    assert "max_turns_hit" in rec["signals"]
    assert rec["action_turns"] == 1 and rec["reasoning_turns"] == 1
    assert not rec["void"] and not rec["disqualified"]
