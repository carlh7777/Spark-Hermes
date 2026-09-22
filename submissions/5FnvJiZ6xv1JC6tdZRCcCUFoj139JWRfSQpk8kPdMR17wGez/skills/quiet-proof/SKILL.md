---
name: quiet-proof
description: Prove a fix with a scratch script and a narrow pytest run, keep the output small, and revert any edit that fails a test which used to pass. Open before the first test run.
---

# Proof that does not spend the budget

Every line a tool prints is re-sent on every later step. Test output is the largest thing you will produce.

## Keep it small

- Always `-q`. First look: `--tb=line`. Re-run a single failure with `--tb=short` only when you need the frame.
- `-x` while iterating, so one failure stops the run.
- Select with `-k` or a node id. Do not run a file to see one test.
- `-p no:cacheprovider` so the run does not write a cache into the tree.
- Never print a whole file. Read a slice around the line. `search_files` for a symbol instead of dumping a module.
- Never redirect a suite into a file in the repository and read it back.

## If pytest is not there

`No module named pytest` is not a problem to solve. There is no network, and an install ends the episode at zero. Fall back to `/tmp/repro.py`: import the package from `/testbed` and call the path the issue describes. That is the proof. Do not hunt for another interpreter.

## Order

1. **Repro.** `/tmp/repro.py`, run from `/testbed`. It must show the bug before the edit and be clean after. The tests written for this bug are not in the tree. Do not look for them and do not add one.
2. **The surviving tests, before you edit if you still have steps.** `search_files` the test tree for the symbol the issue names. What those tests assert is the contract, in exact values.
3. **The file that covers your module.** `<pkg>/<area>/<module>.py` is covered by `tests/<area>/test_<module>.py` or `tests/test_<module>.py`. This is the run that matters: grading draws from here.
4. **Neighbors in the same area**, then the widest set you can still afford, `-q --tb=line`. A partial sweep you read beats a full sweep that ends the episode.

## A new failure

Ask whether this test passed before your change.

- If your change caused it, the edit is too wide or wrong. Narrow it, or `git checkout -- <path>` and try a smaller edit. The smallest diff that keeps the repro clean and the neighbors green is the one to leave in the tree.
- If it failed before you touched anything, leave it. Say so. Fixing a pre-existing failure is an extra edit and extra risk.

`git diff` is the list of what you changed. To tell a regression from a pre-existing failure, restore the file, re-run that one test, then restore your fix. There is no stash and no second commit.

A new **warning** can be the failure. Some of these projects turn warnings into errors, so a deprecation your edit raises fails tests that do not touch your logic. If unrelated tests go red after a small edit, suspect a warning before you suspect the logic.

One project treats expected-failure marks as strict: a test that was supposed to fail and now passes is a failure. If a sweep goes red on a test you did not mean to affect, revert the extra part of the edit.

## Never

- Do not add, edit, move, or delete a test, a `conftest.py`, `pytest.ini`, `tox.ini`, `setup.cfg`, the pytest section of `pyproject.toml`, or any golden or fixture file. It scores zero.
- Do not run a flag that rewrites expected output.
- Do not install anything.
- Scratch files go in `/tmp`, not in `/testbed`.

## Before you stop

- `git diff` contains only source lines you meant.
- The repro is clean.
- The covering test file passes, or you reverted the edit that broke it.
- If the budget ends first, the fix that is already in the tree still counts. Stop. Do not start another investigation.
