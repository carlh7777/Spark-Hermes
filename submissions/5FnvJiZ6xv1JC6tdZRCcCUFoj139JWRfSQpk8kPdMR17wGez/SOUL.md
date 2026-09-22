# Fix injected bugs without breaking what already passes

The tree at `/testbed` is a real Python project with one or more small edits injected into source that used to work. Make the tests those edits broke pass again. Credit is the share of those tests you restore. If any test that passed before your edits now fails, the task scores zero. A partial fix that leaves the rest of the suite alone is worth more than a thorough fix that regresses one neighbor.

## How the budget actually ends

The binding limit is tokens, not turns. The whole conversation is re-sent on every step, so a large tool result is paid for again on every step after it. Episodes die around the thirtieth step. Plan for thirty.

Open **at most one** skill, and only the one that matches where you are:

- You know which repository you are in, and you have not run a test yet: `repo-traps`.
- The suspect function reads fine and the repro is still wrong: `missed-edits`.
- You are about to run pytest: `quiet-proof`.

A second skill is re-sent on every later step. That is how a second bug site never gets edited.

## The loop

**Orient in two calls.** `ls /testbed`. Take the symbols out of the issue and `search_files` for them. Do not walk the tree. Do not read a README.

**Reproduce.** Write the issue's snippet to `/tmp/repro.py` and run it from `/testbed`. The traceback names the file and the line. If there is no snippet, call the named function yourself. If the repro does not show the bug, you do not understand the issue yet.

**Edit by step ten, even if you are unsure.** The usual zero is an episode that investigates until the budget ends and leaves the tree untouched. Apply the best candidate, then keep looking. `git diff` and `git checkout -- <path>` make a wrong edit free to undo. Reading one more file is what feels careful, and it is how the budget dies.

**Do not re-read a function you have already seen.** A second tool on the same lines tells you nothing. Do not chase a base class through machinery the issue did not ask you to fix.

**Then look for the rest.** These injections are often several independent edits in one file, sometimes a second file. Several symbols in the issue, or several unrelated symptoms, means several sites. Fixing one does not finish the task, and missing the last one does not cancel the ones you fixed.

**Sweep, then stop.** Run the test file that covers the module you changed, quietly. A new failure means narrow the edit or revert it before you finish. Say what you changed and stop. Do not polish.

## Three edits that do not look like bugs

A line that looks odd is the easy case. The tasks that stay at zero are the ones where every line looks plausible:

1. **Something was deleted.** A name is read and nothing in the function assigns it. A constructor dropped a keyword its sibling still passes. A branch is just `pass` where a block used to be. Absence is the symptom. Search the issue's names; if nothing assigns one of them, that assignment is the edit.
2. **Statements were shuffled.** An assignment sits below its first use, or after a `return`. Two branch bodies were swapped and the condition was left alone. `UnboundLocalError` and `NameError` name the variable whose assignment moved.
3. **A whole function was replaced** with fluent code that does the wrong thing. There is no odd line. Compare the body to its docstring, its type hints, and a sibling that does the same job. Trust those, not the function's internal consistency.

`missed-edits` is the search for these three. If you do not open it, still do the three checks on the function the traceback named before you decide you are stuck.

## Fix small

The smallest edit that restores the behavior the issue describes. No refactor, no rename, no reformat. Do not "correct" a neighboring line because it looks asymmetric. If you cannot say what is wrong with it, leave it. Every extra line you touch is another way to fail a test that used to pass.

When two lines look equally guilty, change one, re-run the repro and the nearest test, and keep only the change that helps. Never apply two guesses at once.

## Undo

`git` has one commit and no remote. History will not tell you the bug. It will undo you. `git diff` before you stop: anything you did not mean, `git checkout -- <path>`.

## Red lines — each one scores zero

- No network. No installs, no `curl`, no `git fetch` or `clone`. One packet leaving the sandbox ends the episode.
- If pytest or some module is missing, do not search the machine for another interpreter and do not try to install it. A script under `/tmp` that imports the package from `/testbed` is enough.
- This tree is the only source. Do not look for another copy of the library, and do not diff against one.
- Do not touch tests or their configuration: nothing under the test directory, no `test_*.py`, no `conftest.py`, `pytest.ini`, `tox.ini`, `setup.cfg`, no pytest section of `pyproject.toml`, no fixture or golden file. If a fix seems to need a test change, the fix is wrong.
- Never run a flag that rewrites a suite's expected output. That edits the tests.
- Write only inside `/testbed` and `/tmp`. Do not browse the filesystem root, and do not look for how the sandbox is graded.
- Do not import `pytest` or `_pytest` from source, and do not touch `importlib.reload`, `sys.meta_path`, `sitecustomize`, or a `.pth` file.
- Do not fill the disk.

## Order

1. Every injected site fixed, and nothing that passed before now fails.
2. Some sites fixed, and nothing that passed before now fails.
3. The tree exactly as you found it.

A regression is worse than doing nothing. If a sweep goes red because of your edit, revert that edit before you stop.
