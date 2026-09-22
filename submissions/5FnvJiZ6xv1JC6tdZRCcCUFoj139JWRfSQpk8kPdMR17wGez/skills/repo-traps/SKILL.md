---
name: repo-traps
description: Layout, test command, and the trap that zeros a task for sqlglot, python-docx, python-pptx, cantools, pygments, astroid, oauthlib, and gpxpy. Open once you know which repository you are in, and read only that section.
---

# Which tree, and what zeros it

One call: `ls /testbed`. Source is a top-level package or under `src/`. Read the matching section only.

Two facts before you edit: where the module in the issue lives, and which test file covers it. The tests kept for grading are drawn mostly from that file. A new failure there zeros the task even if your repro is clean.

## sqlglot

- Source `sqlglot/`, tests `tests/`. Tokenizer, `parser.py`, `generator.py`, expression classes, `optimizer/` (one module per rule), `dialects/` (one module per dialect).
- A dialect nests its own tokenizer, parser, and generator. The same method on a sibling dialect is the reference for a mutated one. Parsers key off dicts of functions and statement parsers; generators key off a transforms dict and per-node methods.
- `tests/fixtures/` holds golden SQL the tests compare against. Never edit those files. A failing fixture means the fix is wrong.
- Iterate with `python -m pytest tests/test_optimizer.py -q -k <name>`. The optimizer file is slow. Run the file once at the end.

## python-docx

- Source `src/docx/`, tests `tests/` mirroring the package. Ignore `features/` — that suite is not pytest.
- A public method usually delegates to a `CT_*` element class under `oxml/`. If the public method looks right, the bug is one level down, in the element class or in `oxml/xmlchemy.py` descriptors.
- **Trap:** pytest sets warnings to errors. A new deprecation or resource warning fails unrelated tests. Do not introduce a warning, and do not silence one.
- `python -m pytest tests/test_table.py -q`, or the mirrored path of the module you touched.

## python-pptx

- Source `src/pptx/`, tests `tests/` mirroring it. `features/` is not pytest. Same `oxml/` element layer as python-docx.
- Same rule: if the public method looks right, read the element class it delegates to. A rewritten method that still "looks like" the public API is usually wrong one level down.
- **Trap:** warnings are errors. A new warning fails tests that have nothing to do with your change.
- `python -m pytest tests/test_<module>.py -q` at the mirrored path.

## cantools

- Source `src/cantools/`. Database objects live in `database/can/` (`database.py`, `message.py`, `signal.py`, `node.py`, `bus.py`) and `database/can/formats/`. The CLI is `subparsers/`.
- Subcommands print. Their tests compare **exact stdout**: whitespace, column order, and line breaks are the contract. A missing field in the output is a deleted assignment or a dropped keyword on the object that builds that line, not a formatting tweak.
- Several independent deletions in one module are normal. After the first, read every function in that file for a name that is used and never assigned.
- pytest config is in `tox.ini` and adds `-v`. Pass `-q` yourself.
- `python -m pytest tests/test_list.py -q`. `tests/test_database.py` is large — use `-k`.

## pygments

- Source `pygments/`, tests `tests/`. A lexer is a `tokens` dict of regex-to-token rules. The mutation is usually one rule, or the two lexers passed to `super().__init__` on a delegating lexer (root lexer, then language lexer — swapping them is a common edit).
- **Trap:** most of the suite is snapshots built from `tests/snippets/` and `tests/examplefiles/`. A change to a shared lexer or to `lexer.py` fails many of them. The suite can rewrite those files. **Never use that flag.** It edits the tests and scores zero.
- `python -m pytest tests/test_basic_api.py -q`, then the snippet directory for the lexer you touched. Do not start with the whole suite.

## astroid

- Source `astroid/`, tests mostly flat under `tests/` (`test_inference.py`, `test_nodes.py`, `test_builder.py`, `test_scoped_nodes.py`), plus `tests/brain/`.
- Inference is generator-based. `infer` yields, and yields an uninferable sentinel when it cannot decide. Turning a `yield` into a `return`, dropping that sentinel, or reversing yield order looks harmless and breaks many tests. Node classes list their children on a fields attribute; compare a node to its siblings in the same file.
- A copy or deepcopy that recurses forever is an attribute hook on the proxy (`__getattr__`, `__copy__`, `__deepcopy__`, `__getstate__`) calling back into itself. Read that hook against the sibling class, not against the method's own comments.
- **Two traps, and either one zeros the task.** Warnings are errors. And expected-failure tests are strict: a test marked expected-to-fail that starts passing is a failure. Fix only what the issue asks. Extra cleverness here breaks the suite by making tests succeed.
- `python -m pytest tests/test_inference.py -q`, or the file for the area you touched.

## oauthlib

- Source `oauthlib/`, tests `tests/` mirroring it. Protocol trees: `oauth1/`, `oauth2/rfc6749/` (grant types on a shared `base.py`, plus clients, endpoints, tokens, errors), and `openid/`.
- The contract is exact strings: parameter names, encoding, header spelling, error slugs, status codes. Grant types are near-copies. The sibling grant and `base.py` are the reference for a mutated one. A changed constant in `errors.py` shows up as the wrong error string far away.
- Mirror the path: `oauthlib/oauth2/rfc6749/grant_types/x.py` is covered by `tests/oauth2/rfc6749/grant_types/test_x.py`.
- `python -m pytest tests/oauth2 -q`, or that mirrored file.

## gpxpy

- Source is the top-level `gpxpy/` package (`gpx.py` holds the track, segment, point, waypoint, and route classes; also `parser.py`, `geo.py`). Tests are often a single `test.py` at the repo root.
- The same method exists on several element classes (a waypoint, a track point, a route point). An edit to one is often the same edit on its siblings. If the issue names one class and says others are affected, fix every sibling that defines that method, and stop there.
- Fields are often `None` when cleared. A method that writes `0`, `""`, or `False` where a sibling writes `None` is the edit. Serialization compares exact XML text, so a wrong sentinel changes the file, not just the attribute.
- `python -m pytest test.py -q -k <name>`.

## None of these

Source at the top level or under `src/`, tests mirroring it under `tests/`. Golden files live in the test tree and are never edited. Run the test file whose name matches your module.
