---
name: missed-edits
description: Search for injected edits that look like normal code — a deleted assignment, statements moved around a return, or a whole function replaced by fluent wrong code. Open after the repro fails and before the first edit.
---

# Edits that survive a careful read

The bug is a mechanical edit to working code. The line that looks strange is the easy one, and it is usually already fixed by anyone who reads the function. What remains is an edit that still matches the surrounding style. Work the three searches below on the function the traceback named, then on every other function in that file.

## 1. A deleted assignment or a dropped argument

Nothing on the screen is misspelled. The giveaway is a name that is read and never written, or a call that passes fewer arguments than its siblings.

- `search_files` each name the issue quotes. Count assignments against uses in that function. A use with no assignment on that path is the edit. `UnboundLocalError` and `NameError` are this shape.
- A parameter overwritten before it is read is the same edit from the other side.
- Compare every call to the signature it targets, **by name, not by position**. One edit can reorder arguments and drop one. Fixing the order does not finish the site. Two arguments of the same type swapped in a `super().__init__` call is common.
- When the issue says a result is **missing a field** rather than crashing, open the function that builds that object and read it statement by statement. A deleted block often leaves `pass`, or a constructor that no longer passes a keyword its sibling still passes.
- A class missing a method its siblings define is the same shape at class level. Callers of the missing name tell you the signature to restore. Derive it from those callers and from the sibling, not from memory.

Several deletions in one file are independent. After the first fix, repeat this search on the rest of the file. Each one you restore is credit.

## 2. Statements moved, not rewritten

- Search the function for `return`. Read what follows it. Statements stranded after a return never run. An early `return` inserted above the real body does the same thing.
- An assignment moved below its first use is the same family. The error names the variable; find that assignment and see whether it still runs before every use.
- A line that belonged in one branch now runs in the other, or the two bodies were swapped while the condition text stayed put. For every `if`, ask whether the body matches the condition, not whether the condition parses.
- A `continue` or a filter removed, so a loop now handles items it should skip. Paired filters (keep this, drop that) — if one of the pair is gone, that is the edit.
- In generator-style inference, a `yield` turned into a `return`, a sentinel branch removed, or the yield order reversed. Compare the function to the sibling that infers a similar node.

If one site is fixed and the repro still fails, look for the same shuffle in a second file of the same module family before you doubt the fix.

## 3. A function replaced by fluent wrong code

The body is coherent. Names are plausible. Comments may even match. It still does the wrong thing, so there is no odd line to spot.

Rebuild the contract from the docstring, the type hints, and what callers do with the result. Then check the body against that contract one line at a time: order of operations, which value is returned, boundary behavior on empty or `None`, exact strings. A sibling function that does the same job is the second copy of the contract.

Attribute hooks are part of the body. If copying or printing the object recurses, read `__getattr__`, `__copy__`, `__deepcopy__`, and `__getstate__` against the sibling class. A hook that handles a missing attribute by asking for the same attribute will not look wrong in isolation.

## Before you change a line

One sentence: line L of function F in file P should do X and instead does Y. If you cannot say it, apply the single strongest candidate from the three searches anyway. Try candidates one at a time and re-run the repro between them.

Do not "fix" a line you cannot justify. Unusual is not wrong, and a neighbor you break zeros the task, including the sites you already got right.
