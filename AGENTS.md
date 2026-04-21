# Agent Guide

## Always Load
- Read `.agents/project.md` at start of every conversation.
- Load `caveman` skill for every conversation in this repo.

## Load Additional References Only When Needed
- Load `.agents/references/editing.md` for any non-trivial file edit.
- Load `.agents/references/validation.md` for any code change, behavior change, or test decision.
- Load `.agents/references/python.md` when touching Python source, tests, typing, numerical logic, CLI behavior, or public APIs.

## Preferred Skills
- Prefer `python-pro` for Python implementation and refactors.
- Prefer `python-testing-patterns` when adding or reshaping pytest coverage.
- Prefer `documentation-templates` when restructuring `README.md` or other user-facing docs.
- Ignore unrelated skills unless task clearly needs them.

## Repo-Wide Rules
- Prefer smallest correct change.
- Keep instructions concrete and low-ambiguity. Write for fast agent loading, not for humans browsing casually.
- API changes are allowed when they improve clarity, safety, performance, or numerical correctness.
- Inputs are usually controlled. Do not add defensive validation, fallback paths, or compatibility shims unless explicitly requested or clearly required by existing behavior.
- Remove template leftovers that do not match this repo instead of preserving generic wording.
