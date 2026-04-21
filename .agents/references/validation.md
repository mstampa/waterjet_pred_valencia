# Validation Reference

Load when: any code change, behavior change, or test/validation decision.

## General
- Test suite lives in `tests/`.
- Use narrowest meaningful validation first, then widen only as needed.
- Tests support development but are not sole proof of scientific validity.

## Standard Commands
- CLI smoke check: `python -m waterjet_pred_valencia.cli --help`
- Targeted test: `pytest tests/test_<name>.py`
- Full test suite: `pytest`
- Lint: `ruff check .`
- Format when needed: `ruff format <touched-files>`

## Validation Expectations
- After code changes, run the smallest command that exercises changed behavior.
- If Python files changed, run `ruff check` on touched files or whole repo when appropriate.
- If public behavior, CLI output, or numerical flow changed, consider whether pytest coverage should change too.
- If validation is skipped, say why.
