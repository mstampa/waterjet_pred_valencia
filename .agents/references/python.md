# Python Reference

Load when: touching Python source, tests, typing, numerical code, CLI behavior, or runtime error handling.

## Python Style
- Use explicit typing.
- Use Google-style docstrings.
- Follow existing style in touched file when repo is not fully uniform.
- Use 4-space indentation.
- Prefer module-level imports. Use local imports only for cycle breaking or startup-cost reasons.
- Group imports as: standard library, third-party, local package.
- Keep blank line between import groups.
- Use `snake_case` for modules, functions, methods, variables, and parameters.
- Use `PascalCase` for classes.
- Prefer concrete type hints when they help numerical code, for example `NDArray[np.floating]`.
- Prefer short, direct helpers over framework-like indirection.

## Numerical Code
- Keep units and parameter semantics explicit.
- Prefer names that match existing physical-model terminology.
- Avoid refactors that make equations harder to compare against paper, traces, or tests.
- When changing numerical flow, preserve inspectability for debugging and plotting.

## CLI And Logging
- Keep CLI behavior straightforward and script-friendly.
- When changing argument names or defaults, consider impact on `waterjet-pred-valencia` entrypoint and tests.
- Log enough context to debug failing runs without flooding routine output.

## Formatting
- Use `ruff` for linting and formatting.
- Lint all: `ruff check .`
- Lint and autofix: `ruff check . --fix`
- Import ordering only: `ruff check . --select I`
- Format touched files when needed: `ruff format <touched-files>`
- If project is being run through `uv`, `uv run ...` is fine; otherwise call tools directly.
- Keep lines readable and avoid unrelated reformatting.
- Use trailing commas in multiline literals or calls when they improve diffs.
- Keep comments rare and high-signal.

## Error Handling
- Raise early for invalid startup configuration or unsupported modes.
- In long-running simulation paths, prefer errors that preserve diagnostic value.
- When logging exceptions, include enough context to identify failing parameters, state transitions, or output stage.
- Do not add defensive validation for every field when upstream data is controlled.
