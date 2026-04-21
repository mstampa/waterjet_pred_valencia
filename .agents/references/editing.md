# Editing Reference

Load when: making any non-trivial file edit.

## Editing Priorities
- Prefer smallest correct change.
- Keep code close to existing structure unless current structure blocks correctness.
- Do not add abstractions unless they remove real duplication or materially clarify numerical logic.
- Remove stale template wording instead of adapting repo around it.

## Scientific Code Bias
- Preserve meaning of physical quantities, units, and parameter names.
- Do not silently change equations, constants, or integration behavior without reading nearby context.
- When changing numerical logic, keep diffs easy to audit.

## Documentation And Comments
- Prefer concise docstrings for public functions, CLI helpers, and non-obvious numerical helpers.
- Do not add tutorial-style comments for straightforward code.
- Add comments only where physical or numerical intent is otherwise hard to infer.
- Update affected docstrings when behavior or interfaces change.
