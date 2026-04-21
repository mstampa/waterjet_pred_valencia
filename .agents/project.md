# Project Overview

## Purpose
- `waterjet_pred_valencia` implements a physical model of large water jets, or fire streams.
- Repo reproduces and experiments with Valencia et al. trajectory model from `10.1007/s10694-021-01175-1`.
- This repo is part of the user's PhD work.
- User is sole active developer. Original model author is only other expected user.

## Current Reality
- `README.md` states model is still work in progress and ODE system can fail before full integration.
- Treat numerical behavior as scientifically sensitive. Preserve model intent and parameter meaning unless task explicitly asks for physical-model changes.
- Prefer correctness and traceability over generic abstractions.

## Codebase Shape
- Python requirement in `pyproject.toml`: `>=3.10`.
- Main package: `src/waterjet_pred_valencia/`.
- Core simulation logic: `simulator.py`, `parameters.py`, `jet_state.py`, `tracer.py`.
- CLI entrypoint: `src/waterjet_pred_valencia/cli.py` and script `waterjet-pred-valencia`.
- Plotting helpers live under `src/waterjet_pred_valencia/plotting/`.
- Tests live in `tests/`.

## Environment Model
- This repo is imported by sibling repo `narf-targeting`.
- Assume sibling repos may exist under same parent directory.

```text
$HOME/work/narf/
  narf-ros/
  narf-targeting/
  waterjet_pred_valencia/
  <other sibling repos>
```

## Working Assumptions
- API changes are allowed when they improve clarity, safety, maintainability, performance, or numerical correctness.
- Inputs are usually controlled. Do not add defensive validation, fallback behavior, or compatibility layers unless explicitly requested or required by established behavior.
- Keep public names and parameter semantics coherent with physical-model terminology.

## Operational Boundaries
- Keep package structure simple. Avoid framework-like abstractions.
- Preserve clear separation between simulation, tracing, plotting, and CLI layers.
- Generated artifacts such as HTML plots, CSV traces, `__pycache__`, and egg-info are not source of truth and usually should not be edited.
