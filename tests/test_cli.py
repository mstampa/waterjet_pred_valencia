#!/usr/bin/env python3

"""Tests for CLI simulation orchestration."""

from argparse import Namespace
from pathlib import Path

import pytest

from waterjet_pred_valencia import cli
from waterjet_pred_valencia.simulator import SimulationResult


def _make_args(tmp_path: Path) -> Namespace:
    """Build a minimal CLI argument namespace for tests.

    Args:
        tmp_path: Temporary directory provided by pytest.

    Returns:
        Namespace with all fields required by run_simulation().
    """

    return Namespace(
        speed=30.8,
        angle=24.0,
        nozzle=0.0254,
        span=100.0,
        max_step=1e-3,
        debug=False,
        output=tmp_path / "plot.html",
        trace=tmp_path / "trace.csv",
    )


def test_run_simulation_crash_still_writes_plot_and_trace(tmp_path, monkeypatch):
    """Ensure failed simulations still export plot and trace outputs."""

    args = _make_args(tmp_path)

    def _fake_simulate(*_, **kwargs):
        tracer = kwargs["tracer"]
        tracer.snapshot(
            s=0.0,
            scalars={"x": 0.0, "y": 0.0, "theta_a_deg": 24.0, "theta_f_deg": 24.0},
            vectors={},
        )
        raise AssertionError("forced failure for test")

    monkeypatch.setattr(cli, "simulate", _fake_simulate)

    with pytest.raises(AssertionError, match="forced failure"):
        cli.run_simulation(args)

    assert args.output.exists()
    assert args.trace.exists()


def test_run_simulation_uses_solution_plot_on_success(tmp_path, monkeypatch):
    """Ensure successful runs use full-solution plotting path only."""

    args = _make_args(tmp_path)
    called = {"plot_solution": 0, "plot_trace": 0}

    def _fake_simulate(*_, **__):
        return SimulationResult(status="completed", state_idx={}, sol=object())

    def _fake_plot_solution(*_, **__):
        called["plot_solution"] += 1

    def _fake_plot_trace(*_, **__):
        called["plot_trace"] += 1

    monkeypatch.setattr(cli, "simulate", _fake_simulate)
    monkeypatch.setattr(cli, "plot_solution", _fake_plot_solution)
    monkeypatch.setattr(cli, "plot_trace", _fake_plot_trace)

    cli.run_simulation(args)

    assert called["plot_solution"] == 1
    assert called["plot_trace"] == 0
    assert args.trace.exists()
