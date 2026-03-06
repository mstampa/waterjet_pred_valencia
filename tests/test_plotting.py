#!/usr/bin/env python3

"""Tests for plotting utilities."""

import pandas as pd

from waterjet_pred_valencia.jet_state import JetState
from waterjet_pred_valencia.parameters import SimParams
from waterjet_pred_valencia.plotting import plot_trace
from waterjet_pred_valencia.simulator import ode_right_hand_side
from waterjet_pred_valencia.tracer import Tracer


def test_plot_trace_writes_html(tmp_path):
    params = SimParams(30.8, 24.0, 0.0254)
    state = JetState.get_initial(30.8, 24.0, 0.0254)
    tracer = Tracer(s_stride=1e-9)

    ode_right_hand_side(0.0, state.to_array(), params=params, tracer=tracer)
    trace_df = tracer.to_wide_dataframe()

    output_path = tmp_path / "trace_plot.html"
    plot_trace(trace_df, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_trace_rejects_empty_dataframe(tmp_path):
    output_path = tmp_path / "empty_trace_plot.html"

    try:
        plot_trace(pd.DataFrame({"s": []}), output_path)
    except ValueError as err:
        assert "empty" in str(err).lower()
    else:
        raise AssertionError("Expected ValueError for empty trace dataframe")
