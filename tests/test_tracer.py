#!/usr/bin/env python3

"""Tests for tracer capture behavior."""

from waterjet_pred_valencia.jet_state import JetState
from waterjet_pred_valencia.parameters import SimParams, num_drop_classes
from waterjet_pred_valencia.simulator import ode_right_hand_side
from waterjet_pred_valencia.tracer import Tracer


def test_tracer_includes_minimum_plot_fields():
    params = SimParams(30.8, 24.0, 0.0254)
    state = JetState.get_initial(30.8, 24.0, 0.0254)
    tracer = Tracer(s_stride=1e-9)

    ode_right_hand_side(0.0, state.to_array(), params=params, tracer=tracer)
    df = tracer.to_wide_dataframe()

    required_columns = {
        "s",
        "x",
        "y",
        "Uc",
        "Ua",
        "Uf",
        "Dc",
        "Da",
        "Df",
        "rho_f",
        "theta_a_deg",
        "theta_f_deg",
    }
    required_columns.update({f"ND[{i}]" for i in range(num_drop_classes)})

    assert len(df) == 1
    assert required_columns.issubset(df.columns)
