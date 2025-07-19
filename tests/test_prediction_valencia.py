#!/usr/bin/env python

"""
Unit tests for water jet trajectory predictions.
"""

from model.simulator import simulate

import numpy as np
import pytest


# speeds and angles from Tab. 1
# expected x and y from Fig. 4 or text
@pytest.mark.parametrize(
    "speed, angle_deg,exp_x_max,exp_y_max",
    [
        (22.2, 48.0, 28.0, 11.0),
        (26.4, 48.0, 33.0, 13.0),
        (26.4, 37.0, 33.0, 9.0),
        (26.4, 24, 28.0, 5.5),
        (30.8, 24.0, 32.0, 7.0),
        (30.8, 37.0, 40.0, 12.0),
    ],
)
def test_valencia_article_results(speed, angle_deg, exp_x_max, exp_y_max):

    sol, idx = simulate(speed, np.deg2rad(angle_deg), 0.00254, s_span=(0.0, 200.0))
    atol_x = 3.0  # absolute tolerance for x [m]

    x_last = sol.y[idx["x"], -1]
    y_max = np.max(sol.y[idx["y"], :])
    # Df = sol.y[idx["Df"], :]

    assert (
        abs(x_last - exp_x_max) < atol_x
    ), f"x_last = {x_last:.2f} m, expected ~{exp_x_max} m"
    assert y_max <= exp_y_max, f"y_max = {y_max:.2f} m, expected ~{exp_y_max} m"
