"""Uses 'bokeh' to plot simulation results and export as interactive HTML."""

from pathlib import Path
from typing import Dict

import numpy as np
from bokeh.layouts import layout
from bokeh.models import ColumnDataSource, Range1d
from bokeh.palettes import Blues8, Colorblind5
from bokeh.plotting import figure, output_file, save
from numpy.typing import NDArray
from scipy.integrate._ivp.ivp import OdeResult

from .jet_state import JetState


def plot_solution(sol: OdeResult, state_idx: Dict[str, int], path: Path) -> None:
    """Plot trajectory and evolution of variables over the streamwise axis 's'.

    Args:
        sol: ODE solution object from solve_ivp.
        state_idx: Mapping of variable names to indices in sol.y .
        path: Path to export the plot to (html).
    """

    path_str: str = str(path)
    assert path_str.endswith("html"), "Plot path must be an HTML file."
    output_file(path_str, title="Fire stream simulation")

    x_margin: float = 1.0
    # Last 's' value to plot (termination event or pre-defined simulation span)
    s_end = sol.t_events[0][0] if len(sol.t_events[0]) > 0 else sol.t[-1]

    # Define data source for bokeh with all relevant variables.
    source = ColumnDataSource(
        data={
            "s": sol.t,
            "Uc": sol.y[JetState.get_idx("Uc"), :],
            "Dc": sol.y[JetState.get_idx("Dc"), :],
            "Ua": sol.y[JetState.get_idx("Ua"), :],
            "Da": sol.y[JetState.get_idx("Da"), :],
            "theta_a_deg": np.rad2deg(
                np.pi / 2 - sol.y[JetState.get_idx("theta_a"), :]
            ),
            "Uf": sol.y[JetState.get_idx("Uf"), :],
            "Df": sol.y[JetState.get_idx("Df"), :],
            "theta_f_deg": np.rad2deg(
                np.pi / 2 - sol.y[JetState.get_idx("theta_f"), :]
            ),
            "rho_f": sol.y[JetState.get_idx("rho_f"), :],
            "x": sol.y[JetState.get_idx("x"), :],
            "y": sol.y[JetState.get_idx("y"), :],
        }
    )
    for i in range(5):
        source.data[f"ND_{i}"] = sol.y[state_idx[f"ND_{i}"]]  # pyright: ignore

    # Prepare main plot (water jet trajectory).
    p_traj = figure(
        title="Fire stream trajectory",
        x_axis_label="x / m",
        y_axis_label="y / m",
        match_aspect=True,
    )
    p_traj.line(
        "x",
        "y",
        source=source,
        line_width=2,
        color="navy",
        legend_label="Trajectory",
    )

    # The stream has an evolving diameter. Imagine a disc around the trajectory line at
    # a given point. To plot it properly, the disc needs to be rotated by the streams
    # angle at each point.
    upper = np.array([source.data["x"], source.data["y"]]).T
    lower = np.copy(upper)
    for i in range(np.size(sol.t)):
        r = sol.y[state_idx["Df"], i] / 2.0
        theta = np.pi / 2 - sol.y[JetState.get_idx("theta_f"), i]
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        upper[i, :] += np.dot(rot, np.array([0, r]))
        lower[i, :] += np.dot(rot, np.array([0, -r]))

    # Draw diameter evolution as patch (coordinates: upper forward, lower backward)
    x_patch: NDArray = np.concatenate([upper[:, 0], lower[::-1, 0]])
    y_patch: NDArray = np.concatenate([upper[:, 1], lower[::-1, 1]])
    p_traj.patch(
        x_patch,
        y_patch,
        fill_alpha=0.3,
        fill_color="skyblue",
        line_color="gray",
        line_width=1,
        legend_label="Stream Width",
    )

    # --- DEBUG PLOTS --- #

    x_axis_label: str = "Streamwise position s / m"
    x_range = Range1d(0, s_end + x_margin)

    # Speeds of core, air, spray, stream phase (U_c, U_a, U_s, U_f).
    p_speeds = figure(
        title="Phase speeds",
        x_axis_label=x_axis_label,
        x_range=x_range,
        y_axis_label="Speed / m/s",
    )
    p_speeds.line(
        "s",
        "Uc",
        source=source,
        line_width=2,
        line_color=Colorblind5[0],
        legend_label="Uc",
    )
    p_speeds.line(
        "s",
        "Ua",
        source=source,
        line_width=2,
        line_color=Colorblind5[1],
        legend_label="Ua",
    )
    p_speeds.line(
        "s",
        "Uf",
        source=source,
        line_width=2,
        line_color=Colorblind5[2],
        legend_label="Uf",
    )

    # Diameters of each phase (D_c, D_a, D_s, D_f).
    p_diameters = figure(
        title="Phase diameters",
        x_axis_label=x_axis_label,
        x_range=x_range,
        y_axis_label="diameter / m",
    )
    p_diameters.line(
        "s",
        "Dc",
        source=source,
        line_width=2,
        line_color=Colorblind5[0],
        legend_label="Dc",
    )
    p_diameters.line(
        "s",
        "Da",
        source=source,
        line_width=2,
        line_color=Colorblind5[1],
        legend_label="Da",
    )
    p_diameters.line(
        "s",
        "Df",
        source=source,
        line_width=2,
        line_color=Colorblind5[2],
        legend_label="Df",
    )
    p_diameters.x_range = Range1d(0, s_end + x_margin)

    # Angles of each phase (theta_c, theta_a, theta_s, theta_f).
    p_angles = figure(
        title="Phase angles (above horizon)",
        x_axis_label=x_axis_label,
        x_range=x_range,
        y_axis_label="Angle / deg",
    )
    p_angles.line(
        "s",
        "theta_a_deg",
        source=source,
        line_width=2,
        line_color=Colorblind5[0],
        legend_label="theta_a",
    )
    p_angles.line(
        "s",
        "theta_f_deg",
        source=source,
        line_width=2,
        line_color=Colorblind5[1],
        legend_label="theta_f",
    )
    p_angles.x_range = Range1d(0, s_end + x_margin)

    # Drop formation rate for each droplet class (ND[i]).
    p_nd = figure(
        title="Drop count",
        x_axis_label=x_axis_label,
        x_range=x_range,
        y_axis_label="ND / drops/s",
        y_axis_type="log",
        y_range=Range1d(0.0, 1e7),
    )
    for i in range(5):
        p_nd.line(
            "s",
            f"ND_{i}",
            source=source,
            line_width=2,
            line_color=Blues8[i],
            legend_label=f"ND_{i}",
        )

    # Stream density (rho_f).
    # Should go from 100% water (rho_w) to more and more air-like (rho_a).
    p_rho = figure(
        title="Stream density",
        x_axis_label=x_axis_label,
        x_range=x_range,
        y_axis_label="Density / kg/m³",
        y_range=Range1d(0, 1000),
    )
    p_rho.line(
        "s",
        "rho_f",
        source=source,
        line_width=2,
        line_color="purple",
        legend_label="rho_f",
    )
    # TODO: Plot rho_w and rho_a as horizontal reference lines.

    # Place figures in a grid layout.
    plot_layout = layout(
        [
            [p_traj],
            [p_speeds, p_diameters],
            [p_angles, p_nd],
            [p_rho],
        ],  # pyright: ignore
        sizing_mode="stretch_width",
    )

    save(plot_layout)
    return
