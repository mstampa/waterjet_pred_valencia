#!/usr/bin/env python3

"""
Plot simulation results using bokeh and saves them into an HTML file.
"""

from bokeh.layouts import layout
from bokeh.models import ColumnDataSource, Range1d
from bokeh.palettes import Blues8, Colorblind5
from bokeh.plotting import figure, output_file, save
import numpy as np
from pathlib import Path
from scipy.integrate._ivp.ivp import OdeResult


def plot_solution(sol: OdeResult, state_idx: dict[str, int], path: Path) -> None:
    """
    Plots trajectory and evolution of variables over the streamwise axis s.

    Args:
        sol: ODE solution object from solve_ivp
        state_idx: Mapping of variable names to indices in sol.y
        path: Where to save HTML containing the plots to
    """

    path_str = str(path)
    output_file(path_str, title="Fire stream simulation")
    x_margin = 1.0
    s_end = sol.t[-1]  # last domain coordinate to plot
    if len(sol.t_events[0]) > 0:
        s_end = sol.t_events[0][0]  # s at ground impact

    # define data source for bokeh with all relevant variables
    # TODO: Clarify angle usage. This plotting function expects 'above horizon'.
    source = ColumnDataSource(
        data={
            "s": sol.t,
            "Uc": sol.y[state_idx["Uc"], :],
            "Dc": sol.y[state_idx["Dc"], :],
            "Ua": sol.y[state_idx["Ua"], :],
            "Da": sol.y[state_idx["Da"], :],
            "theta_a_deg": np.rad2deg(np.pi / 2 - sol.y[state_idx["theta_a"], :]),
            "Uf": sol.y[state_idx["Uf"], :],
            "Df": sol.y[state_idx["Df"], :],
            "theta_f_deg": np.rad2deg(np.pi / 2 - sol.y[state_idx["theta_f"], :]),
            "rho_f": sol.y[state_idx["rho_f"], :],
            "x": sol.y[state_idx["x"], :],
            "y": sol.y[state_idx["y"], :],
        }
    )
    for i in range(5):
        source.data[f"ND_{i}"] = sol.y[state_idx[f"ND_{i}"]]  # pyright: ignore

    # prepare main plot (water jet trajectory)
    p_traj = figure(
        title="Fire stream trajectory",
        x_axis_label="x [m]",
        y_axis_label="y [m]",
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
        theta = np.pi / 2 - sol.y[state_idx["theta_f"], i]
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        upper[i, :] += np.dot(rot, np.array([0, r]))
        lower[i, :] += np.dot(rot, np.array([0, -r]))

    # build patch coordinates (upper forward, lower backward)
    x_patch = np.concatenate([upper[:, 0], lower[::-1, 0]])
    y_patch = np.concatenate([upper[:, 1], lower[::-1, 1]])

    # draw diameter evolution as patch
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

    x_axis_label = "Streamwise position s / m"
    x_range = Range1d(0, s_end + x_margin)

    # speeds of core, air, spray, stream phase
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

    # diameters of each phase
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

    # angles of each phase
    p_angles = figure(
        title="Phase angles (above horizon)",
        x_axis_label=x_axis_label,
        x_range=x_range,
        y_axis_label="Angle / °",
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

    # Drop formation rate for each droplet class
    p_nd = figure(
        title="Drop count",
        x_axis_label=x_axis_label,
        x_range=x_range,
        y_axis_label="ND / drops/s",
        y_axis_type="log",
        y_range=Range1d(0.0, 1e8),
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

    # Stream density (from 100% water to more and more air-like)
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

    # grid layout for all plots
    mylayout = layout(
        [
            [p_traj],
            [p_speeds, p_diameters],
            [p_angles, p_nd],
            [p_rho],
        ],  # pyright: ignore
        sizing_mode="stretch_width",
    )

    save(mylayout)
    return
