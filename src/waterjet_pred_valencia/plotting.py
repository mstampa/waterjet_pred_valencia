"""Uses bokeh to plot simulation results and export interactive HTML."""

import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import AdaptiveTicker, ColumnDataSource, Range1d
from bokeh.palettes import Blues8, Colorblind5
from bokeh.plotting import figure, output_file, save
from pandas import DataFrame
from scipy.integrate._ivp.ivp import OdeResult

from .jet_state import JetState
from .parameters import num_drop_classes

logger = logging.getLogger(__name__)

# Fixed colors for consistent phase coloring across all plots.
PHASE_COLORS: Dict[str, str] = {
    "core": Colorblind5[2],
    "air": Colorblind5[1],
    "stream": Colorblind5[3],
}
SPRAY_COLORS: Tuple[str, ...] = tuple(Blues8[i] for i in range(num_drop_classes))

DEFAULT_TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
DESIRED_MAJOR_TICKS = 10


def plot_solution(sol: OdeResult, state_idx: Dict[str, int], path: Path) -> None:
    """Plot trajectory and variable evolution based on a successful ODE result.

    Args:
        sol: ODE solution object from solve_ivp.
        state_idx: Mapping of variable names to indices in sol.y.
        path: Path to export the plot to (html).
    """
    logger.info("Plotting OdeResult...")
    source, s_end = _build_source_from_solution(sol, state_idx)
    _save_plot(source=source, s_end=s_end, path=path)
    return


def plot_trace(trace_df: DataFrame, path: Path) -> None:
    """Plot trajectory and variable evolution based on traced partial data.

    Args:
        trace_df: Wide trace dataframe produced by Tracer.to_wide_dataframe().
        path: Path to export the plot to (html).
    """
    logger.info("Plotting traced partial results...")
    source, s_end = _build_source_from_trace(trace_df)
    _save_plot(source=source, s_end=s_end, path=path)
    return


def _build_source_from_solution(
    sol: OdeResult, state_idx: Dict[str, int]
) -> Tuple[ColumnDataSource, float]:
    """Build a bokeh data source from a complete ODE solution.

    Args:
        sol: ODE solution returned by scipy.solve_ivp.
        state_idx: Mapping of variable names to flat state vector indices.

    Returns:
        Tuple of bokeh data source and maximum plotted s-value.
    """

    # Last s value to plot (termination event or predefined simulation span).
    s_end: float = float(sol.t_events[0][0] if len(sol.t_events[0]) > 0 else sol.t[-1])

    data: Dict[str, np.ndarray] = {
        "s": sol.t,
        "Uc": sol.y[JetState.get_idx("Uc"), :],
        "Dc": sol.y[JetState.get_idx("Dc"), :],
        "Ua": sol.y[JetState.get_idx("Ua"), :],
        "Da": sol.y[JetState.get_idx("Da"), :],
        "theta_a_deg": np.rad2deg(np.pi / 2 - sol.y[JetState.get_idx("theta_a"), :]),
        "Uf": sol.y[JetState.get_idx("Uf"), :],
        "Df": sol.y[JetState.get_idx("Df"), :],
        "theta_f_deg": np.rad2deg(np.pi / 2 - sol.y[JetState.get_idx("theta_f"), :]),
        "rho_f": sol.y[JetState.get_idx("rho_f"), :],
        "x": sol.y[JetState.get_idx("x"), :],
        "y": sol.y[JetState.get_idx("y"), :],
        **{f"ND_{i}": sol.y[state_idx[f"ND_{i}"]] for i in range(num_drop_classes)},
        **{f"Us_{i}": sol.y[state_idx[f"Us_{i}"]] for i in range(num_drop_classes)},
        **{
            f"theta_s_deg_{i}": np.rad2deg(np.pi / 2 - sol.y[state_idx[f"theta_s_{i}"]])
            for i in range(num_drop_classes)
        },
    }

    source = ColumnDataSource(data=data)
    return source, s_end


def _build_source_from_trace(trace_df: DataFrame) -> Tuple[ColumnDataSource, float]:
    """Build a bokeh data source from traced partial simulation data.

    Args:
        trace_df: Wide dataframe created by Tracer.

    Returns:
        Tuple of bokeh data source and maximum plotted s-value.

    Raises:
        ValueError: If the dataframe is empty or has no finite s-values.
    """

    if trace_df.empty:
        raise ValueError("Trace dataframe is empty; can not generate plot.")

    n_rows = len(trace_df)

    def _trace_col(name: str) -> np.ndarray:
        if name in trace_df.columns:
            return trace_df[name].to_numpy(dtype=float)
        return np.full((n_rows,), np.nan, dtype=float)

    s = _trace_col("s")
    data: Dict[str, np.ndarray] = {
        "s": s,
        "Uc": _trace_col("Uc"),
        "Dc": _trace_col("Dc"),
        "Ua": _trace_col("Ua"),
        "Da": _trace_col("Da"),
        "theta_a_deg": 90.0 - _trace_col("theta_a_deg"),
        "Uf": _trace_col("Uf"),
        "Df": _trace_col("Df"),
        "theta_f_deg": 90.0 - _trace_col("theta_f_deg"),
        "rho_f": _trace_col("rho_f"),
        "x": _trace_col("x"),
        "y": _trace_col("y"),
        **{f"ND_{i}": _trace_col(f"ND[{i}]") for i in range(num_drop_classes)},
        **{f"Us_{i}": _trace_col(f"Us[{i}]") for i in range(num_drop_classes)},
        **{
            f"theta_s_deg_{i}": 90.0 - _trace_col(f"theta_s_deg[{i}]")
            for i in range(num_drop_classes)
        },
    }

    source = ColumnDataSource(data=data)

    finite_s = s[np.isfinite(s)]
    if finite_s.size == 0:
        raise ValueError(
            "Trace dataframe has no finite 's' values; can not generate plot."
        )

    s_end = float(np.max(finite_s))
    return source, s_end


def _save_plot(source: ColumnDataSource, s_end: float, path: Path) -> None:
    """Render and save the standard multi-panel simulation plot.

    Args:
        source: Prepared bokeh data source with plotting columns.
        s_end: Maximum s-value used for horizontal range limits.
        path: Output path for the generated html file.
    """

    logger.info(f"Saving plot to {path}...")
    logger.info(f"Maximum s-value: {s_end} m")

    path_str = str(path)
    assert path_str.endswith("html"), (
        f"Path suffix must be .html, but is {path.suffix}."
    )
    output_file(path_str, title="Fire stream simulation")

    x_vals = np.asarray(source.data["x"], dtype=float)
    y_vals = np.asarray(source.data["y"], dtype=float)
    finite_x = x_vals[np.isfinite(x_vals)]
    finite_y = y_vals[np.isfinite(y_vals)]
    x_max = float(np.max(finite_x)) if finite_x.size > 0 else 0.0
    y_max = float(np.max(finite_y)) if finite_y.size > 0 else 0.0

    p_traj = figure(
        title="Fire stream trajectory",
        x_axis_label="x / m",
        y_axis_label="y / m",
        x_range=Range1d(0.0, x_max + 1.0),
        y_range=Range1d(0.0, y_max + 1.0),
        match_aspect=True,
        sizing_mode="stretch_width",
        tools=DEFAULT_TOOLS,
    )
    p_traj.line(
        "x",
        "y",
        source=source,
        line_width=2,
        color=PHASE_COLORS["stream"],
        legend_label="Trajectory",
    )
    p_traj.legend.location = "top_left"
    _add_stream_width_patch(p_traj, source)
    _configure_linear_grid_density([p_traj])

    x_axis_label = "Streamwise position s / m"
    x_range = Range1d(0, s_end + 0.1)

    speed_max = 0.0
    speed_keys = ["Uc", "Ua", "Uf"] + [f"Us_{i}" for i in range(num_drop_classes)]
    for key in speed_keys:
        vals = np.asarray(source.data[key], dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size > 0:
            speed_max = max(speed_max, float(np.max(finite)))

    p_speeds = figure(
        title="Phase speeds",
        x_axis_label=x_axis_label,
        x_range=x_range,
        y_axis_label="Speed / m/s",
        y_range=Range1d(0.0, speed_max + 0.5),
        sizing_mode="stretch_width",
        tools=DEFAULT_TOOLS,
    )
    p_speeds.line(
        "s",
        "Uc",
        source=source,
        line_width=2,
        line_color=PHASE_COLORS["core"],
        legend_label="Uc",
    )
    p_speeds.line(
        "s",
        "Ua",
        source=source,
        line_width=2,
        line_color=PHASE_COLORS["air"],
        legend_label="Ua",
    )
    p_speeds.line(
        "s",
        "Uf",
        source=source,
        line_width=2,
        line_color=PHASE_COLORS["stream"],
        legend_label="Uf",
    )
    for i in range(num_drop_classes):
        p_speeds.line(
            "s",
            f"Us_{i}",
            source=source,
            line_width=2,
            line_color=SPRAY_COLORS[i],
            legend_label=f"Us{i}",
        )
    p_speeds.legend.location = "bottom_left"
    _configure_linear_grid_density([p_speeds])

    diameter_max = 0.0
    for key in ("Dc", "Da", "Df"):
        vals = np.asarray(source.data[key], dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size > 0:
            diameter_max = max(diameter_max, float(np.max(finite)))
    p_diameters = figure(
        title="Phase diameters",
        x_axis_label=x_axis_label,
        x_range=x_range,
        y_axis_label="diameter / m",
        y_range=Range1d(0.0, diameter_max),
        sizing_mode="stretch_width",
        tools=DEFAULT_TOOLS,
    )
    p_diameters.line(
        "s",
        "Dc",
        source=source,
        line_width=2,
        line_color=PHASE_COLORS["core"],
        legend_label="Dc",
    )
    p_diameters.line(
        "s",
        "Da",
        source=source,
        line_width=2,
        line_color=PHASE_COLORS["air"],
        legend_label="Da",
    )
    p_diameters.line(
        "s",
        "Df",
        source=source,
        line_width=2,
        line_color=PHASE_COLORS["stream"],
        legend_label="Df",
    )
    p_diameters.legend.location = "top_left"
    _configure_linear_grid_density([p_diameters])

    p_angles = figure(
        title="Phase angles (above horizon)",
        x_axis_label=x_axis_label,
        x_range=x_range,
        y_axis_label="Angle / deg",
        y_range=Range1d(-90.0, 90.0),
        sizing_mode="stretch_width",
        tools=DEFAULT_TOOLS,
    )
    p_angles.line(
        "s",
        "theta_a_deg",
        source=source,
        line_width=2,
        line_color=PHASE_COLORS["air"],
        legend_label="theta_a",
    )
    p_angles.line(
        "s",
        "theta_f_deg",
        source=source,
        line_width=2,
        line_color=PHASE_COLORS["stream"],
        legend_label="theta_f",
    )
    for i in range(num_drop_classes):
        p_angles.line(
            "s",
            f"theta_s_deg_{i}",
            source=source,
            line_width=2,
            line_color=SPRAY_COLORS[i],
            legend_label=f"theta_s{i}",
        )
    p_angles.legend.location = "bottom_left"
    _configure_linear_grid_density([p_angles])

    nd_min_positive = np.inf
    nd_max = 0.0
    for i in range(num_drop_classes):
        nd_vals = np.asarray(source.data[f"ND_{i}"], dtype=float)
        finite_positive = nd_vals[np.isfinite(nd_vals) & (nd_vals > 0.0)]
        if finite_positive.size > 0:
            nd_min_positive = min(nd_min_positive, float(np.min(finite_positive)))
            nd_max = max(nd_max, float(np.max(finite_positive)))

    if not np.isfinite(nd_min_positive):
        nd_min_positive = 1e-3
    if nd_max <= 0.0:
        nd_max = 1.0
    nd_y_start = max(1e-6, nd_min_positive * 0.5)
    nd_y_end = max(nd_y_start * 10.0, nd_max * 1.2)

    p_nd = figure(
        title="Drop count",
        x_axis_label=x_axis_label,
        x_range=x_range,
        y_axis_label="ND / drops/s",
        y_axis_type="log",
        y_range=Range1d(nd_y_start, nd_y_end),
        sizing_mode="stretch_width",
        tools=DEFAULT_TOOLS,
    )

    for i in range(num_drop_classes):
        p_nd.line(
            "s",
            f"ND_{i}",
            source=source,
            line_width=2,
            line_color=SPRAY_COLORS[i],
            legend_label=f"ND{i}",
        )
    p_nd.legend.location = "bottom_right"

    p_rho = figure(
        title="Stream density",
        x_axis_label=x_axis_label,
        x_range=x_range,
        y_axis_label="Density / kg/m³",
        y_range=Range1d(0.0, 1000.0),
        sizing_mode="stretch_width",
        height=300,
        tools=DEFAULT_TOOLS,
    )
    p_rho.line(
        "s",
        "rho_f",
        source=source,
        line_width=2,
        line_color=PHASE_COLORS["stream"],
        legend_label="rho_f",
    )
    _configure_linear_grid_density([p_rho])

    plot_layout = column(
        p_traj,
        row(p_speeds, p_diameters, sizing_mode="stretch_width"),
        row(p_angles, p_nd, sizing_mode="stretch_width"),
        p_rho,
        sizing_mode="stretch_width",
    )

    save(plot_layout)
    logger.info("Plot saved.")
    return


def _add_stream_width_patch(p_traj, source: ColumnDataSource) -> None:
    """Overlay stream-width patch around the trajectory line.

    Args:
        p_traj: Bokeh figure used for trajectory plotting.
        source: Data source containing x, y, Df and theta_f_deg columns.
    """

    x_vals = np.asarray(source.data["x"], dtype=float)
    y_vals = np.asarray(source.data["y"], dtype=float)
    d_vals = np.asarray(source.data["Df"], dtype=float)
    theta_deg = np.asarray(source.data["theta_f_deg"], dtype=float)

    if x_vals.size == 0:
        return

    upper = np.array([x_vals, y_vals], dtype=float).T
    lower = np.copy(upper)

    valid = (
        np.isfinite(x_vals)
        & np.isfinite(y_vals)
        & np.isfinite(d_vals)
        & np.isfinite(theta_deg)
    )
    for i in np.where(valid)[0]:
        r = d_vals[i] / 2.0
        theta = np.deg2rad(theta_deg[i])
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=float)
        upper[i, :] += np.dot(rot, np.array([0.0, r]))
        lower[i, :] += np.dot(rot, np.array([0.0, -r]))

    x_patch = np.concatenate([upper[:, 0], lower[::-1, 0]])
    y_patch = np.concatenate([upper[:, 1], lower[::-1, 1]])
    p_traj.patch(
        x_patch,
        y_patch,
        fill_alpha=0.3,
        fill_color="skyblue",
        line_color="gray",
        line_width=1,
        legend_label="Stream Width",
    )
    logger.debug(f"Added stream width patch with {x_patch.shape[0]} elements")
    return


def _configure_linear_grid_density(figures: Iterable) -> None:
    """Increase major gridline density on linear axes.

    Args:
        figures: Iterable of bokeh figures to configure.
    """

    for fig in figures:
        for axis in fig.xaxis:
            axis.ticker = AdaptiveTicker(desired_num_ticks=DESIRED_MAJOR_TICKS)
        for axis in fig.yaxis:
            if isinstance(axis.ticker, AdaptiveTicker):
                axis.ticker = AdaptiveTicker(desired_num_ticks=DESIRED_MAJOR_TICKS)
    return
