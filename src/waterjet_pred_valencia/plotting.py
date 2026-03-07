"""Plot simulation results with bokeh and export interactive HTML."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Sequence, Tuple

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import AdaptiveTicker, ColumnDataSource, HoverTool, Label, Range1d
from bokeh.models.renderers import GlyphRenderer
from bokeh.palettes import Blues8, Colorblind5
from bokeh.plotting import figure, output_file, save
from pandas import DataFrame
from scipy.integrate._ivp.ivp import OdeResult

from .jet_state import JetState
from .parameters import num_drop_classes

logger = logging.getLogger(__name__)

PHASE_COLORS: Dict[str, str] = {
    "core": Colorblind5[2],
    "air": Colorblind5[1],
    "stream": Colorblind5[3],
}
SPRAY_COLORS: Tuple[str, ...] = tuple(Blues8[i] for i in range(num_drop_classes))
SURROUNDINGS_COLOR = "#4a4a4a"

DEFAULT_TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
DESIRED_MAJOR_TICKS = 10


@dataclass(frozen=True)
class SeriesSpec:
    """Description for rendering one standard line series."""

    field: str
    label: str
    color: str
    width: int = 2
    alpha: float = 1.0
    dash: str = "solid"


@dataclass(frozen=True)
class TransferSpec:
    """Description for rendering one transfer-term series."""

    field: str
    source_phase: str
    target_phase: str
    total: bool = False
    width: int = 2
    alpha: float = 1.0


def plot_solution(sol: OdeResult, state_idx: Dict[str, int], path: Path) -> None:
    """Plot trajectory and variable evolution from a successful ODE result.

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
    """Plot trajectory and variable evolution from traced partial data.

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

    s_end = float(sol.t_events[0][0] if len(sol.t_events[0]) > 0 else sol.t[-1])
    n_rows = sol.t.size
    nan_series = np.full((n_rows,), np.nan, dtype=float)

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
        "m_sur2f": nan_series.copy(),
        "m_a2sur": nan_series.copy(),
        "f_a2sur": nan_series.copy(),
        "f_ra2sur": nan_series.copy(),
        "f_c2a": nan_series.copy(),
        "f_rc2a": nan_series.copy(),
        "f_s2a_total": nan_series.copy(),
        "f_rs2a_total": nan_series.copy(),
        "f_s2sur_total": nan_series.copy(),
        "f_rs2sur_total": nan_series.copy(),
        "m_c2s_total": nan_series.copy(),
        "m_s2sur_total": nan_series.copy(),
        "f_c2s_total": nan_series.copy(),
        "f_rc2s_total": nan_series.copy(),
    }

    for i in range(num_drop_classes):
        data[f"m_c2s_{i}"] = nan_series.copy()
        data[f"m_s2sur_{i}"] = nan_series.copy()
        data[f"f_c2s_{i}"] = nan_series.copy()
        data[f"f_rc2s_{i}"] = nan_series.copy()
        data[f"f_s2a_{i}"] = nan_series.copy()
        data[f"f_rs2a_{i}"] = nan_series.copy()
        data[f"f_s2sur_{i}"] = nan_series.copy()
        data[f"f_rs2sur_{i}"] = nan_series.copy()

    return ColumnDataSource(data=data), s_end


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
        "m_sur2f": _trace_col("m_sur2f"),
        "m_a2sur": _trace_col("m_a2sur"),
        "f_a2sur": _trace_col("f_a2sur"),
        "f_ra2sur": _trace_col("f_ra2sur"),
        "f_c2a": _trace_col("f_c2a"),
        "f_rc2a": _trace_col("f_rc2a"),
        "f_s2a_total": _trace_col("f_s2a_total"),
        "f_rs2a_total": _trace_col("f_rs2a_total"),
        "f_s2sur_total": _trace_col("f_s2sur_total"),
        "f_rs2sur_total": _trace_col("f_rs2sur_total"),
        "m_c2s_total": _sum_trace_columns(_trace_col, "m_c2s", n_rows),
        "m_s2sur_total": _sum_trace_columns(_trace_col, "m_s2sur", n_rows),
        "f_c2s_total": _sum_trace_columns(_trace_col, "f_c2s", n_rows),
        "f_rc2s_total": _sum_trace_columns(_trace_col, "f_rc2s", n_rows),
    }

    for i in range(num_drop_classes):
        data[f"m_c2s_{i}"] = _trace_col(f"m_c2s[{i}]")
        data[f"m_s2sur_{i}"] = _trace_col(f"m_s2sur[{i}]")
        data[f"f_c2s_{i}"] = _trace_col(f"f_c2s[{i}]")
        data[f"f_rc2s_{i}"] = _trace_col(f"f_rc2s[{i}]")
        data[f"f_s2a_{i}"] = _trace_col(f"f_s2a[{i}]")
        data[f"f_rs2a_{i}"] = _trace_col(f"f_rs2a[{i}]")
        data[f"f_s2sur_{i}"] = _trace_col(f"f_s2sur[{i}]")
        data[f"f_rs2sur_{i}"] = _trace_col(f"f_rs2sur[{i}]")

    finite_s = s[np.isfinite(s)]
    if finite_s.size == 0:
        raise ValueError(
            "Trace dataframe has no finite 's' values; can not generate plot."
        )

    return ColumnDataSource(data=data), float(np.max(finite_s))


def _save_plot(source: ColumnDataSource, s_end: float, path: Path) -> None:
    """Render and save the complete multi-panel simulation plot.

    Args:
        source: Prepared bokeh datasource with plotting columns.
        s_end: Maximum s-value used for horizontal range limits.
        path: Output path for the generated html file.
    """

    logger.info(f"Saving plot to {path}...")
    path_str = str(path)
    assert path_str.endswith("html"), (
        f"Path suffix must be .html, but is {path.suffix}."
    )
    output_file(path_str, title="Fire stream simulation")

    x_range = Range1d(0.0, s_end + 0.1)
    p_traj = _build_trajectory_panel(source)
    p_speeds = _build_speed_panel(source, x_range)
    p_diameters = _build_diameter_panel(source, x_range)
    p_angles = _build_angle_panel(source, x_range)
    p_nd = _build_nd_panel(source, x_range)
    p_rho = _build_rho_panel(source, x_range)
    p_mom_stream = _build_transfer_stream_panel(source, x_range)
    p_mom_radial = _build_transfer_radial_panel(source, x_range)
    p_mass = _build_transfer_mass_panel(source, x_range)

    plot_layout = column(
        p_traj,
        row(p_speeds, p_diameters, sizing_mode="stretch_width"),
        row(p_angles, p_nd, sizing_mode="stretch_width"),
        p_rho,
        row(p_mom_stream, p_mom_radial, sizing_mode="stretch_width"),
        p_mass,
        sizing_mode="stretch_width",
    )
    save(plot_layout)
    logger.info("Plot saved.")
    return


def _build_trajectory_panel(source: ColumnDataSource):
    """Create trajectory panel with stream-width patch and hover."""

    x_max = _finite_max(source, "x")
    y_max = _finite_max(source, "y")
    p = _new_panel(
        title="Fire stream trajectory",
        x_axis_label="x / m",
        y_axis_label="y / m",
        x_range=Range1d(0.0, x_max + 0.5),
        y_range=Range1d(0.0, y_max + 0.5),
        match_aspect=True,
    )
    renderer = p.line(
        "x",
        "y",
        source=source,
        line_width=2,
        color=PHASE_COLORS["stream"],
        legend_label="Trajectory",
    )
    p.legend.location = "top_left"
    _add_stream_width_patch(p, source)
    _configure_linear_grid_density([p])
    _add_hover_tool(
        p,
        [
            ("x", "@x{0.000}"),
            ("y", "@y{0.000}"),
            ("s", "@s{0.000}"),
            ("Df", "@Df{0.0000}"),
        ],
        renderer=renderer,
    )
    return p


def _build_speed_panel(source: ColumnDataSource, x_range: Range1d):
    """Create phase-speed panel using series specs."""

    fields = ["Uc", "Ua", "Uf"] + [f"Us_{i}" for i in range(num_drop_classes)]
    speed_max = _finite_max(source, *fields)
    p = _new_panel(
        title="Phase speeds",
        x_axis_label="Streamwise position s / m",
        y_axis_label="Speed / m/s",
        x_range=x_range,
        y_range=Range1d(0.0, speed_max + 0.5),
    )
    specs = [
        SeriesSpec("Uc", "Uc", PHASE_COLORS["core"]),
        SeriesSpec("Ua", "Ua", PHASE_COLORS["air"]),
        SeriesSpec("Uf", "Uf", PHASE_COLORS["stream"]),
        *[
            SeriesSpec(f"Us_{i}", f"Us{i}", SPRAY_COLORS[i])
            for i in range(num_drop_classes)
        ],
    ]
    primary = _add_series_from_specs(p, source, specs)
    p.legend.location = "bottom_left"
    _configure_linear_grid_density([p])
    _add_hover_tool(
        p,
        [
            ("s", "@s{0.000}"),
            ("Uc", "@Uc{0.000}"),
            ("Ua", "@Ua{0.000}"),
            ("Uf", "@Uf{0.000}"),
        ]
        + [(f"Us{i}", f"@Us_{i}{{0.000}}") for i in range(num_drop_classes)],
        renderer=primary,
    )
    return p


def _build_diameter_panel(source: ColumnDataSource, x_range: Range1d):
    """Create phase-diameter panel using series specs."""

    dmax = _finite_max(source, "Dc", "Da", "Df")
    p = _new_panel(
        title="Phase diameters",
        x_axis_label="Streamwise position s / m",
        y_axis_label="diameter / m",
        x_range=x_range,
        y_range=Range1d(0.0, dmax),
    )
    specs = [
        SeriesSpec("Dc", "Dc", PHASE_COLORS["core"]),
        SeriesSpec("Da", "Da", PHASE_COLORS["air"]),
        SeriesSpec("Df", "Df", PHASE_COLORS["stream"]),
    ]
    primary = _add_series_from_specs(p, source, specs)
    p.legend.location = "top_left"
    _configure_linear_grid_density([p])
    _add_hover_tool(
        p,
        [
            ("s", "@s{0.000}"),
            ("Dc", "@Dc{0.0000}"),
            ("Da", "@Da{0.0000}"),
            ("Df", "@Df{0.0000}"),
        ],
        renderer=primary,
    )
    return p


def _build_angle_panel(source: ColumnDataSource, x_range: Range1d):
    """Create phase-angle panel using series specs."""

    p = _new_panel(
        title="Phase angles (above horizon)",
        x_axis_label="Streamwise position s / m",
        y_axis_label="Angle / deg",
        x_range=x_range,
        y_range=Range1d(-90.0, 90.0),
    )
    specs = [
        SeriesSpec("theta_a_deg", "theta_a", PHASE_COLORS["air"]),
        SeriesSpec("theta_f_deg", "theta_f", PHASE_COLORS["stream"]),
        *[
            SeriesSpec(f"theta_s_deg_{i}", f"theta_s{i}", SPRAY_COLORS[i])
            for i in range(num_drop_classes)
        ],
    ]
    primary = _add_series_from_specs(p, source, specs)
    p.legend.location = "bottom_left"
    _configure_linear_grid_density([p])
    _add_hover_tool(
        p,
        [
            ("s", "@s{0.000}"),
            ("theta_a", "@theta_a_deg{0.00}"),
            ("theta_f", "@theta_f_deg{0.00}"),
        ]
        + [
            (f"theta_s{i}", f"@theta_s_deg_{i}{{0.00}}")
            for i in range(num_drop_classes)
        ],
        renderer=primary,
    )
    return p


def _build_nd_panel(source: ColumnDataSource, x_range: Range1d):
    """Create drop-count panel with log y-axis."""

    nd_min, nd_max = _log_range_from_fields(
        source, [f"ND_{i}" for i in range(num_drop_classes)]
    )
    p = _new_panel(
        title="Drop count",
        x_axis_label="Streamwise position s / m",
        y_axis_label="ND / drops/s",
        x_range=x_range,
        y_range=Range1d(nd_min, nd_max),
        y_axis_type="log",
    )
    specs = [
        SeriesSpec(f"ND_{i}", f"ND{i}", SPRAY_COLORS[i])
        for i in range(num_drop_classes)
    ]
    primary = _add_series_from_specs(p, source, specs)
    p.legend.location = "bottom_right"
    _add_hover_tool(
        p,
        [("s", "@s{0.000}")]
        + [(f"ND{i}", f"@ND_{i}{{0.00}}") for i in range(num_drop_classes)],
        renderer=primary,
    )
    return p


def _build_rho_panel(source: ColumnDataSource, x_range: Range1d):
    """Create stream density panel."""

    p = _new_panel(
        title="Stream density",
        x_axis_label="Streamwise position s / m",
        y_axis_label="Density / kg/m³",
        x_range=x_range,
        y_range=Range1d(0.0, 1000.0),
        height=300,
    )
    primary = _add_series_from_specs(
        p,
        source,
        [SeriesSpec("rho_f", "rho_f", PHASE_COLORS["stream"])],
    )
    _configure_linear_grid_density([p])
    _add_hover_tool(
        p, [("s", "@s{0.000}"), ("rho_f", "@rho_f{0.00}")], renderer=primary
    )
    return p


def _build_transfer_mass_panel(source: ColumnDataSource, x_range: Range1d):
    """Create mass-transfer panel using transfer specs."""

    specs = [
        TransferSpec("m_sur2f", "sur", "stream"),
        TransferSpec("m_a2sur", "air", "sur"),
        TransferSpec("m_c2s_total", "core", "spray_mid", total=True),
        TransferSpec("m_s2sur_total", "spray_mid", "sur", total=True),
        *[
            TransferSpec(f"m_c2s_{i}", "core", f"spray_{i}", width=1, alpha=0.8)
            for i in range(num_drop_classes)
        ],
        *[
            TransferSpec(f"m_s2sur_{i}", f"spray_{i}", "sur", width=1, alpha=0.8)
            for i in range(num_drop_classes)
        ],
    ]
    return _build_transfer_panel(
        title="Mass transfer terms",
        y_axis_label="Mass transfer / kg/(m*s)",
        source=source,
        x_range=x_range,
        specs=specs,
        hover_rows=[
            ("s", "@s{0.000}"),
            ("m_sur2f", "@m_sur2f{0.0000}"),
            ("m_a2sur", "@m_a2sur{0.0000}"),
            ("m_c2s_total", "@m_c2s_total{0.0000}"),
            ("m_s2sur_total", "@m_s2sur_total{0.0000}"),
        ],
        primary_field="m_sur2f",
    )


def _build_transfer_stream_panel(source: ColumnDataSource, x_range: Range1d):
    """Create streamwise momentum-transfer panel using transfer specs."""

    specs = [
        TransferSpec("f_a2sur", "air", "sur"),
        TransferSpec("f_c2a", "core", "air"),
        TransferSpec("f_s2a_total", "spray_mid", "air", total=True),
        TransferSpec("f_s2sur_total", "spray_mid", "sur", total=True),
        TransferSpec("f_c2s_total", "core", "spray_mid", total=True),
        *[
            TransferSpec(f"f_c2s_{i}", "core", f"spray_{i}", width=1, alpha=0.8)
            for i in range(num_drop_classes)
        ],
        *[
            TransferSpec(f"f_s2a_{i}", f"spray_{i}", "air", width=1, alpha=0.8)
            for i in range(num_drop_classes)
        ],
        *[
            TransferSpec(f"f_s2sur_{i}", f"spray_{i}", "sur", width=1, alpha=0.8)
            for i in range(num_drop_classes)
        ],
    ]
    return _build_transfer_panel(
        title="Momentum transfer terms (streamwise)",
        y_axis_label="Momentum transfer / N/m",
        source=source,
        x_range=x_range,
        specs=specs,
        hover_rows=[
            ("s", "@s{0.000}"),
            ("f_a2sur", "@f_a2sur{0.0000}"),
            ("f_c2a", "@f_c2a{0.0000}"),
            ("f_s2a_total", "@f_s2a_total{0.0000}"),
            ("f_s2sur_total", "@f_s2sur_total{0.0000}"),
            ("f_c2s_total", "@f_c2s_total{0.0000}"),
        ],
        primary_field="f_a2sur",
    )


def _build_transfer_radial_panel(source: ColumnDataSource, x_range: Range1d):
    """Create radial momentum-transfer panel using transfer specs."""

    specs = [
        TransferSpec("f_ra2sur", "air", "sur"),
        TransferSpec("f_rc2a", "core", "air"),
        TransferSpec("f_rs2a_total", "spray_mid", "air", total=True),
        TransferSpec("f_rs2sur_total", "spray_mid", "sur", total=True),
        TransferSpec("f_rc2s_total", "core", "spray_mid", total=True),
        *[
            TransferSpec(f"f_rc2s_{i}", "core", f"spray_{i}", width=1, alpha=0.8)
            for i in range(num_drop_classes)
        ],
        *[
            TransferSpec(f"f_rs2a_{i}", f"spray_{i}", "air", width=1, alpha=0.8)
            for i in range(num_drop_classes)
        ],
        *[
            TransferSpec(f"f_rs2sur_{i}", f"spray_{i}", "sur", width=1, alpha=0.8)
            for i in range(num_drop_classes)
        ],
    ]
    return _build_transfer_panel(
        title="Momentum transfer terms (radial)",
        y_axis_label="Momentum transfer / N/m",
        source=source,
        x_range=x_range,
        specs=specs,
        hover_rows=[
            ("s", "@s{0.000}"),
            ("f_ra2sur", "@f_ra2sur{0.0000}"),
            ("f_rc2a", "@f_rc2a{0.0000}"),
            ("f_rs2a_total", "@f_rs2a_total{0.0000}"),
            ("f_rs2sur_total", "@f_rs2sur_total{0.0000}"),
            ("f_rc2s_total", "@f_rc2s_total{0.0000}"),
        ],
        primary_field="f_ra2sur",
    )


def _build_transfer_panel(
    title: str,
    y_axis_label: str,
    source: ColumnDataSource,
    x_range: Range1d,
    specs: Sequence[TransferSpec],
    hover_rows: list[tuple[str, str]],
    primary_field: str,
):
    """Create one transfer panel from declarative transfer specs."""

    p = _new_panel(
        title=title,
        x_axis_label="Streamwise position s / m",
        y_axis_label=y_axis_label,
        x_range=x_range,
    )
    primary_renderer: GlyphRenderer | None = None
    for spec in specs:
        renderer = _plot_transfer_term(
            fig=p,
            source=source,
            y_field=spec.field,
            source_color=_resolve_phase_color(spec.source_phase),
            target_color=_resolve_phase_color(spec.target_phase),
            dashed=spec.total,
            line_width=spec.width,
            line_alpha=spec.alpha,
        )
        if spec.field == primary_field:
            primary_renderer = renderer

    _add_transfer_encoding_note(p)
    _configure_linear_grid_density([p])
    _add_hover_tool(p, hover_rows, renderer=primary_renderer)
    return p


def _new_panel(
    title: str,
    x_axis_label: str,
    y_axis_label: str,
    x_range: Range1d,
    y_range: Range1d | None = None,
    y_axis_type: str = "linear",
    height: int | None = None,
    match_aspect: bool = False,
):
    """Create a panel with shared plotting defaults."""

    kwargs = {
        "title": title,
        "x_axis_label": x_axis_label,
        "y_axis_label": y_axis_label,
        "x_range": x_range,
        "sizing_mode": "stretch_width",
        "tools": DEFAULT_TOOLS,
        "y_axis_type": y_axis_type,
    }
    if y_range is not None:
        kwargs["y_range"] = y_range
    if height is not None:
        kwargs["height"] = height
    if match_aspect:
        kwargs["match_aspect"] = True
    return figure(**kwargs)


def _add_series_from_specs(
    fig,
    source: ColumnDataSource,
    specs: Sequence[SeriesSpec],
) -> GlyphRenderer:
    """Render line series based on declarative series specs.

    Args:
        fig: Bokeh figure to draw on.
        source: Shared data source.
        specs: Sequence of line-series specs.

    Returns:
        Renderer of the first plotted series (for hover anchoring).
    """

    primary: GlyphRenderer | None = None
    for spec in specs:
        renderer = fig.line(
            "s",
            spec.field,
            source=source,
            line_width=spec.width,
            line_alpha=spec.alpha,
            line_color=spec.color,
            line_dash=spec.dash,
            legend_label=spec.label,
        )
        if primary is None:
            primary = renderer
    assert primary is not None
    return primary


def _plot_transfer_term(
    fig,
    source: ColumnDataSource,
    y_field: str,
    source_color: str,
    target_color: str,
    dashed: bool = False,
    line_width: int = 2,
    line_alpha: float = 1.0,
    marker_step: int = 16,
) -> GlyphRenderer:
    """Plot transfer term as source-colored line with target-colored markers.

    Args:
        fig: Bokeh figure to draw on.
        source: Shared data source.
        y_field: Data column name.
        source_color: Color representing source phase.
        target_color: Color representing target phase.
        dashed: Whether to draw the term as dashed (used for totals).
        line_width: Width of the line glyph.
        line_alpha: Opacity of line glyph.
        marker_step: Index stride for marker decimation.

    Returns:
        Renderer of the line glyph.
    """

    renderer = fig.line(
        "s",
        y_field,
        source=source,
        line_width=line_width,
        line_alpha=line_alpha,
        line_color=source_color,
        line_dash="dashed" if dashed else "solid",
    )

    x_vals = np.asarray(source.data["s"], dtype=float)
    y_vals = np.asarray(source.data[y_field], dtype=float)
    valid_idx = np.where(np.isfinite(x_vals) & np.isfinite(y_vals))[0]
    if valid_idx.size == 0:
        return renderer

    marker_idx = valid_idx[:: max(1, marker_step)]
    marker_source = ColumnDataSource(
        {"s": x_vals[marker_idx], y_field: y_vals[marker_idx]}
    )
    fig.scatter(
        "s",
        y_field,
        source=marker_source,
        marker="circle",
        size=7,
        fill_color=target_color,
        line_color=target_color,
        fill_alpha=0.9,
        line_alpha=0.9,
    )
    return renderer


def _resolve_phase_color(token: str) -> str:
    """Resolve a phase token to its plot color.

    Args:
        token: Phase token such as core/air/stream/sur/spray_i/spray_mid.

    Returns:
        Hex or named color string.
    """

    if token in PHASE_COLORS:
        return PHASE_COLORS[token]
    if token == "sur":
        return SURROUNDINGS_COLOR
    if token == "spray_mid":
        return SPRAY_COLORS[min(2, num_drop_classes - 1)]
    if token.startswith("spray_"):
        idx = int(token.split("_")[1])
        return SPRAY_COLORS[idx]
    raise KeyError(f"Unknown phase color token: {token}")


def _add_transfer_encoding_note(fig) -> None:
    """Add transfer encoding note to the upper-left corner.

    Args:
        fig: Bokeh figure to annotate.
    """

    y_screen = (fig.height if fig.height is not None else 300) - 12
    fig.add_layout(
        Label(
            x=8,
            y=y_screen,
            x_units="screen",
            y_units="screen",
            text="line=source, marker=target, dashed=total, grey=surroundings",
            text_font_size="9pt",
            text_color="#303030",
            text_baseline="top",
            background_fill_color="#ffffff",
            background_fill_alpha=0.7,
        )
    )
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

    p_traj.patch(
        np.concatenate([upper[:, 0], lower[::-1, 0]]),
        np.concatenate([upper[:, 1], lower[::-1, 1]]),
        fill_alpha=0.3,
        fill_color="skyblue",
        line_color="gray",
        line_width=1,
        legend_label="Stream Width",
    )
    return


def _finite_max(source: ColumnDataSource, *fields: str) -> float:
    """Compute maximum finite value across one or more datasource fields."""

    max_value = 0.0
    for field in fields:
        vals = np.asarray(source.data[field], dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size > 0:
            max_value = max(max_value, float(np.max(finite)))
    return max_value


def _log_range_from_fields(
    source: ColumnDataSource, fields: Sequence[str]
) -> Tuple[float, float]:
    """Compute robust positive log-range bounds from multiple fields."""

    min_positive = np.inf
    max_value = 0.0
    for field in fields:
        vals = np.asarray(source.data[field], dtype=float)
        finite_positive = vals[np.isfinite(vals) & (vals > 0.0)]
        if finite_positive.size > 0:
            min_positive = min(min_positive, float(np.min(finite_positive)))
            max_value = max(max_value, float(np.max(finite_positive)))
    if not np.isfinite(min_positive):
        min_positive = 1e-3
    if max_value <= 0.0:
        max_value = 1.0
    start = max(1e-6, min_positive * 0.5)
    end = max(start * 10.0, max_value * 1.2)
    return start, end


def _sum_trace_columns(
    trace_col_getter: Callable[[str], np.ndarray], prefix: str, n_rows: int
) -> np.ndarray:
    """Sum traced vector columns sharing the same prefix.

    Args:
        trace_col_getter: Function fetching one trace column by name.
        prefix: Column prefix, e.g. "m_c2s" for columns "m_c2s[i]".
        n_rows: Row count used for initialization.

    Returns:
        Element-wise sum across all class-specific columns for the prefix.
    """

    total = np.zeros((n_rows,), dtype=float)
    for i in range(num_drop_classes):
        total += trace_col_getter(f"{prefix}[{i}]")
    return total


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


def _add_hover_tool(
    fig, tooltips: list[tuple[str, str]], renderer: GlyphRenderer | None = None
) -> None:
    """Attach one hover tool with optional renderer filtering.

    Args:
        fig: Bokeh figure to augment.
        tooltips: Label/value tooltip tuples shown on hover.
        renderer: Optional single renderer to avoid duplicate tooltips.
    """

    hover = HoverTool(tooltips=tooltips, mode="vline")
    if renderer is not None:
        hover.renderers = [renderer]
    fig.add_tools(hover)
    return
