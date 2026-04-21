"""Reusable plotting helpers for panel rendering and styling."""

from collections.abc import Sequence

import numpy as np
from bokeh.models import ColumnDataSource, HoverTool, Label, Range1d, Span
from bokeh.models.renderers import GlyphRenderer
from bokeh.plotting import figure

from .constants import DEFAULT_TOOLS, PHASE_COLORS, SPRAY_COLORS, SURROUNDINGS_COLOR
from .types import SeriesSpec


def new_panel(
    title: str,
    x_axis_label: str,
    y_axis_label: str,
    x_range: Range1d,
    y_range: Range1d | None = None,
    y_axis_type: str = "linear",
    height: int | None = None,
    match_aspect: bool = False,
):
    """Create a panel with shared plotting defaults.

    Args:
        title: Figure title.
        x_axis_label: X-axis label.
        y_axis_label: Y-axis label.
        x_range: X-axis range.
        y_range: Optional y-axis range.
        y_axis_type: Axis type (linear/log).
        height: Optional fixed panel height.
        match_aspect: Whether x/y scales should match.

    Returns:
        Configured bokeh figure.
    """

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


def add_series_from_specs(
    fig,
    source: ColumnDataSource,
    specs: Sequence[SeriesSpec],
) -> GlyphRenderer:
    """Render line series from declarative specs.

    Args:
        fig: Bokeh figure to draw on.
        source: Shared data source.
        specs: Sequence of line-series specs.

    Returns:
        Renderer of the first plotted series.
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


def plot_transfer_term(
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
        dashed: Whether to draw the term as dashed (for totals).
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


def add_transfer_encoding_note(fig) -> None:
    """Add transfer encoding note to upper-left corner of a panel.

    Args:
        fig: Bokeh figure to annotate.
    """

    fig.add_layout(
        Label(
            x=8,
            y=6,
            x_units="screen",
            y_units="screen",
            text="line color=source phase"
            + ", marker color=target phase"
            + ", dashed=total, grey=surroundings",
            text_font_size="9pt",
            text_color="#303030",
            background_fill_color="#ffffff",
            background_fill_alpha=0.7,
        )
    )
    return


def resolve_phase_color(token: str) -> str:
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
        return SPRAY_COLORS[min(2, len(SPRAY_COLORS) - 1)]
    if token.startswith("spray_"):
        idx = int(token.split("_")[1])
        return SPRAY_COLORS[idx]
    raise KeyError(f"Unknown phase color token: {token}")


def add_hover_tool(
    fig, tooltips: list[tuple[str, str]], renderer: GlyphRenderer | None = None
) -> None:
    """Attach one hover tool with optional renderer filter.

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


def add_breakup_marker(fig, s_breakup: float | None) -> None:
    """Add a vertical dotted breakup marker for panels using s on x-axis.

    Args:
        fig: Bokeh figure to augment.
        s_breakup: Breakup location along streamwise axis [m]. If None, no marker is added.
    """

    if s_breakup is None:
        return

    fig.add_layout(
        Span(
            location=s_breakup,
            dimension="height",
            line_color="#4b4b4b",
            line_width=2,
            line_dash="dotted",
        )
    )
    return


def finite_max(source: ColumnDataSource, *fields: str) -> float:
    """Compute maximum finite value across one or more datasource fields."""

    max_value = 0.0
    for field in fields:
        vals = np.asarray(source.data[field], dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size > 0:
            max_value = max(max_value, float(np.max(finite)))
    return max_value


def log_range_from_fields(
    source: ColumnDataSource, fields: Sequence[str]
) -> tuple[float, float]:
    """Compute robust positive log-range bounds from multiple fields.

    Args:
        source: Shared data source.
        fields: Column names to inspect.

    Returns:
        Tuple of (start, end) for a log axis.
    """

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


def add_stream_width_patch(p_traj, source: ColumnDataSource) -> None:
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
