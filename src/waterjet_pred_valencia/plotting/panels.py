"""Panel builders for simulation plots."""

from collections.abc import Sequence

from bokeh.models import ColumnDataSource, Range1d
from bokeh.models.renderers import GlyphRenderer

from ..parameters import num_drop_classes
from .constants import PHASE_COLORS, SPRAY_COLORS
from .helpers import (
    add_hover_tool,
    add_series_from_specs,
    add_stream_width_patch,
    add_transfer_encoding_note,
    finite_max,
    log_range_from_fields,
    new_panel,
    plot_transfer_term,
    resolve_phase_color,
)
from .types import SeriesSpec, TransferSpec


def build_trajectory_panel(source: ColumnDataSource):
    """Create trajectory panel with stream-width patch and hover."""

    x_max = finite_max(source, "x")
    y_max = finite_max(source, "y")
    p = new_panel(
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
    add_stream_width_patch(p, source)
    add_hover_tool(
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


def build_speed_panel(source: ColumnDataSource, x_range: Range1d):
    """Create phase-speed panel using series specs."""

    fields = ["Uc", "Ua", "Uf"] + [f"Us_{i}" for i in range(num_drop_classes)]
    speed_max = finite_max(source, *fields)
    p = new_panel(
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
    primary = add_series_from_specs(p, source, specs)
    p.legend.location = "bottom_left"
    add_hover_tool(
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


def build_diameter_panel(source: ColumnDataSource, x_range: Range1d):
    """Create phase-diameter panel using series specs."""

    dmax = finite_max(source, "Dc", "Da", "Df")
    p = new_panel(
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
    primary = add_series_from_specs(p, source, specs)
    p.legend.location = "top_left"
    add_hover_tool(
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


def build_angle_panel(source: ColumnDataSource, x_range: Range1d):
    """Create phase-angle panel using series specs."""

    p = new_panel(
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
    primary = add_series_from_specs(p, source, specs)
    p.legend.location = "bottom_left"
    add_hover_tool(
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


def build_nd_panel(source: ColumnDataSource, x_range: Range1d):
    """Create drop-count panel with log y-axis."""

    nd_min, nd_max = log_range_from_fields(
        source, [f"ND_{i}" for i in range(num_drop_classes)]
    )
    p = new_panel(
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
    primary = add_series_from_specs(p, source, specs)
    p.legend.location = "bottom_right"
    add_hover_tool(
        p,
        [("s", "@s{0.000}")]
        + [(f"ND{i}", f"@ND_{i}{{0.00}}") for i in range(num_drop_classes)],
        renderer=primary,
    )
    return p


def build_rho_panel(source: ColumnDataSource, x_range: Range1d):
    """Create stream density panel."""

    p = new_panel(
        title="Stream density",
        x_axis_label="Streamwise position s / m",
        y_axis_label="Density / kg/m³",
        x_range=x_range,
        y_range=Range1d(0.0, 1000.0),
        height=300,
    )
    primary = add_series_from_specs(
        p,
        source,
        [SeriesSpec("rho_f", "rho_f", PHASE_COLORS["stream"])],
    )
    add_hover_tool(p, [("s", "@s{0.000}"), ("rho_f", "@rho_f{0.00}")], renderer=primary)
    return p


def build_transfer_mass_panel(source: ColumnDataSource, x_range: Range1d):
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


def build_transfer_stream_panel(source: ColumnDataSource, x_range: Range1d):
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


def build_transfer_radial_panel(source: ColumnDataSource, x_range: Range1d):
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
    """Create one transfer panel from declarative transfer specs.

    Args:
        title: Figure title.
        y_axis_label: Label for y-axis.
        source: Shared data source.
        x_range: Shared x-axis range.
        specs: Transfer-term rendering specs.
        hover_rows: Tooltip rows for hover display.
        primary_field: Field used to bind the single hover renderer.

    Returns:
        Configured transfer figure.
    """

    p = new_panel(
        title=title,
        x_axis_label="Streamwise position s / m",
        y_axis_label=y_axis_label,
        x_range=x_range,
    )
    primary_renderer: GlyphRenderer | None = None
    for spec in specs:
        renderer = plot_transfer_term(
            fig=p,
            source=source,
            y_field=spec.field,
            source_color=resolve_phase_color(spec.source_phase),
            target_color=resolve_phase_color(spec.target_phase),
            dashed=spec.total,
            line_width=spec.width,
            line_alpha=spec.alpha,
        )
        if spec.field == primary_field:
            primary_renderer = renderer

    add_transfer_encoding_note(p)
    add_hover_tool(p, hover_rows, renderer=primary_renderer)
    return p
