"""Top-level rendering and layout composition for simulation plots."""

import logging
from pathlib import Path

from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Range1d
from bokeh.plotting import output_file, save

from .panels import (
    build_angle_panel,
    build_diameter_panel,
    build_nd_panel,
    build_rho_panel,
    build_speed_panel,
    build_trajectory_panel,
    build_transfer_mass_panel,
    build_transfer_radial_panel,
    build_transfer_stream_panel,
)
from .style import configure_linear_grid_density

logger = logging.getLogger(__name__)


def save_plot(source: ColumnDataSource, s_end: float, path: Path) -> None:
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
    p_traj = build_trajectory_panel(source)
    p_speeds = build_speed_panel(source, x_range)
    p_diameters = build_diameter_panel(source, x_range)
    p_angles = build_angle_panel(source, x_range)
    p_nd = build_nd_panel(source, x_range)
    p_rho = build_rho_panel(source, x_range)
    p_mass = build_transfer_mass_panel(source, x_range)
    p_mom_stream = build_transfer_stream_panel(source, x_range)
    p_mom_radial = build_transfer_radial_panel(source, x_range)

    configure_linear_grid_density(
        [
            p_traj,
            p_speeds,
            p_diameters,
            p_angles,
            p_rho,
            p_mass,
            p_mom_stream,
            p_mom_radial,
        ]
    )

    plot_layout = column(
        p_traj,
        row(p_speeds, p_diameters, sizing_mode="stretch_width"),
        row(p_angles, p_nd, sizing_mode="stretch_width"),
        row(p_rho, p_mass, sizing_mode="stretch_width"),
        row(p_mom_stream, p_mom_radial, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )
    save(plot_layout)
    logger.info("Plot saved.")
    return
