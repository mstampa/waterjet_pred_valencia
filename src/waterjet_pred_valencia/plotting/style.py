"""Style utilities applied across all generated panels."""

from collections.abc import Iterable

from bokeh.models import AdaptiveTicker

from .constants import DESIRED_MAJOR_TICKS


def configure_linear_grid_density(figures: Iterable) -> None:
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
