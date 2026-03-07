"""Public plotting functions used by the CLI and tests."""

import logging
from pathlib import Path
from typing import Dict

from pandas import DataFrame
from scipy.integrate._ivp.ivp import OdeResult

from .render import save_plot
from .source import build_source_from_solution, build_source_from_trace

logger = logging.getLogger(__name__)


def plot_solution(sol: OdeResult, state_idx: Dict[str, int], path: Path) -> None:
    """Plot trajectory and variable evolution from a successful ODE result.

    Args:
        sol: ODE solution object from solve_ivp.
        state_idx: Mapping of variable names to indices in sol.y.
        path: Path to export the plot to (html).
    """

    logger.info("Plotting OdeResult...")
    source, s_end = build_source_from_solution(sol, state_idx)
    save_plot(source=source, s_end=s_end, path=path)
    return


def plot_trace(trace_df: DataFrame, path: Path) -> None:
    """Plot trajectory and variable evolution from traced partial data.

    Args:
        trace_df: Wide trace dataframe produced by Tracer.to_wide_dataframe().
        path: Path to export the plot to (html).
    """

    logger.info("Plotting traced partial results...")
    source, s_end = build_source_from_trace(trace_df)
    save_plot(source=source, s_end=s_end, path=path)
    return
