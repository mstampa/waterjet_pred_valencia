"""Build bokeh data sources from solver output and trace data."""

from collections.abc import Callable

import numpy as np
from bokeh.models import ColumnDataSource
from pandas import DataFrame
from scipy.integrate._ivp.ivp import OdeResult

from ..jet_state import JetState
from ..parameters import num_drop_classes


def build_source_from_solution(
    sol: OdeResult, state_idx: dict[str, int]
) -> tuple[ColumnDataSource, float]:
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

    data: dict[str, np.ndarray] = {
        "s": sol.t,
        "Uc": sol.y[JetState.get_idx("Uc"), :],
        "Dc": sol.y[JetState.get_idx("Dc"), :],
        "Ua": sol.y[JetState.get_idx("Ua"), :],
        "Da": sol.y[JetState.get_idx("Da"), :],
        "theta_a_deg": np.rad2deg(sol.y[JetState.get_idx("theta_a"), :]),
        "Uf": sol.y[JetState.get_idx("Uf"), :],
        "Df": sol.y[JetState.get_idx("Df"), :],
        "theta_f_deg": np.rad2deg(sol.y[JetState.get_idx("theta_f"), :]),
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


def build_source_from_trace(trace_df: DataFrame) -> tuple[ColumnDataSource, float]:
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
    data: dict[str, np.ndarray] = {
        "s": s,
        "Uc": _trace_col("Uc"),
        "Dc": _trace_col("Dc"),
        "Ua": _trace_col("Ua"),
        "Da": _trace_col("Da"),
        "theta_a_deg": _trace_col("theta_a_deg"),
        "Uf": _trace_col("Uf"),
        "Df": _trace_col("Df"),
        "theta_f_deg": _trace_col("theta_f_deg"),
        "rho_f": _trace_col("rho_f"),
        "x": _trace_col("x"),
        "y": _trace_col("y"),
        **{f"ND_{i}": _trace_col(f"ND[{i}]") for i in range(num_drop_classes)},
        **{f"Us_{i}": _trace_col(f"Us[{i}]") for i in range(num_drop_classes)},
        **{
            f"theta_s_deg_{i}": _trace_col(f"theta_s_deg[{i}]")
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
