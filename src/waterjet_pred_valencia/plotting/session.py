"""Panel session helpers for live simulation plotting."""

import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from pandas import DataFrame

from ..parameters import get_breakup_distance
from ..simulator import simulate
from ..tracer import Tracer
from .api import PlotLayoutParts, build_solution_layout, build_trace_layout

logger = logging.getLogger(__name__)

_CONTROL_PANEL_STYLES = {
    "border": "1px solid #d9d9d9",
    "border-radius": "6px",
    "padding": "10px 12px",
    "background": "#fafafa",
}


@dataclass(frozen=True)
class SimulationPlotSession:
    """Mutable handles for one interactive simulation plotting session.

    Attributes:
        layout: Top-level Panel layout served in the browser.
    """

    layout: Any


def run_simulation_plot(
    injection_angle_deg: float,
    injection_speed: float,
    nozzle_diameter: float,
    injection_height: float = 0.0,
    span: float = 100.0,
    max_step: float = 1e-3,
    debug: bool = False,
    csv_path: Path | None = None,
) -> tuple[PlotLayoutParts | None, str]:
    """Run one simulation and build reusable plot layouts for its result.

    Args:
        injection_angle_deg: theta_0 nozzle elevation angle above horizon [deg].
        injection_speed: U_0 nozzle exit speed [m/s].
        nozzle_diameter: D_0 nozzle diameter [m].
        injection_height: y_0 nozzle height above ground [m].
        span: Maximum simulated streamwise position [m].
        max_step: Maximum ODE integration step size [m].
        debug: Enable solver debug mode.
        csv_path: Optional trace export path written after the run.

    Returns:
        Tuple of reusable plot layouts and a short status message. The layout is
        ``None`` only when the solver fails before any trace rows are recorded.

    Raises:
        Exception: Re-raises the underlying solver error only when no trace data exists.
    """

    tracer = Tracer()
    s_breakup = get_breakup_distance(nozzle_diameter)
    started_at = perf_counter()

    try:
        result = simulate(
            injection_angle_deg=injection_angle_deg,
            injection_speed=injection_speed,
            nozzle_diameter=nozzle_diameter,
            injection_height=injection_height,
            s_span=(0.0, span),
            max_step=max_step,
            debug=debug,
            tracer=tracer,
        )
    except Exception as exc:
        trace_df = tracer.to_wide_dataframe()
        _export_trace_if_requested(trace_df, tracer=tracer, csv_path=csv_path)
        elapsed_s = perf_counter() - started_at
        if trace_df.empty:
            raise
        logger.exception("Simulation failed after partial trace generation.")
        return (
            build_trace_layout(trace_df, s_breakup=s_breakup),
            f"Simulation failed after {elapsed_s:.3f} s. Showing partial trace: `{exc}`.",
        )

    assert result.sol is not None
    trace_df = tracer.to_wide_dataframe()
    _export_trace_if_requested(trace_df, tracer=tracer, csv_path=csv_path)
    elapsed_s = perf_counter() - started_at
    return (
        build_solution_layout(result.sol, result.state_idx, s_breakup=s_breakup),
        f"Simulation updated in {elapsed_s:.3f} s.",
    )


def create_simulation_plot_session(
    injection_angle_deg: float,
    injection_speed: float,
    nozzle_diameter: float,
    injection_height: float = 0.0,
    span: float = 100.0,
    max_step: float = 1e-3,
    debug: bool = False,
    csv_path: Path | None = None,
) -> SimulationPlotSession:
    """Build one interactive Panel app for simulations.

    Args:
        injection_angle_deg: Initial angle shown in the control panel [deg].
        injection_speed: Initial nozzle exit speed shown in the control panel [m/s].
        nozzle_diameter: Initial nozzle diameter shown in the control panel [m].
        injection_height: Initial nozzle height shown in the control panel [m].
        span: Fixed streamwise simulation limit [m].
        max_step: Fixed maximum ODE step size [m].
        debug: Enable solver debug mode.
        csv_path: Optional trace export path written after each run.

    Returns:
        Session object containing the top-level Panel layout.
    """

    pn = _load_panel()
    angle_input = pn.widgets.FloatInput(
        name="Injection angle [deg]",
        value=float(injection_angle_deg),
        step=1.0,
        sizing_mode="stretch_width",
    )
    speed_input = pn.widgets.FloatInput(
        name="Injection speed [m/s]",
        value=float(injection_speed),
        step=0.1,
        sizing_mode="stretch_width",
    )
    nozzle_input = pn.widgets.FloatInput(
        name="Nozzle diameter [m]",
        value=float(nozzle_diameter),
        step=1e-4,
        sizing_mode="stretch_width",
    )
    height_input = pn.widgets.FloatInput(
        name="Initial height [m]",
        value=float(injection_height),
        step=0.1,
        sizing_mode="stretch_width",
    )
    replot_button = pn.widgets.Button(
        name="Simulate",
        button_type="primary",
        sizing_mode="stretch_width",
    )
    status_pane = pn.pane.Markdown("Ready.", sizing_mode="stretch_width")
    trajectory_pane = pn.pane.Bokeh(sizing_mode="stretch_width", min_height=420)
    diagnostics_pane = pn.pane.Bokeh(sizing_mode="stretch_width")

    run_state = {"active": False}

    def _apply_plot(parts: PlotLayoutParts | None, status: str) -> None:
        if parts is not None:
            trajectory_pane.object = parts.trajectory
            diagnostics_pane.object = parts.diagnostics
        status_pane.object = status
        return

    def _current_control_values() -> dict[str, float]:
        return {
            "injection_angle_deg": float(angle_input.value),
            "injection_speed": float(speed_input.value),
            "nozzle_diameter": float(nozzle_input.value),
            "injection_height": float(height_input.value),
        }

    def _render_current_state() -> None:
        try:
            parts, status = run_simulation_plot(
                span=span,
                max_step=max_step,
                debug=debug,
                csv_path=csv_path,
                **_current_control_values(),
            )
        except Exception as exc:
            logger.exception(
                "Interactive simulation failed before any plot data existed."
            )
            _apply_plot(
                None,
                f"Simulation failed before any plot data was recorded: `{exc}`.",
            )
        else:
            _apply_plot(parts, status)
        finally:
            run_state["active"] = False
            replot_button.disabled = False
        return

    def _schedule_refresh() -> None:
        if run_state["active"]:
            return

        run_state["active"] = True
        replot_button.disabled = True
        status_pane.object = "Simulating..."
        curdoc = getattr(pn.state, "curdoc", None)
        if curdoc is None:
            _render_current_state()
            return
        curdoc.add_next_tick_callback(_render_current_state)
        return

    replot_button.on_click(lambda _: _schedule_refresh())
    _schedule_refresh()

    controls = pn.Column(
        pn.pane.Markdown(
            "**Initial conditions**",
            sizing_mode="stretch_width",
        ),
        pn.Row(
            angle_input,
            speed_input,
            nozzle_input,
            height_input,
            replot_button,
            sizing_mode="stretch_width",
        ),
        status_pane,
        sizing_mode="stretch_width",
        styles=_CONTROL_PANEL_STYLES,
    )
    layout = pn.Column(
        trajectory_pane,
        controls,
        diagnostics_pane,
        sizing_mode="stretch_width",
    )
    return SimulationPlotSession(layout=layout)


def start_server(
    injection_angle_deg: float,
    injection_speed: float,
    nozzle_diameter: float,
    injection_height: float = 0.0,
    span: float = 100.0,
    max_step: float = 1e-3,
    debug: bool = False,
    csv_path: Path | None = None,
    port: int = 5007,
    show: bool = True,
) -> None:
    """Start a local Panel server for live water-jet plotting.

    Args:
        injection_angle_deg: Initial angle shown in the control panel [deg].
        injection_speed: Initial nozzle exit speed shown in the control panel [m/s].
        nozzle_diameter: Initial nozzle diameter shown in the control panel [m].
        injection_height: Initial nozzle height shown in the control panel [m].
        span: Fixed streamwise simulation limit [m].
        max_step: Fixed maximum ODE step size [m].
        debug: Enable solver debug mode.
        csv_path: Optional trace export path written after each run.
        port: Bokeh server port.
        show: If ``True``, open the browser automatically.
    """

    pn = _load_panel()
    session = create_simulation_plot_session(
        injection_angle_deg=injection_angle_deg,
        injection_speed=injection_speed,
        nozzle_diameter=nozzle_diameter,
        injection_height=injection_height,
        span=span,
        max_step=max_step,
        debug=debug,
        csv_path=csv_path,
    )
    pn.serve(session.layout, port=port, show=show, threaded=True)
    return


def _load_panel() -> Any:
    """Import and initialize Panel.

    Returns:
        Imported ``panel`` module.

    Raises:
        ImportError: If ``panel`` is not installed.
    """

    try:
        import panel as pn
    except ImportError as exc:
        raise ImportError(
            "Interactive plotting requires 'panel'. Install it via: pip install panel"
        ) from exc

    pn.extension()
    return pn


def _export_trace_if_requested(
    trace_df: DataFrame,
    tracer: Tracer,
    csv_path: Path | None,
) -> None:
    """Write trace CSV after a run when export was requested."""

    if csv_path is None:
        return
    if trace_df.empty:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        trace_df.to_csv(csv_path, index=False)
        return
    tracer.to_csv(csv_path)
    return
