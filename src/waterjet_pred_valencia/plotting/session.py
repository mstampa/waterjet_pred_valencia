"""Panel session helpers for live simulation plotting."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from time import perf_counter
from typing import Any

from bokeh.themes import built_in_themes
from pandas import DataFrame

from ..parameters import get_breakup_distance
from ..simulator import simulate
from ..tracer import Tracer
from .api import PlotLayoutParts, build_solution_layout, build_trace_layout

logger = logging.getLogger(__name__)

_APP_TITLE = "Fire stream trajectory simulation"
_CONTROL_WIDTH = 180

_CONTROL_PANEL_STYLES = {
    "border": "1px solid #3a3a3a",
    "border-radius": "6px",
    "padding": "10px 12px",
    "background": "#1f1f1f",
}

_STATUS_PANE_STYLES = {
    "font-size": "1.1rem",
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
    progress_callback: Callable[[float], None] | None = None,
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
        progress_callback: Optional callback receiving current streamwise `s`.

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
            progress_callback=progress_callback,
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

    First render stays cheap so browser can open immediately. The initial
    simulation then runs in a background thread while the status line reports
    elapsed wall-clock time and latest streamwise `s`.

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
        width=_CONTROL_WIDTH,
    )
    speed_input = pn.widgets.FloatInput(
        name="Injection speed [m/s]",
        value=float(injection_speed),
        step=0.1,
        width=_CONTROL_WIDTH,
    )
    nozzle_input = pn.widgets.FloatInput(
        name="Nozzle diameter [m]",
        value=float(nozzle_diameter),
        step=1e-4,
        width=_CONTROL_WIDTH,
    )
    height_input = pn.widgets.FloatInput(
        name="Initial height [m]",
        value=float(injection_height),
        step=0.1,
        width=_CONTROL_WIDTH,
    )
    replot_button = pn.widgets.Button(
        name="Simulate",
        button_type="primary",
        width=_CONTROL_WIDTH,
    )
    status_pane = pn.pane.Markdown(
        "Ready.",
        sizing_mode="stretch_width",
        styles=_STATUS_PANE_STYLES,
    )
    trajectory_pane = pn.pane.Bokeh(sizing_mode="stretch_width", min_height=420)
    diagnostics_pane = pn.pane.Bokeh(sizing_mode="stretch_width")

    curdoc = getattr(pn.state, "curdoc", None)
    if curdoc is not None:
        curdoc.theme = built_in_themes["dark_minimal"]
    run_state = {
        "active": False,
        "started_at": 0.0,
        "last_s": None,
        "periodic_callback": None,
    }

    def _set_status(status: str) -> None:
        status_pane.object = status
        return

    def _format_running_status() -> str:
        elapsed_s = max(0.0, perf_counter() - float(run_state["started_at"]))
        last_s = run_state["last_s"]
        if last_s is None:
            return f"Simulating... {elapsed_s:.1f} s elapsed, s=n/a."
        return f"Simulating... {elapsed_s:.1f} s elapsed, s={float(last_s):.3f} m."

    def _stop_periodic_status_updates() -> None:
        periodic_callback = run_state["periodic_callback"]
        if curdoc is None or periodic_callback is None:
            return
        curdoc.remove_periodic_callback(periodic_callback)
        run_state["periodic_callback"] = None
        return

    def _queue_ui_callback(callback: Callable[[], None]) -> None:
        if curdoc is None:
            callback()
            return
        curdoc.add_next_tick_callback(callback)
        return

    def _refresh_running_status() -> None:
        if not run_state["active"]:
            return
        _set_status(_format_running_status())
        return

    def _start_periodic_status_updates() -> None:
        if curdoc is None or run_state["periodic_callback"] is not None:
            return
        run_state["periodic_callback"] = curdoc.add_periodic_callback(
            _refresh_running_status,
            1000,
        )
        return

    def _apply_plot(parts: PlotLayoutParts | None, status: str) -> None:
        if parts is not None:
            trajectory_pane.object = parts.trajectory
            diagnostics_pane.object = parts.diagnostics
        _set_status(status)
        return

    def _current_control_values() -> dict[str, float]:
        return {
            "injection_angle_deg": float(angle_input.value),
            "injection_speed": float(speed_input.value),
            "nozzle_diameter": float(nozzle_input.value),
            "injection_height": float(height_input.value),
        }

    def _compute_current_state() -> tuple[PlotLayoutParts | None, str]:
        try:
            return run_simulation_plot(
                span=span,
                max_step=max_step,
                debug=debug,
                csv_path=csv_path,
                progress_callback=lambda s: run_state.__setitem__("last_s", float(s)),
                **_current_control_values(),
            )
        except Exception as exc:
            logger.exception(
                "Interactive simulation failed before any plot data existed."
            )
            return (
                None,
                f"Simulation failed before any plot data was recorded: `{exc}`.",
            )

    def _finish_run(parts: PlotLayoutParts | None, status: str) -> None:
        _stop_periodic_status_updates()
        run_state["active"] = False
        replot_button.disabled = False
        _apply_plot(parts, status)
        return

    def _run_current_state_in_background() -> None:
        parts, status = _compute_current_state()
        _queue_ui_callback(lambda: _finish_run(parts, status))
        return

    def _schedule_refresh() -> None:
        if run_state["active"]:
            return

        run_state["active"] = True
        run_state["started_at"] = perf_counter()
        run_state["last_s"] = 0.0
        replot_button.disabled = True
        _set_status(_format_running_status())
        if curdoc is None:
            parts, status = _compute_current_state()
            _finish_run(parts, status)
            return

        _start_periodic_status_updates()
        Thread(target=_run_current_state_in_background, daemon=True).start()
        return

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
            sizing_mode="stretch_width",
        ),
        replot_button,
        status_pane,
        sizing_mode="stretch_width",
        styles=_CONTROL_PANEL_STYLES,
    )
    replot_button.on_click(lambda _: _schedule_refresh())
    _schedule_refresh()
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

    Each browser session gets a fresh layout and starts its first simulation
    after the page is already servable, which avoids blocking browser startup.

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
    pn.serve(
        lambda: create_simulation_plot_session(
            injection_angle_deg=injection_angle_deg,
            injection_speed=injection_speed,
            nozzle_diameter=nozzle_diameter,
            injection_height=injection_height,
            span=span,
            max_step=max_step,
            debug=debug,
            csv_path=csv_path,
        ).layout,
        port=port,
        show=show,
        threaded=True,
        title=_APP_TITLE,
    )
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
    pn.config.theme = "dark"
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
