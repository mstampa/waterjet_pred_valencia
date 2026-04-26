#!/usr/bin/env python3

"""Tests for interactive Panel plotting session helpers."""

from pathlib import Path
from types import SimpleNamespace

from waterjet_pred_valencia.plotting.render import PlotLayoutParts
from waterjet_pred_valencia.plotting.session import (
    create_simulation_plot_session,
    run_simulation_plot,
    start_server,
)


class _FakeWatcher:
    def __init__(self, widget):
        self._widget = widget

    def watch(self, callback, _name):
        self._widget._watchers.append(callback)


class _FakeWidget:
    def __init__(self, name=None, value=None, **kwargs):
        self.name = name
        self.value = value
        self.disabled = kwargs.get("disabled", False)
        self._click_callbacks = []
        self._watchers = []
        self.param = _FakeWatcher(self)

    def on_click(self, callback):
        self._click_callbacks.append(callback)

    def click(self):
        for callback in self._click_callbacks:
            callback(None)


class _FakePane:
    def __init__(self, object=None, **kwargs):
        self.object = object
        self.kwargs = kwargs


class _FakeLayout:
    def __init__(self, *children, **kwargs):
        self.children = list(children)
        self.kwargs = kwargs


class _FakeDoc:
    def __init__(self):
        self.callbacks = []
        self.periodic_callbacks = []
        self.removed_periodic_callbacks = []

    def add_next_tick_callback(self, callback):
        self.callbacks.append(callback)

    def add_periodic_callback(self, callback, period):
        handle = SimpleNamespace(callback=callback, period=period)
        self.periodic_callbacks.append(handle)
        return handle

    def remove_periodic_callback(self, callback):
        self.removed_periodic_callbacks.append(callback)


class _FakeThread:
    instances = []

    def __init__(self, target, daemon):
        self.target = target
        self.daemon = daemon
        self.started = False
        self.__class__.instances.append(self)

    def start(self):
        self.started = True

    def run(self):
        self.target()


class _FakePanel:
    def __init__(self, doc):
        self.state = SimpleNamespace(curdoc=doc)
        self.widgets = SimpleNamespace(
            FloatInput=_FakeWidget,
            Button=_FakeWidget,
        )
        self.pane = SimpleNamespace(
            Markdown=_FakePane,
            Bokeh=_FakePane,
        )
        self.Row = _FakeLayout
        self.Column = _FakeLayout
        self.served = None

    def extension(self):
        return None

    def serve(self, layout, **kwargs):
        self.served = (layout, kwargs)


def test_run_simulation_plot_uses_trace_layout_after_partial_failure(monkeypatch, tmp_path):
    """Partial traces should still render a fallback layout and export CSV."""

    def _fake_simulate(*_, **kwargs):
        tracer = kwargs["tracer"]
        tracer.snapshot(
            s=0.0,
            scalars={"x": 0.0, "y": 0.0, "theta_a_deg": 24.0, "theta_f_deg": 24.0},
            vectors={},
        )
        raise RuntimeError("forced failure")

    fake_parts = PlotLayoutParts(
        trajectory=SimpleNamespace(title=SimpleNamespace(text="traj")),
        diagnostics=SimpleNamespace(children=[]),
        full_layout=SimpleNamespace(children=[]),
    )

    monkeypatch.setattr(
        "waterjet_pred_valencia.plotting.session.simulate",
        _fake_simulate,
    )
    monkeypatch.setattr(
        "waterjet_pred_valencia.plotting.session.build_trace_layout",
        lambda *_, **__: fake_parts,
    )

    csv_path = tmp_path / "trace.csv"
    parts, status = run_simulation_plot(
        injection_angle_deg=24.0,
        injection_speed=30.8,
        nozzle_diameter=0.0254,
        csv_path=csv_path,
    )

    assert parts is fake_parts
    assert "partial trace" in status.lower()
    assert csv_path.exists()


def test_create_simulation_plot_session_shows_simulating_then_updates(monkeypatch):
    """Startup and button click should show progress before queued rerender runs."""

    _FakeThread.instances = []
    doc = _FakeDoc()
    fake_panel = _FakePanel(doc)
    fake_parts = PlotLayoutParts(
        trajectory=SimpleNamespace(title=SimpleNamespace(text="Fire stream trajectory")),
        diagnostics=SimpleNamespace(children=[1, 2, 3, 4]),
        full_layout=SimpleNamespace(children=[]),
    )
    calls = []

    monkeypatch.setattr(
        "waterjet_pred_valencia.plotting.session._load_panel",
        lambda: fake_panel,
    )
    monkeypatch.setattr(
        "waterjet_pred_valencia.plotting.session.run_simulation_plot",
        lambda **kwargs: calls.append(kwargs)
        or (fake_parts, "Simulation updated in 0.001 s."),
    )
    monkeypatch.setattr(
        "waterjet_pred_valencia.plotting.session.Thread",
        _FakeThread,
    )

    session = create_simulation_plot_session(
        injection_angle_deg=24.0,
        injection_speed=30.8,
        nozzle_diameter=0.0254,
    )

    controls = session.layout.children[1]
    button = controls.children[3]
    status_pane = controls.children[2]
    trajectory_pane = session.layout.children[0]
    diagnostics_pane = session.layout.children[2]

    assert status_pane.object.startswith("Simulating...")
    assert len(doc.periodic_callbacks) == 1
    assert doc.periodic_callbacks[0].period == 1000
    assert len(_FakeThread.instances) == 1
    _FakeThread.instances.pop(0).run()
    assert len(doc.callbacks) == 1
    doc.callbacks.pop()()
    assert status_pane.object.startswith("Simulation updated")
    assert trajectory_pane.object is fake_parts.trajectory
    assert diagnostics_pane.object is fake_parts.diagnostics
    assert doc.removed_periodic_callbacks

    button.click()
    assert status_pane.object.startswith("Simulating...")
    assert len(doc.periodic_callbacks) == 2
    _FakeThread.instances.pop(0).run()
    assert len(doc.callbacks) == 1
    doc.callbacks.pop()()
    assert len(calls) == 2


def test_start_server_serves_created_layout(monkeypatch):
    """Server startup should pass a fresh layout factory into Panel serve."""

    doc = _FakeDoc()
    fake_panel = _FakePanel(doc)
    fake_layout = SimpleNamespace(name="layout")

    monkeypatch.setattr(
        "waterjet_pred_valencia.plotting.session._load_panel",
        lambda: fake_panel,
    )
    monkeypatch.setattr(
        "waterjet_pred_valencia.plotting.session.create_simulation_plot_session",
        lambda **kwargs: SimpleNamespace(layout=fake_layout),
    )

    start_server(
        injection_angle_deg=24.0,
        injection_speed=30.8,
        nozzle_diameter=0.0254,
        csv_path=Path("/tmp/trace.csv"),
        port=5010,
        show=False,
    )

    layout_factory, kwargs = fake_panel.served
    assert layout_factory() is fake_layout
    assert kwargs == {
        "port": 5010,
        "show": False,
        "threaded": True,
        "title": "Fire stream trajectory simulation",
    }
