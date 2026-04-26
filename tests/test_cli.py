#!/usr/bin/env python3

"""Tests for CLI startup and interactive app orchestration."""

from argparse import Namespace
from pathlib import Path

from waterjet_pred_valencia import cli


def _make_args(tmp_path: Path) -> Namespace:
    """Build a minimal CLI argument namespace for tests.

    Args:
        tmp_path: Temporary directory provided by pytest.

    Returns:
        Namespace with all fields required by run_simulation().
    """

    return Namespace(
        speed=30.8,
        angle=24.0,
        nozzle=0.0254,
        y0=0.0,
        csv=tmp_path / "trace.csv",
        debug=False,
        max_step=1e-3,
        port=5012,
        span=100.0,
        show=False,
    )


def test_main_starts_panel_server_with_cli_arguments(tmp_path, monkeypatch):
    """Ensure CLI forwards parsed arguments into Panel server startup."""

    args = _make_args(tmp_path)
    captured: dict[str, object] = {}

    def _fake_get_arguments() -> Namespace:
        return args

    def _fake_start_server(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(cli, "get_arguments", _fake_get_arguments)
    monkeypatch.setattr(cli, "start_server", _fake_start_server)

    assert cli.main() == 0
    assert captured == {
        "injection_angle_deg": args.angle,
        "injection_speed": args.speed,
        "nozzle_diameter": args.nozzle,
        "injection_height": args.y0,
        "span": args.span,
        "max_step": args.max_step,
        "debug": args.debug,
        "csv_path": args.csv,
        "port": args.port,
        "show": args.show,
    }
