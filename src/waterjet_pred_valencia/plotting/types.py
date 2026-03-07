"""Dataclasses for declarative plot-series definitions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SeriesSpec:
    """Description for rendering one standard line series."""

    field: str
    label: str
    color: str
    width: int = 2
    alpha: float = 1.0
    dash: str = "solid"


@dataclass(frozen=True)
class TransferSpec:
    """Description for rendering one transfer-term series."""

    field: str
    source_phase: str
    target_phase: str
    total: bool = False
    width: int = 2
    alpha: float = 1.0
