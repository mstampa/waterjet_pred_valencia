"""Defines Tracer class and helpers."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame

logger = logging.getLogger(__name__)


@dataclass
class TraceRowWide:
    """Represents one 'wide' row: All scalars and vector elements in a single dict."""

    data: Dict[str, Any] = field(default_factory=dict)


class Tracer:
    """Lightweight tracer for all variables within ode_right_hand_side() per sim-step.

    Features:
      - s_stride: Minimum spacing between recorded rows [m].
      - decimals: Round all numeric outputs to this many decimal places in-memory.
      - vector column naming: '<name>[i]' for i-th element.
      - output as CSV.
    """

    def __init__(self, s_stride: float = 0.05, decimals: int = 6):
        """Initialize Tracer.

        Args:
            s_stride: Spacing between recordings along 's' [m]
            decimals: Number of decimals to print into output file.
        """
        self.rows: List[TraceRowWide] = []
        self.s_stride = float(s_stride)
        self.decimals = int(decimals)
        self._next_s_mark: Optional[float] = None  # Set on first snapshot.
        logger.info(f"Tracer initialized with stride={s_stride} m, {decimals=}.")
        return

    def _should_record(self, s: float) -> bool:
        """Decide whether to record this step based on s_stride.

        Args:
            s: Current position along streamwise axis [m].

        Returns:
            True if step should be recorded.
        """

        if self._next_s_mark is None:
            # Align first mark to current s rounded down to nearest stride, then record.
            base = np.floor(s / self.s_stride) * self.s_stride
            self._next_s_mark = base

        # Allow small numerical slack so we don't miss the mark due to float error.
        eps: float = 1e-12
        assert self._next_s_mark is not None
        if s + eps >= self._next_s_mark:
            # Prepare next mark (may jump multiple strides if solver took a big step).
            n_strides = max(
                1, int(np.floor((s - self._next_s_mark) / self.s_stride) + 1)
            )
            self._next_s_mark += n_strides * self.s_stride
            return True

        return False

    def snapshot(
        self,
        s: float,
        scalars: Dict[str, float],
        vectors: Dict[str, NDArray[np.floating]],
    ) -> None:
        """Append a 'wide' row if s advanced past the next stride mark.

        Args:
            s: Current arclength (or independent variable) [m].
            scalars: Mapping of scalar names -> float values.
            vectors: Mapping of vector names -> 1-D arrays (same length per name).
        """
        if not self._should_record(float(s)):
            return

        row: Dict[str, Any] = {}

        # Always include s (rounded).
        row["s"] = np.round(float(s), self.decimals)

        # Round scalars.
        for k, v in scalars.items():
            row[k] = np.round(float(v), self.decimals)

        # Round and flatten vectors: name -> name[i].
        for name, arr in vectors.items():
            a = np.asarray(arr, dtype=float)
            a = np.round(a, self.decimals)
            for i, val in enumerate(a):
                row[f"{name}[{i}]"] = val

        self.rows.append(TraceRowWide(data=row))
        return

    def to_wide_dataframe(self) -> DataFrame:
        """Return a wide pandas.DataFrame.

        One row per logged s. Columns include scalars and vector elements.

        Returns:
            pandas.DataFrame
        """

        if not self.rows:
            return pd.DataFrame(
                {
                    "s": [],
                }
            )
        all_keys = set()
        for r in self.rows:
            all_keys.update(r.data.keys())

        # Stable column order: s first, then others sorted.
        all_keys.discard("s")
        columns = ["s"] + sorted(all_keys)

        # Build records.
        records = []
        for r in self.rows:
            rec = {c: np.nan for c in columns}
            for k, v in r.data.items():
                rec[k] = v
            records.append(rec)

        return DataFrame.from_records(records, columns=columns)

    def to_csv(self, path: Path) -> None:
        """Export data as comma-separated values (CSV) file.

        Args:
            path: Where to save the file to (must have .csv extension).
        """
        logger.info(f"Saving trace to {path}...")
        assert path.suffix == ".csv", "File extension must be '.csv' !"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            logger.warning(f"Trace file already exists and will be overwritten: {path}")
        df: DataFrame = self.to_wide_dataframe()
        df.to_csv(str(path), index=False, float_format=f"%.{self.decimals}f")
        logger.info("Trace saved.")
        return
