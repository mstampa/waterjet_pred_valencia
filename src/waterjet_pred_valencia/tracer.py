#!/usr/bin/env python

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
from pandas import DataFrame


@dataclass
class TraceRowWide:
    """
    One row of 'wide' tracing: all scalars and every vector element in a single dict.
    """

    data: Dict[str, Any] = field(default_factory=dict)


class Tracer:
    """
    Lightweight tracer to record all variables within ode_right_hand_side() per
    simulation step for debugging.

    Features:
      - s_stride: minimum spacing (in meters) between recorded rows (e.g., 0.05 for 5 cm).
      - decimals: round all numeric outputs to this many decimal places in-memory.
      - vector column naming: '<name>[i]' for i-th element.
      - output as CSV or Parquet
    """

    def __init__(self, s_stride: float = 0.01, decimals: int = 6):
        """
        Constructor.

        Args:
            s_stride: spacing between recordings along 's' [m]
            decimals: number of decimals to print into output file
        """
        self.rows: List[TraceRowWide] = []
        self.s_stride = float(s_stride)
        self.decimals = int(decimals)
        self._next_s_mark: Optional[float] = None  # set on first snapshot

    def _should_record(self, s: float) -> bool:
        """
        Decide whether to record this step based on s_stride.
        """

        if self._next_s_mark is None:
            # Align first mark to current s rounded down to the nearest stride, then record.
            base = np.floor(s / self.s_stride) * self.s_stride
            self._next_s_mark = base

        # Allow small numerical slack so we don't miss the mark due to float error
        eps = 1e-12
        assert self._next_s_mark is not None
        if s + eps >= self._next_s_mark:
            # Prepare the next mark (may jump multiple strides if solver took a big step)
            n_strides = max(
                1, int(np.floor((s - self._next_s_mark) / self.s_stride) + 1)
            )
            self._next_s_mark += n_strides * self.s_stride
            return True

        return False

    def snapshot(
        self, s: float, scalars: Dict[str, float], vectors: Dict[str, np.ndarray]
    ):
        """Append a 'wide' row if s advanced past the next stride mark.

        Args:
            s: Current arclength (or independent variable) in meters.
            scalars: Mapping of scalar names -> float values.
            vectors: Mapping of vector names -> 1-D arrays (same length per name).
        """
        if not self._should_record(float(s)):
            return

        row: Dict[str, Any] = {}

        # Always include s (rounded)
        row["s"] = np.round(float(s), self.decimals)

        # Round scalars
        for k, v in scalars.items():
            row[k] = np.round(float(v), self.decimals)

        # Round and flatten vectors: name -> name[i]
        for name, arr in vectors.items():
            a = np.asarray(arr, dtype=float)
            a = np.round(a, self.decimals)
            for i, val in enumerate(a):
                row[f"{name}[{i}]"] = val

        self.rows.append(TraceRowWide(data=row))

    def to_wide_dataframe(self) -> DataFrame:
        """
        Return a wide pandas DataFrame: one row per logged s; columns include scalars
        and vector elements.

        Returns:
            pandas.DataFrame
        """
        import pandas as pd

        if not self.rows:
            return pd.DataFrame(
                {
                    "s": [],
                }
            )
        # Union of all keys (some vectors/scalars might be missing in first rows if you add later)
        all_keys = set()
        for r in self.rows:
            all_keys.update(r.data.keys())

        # Stable column order: s first, then others sorted
        all_keys.discard("s")
        columns = ["s"] + sorted(all_keys)

        # Build records
        records = []
        for r in self.rows:
            rec = {c: np.nan for c in columns}
            for k, v in r.data.items():
                rec[k] = v
            records.append(rec)

        return DataFrame.from_records(records, columns=columns)

    def to_csv(self, path: str):
        """Write wide CSV with numeric formatting."""
        df = self.to_wide_dataframe()
        df.to_csv(path, index=False, float_format=f"%.{self.decimals}f")

    def to_parquet(self, path: str, compression: str = "zstd"):
        """Write Parquet (compact, fast). Use if CSV still too big."""
        df = self.to_wide_dataframe()
        df.to_parquet(path, compression=compression, index=False)
