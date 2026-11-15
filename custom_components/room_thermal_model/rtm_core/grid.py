"""Resampling of raw observations onto the fixed model grid.

Row ``k`` of a :class:`Grid` describes the interval ``[t_k, t_k + step)``:

- ``indoor[k]`` / ``outdoor[k]`` / ``setpoint[k]``: last observation at or before ``t_k``
  (last-observation-carried-forward), or ``None`` when that observation is older than
  ``max_gap`` (``outdoor_max_gap`` for outdoor). Large gaps therefore break the series.
  The setpoint (climate target) changes in steps and is carried forward like the outdoor
  value; ``None`` when the climate reports no target.
- ``effort[k]``: time-weighted mean of the (piecewise constant) effort over the interval.
  Each effort observation is held until the next one, at most ``max_gap``; ``None`` when
  less than half of the interval is covered.
- ``excluded[k]``: an excluded observation is in force somewhere in the interval.

Samples are ``Sample`` objects; ``None`` in a field means "not observed in this sample".
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .base import Sample


def ts_of(dt: datetime) -> float:
    """Epoch seconds of an aware datetime."""
    return dt.timestamp()


def dt_of(ts: float) -> datetime:
    """Aware UTC datetime of epoch seconds."""
    return datetime.fromtimestamp(ts, tz=UTC)


def floor_ts(ts: float, step_s: float) -> float:
    return math.floor(ts / step_s + 1e-9) * step_s


def ceil_ts(ts: float, step_s: float) -> float:
    return math.ceil(ts / step_s - 1e-9) * step_s


@dataclass
class Grid:
    """Contiguous column arrays on a regular grid (``start_ts`` = epoch s of row 0)."""

    step_minutes: float
    start_ts: float | None = None
    indoor: list[float | None] = field(default_factory=list)
    outdoor: list[float | None] = field(default_factory=list)
    effort: list[float | None] = field(default_factory=list)
    excluded: list[bool] = field(default_factory=list)
    setpoint: list[float | None] = field(default_factory=list)

    @property
    def step_s(self) -> float:
        return self.step_minutes * 60.0

    def __len__(self) -> int:
        return len(self.indoor)

    @property
    def end_ts(self) -> float | None:
        """Exclusive end of the last row (== start of the next row to be built)."""
        if self.start_ts is None:
            return None
        return self.start_ts + len(self.indoor) * self.step_s

    def ts(self, k: int) -> float:
        assert self.start_ts is not None
        return self.start_ts + k * self.step_s

    def valid_indoor_count(self) -> int:
        return sum(
            1 for v, ex in zip(self.indoor, self.excluded, strict=True) if v is not None and not ex
        )

    def append_rows(self, other: Grid) -> None:
        """Append ``other`` after this grid, padding the gap between them with empty rows."""
        if not len(other) or other.start_ts is None:
            return
        if self.start_ts is None or not len(self):
            self.start_ts = other.start_ts
            self.indoor = list(other.indoor)
            self.outdoor = list(other.outdoor)
            self.effort = list(other.effort)
            self.excluded = list(other.excluded)
            self.setpoint = list(other.setpoint)
            return
        end = self.end_ts
        assert end is not None
        gap = round((other.start_ts - end) / self.step_s)
        if gap < 0:
            raise ValueError("append_rows: other grid overlaps this grid")
        self._pad(gap)
        self.indoor.extend(other.indoor)
        self.outdoor.extend(other.outdoor)
        self.effort.extend(other.effort)
        self.excluded.extend(other.excluded)
        self.setpoint.extend(other.setpoint)

    def _pad(self, n: int) -> None:
        self.indoor.extend([None] * n)
        self.outdoor.extend([None] * n)
        self.effort.extend([None] * n)
        self.excluded.extend([False] * n)
        self.setpoint.extend([None] * n)

    def trim_before(self, cutoff_ts: float) -> None:
        """Drop rows starting before ``cutoff_ts``."""
        if self.start_ts is None or not len(self):
            return
        n = math.ceil((cutoff_ts - self.start_ts) / self.step_s - 1e-9)
        if n <= 0:
            return
        n = min(n, len(self))
        del (
            self.indoor[:n],
            self.outdoor[:n],
            self.effort[:n],
            self.excluded[:n],
            self.setpoint[:n],
        )
        self.start_ts += n * self.step_s

    def merged(self, other: Grid) -> Grid:
        """Union of two grids; rows of ``self`` win unless their indoor value is missing."""
        if self.start_ts is None or not len(self):
            return other.copy()
        if other.start_ts is None or not len(other):
            return self.copy()
        step = self.step_s
        start = min(self.start_ts, other.start_ts)
        end = max(self.end_ts or start, other.end_ts or start)
        n = round((end - start) / step)
        out = Grid(self.step_minutes, start)
        out._pad(n)
        for src in (other, self):  # self last: wins
            off = round((src.start_ts - start) / step)  # type: ignore[operator]
            for i in range(len(src)):
                j = off + i
                if src.indoor[i] is None and out.indoor[j] is not None:
                    continue
                out.indoor[j] = src.indoor[i]
                out.outdoor[j] = src.outdoor[i]
                out.effort[j] = src.effort[i]
                out.excluded[j] = src.excluded[i]
                out.setpoint[j] = src.setpoint[i] if i < len(src.setpoint) else None
        return out

    def copy(self) -> Grid:
        return Grid(
            self.step_minutes,
            self.start_ts,
            list(self.indoor),
            list(self.outdoor),
            list(self.effort),
            list(self.excluded),
            list(self.setpoint),
        )

    def snapshot(self) -> Grid:
        """Consistent copy, safe against rows appended/trimmed concurrently."""
        start = self.start_ts
        cols = (
            list(self.indoor),
            list(self.outdoor),
            list(self.effort),
            list(self.excluded),
            list(self.setpoint),
        )
        n = min(len(c) for c in cols)
        if start is not None and self.start_ts != start:  # trimmed while copying: retry
            return self.snapshot()
        return Grid(self.step_minutes, start, *(c[:n] for c in cols))

    # -- compact JSON form ------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Integer-scaled columns: °C × 100, effort × 1000, excluded as row indices."""
        return {
            "step_minutes": self.step_minutes,
            "start_ts": self.start_ts,
            "indoor_c100": [None if v is None else round(v * 100) for v in self.indoor],
            "outdoor_c100": [None if v is None else round(v * 100) for v in self.outdoor],
            "effort_e3": [None if v is None else round(v * 1000) for v in self.effort],
            "excluded_rows": [i for i, ex in enumerate(self.excluded) if ex],
            "setpoint_c100": [None if v is None else round(v * 100) for v in self.setpoint],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Grid:
        indoor = [None if v is None else v / 100.0 for v in data.get("indoor_c100", [])]
        n = len(indoor)
        outdoor = [None if v is None else v / 100.0 for v in data.get("outdoor_c100", [])]
        effort = [None if v is None else v / 1000.0 for v in data.get("effort_e3", [])]
        if len(outdoor) != n or len(effort) != n:
            raise ValueError("inconsistent grid columns")
        excluded = [False] * n
        for i in data.get("excluded_rows", []):
            if 0 <= i < n:
                excluded[i] = True
        # Old snapshots have no setpoint column: default to all-None (fallback path).
        raw_sp = data.get("setpoint_c100")
        if raw_sp is None:
            setpoint: list[float | None] = [None] * n
        else:
            if len(raw_sp) != n:
                raise ValueError("inconsistent grid columns")
            setpoint = [None if v is None else v / 100.0 for v in raw_sp]
        start = data.get("start_ts")
        return cls(float(data["step_minutes"]), start, indoor, outdoor, effort, excluded, setpoint)


def resample(
    samples: Iterable[Sample],
    step_minutes: float,
    max_gap_minutes: float,
    outdoor_max_gap_minutes: float | None = None,
    start_ts: float | None = None,
    end_ts: float | None = None,
) -> Grid:
    """Resample raw samples (any spacing, any order) onto the grid.

    ``start_ts`` defaults to the first sample time rounded up to the grid, ``end_ts``
    (exclusive end of the last row) to the last sample time rounded down: a row is only
    built once the data covers its whole interval.
    """
    step_s = step_minutes * 60.0
    max_gap_s = max_gap_minutes * 60.0
    out_gap_s = max(max_gap_minutes, outdoor_max_gap_minutes or 0.0) * 60.0
    rows = sorted(samples, key=lambda s: s.ts)
    grid = Grid(step_minutes)
    if not rows:
        return grid
    times = [ts_of(s.ts) for s in rows]
    if start_ts is None:
        start_ts = ceil_ts(times[0], step_s)
    if end_ts is None:
        end_ts = floor_ts(times[-1], step_s)
    n = int(round((end_ts - start_ts) / step_s))
    grid.start_ts = start_ts
    if n <= 0:
        return grid

    indoor_ev = [
        (t, s.indoor_c) for t, s in zip(times, rows, strict=True) if s.indoor_c is not None
    ]
    outdoor_ev = [
        (t, s.outdoor_c) for t, s in zip(times, rows, strict=True) if s.outdoor_c is not None
    ]
    effort_ev = [(t, s.effort) for t, s in zip(times, rows, strict=True) if s.effort is not None]
    setpoint_ev = [
        (t, s.setpoint_c) for t, s in zip(times, rows, strict=True) if s.setpoint_c is not None
    ]

    grid.indoor = _locf(indoor_ev, start_ts, step_s, n, max_gap_s)
    grid.outdoor = _locf(outdoor_ev, start_ts, step_s, n, out_gap_s)
    grid.effort = _step_mean(effort_ev, start_ts, step_s, n, max_gap_s)
    # Setpoint changes in steps: carry it forward like the outdoor value.
    grid.setpoint = _locf(setpoint_ev, start_ts, step_s, n, out_gap_s)

    excluded = [False] * n
    for i, s in enumerate(rows):
        if not s.excluded:
            continue
        t0 = times[i]
        t1 = min(times[i + 1] if i + 1 < len(rows) else t0, t0 + max_gap_s)
        k0 = max(0, math.floor((t0 - start_ts) / step_s))
        k1 = min(n - 1, math.floor((max(t1, t0) - start_ts) / step_s - 1e-9))
        for k in range(k0, k1 + 1):
            excluded[k] = True
    grid.excluded = excluded
    return grid


def _locf(
    events: list[tuple[float, float]], start: float, step: float, n: int, max_age: float
) -> list[float | None]:
    out: list[float | None] = [None] * n
    j = -1
    m = len(events)
    for k in range(n):
        t = start + k * step
        while j + 1 < m and events[j + 1][0] <= t + 1e-6:
            j += 1
        if j >= 0 and t - events[j][0] <= max_age + 1e-6:
            out[k] = events[j][1]
    return out


def _step_mean(
    events: list[tuple[float, float]], start: float, step: float, n: int, max_hold: float
) -> list[float | None]:
    out: list[float | None] = [None] * n
    m = len(events)
    j = 0
    for k in range(n):
        a = start + k * step
        b = a + step
        # first event that may still be in force at ``a``
        while j + 1 < m and events[j + 1][0] <= a:
            j += 1
        covered = 0.0
        integral = 0.0
        i = j
        while i < m and events[i][0] < b:
            t_i, v = events[i]
            t_next = events[i + 1][0] if i + 1 < m else t_i + max_hold
            hold_end = min(t_next, t_i + max_hold)
            lo = max(a, t_i)
            hi = min(b, hold_end)
            if hi > lo:
                covered += hi - lo
                integral += (hi - lo) * v
            i += 1
        if covered >= 0.5 * step:
            out[k] = integral / covered
    return out
