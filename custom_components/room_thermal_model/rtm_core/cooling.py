"""Time constant from free-cooling segments (cross-check / fallback for ``b``).

Approach borrowed from Ecronika/ha-preheat (MIT License, Copyright (c) 2025 Ecronika),
https://github.com/Ecronika/ha-preheat — re-implemented here on the model grid.

For every segment without heating (≥ 60 min, ≥ 8 points) the indoor temperature
relaxes towards the outdoor temperature: ``ln(T_in − mean(T_out))`` is fitted by
ordinary least squares against time; ``τ = −1/slope``. Segments are accepted when
R² > 0.5 and 0.5 h < τ < 200 h; the result is the length-weighted median.

Internal gains shift the true asymptote above ``T_out`` and bias τ: the main fit is
preferred and this estimate is only used when the main fit lacks excitation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .grid import Grid

MIN_SEGMENT_MINUTES = 60.0
MIN_POINTS = 8
MIN_R2 = 0.5
TAU_MIN_H, TAU_MAX_H = 0.5, 200.0
OFF_EFFORT = 0.01
MIN_DELTA_K = 0.5


@dataclass
class CoolingEstimate:
    tau_hours: float
    segments: int
    taus: list[float]


def _ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Return (slope, r2)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    if sxx <= 0:
        return 0.0, 0.0
    slope = sxy / sxx
    r2 = (sxy * sxy) / (sxx * syy) if syy > 0 else 0.0
    return slope, r2


def cooling_segments(grid: Grid, skip_minutes: float = 30.0) -> list[tuple[int, int]]:
    """Index ranges [k0, k1) of contiguous no-heating rows with indoor and outdoor data.

    The first ``skip_minutes`` after heating stops are skipped (heat still arriving).
    """
    step = grid.step_minutes
    skip = int(math.ceil(skip_minutes / step))
    segs: list[tuple[int, int]] = []
    k0: int | None = None
    n = len(grid)
    for k in range(n + 1):
        ok = (
            k < n
            and grid.indoor[k] is not None
            and grid.outdoor[k] is not None
            and not grid.excluded[k]
            and (e := grid.effort[k]) is not None
            and e <= OFF_EFFORT
        )
        if ok and k0 is None:
            k0 = k
        elif not ok and k0 is not None:
            # skip the first rows only if the segment started right after heating
            start = k0
            if k0 > 0 and (e := grid.effort[k0 - 1]) is not None and e > OFF_EFFORT:
                start = k0 + skip
            if k - start > 0:
                segs.append((start, k))
            k0 = None
    return segs


def cooling_tau(grid: Grid, skip_minutes: float = 30.0) -> CoolingEstimate | None:
    """Length-weighted median τ (hours) over accepted cooling segments."""
    step = grid.step_minutes
    taus: list[tuple[float, int]] = []
    for k0, k1 in cooling_segments(grid, skip_minutes):
        npts = k1 - k0
        if npts < MIN_POINTS or (npts - 1) * step < MIN_SEGMENT_MINUTES:
            continue
        tin = [grid.indoor[k] for k in range(k0, k1)]
        tout = [grid.outdoor[k] for k in range(k0, k1)]
        mean_out = sum(tout) / npts  # type: ignore[arg-type]
        diffs = [t - mean_out for t in tin]  # type: ignore[operator]
        if min(diffs) <= MIN_DELTA_K:
            continue
        xs = [i * step / 60.0 for i in range(npts)]
        ys = [math.log(dv) for dv in diffs]
        slope, r2 = _ols(xs, ys)
        if slope >= 0 or r2 <= MIN_R2:
            continue
        tau = -1.0 / slope
        if TAU_MIN_H < tau < TAU_MAX_H:
            taus.append((tau, npts))
    if not taus:
        return None
    taus.sort()
    total = sum(n for _, n in taus)
    acc = 0
    med = taus[-1][0]
    for tau, n in taus:
        acc += n
        if acc >= total / 2:
            med = tau
            break
    return CoolingEstimate(med, len(taus), [round(t, 3) for t, _ in taus])
