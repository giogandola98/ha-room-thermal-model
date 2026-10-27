"""Heating runs: thermostat regulation mask and heat-up rate from real episodes.

A radiator behind a thermostatic valve reports "heating" for hours while the valve only
trickles to hold the setpoint. Those rows carry almost no information about the full
heating power ``a`` but dominate the regression and drag ``a`` towards the steady-state
loss. Likewise a run with the heat source off (boiler idle, heating season not started)
looks like "heating with no effect".

- :func:`regulation_mask`: inside every heating run, the rows after the run reached its
  highest temperature are regulation (or ineffective heating) and are not learnt from.
- :func:`episode_heating_rate`: the full-effort heating rate measured directly on the
  heat-up part of real episodes (the room was idle before, the temperature rose), with
  the loss and gains of the current parameters added back. Robust (median) and
  independent of the regression, it replaces ``a`` when the two disagree.
"""

from __future__ import annotations

from dataclasses import dataclass

from .grid import Grid

ON_EFFORT = 0.5
PLATEAU_TOLERANCE_K = 0.1
MIN_EPISODE_RISE_K = 0.3
MIN_EPISODE_MINUTES = 30.0


def heating_runs(grid: Grid, on_effort: float = ON_EFFORT) -> list[tuple[int, int]]:
    """(first, last) row of every maximal run with effort > ``on_effort``."""
    eff = grid.effort
    n = len(eff)
    runs: list[tuple[int, int]] = []
    k = 0
    while k < n:
        e = eff[k]
        if e is not None and e > on_effort:
            j = k
            while j + 1 < n and (ej := eff[j + 1]) is not None and ej > on_effort:
                j += 1
            runs.append((k, j))
            k = j + 1
        else:
            k += 1
    return runs


def regulation_mask(grid: Grid, tolerance_k: float = PLATEAU_TOLERANCE_K) -> list[bool]:
    """True for heating rows after the run's maximum temperature (minus tolerance)."""
    mask = [False] * len(grid)
    ind = grid.indoor
    for k, j in heating_runs(grid):
        temps = [(i, t) for i in range(k, j + 1) if (t := ind[i]) is not None]
        if not temps:
            continue
        top = max(t for _, t in temps)
        reach = next(i for i, t in temps if t >= top - tolerance_k)
        for i in range(reach + 1, j + 1):
            mask[i] = True
    return mask


@dataclass
class EpisodeRate:
    a_k_per_h: float  # median full-effort heating rate (loss and gains added back)
    episodes: int
    rates: list[float]


def episode_heating_rate(
    grid: Grid,
    b: float,
    c: float,
    d_steps: int,
    use_outdoor: bool,
    min_episodes: int = 2,
) -> EpisodeRate | None:
    """Median ``a`` over heat-up episodes, or None with fewer than ``min_episodes``.

    Episode: a heating run preceded by at least ``d_steps + 1`` rows without heating.
    The heat-up part goes from ``start + d_steps`` (heat has arrived) to the first row
    at the run's maximum temperature; it must last ``MIN_EPISODE_MINUTES`` and rise by
    ``MIN_EPISODE_RISE_K``. Over that part ``a = ΔT/Δt + b·(mean T − mean Tout) − c``.
    """
    eff, ind, out, exc = grid.effort, grid.indoor, grid.outdoor, grid.excluded
    step_h = grid.step_minutes / 60.0
    pre = d_steps + 1
    rates: list[float] = []
    for k, j in heating_runs(grid):
        if k < pre or any(e is None or e > ON_EFFORT for e in eff[k - pre : k]):
            continue
        s = k + d_steps
        if s > j:
            continue
        seg = range(s, j + 1)
        if any(exc[i] or ind[i] is None for i in seg):
            continue
        top = max(ind[i] for i in seg)  # type: ignore[type-var]
        e_end = next(i for i in seg if ind[i] >= top - 1e-9)  # type: ignore[operator]
        dur_h = (e_end - s) * step_h
        rise = ind[e_end] - ind[s]  # type: ignore[operator]
        if dur_h * 60.0 < MIN_EPISODE_MINUTES or rise < MIN_EPISODE_RISE_K:
            continue
        temps = [ind[i] for i in range(s, e_end + 1)]
        mean_t = sum(temps) / len(temps)  # type: ignore[arg-type]
        if use_outdoor:
            outs = [v for i in range(s, e_end + 1) if (v := out[i]) is not None]
            if not outs:
                continue
            mean_out = sum(outs) / len(outs)
        else:
            mean_out = 0.0
        rates.append(rise / dur_h + b * (mean_t - mean_out) - c)
    if len(rates) < min_episodes:
        return None
    srt = sorted(rates)
    n = len(srt)
    med = srt[n // 2] if n % 2 else 0.5 * (srt[n // 2 - 1] + srt[n // 2])
    return EpisodeRate(med, n, [round(r, 4) for r in rates])
