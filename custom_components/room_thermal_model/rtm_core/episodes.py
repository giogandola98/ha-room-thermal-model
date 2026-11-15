"""Heating runs: thermostat regulation mask and heat-up rate from real episodes.

A radiator behind a thermostatic valve reports "heating" for hours while the valve only
trickles to hold the setpoint. Those rows carry almost no information about the full
heating power ``a`` but dominate the regression and drag ``a`` towards the steady-state
loss. Likewise a run with the heat source off (boiler idle, heating season not started)
looks like "heating with no effect".

- :func:`regulation_mask`: inside every heating run, the rows after the run reached its
  highest temperature are regulation (or ineffective heating) and are not learnt from.
- :func:`learning_effort`: a thermostatic valve throttles inside its proportional band
  while the thermostat still reports "heating" (effort 1), so with on/off effort the
  heating rows within ``APPROACH_BAND_K`` of the run's top have an unknown effort and
  would teach a heating rate well below the full power. A real effort sensor (valve
  opening, fractional values) measures the throttling and is kept as is.
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
# thermostatic valve proportional band (typically 0.5-1 K): reported on/off heating within
# this distance of the run's top temperature may be throttled and is not learnt from
APPROACH_BAND_K = 0.7
FULL_EFFORT = 0.99
ON_OFF_FULL_SHARE = 0.5  # share of heating rows at full effort for on/off reporting


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


def on_off_effort(effort: list[float | None]) -> bool:
    """True when the effort is reported on/off (``hvac_action``), not a measured opening.

    On/off effort is fractional only on the grid steps where it switches (time average),
    a valve-opening sensor mostly in between.
    """
    on = [e for e in effort if e is not None and e > 0.0]
    full = sum(1 for e in on if e >= FULL_EFFORT)
    return bool(on) and full >= ON_OFF_FULL_SHARE * len(on)


def learning_effort(
    grid: Grid, regulation: list[bool] | None = None, band_k: float = APPROACH_BAND_K
) -> list[float | None]:
    """Effort to learn from, None where the reported effort is not the heat delivered.

    - ``regulation`` rows (see :func:`regulation_mask`);
    - with on/off effort, every reported-heating row (any effort > 0, so short blips
      too) within ``band_k`` of its run's top temperature: the valve may be throttling.

    The valve acts on the temperature at that moment and the heat arrives a dead time
    later, so the regression must drop row ``k`` when ``u[k−d]`` is unreliable, not row
    ``k`` itself; an unknown (None) effort does exactly that for every dead-time candidate.
    """
    effort = list(grid.effort)
    if regulation is not None:
        for i, skip in enumerate(regulation):
            if skip:
                effort[i] = None
    if not on_off_effort(grid.effort):
        return effort
    ind = grid.indoor
    for k, j in heating_runs(grid, on_effort=0.0):
        temps = [t for i in range(k, j + 1) if (t := ind[i]) is not None]
        if not temps:
            continue
        cut = max(temps) - band_k
        for i in range(k, j + 1):
            if (t := ind[i]) is not None and t >= cut:
                effort[i] = None
    return effort


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
    at the run's maximum temperature (with on/off effort: at ``APPROACH_BAND_K`` below it,
    before a thermostatic valve throttles); it must last ``MIN_EPISODE_MINUTES`` and rise by
    ``MIN_EPISODE_RISE_K``. Over that part ``a = ΔT/Δt + b·(mean T − mean Tout) − c``.
    """
    eff, ind, out, exc = grid.effort, grid.indoor, grid.outdoor, grid.excluded
    step_h = grid.step_minutes / 60.0
    pre = d_steps + 1
    rates: list[float] = []
    band = APPROACH_BAND_K if on_off_effort(eff) else 1e-9
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
        e_end = next(i for i in seg if ind[i] >= top - band)  # type: ignore[operator]
        # the rise starts at the lowest temperature: a run that began as regulation
        # (throttled valve, setpoint raised later) only heats up from there
        s = min(range(s, e_end + 1), key=lambda i: ind[i])  # type: ignore[arg-type,return-value]
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
