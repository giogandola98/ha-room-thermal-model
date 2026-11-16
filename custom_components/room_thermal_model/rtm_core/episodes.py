"""Heating runs: thermostat regulation mask and heat-up rate from real episodes.

A radiator behind a thermostatic valve reports "heating" for hours while the valve only
trickles to hold the setpoint. Those rows carry almost no information about the full
heating power ``a`` but dominate the regression and drag ``a`` towards the steady-state
loss. Likewise a run with the heat source off (boiler idle, heating season not started)
looks like "heating with no effect".

- :func:`regulation_mask`: inside every heating run, the rows after the run reached its
  highest temperature are regulation (or ineffective heating) and are not learnt from.
- :func:`learning_effort`: a thermostatic valve throttles inside its proportional band
  while the thermostat still reports "heating" (effort 1), so with on/off effort those
  rows have an unknown effort and would teach a heating rate well below the full power.
  When the per-sample setpoint is known, a heating row is throttling when the room is
  within ``SETPOINT_BAND_K`` *below its current setpoint* -- this is what catches the
  overnight-eco -> morning-comfort case, where the eco regulation around ~17 °C and the
  climb to ~21 °C are one long reported-heating run and a band around the run top (21 °C)
  would miss the throttled eco part. Without a setpoint the detection falls back to
  ``APPROACH_BAND_K`` of the run's top (old history). A real effort sensor (valve
  opening, fractional values) measures the throttling and is kept as is.
- :func:`episode_heating_rate`: the full-effort heating rate measured directly on the
  heat-up part of real episodes (the room was idle before, the temperature rose), with
  the loss and gains of the current parameters added back. Robust (median) and
  independent of the regression, it replaces ``a`` when the two disagree. A heat-up is
  often reported as several runs separated by short pauses (burner cycling, a valve
  briefly closing): runs with pauses up to ``MERGE_PAUSE_MINUTES`` form one episode and
  the measured rate is divided by the duty cycle of the effort over the climb.
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
# same idea but measured against the *known* per-sample setpoint instead of the run top:
# a room within this distance below its current setpoint while "heating" is a throttling
# valve, so its effort is unknown
SETPOINT_BAND_K = 0.6
# without a setpoint the episode climb ends this far below the run's top: a wider band
# (``APPROACH_BAND_K``) would discard every heat-up smaller than ~1 K
EPISODE_TOP_BAND_K = 0.3
MERGE_PAUSE_MINUTES = 15.0  # pauses up to this long inside a heat-up are one episode
MIN_DUTY = 0.3
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


def merged_runs(
    grid: Grid, on_effort: float = ON_EFFORT, max_pause_minutes: float = MERGE_PAUSE_MINUTES
) -> list[tuple[int, int]]:
    """:func:`heating_runs` with runs separated by short known-idle pauses merged.

    Only pauses whose rows all have a known effort are bridged (missing data is not a
    pause).
    """
    max_rows = int(max_pause_minutes / grid.step_minutes + 1e-9)
    eff = grid.effort
    out: list[tuple[int, int]] = []
    for k, j in heating_runs(grid, on_effort):
        if out:
            k0, j0 = out[-1]
            gap = range(j0 + 1, k)
            if len(gap) <= max_rows and all(eff[i] is not None for i in gap):
                out[-1] = (k0, j)
                continue
        out.append((k, j))
    return out


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


def _setpoint_column(grid: Grid) -> list[float | None]:
    """The grid's setpoint column, padded to ``len(grid)`` with None.

    Grids built positionally in tests (and very old snapshots) may carry a shorter or
    empty setpoint column; treat the missing rows as "setpoint unknown".
    """
    n = len(grid)
    sp = grid.setpoint
    if len(sp) == n:
        return sp
    return [sp[i] if i < len(sp) else None for i in range(n)]


def learning_effort(
    grid: Grid,
    regulation: list[bool] | None = None,
    band_k: float = APPROACH_BAND_K,
    setpoint_band_k: float = SETPOINT_BAND_K,
) -> list[float | None]:
    """Effort to learn from, None where the reported effort is not the heat delivered.

    - ``regulation`` rows (see :func:`regulation_mask`);
    - with on/off effort, every reported-heating row (any effort > 0, so short blips too)
      that is throttling. A row with a *known* setpoint throttles when the room is within
      ``setpoint_band_k`` below that setpoint; a row without one falls back to ``band_k``
      of its run's top temperature (old history, no setpoint recorded).

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
    setpoint = _setpoint_column(grid)
    for k, j in heating_runs(grid, on_effort=0.0):
        temps = [t for i in range(k, j + 1) if (t := ind[i]) is not None]
        if not temps:
            continue
        top_cut = max(temps) - band_k  # fallback cut for rows without a setpoint
        for i in range(k, j + 1):
            t = ind[i]
            if t is None:
                continue
            sp = setpoint[i]
            if sp is not None:
                if t >= sp - setpoint_band_k:
                    effort[i] = None
            elif t >= top_cut:
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

    Episode: a heating run (pauses ≤ ``MERGE_PAUSE_MINUTES`` merged, see
    :func:`merged_runs`) preceded by at least ``d_steps + 1`` rows without heating.
    The heat-up part goes from ``start + d_steps`` (heat has arrived) to the row at which
    the valve starts throttling, i.e. the room reaches its setpoint band (``SETPOINT_BAND_K``
    below the known setpoint) or, with no setpoint, ``EPISODE_TOP_BAND_K`` below the run's top.
    When the run opens with the valve already throttling against a lower setpoint (overnight
    eco held around ~17 °C before the morning comfort target), those leading throttled rows
    are skipped and only the genuine full-power climb is measured. The heat-up must last
    ``MIN_EPISODE_MINUTES`` and rise by ``MIN_EPISODE_RISE_K``. Over that part
    ``a = (ΔT/Δt + b·(mean T − mean Tout) − c) / duty`` where ``duty`` is the mean effort
    that acted on the climb (``u[i − d]``; 1 for an uninterrupted run).
    """
    eff, ind, out, exc = grid.effort, grid.indoor, grid.outdoor, grid.excluded
    setpoint = _setpoint_column(grid)
    step_h = grid.step_minutes / 60.0
    pre = d_steps + 1
    rates: list[float] = []
    on_off = on_off_effort(eff)
    band = EPISODE_TOP_BAND_K if on_off else 1e-9
    for k, j in merged_runs(grid):
        if k < pre or any(e is None or e > ON_EFFORT for e in eff[k - pre : k]):
            continue
        s = k + d_steps
        if s > j:
            continue
        seg = range(s, j + 1)
        if any(exc[i] or ind[i] is None for i in seg):
            continue
        top = max(ind[i] for i in seg)  # type: ignore[type-var]

        def throttling(i: int, top: float = top) -> bool:
            """True where a reported-heating row is really the valve throttling."""
            if not on_off:
                return ind[i] >= top - band  # type: ignore[operator]
            sp = setpoint[i]
            if sp is not None:
                return ind[i] >= sp - SETPOINT_BAND_K  # type: ignore[operator]
            return ind[i] >= top - band  # type: ignore[operator]

        # skip a throttled opening (eco regulation at a lower setpoint), then take the
        # genuine full-power climb up to where the valve starts throttling again
        start = s
        while start <= j and throttling(start):
            start += 1
        if start > j:
            continue
        e_end = start
        while e_end < j and not throttling(e_end + 1):
            e_end += 1
        if e_end < j and throttling(e_end + 1):
            e_end += 1  # include the row that just reached the setpoint band
        # the rise starts at the lowest temperature of the kept climb
        s = min(range(start, e_end + 1), key=lambda i: ind[i])  # type: ignore[arg-type,return-value]
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
        acting = [eff[i - d_steps] for i in range(s, e_end)]
        duty = sum(e for e in acting if e is not None) / len(acting)
        if duty < MIN_DUTY:
            continue
        rates.append((rise / dur_h + b * (mean_t - mean_out) - c) / min(1.0, duty))
    if len(rates) < min_episodes:
        return None
    srt = sorted(rates)
    n = len(srt)
    med = srt[n // 2] if n % 2 else 0.5 * (srt[n // 2 - 1] + srt[n // 2])
    return EpisodeRate(med, n, [round(r, 4) for r in rates])
