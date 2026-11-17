"""Model quality and maturity.

Evaluation (all on a chronological holdout = the last 20 % of the usable rows, with
parameters fitted on the first 80 % only):

- **forecast skill**: from every holdout row (every 30 min) the model is simulated
  open-loop for 60 min and for 4 h with the *learning* effort (the effort the fit learns
  from: thermostat regulation and throttling valve rows are unknown, so windows that
  contain them are skipped, consistently with the fit) and the observed outdoor
  temperature, starting from the observed indoor temperature; the RMSE of the prediction
  is compared with the persistence baseline (temperature unchanged).
- **heat-up episodes**: effort ≥ 0.5 after at least ``d`` idle steps (pauses of up to
  ``MERGE_PAUSE_ROWS`` steps, burner cycling, belong to the same episode), ending with a
  rise ≥ 0.5 K. The actual duration (from the edge until the temperature first reaches
  the temperature at the end of the episode) is compared with the closed-form preheat
  prediction (mean outdoor over the episode) → MAE in minutes.

Maturity (0..1) = data_factor × skill_factor:

- ``data_factor = min(1, heating_hours / 20) · min(1, cooling_hours / 40)`` where the
  hours count usable steps with effort ≥ 0.5 (heating) and effort < 0.1 (cooling);
- ``skill_factor = clamp((1 − rmse / baseline_rmse) / 0.5, 0, 1)``: a model that halves
  the persistence error is considered fully skilled; 0 without an evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .fit import outdoor_pair
from .grid import Grid
from .physics import preheat_minutes

HORIZON_MIN = 60.0
LONG_HORIZON_MIN = 240.0
MERGE_PAUSE_ROWS = 3  # 15 min on the 5-min grid
STRIDE_MIN = 30.0
HOLDOUT_FRACTION = 0.2
EPISODE_EFFORT = 0.5
EPISODE_MIN_RISE = 0.5
HEATING_HOURS_FULL = 20.0
COOLING_HOURS_FULL = 40.0


@dataclass
class Params:
    a: float
    b: float
    c: float
    d_steps: int
    use_outdoor: bool


@dataclass
class Evaluation:
    rmse: float | None
    baseline_rmse: float | None
    windows: int
    duration_mae: float | None
    episodes: int
    episode_errors: list[float]
    rmse_4h: float | None = None
    baseline_rmse_4h: float | None = None


def bad_rows(grid: Grid, max_step_delta: float) -> list[bool]:
    """Rows unusable for evaluation: missing indoor, excluded or adjacent to a glitch."""
    ind = grid.indoor
    n = len(ind)
    bad = [v is None or ex for v, ex in zip(ind, grid.excluded, strict=True)]
    for k in range(n - 1):
        a, b = ind[k], ind[k + 1]
        if a is not None and b is not None and abs(b - a) > max_step_delta:
            bad[k] = bad[k + 1] = True
    return bad


def step_temperature(
    t: float, u: float, tout: float, a: float, b: float, c: float, dt_h: float
) -> float:
    """One trapezoidal step of the RC model (same discretisation as the fit)."""
    hb = 0.5 * b * dt_h
    return (t * (1.0 - hb) + dt_h * (a * u + b * tout + c)) / (1.0 + hb)


def forecast_rmse(
    grid: Grid,
    p: Params,
    bad: list[bool],
    k_start: int,
    k_end: int | None = None,
    horizon_min: float = HORIZON_MIN,
    effort: list[float | None] | None = None,
) -> tuple[float | None, float | None, int]:
    """Open-loop RMSE vs persistence ``horizon_min`` ahead, windows starting in [k_start, k_end).

    ``effort`` replaces the grid's effort column (e.g. the learning effort, None where
    the reported effort is not the heat delivered: such windows are skipped).
    """
    step = grid.step_minutes
    h = int(round(horizon_min / step))
    stride = max(1, int(round(STRIDE_MIN / step)))
    dt_h = step / 60.0
    ind, out = grid.indoor, grid.outdoor
    eff = grid.effort if effort is None else effort
    n = len(ind)
    k_end = n if k_end is None else k_end
    se = se0 = 0.0
    cnt = 0
    k = max(k_start, p.d_steps)
    while k < k_end and k + h < n:
        ok = not any(bad[k : k + h + 1])
        if ok:
            t = ind[k]
            for j in range(k, k + h):
                u = eff[j - p.d_steps]
                if u is None:
                    ok = False
                    break
                if p.use_outdoor:
                    to = outdoor_pair(out[j], out[j + 1])
                    if to is None:
                        ok = False
                        break
                else:
                    to = 0.0
                t = step_temperature(t, u, to, p.a, p.b, p.c, dt_h)  # type: ignore[arg-type]
            if ok:
                obs = ind[k + h]
                se += (t - obs) ** 2  # type: ignore[operator]
                se0 += (ind[k] - obs) ** 2  # type: ignore[operator]
                cnt += 1
        k += stride if ok else 1
    if not cnt:
        return None, None, 0
    return math.sqrt(se / cnt), math.sqrt(se0 / cnt), cnt


def heat_up_episodes(
    grid: Grid, bad: list[bool], k_start: int, pre_off_steps: int
) -> list[tuple[int, int, int]]:
    """(k_edge, k_end, k_reach) for qualifying episodes starting at or after ``k_start``.

    ``k_end`` is the first row after the episode (temperature at heating stop);
    ``k_reach`` the first row at which that temperature was reached. Pauses of up to
    ``MERGE_PAUSE_ROWS`` known-idle rows followed by heating again continue the episode.
    """
    eff, ind = grid.effort, grid.indoor
    n = len(eff)
    eps: list[tuple[int, int, int]] = []
    pre = max(1, pre_off_steps)
    k = max(k_start, pre)
    while k < n:
        e = eff[k]
        if e is None or e < EPISODE_EFFORT or bad[k]:
            k += 1
            continue
        prev = eff[k - pre : k]
        if any(p is None or p >= EPISODE_EFFORT for p in prev):
            k += 1
            continue
        j = k
        broken = False
        while True:
            while j < n and (ej := eff[j]) is not None and ej >= EPISODE_EFFORT:
                if bad[j]:
                    broken = True
                    break
                j += 1
            if broken:
                break
            # a short idle pause followed by heating again: same episode
            p_end = j
            while p_end < n and p_end - j < MERGE_PAUSE_ROWS:
                ep = eff[p_end]
                if ep is None or ep >= EPISODE_EFFORT or bad[p_end]:
                    break
                p_end += 1
            if (
                p_end > j
                and p_end < n
                and (ep := eff[p_end]) is not None
                and ep >= EPISODE_EFFORT
                and not bad[p_end]
            ):
                j = p_end
                continue
            break
        if broken or j >= n or bad[j]:
            k = j + 1
            continue
        t0, t_end = ind[k], ind[j]
        if t0 is not None and t_end is not None and t_end - t0 >= EPISODE_MIN_RISE:
            reach = next(i for i in range(k, j + 1) if ind[i] >= t_end - 1e-9)  # type: ignore[operator]
            eps.append((k, j, reach))
        k = j + 1
    return eps


def episode_errors(grid: Grid, p: Params, episodes: list[tuple[int, int, int]]) -> list[float]:
    """Predicted − actual minutes for each episode."""
    step = grid.step_minutes
    errs: list[float] = []
    for k0, k1, kr in episodes:
        if p.use_outdoor:
            vals = [v for v in grid.outdoor[k0 : k1 + 1] if v is not None]
            if not vals:
                continue
            tout = sum(vals) / len(vals)
        else:
            tout = 0.0
        t0, t_end = grid.indoor[k0], grid.indoor[k1]
        pred, _ = preheat_minutes(
            t0,
            t_end,
            tout,
            p.a,
            p.b,
            p.c,
            p.d_steps * step,  # type: ignore[arg-type]
        )
        if pred is None:
            continue
        errs.append(pred - (kr - k0) * step)
    return errs


def evaluate(
    grid: Grid,
    p: Params,
    bad: list[bool],
    k_holdout: int,
    effort: list[float | None] | None = None,
) -> Evaluation:
    """Holdout evaluation from row ``k_holdout`` (``effort``: see :func:`forecast_rmse`)."""
    rmse, base, windows = forecast_rmse(grid, p, bad, k_holdout, effort=effort)
    rmse4, base4, _ = forecast_rmse(
        grid, p, bad, k_holdout, horizon_min=LONG_HORIZON_MIN, effort=effort
    )
    eps = heat_up_episodes(grid, bad, k_holdout, p.d_steps + 1)
    errs = episode_errors(grid, p, eps)
    mae = sum(abs(e) for e in errs) / len(errs) if errs else None
    return Evaluation(
        rmse, base, windows, mae, len(errs), [round(e, 1) for e in errs], rmse4, base4
    )


def experience_hours(grid: Grid, bad: list[bool]) -> tuple[float, float]:
    """(heating_hours, cooling_hours) over usable rows."""
    dt_h = grid.step_minutes / 60.0
    heat = cool = 0
    for e, bd in zip(grid.effort, bad, strict=True):
        if bd or e is None:
            continue
        if e >= EPISODE_EFFORT:
            heat += 1
        elif e < 0.1:
            cool += 1
    return heat * dt_h, cool * dt_h


def skill_factor(rmse: float | None, baseline: float | None) -> float:
    if rmse is None or baseline is None or baseline <= 0:
        return 0.0
    return min(1.0, max(0.0, (1.0 - rmse / baseline) / 0.5))


def data_factor(heating_hours: float, cooling_hours: float) -> float:
    return min(1.0, heating_hours / HEATING_HOURS_FULL) * min(
        1.0, cooling_hours / COOLING_HOURS_FULL
    )


def maturity(
    heating_hours: float, cooling_hours: float, rmse: float | None, baseline: float | None
) -> float:
    return data_factor(heating_hours, cooling_hours) * skill_factor(rmse, baseline)
