"""Weighted least-squares fit of the RC model on the grid.

Regression per usable grid step ``k`` (Δt in hours)::

    (T[k+1] − T[k]) / Δt = a·u[k−d] − b·x[k] + c

with ``x[k] = (T[k] + T[k+1])/2 − Tout[k]`` (trapezoidal loss term: consistent with the
continuous model and, unlike the Euler form, unbiased by sensor noise because the
noise of ``T[k+1] − T[k]`` is uncorrelated with ``T[k] + T[k+1]``). Without any outdoor
data ``Tout ≡ 0`` and ``c`` absorbs ``b·mean(Tout)`` (the equilibrium is learnt).

- the indoor column is linearly interpolated between the sensor's change points
  (:func:`interpolate_steps`): most room sensors report on change (~0.1 K) or hourly, so
  on the 5-min grid ~90 % of the steps are exactly zero and the rest carry the whole
  change of an hour; regressing on that staircase biases the dead time and, with a robust
  loss, collapses the fit onto "no change". The stored grid keeps the raw staircase;
- recency weights ``0.5 ** (age / half_life)``;
- ridge prior toward radiator defaults (a≈3 K/h, b≈0.15 1/h, c≈0) worth ~6 h of data;
- plain weighted least squares, with an explicit outlier mask instead of a robust loss:
  rows whose residual exceeds ``OUTLIER_MAD × MAD`` (and ``OUTLIER_FLOOR_K_PER_H``) for at
  least ``OUTLIER_MIN_RUN`` consecutive steps (an open window, not a single sensor step)
  are dropped. A Huber loss is not used: with a staircase sensor its scale is set by the
  zero steps and it down-weights exactly the rows that carry the information;
- dead time grid search ``d ∈ {0, step, …, max_dead_time}`` choosing the lowest
  weighted squared residual (same mask for every candidate);
- projection ``a ≥ 0.05``, ``0.002 ≤ b ≤ 2`` (the remaining parameters are re-solved);
  a projected ``b`` means the data could not identify the loss rate (the model then
  falls back to the free-cooling estimate, see ``model.py``). ``b ≥ 0.002`` allows time
  constants up to 500 h (well-insulated rooms heated by their neighbours).

Stdlib only: the 3×3 normal equations are solved by Gaussian elimination.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from operator import mul
from typing import Any

from .base import ModelConfig
from .grid import Grid

A_MIN = 0.05
B_MIN, B_MAX = 0.002, 2.0
OUTLIER_MAD = 6.0  # residual threshold in MADs
OUTLIER_FLOOR_K_PER_H = 1.0  # never mask residuals below this (K/h)
OUTLIER_MIN_RUN = 3  # consecutive outlying steps needed to mask them
MASK_PASSES = 2
MAX_INTERP_HOURS = 3.0  # longest plateau bridged by the change-point interpolation
PRIOR_HOURS = 6.0
NO_OUTDOOR_REF_C = 10.0  # prior for c' = b·mean(Tout) when no outdoor data exists


@dataclass
class Prior:
    a: float = 3.0
    b: float = 0.15
    c: float = 0.0
    hours: float = PRIOR_HOURS


@dataclass
class Dataset:
    """Usable regression rows (one per grid step with valid ``k`` and ``k+1``)."""

    k: list[int]
    y: list[float]
    x: list[float]
    w: list[float]
    use_outdoor: bool
    dt_h: float

    def __len__(self) -> int:
        return len(self.k)

    def subset(self, k_max: int) -> Dataset:
        """Rows with grid index < ``k_max`` (chronological training split)."""
        n = 0
        while n < len(self.k) and self.k[n] < k_max:
            n += 1
        return Dataset(self.k[:n], self.y[:n], self.x[:n], self.w[:n], self.use_outdoor, self.dt_h)


@dataclass
class FitResult:
    a: float
    b: float
    c: float
    d_steps: int
    rows: int
    scale: float  # robust residual scale 1.4826·MAD (K/h)
    rms_residual: float  # weighted RMS residual over the unmasked rows (K/h)
    downweighted_fraction: float  # share of rows masked as outliers
    criterion: dict[int, float] = field(default_factory=dict)
    flatness: float = 0.0  # (J_max − J_min)/J_min over the dead-time candidates
    projected: list[str] = field(default_factory=list)
    masked: frozenset[int] = frozenset()  # grid rows ``k`` masked as outliers

    def as_dict(self, step_minutes: float) -> dict[str, Any]:
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "dead_time_minutes": self.d_steps * step_minutes,
            "rows": self.rows,
            "scale_k_per_h": round(self.scale, 4),
            "rms_residual_k_per_h": round(self.rms_residual, 4),
            "downweighted_fraction": round(self.downweighted_fraction, 4),
            "dead_time_flatness": round(self.flatness, 5),
            "projected": list(self.projected),
        }


def outdoor_pair(o0: float | None, o1: float | None) -> float | None:
    if o0 is None:
        return o1
    if o1 is None:
        return o0
    return 0.5 * (o0 + o1)


def interpolate_steps(
    indoor: list[float | None],
    excluded: list[bool],
    max_rows: int,
    max_jump: float,
) -> list[float | None]:
    """Linear interpolation of the indoor staircase between consecutive change points.

    A change point is a row whose value differs from the previous row. Between two change
    points ``p < q`` the rows are replaced by the straight line from ``indoor[p]`` to
    ``indoor[q]`` when the stretch has no missing/excluded row, ``q − p ≤ max_rows`` and
    the jump is at most ``max_jump`` (a larger jump is a glitch: leaving it as a single
    step keeps it visible to the glitch guard). Longer plateaus stay flat.
    """
    out = list(indoor)
    p: int | None = None
    prev: float | None = None
    for k, v in enumerate(indoor):
        if v is None or excluded[k]:
            p = prev = None
            continue
        if prev is None or v != prev:
            if p is not None and k - p <= max_rows:
                v0 = indoor[p]
                assert v0 is not None
                if abs(v - v0) <= max_jump:
                    span = k - p
                    for i in range(p + 1, k):
                        out[i] = v0 + (v - v0) * (i - p) / span
            p = k
        prev = v
    return out


def build_dataset(
    grid: Grid,
    config: ModelConfig,
    now_ts: float,
    use_outdoor: bool | None = None,
    skip: list[bool] | None = None,
) -> Dataset:
    """Turn the grid into regression rows, dropping gaps, exclusions and glitches.

    ``use_outdoor=None`` decides automatically: outdoor mode when at least half of the
    usable rows (and ≥ 3 h) have an outdoor value. ``skip`` marks further rows not to
    learn from (thermostat regulation, see :func:`episodes.regulation_mask`).
    """
    dt_h = grid.step_minutes / 60.0
    raw, out, exc = grid.indoor, grid.outdoor, grid.excluded
    max_dt = config.max_step_delta_k
    max_rows = int(round(MAX_INTERP_HOURS * 60.0 / grid.step_minutes))
    ind = interpolate_steps(raw, exc, max_rows, max_dt)
    cand: list[tuple[int, float, float, float | None]] = []
    for k in range(len(ind) - 1):
        t0 = ind[k]
        t1 = ind[k + 1]
        r0, r1 = raw[k], raw[k + 1]
        if t0 is None or t1 is None or r0 is None or r1 is None or exc[k] or exc[k + 1]:
            continue
        if skip is not None and skip[k]:
            continue
        if abs(r1 - r0) > max_dt:  # glitch guard on the raw readings
            continue
        cand.append((k, t0, t1, outdoor_pair(out[k], out[k + 1])))
    if use_outdoor is None:
        n_out = sum(1 for r in cand if r[3] is not None)
        use_outdoor = n_out >= 0.5 * len(cand) and n_out * dt_h >= 3.0
    half_life_s = max(config.half_life_days, 0.01) * 86400.0
    start = grid.start_ts or 0.0
    step_s = grid.step_s
    ks: list[int] = []
    ys: list[float] = []
    xs: list[float] = []
    ws: list[float] = []
    for k, t0, t1, to in cand:
        if use_outdoor:
            if to is None:
                continue
            x = 0.5 * (t0 + t1) - to
        else:
            x = 0.5 * (t0 + t1)
        age = max(0.0, now_ts - (start + (k + 1) * step_s))
        ks.append(k)
        ys.append((t1 - t0) / dt_h)
        xs.append(x)
        ws.append(0.5 ** (age / half_life_s))
    return Dataset(ks, ys, xs, ws, bool(use_outdoor), dt_h)


# -- linear algebra -------------------------------------------------------------------
def solve_linear(m: list[list[float]], v: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting (small dense systems)."""
    n = len(v)
    a = [row[:] + [v[i]] for i, row in enumerate(m)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            return None
        a[col], a[piv] = a[piv], a[col]
        for r in range(col + 1, n):
            f = a[r][col] / a[col][col]
            if f:
                for cc in range(col, n + 1):
                    a[r][cc] -= f * a[col][cc]
    xs = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = a[r][n] - sum(a[r][cc] * xs[cc] for cc in range(r + 1, n))
        xs[r] = s / a[r][r]
    return xs


def _solve_fixed(m: list[list[float]], v: list[float], fixed: dict[int, float]) -> list[float]:
    free = [i for i in range(3) if i not in fixed]
    theta = [0.0, 0.0, 0.0]
    for i, val in fixed.items():
        theta[i] = val
    if free:
        mm = [[m[i][j] for j in free] for i in free]
        vv = [v[i] - sum(m[i][j] * fixed[j] for j in fixed) for i in free]
        sol = solve_linear(mm, vv)
        if sol is None:
            sol = [0.0] * len(free)
        for i, val in zip(free, sol, strict=True):
            theta[i] = val
    return theta


def solve_projected(
    m: list[list[float]], v: list[float], fixed: dict[int, float] | None = None
) -> tuple[list[float], list[str]]:
    """Solve the normal equations, then project a ≥ A_MIN and B_MIN ≤ b ≤ B_MAX."""
    fx = dict(fixed or {})
    projected: list[str] = []
    theta = _solve_fixed(m, v, fx)
    for _ in range(3):
        changed = False
        if 0 not in fx and theta[0] < A_MIN:
            fx[0] = A_MIN
            projected.append("a")
            changed = True
        if 1 not in fx and not B_MIN <= theta[1] <= B_MAX:
            fx[1] = min(B_MAX, max(B_MIN, theta[1]))
            projected.append("b")
            changed = True
        if not changed:
            break
        theta = _solve_fixed(m, v, fx)
    return theta, projected


def _normal(
    u: list[float], x: list[float], y: list[float], w: list[float]
) -> tuple[list[list[float]], list[float]]:
    wu = list(map(mul, w, u))
    wx = list(map(mul, w, x))
    suu = sum(map(mul, wu, u))
    sux = sum(map(mul, wu, x))
    su = sum(wu)
    sxx = sum(map(mul, wx, x))
    sx = sum(wx)
    s1 = sum(w)
    suy = sum(map(mul, wu, y))
    sxy = sum(map(mul, wx, y))
    sy = sum(map(mul, w, y))
    m = [[suu, -sux, su], [-sux, sxx, -sx], [su, -sx, s1]]
    return m, [suy, -sxy, sy]


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if not n:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def _wvar(values: list[float], w: list[float]) -> float:
    sw = sum(w)
    if sw <= 0:
        return 0.0
    mean = sum(map(mul, w, values)) / sw
    return sum(wi * (vi - mean) ** 2 for vi, wi in zip(values, w, strict=True)) / sw


@dataclass
class _Solved:
    theta: list[float]
    resid: list[float]
    k: list[int]  # grid row of every residual
    w: list[float]
    keep: list[bool]  # False for rows masked as outliers
    projected: list[str]


class _Problem:
    """Shared state of one fit (dataset, effort series, prior)."""

    def __init__(
        self,
        ds: Dataset,
        effort: list[float | None],
        prior: Prior,
        fixed: dict[int, float] | None,
    ) -> None:
        self.ds = ds
        self.effort = effort
        self.fixed = fixed or {}
        n = len(ds)
        u0 = [e if (e := effort[k]) is not None else 0.0 for k in ds.k]
        mean_w = (sum(ds.w) / n) if n else 1.0
        n0 = prior.hours / ds.dt_h * mean_w
        var_u = max(_wvar(u0, ds.w), 0.05)
        var_x = max(_wvar(ds.x, ds.w), 1.0)
        self.prior_diag = [n0 * var_u, n0 * var_x, n0]
        c0 = prior.c if ds.use_outdoor else prior.c + prior.b * NO_OUTDOOR_REF_C
        self.theta0 = [prior.a, prior.b, c0]
        self._cache: dict[
            int, tuple[list[float], list[float], list[float], list[float], list[int]]
        ] = {}

    def rows(self, d: int) -> tuple[list[float], list[float], list[float], list[float], list[int]]:
        """(u, x, y, w, k) for dead time ``d`` steps (rows lacking u[k−d] are dropped)."""
        if d in self._cache:
            return self._cache[d]
        ds, eff = self.ds, self.effort
        u: list[float] = []
        idx: list[int] = []
        for i, k in enumerate(ds.k):
            j = k - d
            if j >= 0 and (e := eff[j]) is not None:
                u.append(e)
                idx.append(i)
        if len(idx) == len(ds.k):
            res = (u, ds.x, ds.y, ds.w, ds.k)
        else:
            res = (
                u,
                [ds.x[i] for i in idx],
                [ds.y[i] for i in idx],
                [ds.w[i] for i in idx],
                [ds.k[i] for i in idx],
            )
        self._cache[d] = res
        return res

    def solve(self, d: int, masked: frozenset[int] = frozenset()) -> _Solved:
        u, x, y, w, ks = self.rows(d)
        keep = [k not in masked for k in ks]
        ww = [wi if kp else 0.0 for wi, kp in zip(w, keep, strict=True)] if masked else w
        m, v = _normal(u, x, y, ww)
        for i in range(3):
            m[i][i] += self.prior_diag[i]
            v[i] += self.prior_diag[i] * self.theta0[i]
        theta, projected = solve_projected(m, v, self.fixed)
        a, b, c = theta
        resid = [yi - a * ui + b * xi - c for yi, ui, xi in zip(y, u, x, strict=True)]
        return _Solved(theta, resid, ks, w, keep, projected)

    def mask_residuals(self, theta: list[float], d: int) -> list[float]:
        """Residual of every dataset row for outlier screening, whatever its effort.

        Rows whose effort ``u[k−d]`` is unknown (a throttling valve, no effort data) are
        not in the regression for this ``d`` but may still be an open window: their
        residual is the distance to the range explained by any effort in [0, 1].
        """
        a, b, c = theta
        ds, eff = self.ds, self.effort
        out: list[float] = []
        for k, x, y in zip(ds.k, ds.x, ds.y, strict=True):
            base = y + b * x - c
            j = k - d
            e = eff[j] if j >= 0 else None
            if e is not None:
                out.append(base - a * e)
            elif base < 0.0:
                out.append(base)
            else:
                out.append(max(0.0, base - max(a, 0.0)))
        return out


def _mad(values: list[float]) -> float:
    med = _median(values)
    return _median([abs(v - med) for v in values])


def outlier_rows(
    ks: list[int],
    resid: list[float],
    threshold: float,
    min_run: int = OUTLIER_MIN_RUN,
) -> frozenset[int]:
    """Grid rows of sustained outlying residuals (``|resid| > threshold``).

    A run of consecutive outlying rows is masked when it spans at least ``min_run`` grid
    rows. Rows missing from the regression right next to the run (no effort, thermostat
    regulation, a glitch) are unknown, not normal: up to ``min_run − 1`` of them on each
    side count towards the span. An open window (the room loses 1-3 K in 10-30 min) is
    therefore masked even when the thermostat's reaction removed some of its rows, while
    an isolated large residual between normal rows (a staircase sensor reporting the
    change of hours in one step, one noisy reading) is kept.
    """
    present = set(ks)
    out: set[int] = set()

    def flush(run: list[int]) -> None:
        if not run:
            return
        span = len(run)
        for direction, edge in ((-1, run[0]), (1, run[-1])):
            for i in range(1, min_run):
                if edge + direction * i in present:
                    break
                span += 1
        if span >= min_run:
            out.update(run)

    run: list[int] = []
    prev: int | None = None
    for k, r in zip(ks, resid, strict=True):
        big = abs(r) > threshold
        if big and run and prev == k - 1:
            run.append(k)
        else:
            flush(run)
            run = [k] if big else []
        prev = k
    flush(run)
    return frozenset(out)


def _mse(sol: _Solved) -> float:
    sw = se = 0.0
    for r, wi, kp in zip(sol.resid, sol.w, sol.keep, strict=True):
        if kp:
            sw += wi
            se += wi * r * r
    return se / sw if sw > 0 else math.inf


def fit_rc(
    ds: Dataset,
    effort: list[float | None],
    config: ModelConfig,
    prior: Prior | None = None,
    d_candidates: list[int] | None = None,
    fixed: dict[int, float] | None = None,
    mask: frozenset[int] | None = None,
) -> FitResult | None:
    """Fit (a, b, c, d). ``fixed`` maps parameter index (0=a, 1=b, 2=c) to a value.

    ``mask`` reuses the outlier rows of a previous fit (refits with fixed parameters and
    the holdout fit must learn from the same rows); by default it is computed here.
    """
    if len(ds) < 12:
        return None
    if d_candidates is None:
        dmax = int(round(config.max_dead_time_minutes / config.step_minutes))
        d_candidates = list(range(0, max(dmax, 0) + 1))
    prob = _Problem(ds, effort, prior or Prior(), fixed)
    cands = [d for d in d_candidates if len(prob.rows(d)[0]) >= 12]
    if not cands:
        return None

    # 1) outlier mask from the best plain (prior-regularised) WLS solution
    plain = {d: prob.solve(d) for d in cands}
    d0 = min(cands, key=lambda d: (_mse(plain[d]), d))
    sol = plain[d0]
    scale = 1.4826 * _mad(sol.resid)
    if mask is None:
        masked: frozenset[int] = frozenset()
        for _ in range(MASK_PASSES):
            kept = [r for r, kp in zip(sol.resid, sol.keep, strict=True) if kp]
            scale = 1.4826 * _mad(kept)
            thr = max(OUTLIER_MAD * scale / 1.4826, OUTLIER_FLOOR_K_PER_H)
            new = outlier_rows(ds.k, prob.mask_residuals(sol.theta, d0), thr)
            if new == masked:
                break
            masked = new
            sol = prob.solve(d0, masked)
    else:
        masked = mask

    # 2) dead time: lowest weighted squared residual, same mask for every candidate
    crit: dict[int, float] = {}
    sols: dict[int, _Solved] = {}
    for d in cands:
        sols[d] = prob.solve(d, masked) if masked else plain[d]
        crit[d] = _mse(sols[d])
    best = min(cands, key=lambda d: (crit[d], d))
    sol = sols[best]
    jmin = crit[best]
    flat = (max(crit.values()) - jmin) / jmin if jmin > 0 else 0.0
    rms = math.sqrt(jmin) if math.isfinite(jmin) else 0.0
    used = frozenset(k for k, kp in zip(sol.k, sol.keep, strict=True) if not kp)
    a, b, c = sol.theta
    return FitResult(
        a=a,
        b=b,
        c=c,
        d_steps=best,
        rows=len(sol.resid),
        scale=max(scale, 1e-3),
        rms_residual=rms,
        downweighted_fraction=len(used) / len(sol.resid),
        criterion=crit,
        flatness=flat,
        projected=sorted(set(sol.projected)),
        masked=masked,
    )
