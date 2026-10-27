"""Robust weighted least-squares fit of the RC model on the grid.

Regression per usable grid step ``k`` (Δt in hours)::

    (T[k+1] − T[k]) / Δt = a·u[k−d] − b·x[k] + c

with ``x[k] = (T[k] + T[k+1])/2 − Tout[k]`` (trapezoidal loss term: consistent with the
continuous model and, unlike the Euler form, unbiased by sensor noise because the
noise of ``T[k+1] − T[k]`` is uncorrelated with ``T[k] + T[k+1]``). Without any outdoor
data ``Tout ≡ 0`` and ``c`` absorbs ``b·mean(Tout)`` (the equilibrium is learnt).

- recency weights ``0.5 ** (age / half_life)``;
- ridge prior toward radiator defaults (a≈3 K/h, b≈0.15 1/h, c≈0) worth ~6 h of data;
- Huber IRLS (3 passes, fixed robust scale) down-weights outliers such as open windows;
- dead time grid search ``d ∈ {0, step, …, max_dead_time}`` choosing the lowest
  weighted Huber loss;
- projection ``a ≥ 0.05``, ``0.01 ≤ b ≤ 2`` (the remaining parameters are re-solved);
  a projected ``b`` means the data could not identify the loss rate (the model then
  falls back to the free-cooling estimate, see ``model.py``).

Stdlib only: the 3×3 normal equations are solved by Gaussian elimination.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from operator import mul
from typing import Any

from .base import ModelConfig
from .grid import Grid

HUBER_K = 1.345
A_MIN = 0.05
B_MIN, B_MAX = 0.01, 2.0
IRLS_PASSES = 3
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
    scale: float  # robust residual scale (K/h)
    rms_residual: float  # weighted RMS residual (K/h)
    downweighted_fraction: float
    criterion: dict[int, float] = field(default_factory=dict)
    flatness: float = 0.0  # (J_max − J_min)/J_min over the dead-time candidates
    projected: list[str] = field(default_factory=list)

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
    ind, out, exc = grid.indoor, grid.outdoor, grid.excluded
    max_dt = config.max_step_delta_k
    cand: list[tuple[int, float, float, float | None]] = []
    for k in range(len(ind) - 1):
        t0 = ind[k]
        t1 = ind[k + 1]
        if t0 is None or t1 is None or exc[k] or exc[k + 1]:
            continue
        if skip is not None and skip[k]:
            continue
        if abs(t1 - t0) > max_dt:
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
    w: list[float]
    h: list[float]
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
        self._cache: dict[int, tuple[list[float], list[float], list[float], list[float]]] = {}

    def rows(self, d: int) -> tuple[list[float], list[float], list[float], list[float]]:
        """(u, x, y, w) for dead time ``d`` steps (rows lacking u[k−d] are dropped)."""
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
            res = (u, ds.x, ds.y, ds.w)
        else:
            res = (u, [ds.x[i] for i in idx], [ds.y[i] for i in idx], [ds.w[i] for i in idx])
        self._cache[d] = res
        return res

    def solve(self, d: int, h: list[float] | None) -> _Solved:
        u, x, y, w = self.rows(d)
        ww = w if h is None else list(map(mul, w, h))
        m, v = _normal(u, x, y, ww)
        for i in range(3):
            m[i][i] += self.prior_diag[i]
            v[i] += self.prior_diag[i] * self.theta0[i]
        theta, projected = solve_projected(m, v, self.fixed)
        a, b, c = theta
        resid = [yi - a * ui + b * xi - c for yi, ui, xi in zip(y, u, x, strict=True)]
        return _Solved(theta, resid, w, h or [1.0] * len(y), projected)

    def irls(self, d: int, scale: float, passes: int = IRLS_PASSES) -> _Solved:
        sol = self.solve(d, None)
        kk = HUBER_K * scale
        for _ in range(passes):
            h = [1.0 if abs(r) <= kk else kk / abs(r) for r in sol.resid]
            sol = self.solve(d, h)
        sol.h = [1.0 if abs(r) <= kk else kk / abs(r) for r in sol.resid]
        return sol


def huber_criterion(resid: list[float], w: list[float], scale: float) -> float:
    """Weighted mean Huber loss of ``resid / scale``."""
    k = HUBER_K
    tot = 0.0
    sw = 0.0
    for r, wi in zip(resid, w, strict=True):
        z = abs(r) / scale
        tot += wi * (0.5 * z * z if z <= k else k * z - 0.5 * k * k)
        sw += wi
    return tot / sw if sw > 0 else math.inf


def fit_rc(
    ds: Dataset,
    effort: list[float | None],
    config: ModelConfig,
    prior: Prior | None = None,
    d_candidates: list[int] | None = None,
    fixed: dict[int, float] | None = None,
    scale: float | None = None,
) -> FitResult | None:
    """Fit (a, b, c, d). ``fixed`` maps parameter index (0=a, 1=b, 2=c) to a value."""
    if len(ds) < 12:
        return None
    if d_candidates is None:
        dmax = int(round(config.max_dead_time_minutes / config.step_minutes))
        d_candidates = list(range(0, max(dmax, 0) + 1))
    prob = _Problem(ds, effort, prior or Prior(), fixed)
    cands = [d for d in d_candidates if len(prob.rows(d)[0]) >= 12]
    if not cands:
        return None

    # 1) fixed robust scale from the best plain (prior-regularised) WLS solution
    if scale is None:
        best_sse = math.inf
        best_res: list[float] = []
        for d in cands:
            sol = prob.solve(d, None)
            sse = sum(wi * r * r for r, wi in zip(sol.resid, sol.w, strict=True)) / sum(sol.w)
            if sse < best_sse:
                best_sse, best_res = sse, sol.resid
        scale = 1.4826 * _median([abs(r) for r in best_res])
    scale = max(scale, 1e-3)

    # 2) Huber IRLS for each candidate, compared with the same scale
    crit: dict[int, float] = {}
    sols: dict[int, _Solved] = {}
    for d in cands:
        sol = prob.irls(d, scale)
        crit[d] = huber_criterion(sol.resid, sol.w, scale)
        sols[d] = sol
    best = min(cands, key=lambda d: (crit[d], d))
    sol = sols[best]
    jmin = crit[best]
    flat = (max(crit.values()) - jmin) / jmin if jmin > 0 else 0.0
    sw = sum(sol.w)
    rms = math.sqrt(sum(wi * r * r for r, wi in zip(sol.resid, sol.w, strict=True)) / sw)
    down = sum(1 for hi in sol.h if hi < 1.0) / len(sol.h)
    a, b, c = sol.theta
    return FitResult(
        a=a,
        b=b,
        c=c,
        d_steps=best,
        rows=len(sol.resid),
        scale=scale,
        rms_residual=rms,
        downweighted_fraction=down,
        criterion=crit,
        flatness=flat,
        projected=sorted(set(sol.projected)),
    )
