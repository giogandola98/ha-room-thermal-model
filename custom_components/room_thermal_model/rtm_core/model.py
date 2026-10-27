"""RC (single thermal capacity + dead time) implementation of :class:`ThermalModel`."""

from __future__ import annotations

import bisect
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from datetime import datetime
from typing import Any

from .base import (
    ForecastPoint,
    ModelConfig,
    PreheatEstimate,
    QualityReport,
    Sample,
    ThermalModel,
    ThermalParameters,
)
from .cooling import cooling_tau
from .deadtime import tangent_dead_time
from .episodes import episode_heating_rate, regulation_mask
from .fit import Dataset, FitResult, build_dataset, fit_rc
from .forecast import mean_over, outdoor_curve, solve_duration
from .grid import Grid, ceil_ts, dt_of, floor_ts, resample, ts_of
from .physics import coast_temperature, equilibrium, preheat_minutes
from .quality import (
    HOLDOUT_FRACTION,
    Params,
    bad_rows,
    evaluate,
    experience_hours,
    maturity,
)

STATE_VERSION = 1
MIN_FIT_HOURS = 3.0  # usable data needed before the first fit
MIN_QUALITY_HOURS = 12.0  # usable data needed for a holdout evaluation
REFIT_SECONDS = 3600.0
WEAK_EXCITATION_HOURS = 3.0
OUTDOOR_REF_HOURS = 24.0
EPISODE_DISAGREEMENT = 0.3  # relative gap between fitted and episode ``a`` that triggers it
PLAUSIBLE_MAX_RISE_K = 50.0  # full-effort equilibrium above outdoor (radiators: ~20-35 K)
NO_OUTDOOR_MEAN_C = 15.0  # no outdoor data: c absorbs b·mean(Tout); allow for it


def _r(v: float | None, nd: int = 4) -> float | None:
    return None if v is None else round(v, nd)


class RCThermalModel(ThermalModel):
    """Lumped RC room model ``T[k+1] − T[k] = Δt·(a·u[k−d] − b·(T[k] − Tout[k]) + c)``.

    Keeps a bounded 5-min grid of the last ``history_days`` (plus a short tail of raw
    samples not yet gridded), fits (a, b, c, d) with robust weighted least squares and
    answers preheat / coast questions in closed form.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.reset()

    # -- data ingestion -----------------------------------------------------------------
    @property
    def _step_s(self) -> float:
        return self.config.step_minutes * 60.0

    @property
    def _lookback_s(self) -> float:
        cfg = self.config
        return max(cfg.max_gap_minutes, cfg.outdoor_max_gap_minutes) * 60.0

    @property
    def _history_s(self) -> float:
        return self.config.history_days * 86400.0

    def reset(self) -> None:
        self._grid = Grid(self.config.step_minutes)
        self._pending: list[tuple[float, Sample]] = []
        self._last_sample_ts: float | None = None
        self._params: dict[str, Any] | None = None
        self._outdoor_ref: float | None = None
        self._quality: dict[str, Any] | None = None
        self._last_fit_ts: float | None = None
        self._data_end_at_fit: float | None = None
        self._summary: dict[str, Any] = {}

    def load_history(self, samples: Iterable[Sample], replace: bool = False) -> None:
        rows = sorted(samples, key=lambda s: s.ts)
        if replace:
            self._grid = Grid(self.config.step_minutes)
            self._pending = []
            self._last_sample_ts = None
        if not rows:
            return
        cfg = self.config
        new = resample(rows, cfg.step_minutes, cfg.max_gap_minutes, cfg.outdoor_max_gap_minutes)
        if len(new):
            self._grid = new if not len(self._grid) else self._grid.merged(new)
        latest = ts_of(rows[-1].ts)
        if self._last_sample_ts is None or latest > self._last_sample_ts:
            self._last_sample_ts = latest
        end = self._grid.end_ts
        horizon = (end if end is not None else latest) - self._lookback_s
        merged = self._pending + [(ts_of(s.ts), s) for s in rows if ts_of(s.ts) >= horizon]
        merged.sort(key=lambda p: p[0])
        self._pending = [p for p in merged if p[0] >= horizon]
        self._advance()
        self._trim()

    def add_sample(self, sample: Sample) -> None:
        t = ts_of(sample.ts)
        end = self._grid.end_ts
        if end is not None and t < end - self._lookback_s:
            return  # too old to matter: rows up to ``end`` are final
        if not self._pending or t >= self._pending[-1][0]:
            self._pending.append((t, sample))
        else:
            bisect.insort(self._pending, (t, sample), key=lambda p: p[0])
        if self._last_sample_ts is None or t > self._last_sample_ts:
            self._last_sample_ts = t
        self._advance()

    def _advance(self) -> None:
        """Build grid rows fully covered by the pending samples."""
        if not self._pending:
            return
        step = self._step_s
        latest = self._pending[-1][0]
        if self._grid.start_ts is None or ((self._grid.end_ts or 0.0) < latest - self._history_s):
            first = max(self._pending[0][0], latest - self._history_s)
            self._grid = Grid(self.config.step_minutes, ceil_ts(first, step))
        end = self._grid.end_ts
        assert end is not None
        new_end = floor_ts(latest, step)
        if new_end - end < step - 1e-6:
            return
        cfg = self.config
        chunk = resample(
            (s for _, s in self._pending),
            cfg.step_minutes,
            cfg.max_gap_minutes,
            cfg.outdoor_max_gap_minutes,
            start_ts=end,
            end_ts=new_end,
        )
        self._grid.append_rows(chunk)
        horizon = new_end - self._lookback_s
        if self._pending[0][0] < horizon:
            idx = bisect.bisect_left(self._pending, horizon, key=lambda p: p[0])
            del self._pending[:idx]
        if len(self._grid) * step > self._history_s + 86400.0:
            self._trim()

    def _trim(self) -> None:
        if self._last_sample_ts is not None:
            self._grid.trim_before(self._last_sample_ts - self._history_s)

    # -- fitting ------------------------------------------------------------------------
    def fit_due(self, now: datetime) -> bool:
        if self._last_sample_ts is None or self._grid.end_ts is None:
            return False
        if self._last_fit_ts is None:
            return self._usable_hours() >= MIN_FIT_HOURS
        if ts_of(now) - self._last_fit_ts < REFIT_SECONDS:
            return False
        return self._data_end_at_fit is None or self._grid.end_ts > self._data_end_at_fit

    def _usable_hours(self) -> float:
        return self._grid.valid_indoor_count() * self.config.step_minutes / 60.0

    def fit(self, now: datetime) -> dict[str, Any]:
        t_start = time.perf_counter()
        cfg = self.config
        now_ts = ts_of(now)
        self._last_fit_ts = now_ts
        # fit runs in an executor while samples keep arriving: work on a snapshot
        grid = self._grid.snapshot()
        self._data_end_at_fit = grid.end_ts
        regulation = regulation_mask(grid)
        ds = build_dataset(grid, cfg, now_ts, skip=regulation)
        if len(ds) * ds.dt_h < MIN_FIT_HOURS:
            self._summary = {"status": "insufficient_data", "rows": len(ds)}
            return dict(self._summary)
        res = fit_rc(ds, grid.effort, cfg)
        if res is None:
            self._summary = {"status": "insufficient_data", "rows": len(ds)}
            return dict(self._summary)

        bad = bad_rows(grid, cfg.max_step_delta_k)
        heat_h, cool_h = experience_hours(grid, bad)
        step = cfg.step_minutes
        dmax = int(round(cfg.max_dead_time_minutes / step))

        # cross-checks (Ecronika/ha-preheat): fallbacks when excitation is weak
        cool = cooling_tau(grid, skip_minutes=max(30.0, (res.d_steps + 1) * step))
        if not ds.use_outdoor:
            cool = None  # needs an outdoor reference
        tangent = tangent_dead_time(grid)
        fixed: dict[int, float] = {}
        fallbacks: list[str] = []
        weak = heat_h < WEAK_EXCITATION_HOURS or cool_h < WEAK_EXCITATION_HOURS
        d_steps = res.d_steps
        if weak and tangent is not None:
            d_steps = min(dmax, max(0, round(tangent.minutes / step)))
            fallbacks.append("dead_time_from_tangent")
        b_unidentified = "b" in res.projected
        if (weak or b_unidentified) and cool is not None:
            fixed[1] = 1.0 / cool.tau_hours
            fallbacks.append("loss_rate_from_cooling")
        if fallbacks:
            res = self._refit(ds, grid, res, d_steps, fixed)

        # full-effort heating rate measured on real heat-up episodes
        episode = episode_heating_rate(grid, res.b, res.c, res.d_steps, ds.use_outdoor)
        if episode is not None and episode.a_k_per_h > 0:
            gap = abs(res.a - episode.a_k_per_h) / episode.a_k_per_h
            if gap > EPISODE_DISAGREEMENT:
                fixed[0] = episode.a_k_per_h
                fixed[1] = res.b
                fallbacks.append("heating_rate_from_episodes")
                res = self._refit(ds, grid, res, res.d_steps, fixed)

        reliable, unreliable_reason = self._plausible(res, ds.use_outdoor, fixed)

        # holdout evaluation with parameters fitted on the first 80 %
        evaluation = None
        n = len(ds)
        if n * ds.dt_h >= MIN_QUALITY_HOURS:
            k_hold = ds.k[int((1.0 - HOLDOUT_FRACTION) * n)]
            train = ds.subset(k_hold)
            r_tr = fit_rc(
                train, grid.effort, cfg, d_candidates=[res.d_steps], fixed=fixed, scale=res.scale
            )
            if r_tr is not None:
                p_tr = Params(r_tr.a, r_tr.b, r_tr.c, r_tr.d_steps, ds.use_outdoor)
                evaluation = evaluate(grid, p_tr, bad, k_hold)

        self._params = {
            "a": res.a,
            "b": res.b,
            "c": res.c,
            "d_minutes": res.d_steps * step,
            "use_outdoor": ds.use_outdoor,
            "reliable": reliable,
        }
        self._outdoor_ref = self._recent_outdoor()
        rmse = evaluation.rmse if evaluation else None
        base = evaluation.baseline_rmse if evaluation else None
        mat = maturity(heat_h, cool_h, rmse, base) if reliable else 0.0
        self._quality = {
            "maturity": mat,
            "samples": grid.valid_indoor_count(),
            "heating_hours": heat_h,
            "cooling_hours": cool_h,
            "rmse_c": rmse,
            "baseline_rmse_c": base,
            "duration_mae_minutes": evaluation.duration_mae if evaluation else None,
            "episodes": evaluation.episodes if evaluation else 0,
        }
        cross: dict[str, Any] = {}
        if cool is not None:
            cross["cooling_tau_h"] = round(cool.tau_hours, 3)
            cross["cooling_segments"] = cool.segments
            cross["tau_ratio"] = round(cool.tau_hours * res.b, 3)
        if tangent is not None:
            cross["tangent_dead_time_min"] = round(tangent.minutes, 1)
            cross["tangent_edges"] = tangent.edges
        self._summary = {
            "status": "ok",
            "mode": "outdoor" if ds.use_outdoor else "no_outdoor",
            **self._fit_dict(res),
            "time_constant_h": round(1.0 / res.b, 3),
            "fallbacks": fallbacks,
            "reliable": reliable,
            "unreliable_reason": unreliable_reason,
            "regulation_rows_skipped": sum(regulation),
            "episode_heating_rate": (
                {
                    "a_k_per_h": round(episode.a_k_per_h, 4),
                    "episodes": episode.episodes,
                    "rates": episode.rates,
                }
                if episode is not None
                else None
            ),
            "cross_check": cross,
            "quality": {k: _r(v) if isinstance(v, float) else v for k, v in self._quality.items()},
            "evaluation_windows": evaluation.windows if evaluation else 0,
            "episode_errors_min": evaluation.episode_errors if evaluation else [],
            "fit_seconds": round(time.perf_counter() - t_start, 3),
        }
        return dict(self._summary)

    def _refit(
        self, ds: Dataset, grid: Grid, res: FitResult, d_steps: int, fixed: dict[int, float]
    ) -> FitResult:
        """Refit with some parameters fixed, keeping the dead-time diagnostics."""
        res2 = fit_rc(
            ds, grid.effort, self.config, d_candidates=[d_steps], fixed=fixed, scale=res.scale
        )
        if res2 is None:
            return res
        res2.criterion, res2.flatness = res.criterion, res.flatness
        return res2

    def _plausible(
        self, res: FitResult, use_outdoor: bool, fixed: dict[int, float]
    ) -> tuple[bool, str | None]:
        """Physical sanity of the fitted parameters (an unreliable model gives no preheat)."""
        if "b" in res.projected and 1 not in fixed:
            return False, "loss_rate_unidentified"
        # (a + c)/b: how far above the outdoor (or the learnt mean) full heating settles
        if equilibrium(0.0, res.a, res.b, res.c) > PLAUSIBLE_MAX_RISE_K + (
            0.0 if use_outdoor else NO_OUTDOOR_MEAN_C
        ):
            return False, "max_reachable_implausible"
        return True, None

    def _fit_dict(self, res: FitResult) -> dict[str, Any]:
        d = res.as_dict(self.config.step_minutes)
        d["criterion"] = {
            str(int(k * self.config.step_minutes)): round(v, 5) for k, v in res.criterion.items()
        }
        return d

    def _recent_outdoor(self) -> float | None:
        n = int(OUTDOOR_REF_HOURS * 60 / self.config.step_minutes)
        vals = [v for v in self._grid.outdoor[-n:] if v is not None]
        return sum(vals) / len(vals) if vals else None

    # -- queries ------------------------------------------------------------------------
    def parameters(self) -> ThermalParameters | None:
        p = self._params
        if p is None:
            return None
        return ThermalParameters(p["a"], p["b"], p["c"], p["d_minutes"])

    def last_fit_summary(self) -> dict[str, Any]:
        """Summary of the last fit (diagnostics)."""
        return dict(self._summary)

    def _outdoor_function(
        self, now: datetime, outdoor_c: float | None, forecast: Sequence[ForecastPoint]
    ) -> tuple[Any, str]:
        """(minutes → mean outdoor, source) or (None, reason)."""
        p = self._params
        assert p is not None
        if not p["use_outdoor"]:
            return (lambda _m: 0.0), "not_used"
        pts = outdoor_curve(forecast, now, outdoor_c)
        if forecast and pts:
            source = "forecast"
        elif pts:
            source = "current"
        elif self._outdoor_ref is not None:
            pts = [(ts_of(now), self._outdoor_ref)]
            source = "model_reference"
        else:
            return None, "no_outdoor"
        t0 = ts_of(now)
        return (lambda m: mean_over(pts, t0, t0 + m * 60.0)), source

    def preheat(
        self,
        now: datetime,
        start_c: float,
        target_c: float,
        outdoor_c: float | None,
        forecast: Sequence[ForecastPoint] = (),
    ) -> PreheatEstimate:
        p = self._params
        if p is None:
            return PreheatEstimate(
                None, False, target_c, start_c, None, None, {"status": "not_fitted"}
            )
        a, b, c, d = p["a"], p["b"], p["c"], p["d_minutes"]
        outdoor_for, source = self._outdoor_function(now, outdoor_c, forecast)
        reason: dict[str, Any] = {
            "model": "rc",
            "mode": "outdoor" if p["use_outdoor"] else "no_outdoor",
            "a_k_per_h": a,
            "b_per_h": b,
            "c_k_per_h": c,
            "dead_time_minutes": d,
            "time_constant_h": 1.0 / b,
            "outdoor_source": source,
        }
        if not p.get("reliable", True):
            reason["status"] = "unreliable"
            reason["unreliable_reason"] = self._summary.get("unreliable_reason")
            return PreheatEstimate(None, False, target_c, start_c, None, None, reason)
        if outdoor_for is None:
            reason["status"] = "no_outdoor"
            return PreheatEstimate(None, False, target_c, start_c, None, None, reason)

        def minutes_for(tout: float) -> float | None:
            return preheat_minutes(start_c, target_c, tout, a, b, c, d)[0]

        minutes, tout_eff, info = solve_duration(minutes_for, outdoor_for)
        _, pinfo = preheat_minutes(start_c, target_c, tout_eff, a, b, c, d)
        t_inf = equilibrium(tout_eff, a, b, c)
        reason.update(pinfo)
        reason["solve"] = info
        reason["status"] = "ok" if minutes is not None else "unreachable"
        return PreheatEstimate(
            minutes=minutes,
            reachable=minutes is not None,
            target_c=target_c,
            start_c=start_c,
            outdoor_used_c=tout_eff if p["use_outdoor"] else None,
            max_reachable_c=t_inf,
            reason=reason,
        )

    def coast(
        self,
        now: datetime,
        start_c: float,
        minutes: float,
        outdoor_c: float | None,
        forecast: Sequence[ForecastPoint] = (),
    ) -> float | None:
        p = self._params
        if p is None:
            return None
        outdoor_for, _ = self._outdoor_function(now, outdoor_c, forecast)
        if outdoor_for is None:
            return None
        tout = outdoor_for(max(0.0, minutes))
        return coast_temperature(start_c, max(0.0, minutes), tout, p["b"], p["c"])

    def quality(self) -> QualityReport:
        q = self._quality
        samples = self._grid.valid_indoor_count()
        if q is None:
            return QualityReport(0.0, samples, 0.0, 0.0, None, None, None, 0)
        return QualityReport(
            maturity=q["maturity"],
            samples=samples,
            heating_hours=q["heating_hours"],
            cooling_hours=q["cooling_hours"],
            rmse_c=q["rmse_c"],
            baseline_rmse_c=q["baseline_rmse_c"],
            duration_mae_minutes=q["duration_mae_minutes"],
            episodes=q["episodes"],
        )

    # -- persistence --------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "model": "rc",
            "config": asdict(self.config),
            "grid": self._grid.to_dict(),
            "pending": [
                [t, s.indoor_c, s.outdoor_c, s.effort, int(s.excluded)] for t, s in self._pending
            ],
            "last_sample_ts": self._last_sample_ts,
            "params": dict(self._params) if self._params else None,
            "outdoor_ref": self._outdoor_ref,
            "quality": dict(self._quality) if self._quality else None,
            "last_fit_ts": self._last_fit_ts,
            "data_end_at_fit": self._data_end_at_fit,
            "summary": self._summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], config: ModelConfig) -> RCThermalModel:
        if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
            raise ValueError("unsupported model state")
        model = cls(config)
        params = data.get("params")
        if params:
            model._params = {
                "a": float(params["a"]),
                "b": float(params["b"]),
                "c": float(params["c"]),
                "d_minutes": float(params["d_minutes"]),
                "use_outdoor": bool(params["use_outdoor"]),
                "reliable": bool(params.get("reliable", True)),
            }
        model._outdoor_ref = data.get("outdoor_ref")
        model._quality = data.get("quality")
        model._last_fit_ts = data.get("last_fit_ts")
        model._data_end_at_fit = data.get("data_end_at_fit")
        model._summary = data.get("summary") or {}
        model._last_sample_ts = data.get("last_sample_ts")
        grid_data = data.get("grid")
        if grid_data and float(grid_data.get("step_minutes", 0)) == config.step_minutes:
            try:
                model._grid = Grid.from_dict(grid_data)
            except (KeyError, ValueError, TypeError):
                model._grid = Grid(config.step_minutes)
            for row in data.get("pending", []):
                t, ind, out, eff, exc = row
                model._pending.append((t, Sample(dt_of(t), ind, out, eff, bool(exc))))
        model._trim()
        return model
