"""Report-on-change ("staircase") room sensors, change-point interpolation, outlier mask."""

from __future__ import annotations

from datetime import timedelta
from functools import cache

import pytest

from rtm_core import ModelConfig, RCThermalModel
from rtm_core import fit as fit_mod
from rtm_core import synthetic as syn
from rtm_core.episodes import learning_effort, regulation_mask
from rtm_core.fit import build_dataset, fit_rc, interpolate_steps, outlier_rows

DAYS = 21


def _rel(est: float, true: float) -> float:
    return abs(est - true) / true


@cache
def _slow_run(window_minutes: tuple[int, int] | None = None) -> syn.SyntheticRun:
    if window_minutes is None:
        return syn.generate(syn.slow_radiator_room(), DAYS, windows=False)
    spec = syn.slow_radiator_room(window_minutes=window_minutes, windows_per_week=2.0)
    return syn.generate(spec, DAYS, windows=True)


def _fit(samples) -> tuple[RCThermalModel, dict]:
    model = RCThermalModel(ModelConfig())
    model.load_history(samples, replace=True)
    return model, model.fit(samples[-1].ts)


# -- synthetic generator ------------------------------------------------------------------
def test_report_on_change_produces_a_staircase():
    run = syn.generate(syn.slow_radiator_room(), 7, windows=False, disturbances=False)
    stepped = syn.report_on_change(run, threshold_k=0.07, max_interval_min=60)
    assert len(stepped) == len(run.samples)
    changes = [
        (a, b)
        for a, b in zip(stepped, stepped[1:], strict=False)
        if b.indoor_c != a.indoor_c and b.indoor_c is not None and a.indoor_c is not None
    ]
    # far fewer distinct readings than samples, each report ≥ threshold or ≤ 60 min apart
    assert len(changes) < 0.1 * len(stepped)
    last_change = stepped[0].ts
    for a, b in changes:
        assert abs(b.indoor_c - a.indoor_c) >= 0.07 - 1e-9 or (
            b.ts - last_change >= timedelta(minutes=60)
        )
        last_change = b.ts
    # effort/outdoor/setpoint are untouched
    assert [s.effort for s in stepped] == [s.effort for s in run.samples]


# -- fit on a staircase sensor ------------------------------------------------------------
def test_slow_room_with_report_on_change_sensor_is_reliable():
    spec = syn.slow_radiator_room()
    _, s = _fit(syn.report_on_change(_slow_run()))
    assert s["reliable"] is True, s["unreliable_reason"]
    assert "b" not in s["projected"]
    assert _rel(s["a"], spec.a) <= 0.15
    assert _rel(1.0 / s["b"], 1.0 / spec.b) <= 0.30
    assert abs(s["dead_time_minutes"] - spec.dead_time_minutes) <= 15


@pytest.mark.parametrize("room", ["slow_radiator", "radiator", "fast_radiator"])
def test_dense_sampling_unchanged_by_interpolation(monkeypatch, room):
    """With a dense sensor the change-point interpolation is (almost) a no-op."""
    spec = syn.ROOMS[room]()
    run = syn.generate(spec, 14, windows=False)
    _, with_interp = _fit(run.samples)
    monkeypatch.setattr(fit_mod, "MAX_INTERP_HOURS", 0.0)
    _, without = _fit(run.samples)
    for key in ("a", "b"):
        assert with_interp[key] == pytest.approx(without[key], rel=0.05)
    assert with_interp["dead_time_minutes"] == without["dead_time_minutes"]
    assert _rel(with_interp["a"], spec.a) < 0.15


def test_steep_real_heat_ups_are_not_masked():
    """A fast radiator (5 K/h) on a dense sensor: its heat-ups are dynamics, not outliers."""
    run = syn.generate(syn.fast_radiator_room(), 14, windows=False, disturbances=False)
    _, s = _fit(run.samples)
    assert s["downweighted_fraction"] < 0.002


def test_thirty_minute_window_is_masked():
    run = _slow_run(window_minutes=(30, 30))
    assert len(run.windows) >= 3
    spec = run.spec
    model = RCThermalModel(ModelConfig())
    samples = syn.report_on_change(run)
    model.load_history(samples, replace=True)
    grid = model._grid
    now = samples[-1].ts.timestamp()
    reg = regulation_mask(grid)
    ds = build_dataset(grid, model.config, now, skip=reg)
    res = fit_rc(ds, learning_effort(grid, reg), model.config)
    assert res is not None
    inside: list[int] = []
    for ts, _drop in run.windows:
        k = int((ts.timestamp() - grid.start_ts) // grid.step_s)
        # rows fully inside the opening (some may be skipped as thermostat regulation; the
        # first reported change of a staircase sensor can lag the opening by a row)
        inside += [j for j in ds.k if k + 2 <= j <= k + 5]
    assert len(inside) >= 8
    assert sum(j in res.masked for j in inside) >= 0.9 * len(inside)
    assert 0.0 < res.downweighted_fraction < 0.02
    assert _rel(res.a, spec.a) <= 0.15
    s = model.fit(samples[-1].ts)
    assert s["reliable"] is True
    assert _rel(s["a"], spec.a) <= 0.15


# -- building blocks ----------------------------------------------------------------------
def test_interpolate_steps_between_change_points():
    ind = [20.0, 20.0, 20.0, 20.3, 20.3, 20.3, 20.3, 20.0]
    out = interpolate_steps(ind, [False] * len(ind), max_rows=36, max_jump=1.5)
    assert out[:4] == pytest.approx([20.0, 20.1, 20.2, 20.3])
    assert out[4:] == pytest.approx([20.225, 20.15, 20.075, 20.0])


def test_interpolate_steps_keeps_gaps_long_plateaus_and_glitches():
    exc = [False] * 8
    # a missing row breaks the stretch
    out = interpolate_steps([20.0, 20.0, None, 20.0, 20.3, 20.3], exc[:6], 36, 1.5)
    assert out == [20.0, 20.0, None, 20.0, 20.3, 20.3]
    # an excluded row breaks it too
    out = interpolate_steps([20.0, 20.0, 20.0, 20.3], [False, True, False, False], 36, 1.5)
    assert out == [20.0, 20.0, 20.0, 20.3]
    # plateau longer than max_rows stays flat
    out = interpolate_steps([20.0, 20.0, 20.0, 20.3], exc[:4], 2, 1.5)
    assert out == [20.0, 20.0, 20.0, 20.3]
    # a jump larger than max_jump (glitch) is left as a single step
    out = interpolate_steps([20.0, 20.0, 20.0, 24.0, 20.0], exc[:5], 36, 1.5)
    assert out == [20.0, 20.0, 20.0, 24.0, 20.0]


def test_glitch_guard_uses_raw_readings():
    """A spike after a plateau is not smeared into a ramp that passes the guard."""
    from rtm_core.grid import Grid

    n = 40
    ind = [20.0] * n
    ind[20] = 24.0  # one-row spike after a 20-row plateau
    grid = Grid(5.0, 0.0, ind, [5.0] * n, [0.0] * n, [False] * n, [None] * n)
    ds = build_dataset(grid, ModelConfig(), now_ts=n * 300.0)
    assert 19 not in ds.k and 20 not in ds.k
    assert max(abs(y) for y in ds.y) == pytest.approx(0.0)


def test_outlier_rows_need_a_sustained_run():
    thr = 1.0
    ks = list(range(10))
    # isolated large residual between normal rows: kept
    assert outlier_rows(ks, [0, 0, 0, 5, 0, 0, 0, 0, 0, 0], thr) == frozenset()
    # two consecutive: kept; three: masked
    assert outlier_rows(ks, [0, 0, 5, -5, 0, 0, 0, 0, 0, 0], thr) == frozenset()
    assert outlier_rows(ks, [0, 0, 5, -5, 5, 0, 0, 0, 0, 0], thr) == frozenset({2, 3, 4})
    # a single outlier whose neighbours are missing from the regression counts as a run
    ks_gap = [0, 1, 2, 5, 8, 9]
    assert outlier_rows(ks_gap, [0, 0, 0, -6, 0, 0], thr) == frozenset({5})
    # ... but one missing neighbour on one side only is not enough for a single row
    ks_one = [0, 1, 2, 4, 5, 6]
    assert outlier_rows(ks_one, [0, 0, 0, -6, 0, 0], thr) == frozenset()
