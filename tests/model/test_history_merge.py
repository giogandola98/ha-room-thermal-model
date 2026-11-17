"""Merged heating runs, history merging and the holdout evaluation."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from rtm_core import ModelConfig, RCThermalModel
from rtm_core import synthetic as syn
from rtm_core.episodes import episode_heating_rate, heating_runs, merged_runs
from rtm_core.grid import Grid
from rtm_core.quality import Params, bad_rows, forecast_rmse, heat_up_episodes


def _grid(indoor, effort, setpoint=None, outdoor=10.0):
    n = len(indoor)
    sp = list(setpoint) if setpoint is not None else [None] * n
    return Grid(5.0, 0.0, list(indoor), [outdoor] * n, list(effort), [False] * n, sp)


def _paused_episode(pause_rows: int):
    """Idle hour, 40 min heating, a pause, 40 min heating, idle (0.1 K per heating row)."""
    effort = [0.0] * 12 + [1.0] * 8 + [0.0] * pause_rows + [1.0] * 8 + [0.0] * 12
    indoor = []
    t = 18.0
    for e in effort:
        indoor.append(round(t, 3))
        t += 0.1 if e else -0.005
    return indoor, effort


@pytest.mark.parametrize("pause_rows", [1, 2])  # 5 and 10 min
def test_short_pauses_are_one_episode(pause_rows):
    indoor, effort = _paused_episode(pause_rows)
    g = _grid(indoor, effort)
    assert len(heating_runs(g)) == 2
    assert len(merged_runs(g)) == 1
    eps = heat_up_episodes(g, bad_rows(g, 1.5), 0, 2)
    assert len(eps) == 1
    k0, k1, _ = eps[0]
    assert k0 == 12 and k1 == 12 + 16 + pause_rows


def test_long_pause_splits_the_episode():
    indoor, effort = _paused_episode(6)  # 30 min
    g = _grid(indoor, effort)
    assert len(merged_runs(g)) == 2
    assert len(heat_up_episodes(g, bad_rows(g, 1.5), 0, 2)) == 2


def test_episode_rate_divides_by_duty():
    """With pauses the climb is slower; the full-effort rate is the duty-corrected one."""
    indoor, effort = _paused_episode(2)
    g = _grid(indoor * 2, effort * 2)
    est = episode_heating_rate(g, 0.0, 0.0, 0, True)
    assert est is not None and est.episodes == 2
    assert est.a_k_per_h == pytest.approx(1.2, rel=0.1)  # 0.1 K / 5 min at full effort


def test_small_rises_count_without_setpoint():
    """A 0.8 K heat-up without setpoint is an episode (band 0.3 K below the top)."""
    effort = [0.0] * 12 + [1.0] * 10 + [0.0] * 6
    indoor = [18.0] * 12 + [18.0 + 0.08 * i for i in range(10)] + [18.8] * 6
    g = _grid(indoor * 2, effort * 2)
    est = episode_heating_rate(g, 0.0, 0.0, 0, True)
    assert est is not None and est.episodes == 2


# -- grid merging -------------------------------------------------------------------------
def test_merged_fills_missing_setpoint_and_outdoor():
    live = Grid(5.0, 0.0, [20.0, 20.1, 20.2], [None, 5.0, None], [0.0] * 3, [False] * 3)
    live.setpoint = [None, None, 21.0]
    hist = Grid(5.0, 300.0, [19.0, 19.5, 19.9], [6.0, 6.5, 7.0], [1.0] * 3, [False] * 3)
    hist.setpoint = [18.0, 18.5, None]
    m = hist.merged(live)  # history wins where it has indoor
    assert m.start_ts == 0.0 and len(m) == 4
    assert m.indoor == [20.0, 19.0, 19.5, 19.9]
    assert m.setpoint == [None, 18.0, 18.5, None]
    assert m.outdoor == [None, 6.0, 6.5, 7.0]
    m2 = live.merged(hist)  # live wins, but takes the missing values from history
    assert m2.indoor == [20.0, 20.1, 20.2, 19.9]
    assert m2.setpoint == [None, 18.0, 21.0, None]
    assert m2.outdoor == [None, 5.0, 6.5, 7.0]


def test_history_merge_keeps_older_grid_and_history_wins():
    run = syn.generate(syn.radiator_room(), 30, windows=False, disturbances=False)
    stored = [replace(s, setpoint_c=None) for s in run.samples]  # old data: no setpoint
    model = RCThermalModel(ModelConfig())
    model.load_history(stored, replace=True)
    days_before = len(model._grid) * 5 / 1440
    cutoff = run.samples[-1].ts - timedelta(days=10)
    recorder = [s for s in run.samples if s.ts >= cutoff]  # 10 days with setpoint
    model.load_history(recorder)  # retrain default: merge
    grid = model._grid
    assert len(grid) * 5 / 1440 == pytest.approx(days_before, abs=0.01)  # 30 days kept
    k_cut = int((cutoff.timestamp() - grid.start_ts) // grid.step_s) + 1
    assert all(v is None for v in grid.setpoint[: k_cut - 1])
    recent = grid.setpoint[k_cut:]
    assert sum(v is not None for v in recent) >= 0.99 * len(recent)

    model.load_history(recorder, replace=True)  # old behaviour on request
    assert len(model._grid) * 5 / 1440 == pytest.approx(10, abs=0.1)


def test_reset_parameters_keeps_data():
    run = syn.generate(syn.radiator_room(), 4, windows=False)
    model = RCThermalModel(ModelConfig())
    model.load_history(run.samples, replace=True)
    model.fit(run.samples[-1].ts)
    rows = len(model._grid)
    model.reset_parameters()
    assert model.parameters() is None
    assert len(model._grid) == rows
    assert model.fit_due(run.samples[-1].ts)


# -- evaluation ---------------------------------------------------------------------------
def test_evaluation_skips_unknown_learning_effort():
    n = 60
    g = _grid([20.0] * n, [0.0] * n)
    p = Params(1.0, 0.01, 0.1, 0, True)
    bad = bad_rows(g, 1.5)
    _, _, all_windows = forecast_rmse(g, p, bad, 0)
    eff = [None if 20 <= i < 30 else 0.0 for i in range(n)]
    _, _, some = forecast_rmse(g, p, bad, 0, effort=eff)
    assert 0 < some < all_windows


def test_fit_reports_4h_skill_and_episode_count(trained):
    s = trained("radiator", 14).last_fit_summary()
    q = s["quality"]
    assert q["rmse_4h_c"] is not None and q["baseline_rmse_4h_c"] is not None
    assert q["rmse_4h_c"] < q["baseline_rmse_4h_c"]
    assert s["heat_up_episodes_total"] >= 10


def test_holdout_fallbacks_use_training_rows_only(monkeypatch):
    import rtm_core.model as model_mod

    seen: list[int] = []
    real = model_mod.episode_heating_rate

    def spy(grid, *a, **k):
        seen.append(len(grid))
        return real(grid, *a, **k)

    monkeypatch.setattr(model_mod, "episode_heating_rate", spy)
    run = syn.generate(syn.radiator_room(), 10, windows=False)
    model = RCThermalModel(ModelConfig())
    model.load_history(run.samples, replace=True)
    model.fit(run.samples[-1].ts)
    assert len(seen) == 2
    assert seen[1] < 0.85 * seen[0]  # second call: the training part only
