"""Preheat correction from real heat-ups (measured vs full-power predicted minutes)."""

from __future__ import annotations

from functools import cache

import pytest

from rtm_core import ModelConfig, RCThermalModel
from rtm_core import synthetic as syn
from rtm_core.grid import dt_of
from rtm_core.quality import (
    FEW_EPISODES_P80,
    EpisodeError,
    correction_from,
    leave_one_out,
    merge_errors,
)


@cache
def _state() -> dict:
    run = syn.generate(syn.trv_cycling_room(), 21, windows=False)
    model = RCThermalModel(ModelConfig())
    model.load_history(run.samples, replace=True)
    model.fit(run.samples[-1].ts)
    return model.to_dict()


def _model() -> RCThermalModel:
    return RCThermalModel.from_dict(_state(), ModelConfig())


def _now(m: RCThermalModel):
    return dt_of(m._grid.ts(len(m._grid) - 1))


def test_correction_rules():
    few = correction_from([1.3, 1.2])
    assert not few.active and few.factor == 1.0 and few.p80 == FEW_EPISODES_P80
    ratios = [1.0, 1.1, 1.2, 1.3, 1.4]
    c = correction_from(ratios)
    assert c.active and c.factor == pytest.approx(1.2) and c.p80 == pytest.approx(1.32)
    assert c.at(0.5) == c.factor and c.at(0.8) == pytest.approx(c.p80)
    assert correction_from([3.0] * 6).factor == 1.5  # clamped
    assert correction_from([0.5] * 6).factor == 0.8


def test_merge_keeps_older_episodes_and_caps():
    old = [EpisodeError(float(i), 100.0, 120.0, 1.0) for i in range(30)]
    fresh = [EpisodeError(100.0 + i, 100.0, 110.0, 1.0) for i in range(20)]
    merged = merge_errors(old, fresh, grid_start=20.0)
    # stored episodes inside the grid (ts >= 20) are replaced by the fresh evaluation
    assert len(merged) == 40 and merged[0].ts == 0.0 and merged[-1].ts == 119.0
    assert not [e for e in merged if 20.0 <= e.ts < 100.0]
    more = [EpisodeError(200.0 + i, 100.0, 110.0, 1.0) for i in range(30)]
    capped = merge_errors(merged, more, grid_start=150.0)
    assert len(capped) == 40 and capped[0].ts == 110.0


def test_trv_cycling_room_learns_the_correction():
    m = _model()
    s = m.last_fit_summary()
    corr = s["preheat_correction"]
    loo = s["preheat_correction_loo"]
    assert corr["active"] and corr["episodes"] >= 10
    assert corr["factor"] >= 1.0 and corr["p80_factor"] > corr["factor"]
    raw_bias = sum(e.predicted_min - e.measured_min for e in m._episode_errors) / len(
        m._episode_errors
    )
    assert raw_bias < 0  # full power is early
    assert abs(loo["bias_min"]) <= abs(raw_bias)
    assert loo["p80_coverage"] >= 0.7
    est = m.preheat(_now(m), 17.5, 20.5, 5.0)
    assert est.minutes == pytest.approx(est.minutes_p50)
    assert est.minutes_full_power < est.minutes_p50 < est.minutes_p80
    q = m.preheat(_now(m), 17.5, 20.5, 5.0, quantile=0.8)
    assert q.minutes == pytest.approx(est.minutes_p80)
    assert m.quality().p80_coverage == pytest.approx(loo["p80_coverage"], abs=1e-3)


def test_v2_state_loads_without_episode_errors():
    state = dict(_state())
    state["version"] = 2
    state.pop("episode_errors")
    m = RCThermalModel.from_dict(state, ModelConfig())
    assert m._episode_errors == []
    assert m.parameters() == _model().parameters()
    est = m.preheat(_now(m), 17.5, 20.5, 5.0)
    # no correction until the next fit rebuilds it from the stored grid
    assert est.minutes == pytest.approx(est.minutes_full_power)
    assert est.minutes_p80 == pytest.approx(FEW_EPISODES_P80 * est.minutes_full_power)
    m.fit(_now(m))
    assert len(m._episode_errors) == len(_model()._episode_errors)


def test_v3_round_trip():
    m = _model()
    again = RCThermalModel.from_dict(m.to_dict(), ModelConfig())
    assert m.to_dict()["version"] == 3
    assert [e.to_list() for e in again._episode_errors] == [e.to_list() for e in m._episode_errors]


def test_hold_cost():
    m = _model()
    p = m.parameters()
    loss, duty = m.hold_cost(20.0, 5.0)
    assert loss == pytest.approx(p.loss_rate_per_h * 15.0 - p.gains_k_per_h)
    assert duty == pytest.approx(loss / p.heating_rate_k_per_h)
    assert leave_one_out([])["episodes"] == 0
