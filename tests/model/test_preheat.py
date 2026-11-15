"""Preheat / coast predictions against the simulated truth."""

from __future__ import annotations

import copy
import json
from datetime import timedelta

import pytest

from rtm_core import ForecastPoint, ModelConfig, RCThermalModel
from rtm_core import synthetic as syn
from rtm_core.physics import preheat_minutes

SCENARIOS = [
    (t0, target, tout)
    for t0 in (16.0, 17.0, 18.5)
    for target in (20.0, 21.0, 21.5)
    for tout in (-3.0, 4.0, 11.0)
]


def _errors(model, spec, now, scenarios=SCENARIOS, model_outdoor=True):
    errs = []
    for t0, target, tout in scenarios:
        truth = syn.true_preheat_minutes(spec, t0, target, tout)
        est = model.preheat(now, t0, target, tout if model_outdoor else None)
        if truth is None:
            continue
        assert est.reachable, (t0, target, tout)
        errs.append((est.minutes - truth, truth))
    return errs


@pytest.mark.parametrize("name", ["radiator", "fast_radiator"])
@pytest.mark.parametrize("windows", [False, True])
def test_preheat_mae_vs_truth(trained, run_of, name, windows):
    m = trained(name, 14, windows)
    now = run_of(name, 14, windows).samples[-1].ts
    errs = _errors(m, syn.ROOMS[name](), now)
    mae = sum(abs(e) for e, _ in errs) / len(errs)
    assert len(errs) >= 15
    assert mae < 10.0


def test_preheat_slow_underfloor_relative(trained, run_of):
    m = trained("underfloor", 14)
    now = run_of("underfloor", 14).samples[-1].ts
    errs = _errors(m, syn.underfloor_room(), now)
    rel = sum(abs(e) / t for e, t in errs) / len(errs)
    assert rel < 0.10


def test_no_outdoor_room_predicts_reasonably(trained, run_of):
    run = run_of("no_outdoor", 14)
    m = trained("no_outdoor", 14)
    now = run.samples[-1].ts
    tout = sum(run.outdoor[-1440 * 3 :]) / (1440 * 3)  # typical recent weather
    spec = syn.no_outdoor_room()
    scen = [(t0, tg, tout) for t0 in (16.0, 17.0, 18.0) for tg in (20.0, 21.0)]
    errs = _errors(m, spec, now, scen, model_outdoor=False)
    rel = sum(abs(e) / t for e, t in errs) / len(errs)
    assert rel < 0.3
    est = m.preheat(now, 17.0, 21.0, 5.0)
    assert est.outdoor_used_c is None and est.reason["outdoor_source"] == "not_used"


def test_unreachable_on_very_cold_outdoor(trained, run_of):
    now = run_of("fast_radiator", 14).samples[-1].ts
    m = trained("fast_radiator", 14)
    est = m.preheat(now, 15.0, 21.0, -15.0)
    assert not est.reachable and est.minutes is None
    assert est.max_reachable_c < 21.1
    assert est.reason["status"] == "unreachable"
    assert syn.true_preheat_minutes(syn.fast_radiator_room(), 15.0, 21.0, -15.0) is None


def test_already_at_target_and_not_fitted(trained, run_of):
    now = run_of("radiator", 14).samples[-1].ts
    est = trained("radiator", 14).preheat(now, 21.0, 20.5, 5.0)
    assert est.minutes == 0.0 and est.reachable
    fresh = RCThermalModel(ModelConfig())
    est = fresh.preheat(now, 17.0, 21.0, 5.0)
    assert est.minutes is None and est.reason["status"] == "not_fitted"
    assert fresh.coast(now, 20.0, 60, 5.0) is None


def test_reason_contains_parameters(trained, run_of):
    now = run_of("radiator", 14).samples[-1].ts
    est = trained("radiator", 14).preheat(now, 17.0, 21.0, 5.0)
    for key in (
        "a_k_per_h",
        "b_per_h",
        "c_k_per_h",
        "dead_time_minutes",
        "t_infinity_c",
        "minutes_simple",
        "start_after_dead_time_c",
        "outdoor_source",
        "solve",
    ):
        assert key in est.reason
    assert est.max_reachable_c == pytest.approx(min(30.0, est.reason["t_infinity_c"]))
    assert est.outdoor_used_c == 5.0


def test_preheat_with_forecast(trained, run_of):
    run = run_of("radiator", 14)
    now = run.samples[-1].ts
    m = trained("radiator", 14)
    spec = syn.radiator_room()
    # front arriving: −3 °C/h for 3 hours
    fc = [ForecastPoint(now + timedelta(hours=h), 8.0 - 3.0 * h) for h in range(0, 4)]
    est = m.preheat(now, 16.0, 21.0, 8.0, fc)
    flat = m.preheat(now, 16.0, 21.0, 8.0)
    assert est.minutes > flat.minutes
    assert est.outdoor_used_c < 8.0

    def tout(t_min: float) -> float:
        return max(-1.0, 8.0 - 3.0 * t_min / 60.0)

    truth = syn.true_preheat_minutes(spec, 16.0, 21.0, tout)
    assert abs(est.minutes - truth) < 10.0


def test_outdoor_fallback_to_model_reference(trained, run_of):
    now = run_of("radiator", 14).samples[-1].ts
    est = trained("radiator", 14).preheat(now, 17.0, 21.0, None)
    assert est.reachable and est.reason["outdoor_source"] == "model_reference"


def test_coast_vs_truth(trained, run_of):
    now = run_of("radiator", 14).samples[-1].ts
    m = trained("radiator", 14)
    spec = syn.radiator_room()
    for t0, minutes, tout in [(21.0, 120, 0.0), (20.0, 360, 8.0), (19.0, 30, -5.0)]:
        pred = m.coast(now, t0, minutes, tout)
        assert abs(pred - syn.true_coast(spec, t0, minutes, tout)) < 0.3
    assert m.coast(now, 21.0, 0, 5.0) == pytest.approx(21.0)


def test_queries_do_not_mutate(trained, run_of):
    now = run_of("radiator", 14).samples[-1].ts
    m = trained("radiator", 14)
    before = copy.deepcopy(m.to_dict())
    fc = [ForecastPoint(now + timedelta(hours=1), 2.0)]
    m.preheat(now, 17.0, 21.0, 5.0, fc)
    m.coast(now, 21.0, 120, 5.0, fc)
    m.parameters()
    m.quality()
    assert m.to_dict() == before


def test_json_round_trip_identical_outputs(trained, run_of):
    now = run_of("radiator", 14).samples[-1].ts
    m = trained("radiator", 14)
    text = json.dumps(m.to_dict())
    m2 = RCThermalModel.from_dict(json.loads(text), ModelConfig())
    fc = [ForecastPoint(now + timedelta(hours=h), 5.0 - h) for h in range(4)]
    for args in [(17.0, 21.0, 5.0), (16.0, 20.5, None), (18.0, 21.0, -2.0)]:
        a = m.preheat(now, *args, fc)
        b = m2.preheat(now, *args, fc)
        assert (a.minutes, a.reachable, a.outdoor_used_c, a.max_reachable_c) == (
            b.minutes,
            b.reachable,
            b.outdoor_used_c,
            b.max_reachable_c,
        )
        assert a.reason == b.reason
    assert m.coast(now, 21.0, 90, 3.0) == m2.coast(now, 21.0, 90, 3.0)
    assert m.parameters() == m2.parameters()
    assert m.quality() == m2.quality()
    assert len(text) < 200_000


def test_from_dict_tolerates_garbage_and_reset(trained):
    with pytest.raises(ValueError):
        RCThermalModel.from_dict({"version": 999}, ModelConfig())
    with pytest.raises(ValueError):
        RCThermalModel.from_dict("garbage", ModelConfig())  # type: ignore[arg-type]
    m = trained("radiator", 14)
    m.reset()
    assert m.parameters() is None and m.quality().samples == 0


def test_max_reachable_is_capped(trained, run_of):
    """The linear T∞ (~45 °C here) is capped: 30 °C by default, else the climate max_temp."""
    now = run_of("radiator", 14).samples[-1].ts
    m = trained("radiator", 14)
    est = m.preheat(now, 17.0, 21.0, 5.0)
    assert est.reason["t_infinity_c"] > 30.0
    assert est.max_reachable_c == 30.0
    capped = m.preheat(now, 17.0, 21.0, 5.0, max_c=24.0)
    assert capped.max_reachable_c == 24.0
    assert capped.minutes == pytest.approx(est.minutes)  # dynamics unchanged
    over = m.preheat(now, 17.0, 25.0, 5.0, max_c=24.0)
    assert not over.reachable and over.minutes is None
    assert over.reason["status"] == "unreachable"


def test_preheat_zero_when_room_drifts_to_target():
    """Gains warm the room past the target during the dead time: no heating needed."""
    # coasting equilibrium 25 °C; after 90 min the room is at ~20.7 °C > 20.5 °C
    minutes, info = preheat_minutes(20.0, 20.5, 0.0, 3.0, 0.1, 2.5, 90.0)
    assert info["start_after_dead_time_c"] >= 20.5
    assert minutes == 0.0
