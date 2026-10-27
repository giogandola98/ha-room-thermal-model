"""Thermostatic valves, ineffective heating, episode heating rate and plausibility."""

from __future__ import annotations

from rtm_core import ModelConfig, RCThermalModel
from rtm_core import synthetic as syn
from rtm_core.episodes import episode_heating_rate, regulation_mask
from rtm_core.grid import Grid

SCENARIOS = [
    (t0, target, tout)
    for t0 in (16.0, 17.0, 18.5)
    for target in (20.0, 21.0)
    for tout in (4.0, 11.0)
]


def _grid(indoor, effort):
    n = len(indoor)
    return Grid(5.0, 0.0, list(indoor), [10.0] * n, list(effort), [False] * n)


def test_regulation_mask_keeps_heat_up_drops_plateau():
    indoor = [18.0, 18.0, 18.5, 19.0, 19.5, 20.0, 20.0, 19.95, 20.0, 18.0]
    effort = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    mask = regulation_mask(_grid(indoor, effort))
    assert mask[:6] == [False] * 6  # heat-up up to the first row at the maximum
    assert mask[6:9] == [True, True, True]  # regulation around the setpoint
    assert mask[9] is False


def test_regulation_mask_drops_ineffective_run():
    # thermostat reports heating for hours, the temperature only falls (boiler off)
    indoor = [21.0 - 0.01 * i for i in range(60)]
    effort = [1.0] * 60
    mask = regulation_mask(_grid(indoor, effort))
    assert not mask[0] and all(mask[1:])


def test_episode_heating_rate_needs_two_episodes():
    indoor = [18.0] * 4 + [18.0 + 0.05 * i for i in range(12)] + [18.6] * 4
    effort = [0.0] * 4 + [1.0] * 12 + [0.0] * 4
    assert episode_heating_rate(_grid(indoor, effort), 0.05, 0.0, 0, True) is None
    two = _grid(indoor * 2, effort * 2)
    est = episode_heating_rate(two, 0.05, 0.0, 0, True)
    assert est is not None and est.episodes == 2
    # 0.05 K per 5 min = 0.6 K/h, plus the loss b·(mean T − Tout) = 0.05·8.3
    assert abs(est.a_k_per_h - (0.6 + 0.05 * (18.3 - 10.0))) < 0.05


def _trv_model(days=21.0):
    spec = syn.trv_room()
    run = syn.generate(spec, days, windows=False)
    model = RCThermalModel(ModelConfig())
    model.load_history(run.samples, replace=True)
    summary = model.fit(run.samples[-1].ts)
    return spec, run, model, summary


def test_trv_room_heating_rate_and_preheat():
    spec, run, model, summary = _trv_model()
    assert summary["reliable"] is True
    assert summary["regulation_rows_skipped"] > 0
    p = model.parameters()
    assert abs(p.heating_rate_k_per_h - spec.a) / spec.a < 0.3
    # before the regulation mask the loss rate collapsed to ~0.012 (τ ≈ 85 h)
    assert abs(p.loss_rate_per_h - spec.b) / spec.b < 0.35
    now = run.samples[-1].ts
    rel = []
    for t0, target, tout in SCENARIOS:
        truth = syn.true_preheat_minutes(spec, t0, target, tout)
        est = model.preheat(now, t0, target, tout)
        assert truth is not None and est.minutes is not None
        rel.append(abs(est.minutes - truth) / truth)
    # the truth assumes a fully open valve up to the target, while the learnt rate includes
    # the throttling near the setpoint: ~30 % here (it was 100-150 % before the fix)
    assert sum(rel) / len(rel) < 0.4
    assert model.quality().maturity > 0


def test_unreliable_parameters_give_no_preheat():
    _, run, model, _ = _trv_model(10.0)
    state = model.to_dict()
    state["params"] = {"a": 2.0, "b": 0.01, "c": 0.0, "d_minutes": 30.0, "use_outdoor": True}
    state["params"]["reliable"] = False
    restored = RCThermalModel.from_dict(state, ModelConfig())
    est = restored.preheat(run.samples[-1].ts, 18.0, 20.0, 10.0)
    assert est.minutes is None and not est.reachable
    assert est.max_reachable_c is None
    assert est.reason["status"] == "unreliable"


def test_old_state_without_reliable_flag_is_usable():
    _, run, model, _ = _trv_model(10.0)
    state = model.to_dict()
    state["params"].pop("reliable")
    restored = RCThermalModel.from_dict(state, ModelConfig())
    assert restored.preheat(run.samples[-1].ts, 18.0, 20.0, 10.0).reason["status"] != "unreliable"
