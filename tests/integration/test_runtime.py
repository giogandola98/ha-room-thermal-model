"""Live sampling, target, preheat estimate, preheat start / preheat now."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

from freezegun.api import FrozenDateTimeFactory
from homeassistant.const import STATE_UNKNOWN
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_fire_time_changed

from custom_components.room_thermal_model.coordinator import RTMCoordinator
from custom_components.room_thermal_model.rtm_core.base import PreheatEstimate

from .conftest import (
    ARRIVAL,
    CLIMATE,
    EFFORT,
    MISSING,
    SLUG,
    WEATHER,
    FakeModel,
    climate_attrs,
    make_entry,
    set_inputs,
    setup_entry,
)
from .test_forecast import register_weather

PREHEAT = f"sensor.{SLUG}_preheat_minutes"
START = f"sensor.{SLUG}_preheat_start"
NOW_SENSOR = f"binary_sensor.{SLUG}_preheat_now"


async def _fitted(hass: HomeAssistant, entry) -> RTMCoordinator:
    coordinator: RTMCoordinator = entry.runtime_data
    coordinator.model.due = True
    await coordinator.async_refresh()
    await hass.async_block_till_done(wait_background_tasks=True)
    assert coordinator.model.fits == 1
    return coordinator


async def test_learning_then_estimate(hass: HomeAssistant, fake_model: None) -> None:
    """Unknown (not 0) while learning; minutes and parameters after the first fit."""
    set_inputs(hass)
    entry = await setup_entry(hass, make_entry(hass))
    state = hass.states.get(PREHEAT)
    assert state.state == STATE_UNKNOWN
    assert state.attributes["reason"] == "learning"
    await _fitted(hass, entry)
    state = hass.states.get(PREHEAT)
    assert float(state.state) == 30.0  # (20.5 - 19) K at 3 K/h
    attrs = state.attributes
    assert attrs["reason"] == "ok"
    assert attrs["reachable"] is True
    assert attrs["target_c"] == 20.5
    assert attrs["start_c"] == 19.0
    assert attrs["max_reachable_c"] == 24.0
    assert attrs["forecast_used"] is False
    assert attrs["maturity"] == 50.0
    assert attrs["details"] == {"iterations": 1, "outdoor_source": "current"}
    assert hass.states.get(f"sensor.{SLUG}_time_constant").state == "8.0"
    assert hass.states.get(f"sensor.{SLUG}_heating_rate").state == "3.0"
    assert hass.states.get(f"sensor.{SLUG}_dead_time").state == "30.0"
    assert hass.states.get(f"sensor.{SLUG}_max_reachable_temperature").state == "24.0"
    maturity = hass.states.get(f"sensor.{SLUG}_model_maturity")
    assert maturity.state == "50.0"
    assert maturity.attributes["rmse_c"] == 0.1
    assert maturity.attributes["last_fit"] is not None


async def test_follow_climate_target(hass: HomeAssistant, fake_model: None) -> None:
    """Target from the climate ``temperature``; fixed value as fallback."""
    set_inputs(hass)
    entry = await setup_entry(hass, make_entry(hass, target_mode="climate"))
    coordinator = await _fitted(hass, entry)
    state = hass.states.get(PREHEAT)
    assert state.attributes["target_c"] == 21.0
    assert state.attributes["target_source"] == "climate"
    assert float(state.state) == 40.0
    hass.states.async_set(CLIMATE, "heat", climate_attrs(target=None))
    await coordinator.async_refresh()
    state = hass.states.get(PREHEAT)
    assert state.attributes["target_c"] == 20.5
    assert state.attributes["target_source"] == "fixed"


async def test_status_unreachable_at_target_no_temperature(
    hass: HomeAssistant, fake_model: None
) -> None:
    """Reasons other than ok."""
    set_inputs(hass)
    entry = await setup_entry(hass, make_entry(hass))
    coordinator = await _fitted(hass, entry)
    coordinator.model.reachable = False
    await coordinator.async_refresh()
    state = hass.states.get(PREHEAT)
    assert state.state == STATE_UNKNOWN
    assert state.attributes["reason"] == "unreachable"
    assert state.attributes["reachable"] is False

    coordinator.model.reachable = True
    hass.states.async_set(CLIMATE, "heat", climate_attrs(current=21.0))
    await coordinator.async_refresh()
    state = hass.states.get(PREHEAT)
    assert float(state.state) == 0.0
    assert state.attributes["reason"] == "at_target"

    hass.states.async_set(CLIMATE, "heat", climate_attrs(current=None))
    await coordinator.async_refresh()
    state = hass.states.get(PREHEAT)
    assert state.state == STATE_UNKNOWN
    assert state.attributes["reason"] == "no_temperature"


async def test_live_samples_effort(hass: HomeAssistant, fake_model: None) -> None:
    """State changes feed samples; effort from hvac_action, fallback and sensor."""
    set_inputs(hass)
    entry = await setup_entry(hass, make_entry(hass))
    model: FakeModel = entry.runtime_data.model
    hass.states.async_set(CLIMATE, "heat", climate_attrs(current=19.2, action="heating"))
    await hass.async_block_till_done(wait_background_tasks=True)
    assert (model.samples[-1].indoor_c, model.samples[-1].effort) == (19.2, 1.0)
    hass.states.async_set(CLIMATE, "heat", climate_attrs(current=19.4, action=MISSING))
    await hass.async_block_till_done(wait_background_tasks=True)
    assert model.samples[-1].effort == 1.0  # 21 > 19.4 + 0.2
    hass.states.async_set(CLIMATE, "off", climate_attrs(current=19.4, action=MISSING))
    await hass.async_block_till_done(wait_background_tasks=True)
    assert model.samples[-1].effort == 0.0
    hass.states.async_set(CLIMATE, "unavailable", {})
    await hass.async_block_till_done(wait_background_tasks=True)
    assert model.samples[-1].excluded


async def test_effort_sensor_samples(hass: HomeAssistant, fake_model: None) -> None:
    """With an effort sensor the valve opening is the effort."""
    set_inputs(hass, full=True)
    entry = await setup_entry(hass, make_entry(hass, full=True))
    model: FakeModel = entry.runtime_data.model
    hass.states.async_set(EFFORT, "75", {"unit_of_measurement": "%"})
    await hass.async_block_till_done(wait_background_tasks=True)
    sample = model.samples[-1]
    assert sample.effort == 0.75
    assert sample.indoor_c == 19.5  # room sensor
    assert sample.outdoor_c == 5.0  # outdoor sensor wins over the weather attribute


async def test_grid_sample_and_poll(
    hass: HomeAssistant, fake_model: None, freezer: FrozenDateTimeFactory
) -> None:
    """Without state changes a sample is still fed every 5 minutes."""
    set_inputs(hass)
    entry = await setup_entry(hass, make_entry(hass))
    model: FakeModel = entry.runtime_data.model
    count = len(model.samples)
    freezer.tick(timedelta(minutes=5, seconds=1))
    async_fire_time_changed(hass)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert len(model.samples) >= count + 1
    assert model.samples[-1].ts >= dt_util.utcnow() - timedelta(seconds=2)


async def test_forecast_passed_to_model(hass: HomeAssistant, fake_model: None) -> None:
    """The weather forecast reaches ``preheat`` and is flagged in the attributes."""
    set_inputs(hass, full=True)
    register_weather(hass)
    entry = await setup_entry(hass, make_entry(hass, full=True))
    coordinator = await _fitted(hass, entry)
    call = coordinator.model.preheat_calls[-1]
    assert call["forecast"]
    assert call["outdoor"] == 5.0
    state = hass.states.get(PREHEAT)
    assert state.attributes["forecast_used"] is True
    assert state.attributes["outdoor_used_c"] == call["forecast"][0].outdoor_c
    assert coordinator.data.forecast_type == "hourly"


async def test_weather_outdoor_fallback(hass: HomeAssistant, fake_model: None) -> None:
    """Weather only (no outdoor sensor): its temperature attribute is the outdoor value."""
    set_inputs(hass)
    hass.states.async_set(WEATHER, "sunny", {"temperature": 7.5, "temperature_unit": "°C"})
    entry = make_entry(hass)
    hass.config_entries.async_update_entry(entry, data={**entry.data, "weather_entity": WEATHER})
    await setup_entry(hass, entry)
    assert entry.runtime_data.model.samples[-1].outdoor_c == 7.5


async def test_preheat_start_and_now(
    hass: HomeAssistant, fake_model: None, freezer: FrozenDateTimeFactory
) -> None:
    """preheat_start = arrival − minutes; preheat_now flips at the start, off at arrival."""
    set_inputs(hass)
    now = dt_util.utcnow()
    arrival = now + timedelta(hours=2)
    hass.states.async_set(ARRIVAL, arrival.isoformat(), {"device_class": "timestamp"})
    entry = await setup_entry(hass, make_entry(hass, arrival=True))
    coordinator = await _fitted(hass, entry)
    start = dt_util.parse_datetime(hass.states.get(START).state)
    # Cooling 1 K/h until the start: fixed point lead = 70 - lead / 3 = 52.5 min.
    assert abs(start - (arrival - timedelta(minutes=52.5))) < timedelta(seconds=10)
    assert hass.states.get(NOW_SENSOR).state == "off"
    assert coordinator.data.coast_c is not None
    assert hass.states.get(PREHEAT).attributes["arrival"] == arrival.isoformat()

    # Exactly at the start (timer), before the next poll.
    freezer.move_to(start + timedelta(seconds=5))
    async_fire_time_changed(hass)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert hass.states.get(NOW_SENSOR).state == "on"

    freezer.move_to(arrival + timedelta(seconds=5))
    async_fire_time_changed(hass)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert hass.states.get(NOW_SENSOR).state == "off"


async def test_arrival_soon_past_invalid(hass: HomeAssistant, fake_model: None) -> None:
    """Arrival sooner than needed → on; past / invalid → off and no start."""
    set_inputs(hass)
    now = dt_util.utcnow()
    hass.states.async_set(
        ARRIVAL, (now + timedelta(minutes=10)).isoformat(), {"device_class": "timestamp"}
    )
    entry = await setup_entry(hass, make_entry(hass, arrival=True))
    coordinator = await _fitted(hass, entry)
    assert hass.states.get(NOW_SENSOR).state == "on"

    hass.states.async_set(
        ARRIVAL, (now - timedelta(minutes=10)).isoformat(), {"device_class": "timestamp"}
    )
    await coordinator.async_refresh()
    assert hass.states.get(NOW_SENSOR).state == "off"
    assert hass.states.get(START).state == STATE_UNKNOWN

    hass.states.async_set(ARRIVAL, "unavailable", {"device_class": "timestamp"})
    await coordinator.async_refresh()
    assert hass.states.get(NOW_SENSOR).state == "off"


async def test_preheat_now_unknown_while_learning(hass: HomeAssistant, fake_model: None) -> None:
    """Future arrival but no estimate yet: preheat now is unknown, start unknown."""
    set_inputs(hass)
    arrival = dt_util.utcnow() + timedelta(hours=1)
    hass.states.async_set(ARRIVAL, arrival.isoformat(), {"device_class": "timestamp"})
    await setup_entry(hass, make_entry(hass, arrival=True))
    assert hass.states.get(NOW_SENSOR).state == STATE_UNKNOWN
    assert hass.states.get(START).state == STATE_UNKNOWN


async def test_arrival_change_refreshes(hass: HomeAssistant, fake_model: None) -> None:
    """A new arrival time is picked up without waiting for the poll."""
    set_inputs(hass)
    hass.states.async_set(ARRIVAL, STATE_UNKNOWN, {"device_class": "timestamp"})
    entry = await setup_entry(hass, make_entry(hass, arrival=True))
    await _fitted(hass, entry)
    model: FakeModel = entry.runtime_data.model
    samples = len(model.samples)
    arrival = dt_util.utcnow() + timedelta(hours=3)
    hass.states.async_set(ARRIVAL, arrival.isoformat(), {"device_class": "timestamp"})
    await hass.async_block_till_done(wait_background_tasks=True)
    start = dt_util.parse_datetime(hass.states.get(START).state)
    # Cooling 1 K/h for 3 h: lead = 90 - lead / 3 = 67.5 min.
    assert abs(start - (arrival - timedelta(minutes=67.5))) < timedelta(seconds=10)
    # Arrival changes do not create samples by themselves (the refresh does one poll sample).
    assert len(model.samples) <= samples + 1


async def test_unreachable_target_with_arrival(hass: HomeAssistant, fake_model: None) -> None:
    """Unreachable target: minutes stay unknown, but preheat starts towards the maximum."""
    set_inputs(hass)
    hass.states.async_set(CLIMATE, "heat", climate_attrs(current=17.0))
    arrival = dt_util.utcnow() + timedelta(hours=2)
    hass.states.async_set(ARRIVAL, arrival.isoformat(), {"device_class": "timestamp"})
    entry = await setup_entry(hass, make_entry(hass, arrival=True))
    coordinator = await _fitted(hass, entry)
    coordinator.model.reachable = False  # max reachable 19.5 -> goal 19.0
    await coordinator.async_refresh()
    assert hass.states.get(PREHEAT).state == STATE_UNKNOWN
    assert hass.states.get(PREHEAT).attributes["reason"] == "unreachable"
    # Cooling 1 K/h towards a 19.0 goal: lead = 80 - lead / 3 = 60 min.
    start = dt_util.parse_datetime(hass.states.get(START).state)
    assert abs(start - (arrival - timedelta(minutes=60))) < timedelta(seconds=10)
    assert hass.states.get(NOW_SENSOR).state == "off"

    # Already at the goal: no heating now, but the room cools until the start.
    # lead = (120 - lead) / 3 = 30 min.
    hass.states.async_set(CLIMATE, "heat", climate_attrs(current=19.0))
    await coordinator.async_refresh()
    start = dt_util.parse_datetime(hass.states.get(START).state)
    assert abs(start - (arrival - timedelta(minutes=30))) < timedelta(seconds=10)
    assert hass.states.get(NOW_SENSOR).state == "off"


async def test_preheat_now_latch_holds_and_releases(
    hass: HomeAssistant, fake_model: None, freezer: FrozenDateTimeFactory
) -> None:
    """On stays on when the room reaches the target; a far postponement releases it."""
    set_inputs(hass)
    arrival = dt_util.utcnow() + timedelta(minutes=20)
    hass.states.async_set(ARRIVAL, arrival.isoformat(), {"device_class": "timestamp"})
    entry = await setup_entry(hass, make_entry(hass, arrival=True))
    coordinator = await _fitted(hass, entry)
    assert hass.states.get(NOW_SENSOR).state == "on"

    hass.states.async_set(CLIMATE, "heat", climate_attrs(current=21.0))
    await coordinator.async_refresh()
    assert hass.states.get(PREHEAT).attributes["reason"] == "at_target"
    assert coordinator.data.latched
    assert hass.states.get(NOW_SENSOR).state == "on"

    new_arrival = dt_util.utcnow() + timedelta(hours=5)
    hass.states.async_set(ARRIVAL, new_arrival.isoformat(), {"device_class": "timestamp"})
    await hass.async_block_till_done(wait_background_tasks=True)
    await coordinator.async_refresh()
    assert not coordinator.data.latched
    assert hass.states.get(NOW_SENSOR).state == "off"


async def test_preheat_now_latch_holds_until_arrival_when_at_target_early(
    hass: HomeAssistant, fake_model: None, freezer: FrozenDateTimeFactory
) -> None:
    """Target reached long before the arrival: on until the arrival, no off/on flap."""
    set_inputs(hass)
    arrival = dt_util.utcnow() + timedelta(minutes=60)
    hass.states.async_set(ARRIVAL, arrival.isoformat(), {"device_class": "timestamp"})
    entry = await setup_entry(hass, make_entry(hass, arrival=True))
    coordinator = await _fitted(hass, entry)
    # Cooling 1 K/h until the start: lead = 50 - lead / 3 = 37.5 min.
    start = dt_util.parse_datetime(hass.states.get(START).state)
    assert abs(start - (arrival - timedelta(minutes=37.5))) < timedelta(seconds=10)
    freezer.move_to(start + timedelta(seconds=5))
    async_fire_time_changed(hass)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert hass.states.get(NOW_SENSOR).state == "on"

    # At target 37 min before the arrival: the lead drops to ~2 min, the latch holds.
    hass.states.async_set(CLIMATE, "heat", climate_attrs(current=21.0))
    await coordinator.async_refresh()
    assert coordinator.data.preheat_start > dt_util.utcnow() + timedelta(minutes=30)
    assert coordinator.data.latched
    assert hass.states.get(NOW_SENSOR).state == "on"

    # A small postponement (within the hold margin) keeps it on as well.
    arrival += timedelta(minutes=10)
    hass.states.async_set(ARRIVAL, arrival.isoformat(), {"device_class": "timestamp"})
    await hass.async_block_till_done(wait_background_tasks=True)
    for minutes in (10, 20, 30, 40):
        freezer.move_to(start + timedelta(minutes=minutes))
        await coordinator.async_refresh()
        assert hass.states.get(NOW_SENSOR).state == "on", minutes

    freezer.move_to(arrival + timedelta(seconds=5))
    async_fire_time_changed(hass)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert hass.states.get(NOW_SENSOR).state == "off"


async def test_status_unreliable(hass: HomeAssistant, fake_model: None) -> None:
    """Implausible parameters give no preheat value and no preheat start."""
    set_inputs(hass)
    entry = await setup_entry(hass, make_entry(hass))
    coordinator = await _fitted(hass, entry)
    estimate = PreheatEstimate(
        minutes=None,
        reachable=False,
        target_c=20.5,
        start_c=19.0,
        outdoor_used_c=None,
        max_reachable_c=None,
        reason={"status": "unreliable", "unreliable_reason": "max_reachable_implausible"},
    )
    with patch.object(coordinator.model, "preheat", return_value=estimate):
        await coordinator.async_refresh()
    state = hass.states.get(PREHEAT)
    assert state.state == STATE_UNKNOWN
    assert state.attributes["reason"] == "unreliable"
    assert coordinator.data.preheat_start is None


async def test_status_no_outdoor(hass: HomeAssistant, fake_model: None) -> None:
    """A model without an outdoor temperature reports its own status, not unreachable."""
    set_inputs(hass)
    entry = await setup_entry(hass, make_entry(hass))
    coordinator = await _fitted(hass, entry)
    estimate = PreheatEstimate(
        minutes=None,
        reachable=False,
        target_c=20.5,
        start_c=19.0,
        outdoor_used_c=None,
        max_reachable_c=None,
        reason={"status": "no_outdoor", "outdoor_source": "no_outdoor"},
    )
    with patch.object(coordinator.model, "preheat", return_value=estimate):
        await coordinator.async_refresh()
    state = hass.states.get(PREHEAT)
    assert state.state == STATE_UNKNOWN
    assert state.attributes["reason"] == "no_outdoor"


async def test_future_preheat_uses_forecast_not_current_outdoor(
    hass: HomeAssistant, fake_model: None
) -> None:
    """Preheat evaluated at a future start: the forecast, not today's reading, anchors it."""
    set_inputs(hass, full=True)
    register_weather(hass)
    arrival = dt_util.utcnow() + timedelta(hours=8)
    hass.states.async_set(ARRIVAL, arrival.isoformat(), {"device_class": "timestamp"})
    entry = await setup_entry(hass, make_entry(hass, full=True, arrival=True))
    coordinator = await _fitted(hass, entry)
    now = coordinator.data.updated_at
    calls = coordinator.model.preheat_calls
    current = [c for c in calls if c["now"] == now]
    future = [c for c in calls if c["now"] > now]
    assert current and all(c["outdoor"] == 5.0 for c in current)
    assert future and all(c["forecast"] and c["outdoor"] is None for c in future)


async def test_future_preheat_without_forecast_keeps_current_outdoor(
    hass: HomeAssistant, fake_model: None
) -> None:
    """Without a forecast the current reading is the only outdoor information."""
    set_inputs(hass, full=True)
    arrival = dt_util.utcnow() + timedelta(hours=8)
    hass.states.async_set(ARRIVAL, arrival.isoformat(), {"device_class": "timestamp"})
    entry = make_entry(hass, full=True, arrival=True)
    data = {k: v for k, v in entry.data.items() if k != "weather_entity"}
    hass.config_entries.async_update_entry(entry, data=data)
    entry = await setup_entry(hass, entry)
    coordinator = await _fitted(hass, entry)
    now = coordinator.data.updated_at
    future = [c for c in coordinator.model.preheat_calls if c["now"] > now]
    assert future and all(c["outdoor"] == 5.0 for c in future)


async def test_climate_setpoint_below_current(hass: HomeAssistant, fake_model: None) -> None:
    """Follow-the-climate with a setback setpoint: explicit reason instead of at_target."""
    set_inputs(hass)
    hass.states.async_set(CLIMATE, "heat", climate_attrs(current=18.0, target=17.0))
    entry = await setup_entry(hass, make_entry(hass, target_mode="climate"))
    await _fitted(hass, entry)
    state = hass.states.get(PREHEAT)
    assert float(state.state) == 0.0
    assert state.attributes["reason"] == "setpoint_below_current"


async def test_climate_max_temp_caps_max_reachable(hass: HomeAssistant, fake_model: None) -> None:
    """The climate's max_temp is passed to the model as the reachable ceiling."""
    set_inputs(hass)
    hass.states.async_set(CLIMATE, "heat", climate_attrs(max_temp=26.0))
    entry = await setup_entry(hass, make_entry(hass))
    coordinator = await _fitted(hass, entry)
    assert coordinator.model.preheat_calls[-1]["max_c"] == 26.0
    hass.states.async_set(CLIMATE, "heat", climate_attrs())
    await coordinator.async_refresh()
    assert coordinator.model.preheat_calls[-1]["max_c"] is None
