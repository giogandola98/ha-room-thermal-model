"""State → sample conversion and recorder history reduction."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from homeassistant.core import State

from custom_components.room_thermal_model.history import reduce_states
from custom_components.room_thermal_model.inputs import (
    ZoneInputs,
    build_sample,
    effort_from_climate,
    heat_source_on,
    heating_effort,
    indoor_temperature,
    outdoor_temperature,
)

from .conftest import CLIMATE, EFFORT, MISSING, OUTDOOR, ROOM, WEATHER, climate_attrs

C = "°C"
T0 = datetime(2026, 1, 10, 8, 0, tzinfo=UTC)
MIN = ZoneInputs(climate=CLIMATE)
FULL = ZoneInputs(
    climate=CLIMATE, room_sensor=ROOM, effort_sensor=EFFORT, outdoor_sensor=OUTDOOR, weather=WEATHER
)


def _climate(state: str = "heat", **kwargs) -> State:
    return State(CLIMATE, state, climate_attrs(**kwargs))


def _temp(entity_id: str, value: str, unit: str = C, ts: datetime | None = None) -> State:
    return State(entity_id, value, {"unit_of_measurement": unit}, last_updated=ts, last_changed=ts)


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("heating", 1.0),
        ("preheating", 1.0),
        ("idle", 0.0),
        ("off", 0.0),
        ("fan", 0.0),
        (None, None),  # reported but unknown: not observed
        ("something_new", None),
    ],
)
def test_effort_from_hvac_action(action, expected) -> None:
    """hvac_action drives the effort when no effort sensor is configured."""
    assert effort_from_climate(_climate(action=action), C) == expected


@pytest.mark.parametrize(
    ("mode", "current", "target", "expected"),
    [
        ("heat", 19.0, 21.0, 1.0),  # target > current + 0.2
        ("heat", 20.9, 21.0, 0.0),  # within the margin
        ("off", 15.0, 21.0, 0.0),  # mode off
        ("heat", None, 21.0, 0.0),  # missing current
    ],
)
def test_effort_fallback_without_hvac_action(mode, current, target, expected) -> None:
    """Climates that never report hvac_action: mode + setpoint fallback."""
    state = _climate(mode, current=current, target=target, action=MISSING)
    assert effort_from_climate(state, C) == expected


def test_effort_sensor_preferred_and_clamped() -> None:
    """Effort sensor in % → 0..1, clamped; invalid → climate derivation."""
    climate = _climate(action="heating")
    inputs = ZoneInputs(climate=CLIMATE, effort_sensor=EFFORT)
    for raw, expected in (("40", 0.4), ("150", 1.0), ("-5", 0.0), ("unavailable", 1.0)):
        states = {CLIMATE: climate, EFFORT: State(EFFORT, raw)}
        assert heating_effort(inputs, states, C) == pytest.approx(expected)


BOILER = "binary_sensor.boiler_heating"
BURNER = "sensor.burner_power"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("on", True), ("off", False), ("35", True), ("0", False), ("unavailable", None), ("x", None)],
)
def test_heat_source_on(raw, expected) -> None:
    """On/off entities and numeric sensors (> 0 = on); unknown → None."""
    assert heat_source_on(State(BOILER, raw)) is expected


def test_heat_source_gates_effort() -> None:
    """Heating counts only while the heat source is on; unknown source → not observed."""
    inputs = ZoneInputs(climate=CLIMATE, heat_source=BOILER)
    heating = _climate(action="heating")
    assert heating_effort(inputs, {CLIMATE: heating, BOILER: State(BOILER, "on")}, C) == 1.0
    assert heating_effort(inputs, {CLIMATE: heating, BOILER: State(BOILER, "off")}, C) == 0.0
    assert heating_effort(inputs, {CLIMATE: heating, BOILER: None}, C) is None
    idle = _climate(action="idle")
    assert heating_effort(inputs, {CLIMATE: idle, BOILER: None}, C) == 0.0
    assert BOILER in inputs.sample_entities  # its history is imported too


def test_heat_source_numeric_with_effort_sensor() -> None:
    """A burner power sensor gates an effort sensor."""
    inputs = ZoneInputs(climate=CLIMATE, effort_sensor=EFFORT, heat_source=BURNER)
    states = {CLIMATE: _climate(), EFFORT: State(EFFORT, "60"), BURNER: State(BURNER, "40")}
    assert heating_effort(inputs, states, C) == pytest.approx(0.6)
    states[BURNER] = State(BURNER, "0")
    assert heating_effort(inputs, states, C) == 0.0


def test_unavailable_climate_gives_no_effort() -> None:
    """Unknown/unavailable climate: no effort, sample excluded."""
    states = {CLIMATE: State(CLIMATE, "unavailable")}
    sample = build_sample(MIN, states, T0, C)
    assert sample.effort is None
    assert sample.indoor_c is None
    assert sample.excluded


def test_cooling_is_excluded() -> None:
    """Active cooling is not described by the heating model."""
    sample = build_sample(MIN, {CLIMATE: _climate("cool", action="cooling")}, T0, C)
    assert sample.excluded
    assert sample.effort == 0.0


def test_room_sensor_overrides_climate() -> None:
    """Room sensor wins (no mixing when it is unavailable); °F converted."""
    states = {CLIMATE: _climate(current=19.0), ROOM: _temp(ROOM, "68", "°F")}
    assert indoor_temperature(FULL, states, C) == pytest.approx(20.0)
    states[ROOM] = State(ROOM, "unavailable")
    assert indoor_temperature(FULL, states, C) is None
    assert indoor_temperature(MIN, states, C) == 19.0


def test_climate_attributes_follow_system_unit() -> None:
    """Climate attributes are in the system unit."""
    state = _climate(current=68.0, target=70.0)
    assert indoor_temperature(MIN, {CLIMATE: state}, "°F") == pytest.approx(20.0)


def test_outdoor_sensor_then_weather_fallback() -> None:
    """Outdoor sensor first, weather temperature attribute as fallback."""
    weather = State(WEATHER, "cloudy", {"temperature": 41.0, "temperature_unit": "°F"})
    states = {CLIMATE: _climate(), OUTDOOR: _temp(OUTDOOR, "3.5"), WEATHER: weather}
    assert outdoor_temperature(FULL, states, C) == 3.5
    states[OUTDOOR] = State(OUTDOOR, "unknown")
    assert outdoor_temperature(FULL, states, C) == pytest.approx(5.0)
    assert outdoor_temperature(MIN, states, C) is None


def _climate_at(ts: datetime, state: str = "heat", **kwargs) -> State:
    return State(CLIMATE, state, climate_attrs(**kwargs), last_updated=ts, last_changed=ts)


def test_history_forward_fill_and_gaps() -> None:
    """History is forward-filled on the 5-min grid; unavailable gives a real gap."""
    rows = {
        CLIMATE: [
            _climate_at(T0, current=18.0, action="heating"),
            _climate_at(T0 + timedelta(minutes=30), current=19.0, action="idle"),
            _climate_at(T0 + timedelta(minutes=60), state="unavailable"),
            _climate_at(T0 + timedelta(minutes=75), current=18.5, action="idle"),
        ],
        OUTDOOR: [_temp(OUTDOOR, "4.0", ts=T0 + timedelta(seconds=30))],
    }
    inputs = ZoneInputs(climate=CLIMATE, outdoor_sensor=OUTDOOR)
    samples = reduce_states(inputs, C, [rows], T0 + timedelta(minutes=90))
    times = [s.ts for s in samples]
    assert times == sorted(times)
    gaps = [(b - a).total_seconds() / 60 for a, b in zip(times, times[1:], strict=False)]
    assert max(gaps) <= 5.0 + 1e-6
    first = samples[0]
    # The outdoor change 30 s later is coalesced into the first sample.
    assert (first.indoor_c, first.effort, first.outdoor_c) == (18.0, 1.0, 4.0)
    at_25 = next(s for s in samples if s.ts == T0 + timedelta(minutes=25))
    assert (at_25.indoor_c, at_25.effort) == (18.0, 1.0)
    at_45 = next(s for s in samples if s.ts >= T0 + timedelta(minutes=45))
    assert (at_45.indoor_c, at_45.effort) == (19.0, 0.0)
    gap = [s for s in samples if T0 + timedelta(minutes=60) <= s.ts < T0 + timedelta(minutes=75)]
    assert gap and all(s.indoor_c is None and s.excluded for s in gap)
    assert samples[-1].indoor_c == 18.5
    assert samples[-1].ts <= T0 + timedelta(minutes=90)


def test_history_across_windows() -> None:
    """State is carried across recorder windows."""
    first = {CLIMATE: [_climate_at(T0, current=18.0, action="idle")]}
    second = {OUTDOOR: [_temp(OUTDOOR, "2.0", ts=T0 + timedelta(minutes=20))]}
    inputs = ZoneInputs(climate=CLIMATE, outdoor_sensor=OUTDOOR)
    samples = reduce_states(inputs, C, [first, second], T0 + timedelta(minutes=30))
    last = samples[-1]
    assert last.indoor_c == 18.0
    assert last.outdoor_c == 2.0
