"""Turn Home Assistant states into model inputs (shared by live sampling and history).

Only standard Home Assistant attributes are used:

* ``climate``: state = HVAC mode, ``current_temperature``, ``temperature`` (target),
  ``hvac_action``;
* temperature sensors: numeric state with ``unit_of_measurement``;
* ``weather``: ``temperature`` attribute (+ ``temperature_unit``).

Rules:

* indoor temperature: the room sensor when configured (no mixing: an unavailable room
  sensor gives no indoor value), else the climate ``current_temperature``;
* heat source (optional): the boiler / heat pump really producing heat (on/off entity,
  or a numeric sensor such as the burner power: > 0 = on). The effort counts only while
  it is on: off → 0, unknown → not observed;
* heating effort: the effort sensor (% → 0..1, clamped) when it has a valid value, else
  ``hvac_action`` (``heating``/``preheating`` = 1, ``idle``/``off``/``fan``/``drying``/
  ``defrosting``/``cooling`` = 0, ``None`` or an unknown action = not observed). A climate
  without the ``hvac_action`` attribute falls back to: HVAC mode not ``off`` and target >
  current + 0.2 K → 1, else 0;
* outdoor temperature: the outdoor sensor when valid, else the weather entity's
  ``temperature`` attribute, else unknown;
* a sample is ``excluded`` from learning while the climate is unavailable/unknown or
  actively cooling (the model only describes heating).

All temperatures are converted to °C.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from homeassistant.components.climate import ATTR_CURRENT_TEMPERATURE, ATTR_HVAC_ACTION
from homeassistant.const import (
    ATTR_TEMPERATURE,
    ATTR_UNIT_OF_MEASUREMENT,
    STATE_OFF,
    STATE_ON,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    UnitOfTemperature,
)
from homeassistant.core import State
from homeassistant.util.unit_conversion import TemperatureConverter

from .const import (
    CONF_ARRIVAL_SENSOR,
    CONF_CLIMATE,
    CONF_EFFORT_SENSOR,
    CONF_HEAT_SOURCE,
    CONF_OUTDOOR_SENSOR,
    CONF_ROOM_SENSOR,
    CONF_WEATHER,
    COOLING_ACTIONS,
    FALLBACK_EFFORT_MARGIN_K,
    HEATING_ACTIONS,
    NON_HEATING_ACTIONS,
)
from .rtm_core.base import Sample

INVALID_STATES = frozenset({STATE_UNAVAILABLE, STATE_UNKNOWN, ""})
ATTR_WEATHER_TEMPERATURE_UNIT = "temperature_unit"
COOL_MODES = frozenset({"cool"})


@dataclass(frozen=True, slots=True)
class ZoneInputs:
    """Entity ids of one zone (from the config entry data)."""

    climate: str
    room_sensor: str | None = None
    effort_sensor: str | None = None
    heat_source: str | None = None
    outdoor_sensor: str | None = None
    weather: str | None = None
    arrival_sensor: str | None = None

    @classmethod
    def from_data(cls, data: Any) -> ZoneInputs:
        """Build from config entry data."""
        return cls(
            climate=data[CONF_CLIMATE],
            room_sensor=data.get(CONF_ROOM_SENSOR) or None,
            effort_sensor=data.get(CONF_EFFORT_SENSOR) or None,
            heat_source=data.get(CONF_HEAT_SOURCE) or None,
            outdoor_sensor=data.get(CONF_OUTDOOR_SENSOR) or None,
            weather=data.get(CONF_WEATHER) or None,
            arrival_sensor=data.get(CONF_ARRIVAL_SENSOR) or None,
        )

    @property
    def sample_entities(self) -> list[str]:
        """Entities whose states build a sample."""
        return [
            e
            for e in (
                self.climate,
                self.room_sensor,
                self.effort_sensor,
                self.heat_source,
                self.outdoor_sensor,
                self.weather,
            )
            if e
        ]

    @property
    def all_entities(self) -> list[str]:
        """Every configured input entity."""
        return [*self.sample_entities, *([self.arrival_sensor] if self.arrival_sensor else [])]


def to_float(value: Any) -> float | None:
    """Parse a finite float (None otherwise)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def to_celsius(value: Any, unit: str | None) -> float | None:
    """Parse a temperature and convert it to °C (unknown units are taken as °C)."""
    number = to_float(value)
    if number is None:
        return None
    if unit in (UnitOfTemperature.FAHRENHEIT, UnitOfTemperature.KELVIN):
        return TemperatureConverter.convert(number, unit, UnitOfTemperature.CELSIUS)
    return number


def _valid(state: State | None) -> bool:
    return state is not None and state.state not in INVALID_STATES


def sensor_celsius(state: State | None) -> float | None:
    """Temperature of a sensor state in °C."""
    if not _valid(state):
        return None
    assert state is not None
    return to_celsius(state.state, state.attributes.get(ATTR_UNIT_OF_MEASUREMENT))


def climate_current(state: State | None, system_unit: str) -> float | None:
    """``current_temperature`` of a climate in °C (attributes use the system unit)."""
    if not _valid(state):
        return None
    assert state is not None
    return to_celsius(state.attributes.get(ATTR_CURRENT_TEMPERATURE), system_unit)


def climate_target(state: State | None, system_unit: str) -> float | None:
    """Target ``temperature`` of a climate in °C."""
    if not _valid(state):
        return None
    assert state is not None
    return to_celsius(state.attributes.get(ATTR_TEMPERATURE), system_unit)


def weather_temperature(state: State | None, system_unit: str) -> float | None:
    """Current outdoor temperature reported by a weather entity in °C."""
    if not _valid(state):
        return None
    assert state is not None
    unit = state.attributes.get(ATTR_WEATHER_TEMPERATURE_UNIT) or system_unit
    return to_celsius(state.attributes.get(ATTR_TEMPERATURE), unit)


def effort_from_sensor(state: State | None) -> float | None:
    """Effort sensor in % → 0..1 (clamped), None when invalid."""
    if not _valid(state):
        return None
    assert state is not None
    value = to_float(state.state)
    if value is None:
        return None
    return min(1.0, max(0.0, value / 100.0))


def effort_from_climate(state: State | None, system_unit: str) -> float | None:
    """Effort derived from ``hvac_action`` (or the target/current fallback)."""
    if not _valid(state):
        return None
    assert state is not None
    if ATTR_HVAC_ACTION in state.attributes:
        action = state.attributes[ATTR_HVAC_ACTION]
        if action is None:
            return None  # reported but unknown right now: not observed
        if str(action) in HEATING_ACTIONS:
            return 1.0
        if str(action) in NON_HEATING_ACTIONS:
            return 0.0
        return None
    # The climate does not report hvac_action at all: infer it from mode and setpoint.
    if state.state == STATE_OFF:
        return 0.0
    target = climate_target(state, system_unit)
    current = climate_current(state, system_unit)
    if target is None or current is None:
        return 0.0
    return 1.0 if target > current + FALLBACK_EFFORT_MARGIN_K else 0.0


def climate_cooling(state: State | None) -> bool:
    """True while the climate is cooling (not described by the heating model)."""
    if not _valid(state):
        return False
    assert state is not None
    action = state.attributes.get(ATTR_HVAC_ACTION)
    if action is not None:
        return str(action) in COOLING_ACTIONS
    return state.state in COOL_MODES


def indoor_temperature(
    inputs: ZoneInputs, states: dict[str, State | None], system_unit: str
) -> float | None:
    """Indoor temperature in °C."""
    if inputs.room_sensor:
        return sensor_celsius(states.get(inputs.room_sensor))
    return climate_current(states.get(inputs.climate), system_unit)


def outdoor_temperature(
    inputs: ZoneInputs, states: dict[str, State | None], system_unit: str
) -> float | None:
    """Outdoor temperature in °C (sensor, else weather attribute)."""
    value: float | None = None
    if inputs.outdoor_sensor:
        value = sensor_celsius(states.get(inputs.outdoor_sensor))
    if value is None and inputs.weather:
        value = weather_temperature(states.get(inputs.weather), system_unit)
    return value


def heat_source_on(state: State | None) -> bool | None:
    """Heat source producing heat: on/off states, or a number > 0. None when unknown."""
    if not _valid(state):
        return None
    assert state is not None
    if state.state in (STATE_ON, STATE_OFF):
        return state.state == STATE_ON
    value = to_float(state.state)
    return None if value is None else value > 0


def heating_effort(
    inputs: ZoneInputs, states: dict[str, State | None], system_unit: str
) -> float | None:
    """Heating effort in [0, 1] (sensor, else derived from the climate), gated by the
    heat source when one is configured."""
    effort = _room_effort(inputs, states, system_unit)
    if inputs.heat_source is None or effort is None or effort == 0.0:
        return effort
    on = heat_source_on(states.get(inputs.heat_source))
    if on is None:
        return None
    return effort if on else 0.0


def _room_effort(
    inputs: ZoneInputs, states: dict[str, State | None], system_unit: str
) -> float | None:
    if inputs.effort_sensor:
        value = effort_from_sensor(states.get(inputs.effort_sensor))
        if value is not None:
            return value
    return effort_from_climate(states.get(inputs.climate), system_unit)


def build_sample(
    inputs: ZoneInputs, states: dict[str, State | None], ts: datetime, system_unit: str
) -> Sample:
    """Build one model sample from the current states of the zone's inputs."""
    climate = states.get(inputs.climate)
    return Sample(
        ts=ts,
        indoor_c=indoor_temperature(inputs, states, system_unit),
        outdoor_c=outdoor_temperature(inputs, states, system_unit),
        effort=heating_effort(inputs, states, system_unit),
        excluded=not _valid(climate) or climate_cooling(climate),
    )
