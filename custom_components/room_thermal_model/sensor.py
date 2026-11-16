"""Sensor platform."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from homeassistant.components.sensor import (
    ENTITY_ID_FORMAT,
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import PERCENTAGE, EntityCategory, UnitOfTemperature, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import (
    KEY_ARRIVAL_TEMPERATURE,
    KEY_DEAD_TIME,
    KEY_HEATING_RATE,
    KEY_MAX_REACHABLE,
    KEY_MODEL_MATURITY,
    KEY_PREHEAT_MINUTES,
    KEY_PREHEAT_START,
    KEY_TIME_CONSTANT,
)
from .coordinator import RTMConfigEntry, RTMCoordinator
from .entity import RTMEntity

PARALLEL_UPDATES = 0

UNIT_KELVIN_PER_HOUR = "K/h"


def _round(value: float | None, digits: int = 2) -> float | None:
    return round(value, digits) if value is not None else None


async def async_setup_entry(
    hass: HomeAssistant,
    entry: RTMConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up sensors."""
    coordinator = entry.runtime_data
    entities: list[SensorEntity] = [
        PreheatMinutesSensor(coordinator),
        MaxReachableSensor(coordinator),
        ModelMaturitySensor(coordinator),
        TimeConstantSensor(coordinator),
        HeatingRateSensor(coordinator),
        DeadTimeSensor(coordinator),
    ]
    if coordinator.inputs.arrival_sensor:
        entities.append(PreheatStartSensor(coordinator))
        entities.append(ArrivalTemperatureSensor(coordinator))
    async_add_entities(entities)


class RTMSensor(RTMEntity, SensorEntity):
    """Base zone sensor."""

    _key_name: str

    def __init__(self, coordinator: RTMCoordinator) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator, self._key_name, ENTITY_ID_FORMAT)


class PreheatMinutesSensor(RTMSensor):
    """Minutes of full heating needed to reach the target (unknown while learning)."""

    _key_name = KEY_PREHEAT_MINUTES
    _attr_native_unit_of_measurement = UnitOfTime.MINUTES
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 0

    @property
    def native_value(self) -> float | None:
        """Minutes (None when learning, unreachable or no temperature)."""
        zone = self.zone
        if zone is None or zone.preheat_minutes is None:
            return None
        return round(max(0.0, zone.preheat_minutes), 1)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Inputs and explanation of the estimate."""
        zone = self.zone
        if zone is None:
            return {}
        estimate = zone.preheat
        return {
            "target_c": _round(zone.target_c),
            "target_source": zone.target_source,
            "reachable": estimate.reachable if estimate else None,
            "start_c": _round(estimate.start_c if estimate else zone.indoor_c),
            "outdoor_used_c": _round(estimate.outdoor_used_c if estimate else zone.outdoor_c),
            "max_reachable_c": _round(estimate.max_reachable_c if estimate else None),
            "forecast_used": zone.forecast_used,
            "maturity": round(zone.quality.maturity * 100, 1),
            "reason": zone.status,
            "details": dict(estimate.reason) if estimate else {},
            "arrival": zone.arrival.isoformat() if zone.arrival else None,
            "coast_c": _round(zone.coast_c),
        }


class PreheatStartSensor(RTMSensor):
    """When heating must start to reach the target at the arrival."""

    _key_name = KEY_PREHEAT_START
    _attr_device_class = SensorDeviceClass.TIMESTAMP

    @property
    def native_value(self) -> datetime | None:
        """Arrival minus preheat minutes (unknown without a future arrival or estimate)."""
        zone = self.zone
        return zone.preheat_start if zone else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Arrival used."""
        zone = self.zone
        if zone is None:
            return {}
        return {
            "arrival": zone.arrival.isoformat() if zone.arrival else None,
            "reason": zone.status,
        }


class ArrivalTemperatureSensor(RTMSensor):
    """Room temperature at the arrival without preheating (not below a setback setpoint)."""

    _key_name = KEY_ARRIVAL_TEMPERATURE
    _attr_device_class = SensorDeviceClass.TEMPERATURE
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 1

    @property
    def native_value(self) -> float | None:
        """Coasted temperature at the arrival (unknown without arrival or reliable model)."""
        zone = self.zone
        return _round(zone.coast_c) if zone else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Arrival and setback floor used."""
        zone = self.zone
        if zone is None:
            return {}
        return {
            "arrival": zone.arrival.isoformat() if zone.arrival else None,
            "floor_c": _round(zone.floor_c),
            "target_c": _round(zone.target_c),
        }


class MaxReachableSensor(RTMSensor):
    """Equilibrium temperature at full heating effort."""

    _key_name = KEY_MAX_REACHABLE
    _attr_device_class = SensorDeviceClass.TEMPERATURE
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 1

    @property
    def native_value(self) -> float | None:
        """T_infinity (unknown before the first fit)."""
        zone = self.zone
        if zone is None or zone.preheat is None:
            return None
        return _round(zone.preheat.max_reachable_c)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Reliability of the model (unknown value when unreliable)."""
        zone = self.zone
        if zone is None or zone.parameters is None:
            return {}
        return {"reliable": zone.parameters.reliable}


class ModelMaturitySensor(RTMSensor):
    """How well the model knows the zone."""

    _key_name = KEY_MODEL_MATURITY
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 0

    @property
    def native_value(self) -> float | None:
        """Maturity in percent."""
        zone = self.zone
        return round(zone.quality.maturity * 100, 1) if zone else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Quality report."""
        zone = self.zone
        if zone is None:
            return {}
        quality = zone.quality
        return {
            "samples": quality.samples,
            "heating_hours": _round(quality.heating_hours, 1),
            "cooling_hours": _round(quality.cooling_hours, 1),
            "rmse_c": _round(quality.rmse_c, 3),
            "baseline_rmse_c": _round(quality.baseline_rmse_c, 3),
            "rmse_4h_c": _round(quality.rmse_4h_c, 3),
            "baseline_rmse_4h_c": _round(quality.baseline_rmse_4h_c, 3),
            "duration_mae_minutes": _round(quality.duration_mae_minutes, 1),
            "episodes": quality.episodes,
            "last_fit": zone.last_fit.isoformat() if zone.last_fit else None,
            "bootstrapping": self.coordinator.bootstrapping,
        }


class RTMParameterSensor(RTMSensor):
    """Diagnostic sensor exposing one learnt parameter."""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_state_class = SensorStateClass.MEASUREMENT

    def _value(self) -> float | None:
        raise NotImplementedError

    def _raw(self) -> float | None:
        zone = self.zone
        if zone is None or zone.parameters is None:
            return None
        value = self._value()
        return _round(value) if value is not None and value != float("inf") else None

    @property
    def native_value(self) -> float | None:
        """Unknown before the first fit and while the model is unreliable."""
        zone = self.zone
        if zone is None or zone.parameters is None or not zone.parameters.reliable:
            return None
        return self._raw()

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Reliability and, when unreliable, the raw learnt value (diagnostics)."""
        zone = self.zone
        if zone is None or zone.parameters is None:
            return {}
        attrs: dict[str, Any] = {"reliable": zone.parameters.reliable}
        if not zone.parameters.reliable:
            attrs["raw_value"] = self._raw()
        return attrs


class TimeConstantSensor(RTMParameterSensor):
    """Time constant tau = 1/b."""

    _key_name = KEY_TIME_CONSTANT
    _attr_native_unit_of_measurement = UnitOfTime.HOURS
    _attr_suggested_display_precision = 1

    def _value(self) -> float | None:
        assert self.zone is not None and self.zone.parameters is not None
        return self.zone.parameters.time_constant_hours


class HeatingRateSensor(RTMParameterSensor):
    """Heating gain a (K/h at full effort)."""

    _key_name = KEY_HEATING_RATE
    _attr_native_unit_of_measurement = UNIT_KELVIN_PER_HOUR
    _attr_suggested_display_precision = 2

    def _value(self) -> float | None:
        assert self.zone is not None and self.zone.parameters is not None
        return self.zone.parameters.heating_rate_k_per_h


class DeadTimeSensor(RTMParameterSensor):
    """Dead time d (radiator and valve lag)."""

    _key_name = KEY_DEAD_TIME
    _attr_native_unit_of_measurement = UnitOfTime.MINUTES
    _attr_suggested_display_precision = 0

    def _value(self) -> float | None:
        assert self.zone is not None and self.zone.parameters is not None
        return self.zone.parameters.dead_time_minutes
