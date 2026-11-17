"""Fixtures for the Home Assistant integration tests."""

from __future__ import annotations

import sys
from collections.abc import Generator, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from homeassistant.core import HomeAssistant  # noqa: E402
from pytest_homeassistant_custom_component.common import MockConfigEntry  # noqa: E402

from custom_components.room_thermal_model.const import (  # noqa: E402
    CONF_ARRIVAL_SENSOR,
    CONF_CLIMATE,
    CONF_EFFORT_SENSOR,
    CONF_OUTDOOR_SENSOR,
    CONF_ROOM_SENSOR,
    CONF_TARGET_MODE,
    CONF_TARGET_TEMPERATURE,
    CONF_WEATHER,
    CONF_ZONE_NAME,
    DOMAIN,
)
from custom_components.room_thermal_model.rtm_core.base import (  # noqa: E402
    ForecastPoint,
    ModelConfig,
    PreheatEstimate,
    QualityReport,
    Sample,
    ThermalModel,
    ThermalParameters,
)

CLIMATE = "climate.living_room"
ROOM = "sensor.living_room_temperature"
EFFORT = "sensor.living_room_valve"
OUTDOOR = "sensor.outdoor_temperature"
WEATHER = "weather.home"
ARRIVAL = "sensor.house_next_return_eta"
ZONE = "Living Room"
SLUG = "living_room"


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Enable custom integrations in every test."""


@pytest.fixture(autouse=True)
def no_history() -> Generator[Any]:
    """Never hit the recorder: the bootstrap gets an empty history by default."""
    with patch(
        "custom_components.room_thermal_model.coordinator.async_load_zone_history",
        return_value=[],
    ) as mock:
        yield mock


def climate_attrs(
    current: float | None = 19.0, target: float | None = 21.0, action: Any = "idle", **extra: Any
) -> dict[str, Any]:
    """Attributes of a heating climate entity (``action=...`` omitted when ``MISSING``)."""
    attrs: dict[str, Any] = {
        "current_temperature": current,
        "temperature": target,
        "friendly_name": "Living room",
        **extra,
    }
    if action is not MISSING:
        attrs["hvac_action"] = action
    return attrs


MISSING = object()


def set_inputs(hass: HomeAssistant, full: bool = False) -> None:
    """Create the input entities."""
    hass.states.async_set(CLIMATE, "heat", climate_attrs())
    if full:
        hass.states.async_set(
            ROOM, "19.5", {"unit_of_measurement": "°C", "device_class": "temperature"}
        )
        hass.states.async_set(EFFORT, "40", {"unit_of_measurement": "%"})
        hass.states.async_set(
            OUTDOOR, "5.0", {"unit_of_measurement": "°C", "device_class": "temperature"}
        )
        hass.states.async_set(WEATHER, "cloudy", {"temperature": 6.0, "temperature_unit": "°C"})


def make_entry(
    hass: HomeAssistant, full: bool = False, arrival: bool = False, **options: Any
) -> MockConfigEntry:
    """Config entry of the test zone."""
    data: dict[str, Any] = {CONF_ZONE_NAME: ZONE, CONF_CLIMATE: CLIMATE}
    if full:
        data |= {
            CONF_ROOM_SENSOR: ROOM,
            CONF_EFFORT_SENSOR: EFFORT,
            CONF_OUTDOOR_SENSOR: OUTDOOR,
            CONF_WEATHER: WEATHER,
        }
    if arrival:
        data[CONF_ARRIVAL_SENSOR] = ARRIVAL
    entry = MockConfigEntry(
        domain=DOMAIN,
        title=ZONE,
        unique_id=CLIMATE,
        data=data,
        options={CONF_TARGET_MODE: "fixed", CONF_TARGET_TEMPERATURE: 20.5, **options},
    )
    entry.add_to_hass(hass)
    return entry


async def setup_entry(hass: HomeAssistant, entry: MockConfigEntry) -> MockConfigEntry:
    """Set up an entry and wait for the background tasks."""
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    return entry


@dataclass
class FakeModel(ThermalModel):
    """Deterministic model double (independent of the real RC implementation).

    preheat: ``minutes`` = (target - start) / rate * 60 with rate ``a`` K/h; with
    ``reachable`` False the maximum reachable temperature is 19.5 °C.
    """

    config: ModelConfig = field(default_factory=ModelConfig)
    samples: list[Sample] = field(default_factory=list)
    history: list[Sample] = field(default_factory=list)
    params: ThermalParameters | None = None
    fits: int = 0
    due: bool = False
    reachable: bool = True
    preheat_calls: list[dict[str, Any]] = field(default_factory=list)
    resets: int = 0
    parameter_resets: int = 0
    history_calls: list[bool] = field(default_factory=list)  # ``replace`` of each call

    def load_history(self, samples: Iterable[Sample], replace: bool = False) -> None:
        self.history_calls.append(replace)
        if replace:
            self.history = []
        self.history.extend(samples)

    def add_sample(self, sample: Sample) -> None:
        self.samples.append(sample)

    def fit_due(self, now: datetime) -> bool:
        return self.due

    def fit(self, now: datetime) -> dict[str, Any]:
        self.fits += 1
        self.due = False
        self.params = ThermalParameters(3.0, 0.125, 0.0, 30.0)
        return {"ok": True}

    def parameters(self) -> ThermalParameters | None:
        return self.params

    def preheat(
        self,
        now: datetime,
        start_c: float,
        target_c: float,
        outdoor_c: float | None,
        forecast: Sequence[ForecastPoint] = (),
        max_c: float | None = None,
    ) -> PreheatEstimate:
        self.preheat_calls.append(
            {
                "now": now,
                "start": start_c,
                "target": target_c,
                "outdoor": outdoor_c,
                "forecast": forecast,
                "max_c": max_c,
            }
        )
        assert self.params is not None
        minutes = max(0.0, (target_c - start_c) / self.params.heating_rate_k_per_h * 60)
        max_reachable = 24.0 if self.reachable else 19.5
        reachable = target_c <= max_reachable
        return PreheatEstimate(
            minutes=minutes if reachable else None,
            reachable=reachable,
            target_c=target_c,
            start_c=start_c,
            outdoor_used_c=forecast[0].outdoor_c if forecast else outdoor_c,
            max_reachable_c=max_reachable,
            reason={
                "iterations": 1,
                "outdoor_source": "forecast" if forecast else "current",
            },
        )

    def coast(
        self,
        now: datetime,
        start_c: float,
        minutes: float,
        outdoor_c: float | None,
        forecast: Sequence[ForecastPoint] = (),
        floor_c: float | None = None,
    ) -> float | None:
        temp = start_c - minutes / 60
        return max(temp, floor_c) if floor_c is not None and floor_c < start_c else temp

    def quality(self) -> QualityReport:
        return QualityReport(
            maturity=0.5 if self.params else 0.0,
            samples=len(self.samples) + len(self.history),
            heating_hours=1.0,
            cooling_hours=2.0,
            rmse_c=0.1 if self.params else None,
            baseline_rmse_c=0.3 if self.params else None,
            duration_mae_minutes=None,
        )

    def reset_parameters(self) -> None:
        self.parameter_resets += 1
        self.params = None

    def reset(self) -> None:
        self.resets += 1
        self.samples.clear()
        self.history.clear()
        self.params = None

    def to_dict(self) -> dict[str, Any]:
        return {"fake": True, "samples": len(self.samples), "fitted": self.params is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any], config: ModelConfig) -> FakeModel:
        if not data.get("fake"):
            raise ValueError("not a fake model")
        model = cls(config=config)
        if data.get("fitted"):
            model.params = ThermalParameters(3.0, 0.125, 0.0, 30.0)
        return model


@pytest.fixture
def fake_model() -> Generator[None]:
    """Replace the real model with :class:`FakeModel`."""
    with patch(
        "custom_components.room_thermal_model.coordinator.create_model",
        side_effect=lambda config: FakeModel(config=config),
    ):
        yield
