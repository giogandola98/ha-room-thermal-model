"""Setup/unload, entities, services, diagnostics, storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import STATE_UNKNOWN
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.room_thermal_model.const import DOMAIN
from custom_components.room_thermal_model.diagnostics import (
    async_get_config_entry_diagnostics,
)
from custom_components.room_thermal_model.rtm_core.base import Sample

from .conftest import (
    ARRIVAL,
    CLIMATE,
    OUTDOOR,
    SLUG,
    ZONE,
    FakeModel,
    make_entry,
    set_inputs,
    setup_entry,
)

BASE_SENSORS = (
    "preheat_minutes",
    "max_reachable_temperature",
    "model_maturity",
    "time_constant",
    "heating_rate",
    "dead_time",
)


async def test_setup_unload_real_model(hass: HomeAssistant) -> None:
    """With the real model: entities exist, preheat unknown while learning."""
    set_inputs(hass)
    entry = await setup_entry(hass, make_entry(hass))
    assert entry.state is ConfigEntryState.LOADED
    state = hass.states.get(f"sensor.{SLUG}_preheat_minutes")
    assert state is not None
    assert state.state == STATE_UNKNOWN
    assert state.attributes["reason"] == "learning"
    assert state.attributes["target_c"] == 20.5
    for key in ("time_constant", "heating_rate", "dead_time", "max_reachable_temperature"):
        assert hass.states.get(f"sensor.{SLUG}_{key}").state == STATE_UNKNOWN
    assert hass.states.get(f"sensor.{SLUG}_model_maturity").state not in ("unavailable",)
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert entry.state is ConfigEntryState.NOT_LOADED


async def test_entity_ids_without_arrival(hass: HomeAssistant, fake_model: None) -> None:
    """Explicit entity ids / unique ids; arrival entities absent without the sensor."""
    set_inputs(hass)
    entry = await setup_entry(hass, make_entry(hass))
    registry = er.async_get(hass)
    entities = er.async_entries_for_config_entry(registry, entry.entry_id)
    assert {e.entity_id for e in entities} == {f"sensor.{SLUG}_{k}" for k in BASE_SENSORS}
    for entity in entities:
        key = entity.entity_id.split(".", 1)[1].removeprefix(f"{SLUG}_")
        assert entity.unique_id == f"{entry.entry_id}_{key}"
        assert entity.translation_key == key
    diag = {"time_constant", "heating_rate", "dead_time"}
    for entity in entities:
        is_diag = entity.unique_id.removeprefix(f"{entry.entry_id}_") in diag
        assert (entity.entity_category is not None) == is_diag
    devices = dr.async_entries_for_config_entry(dr.async_get(hass), entry.entry_id)
    device = next((d for d in devices if (DOMAIN, entry.entry_id) in d.identifiers), None)
    assert device is not None
    assert device.name == ZONE
    assert device.entry_type is dr.DeviceEntryType.SERVICE
    assert hass.states.get(f"sensor.{SLUG}_preheat_start") is None
    assert hass.states.get(f"binary_sensor.{SLUG}_preheat_now") is None
    assert hass.states.get(f"sensor.{SLUG}_arrival_temperature") is None


async def test_entity_ids_with_arrival(hass: HomeAssistant, fake_model: None) -> None:
    """Arrival sensor configured: preheat start + preheat now exist."""
    set_inputs(hass)
    hass.states.async_set(ARRIVAL, STATE_UNKNOWN, {"device_class": "timestamp"})
    entry = await setup_entry(hass, make_entry(hass, arrival=True))
    registry = er.async_get(hass)
    assert registry.async_get(f"sensor.{SLUG}_preheat_start").unique_id == (
        f"{entry.entry_id}_preheat_start"
    )
    assert registry.async_get(f"binary_sensor.{SLUG}_preheat_now").unique_id == (
        f"{entry.entry_id}_preheat_now"
    )
    assert hass.states.get(f"binary_sensor.{SLUG}_preheat_now").state == "off"
    assert hass.states.get(f"sensor.{SLUG}_preheat_start").state == STATE_UNKNOWN
    assert registry.async_get(f"sensor.{SLUG}_arrival_temperature").unique_id == (
        f"{entry.entry_id}_arrival_temperature"
    )


async def test_two_zones(hass: HomeAssistant, fake_model: None) -> None:
    """Several zones coexist with their own entities."""
    set_inputs(hass)
    hass.states.async_set("climate.bedroom", "heat", {"current_temperature": 18.0})
    await setup_entry(hass, make_entry(hass))
    other = MockConfigEntry(
        domain=DOMAIN,
        title="Bedroom",
        unique_id="climate.bedroom",
        data={"zone_name": "Bedroom", "climate_entity": "climate.bedroom"},
        options={},
    )
    other.add_to_hass(hass)
    await setup_entry(hass, other)
    assert hass.states.get("sensor.bedroom_preheat_minutes") is not None
    assert hass.states.get(f"sensor.{SLUG}_preheat_minutes") is not None


async def test_bootstrap_loads_history(
    hass: HomeAssistant, fake_model: None, no_history: Any
) -> None:
    """First start merges recorder history (replace=False) and fits right away."""
    set_inputs(hass)
    from datetime import UTC, datetime

    history = [Sample(datetime(2026, 1, 1, tzinfo=UTC), 19.0, 5.0, 1.0)]
    no_history.return_value = history
    entry = await setup_entry(hass, make_entry(hass))
    coordinator = entry.runtime_data
    model: FakeModel = coordinator.model
    assert model.history == history
    assert model.fits == 1
    assert coordinator.bootstrapped
    assert no_history.call_args.args[2] == 30.0
    state = hass.states.get(f"sensor.{SLUG}_preheat_minutes")
    assert state.state == "30.0"  # (20.5 - 19) K at 3 K/h


async def test_services(hass: HomeAssistant, fake_model: None, no_history: Any, tmp_path) -> None:
    """retrain / reset / export_diagnostics with and without targets."""
    set_inputs(hass, full=True)
    entry = await setup_entry(hass, make_entry(hass, full=True))
    coordinator = entry.runtime_data
    calls_before = no_history.call_count
    coordinator.model.due = True  # the next retrain fits

    response = await hass.services.async_call(
        DOMAIN, "retrain", {"entity_id": OUTDOOR}, blocking=True, return_response=True
    )
    assert response == {ZONE: {"samples": 0}}
    assert no_history.call_count == calls_before + 1

    await hass.services.async_call(
        DOMAIN, "retrain", {"entity_id": f"sensor.{SLUG}_model_maturity"}, blocking=True
    )
    await hass.services.async_call(DOMAIN, "retrain", {}, blocking=True)
    assert no_history.call_count == calls_before + 3

    with pytest.raises(ServiceValidationError):
        await hass.services.async_call(
            DOMAIN, "retrain", {"entity_id": "sensor.unrelated"}, blocking=True
        )
    with pytest.raises(ServiceValidationError):
        await hass.services.async_call(DOMAIN, "reset", {"config_entry_id": "nope"}, blocking=True)

    assert coordinator.model.parameters() is not None
    await hass.services.async_call(
        DOMAIN, "reset", {"config_entry_id": entry.entry_id}, blocking=True
    )
    assert coordinator.model.parameters() is None
    assert coordinator.last_fit is None
    assert hass.states.get(f"sensor.{SLUG}_preheat_minutes").attributes["reason"] == "learning"

    hass.config.config_dir = str(tmp_path)
    response = await hass.services.async_call(
        DOMAIN, "export_diagnostics", {"entity_id": CLIMATE}, blocking=True, return_response=True
    )
    path = Path(response["path"])
    assert path.parent == tmp_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert entry.entry_id in payload["zones"]
    assert payload["zones"][entry.entry_id]["entry"]["data"]["climate_entity"] == CLIMATE


async def test_services_not_loaded(hass: HomeAssistant) -> None:
    """Services exist without zones but refuse to run."""
    from homeassistant.setup import async_setup_component

    assert await async_setup_component(hass, DOMAIN, {})
    with pytest.raises(ServiceValidationError):
        await hass.services.async_call(DOMAIN, "retrain", {}, blocking=True)


async def test_diagnostics(hass: HomeAssistant, fake_model: None) -> None:
    """Diagnostics contain inputs, zone result, model snapshot."""
    set_inputs(hass, full=True)
    entry = await setup_entry(hass, make_entry(hass, full=True))
    diag = await async_get_config_entry_diagnostics(hass, entry)
    assert diag["entry"]["data"]["climate_entity"] == CLIMATE
    assert CLIMATE in diag["inputs"]
    assert diag["zone"]["status"] == "learning"
    assert diag["zone"]["indoor_c"] == 19.5  # room sensor overrides the climate
    assert diag["zone"]["effort"] == 0.4  # effort sensor 40 %
    assert diag["model"]["fake"] is True
    json.dumps(diag, default=str)


async def test_store_round_trip(
    hass: HomeAssistant, fake_model: None, hass_storage: dict[str, Any]
) -> None:
    """Unload saves the model; the next setup restores it and skips the bootstrap."""
    set_inputs(hass)
    entry = await setup_entry(hass, make_entry(hass))
    coordinator = entry.runtime_data
    await coordinator.async_retrain()  # fit (force) → params
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    key = f"{DOMAIN}.{entry.entry_id}"
    stored = hass_storage[key]
    assert stored["version"] == 1
    assert stored["minor_version"] == 3
    assert stored["data"]["schema"] == 1
    assert stored["data"]["history_merged"] == 2
    assert stored["data"]["bootstrapped"] is True
    assert stored["data"]["model"]["fake"] is True

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    restored = entry.runtime_data
    assert restored is not coordinator
    assert restored.bootstrapped
    assert restored.last_fit == coordinator.last_fit


async def test_unreadable_store_discarded(
    hass: HomeAssistant, fake_model: None, hass_storage: dict[str, Any]
) -> None:
    """A model that cannot be restored is replaced and bootstrapped again."""
    set_inputs(hass)
    entry = make_entry(hass)
    hass_storage[f"{DOMAIN}.{entry.entry_id}"] = {
        "version": 1,
        "minor_version": 1,
        "key": f"{DOMAIN}.{entry.entry_id}",
        "data": {"schema": 1, "model": {"bogus": 1}, "last_fit": None, "bootstrapped": True},
    }
    await setup_entry(hass, entry)
    assert entry.state is ConfigEntryState.LOADED
    assert entry.runtime_data.bootstrapped  # bootstrap ran again


async def test_unknown_model_version_bootstraps(
    hass: HomeAssistant, hass_storage: dict[str, Any], no_history: Any
) -> None:
    """A stored real model of an unknown version is discarded and the history re-read."""
    set_inputs(hass)
    entry = make_entry(hass)
    hass_storage[f"{DOMAIN}.{entry.entry_id}"] = {
        "version": 1,
        "minor_version": 1,
        "key": f"{DOMAIN}.{entry.entry_id}",
        "data": {
            "schema": 1,
            "model": {"version": 999, "params": None},
            "last_fit": None,
            "bootstrapped": True,
        },
    }
    await setup_entry(hass, entry)
    assert entry.state is ConfigEntryState.LOADED
    assert entry.runtime_data.snapshot is not None  # replaced by the fresh model's snapshot
    assert entry.runtime_data.snapshot.get("version") != 999
    no_history.assert_called_once()


async def test_changed_inputs_keep_data_reset_parameters(
    hass: HomeAssistant, fake_model: None, hass_storage: dict[str, Any], no_history: Any
) -> None:
    """Other inputs: the data stays, the parameters are refitted, the history re-merged."""
    set_inputs(hass)
    entry = make_entry(hass)
    key = f"{DOMAIN}.{entry.entry_id}"
    document = {
        "schema": 1,
        "model": {"fake": True, "fitted": True},
        "last_fit": None,
        "bootstrapped": True,
        "inputs": [CLIMATE],
        "history_merged": 2,
    }
    stored = {"version": 1, "minor_version": 2, "key": key, "data": document}

    hass_storage[key] = stored
    await setup_entry(hass, entry)
    no_history.assert_not_called()  # same inputs, already merged: restored as is
    assert entry.runtime_data.model.params is not None
    assert entry.runtime_data.model.parameter_resets == 0
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert hass_storage[key]["data"]["inputs"] == [CLIMATE]

    document["inputs"] = [CLIMATE, "sensor.old_room"]
    hass_storage[key] = stored
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    model = entry.runtime_data.model
    no_history.assert_called_once()
    assert no_history.call_count == 1
    assert model.params is None
    assert model.parameter_resets == 1 and model.resets == 0  # data kept
    assert entry.runtime_data.history_merged == 2


async def test_store_v1_1_is_merged_with_history_once(
    hass: HomeAssistant, fake_model: None, hass_storage: dict[str, Any], no_history: Any
) -> None:
    """A 1.1 document loads, the recorder history is merged once, then never again."""
    set_inputs(hass)
    entry = make_entry(hass)
    key = f"{DOMAIN}.{entry.entry_id}"
    document = {
        "schema": 1,
        "model": {"fake": True, "fitted": True},
        "last_fit": None,
        "bootstrapped": True,
        "inputs": [CLIMATE],
    }
    hass_storage[key] = {"version": 1, "minor_version": 1, "key": key, "data": document}
    from datetime import UTC, datetime

    no_history.return_value = [
        Sample(datetime(2026, 1, 1, tzinfo=UTC), 19.0, 5.0, 1.0, False, 20.0)
    ]
    await setup_entry(hass, entry)
    coordinator = entry.runtime_data
    assert no_history.call_count == 1  # one-off merge
    assert coordinator.model.params is not None  # learning kept
    assert coordinator.model.resets == 0
    assert coordinator.model.history_calls == [False]  # merged, not replaced
    assert coordinator.history_merged == 2
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert hass_storage[key]["minor_version"] == 3
    assert hass_storage[key]["data"]["history_merged"] == 2

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert no_history.call_count == 1  # not again


async def test_store_v1_1_flag_set_without_recorder_data(
    hass: HomeAssistant, fake_model: None, hass_storage: dict[str, Any], no_history: Any
) -> None:
    """No recorder history (retention, recorder off): the one-off flag is set anyway."""
    set_inputs(hass)
    entry = make_entry(hass)
    key = f"{DOMAIN}.{entry.entry_id}"
    document = {"schema": 1, "model": {"fake": True, "fitted": True}, "bootstrapped": True}
    hass_storage[key] = {"version": 1, "minor_version": 1, "key": key, "data": document}
    await setup_entry(hass, entry)
    assert no_history.call_count == 1
    assert entry.runtime_data.history_merged == 2
    assert entry.runtime_data.model.params is not None


async def test_remove_entry_deletes_store(
    hass: HomeAssistant, fake_model: None, hass_storage: dict[str, Any]
) -> None:
    """Removing the zone removes its stored model."""
    set_inputs(hass)
    entry = await setup_entry(hass, make_entry(hass))
    await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert f"{DOMAIN}.{entry.entry_id}" in hass_storage
    await hass.config_entries.async_remove(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert f"{DOMAIN}.{entry.entry_id}" not in hass_storage


async def test_real_model_end_to_end(hass: HomeAssistant, no_history: Any) -> None:
    """Real RC model bootstrapped from (synthetic) history gives a preheat estimate."""
    from datetime import timedelta

    from homeassistant.util import dt as dt_util

    from custom_components.room_thermal_model.rtm_core.synthetic import (
        generate,
        radiator_room,
    )

    run = await hass.async_add_executor_job(
        lambda: generate(
            radiator_room(), days=10, start=dt_util.utcnow() - timedelta(days=10), sample_minutes=5
        )
    )
    no_history.return_value = run.samples
    set_inputs(hass)
    entry = await setup_entry(hass, make_entry(hass))
    coordinator = entry.runtime_data
    assert coordinator.model.parameters() is not None
    state = hass.states.get(f"sensor.{SLUG}_preheat_minutes")
    assert state.attributes["reason"] in ("ok", "unreachable", "at_target")
    if state.attributes["reason"] == "ok":
        assert 0 < float(state.state) < 24 * 60
    assert float(hass.states.get(f"sensor.{SLUG}_time_constant").state) > 0
    assert float(hass.states.get(f"sensor.{SLUG}_heating_rate").state) > 0
    assert hass.states.get(f"sensor.{SLUG}_model_maturity").attributes["last_fit"]
    diag = await async_get_config_entry_diagnostics(hass, entry)
    json.dumps(diag, default=str)


async def test_store_v1_2_loads_unlatched(
    hass: HomeAssistant, fake_model: None, hass_storage: dict[str, Any], no_history: Any
) -> None:
    """A 1.2 document (no latch) loads as not latched, without a history re-merge."""
    set_inputs(hass)
    entry = make_entry(hass)
    key = f"{DOMAIN}.{entry.entry_id}"
    document = {
        "schema": 1,
        "model": {"fake": True, "fitted": True},
        "last_fit": None,
        "bootstrapped": True,
        "inputs": [CLIMATE],
        "history_merged": 2,
    }
    hass_storage[key] = {"version": 1, "minor_version": 2, "key": key, "data": document}
    await setup_entry(hass, entry)
    assert entry.runtime_data.latch is None
    assert entry.runtime_data.model.params is not None
    no_history.assert_not_called()
