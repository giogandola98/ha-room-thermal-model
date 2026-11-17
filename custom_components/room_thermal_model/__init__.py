"""Room Thermal Model: learns how fast each zone heats up and when to start heating."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import voluptuous as vol
from homeassistant.const import ATTR_ENTITY_ID, Platform
from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse, SupportsResponse
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.json import ExtendedJSONEncoder
from homeassistant.helpers.typing import ConfigType
from homeassistant.util import dt as dt_util

from .const import (
    ATTR_CONFIG_ENTRY_ID,
    DOMAIN,
    SERVICE_EXPORT_DIAGNOSTICS,
    SERVICE_RESET,
    SERVICE_RETRAIN,
)
from .coordinator import RTMConfigEntry, RTMCoordinator
from .diagnostics import async_get_config_entry_diagnostics
from .storage import RTMStorage

PLATFORMS: list[Platform] = [Platform.BINARY_SENSOR, Platform.SENSOR]

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

TARGET_SCHEMA = vol.Schema(
    {
        vol.Optional(ATTR_ENTITY_ID): cv.entity_ids,
        vol.Optional(ATTR_CONFIG_ENTRY_ID): cv.string,
    }
)

ATTR_REPLACE = "replace"
RETRAIN_SCHEMA = TARGET_SCHEMA.extend({vol.Optional(ATTR_REPLACE, default=False): cv.boolean})


def _zone_entities(hass: HomeAssistant, entry: RTMConfigEntry) -> set[str]:
    """Inputs of the zone plus the entities it provides."""
    registry = er.async_get(hass)
    own = {e.entity_id for e in er.async_entries_for_config_entry(registry, entry.entry_id)}
    return own | set(entry.runtime_data.inputs.all_entities)


def _target_entries(hass: HomeAssistant, call: ServiceCall) -> list[RTMConfigEntry]:
    """Zones targeted by a service call (no target = every loaded zone)."""
    entries: list[RTMConfigEntry] = hass.config_entries.async_loaded_entries(DOMAIN)
    if not entries:
        raise ServiceValidationError(translation_domain=DOMAIN, translation_key="not_loaded")
    entry_id: str | None = call.data.get(ATTR_CONFIG_ENTRY_ID)
    entity_ids: list[str] = call.data.get(ATTR_ENTITY_ID) or []
    if entry_id is None and not entity_ids:
        return entries
    selected: list[RTMConfigEntry] = []
    if entry_id is not None:
        match = [e for e in entries if e.entry_id == entry_id]
        if not match:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="unknown_zone",
                translation_placeholders={"target": entry_id},
            )
        selected += match
    for entity_id in entity_ids:
        match = [e for e in entries if entity_id in _zone_entities(hass, e)]
        if not match:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="unknown_zone",
                translation_placeholders={"target": entity_id},
            )
        selected += [e for e in match if e not in selected]
    return selected


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Register services (once, independent of config entries)."""

    async def _retrain(call: ServiceCall) -> ServiceResponse:
        result: dict[str, Any] = {}
        for entry in _target_entries(hass, call):
            coordinator: RTMCoordinator = entry.runtime_data
            samples = await coordinator.async_retrain(replace=bool(call.data.get(ATTR_REPLACE)))
            result[coordinator.zone_name] = {"samples": samples}
        return result

    async def _reset(call: ServiceCall) -> None:
        for entry in _target_entries(hass, call):
            await entry.runtime_data.async_reset()

    async def _export(call: ServiceCall) -> ServiceResponse:
        zones: dict[str, Any] = {}
        for entry in _target_entries(hass, call):
            zones[entry.entry_id] = await async_get_config_entry_diagnostics(hass, entry)
        stamp = dt_util.now().strftime("%Y%m%d_%H%M%S")
        path = hass.config.path(f"{DOMAIN}_diagnostics_{stamp}.json")
        payload = {"generated_at": dt_util.utcnow(), "zones": zones}

        def _write() -> None:
            Path(path).write_text(
                json.dumps(payload, cls=ExtendedJSONEncoder, indent=2), encoding="utf-8"
            )

        await hass.async_add_executor_job(_write)
        return {"path": path}

    hass.services.async_register(
        DOMAIN,
        SERVICE_RETRAIN,
        _retrain,
        schema=RETRAIN_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
    hass.services.async_register(DOMAIN, SERVICE_RESET, _reset, schema=TARGET_SCHEMA)
    hass.services.async_register(
        DOMAIN,
        SERVICE_EXPORT_DIAGNOSTICS,
        _export,
        schema=TARGET_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
    return True


async def async_setup_entry(hass: HomeAssistant, entry: RTMConfigEntry) -> bool:
    """Set up one zone."""
    coordinator = RTMCoordinator(hass, entry)
    await coordinator.async_config_entry_first_refresh()
    entry.runtime_data = coordinator
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: RTMConfigEntry) -> bool:
    """Unload a zone and persist its model."""
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unloaded:
        await entry.runtime_data.async_unload()
    return unloaded


async def async_remove_entry(hass: HomeAssistant, entry: RTMConfigEntry) -> None:
    """Delete the stored model when the zone is removed."""
    await RTMStorage(hass, entry.entry_id).async_remove()
