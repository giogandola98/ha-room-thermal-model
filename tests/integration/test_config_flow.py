"""Config, reconfigure and options flow."""

from __future__ import annotations

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.room_thermal_model.const import (
    CONF_ARRIVAL_SENSOR,
    CONF_CLIMATE,
    CONF_EFFORT_SENSOR,
    CONF_HALF_LIFE_DAYS,
    CONF_HEAT_SOURCE,
    CONF_HISTORY_DAYS,
    CONF_OUTDOOR_SENSOR,
    CONF_ROOM_SENSOR,
    CONF_TARGET_MODE,
    CONF_TARGET_TEMPERATURE,
    CONF_UPDATE_INTERVAL_MINUTES,
    CONF_WEATHER,
    CONF_ZONE_NAME,
    DOMAIN,
)

from .conftest import (
    ARRIVAL,
    CLIMATE,
    EFFORT,
    OUTDOOR,
    ROOM,
    WEATHER,
    ZONE,
    make_entry,
    set_inputs,
    setup_entry,
)


async def _start(hass: HomeAssistant):
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"
    return result


async def test_minimal_flow(hass: HomeAssistant) -> None:
    """Only name + climate: optional inputs absent, default options stored."""
    set_inputs(hass)
    result = await _start(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_ZONE_NAME: f"  {ZONE} ", CONF_CLIMATE: CLIMATE}
    )
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == ZONE
    assert result["data"] == {CONF_ZONE_NAME: ZONE, CONF_CLIMATE: CLIMATE}
    assert result["options"] == {
        CONF_TARGET_MODE: "fixed",
        CONF_TARGET_TEMPERATURE: 20.5,
        CONF_HALF_LIFE_DAYS: 21,
        CONF_HISTORY_DAYS: 30,
        CONF_UPDATE_INTERVAL_MINUTES: 5,
    }
    assert result["result"].unique_id == CLIMATE


async def test_full_flow(hass: HomeAssistant) -> None:
    """Every optional input, follow-the-climate target."""
    set_inputs(hass, full=True)
    hass.states.async_set(ARRIVAL, "unknown", {"device_class": "timestamp"})
    result = await _start(hass)
    user_input = {
        CONF_ZONE_NAME: ZONE,
        CONF_CLIMATE: CLIMATE,
        CONF_ROOM_SENSOR: ROOM,
        CONF_EFFORT_SENSOR: EFFORT,
        CONF_OUTDOOR_SENSOR: OUTDOOR,
        CONF_WEATHER: WEATHER,
        CONF_ARRIVAL_SENSOR: ARRIVAL,
        CONF_TARGET_MODE: "climate",
        CONF_TARGET_TEMPERATURE: 21.5,
    }
    result = await hass.config_entries.flow.async_configure(result["flow_id"], user_input)
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {
        k: v for k, v in user_input.items() if k not in (CONF_TARGET_MODE, CONF_TARGET_TEMPERATURE)
    }
    assert result["options"][CONF_TARGET_MODE] == "climate"
    assert result["options"][CONF_TARGET_TEMPERATURE] == 21.5


async def test_empty_name(hass: HomeAssistant) -> None:
    """A blank name is rejected."""
    set_inputs(hass)
    result = await _start(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_ZONE_NAME: "  ", CONF_CLIMATE: CLIMATE}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {CONF_ZONE_NAME: "name_required"}


async def test_already_configured(hass: HomeAssistant) -> None:
    """The same climate cannot be used by two zones; another climate can."""
    set_inputs(hass)
    make_entry(hass)
    result = await _start(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_ZONE_NAME: "Other", CONF_CLIMATE: CLIMATE}
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"

    hass.states.async_set("climate.bedroom", "heat", {"current_temperature": 18})
    result = await _start(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_ZONE_NAME: "Bedroom", CONF_CLIMATE: "climate.bedroom"}
    )
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert len(hass.config_entries.async_entries(DOMAIN)) == 2


async def test_options_flow(hass: HomeAssistant) -> None:
    """Options are stored with proper types and the entry reloads."""
    set_inputs(hass)
    entry = await setup_entry(hass, make_entry(hass))
    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] is FlowResultType.FORM
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            CONF_TARGET_MODE: "climate",
            CONF_TARGET_TEMPERATURE: 19.5,
            CONF_HALF_LIFE_DAYS: 14,
            CONF_HISTORY_DAYS: 60,
            CONF_UPDATE_INTERVAL_MINUTES: 10,
        },
    )
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options == {
        CONF_TARGET_MODE: "climate",
        CONF_TARGET_TEMPERATURE: 19.5,
        CONF_HALF_LIFE_DAYS: 14.0,
        CONF_HISTORY_DAYS: 60,
        CONF_UPDATE_INTERVAL_MINUTES: 10,
    }
    coordinator = entry.runtime_data
    assert coordinator.target_mode == "climate"
    assert coordinator.history_days == 60
    assert coordinator.model_config.half_life_days == 14.0
    assert coordinator.update_interval.total_seconds() == 600


async def test_reconfigure(hass: HomeAssistant) -> None:
    """Optional inputs can be added/removed later; the climate stays."""
    set_inputs(hass, full=True)
    entry = await setup_entry(hass, make_entry(hass, full=True))
    result = await entry.start_reconfigure_flow(hass)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reconfigure"
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_ZONE_NAME: "Lounge", CONF_OUTDOOR_SENSOR: OUTDOOR}
    )
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert entry.title == "Lounge"
    assert entry.data == {
        CONF_ZONE_NAME: "Lounge",
        CONF_CLIMATE: CLIMATE,
        CONF_OUTDOOR_SENSOR: OUTDOOR,
    }


async def test_reconfigure_adds_heat_source(hass: HomeAssistant) -> None:
    """The heat source is an optional input of the zone (stored in the entry data)."""
    set_inputs(hass)
    hass.states.async_set("binary_sensor.boiler_heating", "off")
    entry = await setup_entry(hass, make_entry(hass))
    result = await entry.start_reconfigure_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_ZONE_NAME: "Living", CONF_HEAT_SOURCE: "binary_sensor.boiler_heating"},
    )
    await hass.async_block_till_done()
    assert result["reason"] == "reconfigure_successful"
    assert entry.data[CONF_HEAT_SOURCE] == "binary_sensor.boiler_heating"
    assert "binary_sensor.boiler_heating" in entry.runtime_data.inputs.sample_entities
