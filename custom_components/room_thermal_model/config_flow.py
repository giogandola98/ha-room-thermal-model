"""Config, reconfigure and options flow (one config entry per zone)."""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.config_entries import ConfigFlow, ConfigFlowResult, OptionsFlowWithReload
from homeassistant.core import callback
from homeassistant.helpers.selector import (
    EntitySelector,
    EntitySelectorConfig,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
)

from .const import (
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
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_HISTORY_DAYS,
    DEFAULT_TARGET_MODE,
    DEFAULT_TARGET_TEMPERATURE,
    DEFAULT_UPDATE_INTERVAL_MINUTES,
    DOMAIN,
    HALF_LIFE_DAYS_MAX,
    HALF_LIFE_DAYS_MIN,
    HISTORY_DAYS_MAX,
    HISTORY_DAYS_MIN,
    TARGET_MODE_CLIMATE,
    TARGET_MODE_FIXED,
    TARGET_TEMPERATURE_MAX,
    TARGET_TEMPERATURE_MIN,
    TARGET_TEMPERATURE_STEP,
    UPDATE_INTERVAL_MAX,
    UPDATE_INTERVAL_MIN,
)

OPTIONAL_INPUTS = (
    CONF_ROOM_SENSOR,
    CONF_EFFORT_SENSOR,
    CONF_HEAT_SOURCE,
    CONF_OUTDOOR_SENSOR,
    CONF_WEATHER,
    CONF_ARRIVAL_SENSOR,
)


def _number(minimum: float, maximum: float, step: float, unit: str | None = None) -> NumberSelector:
    config = NumberSelectorConfig(min=minimum, max=maximum, step=step, mode=NumberSelectorMode.BOX)
    if unit:
        config["unit_of_measurement"] = unit
    return NumberSelector(config)


def _target_fields() -> dict[Any, Any]:
    return {
        vol.Required(CONF_TARGET_MODE, default=DEFAULT_TARGET_MODE): SelectSelector(
            SelectSelectorConfig(
                options=[TARGET_MODE_FIXED, TARGET_MODE_CLIMATE],
                translation_key=CONF_TARGET_MODE,
                mode=SelectSelectorMode.LIST,
            )
        ),
        vol.Required(CONF_TARGET_TEMPERATURE, default=DEFAULT_TARGET_TEMPERATURE): _number(
            TARGET_TEMPERATURE_MIN, TARGET_TEMPERATURE_MAX, TARGET_TEMPERATURE_STEP, "°C"
        ),
    }


def _input_fields(include_climate: bool) -> dict[Any, Any]:
    fields: dict[Any, Any] = {vol.Required(CONF_ZONE_NAME): TextSelector()}
    if include_climate:
        fields[vol.Required(CONF_CLIMATE)] = EntitySelector(EntitySelectorConfig(domain="climate"))
    fields.update(
        {
            vol.Optional(CONF_ROOM_SENSOR): EntitySelector(
                EntitySelectorConfig(domain="sensor", device_class="temperature")
            ),
            vol.Optional(CONF_EFFORT_SENSOR): EntitySelector(
                EntitySelectorConfig(domain=["sensor", "number", "input_number"])
            ),
            vol.Optional(CONF_HEAT_SOURCE): EntitySelector(
                EntitySelectorConfig(domain=["binary_sensor", "switch", "input_boolean", "sensor"])
            ),
            vol.Optional(CONF_OUTDOOR_SENSOR): EntitySelector(
                EntitySelectorConfig(domain="sensor", device_class="temperature")
            ),
            vol.Optional(CONF_WEATHER): EntitySelector(EntitySelectorConfig(domain="weather")),
            vol.Optional(CONF_ARRIVAL_SENSOR): EntitySelector(
                EntitySelectorConfig(domain="sensor", device_class="timestamp")
            ),
        }
    )
    return fields


def _clean(user_input: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Keep non-empty values of ``keys``."""
    return {k: user_input[k] for k in keys if user_input.get(k)}


class RoomThermalModelConfigFlow(ConfigFlow, domain=DOMAIN):
    """Add a zone."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Zone name, climate, optional inputs and target."""
        errors: dict[str, str] = {}
        if user_input is not None:
            name = str(user_input.get(CONF_ZONE_NAME, "")).strip()
            if not name:
                errors[CONF_ZONE_NAME] = "name_required"
            else:
                await self.async_set_unique_id(user_input[CONF_CLIMATE])
                self._abort_if_unique_id_configured()
                data = {
                    CONF_ZONE_NAME: name,
                    CONF_CLIMATE: user_input[CONF_CLIMATE],
                    **_clean(user_input, OPTIONAL_INPUTS),
                }
                options = {
                    CONF_TARGET_MODE: user_input.get(CONF_TARGET_MODE, DEFAULT_TARGET_MODE),
                    CONF_TARGET_TEMPERATURE: float(
                        user_input.get(CONF_TARGET_TEMPERATURE, DEFAULT_TARGET_TEMPERATURE)
                    ),
                    CONF_HALF_LIFE_DAYS: DEFAULT_HALF_LIFE_DAYS,
                    CONF_HISTORY_DAYS: DEFAULT_HISTORY_DAYS,
                    CONF_UPDATE_INTERVAL_MINUTES: DEFAULT_UPDATE_INTERVAL_MINUTES,
                }
                return self.async_create_entry(title=name, data=data, options=options)
        schema = vol.Schema({**_input_fields(include_climate=True), **_target_fields()})
        return self.async_show_form(
            step_id="user",
            data_schema=self.add_suggested_values_to_schema(schema, user_input or {}),
            errors=errors,
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Change the zone name and the optional inputs (the climate stays)."""
        entry = self._get_reconfigure_entry()
        errors: dict[str, str] = {}
        if user_input is not None:
            name = str(user_input.get(CONF_ZONE_NAME, "")).strip()
            if not name:
                errors[CONF_ZONE_NAME] = "name_required"
            else:
                data = {
                    CONF_ZONE_NAME: name,
                    CONF_CLIMATE: entry.data[CONF_CLIMATE],
                    **_clean(user_input, OPTIONAL_INPUTS),
                }
                return self.async_update_reload_and_abort(entry, title=name, data=data)
        schema = vol.Schema(_input_fields(include_climate=False))
        return self.async_show_form(
            step_id="reconfigure",
            data_schema=self.add_suggested_values_to_schema(schema, user_input or entry.data),
            errors=errors,
            description_placeholders={"climate": entry.data[CONF_CLIMATE]},
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: Any) -> RoomThermalModelOptionsFlow:
        """Return the options flow."""
        return RoomThermalModelOptionsFlow()


class RoomThermalModelOptionsFlow(OptionsFlowWithReload):
    """Target and learning options (the entry reloads on change)."""

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Manage options."""
        if user_input is not None:
            return self.async_create_entry(
                data={
                    CONF_TARGET_MODE: user_input[CONF_TARGET_MODE],
                    CONF_TARGET_TEMPERATURE: float(user_input[CONF_TARGET_TEMPERATURE]),
                    CONF_HALF_LIFE_DAYS: float(user_input[CONF_HALF_LIFE_DAYS]),
                    CONF_HISTORY_DAYS: int(user_input[CONF_HISTORY_DAYS]),
                    CONF_UPDATE_INTERVAL_MINUTES: int(user_input[CONF_UPDATE_INTERVAL_MINUTES]),
                }
            )
        current = {
            CONF_TARGET_MODE: DEFAULT_TARGET_MODE,
            CONF_TARGET_TEMPERATURE: DEFAULT_TARGET_TEMPERATURE,
            CONF_HALF_LIFE_DAYS: DEFAULT_HALF_LIFE_DAYS,
            CONF_HISTORY_DAYS: DEFAULT_HISTORY_DAYS,
            CONF_UPDATE_INTERVAL_MINUTES: DEFAULT_UPDATE_INTERVAL_MINUTES,
            **self.config_entry.options,
        }
        schema = vol.Schema(
            {
                **_target_fields(),
                vol.Required(CONF_HALF_LIFE_DAYS): _number(
                    HALF_LIFE_DAYS_MIN, HALF_LIFE_DAYS_MAX, 1, "d"
                ),
                vol.Required(CONF_HISTORY_DAYS): _number(
                    HISTORY_DAYS_MIN, HISTORY_DAYS_MAX, 1, "d"
                ),
                vol.Required(CONF_UPDATE_INTERVAL_MINUTES): _number(
                    UPDATE_INTERVAL_MIN, UPDATE_INTERVAL_MAX, 1, "min"
                ),
            }
        )
        return self.async_show_form(
            step_id="init", data_schema=self.add_suggested_values_to_schema(schema, current)
        )
