"""Constants for Room Thermal Model."""

from __future__ import annotations

from typing import Final

DOMAIN: Final = "room_thermal_model"
NAME: Final = "Room Thermal Model"

# Config entry data (one entry per zone)
CONF_ZONE_NAME: Final = "zone_name"
CONF_CLIMATE: Final = "climate_entity"
CONF_ROOM_SENSOR: Final = "room_temperature_sensor"  # optional
CONF_EFFORT_SENSOR: Final = "heating_effort_sensor"  # optional, %
CONF_OUTDOOR_SENSOR: Final = "outdoor_temperature_sensor"  # optional
CONF_WEATHER: Final = "weather_entity"  # optional, forecast (+ outdoor fallback)
CONF_ARRIVAL_SENSOR: Final = "arrival_sensor"  # optional timestamp sensor

# Options
CONF_TARGET_MODE: Final = "target_mode"
TARGET_MODE_FIXED: Final = "fixed"
TARGET_MODE_CLIMATE: Final = "climate"
CONF_TARGET_TEMPERATURE: Final = "target_temperature"
CONF_HALF_LIFE_DAYS: Final = "half_life_days"
CONF_HISTORY_DAYS: Final = "history_days"
CONF_UPDATE_INTERVAL_MINUTES: Final = "update_interval_minutes"

DEFAULT_TARGET_MODE: Final = TARGET_MODE_FIXED
DEFAULT_TARGET_TEMPERATURE: Final = 20.5
DEFAULT_HALF_LIFE_DAYS: Final = 21
DEFAULT_HISTORY_DAYS: Final = 30
DEFAULT_UPDATE_INTERVAL_MINUTES: Final = 5

# Services
SERVICE_RETRAIN: Final = "retrain"
SERVICE_RESET: Final = "reset"
SERVICE_EXPORT_DIAGNOSTICS: Final = "export_diagnostics"

# Storage
STORAGE_VERSION: Final = 1

# Entity keys (translation_key == key; unique_id = f"{entry_id}_{key}";
# entity_id = f"{platform}.{zone_slug}_{key}")
KEY_PREHEAT_MINUTES: Final = "preheat_minutes"
KEY_PREHEAT_START: Final = "preheat_start"
KEY_PREHEAT_NOW: Final = "preheat_now"
KEY_MAX_REACHABLE: Final = "max_reachable_temperature"
KEY_MODEL_MATURITY: Final = "model_maturity"
KEY_TIME_CONSTANT: Final = "time_constant"
KEY_HEATING_RATE: Final = "heating_rate"
KEY_DEAD_TIME: Final = "dead_time"

# Option ranges
TARGET_TEMPERATURE_MIN: Final = 5.0
TARGET_TEMPERATURE_MAX: Final = 30.0
TARGET_TEMPERATURE_STEP: Final = 0.5
HALF_LIFE_DAYS_MIN: Final = 3
HALF_LIFE_DAYS_MAX: Final = 120
HISTORY_DAYS_MIN: Final = 7
HISTORY_DAYS_MAX: Final = 90
UPDATE_INTERVAL_MIN: Final = 1
UPDATE_INTERVAL_MAX: Final = 30

# Runtime tuning
SAMPLE_INTERVAL_MINUTES: Final = 5  # live grid sample (independent of the update interval)
HISTORY_CHUNK_DAYS: Final = 7  # recorder query window
HISTORY_MIN_SPACING_S: Final = 60  # coalesce history events closer than this
SAVE_DELAY_SECONDS: Final = 600  # Store delayed save
SNAPSHOT_MIN_INTERVAL_S: Final = 600  # at most one model.to_dict() per 10 min (unless fitted)
FORECAST_CACHE_MINUTES: Final = 30
FORECAST_TYPES: Final = ("hourly", "twice_daily", "daily")
FALLBACK_EFFORT_MARGIN_K: Final = 0.2  # no hvac_action: heating when target > current + this
LEAD_MAX_ITERATIONS: Final = 10  # fixed-point iterations of the preheat lead time
LEAD_TOLERANCE_MINUTES: Final = 0.5  # lead time convergence threshold
UNREACHABLE_MARGIN_K: Final = 0.5  # unreachable target: aim this far below the maximum
PREHEAT_HOLD_MINUTES: Final = 30  # preheat_now stays on this long past the expected end

# hvac_action values (standard HVACAction strings)
HEATING_ACTIONS: Final = frozenset({"heating", "preheating"})
COOLING_ACTIONS: Final = frozenset({"cooling"})
NON_HEATING_ACTIONS: Final = frozenset({"idle", "off", "fan", "drying", "defrosting", "cooling"})

# Preheat status (attribute ``reason`` of the preheat_minutes sensor)
STATUS_OK: Final = "ok"
STATUS_LEARNING: Final = "learning"
STATUS_NO_TEMPERATURE: Final = "no_temperature"
STATUS_UNREACHABLE: Final = "unreachable"
STATUS_NO_OUTDOOR: Final = "no_outdoor"
STATUS_AT_TARGET: Final = "at_target"
STATUS_UNRELIABLE: Final = "unreliable"  # learnt parameters are physically implausible

# Service fields
ATTR_CONFIG_ENTRY_ID: Final = "config_entry_id"
