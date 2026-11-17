"""Binary sensor platform: preheat now (only with an arrival sensor)."""

from __future__ import annotations

from typing import Any

from homeassistant.components.binary_sensor import ENTITY_ID_FORMAT, BinarySensorEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util import dt as dt_util

from .const import KEY_PREHEAT_NOW
from .coordinator import RTMConfigEntry, RTMCoordinator
from .entity import RTMEntity

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    entry: RTMConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up binary sensors."""
    coordinator = entry.runtime_data
    if coordinator.inputs.arrival_sensor:
        async_add_entities([PreheatNowBinarySensor(coordinator)])


class PreheatNowBinarySensor(RTMEntity, BinarySensorEntity):
    """On from the preheat start until the arrival (held on once started)."""

    def __init__(self, coordinator: RTMCoordinator) -> None:
        """Initialize the binary sensor."""
        super().__init__(coordinator, KEY_PREHEAT_NOW, ENTITY_ID_FORMAT)

    @property
    def is_on(self) -> bool | None:
        """Evaluated against the current time (timers refresh it at the edges).

        Once on, it stays on until the arrival even if the room reaches the target early,
        and for ``arrival_hold_minutes`` after an arrival time that passed without anyone
        arriving (the coordinator releases it, see ``coordinator.Latch``).
        """
        zone = self.zone
        return zone.preheat_now(dt_util.utcnow()) if zone else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Arrival and start used."""
        zone = self.zone
        if zone is None:
            return {}
        return {
            "arrival": zone.arrival.isoformat() if zone.arrival else None,
            "preheat_start": zone.preheat_start.isoformat() if zone.preheat_start else None,
            "preheat_minutes": (
                round(zone.preheat_minutes) if zone.preheat_minutes is not None else None
            ),
            "reason": zone.status,
            "latched": zone.latched,
            "arrival_at_latch": (
                zone.arrival_at_latch.isoformat() if zone.arrival_at_latch else None
            ),
            "hold_until": zone.hold_until.isoformat() if zone.hold_until else None,
            "lead_minutes": round(zone.lead_minutes) if zone.lead_minutes is not None else None,
            "lead_quantile": zone.lead_quantile,
        }
