"""Recorder bootstrap: reduce the zone's input history to model samples.

The recorder is queried in ``HISTORY_CHUNK_DAYS`` windows (all input entities at once,
attributes included) inside the recorder executor. Each window is reduced right away,
so raw state rows never pile up:

* state rows of every input are merged chronologically and replayed; after each
  change a sample is built from the latest state of every input (same rules as the
  live sampling, see ``inputs.py``), including the climate ``temperature`` (target)
  attribute reconstructed as ``Sample.setpoint_c`` — it changes in steps and is carried
  forward with the same anti-downtime limit as the other signals;
* changes closer than ``HISTORY_MIN_SPACING_S`` are coalesced (last one wins);
* between changes the last known states are carried forward on a
  ``SAMPLE_INTERVAL_MINUTES`` grid (recorder rows only store changes; the state holds
  until the next row), like the live grid sampling does, but at most
  ``HISTORY_MAX_CARRY_MINUTES`` after the last row: a longer silence of every input is
  HA downtime or a recorder gap, which the model then sees as missing data (indoor
  None) instead of a flat temperature with the pre-shutdown effort.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime, timedelta

from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from .const import (
    HISTORY_CHUNK_DAYS,
    HISTORY_MAX_CARRY_MINUTES,
    HISTORY_MIN_SPACING_S,
    SAMPLE_INTERVAL_MINUTES,
)
from .inputs import ZoneInputs, build_sample
from .rtm_core.base import Sample

_LOGGER = logging.getLogger(__name__)


class HistoryReducer:
    """Incrementally turn chronological state rows into samples."""

    def __init__(self, inputs: ZoneInputs, system_unit: str) -> None:
        self.inputs = inputs
        self.system_unit = system_unit
        self.samples: list[Sample] = []
        self._current: dict[str, State | None] = dict.fromkeys(inputs.sample_entities)
        self._last_ts: datetime | None = None
        self._last_row_ts: datetime | None = None
        self._max_carry = timedelta(minutes=HISTORY_MAX_CARRY_MINUTES)
        self._grid = timedelta(minutes=SAMPLE_INTERVAL_MINUTES)
        self._spacing = timedelta(seconds=HISTORY_MIN_SPACING_S)

    def _emit(self, ts: datetime) -> None:
        sample = build_sample(self.inputs, self._current, ts, self.system_unit)
        if self.samples and self._last_ts is not None and ts - self._last_ts < self._spacing:
            # Coalesce bursts: keep the timestamp of the first change, latest values.
            first = self.samples[-1]
            self.samples[-1] = Sample(
                ts=first.ts,
                indoor_c=sample.indoor_c,
                outdoor_c=sample.outdoor_c,
                effort=sample.effort,
                excluded=sample.excluded,
                setpoint_c=sample.setpoint_c,
            )
            return
        self.samples.append(sample)
        self._last_ts = ts

    def _fill_until(self, ts: datetime) -> None:
        """Carry the last states forward on the grid up to (excluding) ``ts``.

        Never further than ``HISTORY_MAX_CARRY_MINUTES`` past the last recorder row.
        """
        if self._last_ts is None or self._current.get(self.inputs.climate) is None:
            return
        assert self._last_row_ts is not None
        limit = self._last_row_ts + self._max_carry
        step = self._last_ts + self._grid
        while step < ts and step <= limit:
            self._emit(step)
            step += self._grid

    def feed(self, rows: dict[str, list[State]]) -> None:
        """Reduce one window (``{entity_id: chronological states}``)."""
        events: list[tuple[datetime, int, str, State]] = []
        for entity_id, states in rows.items():
            if entity_id not in self._current:
                continue
            for index, state in enumerate(states):
                events.append((state.last_updated, index, entity_id, state))
        events.sort(key=lambda e: (e[0], e[1]))
        pending_ts: datetime | None = None
        for ts, _index, entity_id, state in events:
            if pending_ts is not None and ts != pending_ts:
                self._emit(pending_ts)
                pending_ts = None
            if pending_ts is None:
                self._fill_until(ts)
            self._current[entity_id] = state
            self._last_row_ts = ts
            pending_ts = ts
        if pending_ts is not None:
            self._emit(pending_ts)

    def finish(self, end: datetime) -> list[Sample]:
        """Carry the last states forward up to ``end`` and return the samples."""
        self._fill_until(end)
        return self.samples


def reduce_states(
    inputs: ZoneInputs, system_unit: str, windows: Iterable[dict[str, list[State]]], end: datetime
) -> list[Sample]:
    """Reduce chronological windows of state rows (pure helper, used by tests)."""
    reducer = HistoryReducer(inputs, system_unit)
    for rows in windows:
        reducer.feed(rows)
    return reducer.finish(end)


def _fetch_states(
    hass: HomeAssistant, entity_ids: list[str], start: datetime, end: datetime
) -> dict[str, list[State]]:
    """Read every state row (attributes included) of the inputs. Recorder executor."""
    from homeassistant.components.recorder import history

    result = history.get_significant_states(
        hass,
        start,
        end,
        entity_ids=entity_ids,
        include_start_time_state=True,
        significant_changes_only=False,
        minimal_response=False,
        no_attributes=False,
    )
    return {
        entity_id: [s for s in states if isinstance(s, State)]
        for entity_id, states in result.items()
    }


def _load_chunked(
    hass: HomeAssistant,
    inputs: ZoneInputs,
    system_unit: str,
    start: datetime,
    end: datetime,
) -> list[Sample]:
    """Query the recorder window by window and reduce each one (recorder executor)."""
    reducer = HistoryReducer(inputs, system_unit)
    window = timedelta(days=HISTORY_CHUNK_DAYS)
    entity_ids = inputs.sample_entities
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + window, end)
        reducer.feed(_fetch_states(hass, entity_ids, chunk_start, chunk_end))
        chunk_start = chunk_end
    return reducer.finish(end)


async def async_load_zone_history(
    hass: HomeAssistant, inputs: ZoneInputs, days: float
) -> list[Sample]:
    """Load ``days`` of history as samples (empty when the recorder is unavailable)."""
    if "recorder" not in hass.config.components:
        _LOGGER.debug("Recorder not loaded, skipping history bootstrap for %s", inputs.climate)
        return []
    from homeassistant.components.recorder import get_instance
    from sqlalchemy.exc import SQLAlchemyError

    end = dt_util.utcnow()
    start = end - timedelta(days=days)
    system_unit = hass.config.units.temperature_unit
    try:
        return await get_instance(hass).async_add_executor_job(
            _load_chunked, hass, inputs, system_unit, start, end
        )
    except (HomeAssistantError, SQLAlchemyError, KeyError, ValueError) as err:
        _LOGGER.warning("Could not read history for %s: %s", inputs.climate, err)
        return []
