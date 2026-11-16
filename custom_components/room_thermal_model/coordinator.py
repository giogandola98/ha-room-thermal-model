"""Per-zone coordinator: live samples, periodic fit, preheat estimate, persistence.

Concurrency rules:

* the model is never used concurrently: every model call holds ``self._lock``. Cheap
  calls (``add_sample``, ``fit_due``, ``parameters``, ``quality``, ``preheat``,
  ``coast``) run in the event loop; CPU work (``fit``, ``load_history``, ``to_dict``,
  ``from_dict``, ``reset``) runs in the executor while the lock is held;
* the lock is held only around model calls (never around recorder reads or service
  calls such as the weather forecast);
* live samples are queued by state listeners (callbacks) and flushed into the model by
  one task under the lock, so a running fit only delays them.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EVENT_HOMEASSISTANT_STOP
from homeassistant.core import (
    CALLBACK_TYPE,
    Event,
    EventStateChangedData,
    HomeAssistant,
    State,
    callback,
)
from homeassistant.helpers.event import (
    async_call_later,
    async_track_point_in_utc_time,
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .const import (
    ARRIVAL_MISSING_UPDATES,
    CONF_ARRIVAL_HOLD_MINUTES,
    CONF_HALF_LIFE_DAYS,
    CONF_HISTORY_DAYS,
    CONF_TARGET_MODE,
    CONF_TARGET_TEMPERATURE,
    CONF_UPDATE_INTERVAL_MINUTES,
    CONF_ZONE_NAME,
    DEFAULT_ARRIVAL_HOLD_MINUTES,
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_HISTORY_DAYS,
    DEFAULT_TARGET_MODE,
    DEFAULT_TARGET_TEMPERATURE,
    DEFAULT_UPDATE_INTERVAL_MINUTES,
    DOMAIN,
    HISTORY_MERGE_VERSION,
    LEAD_MAX_ITERATIONS,
    LEAD_TOLERANCE_MINUTES,
    PREHEAT_HOLD_MINUTES,
    SAMPLE_DEBOUNCE_S,
    SAMPLE_INTERVAL_MINUTES,
    SAVE_DELAY_SECONDS,
    SNAPSHOT_MIN_INTERVAL_S,
    STATUS_AT_TARGET,
    STATUS_LEARNING,
    STATUS_NO_OUTDOOR,
    STATUS_NO_TEMPERATURE,
    STATUS_OK,
    STATUS_SETPOINT_BELOW_CURRENT,
    STATUS_UNREACHABLE,
    STATUS_UNRELIABLE,
    TARGET_MODE_CLIMATE,
    UNREACHABLE_MARGIN_K,
)
from .forecast import WeatherForecast
from .history import async_load_zone_history
from .inputs import (
    INVALID_STATES,
    ZoneInputs,
    build_sample,
    climate_max_temp,
    climate_target,
)
from .rtm_core import create_model
from .rtm_core.base import (
    ForecastPoint,
    ModelConfig,
    PreheatEstimate,
    QualityReport,
    Sample,
    ThermalModel,
    ThermalParameters,
)
from .storage import STORAGE_SCHEMA, RTMStorage, StoredData

_LOGGER = logging.getLogger(__name__)

type RTMConfigEntry = ConfigEntry[RTMCoordinator]

MODEL_ERRORS = (ArithmeticError, KeyError, TypeError, ValueError)


@dataclass
class ZoneData:
    """Result of one update (read by entities)."""

    updated_at: datetime
    indoor_c: float | None
    outdoor_c: float | None
    effort: float | None
    target_c: float
    target_source: str  # "fixed" / "climate" (falls back to fixed without a climate target)
    status: str
    preheat: PreheatEstimate | None
    parameters: ThermalParameters | None
    quality: QualityReport
    forecast_used: bool = False
    forecast_type: str | None = None
    arrival: datetime | None = None  # valid future arrival only
    preheat_start: datetime | None = None
    coast_c: float | None = None  # temperature at arrival without heating
    latched: bool = False  # preheat_now held on (hysteresis)
    arrival_at_latch: datetime | None = None  # arrival when preheat_now latched
    hold_until: datetime | None = None  # latched: off at this time (arrival passed)
    lead_minutes: float | None = None
    last_fit: datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """True when the model has learnt parameters."""
        return self.parameters is not None

    @property
    def preheat_minutes(self) -> float | None:
        """Minutes of heating needed (None when unknown or unreachable)."""
        return self.preheat.minutes if self.preheat is not None else None

    def preheat_now(self, now: datetime) -> bool | None:
        """On from the preheat start until the arrival; unknown without a start.

        Once on it is held (``latched``), so it neither drops when the room reaches the
        target early nor flickers while heating, nor on a single update without arrival.
        After the arrival time it stays on until ``hold_until`` (``arrival_hold_minutes``,
        nobody arrived yet). The coordinator releases the latch (see :class:`Latch`).
        """
        if self.latched:
            return self.hold_until is None or now < self.hold_until
        if self.arrival is None or self.arrival <= now:
            return False
        if self.preheat_start is None:
            return None
        return now >= self.preheat_start


@dataclass
class Latch:
    """State of a latched ``preheat_now`` (persisted: it survives a restart).

    Released when the arrival moves later than ``arrival_at_latch`` by more than
    ``PREHEAT_HOLD_MINUTES`` (cumulative, so an estimate that keeps slipping cannot hold
    it for hours), when the arrival is missing for more than ``ARRIVAL_MISSING_UPDATES``
    updates before its time (someone arrived, the estimate was withdrawn), or
    ``arrival_hold_minutes`` after the last known arrival time has passed.
    """

    arrival_at_latch: datetime
    last_arrival: datetime
    missing: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "arrival_at_latch": self.arrival_at_latch.isoformat(),
            "last_arrival": self.last_arrival.isoformat(),
            "missing": self.missing,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Latch | None:
        if not isinstance(data, dict):
            return None
        at = _parse_dt(data.get("arrival_at_latch"))
        last = _parse_dt(data.get("last_arrival")) or at
        if at is None or last is None:
            return None
        return cls(at, last, int(data.get("missing") or 0))

    def update(self, arrival: datetime | None, now: datetime, hold: timedelta) -> bool:
        """Advance with the current arrival; False when the latch is released."""
        if arrival is not None:
            self.missing = 0
            self.last_arrival = arrival
            return arrival <= self.arrival_at_latch + timedelta(minutes=PREHEAT_HOLD_MINUTES)
        if now < self.last_arrival:
            self.missing += 1
            return self.missing <= ARRIVAL_MISSING_UPDATES
        return now < self.last_arrival + hold


def parse_arrival(state: State | None, now: datetime) -> datetime | None:
    """Arrival timestamp from a timestamp sensor, only when valid and in the future."""
    if state is None or state.state in INVALID_STATES:
        return None
    parsed = dt_util.parse_datetime(state.state)
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt_util.get_default_time_zone())
    parsed = dt_util.as_utc(parsed)
    return parsed if parsed > now else None


def _parse_dt(value: Any) -> datetime | None:
    return dt_util.parse_datetime(value) if isinstance(value, str) and value else None


class RTMCoordinator(DataUpdateCoordinator[ZoneData]):
    """Coordinate one zone (one config entry)."""

    config_entry: RTMConfigEntry

    def __init__(self, hass: HomeAssistant, entry: RTMConfigEntry) -> None:
        """Initialize the coordinator."""
        options = entry.options
        super().__init__(
            hass,
            _LOGGER,
            config_entry=entry,
            name=f"{DOMAIN}_{entry.entry_id}",
            update_interval=timedelta(
                minutes=float(
                    options.get(CONF_UPDATE_INTERVAL_MINUTES, DEFAULT_UPDATE_INTERVAL_MINUTES)
                )
            ),
        )
        self.zone_name: str = str(entry.data.get(CONF_ZONE_NAME) or entry.title)
        self.inputs = ZoneInputs.from_data(entry.data)
        self.target_mode: str = options.get(CONF_TARGET_MODE, DEFAULT_TARGET_MODE)
        self.fixed_target: float = float(
            options.get(CONF_TARGET_TEMPERATURE, DEFAULT_TARGET_TEMPERATURE)
        )
        self.history_days: float = float(options.get(CONF_HISTORY_DAYS, DEFAULT_HISTORY_DAYS))
        self.arrival_hold = timedelta(
            minutes=float(options.get(CONF_ARRIVAL_HOLD_MINUTES, DEFAULT_ARRIVAL_HOLD_MINUTES))
        )
        self.latch: Latch | None = None
        self.model_config = ModelConfig(
            step_minutes=float(SAMPLE_INTERVAL_MINUTES),
            half_life_days=float(options.get(CONF_HALF_LIFE_DAYS, DEFAULT_HALF_LIFE_DAYS)),
            history_days=self.history_days,
        )
        self.model: ThermalModel = create_model(self.model_config)
        self.forecast: WeatherForecast | None = (
            WeatherForecast(hass, self.inputs.weather) if self.inputs.weather else None
        )
        self.storage = RTMStorage(hass, entry.entry_id)
        self.snapshot: dict[str, Any] | None = None
        self.last_fit: datetime | None = None
        self.last_fit_summary: dict[str, Any] | None = None
        self.bootstrapped = False
        self.bootstrapping = False
        self.history_merged = 0
        self._sample_at: dict[str, datetime] = {}  # last state-change sample per entity
        self._last_sample_at: datetime | None = None
        self._deferred: dict[str, CALLBACK_TYPE] = {}
        self._lock = asyncio.Lock()
        self._retrain_lock = asyncio.Lock()
        self._pending: list[Sample] = []
        self._flush_task: asyncio.Task[None] | None = None
        self._dirty = False
        self._snapshot_at: datetime | None = None
        self._save_scheduled_at: datetime | None = None
        self._unsubs: list[CALLBACK_TYPE] = []
        self._unsub_stop: CALLBACK_TYPE | None = None
        self._unsub_timers: list[CALLBACK_TYPE] = []
        self._unloading = False

    # ------------------------------------------------------------------ helpers

    @property
    def system_unit(self) -> str:
        """Temperature unit of climate attributes."""
        return self.hass.config.units.temperature_unit

    def _states(self) -> dict[str, State | None]:
        return {e: self.hass.states.get(e) for e in self.inputs.sample_entities}

    def current_sample(self, ts: datetime | None = None) -> Sample:
        """Sample built from the current states of the inputs."""
        return build_sample(self.inputs, self._states(), ts or dt_util.utcnow(), self.system_unit)

    async def _async_model_job(self, func: Any, *args: Any) -> Any:
        """Run one CPU-bound model call in the executor under the lock."""
        async with self._lock:
            return await self.hass.async_add_executor_job(func, *args)

    # ------------------------------------------------------------------ setup

    async def _async_setup(self) -> None:
        """Restore the stored model, subscribe, schedule the bootstrap."""
        self.config_entry.async_on_unload(self._async_unsubscribe)
        stored = await self.storage.async_load()
        model_data = stored.get("model")
        stored_inputs = stored.get("inputs")
        inputs_changed = stored_inputs is not None and stored_inputs != sorted(
            self.inputs.sample_entities
        )
        if isinstance(model_data, dict) and model_data:
            try:
                self.model = await self.hass.async_add_executor_job(
                    type(self.model).from_dict, model_data, self.model_config
                )
            except MODEL_ERRORS as err:
                _LOGGER.warning("Discarding unreadable model of %s: %s", self.zone_name, err)
            else:
                self.snapshot = model_data
        self.last_fit = _parse_dt(stored.get("last_fit"))
        self.bootstrapped = bool(stored.get("bootstrapped")) and self.snapshot is not None
        self.history_merged = int(stored.get("history_merged") or 0)
        self.latch = Latch.from_dict(stored.get("latch")) if self.inputs.arrival_sensor else None
        if inputs_changed and self.snapshot is not None:
            # the data stays (and the recorder history of the new inputs is merged over
            # it below); only the parameters learnt from the old inputs are dropped
            _LOGGER.info(
                "Inputs of %s changed since the model was learnt: refitting its parameters",
                self.zone_name,
            )
            await self._async_model_job(self.model.reset_parameters)
            self.last_fit = None
            self.history_merged = 0
            self._dirty = True
        self._async_subscribe()
        self._queue_sample(dt_util.utcnow())
        if not self.bootstrapped:
            self.bootstrapping = True
            self.config_entry.async_create_background_task(
                self.hass, self.async_retrain(bootstrap=True), f"{DOMAIN}_bootstrap"
            )
        elif self.history_merged < HISTORY_MERGE_VERSION:
            # one-off merge of the recorder history after an upgrade (fills the setpoint
            # column of old data) or an input change
            self.config_entry.async_create_background_task(
                self.hass, self.async_retrain(), f"{DOMAIN}_history_merge"
            )

    @callback
    def _async_subscribe(self) -> None:
        self._unsubs.append(
            async_track_state_change_event(
                self.hass, self.inputs.all_entities, self._async_handle_state_event
            )
        )
        self._unsubs.append(
            async_track_time_interval(
                self.hass,
                self._async_grid_sample,
                timedelta(minutes=SAMPLE_INTERVAL_MINUTES),
                name=f"{DOMAIN} grid sample",
            )
        )
        self._unsub_stop = self.hass.bus.async_listen_once(
            EVENT_HOMEASSISTANT_STOP, self._async_handle_stop
        )

    @callback
    def _async_unsubscribe(self) -> None:
        """Remove every listener (idempotent)."""
        for unsub in self._unsubs:
            unsub()
        self._unsubs.clear()
        for unsub in self._deferred.values():
            unsub()
        self._deferred.clear()
        self._cancel_timers()
        if self._unsub_stop:
            self._unsub_stop()
            self._unsub_stop = None

    @callback
    def _cancel_timers(self) -> None:
        for unsub in self._unsub_timers:
            unsub()
        self._unsub_timers.clear()

    async def _async_handle_stop(self, _event: Event) -> None:
        """HA stops: snapshot now; Store writes the delayed save on its final write."""
        self._unsub_stop = None
        await self._async_flush_samples()
        await self._async_snapshot(force=True)
        self.storage.async_schedule_save(self._store_data)

    async def async_unload(self) -> None:
        """Stop listening and persist the model."""
        self._unloading = True
        self._async_unsubscribe()
        await self._async_flush_samples()
        await self._async_snapshot(force=True)
        await self.storage.async_save(self._store_data())

    # ------------------------------------------------------------------ samples

    @callback
    def _async_handle_state_event(self, event: Event[EventStateChangedData]) -> None:
        if event.data["new_state"] is None or self._unloading:
            return
        entity_id = event.data["entity_id"]
        if entity_id != self.inputs.arrival_sensor:
            self._debounced_sample(entity_id, event.data["new_state"].last_updated)
        # Debounced by the coordinator (immediate first call, then 10 s cooldown).
        self.config_entry.async_create_background_task(
            self.hass, self.async_request_refresh(), f"{DOMAIN}_state_refresh"
        )

    @callback
    def _debounced_sample(self, entity_id: str, ts: datetime) -> None:
        """At most one state-change sample per entity per ``SAMPLE_DEBOUNCE_S``.

        Thermostats publish bursts of attribute updates (hundreds per hour); a change
        inside the window is not lost: one sample with the latest states is taken when
        the window ends.
        """
        last = self._sample_at.get(entity_id)
        if last is None or (ts - last).total_seconds() >= SAMPLE_DEBOUNCE_S:
            self._sample_at[entity_id] = ts
            self._queue_sample(ts)
            return
        if entity_id in self._deferred:
            return

        @callback
        def _fire(_now: datetime) -> None:
            self._deferred.pop(entity_id, None)
            if self._unloading:
                return
            now = dt_util.utcnow()
            self._sample_at[entity_id] = now
            self._queue_sample(now)

        delay = SAMPLE_DEBOUNCE_S - (ts - last).total_seconds()
        self._deferred[entity_id] = async_call_later(self.hass, max(1.0, delay), _fire)

    @callback
    def _async_grid_sample(self, now: datetime) -> None:
        if not self._unloading:
            self._queue_sample(dt_util.utcnow())

    @callback
    def _queue_sample(self, ts: datetime) -> None:
        """Build a sample from the current states and queue it for the model."""
        self._pending.append(self.current_sample(ts))
        self._last_sample_at = ts
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = self.config_entry.async_create_background_task(
                self.hass, self._async_flush_samples(), f"{DOMAIN}_flush_samples"
            )

    async def _async_flush_samples(self) -> None:
        """Feed queued samples to the model (cheap calls, under the lock)."""
        if not self._pending:
            return
        async with self._lock:
            pending, self._pending = self._pending, []
            for sample in pending:
                self.model.add_sample(sample)
            self._dirty = True

    # ------------------------------------------------------------------ update

    def _target(self) -> tuple[float, str]:
        if self.target_mode == TARGET_MODE_CLIMATE:
            target = climate_target(self.hass.states.get(self.inputs.climate), self.system_unit)
            if target is not None:
                return target, TARGET_MODE_CLIMATE
        return self.fixed_target, "fixed"

    async def _async_get_forecast(self, now: datetime) -> list[ForecastPoint]:
        if self.forecast is None:
            return []
        try:
            return await self.forecast.async_get(now)
        except Exception:  # a broken weather integration must not stop the zone
            _LOGGER.warning("Forecast of %s failed", self.inputs.weather, exc_info=True)
            return []

    async def _async_fit_if_due(self, now: datetime, force: bool = False) -> bool:
        """Refit in the executor when due. Returns True when a fit succeeded."""
        async with self._lock:
            if not force and not self.model.fit_due(now):
                return False
            try:
                summary = await self.hass.async_add_executor_job(self.model.fit, now)
            except MODEL_ERRORS:
                _LOGGER.warning("Fit of %s failed", self.zone_name, exc_info=True)
                return False
        self.last_fit = now
        self.last_fit_summary = summary if isinstance(summary, dict) else None
        self._dirty = True
        _LOGGER.debug("Fit of %s: %s", self.zone_name, summary)
        return True

    async def _async_update_data(self) -> ZoneData:
        now = dt_util.utcnow()
        # Every poll feeds the current values too (the model treats indoor/effort older
        # than max_gap_minutes as missing; the grid timer covers long update intervals),
        # unless a sample was just taken (state-change bursts refresh every few seconds).
        if (
            self._last_sample_at is None
            or (now - self._last_sample_at).total_seconds() >= SAMPLE_DEBOUNCE_S
        ):
            self._pending.append(self.current_sample(now))
            self._last_sample_at = now
        await self._async_flush_samples()
        fitted = await self._async_fit_if_due(now)
        forecast = await self._async_get_forecast(now)
        current = self.current_sample(now)
        target_c, target_source = self._target()
        arrival_state = (
            self.hass.states.get(self.inputs.arrival_sensor) if self.inputs.arrival_sensor else None
        )
        arrival = parse_arrival(arrival_state, now)
        indoor = current.indoor_c
        outdoor = current.outdoor_c
        # the climate's max setpoint caps the maximum reachable temperature
        max_c = climate_max_temp(self.hass.states.get(self.inputs.climate), self.system_unit)
        estimate: PreheatEstimate | None = None
        coast_c: float | None = None
        lead: float | None = None
        try:
            async with self._lock:
                parameters = self.model.parameters()
                quality = self.model.quality()
                if indoor is not None and parameters is not None:
                    estimate = self.model.preheat(now, indoor, target_c, outdoor, forecast, max_c)
                    if arrival is not None:
                        minutes = (arrival - now).total_seconds() / 60.0
                        coast_c = self.model.coast(now, indoor, minutes, outdoor, forecast)
                        lead = self._lead_minutes(
                            now, indoor, target_c, outdoor, forecast, estimate, minutes, max_c
                        )
        except MODEL_ERRORS as err:
            raise UpdateFailed(
                translation_domain=DOMAIN,
                translation_key="update_failed",
                translation_placeholders={"zone": self.zone_name, "error": str(err)},
            ) from err
        if indoor is None:
            status = STATUS_NO_TEMPERATURE
        elif parameters is None or estimate is None:
            status = STATUS_LEARNING
        elif estimate.reason.get("status") == "no_outdoor":
            status = STATUS_NO_OUTDOOR
        elif estimate.reason.get("status") == "unreliable":
            status = STATUS_UNRELIABLE
        elif not estimate.reachable:
            status = STATUS_UNREACHABLE
        elif estimate.minutes is None:
            status = STATUS_LEARNING
        elif target_source == TARGET_MODE_CLIMATE and indoor > target_c:
            # a setback setpoint never asks for a preheat (see README)
            status = STATUS_SETPOINT_BELOW_CURRENT
        elif estimate.minutes <= 0 or indoor >= target_c:
            status = STATUS_AT_TARGET
        else:
            status = STATUS_OK
        preheat_start = arrival - timedelta(minutes=lead) if arrival and lead is not None else None
        # Latch: held whatever the lead does (it collapses once the room is at target);
        # released only by the rules of :class:`Latch`.
        latch_before = self.latch.to_dict() if self.latch else None
        released = False
        if self.latch is not None and not self.latch.update(arrival, now, self.arrival_hold):
            self.latch = None
            released = True
        # on now by the new start, or by the previous one (the timer fired at the start and
        # the recomputed start moved a little later, e.g. because the room cooled less)
        prev = self.data
        if (
            self.latch is None
            and not released
            and arrival is not None
            and arrival > now
            and (
                (preheat_start is not None and now >= preheat_start)
                or (
                    prev is not None
                    and prev.arrival == arrival
                    and prev.preheat_start is not None
                    and now >= prev.preheat_start
                )
            )
        ):
            self.latch = Latch(arrival, arrival)
        if (self.latch.to_dict() if self.latch else None) != latch_before:
            self._schedule_save(now, force=True)
        latch = self.latch
        forecast_used = estimate is not None and estimate.reason.get("outdoor_source") == "forecast"
        data = ZoneData(
            updated_at=now,
            indoor_c=indoor,
            outdoor_c=outdoor,
            effort=current.effort,
            target_c=target_c,
            target_source=target_source,
            status=status,
            preheat=estimate,
            parameters=parameters,
            quality=quality,
            forecast_used=forecast_used,
            forecast_type=self.forecast.cache.forecast_type if self.forecast else None,
            arrival=arrival,
            preheat_start=preheat_start,
            coast_c=coast_c,
            latched=latch is not None,
            arrival_at_latch=latch.arrival_at_latch if latch else None,
            hold_until=latch.last_arrival + self.arrival_hold if latch else None,
            lead_minutes=lead,
            last_fit=self.last_fit,
        )
        self._schedule_timers(data, now)
        await self._async_snapshot(force=fitted)
        return data

    def _lead_minutes(
        self,
        now: datetime,
        indoor: float,
        target_c: float,
        outdoor: float | None,
        forecast: list[ForecastPoint],
        estimate: PreheatEstimate,
        total: float,
        max_c: float | None = None,
    ) -> float | None:
        """Minutes before the arrival at which heating must start (model lock held).

        The room cools until the heating starts, so the duration is iterated to a fixed
        point: lead -> start temperature after ``total - lead`` minutes of coasting ->
        new duration. An unreachable target is replaced by the maximum reachable
        temperature minus a margin (the sensors stay unreachable; only the start moves).
        Returns None when no lead can be determined.
        """
        goal = target_c
        if estimate.minutes is None:
            if (
                estimate.reachable
                or estimate.reason.get("status") in ("no_outdoor", "unreliable")
                or estimate.max_reachable_c is None
            ):
                return None
            goal = estimate.max_reachable_c - UNREACHABLE_MARGIN_K
            # already near the maximum: lead 0, the loop still accounts for the cooling
            estimate = self.model.preheat(now, indoor, goal, outdoor, forecast, max_c)
            if estimate.minutes is None:
                return total
        lead = max(0.0, estimate.minutes)
        # From a future start the forecast describes the outdoor temperature; the value
        # measured now would wrongly anchor the curve there. Without a forecast the
        # current reading is the best guess.
        future_outdoor = None if any(p.outdoor_c is not None for p in forecast) else outdoor
        for _ in range(LEAD_MAX_ITERATIONS):
            wait = total - lead
            if wait <= 0:
                break
            start_c = self.model.coast(now, indoor, wait, outdoor, forecast)
            if start_c is None:
                break
            nxt = self.model.preheat(
                now + timedelta(minutes=wait), start_c, goal, future_outdoor, forecast, max_c
            ).minutes
            if nxt is None:
                break
            if abs(nxt - lead) < LEAD_TOLERANCE_MINUTES:
                lead = nxt
                break
            lead = nxt
        return lead

    @callback
    def _schedule_timers(self, data: ZoneData, now: datetime) -> None:
        """Refresh entities exactly at the preheat start and at the arrival."""
        self._cancel_timers()
        for when in (data.preheat_start, data.arrival, data.hold_until):
            if when is not None and when > now:
                self._unsub_timers.append(
                    async_track_point_in_utc_time(self.hass, self._async_timer_fired, when)
                )

    @callback
    def _async_timer_fired(self, _now: datetime) -> None:
        self.async_update_listeners()
        # latch the preheat start / release the hold without waiting for the poll
        self.config_entry.async_create_background_task(
            self.hass, self.async_request_refresh(), f"{DOMAIN}_timer_refresh"
        )

    # ------------------------------------------------------------------ storage

    async def _async_snapshot(self, force: bool = False) -> None:
        """Snapshot the model (executor) when changed; at most every 10 min unless forced."""
        now = dt_util.utcnow()
        if not self._dirty and self.snapshot is not None and not force:
            return
        if (
            not force
            and self._snapshot_at is not None
            and (now - self._snapshot_at).total_seconds() < SNAPSHOT_MIN_INTERVAL_S
        ):
            return
        self._dirty = False
        self.snapshot = await self._async_model_job(self.model.to_dict)
        self._snapshot_at = now
        self._schedule_save(now)

    @callback
    def _schedule_save(self, now: datetime, force: bool = False) -> None:
        """Delayed save; not re-armed while one is pending (it writes the latest data)."""
        if force:
            # small state (the latch): write soon, the model snapshot is reused
            self.storage.async_schedule_save(self._store_data, delay=10.0)
            self._save_scheduled_at = now
            return
        if (
            self._save_scheduled_at is not None
            and (now - self._save_scheduled_at).total_seconds() < SAVE_DELAY_SECONDS
        ):
            return
        self._save_scheduled_at = now
        self.storage.async_schedule_save(self._store_data)

    def _store_data(self) -> StoredData:
        return {
            "schema": STORAGE_SCHEMA,
            "model": self.snapshot,
            "last_fit": self.last_fit.isoformat() if self.last_fit else None,
            "bootstrapped": self.bootstrapped,
            "inputs": sorted(self.inputs.sample_entities),
            "history_merged": self.history_merged,
            "latch": self.latch.to_dict() if self.latch else None,
        }

    # ------------------------------------------------------------------ actions

    async def async_retrain(self, bootstrap: bool = False, replace: bool = False) -> int:
        """Merge the recorder history into the model data (or replace the data with it).

        The history wins where it has data; older stored data (beyond the recorder
        retention) is kept unless ``replace``. The one-off history merge flag is set
        even when the recorder returns nothing. Returns the number of samples loaded.
        """
        async with self._retrain_lock:
            try:
                samples = await async_load_zone_history(self.hass, self.inputs, self.history_days)
                _LOGGER.debug("History for %s: %d samples", self.zone_name, len(samples))
                if samples:
                    try:
                        await self._async_model_job(
                            self.model.load_history, samples, replace and not bootstrap
                        )
                    except MODEL_ERRORS:
                        _LOGGER.warning(
                            "Loading history of %s failed", self.zone_name, exc_info=True
                        )
                    else:
                        self._dirty = True
                await self._async_flush_samples()
                await self._async_fit_if_due(dt_util.utcnow(), force=bool(samples))
                self.bootstrapped = True
                self.history_merged = HISTORY_MERGE_VERSION
            finally:
                self.bootstrapping = False
            await self._async_snapshot(force=True)
        if not self._unloading:
            await self.async_request_refresh()
        return len(samples)

    async def async_reset(self) -> None:
        """Forget everything learnt (learning restarts from live samples)."""
        self._pending.clear()
        await self._async_model_job(self.model.reset)
        self.last_fit = None
        self.last_fit_summary = None
        self._dirty = True
        await self._async_snapshot(force=True)
        await self.async_refresh()
