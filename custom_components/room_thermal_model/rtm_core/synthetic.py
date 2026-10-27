"""Deterministic synthetic rooms for tests and the simulation script.

The true room follows the continuous RC model with a pure input delay, integrated
exactly minute by minute::

    dT/dt = a·u(t − d) − b·(T − Tout(t)) + c   (− extra loss while a window is open)

- on/off thermostat (hysteresis around a comfort/eco schedule) → effort 1/0 like
  ``hvac_action`` heating/idle;
- optional thermostatic valve (``trv_band_k > 0``): the reported effort stays 1 while the
  room is below the setpoint, but the valve throttles proportionally inside the band, so
  near the setpoint the real heating is a fraction of the full power;
- optional heat source off (``heat_source_off_days``): the thermostat keeps reporting
  "heating" but no heat arrives (boiler idle, heating season not started);
- sensor readings with Gaussian noise (σ≈0.05 K) and 0.1 K quantisation;
- outdoor temperature: daily sinusoid + weather fronts (smooth level changes);
- disturbances: unflagged window openings (1–3 K drop in 10–20 min), missing data
  (no samples for 30–150 min), single-sample glitches, short "excluded" periods.

Everything is driven by ``random.Random(seed)``: identical inputs give identical runs.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from .base import Sample, ThermalParameters

DEFAULT_START = datetime(2025, 1, 6, 0, 0, tzinfo=UTC)  # a Monday


@dataclass(frozen=True)
class RoomSpec:
    name: str
    a: float  # K/h at full effort
    b: float  # 1/h
    c: float  # K/h
    dead_time_minutes: float
    has_outdoor: bool = True
    comfort_c: float = 21.0
    eco_c: float = 17.0
    hysteresis_k: float = 0.2
    noise_sigma: float = 0.05
    quantum: float = 0.1
    windows_per_week: float = 2.0
    gaps_per_week: float = 1.0
    glitches_per_week: float = 3.0
    excluded_per_week: float = 0.5
    outdoor_mean: float = 6.0
    outdoor_daily_amp: float = 4.0
    front_amp: float = 5.0
    weekday_comfort: tuple[tuple[float, float], ...] = ((6.5, 8.5), (17.0, 22.5))
    weekend_comfort: tuple[tuple[float, float], ...] = ((8.0, 23.0),)
    trv_band_k: float = 0.0
    heat_source_off_days: tuple[tuple[float, float], ...] = ()
    seed: int = 1

    @property
    def parameters(self) -> ThermalParameters:
        return ThermalParameters(self.a, self.b, self.c, self.dead_time_minutes)


def radiator_room(**kw: object) -> RoomSpec:
    base: dict[str, object] = {
        "name": "radiator",
        "a": 4.0,
        "b": 0.10,
        "c": 0.3,
        "dead_time_minutes": 20.0,
        "seed": 11,
    }
    base.update(kw)
    return RoomSpec(**base)  # type: ignore[arg-type]


def fast_radiator_room(**kw: object) -> RoomSpec:
    base: dict[str, object] = {
        "name": "fast_radiator",
        "a": 5.0,
        "b": 0.25,
        "c": 0.5,
        "dead_time_minutes": 10.0,
        "eco_c": 16.0,
        "seed": 12,
    }
    base.update(kw)
    return RoomSpec(**base)  # type: ignore[arg-type]


def underfloor_room(**kw: object) -> RoomSpec:
    base: dict[str, object] = {
        "name": "underfloor",
        "a": 1.5,
        "b": 0.06,
        "c": 0.1,
        "dead_time_minutes": 60.0,
        "eco_c": 19.0,
        "weekday_comfort": ((6.0, 9.0), (16.0, 22.0)),
        "weekend_comfort": ((7.0, 22.0),),
        "seed": 13,
    }
    base.update(kw)
    return RoomSpec(**base)  # type: ignore[arg-type]


def no_outdoor_room(**kw: object) -> RoomSpec:
    base: dict[str, object] = {
        "name": "no_outdoor",
        "a": 3.5,
        "b": 0.12,
        "c": 0.4,
        "dead_time_minutes": 15.0,
        "has_outdoor": False,
        "seed": 14,
    }
    base.update(kw)
    return RoomSpec(**base)  # type: ignore[arg-type]


class OutdoorProfile:
    """Daily sinusoid (min ≈ 03:00, max ≈ 15:00) + weather fronts every 2–4 days."""

    def __init__(self, spec: RoomSpec, days: float, start: datetime) -> None:
        rng = random.Random(spec.seed * 7919 + 1)
        self.mean = spec.outdoor_mean
        self.amp = spec.outdoor_daily_amp
        self.start = start
        self.fronts: list[tuple[float, float, float]] = []  # (t_min, from, to)
        level = rng.uniform(-spec.front_amp, spec.front_amp) * 0.5
        self.level0 = level
        t = rng.uniform(1.0, 3.0) * 1440
        while t < (days + 5) * 1440:
            new = rng.uniform(-spec.front_amp, spec.front_amp)
            self.fronts.append((t, level, new))
            level = new
            t += rng.uniform(2.0, 4.0) * 1440
        self.hour0 = start.hour + start.minute / 60.0
        self.ramp = 12 * 60.0

    def __call__(self, t_min: float) -> float:
        level = self.level0
        for tf, lo, hi in self.fronts:
            if t_min < tf:
                break
            if t_min < tf + self.ramp:
                x = (t_min - tf) / self.ramp
                level = lo + (hi - lo) * 0.5 * (1 - math.cos(math.pi * x))
                break
            level = hi
        hour = (self.hour0 + t_min / 60.0) % 24.0
        return self.mean + level + self.amp * math.sin(2 * math.pi * (hour - 9.0) / 24.0)


@dataclass
class SyntheticRun:
    spec: RoomSpec
    start: datetime
    samples: list[Sample]
    true_temp: list[float]  # per minute (state at the start of the minute)
    outdoor: list[float]  # per minute (true)
    effort: list[float]  # per minute (real heating; samples carry the reported effort)
    outdoor_profile: OutdoorProfile
    windows: list[tuple[datetime, float]] = field(default_factory=list)  # (start, drop K)

    def minute(self, ts: datetime) -> int:
        return int((ts - self.start).total_seconds() // 60)

    def samples_between(self, t0: datetime, t1: datetime) -> list[Sample]:
        return [s for s in self.samples if t0 <= s.ts < t1]


def _in_comfort(spec: RoomSpec, ts: datetime) -> bool:
    hour = ts.hour + ts.minute / 60.0
    periods = spec.weekend_comfort if ts.weekday() >= 5 else spec.weekday_comfort
    return any(lo <= hour < hi for lo, hi in periods)


def _events(rng: random.Random, per_week: float, days: float) -> list[float]:
    """Random event times (minutes), Poisson-like with the given weekly rate."""
    out: list[float] = []
    if per_week <= 0:
        return out
    mean_gap = 7 * 1440 / per_week
    t = rng.expovariate(1.0 / mean_gap)
    while t < days * 1440:
        out.append(t)
        t += rng.expovariate(1.0 / mean_gap)
    return out


def generate(
    spec: RoomSpec,
    days: float,
    start: datetime = DEFAULT_START,
    sample_minutes: int = 1,
    windows: bool = True,
    disturbances: bool = True,
    initial_temp: float | None = None,
) -> SyntheticRun:
    """Simulate ``days`` of the room and produce raw samples every ``sample_minutes``."""
    rng = random.Random(spec.seed)
    ev_rng = random.Random(spec.seed * 31 + 5)
    n = int(round(days * 1440))
    outdoor = OutdoorProfile(spec, days, start)
    d = int(round(spec.dead_time_minutes))

    window_at: dict[int, tuple[int, float]] = {}
    win_list: list[tuple[datetime, float]] = []
    if windows:
        for t in _events(ev_rng, spec.windows_per_week, days):
            # windows are opened during the day
            m = int(t // 1440) * 1440 + int(ev_rng.uniform(7.5, 21.5) * 60)
            dur = ev_rng.randint(10, 20)
            drop = ev_rng.uniform(1.0, 3.0)
            if m < n:
                window_at[m] = (dur, drop)
                win_list.append((start + timedelta(minutes=m), drop))
    gap = [False] * n
    glitch: dict[int, float] = {}
    excluded = [False] * n
    if disturbances:
        for t in _events(ev_rng, spec.gaps_per_week, days):
            m0 = int(t)
            for m in range(m0, min(n, m0 + ev_rng.randint(30, 150))):
                gap[m] = True
        for t in _events(ev_rng, spec.glitches_per_week, days):
            glitch[int(t)] = ev_rng.choice((-1, 1)) * ev_rng.uniform(3.0, 6.0)
        for t in _events(ev_rng, spec.excluded_per_week, days):
            m0 = int(t)
            for m in range(m0, min(n, m0 + ev_rng.randint(20, 40))):
                excluded[m] = True

    temp = initial_temp if initial_temp is not None else spec.eco_c
    decay = math.exp(-spec.b / 60.0)
    heating = False
    u_hist: list[float] = []
    true_temp: list[float] = []
    out_list: list[float] = []
    samples: list[Sample] = []
    win_left = 0
    win_rate = 0.0
    q = spec.quantum
    for m in range(n):
        ts = start + timedelta(minutes=m)
        tout = outdoor(m)
        sp = spec.comfort_c if _in_comfort(spec, ts) else spec.eco_c
        if temp < sp - spec.hysteresis_k:
            heating = True
        elif temp > sp + spec.hysteresis_k:
            heating = False
        if spec.trv_band_k > 0:
            # the thermostat reports heating below the setpoint, the valve throttles
            reported = 1.0 if temp < sp else 0.0
            u = min(1.0, max(0.0, (sp - temp) / spec.trv_band_k)) if reported else 0.0
        else:
            reported = u = 1.0 if heating else 0.0
        day = m / 1440.0
        if any(lo <= day < hi for lo, hi in spec.heat_source_off_days):
            u = 0.0
        u_hist.append(u)
        true_temp.append(temp)
        out_list.append(tout)
        if m % sample_minutes == 0 and not gap[m]:
            meas = round((temp + rng.gauss(0.0, spec.noise_sigma)) / q) * q
            if m in glitch:
                meas = round(meas + glitch[m], 1)
            o = round(tout + rng.gauss(0.0, 0.1), 1) if spec.has_outdoor else None
            samples.append(
                Sample(
                    ts,
                    round(meas, 2),
                    o,
                    None if excluded[m] else reported,
                    excluded[m],
                )
            )
        # advance one minute
        u_del = u_hist[m - d] if m >= d else 0.0
        teq = tout + (spec.a * u_del + spec.c) / spec.b
        temp = teq + (temp - teq) * decay
        if m in window_at:
            win_left, drop = window_at[m]
            win_rate = drop / win_left
        if win_left > 0:
            temp -= win_rate
            win_left -= 1
    return SyntheticRun(spec, start, samples, true_temp, out_list, u_hist, outdoor, win_list)


def true_preheat_minutes(
    spec: RoomSpec,
    start_c: float,
    target_c: float,
    tout: float | Callable[[float], float],
    max_minutes: float = 24 * 60.0,
    dt_minutes: float = 0.25,
) -> float | None:
    """Simulated truth: idle before t=0, full effort from t=0 (heat arrives after d)."""
    if start_c >= target_c:
        return 0.0
    f = tout if callable(tout) else (lambda _t, v=float(tout): v)
    temp = start_c
    t = 0.0
    decay = math.exp(-spec.b * dt_minutes / 60.0)
    while t < max_minutes:
        u = 1.0 if t >= spec.dead_time_minutes - 1e-9 else 0.0
        teq = f(t + 0.5 * dt_minutes) + (spec.a * u + spec.c) / spec.b
        new = teq + (temp - teq) * decay
        if new >= target_c:
            return t + dt_minutes * (target_c - temp) / (new - temp)
        temp = new
        t += dt_minutes
    return None


def true_coast(
    spec: RoomSpec,
    start_c: float,
    minutes: float,
    tout: float | Callable[[float], float],
    dt_minutes: float = 0.25,
) -> float:
    """Simulated truth for ``minutes`` without heating."""
    f = tout if callable(tout) else (lambda _t, v=float(tout): v)
    temp = start_c
    t = 0.0
    while t < minutes - 1e-9:
        h = min(dt_minutes, minutes - t)
        teq = f(t + 0.5 * h) + spec.c / spec.b
        temp = teq + (temp - teq) * math.exp(-spec.b * h / 60.0)
        t += h
    return temp


def trv_room(**kw: object) -> RoomSpec:
    """Radiator behind a thermostatic valve, with two days of boiler off."""
    base: dict[str, object] = {
        "name": "trv",
        "a": 3.0,
        "b": 0.08,
        "c": 0.2,
        "dead_time_minutes": 25.0,
        "trv_band_k": 0.6,
        "hysteresis_k": 0.1,
        "heat_source_off_days": ((3.0, 5.0),),
        "seed": 17,
    }
    base.update(kw)
    return RoomSpec(**base)  # type: ignore[arg-type]


ROOMS: dict[str, Callable[..., RoomSpec]] = {
    "radiator": radiator_room,
    "fast_radiator": fast_radiator_room,
    "underfloor": underfloor_room,
    "no_outdoor": no_outdoor_room,
    "trv": trv_room,
}
