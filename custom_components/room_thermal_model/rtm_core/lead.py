"""Preheat lead time: when to start heating to be at the target on arrival.

The room cools until the heating starts, so the duration is iterated to a fixed point:
lead → start temperature after ``total − lead`` minutes of coasting → new duration. A
setback setpoint (``floor_c``) stops the cooling: the thermostat holds it. An
unreachable target is replaced by the maximum reachable temperature minus a margin.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

from .base import ForecastPoint, PreheatEstimate, ThermalModel

MAX_ITERATIONS = 10
TOLERANCE_MINUTES = 0.5
UNREACHABLE_MARGIN_K = 0.5


def lead_minutes(
    model: ThermalModel,
    now: datetime,
    indoor: float,
    target_c: float,
    outdoor: float | None,
    forecast: Sequence[ForecastPoint],
    estimate: PreheatEstimate,
    total: float,
    max_c: float | None = None,
    floor_c: float | None = None,
    margin_k: float = UNREACHABLE_MARGIN_K,
    max_iterations: int = MAX_ITERATIONS,
    tolerance_minutes: float = TOLERANCE_MINUTES,
) -> float | None:
    """Minutes before an arrival ``total`` minutes away at which heating must start.

    ``estimate`` is the preheat estimate from now. Returns None when no lead can be
    determined (no outdoor data, unreliable model).
    """
    goal = target_c
    if estimate.minutes is None:
        if (
            estimate.reachable
            or estimate.reason.get("status") in ("no_outdoor", "unreliable")
            or estimate.max_reachable_c is None
        ):
            return None
        goal = estimate.max_reachable_c - margin_k
        # already near the maximum: lead 0, the loop still accounts for the cooling
        estimate = model.preheat(now, indoor, goal, outdoor, forecast, max_c)
        if estimate.minutes is None:
            return total
    lead = max(0.0, estimate.minutes)
    # From a future start the forecast describes the outdoor temperature; the value
    # measured now would wrongly anchor the curve there. Without a forecast the current
    # reading is the best guess.
    future_outdoor = None if any(p.outdoor_c is not None for p in forecast) else outdoor
    for _ in range(max_iterations):
        wait = total - lead
        if wait <= 0:
            break
        start_c = model.coast(now, indoor, wait, outdoor, forecast, floor_c=floor_c)
        if start_c is None:
            break
        nxt = model.preheat(
            now + timedelta(minutes=wait), start_c, goal, future_outdoor, forecast, max_c
        ).minutes
        if nxt is None:
            break
        if abs(nxt - lead) < tolerance_minutes:
            lead = nxt
            break
        lead = nxt
    return lead
