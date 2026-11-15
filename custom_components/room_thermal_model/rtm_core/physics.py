"""Closed-form solutions of the first-order RC room model.

    dT/dt = a·u(t − d) − b·(T − Tout) + c        (T in °C, t in hours)

With constant inputs the temperature relaxes exponentially (time constant τ = 1/b)
towards ``Teq = Tout + (a·u + c)/b``. In "no outdoor" mode ``Tout`` is 0 and ``c``
absorbs ``b·mean(Tout)``, so the same formulas apply.
"""

from __future__ import annotations

import math
from typing import Any

REACH_MARGIN_K = 0.1
# The linear model has no upper limit (T∞ = Tout + a/b easily exceeds 40 °C), but a real
# room never gets that warm: thermostats stop at their max setpoint. The reported and
# used maximum reachable temperature is capped there (the climate's ``max_temp`` when
# known, otherwise this constant). The heat-up dynamics keep the uncapped T∞.
DEFAULT_MAX_REACHABLE_C = 30.0


def equilibrium(tout: float, a: float, b: float, c: float, u: float = 1.0) -> float:
    """Steady-state temperature at effort ``u``."""
    return tout + (a * u + c) / b


def coast_temperature(t0: float, minutes: float, tout: float, b: float, c: float) -> float:
    """Temperature after ``minutes`` without heating."""
    teq = tout + c / b
    return teq + (t0 - teq) * math.exp(-b * minutes / 60.0)


def max_reachable(t_inf: float, max_c: float | None = None) -> float:
    """``T∞`` capped at ``max_c`` (default :data:`DEFAULT_MAX_REACHABLE_C`)."""
    return min(t_inf, DEFAULT_MAX_REACHABLE_C if max_c is None else max_c)


def preheat_minutes(
    t0: float,
    target: float,
    tout: float,
    a: float,
    b: float,
    c: float,
    dead_time_minutes: float,
    max_c: float | None = None,
) -> tuple[float | None, dict[str, Any]]:
    """Minutes of full-effort heating (starting now, idle before) to reach ``target``.

    The plain formula ``t = d + τ·ln((T∞ − T0)/(T∞ − Ttarget))`` is refined by letting the
    room coast during the dead time (heat has not arrived yet): ``T0`` is replaced by
    the coasted temperature ``T_d``. Both are reported in the info dict. The target is
    unreachable above :func:`max_reachable` (``T∞`` capped at ``max_c``).
    """
    d = dead_time_minutes
    t_inf = equilibrium(tout, a, b, c)
    info: dict[str, Any] = {
        "t_infinity_c": t_inf,
        "max_reachable_c": max_reachable(t_inf, max_c),
        "time_constant_h": 1.0 / b,
    }
    if t0 >= target:
        info["already_at_target"] = True
        return 0.0, info
    if info["max_reachable_c"] <= target + REACH_MARGIN_K:
        info["unreachable"] = True
        return None, info
    simple = d + 60.0 * math.log((t_inf - t0) / (t_inf - target)) / b
    t_d = coast_temperature(t0, d, tout, b, c)
    info["minutes_simple"] = simple
    info["start_after_dead_time_c"] = t_d
    if t_d >= target:
        # the room drifts up to the target by itself before the heat arrives: no heating
        info["reached_without_heating"] = True
        return 0.0, info
    minutes = d + 60.0 * math.log((t_inf - t_d) / (t_inf - target)) / b
    return minutes, info
