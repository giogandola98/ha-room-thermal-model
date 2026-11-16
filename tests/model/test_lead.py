"""Lead time with a setback floor; coast of an unreliable model."""

from __future__ import annotations

import pytest

from rtm_core.lead import lead_minutes


def test_coast_floor_and_lead(trained):
    m = trained("radiator", 14)
    now = m._grid.ts(len(m._grid) - 1)
    from rtm_core.grid import dt_of

    now = dt_of(now)
    free = m.coast(now, 20.0, 24 * 60, 0.0)
    held = m.coast(now, 20.0, 24 * 60, 0.0, floor_c=18.0)
    assert free < 18.0 and held == pytest.approx(18.0)
    # a floor above the start does not heat the room
    assert m.coast(now, 17.0, 60, 0.0, floor_c=18.0) < 17.0
    est = m.preheat(now, 20.0, 21.0, 0.0)
    no_floor = lead_minutes(m, now, 20.0, 21.0, 0.0, (), est, 24 * 60)
    floor = lead_minutes(m, now, 20.0, 21.0, 0.0, (), est, 24 * 60, floor_c=18.0)
    from_18 = m.preheat(now, 18.0, 21.0, 0.0).minutes
    assert floor == pytest.approx(from_18, abs=1.0)
    assert no_floor > floor


def test_unreliable_model_does_not_coast(trained):
    m = trained("radiator", 14)
    from rtm_core.grid import dt_of

    now = dt_of(m._grid.ts(len(m._grid) - 1))
    m._params["reliable"] = False
    assert m.coast(now, 20.0, 60, 5.0) is None
    assert m.parameters().reliable is False
