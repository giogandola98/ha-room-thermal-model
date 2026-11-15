"""Resampling onto the 5-min grid and gap handling."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from rtm_core import ModelConfig, RCThermalModel, Sample
from rtm_core.grid import Grid, resample, ts_of

T0 = datetime(2025, 1, 6, 0, 0, tzinfo=UTC)


def at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def test_locf_and_alignment():
    samples = [
        Sample(at(1), 20.0, 5.0, 0.0),
        Sample(at(7), 20.5, None, 1.0),
        Sample(at(21), 21.0, None, 1.0),
    ]
    g = resample(samples, 5, 15, 180)
    # first row rounded up to 00:05, last row ends at floor(00:21) = 00:20
    assert g.start_ts == ts_of(at(5))
    assert len(g) == 3
    assert g.indoor == [20.0, 20.5, 20.5]
    assert g.outdoor == [5.0, 5.0, 5.0]  # carried forward (outdoor hold 180 min)


def test_effort_is_time_averaged_over_step():
    samples = [
        Sample(at(0), 20.0, None, 0.0),
        Sample(at(2), 20.0, None, 1.0),  # on for 3 of the 5 minutes
        Sample(at(5), 20.0, None, 1.0),
        Sample(at(9), 20.0, None, 0.0),  # off for the last minute
        Sample(at(10), 20.0, None, 0.0),
    ]
    g = resample(samples, 5, 15)
    assert g.effort[0] == 0.6
    assert abs(g.effort[1] - 0.8) < 1e-9


def test_gap_breaks_series():
    samples = [Sample(at(m), 20.0, 3.0, 0.0) for m in range(0, 31, 5)]
    samples += [Sample(at(m), 19.0, 3.0, 0.0) for m in range(90, 121, 5)]
    g = resample(samples, 5, 15, 180)
    by_min = {round((g.ts(k) - ts_of(T0)) / 60): k for k in range(len(g))}
    assert g.indoor[by_min[45]] == 20.0  # 15 min old: still valid
    assert g.indoor[by_min[50]] is None  # 20 min old: gap
    assert g.effort[by_min[50]] is None
    assert g.outdoor[by_min[50]] == 3.0  # outdoor held longer
    assert g.indoor[by_min[90]] == 19.0


def test_excluded_marks_rows():
    samples = [Sample(at(m), 20.0, None, 0.0, excluded=(10 <= m < 20)) for m in range(0, 41)]
    g = resample(samples, 5, 15)
    assert g.excluded[:5] == [False, False, True, True, False]


def test_grid_dict_round_trip_is_compact():
    samples = [Sample(at(m), 20.123, -1.27, 0.3333, excluded=m == 20) for m in range(0, 60, 5)]
    g = resample(samples, 5, 15)
    data = json.loads(json.dumps(g.to_dict()))
    g2 = Grid.from_dict(data)
    assert g2.indoor[0] == 20.12 and g2.outdoor[0] == -1.27 and g2.effort[0] == 0.333
    assert g2.excluded == g.excluded and g2.start_ts == g.start_ts


def test_incremental_equals_batch(run_of):
    run = run_of("radiator", 2)
    batch = RCThermalModel(ModelConfig())
    batch.load_history(run.samples, replace=True)
    live = RCThermalModel(ModelConfig())
    for s in run.samples:
        live.add_sample(s)
    gb, gl = batch._grid, live._grid
    n = min(len(gb), len(gl))
    assert n > 500
    off = round((gl.start_ts - gb.start_ts) / gb.step_s)
    assert off == 0
    assert gb.indoor[:n] == gl.indoor[:n]
    assert gb.effort[:n] == gl.effort[:n]
    assert gb.excluded[:n] == gl.excluded[:n]


def test_merge_history_and_trim(run_of):
    run = run_of("radiator", 3)
    half = len(run.samples) // 2
    m = RCThermalModel(ModelConfig(history_days=2.0))
    for s in run.samples[half:]:
        m.add_sample(s)
    m.load_history(run.samples[:half], replace=False)
    g = m._grid
    span_days = len(g) * g.step_s / 86400
    assert span_days <= 2.0 + 1e-6
    assert g.end_ts >= ts_of(run.samples[-1].ts) - 300


def test_old_live_samples_are_ignored():
    m = RCThermalModel(ModelConfig())
    for k in range(0, 60):
        m.add_sample(Sample(at(k), 20.0, 5.0, 0.0))
    n = len(m._grid)
    m.add_sample(Sample(at(-600), 30.0, 5.0, 1.0))
    assert len(m._grid) == n


def test_setpoint_locf_and_round_trip():
    samples = [
        Sample(at(1), 20.0, 5.0, 1.0, setpoint_c=17.0),
        Sample(at(7), 20.5, None, 1.0, setpoint_c=21.0),
        Sample(at(21), 21.0, None, 1.0, setpoint_c=21.0),
    ]
    g = resample(samples, 5, 15, 180)
    # setpoint carried forward like the outdoor value (step change, long hold)
    assert g.setpoint == [17.0, 21.0, 21.0]
    data = json.loads(json.dumps(g.to_dict()))
    assert data["setpoint_c100"] == [1700, 2100, 2100]
    g2 = Grid.from_dict(data)
    assert g2.setpoint == g.setpoint
    assert len(g2.setpoint) == len(g2)


def test_setpoint_none_round_trips():
    samples = [Sample(at(m), 20.0, 5.0, 0.0) for m in range(0, 20, 5)]
    g = resample(samples, 5, 15)
    assert all(v is None for v in g.setpoint) and len(g.setpoint) == len(g)
    g2 = Grid.from_dict(json.loads(json.dumps(g.to_dict())))
    assert g2.setpoint == [None] * len(g2)


def test_old_dict_without_setpoint_loads_as_none():
    samples = [Sample(at(m), 20.0, 5.0, 0.0, setpoint_c=19.0) for m in range(0, 20, 5)]
    g = resample(samples, 5, 15)
    data = g.to_dict()
    del data["setpoint_c100"]  # pre-feature snapshot
    g2 = Grid.from_dict(data)
    assert g2.setpoint == [None] * len(g2)
    assert len(g2) == len(g) and g2.indoor == g.indoor


def test_merge_and_append_preserve_setpoint_and_alignment():
    a = resample([Sample(at(m), 20.0, 5.0, 1.0, setpoint_c=17.0) for m in range(0, 26, 5)], 5, 15)
    b = resample([Sample(at(m), 21.0, 5.0, 1.0, setpoint_c=21.0) for m in range(30, 56, 5)], 5, 15)
    merged = a.merged(b)
    cols = (merged.indoor, merged.outdoor, merged.effort, merged.excluded, merged.setpoint)
    assert len({len(c) for c in cols}) == 1  # all columns aligned
    assert merged.setpoint[0] == 17.0 and merged.setpoint[-1] == 21.0

    appended = a.copy()
    appended.append_rows(b)
    cols2 = (
        appended.indoor,
        appended.outdoor,
        appended.effort,
        appended.excluded,
        appended.setpoint,
    )
    assert len({len(c) for c in cols2}) == 1
    assert appended.setpoint[0] == 17.0 and appended.setpoint[-1] == 21.0
    # the padded gap between the two grids carries no setpoint
    assert None in appended.setpoint


def test_trim_before_keeps_setpoint_aligned():
    g = resample([Sample(at(m), 20.0, 5.0, 1.0, setpoint_c=18.0) for m in range(0, 61, 5)], 5, 15)
    g.trim_before(ts_of(at(30)))
    cols = (g.indoor, g.outdoor, g.effort, g.excluded, g.setpoint)
    assert len({len(c) for c in cols}) == 1
    assert all(v == 18.0 for v in g.setpoint)
