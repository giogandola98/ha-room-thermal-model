#!/usr/bin/env python3
"""Refit the thermal model from Room Thermal Model diagnostics dumps and score it.

Standalone (imports only ``rtm_core``). Each argument is a diagnostics JSON downloaded from
Home Assistant (Settings → Devices → zone → Download diagnostics); the learnt state in
``data.model`` is restored, refitted at its last fit time with the code of this checkout
and scored on the stored grid:

- fit summary: parameters, reliability, fallbacks, masked rows, the model's own holdout
  quality (60-min RMSE vs persistence, maturity);
- open-loop forecast RMSE of the final parameters at several horizons (default 1, 4 and
  12 h) on the last 20 % of the grid, with the observed effort and outdoor temperature,
  versus persistence;
- heat-up episodes: merged heating runs (pauses ≤ 15 min) after ≥ 1 h idle that rise by
  ≥ 1.2 K; the time to reach ``top − band`` is simulated at full effort from the start
  (what ``preheat`` assumes) and compared with the measured time (bias / MAE in minutes,
  negative = predicted too early).

``--set module.NAME=value`` overrides a module constant before the fit, e.g.
``--set fit.OUTLIER_MAD=8`` (variants for experiments). Run the same script on two
checkouts to compare before/after; the data files never belong in the repository.

    python -I scripts/replay/replay_fit.py diag_zone1.json [diag_zone2.json ...]
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "custom_components" / "room_thermal_model"))

from rtm_core import ModelConfig, RCThermalModel  # noqa: E402
from rtm_core.grid import Grid  # noqa: E402
from rtm_core.quality import bad_rows, step_temperature  # noqa: E402

HOLDOUT = 0.2
STRIDE_ROWS = 6


def load_state(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["data"]["model"] if "data" in data else data["model"]


def apply_overrides(items: list[str]) -> None:
    for item in items:
        name, value = item.split("=", 1)
        mod_name, attr = name.rsplit(".", 1)
        mod = importlib.import_module(f"rtm_core.{mod_name}")
        if not hasattr(mod, attr):
            raise SystemExit(f"unknown constant {name}")
        setattr(mod, attr, type(getattr(mod, attr))(float(value)))


def outdoor_filled(grid: Grid) -> list[float | None]:
    out: list[float | None] = []
    last = next((v for v in grid.outdoor if v is not None), None)
    for v in grid.outdoor:
        if v is not None:
            last = v
        out.append(last)
    return out


def horizon_rmse(grid: Grid, p: dict, hours: float) -> tuple[float | None, float | None, int]:
    """Open-loop RMSE of the parameters ``p`` at ``hours`` ahead (holdout = last 20 %)."""
    step = grid.step_minutes
    h = int(round(hours * 60 / step))
    d = int(round(p["d_minutes"] / step))
    dt_h = step / 60.0
    ind, eff = grid.indoor, grid.effort
    out = grid.outdoor
    bad = bad_rows(grid, 1.5)
    n = len(grid)
    se = se0 = 0.0
    cnt = 0
    k = max(int((1 - HOLDOUT) * n), d)
    while k + h < n:
        ok = not any(bad[k : k + h + 1])
        t = ind[k]
        if ok:
            for j in range(k, k + h):
                u = eff[j - d]
                if u is None:
                    ok = False
                    break
                if p["use_outdoor"]:
                    o0, o1 = out[j], out[j + 1]
                    if o0 is None and o1 is None:
                        ok = False
                        break
                    to = o0 if o1 is None else o1 if o0 is None else 0.5 * (o0 + o1)
                else:
                    to = 0.0
                t = step_temperature(t, u, to, p["a"], p["b"], p["c"], dt_h)
        if ok:
            se += (t - ind[k + h]) ** 2
            se0 += (ind[k] - ind[k + h]) ** 2
            cnt += 1
            k += STRIDE_ROWS
        else:
            k += 1
    if not cnt:
        return None, None, 0
    return math.sqrt(se / cnt), math.sqrt(se0 / cnt), cnt


def heat_up_episodes(
    grid: Grid, band: float, pause: int = 3, min_rise: float = 1.2, idle_rows: int = 12
) -> list[tuple[int, int]]:
    """(start, reach) rows of merged heat-up runs (see module docstring)."""
    eff = [0.0 if e is None else e for e in grid.effort]
    ind = grid.indoor
    bad = [v is None or ex for v, ex in zip(ind, grid.excluded, strict=True)]
    n = len(grid)
    on = [e > 0.5 for e in eff]
    eps: list[tuple[int, int]] = []
    k = 0
    while k < n:
        if not on[k]:
            k += 1
            continue
        j = k
        while True:
            nxt = next((i for i in range(j + 1, min(n, j + 2 + pause)) if on[i]), None)
            if nxt is None:
                break
            j = nxt
        e = min(j + 6, n - 1)
        if k >= idle_rows and not any(on[k - idle_rows : k]) and not any(bad[k : e + 1]):
            top = max(ind[k : e + 1])
            if top - ind[k] >= min_rise:
                r = next(i for i in range(k, e + 1) if ind[i] >= top - band)
                eps.append((k, r))
        k = j + 1
    return eps


def simulate_full_power(grid: Grid, p: dict, k: int, target: float, max_rows: int = 500):
    """Rows until the full-effort simulation from row ``k`` reaches ``target`` (or None)."""
    step = grid.step_minutes
    dt_h = step / 60.0
    d = int(round(p["d_minutes"] / step))
    eff = [0.0 if e is None else e for e in grid.effort]
    out = outdoor_filled(grid)
    b = p["b"]
    f = math.exp(-b * dt_h)
    g = (1 - f) / b
    t = grid.indoor[k]
    for i in range(min(max_rows, len(grid) - 1 - k)):
        j = k + i
        u = 1.0 if j - d >= k else eff[j - d]
        to = (out[j] or 0.0) if p["use_outdoor"] else 0.0
        t = t * f + g * (p["a"] * u + b * to + p["c"])
        if t >= target:
            return i + 1
    return None


def episode_scores(grid: Grid, p: dict, band: float) -> dict:
    errs: list[float] = []
    for k, r in heat_up_episodes(grid, band):
        pred = simulate_full_power(grid, p, k, grid.indoor[r])
        if pred is not None:
            errs.append((pred - (r - k)) * grid.step_minutes)
    if not errs:
        return {"episodes": 0}
    return {
        "episodes": len(errs),
        "bias_min": round(sum(errs) / len(errs), 1),
        "mae_min": round(sum(abs(e) for e in errs) / len(errs), 1),
    }


def run(path: str, horizons: list[float], bands: list[float], verbose: bool = False) -> None:
    state = load_state(path)
    cfg = ModelConfig(**state["config"])
    model = RCThermalModel.from_dict(copy.deepcopy(state), cfg)
    now = datetime.fromtimestamp(state["last_fit_ts"], UTC)
    s = model.fit(now)
    print(f"===== {Path(path).name}")
    if verbose:
        print("  dead-time criterion:", s.get("criterion"))
    if s.get("status") != "ok":
        print("  fit:", s)
        return
    q = s["quality"]
    print(
        f"  a={s['a']:.3f} K/h  b={s['b']:.4f}/h (tau {1 / s['b']:.0f} h)  c={s['c']:.3f}  "
        f"d={s['dead_time_minutes']:.0f} min  mode={s['mode']}"
    )
    print(
        f"  reliable={s['reliable']} ({s['unreliable_reason']})  fallbacks={s['fallbacks']}  "
        f"projected={s['projected']}  masked={s['downweighted_fraction']}"
    )
    print(
        f"  holdout (model): rmse_1h={q['rmse_c']} base={q['baseline_rmse_c']} "
        f"maturity={q['maturity']} episodes={q['episodes']} "
        f"duration_mae={q['duration_mae_minutes']}"
    )
    p = model._params
    assert p is not None
    grid = Grid.from_dict(state["grid"])  # the whole stored grid (the model trims to 30 days)
    parts = []
    for hh in horizons:
        rm, base, cnt = horizon_rmse(grid, p, hh)
        if rm is None:
            parts.append(f"{hh:g}h -")
        else:
            parts.append(f"{hh:g}h {rm:.3f}/{base:.3f} (n={cnt})")
    print("  horizon rmse/persistence: " + " | ".join(parts))
    for band in bands:
        print(f"  heat-ups to top-{band:g} K at full effort: {episode_scores(grid, p, band)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--horizons", default="1,4,12", help="hours, comma separated")
    ap.add_argument("--bands", default="0.15,0.5", help="K below the run top")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="module.NAME=value")
    args = ap.parse_args()
    apply_overrides(args.set)
    horizons = [float(x) for x in args.horizons.split(",")]
    bands = [float(x) for x in args.bands.split(",")]
    for f in args.files:
        run(f, horizons, bands, args.verbose)


if __name__ == "__main__":
    main()
