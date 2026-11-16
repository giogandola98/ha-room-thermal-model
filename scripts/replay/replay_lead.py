#!/usr/bin/env python3
"""Preheat lead for an absence scenario on a model restored from a diagnostics dump.

The zone is refitted at its last fit time (as ``replay_fit.py``), then: the room is at
``--start`` °C now, the thermostat holds a setback ``--setback`` °C, the arrival is in
``--hours`` h, the outdoor temperature stays at ``--outdoor`` °C and the target is
``--target`` °C. Prints the lead (minutes before the arrival at which heating starts)
and the temperature at the arrival without preheating, with and without the setback
floor (without it the model lets the room coast towards the outdoor temperature).

    python -I scripts/replay/replay_lead.py diag_zone.json --hours 24 --outdoor 8
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "custom_components" / "room_thermal_model"))

from rtm_core import ModelConfig, RCThermalModel  # noqa: E402
from rtm_core.lead import lead_minutes  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--outdoor", type=float, default=8.0)
    ap.add_argument("--start", type=float, default=20.0)
    ap.add_argument("--setback", type=float, default=18.0)
    ap.add_argument("--target", type=float, default=20.0)
    args = ap.parse_args()
    for path in args.files:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        state = data["data"]["model"] if "data" in data else data["model"]
        model = RCThermalModel.from_dict(copy.deepcopy(state), ModelConfig(**state["config"]))
        now = datetime.fromtimestamp(state["last_fit_ts"], UTC)
        s = model.fit(now)
        total = args.hours * 60.0
        print(f"===== {Path(path).name}: a={s.get('a'):.3f} tau={1 / s.get('b'):.0f} h")
        est = model.preheat(now, args.start, args.target, args.outdoor)
        for label, floor in (("no floor", None), (f"floor {args.setback:g}", args.setback)):
            lead = lead_minutes(
                model, now, args.start, args.target, args.outdoor, (), est, total, floor_c=floor
            )
            arr = model.coast(now, args.start, total, args.outdoor, floor_c=floor)
            lead_s = "-" if lead is None else f"{lead:.0f} min ({lead / 60:.2f} h)"
            arr_s = "-" if arr is None else f"{arr:.2f} °C"
            print(f"  {label:10s} lead {lead_s}, temperature at arrival {arr_s}")


if __name__ == "__main__":
    main()
