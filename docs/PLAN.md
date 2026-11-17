# Implementation plan — Room Thermal Model

Generic Home Assistant integration (any `climate` entity, any brand) that learns a
physical thermal model per zone from history and exposes how many minutes of heating a
zone needs to reach a target temperature. Companion of Home Return Predictor, but fully
independent and usable by anyone.

## 1. Scope

- One config entry per **zone**. Inputs (all standard HA attributes):
  - `climate` entity (required): `current_temperature`, `hvac_action` (`heating`/`idle`/`off`),
    `temperature` (target).
  - optional room temperature sensor (overrides the climate's `current_temperature`);
  - optional heating effort sensor in % (valve opening, PI demand...): better than the
    on/off `hvac_action`;
  - outdoor temperature: a sensor **and/or** a `weather` entity. The sensor gives the
    current value and history; the weather entity gives the forecast for the preheat
    window (hourly → twice_daily → daily fallback) and is the outdoor fallback when no
    sensor is set. With neither, the model learns a fixed equilibrium (no outdoor term);
  - no window sensor (user decision): window openings are handled by an explicit
    outlier mask (sustained residuals > 6·MAD for ≥ 3 steps) and the glitch guard;
  - optional arrival timestamp sensor (e.g. `sensor.house_next_return_eta`) to compute
    the preheat start time.
- Target temperature: fixed number (default 20.5 °C) or "follow the climate target".
- No device control. Outputs are sensors only.

## 2. Model (pure Python, stdlib only, `rtm_core/`, no Home Assistant imports)

Lumped single-capacity (RC) model on a 5-minute grid:

    T[k+1] − T[k] = Δt · ( a·u[k−d] − b·(T[k] − Tout[k]) + c )

- `u ∈ [0,1]`: heating effort (effort sensor / 100, else hvac_action heating=1, idle/off=0)
- `a` [K/h] heating gain, `b` [1/h] loss rate (τ = 1/b), `c` [K/h] internal/solar gains,
  `d` dead time in steps (radiator + valve lag).
- Fit: recency-weighted least squares (half-life 21 d) via 3×3 normal equations with a weak
  ridge prior toward radiator defaults (a≈3 K/h, b≈0.15 1/h, c≈0); grid search d ∈ 0..90 min
  choosing the lowest residual; constraints a>0, b>0 by projection.
- Samples excluded: gaps > 15 min, |ΔT| > 1.5 K per step (glitches), missing indoor;
  sustained residual outliers (e.g. open window: |r| > 6·MAD for ≥ 3 consecutive steps)
  masked. The indoor staircase of report-on-change sensors is interpolated linearly between
  change points (≤ 3 h) before building the regression rows; the stored grid stays raw.
- Cross-check / initialisation borrowed from Ecronika/ha-preheat (MIT, © 2025 Ecronika):
  tau from cooling segments (OLS of ln(T−Tout)), dead time by max-slope tangent after an
  off→on edge, trapezoidal forecast averaging with fixed-point duration solve.
- Derived quantities at current conditions (u = 1):
  - equilibrium `T∞ = Tout + (a + c)/b` — max reachable temperature;
  - **preheat minutes** to go from T0 to Ttarget:
    `t = d + τ·ln((T∞ − T0)/(T∞ − Ttarget))` if `T∞ > Ttarget` else unreachable;
    Tout averaged over the preheat window from the weather forecast (fixed-point, ≤ 3 iters);
  - coasting temperature after t minutes without heating (`u = 0`), for "will it still be
    warm when I'm back?".
- Quality: chronological holdout (currently a single 80/20 split: fit on the first 80 % of
  the usable rows, evaluate on the last 20 %; a rolling holdout is future work); 60-min-ahead simulated forecast RMSE vs a
  persistence baseline (temperature stays the same) → skill; observed heat-up episodes
  (heating on, ≥ 0.5 K rise) → preheat-duration error (minutes, MAE). Maturity =
  data factor (hours of heating and of cooling seen) × skill factor.
- Interface `ThermalModel` (ABC) so another backend can replace it.

## 3. Home Assistant glue

- `config_flow.py` (zone: name, climate, optional sensors, outdoor source, target mode) +
  options (target, half-life, history days 30, update interval 5 min).
- `coordinator.py`: samples on a 5-min grid (+ state listeners for immediate updates),
  refits hourly in the executor, bootstrap from recorder (chunked, executor), Store
  persistence (schema version), services `retrain`, `reset`, `export_diagnostics`,
  `diagnostics.py`.
- Entities per zone (device per zone, stable unique_ids, en, it, fr, de translations):
  - `sensor.<zone>_preheat_minutes` (min; attrs `target_c`, `reachable`, `outdoor_used_c`, `max_reachable_c`; `t_infinity_c` inside `details`)
  - `sensor.<zone>_preheat_start` (timestamp; only with arrival sensor)
  - `binary_sensor.<zone>_preheat_now` (on when now ≥ preheat start; latched until arrival)
  - `sensor.<zone>_max_reachable_temperature` (°C)
  - `sensor.<zone>_model_maturity` (%; attrs rmse, baseline_rmse, duration_mae, samples)
  - diagnostic: `_time_constant` (h), `_heating_rate` (K/h), `_dead_time` (min)

## 4. Parallel work

```
[me] PLAN + rtm_core/base.py contract + const.py           (serial, first)
   ├── Agent A: rtm_core model + synthetic rooms + pure tests + scripts/simulate.py
   ├── Agent B: HA glue + translations + integration tests (against the contract)
   └── [me]   : packaging (manifest, hacs.json, CI, pyproject), README en + it,
                cross-link from Home Return Predictor README/blueprint docs
   ▼
 full suite → independent verification agent → fixes → public repo, tag v0.1.0
```

## 5. Verification

| Piece | Check |
|---|---|
| model | parameter recovery (±15 %) on synthetic rooms with known a, b, c, d and noise; dead-time detection; unreachable detection; preheat minutes vs simulated truth (MAE < 10 min); forecast averaging; guards drop glitches/window periods; JSON round trip; predict pure |
| simulate.py | standalone, prints learning curve (parameters, RMSE, duration MAE per day) |
| HA | config flow, options, setup/unload, entity ids & unique_ids, history bootstrap (mocked recorder), services, diagnostics |
| packaging | ruff, pytest, hassfest, HACS action in CI |
