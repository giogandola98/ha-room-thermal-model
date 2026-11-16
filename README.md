# Room Thermal Model

A generic Home Assistant integration that **learns how each room heats up and cools
down** from your own history, and tells you **how many minutes of heating a room needs**
to reach a target temperature, so that automations can start heating at the right time.

It works with **any `climate` entity** (TRVs, generic_thermostat, Versatile Thermostat,
smart thermostats…) and only reads standard Home Assistant attributes. It never controls
anything: it only exposes sensors.

## How it works

Each zone is modelled as a single thermal mass that gains heat from the heater and loses
it to the outside (a lumped *RC* model), on a 5-minute grid:

```
ΔT/Δt = a · effort(t − dead time) − b · (T_room − T_outdoor) + c
```

| Parameter | Meaning |
|---|---|
| `a` (K/h) | how fast the heater warms the room at full effort |
| `b` (1/h) | heat loss to the outside — time constant τ = 1/b |
| `c` (K/h) | internal and solar gains |
| dead time (min) | lag before the radiator starts warming the room |

- **Learning** uses *all* history, not only preheat sessions: periods with the heater off
  teach heat loss, periods with it on teach heating power. A weighted least-squares fit
  drops sustained anomalies such as an open window (residuals far outside the usual
  spread for 15 minutes or more), and recent data weighs more (half-life 21 days). Many
  room sensors only report when the temperature changed by ~0.1 K (or once an hour): for
  fitting, the readings are interpolated linearly between those changes (up to 3 h), so a
  slow room is not mistaken for one whose temperature never moves. On first start the recorder history is imported, so
  the model is useful from day one if you have a few days of history.
- **Thermostatic valves**: a thermostat (or Versatile Thermostat over TRVs) keeps
  reporting `heating` for hours while the valve only trickles to hold the setpoint, or
  while the boiler is idle. Inside every heating period, the part after the room reached
  its highest temperature is therefore not learnt from, and neither is reported heating
  while the room sits within 0.6 K below the climate's current setpoint (the valve is
  throttling). Because the setpoint is followed over time, an overnight eco regulation
  followed by the morning climb to comfort is handled correctly; history recorded without
  a setpoint falls back to a band below the heating period's top. The full heating power is also
  measured directly on real heat-ups (the room was idle, then warmed up); when the
  regression disagrees by more than 30 %, that measured rate is used. When the data
  cannot identify the heat loss, the time constant comes from free-cooling periods.
- **Plausibility**: if the learnt parameters are physically implausible (full heating
  would settle more than 150 K above the outdoor temperature, or the heat loss cannot be
  identified) the zone reports **unreliable**: no preheat value, maturity 0.
- **Preheat minutes** come from the exact solution of the model. The heat takes the dead
  time `d` to arrive, so the room first coasts (no heating) for `d` minutes to a
  temperature `T_d`; then `t = d + τ · ln((T∞ − T_d) / (T∞ − T_target))`. `T∞` is the
  highest temperature the room can reach with the heater at full effort. It uses the
  outdoor temperature averaged over the preheat window: from the weather forecast when
  available, otherwise the current outdoor value, otherwise the model's reference
  outdoor temperature. The linear model alone would let `T∞` climb far above anything a
  room reaches (40 °C and more), so the maximum reachable temperature is capped at the
  climate's `max_temp` (its highest setpoint) or, when the climate does not report one,
  at 30 °C. If that capped value is below the target, the zone reports **unreachable**.
  When the room coasts up to the target by itself during the dead time, no heating is
  needed (0 minutes). For a preheat that starts later (preheat start), the outdoor
  temperature at that time comes from the forecast, not from the current reading.
- **Maturity** compares the model's 60-minute-ahead temperature forecasts with what
  really happened and with a naive "temperature stays the same" baseline, and checks
  predicted vs observed heat-up durations.

Ideas for the dead-time detection, cooling-curve cross-check and forecast averaging come
from [Ecronika/ha-preheat](https://github.com/Ecronika/ha-preheat) (MIT, © 2025 Ecronika).

## Installation

HACS → ⋮ → **Custom repositories** → `https://github.com/giogandola98/ha-room-thermal-model`,
category **Integration** → install → restart Home Assistant.
Manual: copy `custom_components/room_thermal_model/` into your config directory.

Requires Home Assistant 2026.1+ and the `recorder` (for the history import).

## Configuration (one entry per zone)

Settings → Devices & services → Add integration → **Room Thermal Model**.

| Input | Required | Notes |
|---|---|---|
| Zone name | yes | used for entity ids |
| Climate entity | yes | `current_temperature`, `hvac_action`, target `temperature`, HVAC mode |
| Room temperature sensor | no | overrides the climate's temperature (e.g. a sensor away from the radiator) |
| Heating effort sensor (%) | no | valve opening / heating demand 0–100 %; otherwise `hvac_action` is used (heating = 100 %, idle/off = 0 %); if the climate has no `hvac_action`, "mode on and target > current" |
| Heat source | no | on while the boiler / heat pump really produces heat (binary sensor, switch, or a numeric sensor such as the burner power: > 0 = on). Recommended with a central boiler: thermostats often report `heating` while the boiler is idle. Heating counts only while it is on |
| Outdoor temperature sensor | no | current value and history |
| Weather entity | no | forecast for the preheat window (hourly → twice daily → daily) and outdoor fallback |
| Arrival sensor | no | any timestamp sensor (e.g. `sensor.house_next_return_eta` from [Home Return Predictor](https://github.com/giogandola98/ha-home-return-predictor)); enables preheat start / preheat now |

Options: target temperature (fixed, default 20.5 °C; follow the climate's target; or a
climate preset temperature), preheat preset (`comfort`), data half-life (21 days),
history window (30 days), update interval (5 min), hold after the arrival time (30 min),
maximum sensor age (360 min).

**Target modes and setback schedules.** "Follow the climate's target" uses the setpoint
the climate has *right now*. With a schedule that lowers the setpoint while nobody is
home (e.g. eco 18 °C while the room is at 19 °C), the zone sees no need to heat: preheat
minutes are 0 with reason `setpoint_below_current`, and preheat start / preheat now never
fire. After 24 hours in that state with an arrival sensor, a repair issue suggests the
preset mode. **"Thermostat preset temperature"** reads the climate's
`preset_temperatures` attribute (Versatile Thermostat): the target is
`preset_temperatures.<preset>_temp` of the configured preheat preset (default
`comfort`), whatever preset is active; without that attribute the fixed target is used.

**Setback floor.** When the climate's current setpoint is below the room temperature, the
thermostat holds it: the predicted cooling before a preheat (and the temperature at
arrival) does not go below that setpoint, so the preheat starts from the setback
temperature rather than from a room that has cooled towards the outdoor temperature.

**Dead sensors.** The indoor temperature counts as missing when the entity providing it
has not reported for longer than the maximum sensor age (`last_reported`, which also
moves when an unchanged value is reported); the zone then reports `no_temperature`. The
same limit applies, per entity, to the history import.

The history import replays the recorder: between state changes the last states are
carried forward for at most 15 minutes. Longer silences (Home Assistant down, recorder
gaps) are treated as missing data, like in live operation.
Without any outdoor source the model still works but cannot follow the weather, so
expect larger errors.

## Entities (per zone)

| Entity | Content |
|---|---|
| `sensor.<zone>_preheat_minutes` | minutes of full heating needed to reach the target; unknown while learning, when unreachable or without outdoor data (attribute `reason`, e.g. `unreachable`, `no_outdoor`) |
| `sensor.<zone>_max_reachable_temperature` | highest temperature reachable at the outdoor temperature averaged over the preheat window, capped at the climate's `max_temp` (30 °C when unknown); unknown while the model is unreliable (attribute `reliable`); the uncapped model value is in the `details` attribute of the preheat sensor as `t_infinity_c` |
| `sensor.<zone>_model_maturity` | 0–100 %; attributes `rmse_c`, `baseline_rmse_c` (1 h ahead), `rmse_4h_c`, `baseline_rmse_4h_c` (4 h ahead), `duration_mae_minutes`, `heating_hours`, `cooling_hours` |
| `sensor.<zone>_preheat_start` | latest time to start heating to be at the target on arrival (only with an arrival sensor). The room keeps cooling until then, so it is computed from the temperature the room will have at the start time, not the current one. When the target is unreachable it aims at 0.5 K below the max reachable temperature instead |
| `binary_sensor.<zone>_preheat_now` | on when it is time to start heating (only with an arrival sensor). It latches: once on it stays on until the arrival, also if the room reaches the target early or the arrival sensor is unknown for one or two updates. It is released when the arrival moves more than 30 min later than the arrival it latched on (attribute `arrival_at_latch`), when the arrival stays unavailable for more than two updates before its time (someone came home), or `arrival_hold_minutes` (option, default 30) after an arrival time that passed without anybody arriving. The latch survives a restart. Attributes: `arrival`, `preheat_start`, `arrival_at_latch`, `hold_until`, `lead_minutes` |
| `sensor.<zone>_arrival_temperature` | temperature expected at the arrival without preheating, not below the setback setpoint (only with an arrival sensor; unknown while the model is unreliable) |
| `sensor.<zone>_time_constant`, `_heating_rate`, `_dead_time` | diagnostic: learnt parameters; **unknown while the model is unreliable** (attribute `reliable`, the raw value in `raw_value`) |

When the target is unreachable, `sensor.<zone>_preheat_minutes` stays unknown with reason
`unreachable`, while preheat start and preheat now still work as described above.

The `reason` attribute of the preheat sensors explains why there is no value: for example
`setpoint_below_current` (target followed from the climate, whose current setpoint is
below the room temperature: no preheat; see the setback note above),
`unreachable` (the target is above the max reachable temperature), `no_outdoor` (the
model needs outdoor data but none is available: no sensor, no forecast and no reference
learnt yet) or `unreliable` (the learnt parameters failed the plausibility check; the
diagnostics show `unreliable_reason`).

Changing the input sensors via **Reconfigure** keeps the stored data, drops only the
learnt parameters and merges the recorder history of the new inputs over the stored data
(the recorder wins where it has data); the next fit relearns the parameters.

`retrain` merges the recorder history into the stored data (the recorder wins where it
has data, so data older than the recorder retention is kept); `replace: true` discards the
stored data and rebuilds it from the recorder only (the previous behaviour). After the
upgrade to this version the history is merged once automatically, which fills in the
climate setpoint of data recorded before the setpoint was tracked.

Services: `room_thermal_model.retrain`, `room_thermal_model.reset`,
`room_thermal_model.export_diagnostics`. Each accepts an optional `entity_id` (any entity
of the zone, multiple allowed) and an optional `config_entry_id`; a call without a target
acts on every zone. Diagnostics download is supported.

## Example: start heating in time

```yaml
automation:
  - alias: Living room preheat
    triggers:
      - trigger: state
        entity_id: binary_sensor.living_room_preheat_now
        to: "on"
    actions:
      - action: climate.set_preset_mode
        target:
          entity_id: climate.living_room
        data:
          preset_mode: comfort
```

With Home Return Predictor's VTherm blueprint, set the blueprint's "preheat minutes
sensor" input to `sensor.<zone>_preheat_minutes`.

## Development

```bash
pip install -r requirements_test.txt
pytest                     # pure model tests (no HA) + integration tests
python scripts/simulate.py --days 30   # synthetic rooms: learning curve
python -I scripts/replay/replay_fit.py diag_zone.json   # refit + score a diagnostics dump
```

The model core (`rtm_core/`) is pure standard-library Python behind a `ThermalModel`
interface, so it can be tested alone or replaced by another backend.

## Limitations

- One thermal mass per zone: adjacent rooms heating each other and strong solar gains
  are only captured on average.
- The heating effort from `hvac_action` is on/off; a real valve-opening sensor gives
  more precise results.
- Without an outdoor source, preheat estimates cannot adapt to the weather.
