"""Thermal model contract shared by the Home Assistant glue and model implementations.

Pure standard-library Python: never import Home Assistant here or anywhere in
``rtm_core``. Only relative imports inside the package. All datetimes are timezone-aware.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Tunable parameters (mirrors the options flow)."""

    step_minutes: float = 5.0
    half_life_days: float = 21.0
    max_dead_time_minutes: float = 90.0
    max_gap_minutes: float = 15.0
    max_step_delta_k: float = 1.5
    history_days: float = 30.0
    # Outdoor temperature changes slowly and is often reported sparsely (weather entity):
    # its last value is carried forward for longer than indoor/effort values.
    outdoor_max_gap_minutes: float = 180.0


@dataclass(frozen=True)
class Sample:
    """One observation, already resampled on the model grid by the caller or the model.

    ``effort`` is the heating effort in [0, 1] (None = unknown). ``outdoor_c`` may be None
    when no outdoor source is configured or the value is missing. ``excluded`` marks
    periods that must not be used for learning (e.g. climate unavailable). ``setpoint_c``
    is the climate target temperature at this instant (None when the climate has no
    ``temperature`` attribute); it lets the model detect a thermostatic valve throttling
    as "the room is close below its current setpoint" instead of "close to a run's top".
    """

    ts: datetime
    indoor_c: float | None
    outdoor_c: float | None
    effort: float | None
    excluded: bool = False
    setpoint_c: float | None = None


@dataclass(frozen=True)
class ForecastPoint:
    """Outdoor temperature forecast point."""

    ts: datetime
    outdoor_c: float


@dataclass(frozen=True)
class ThermalParameters:
    """Learnt parameters of the RC model."""

    heating_rate_k_per_h: float  # a
    loss_rate_per_h: float  # b (time constant = 1/b)
    gains_k_per_h: float  # c
    dead_time_minutes: float  # d
    reliable: bool = True  # False: implausible fit (no preheat, parameters not shown)

    @property
    def time_constant_hours(self) -> float:
        return 1.0 / self.loss_rate_per_h if self.loss_rate_per_h > 0 else float("inf")


@dataclass
class PreheatEstimate:
    """Output of ``ThermalModel.preheat``."""

    minutes: float | None  # None when unreachable or model not usable
    reachable: bool
    target_c: float
    start_c: float | None
    outdoor_used_c: float | None  # effective (forecast-averaged) outdoor temperature
    max_reachable_c: float | None  # T_infinity at full effort, capped (max setpoint)
    reason: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """How well the model knows this zone."""

    maturity: float  # 0..1
    samples: int
    heating_hours: float
    cooling_hours: float
    rmse_c: float | None  # 60-min-ahead simulated forecast RMSE
    baseline_rmse_c: float | None  # persistence baseline RMSE
    duration_mae_minutes: float | None  # observed heat-up episodes
    episodes: int = 0
    rmse_4h_c: float | None = None  # 4-h-ahead simulated forecast RMSE
    baseline_rmse_4h_c: float | None = None


class ThermalModel(ABC):
    """Abstract per-zone thermal model.

    Lifecycle used by the integration:
      1. ``load_history`` on bootstrap / retrain (raw samples, any spacing; the model
         resamples to its grid). ``replace=False`` merges with what is already stored
         (the history wins where it has data).
      2. ``add_sample`` for every live observation (any spacing).
      3. ``fit_due`` / ``fit`` periodically (fit is CPU work: caller runs it in an executor).
      4. ``preheat`` / ``coast`` / ``parameters`` / ``quality`` (pure, cheap).
      5. ``to_dict`` / ``from_dict`` (JSON-safe) for persistence.
    """

    @abstractmethod
    def load_history(self, samples: Iterable[Sample], replace: bool = False) -> None:
        """Ingest historical samples."""

    @abstractmethod
    def add_sample(self, sample: Sample) -> None:
        """Ingest one live sample (cheap)."""

    @abstractmethod
    def fit_due(self, now: datetime) -> bool:
        """True when a refit is worthwhile (new data since last fit, ≥ 1 h)."""

    @abstractmethod
    def fit(self, now: datetime) -> dict[str, Any]:
        """Refit parameters and quality; returns a short summary. CPU-bound."""

    @abstractmethod
    def parameters(self) -> ThermalParameters | None:
        """Current parameters, or None before the first successful fit."""

    @abstractmethod
    def preheat(
        self,
        now: datetime,
        start_c: float,
        target_c: float,
        outdoor_c: float | None,
        forecast: Sequence[ForecastPoint] = (),
        max_c: float | None = None,
    ) -> PreheatEstimate:
        """Minutes of full-effort heating needed to reach ``target_c`` from ``start_c``.

        ``max_c`` caps the maximum reachable temperature (e.g. the climate's
        ``max_temp``); the model applies its own default ceiling when None.
        """

    @abstractmethod
    def coast(
        self,
        now: datetime,
        start_c: float,
        minutes: float,
        outdoor_c: float | None,
        forecast: Sequence[ForecastPoint] = (),
        floor_c: float | None = None,
    ) -> float | None:
        """Predicted temperature after ``minutes`` without heating.

        ``floor_c``: the thermostat holds this setpoint (a setback), so the room does not
        cool below it. None when the model is not usable (not fitted, unreliable).
        """

    @abstractmethod
    def quality(self) -> QualityReport:
        """Return the quality / maturity report."""

    @abstractmethod
    def reset(self) -> None:
        """Forget everything learnt."""

    def reset_parameters(self) -> None:
        """Forget the learnt parameters but keep the data (default: forget everything)."""
        self.reset()

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable state."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any], config: ModelConfig) -> ThermalModel:
        """Restore from ``to_dict`` output."""
