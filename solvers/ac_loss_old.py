"""Geometry-aware AC-loss estimator for HTS solenoids."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import numpy as np

from data.coils import Solenoid, PancakeLayer
from solvers.magnetic_field import BFieldSolver
from utils.constants import MU0


@dataclass
class TurnACLoss:
    """Per-turn AC-loss estimate for one winding turn."""
    turn_index: int
    r_m: float
    z_m: float
    B_local_T: float
    Ic_local_A: float
    loss_per_cycle_J_per_m: float
    power_W_per_m: float


@dataclass
class ACLossSolution:
    """Total AC-loss estimate for the full solenoid."""
    current_A: float
    ramp_rate_A_per_s: float
    temperature_K: float
    turns: List[TurnACLoss] = field(default_factory=list)

    @property
    def total_loss_per_cycle_J_per_m(self) -> float:
        return float(sum(turn.loss_per_cycle_J_per_m for turn in self.turns))

    @property
    def total_power_W_per_m(self) -> float:
        return float(sum(turn.power_W_per_m for turn in self.turns))

    @property
    def peak_turn_B_T(self) -> float:
        return float(max((turn.B_local_T for turn in self.turns), default=0.0))

    @property
    def cycle_period_s(self) -> float:
        if abs(self.ramp_rate_A_per_s) < 1e-12:
            return np.inf
        return abs(2.0 * self.current_A / self.ramp_rate_A_per_s)

    @property
    def cycles_per_second(self) -> float:
        period = self.cycle_period_s
        if not np.isfinite(period) or period <= 0.0:
            return 0.0
        return 1.0 / period


class ACLossSolver:
    """First-order, geometry-aware AC-loss estimator for a current-ramped HTS coil.

    The model is intentionally based on the actual solenoid geometry and the local field
    sampled at each turn center. If the cable arrangement or winding geometry changes,
    the field map and the per-turn loss estimates automatically update by re-running the
    solver on the new Solenoid object.
    """

    def __init__(self, solenoid: Solenoid):
        self.sol = solenoid

    def _field_map(self, filament_grid_size: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the field-map arrays for the current solenoid geometry."""
        solver = BFieldSolver(self.sol)
        field_sol = solver.compute(
            r_max=self.sol.r_outer * 2.0,
            z_max=self.sol.height * 0.75,
            nr=60,
            nz=80,
            filament_grid_size=filament_grid_size,
        )
        return field_sol.r, field_sol.z, field_sol.Bmag

    def _field_at_turn(self, turn: PancakeLayer, r_grid: np.ndarray, z_grid: np.ndarray, Bmag: np.ndarray) -> float:
        """Read the local field magnitude at the supplied turn center."""
        r_idx = int(np.argmin(np.abs(r_grid - turn.r_center)))
        z_idx = int(np.argmin(np.abs(z_grid - turn.z_center)))
        return float(Bmag[r_idx, z_idx])

    def _per_turn_hysteresis_loss(
        self,
        turn: PancakeLayer,
        current_A: float,
        ramp_rate_A_per_s: float,
        temperature_K: float,
        r_grid: np.ndarray,
        z_grid: np.ndarray,
        Bmag: np.ndarray,
    ) -> TurnACLoss:
        """Estimate hysteresis loss for a single turn using a first-order HTS slab model."""
        tape = self.sol.cable.stack.tape
        B_local = self._field_at_turn(turn, r_grid, z_grid, Bmag)
        Ic_local = tape.critical_current(temperature_K, B_local)

        current_ratio = abs(current_A) / max(Ic_local, 1.0)
        q_cycle = (2.0 / 3.0) * MU0 * (tape.width * tape.t_rebco) * max(B_local, 1e-6) * (current_ratio ** 2)

        if not np.isfinite(q_cycle) or q_cycle < 0.0:
            q_cycle = 0.0

        if abs(ramp_rate_A_per_s) < 1e-12:
            power = 0.0
        else:
            cycle_period = abs(2.0 * current_A / ramp_rate_A_per_s)
            cycles_per_second = 1.0 / max(cycle_period, 1e-12)
            power = q_cycle * cycles_per_second

        return TurnACLoss(
            turn_index=0,
            r_m=turn.r_center,
            z_m=turn.z_center,
            B_local_T=B_local,
            Ic_local_A=Ic_local,
            loss_per_cycle_J_per_m=q_cycle,
            power_W_per_m=power,
        )

    def compute(self, current_A: float, ramp_rate_A_per_s: float, temperature_K: float = 20.0) -> ACLossSolution:
        """Compute total AC loss for the current solenoid geometry."""
        r_grid, z_grid, Bmag = self._field_map()
        turn_losses: List[TurnACLoss] = []
        for idx, turn in enumerate(self.sol.layers):
            result = self._per_turn_hysteresis_loss(
                turn,
                current_A,
                ramp_rate_A_per_s,
                temperature_K,
                r_grid,
                z_grid,
                Bmag,
            )
            result.turn_index = idx
            turn_losses.append(result)

        return ACLossSolution(
            current_A=current_A,
            ramp_rate_A_per_s=ramp_rate_A_per_s,
            temperature_K=temperature_K,
            turns=turn_losses,
        )
