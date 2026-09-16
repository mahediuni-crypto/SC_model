"""Geometry builders for solenoid coils."""

import numpy as np
import sys
import os
from dataclasses import dataclass
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.coils import Solenoid, PancakeLayer
from data.conductors import StackedSlotCable
from solvers.magnetic_field import BFieldSolver
from utils.constants import MU0


@dataclass
class TargetFieldSolution:
    """Summary of a converged target-field design."""
    solenoid: Solenoid
    target_field: float
    achieved_field: float
    current: float
    inner_radius: float
    outer_radius: float
    turns: int
    radial_layers: int
    axial_layers: int

    @property
    def n_radial(self) -> int:
        return self.radial_layers

    @property
    def n_axial(self) -> int:
        return self.axial_layers

    @property
    def total_turns(self) -> int:
        return self.turns

    @property
    def summary(self) -> dict:
        amp_turns = self.current * self.turns
        return {
            "target_field_T": self.target_field,
            "achieved_field_T": self.achieved_field,
            "current_A": self.current,
            "inner_radius_m": self.inner_radius,
            "outer_radius_m": self.outer_radius,
            "turns": self.turns,
            "radial_layers": self.radial_layers,
            "axial_layers": self.axial_layers,
            "ampere_turns_total": amp_turns,
            "ampere_turns_MAt": amp_turns / 1e6,
        }


class SolenoidBuilder:
    """Builder for solenoid geometry from target field."""

    @staticmethod
    def from_target_field(
        cable: StackedSlotCable,
        r_inner: float,
        height: float,
        B_target: float,
        current: float
    ) -> Solenoid:
        """Build solenoid geometry to achieve target field."""
        # Compute total turns needed
        NI_needed = B_target * height / MU0
        N_total = NI_needed / current

        # Compute layers
        d = cable.conduit_side
        n_axial = max(1, round(height / d))
        n_radial = max(1, int(np.ceil(N_total / n_axial)))
        r_outer = r_inner + n_radial * d

        # Build solenoid
        sol = Solenoid(
            cable=cable,
            r_inner=r_inner,
            height=height,
        )
        sol.r_outer = r_outer
        sol.n_radial = n_radial
        sol.n_axial = n_axial

        # Generate layers
        sol.layers = SolenoidBuilder.build_turns(sol, current)

        return sol

    @staticmethod
    def _build_from_counts(
        cable: StackedSlotCable,
        r_inner: float,
        height: float,
        current: float,
        n_radial: int,
        n_axial: int,
    ) -> Solenoid:
        """Create a solenoid from explicit radial and axial turn counts."""
        d = cable.conduit_side
        sol = Solenoid(
            cable=cable,
            r_inner=r_inner,
            height=height,
        )
        sol.r_outer = r_inner + n_radial * d
        sol.n_radial = n_radial
        sol.n_axial = n_axial
        sol.layers = SolenoidBuilder.build_turns(sol, current)
        return sol

    @staticmethod
    def _turn_filament_grid(turn: PancakeLayer, current: float, n_side: int = 5):
        """Distribute the turn current over a finite-sized filament grid.

        This avoids the 1/r singularity of a single filament model when the
        evaluation point is taken at the turn surface.
        """
        area = max(turn.cable.A_sc_total / max(1, turn.cable.N_slots), 1e-18)
        current_density = current / area
        half_side = 0.5 * turn.cable.stack.block_side
        filaments = []
        for i in range(n_side):
            for j in range(n_side):
                rx = (-0.5 + (i + 0.5) / n_side) * 2.0 * half_side
                rz = (-0.5 + (j + 0.5) / n_side) * 2.0 * half_side
                r_f = turn.r_center + rx
                z_f = turn.z_center + rz
                dA = area / (n_side * n_side)
                filament_current = current_density * dA
                filaments.append((r_f, z_f, filament_current))
        return filaments

    @staticmethod
    def _peak_field_at_innermost_turn(sol: Solenoid, current: float | None = None) -> float:
        """Field at the inner edge of the innermost turn in the mid-plane.

        This is the correct physical location for the peak field: the innermost
        radial turn at z = 0, evaluated on the inside edge of the HTS stack.
        """
        if not sol.layers:
            return 0.0

        min_r = min(lay.r_center for lay in sol.layers)
        innermost_turns = [lay for lay in sol.layers if np.isclose(lay.r_center, min_r, rtol=1e-9, atol=1e-12)]
        turn = min(innermost_turns, key=lambda t: abs(t.z_center))
        stack_half_side = 0.5 * sol.cable.stack.block_side
        eval_r = turn.r_center - stack_half_side
        eval_z = turn.z_center

        field_solver = BFieldSolver(sol)
        R = np.array([[eval_r]])
        Z = np.array([[eval_z]])

        total_b = 0.0
        for lay in sol.layers:
            current_val = lay.current if current is None else current
            for r_f, z_f, I_f in SolenoidBuilder._turn_filament_grid(lay, current_val, n_side=9):
                br, bz = field_solver._loop_field(r_f, z_f, I_f, R, Z)
                total_b += np.hypot(br[0, 0], bz[0, 0])
        return float(total_b)

    @staticmethod
    def _design_sweep_rows(
        cable: StackedSlotCable,
        r_inner: float,
        height: float,
        current: float,
        B_target: float,
        metric: str = "peak_inner_turn",
    ) -> list[dict]:
        """Evaluate the full valid design sweep for Nr=8..15 and Nz=48..54."""
        d = cable.conduit_side
        max_axial_turns = max(1, int(np.floor(1.5 / d)))
        axial_candidates = [n for n in range(48, max_axial_turns + 1) if n * d <= 1.5 + 1e-9]
        if not axial_candidates:
            axial_candidates = [max_axial_turns]

        rows = []
        for n_axial in axial_candidates:
            for n_radial in range(8, 16):
                sol_height = min(float(height), n_axial * d)
                sol = SolenoidBuilder._build_from_counts(
                    cable=cable,
                    r_inner=r_inner,
                    height=sol_height,
                    current=current,
                    n_radial=n_radial,
                    n_axial=n_axial,
                )
                if metric == "center":
                    field_solver = BFieldSolver(sol)
                    field_sol = field_solver.compute(r_max=sol.r_outer * 2.0, z_max=sol_height * 0.75, nr=120, nz=180, filament_grid_size=9)
                    z_idx = field_sol.z.shape[0] // 2
                    field_value = abs(field_sol.Bz[0, z_idx])
                elif metric == "peak":
                    field_solver = BFieldSolver(sol)
                    field_sol = field_solver.compute(r_max=sol.r_outer * 2.0, z_max=sol_height * 0.75, nr=120, nz=180, filament_grid_size=9)
                    field_value = float(np.nanmax(np.nan_to_num(field_sol.Bmag, nan=0.0)))
                else:
                    field_value = SolenoidBuilder._peak_field_at_innermost_turn(sol, current=current)
                rows.append({
                    "n_radial": n_radial,
                    "n_axial": n_axial,
                    "turns": sol.total_turns,
                    "r_outer_m": sol.r_outer,
                    "height_m": sol.height,
                    "peak_field_T": field_value,
                    "NI_MAt": (sol.total_turns * current) / 1e6,
                    "delta_target_T": abs(field_value - B_target),
                })
        return rows

    @staticmethod
    def solve_to_target_field(
        cable: StackedSlotCable,
        r_inner: float,
        height: float,
        B_target: float,
        current: float,
        tol: float = 1e-3,
        max_iter: int = 80,
        metric: str = "peak_inner_turn",
    ) -> TargetFieldSolution:
        """Iterate over realistic radial/axial layouts until the peak-turn field reaches the target."""
        d = cable.conduit_side
        max_axial_turns = max(1, int(np.floor(1.5 / d)))
        axial_candidates = [n for n in range(48, max_axial_turns + 1) if n * d <= 1.5 + 1e-9]
        if not axial_candidates:
            axial_candidates = [max_axial_turns]

        current_eff = float(current)
        r_inner_eff = float(r_inner)

        if metric not in {"peak_inner_turn", "center", "peak"}:
            raise ValueError("metric must be 'peak_inner_turn', 'center', or 'peak'")

        def build_and_eval(n_radial: int, n_axial: int, current_value: float, r_value: float) -> tuple[Solenoid, float]:
            sol_height = min(float(height), n_axial * d)
            sol = SolenoidBuilder._build_from_counts(
                cable=cable,
                r_inner=r_value,
                height=sol_height,
                current=current_value,
                n_radial=n_radial,
                n_axial=n_axial,
            )
            if metric == "center":
                field_solver = BFieldSolver(sol)
                field_sol = field_solver.compute(r_max=sol.r_outer * 2.0, z_max=sol_height * 0.75, nr=120, nz=180, filament_grid_size=9)
                z_idx = field_sol.z.shape[0] // 2
                field_value = abs(field_sol.Bz[0, z_idx])
                return sol, field_value
            if metric == "peak":
                field_solver = BFieldSolver(sol)
                field_sol = field_solver.compute(r_max=sol.r_outer * 2.0, z_max=sol_height * 0.75, nr=120, nz=180, filament_grid_size=9)
                field_value = float(np.nanmax(np.nan_to_num(field_sol.Bmag, nan=0.0)))
                return sol, field_value
            field_value = SolenoidBuilder._peak_field_at_innermost_turn(sol, current=current_value)
            return sol, field_value

        sweep_rows = SolenoidBuilder._design_sweep_rows(cable, r_inner_eff, height, current_eff, B_target, metric=metric)
        print("\nPeak-field sweep table (valid range: Nr 8..15, Nz 48..54):")
        print(f"{'Nr':>3} {'Nz':>3} {'turns':>6} {'outer_m':>8} {'height_m':>9} {'Bpeak_T':>10} {'NI_MAt':>9} {'delta_T':>9}")
        for row in sorted(sweep_rows, key=lambda r: (r["delta_target_T"], r["n_radial"], r["n_axial"])):
            print(f"{row['n_radial']:>3} {row['n_axial']:>3} {row['turns']:>6} {row['r_outer_m']:>8.4f} {row['height_m']:>9.3f} {row['peak_field_T']:>10.3f} {row['NI_MAt']:>9.3f} {row['delta_target_T']:>9.3f}")

        best_sol = None
        best_field = np.inf
        best_row = None
        for row in sweep_rows:
            if row["delta_target_T"] < abs(best_field - B_target):
                best_row = row
                best_field = row["peak_field_T"]
                best_sol = SolenoidBuilder._build_from_counts(
                    cable=cable,
                    r_inner=r_inner_eff,
                    height=min(float(height), row["n_axial"] * d),
                    current=current_eff,
                    n_radial=row["n_radial"],
                    n_axial=row["n_axial"],
                )

        for _ in range(max_iter):
            for n_axial in axial_candidates:
                for n_radial in range(8, 16):
                    sol, field_value = build_and_eval(n_radial, n_axial, current_eff, r_inner_eff)
                    if abs(field_value - B_target) < abs(best_field - B_target):
                        best_sol = sol
                        best_field = field_value

                    if abs(field_value - B_target) <= tol * max(abs(B_target), 1.0):
                        print(f"\nSelected design: Nr={sol.n_radial}, Nz={sol.n_axial}, turns={sol.total_turns}, Bpeak={field_value:.3f} T")
                        return TargetFieldSolution(
                            solenoid=sol,
                            target_field=B_target,
                            achieved_field=field_value,
                            current=current_eff,
                            inner_radius=r_inner_eff,
                            outer_radius=sol.r_outer,
                            turns=sol.total_turns,
                            radial_layers=sol.n_radial,
                            axial_layers=sol.n_axial,
                        )

            if best_field == np.inf:
                break
            target_ratio = B_target / max(best_field, 1e-12)
            if best_field < B_target:
                current_eff *= min(1.5, max(1.05, target_ratio))
            else:
                current_eff *= max(0.7, min(0.98, target_ratio))

        if best_sol is None:
            raise RuntimeError("Target-field iteration failed to initialize a valid solenoid.")
        print(f"\nSelected design: Nr={best_sol.n_radial}, Nz={best_sol.n_axial}, turns={best_sol.total_turns}, Bpeak={best_field:.3f} T")
        return TargetFieldSolution(
            solenoid=best_sol,
            target_field=B_target,
            achieved_field=best_field,
            current=current_eff,
            inner_radius=r_inner_eff,
            outer_radius=best_sol.r_outer,
            turns=best_sol.total_turns,
            radial_layers=best_sol.n_radial,
            axial_layers=best_sol.n_axial,
        )

    @staticmethod
    def build_turns(sol: Solenoid, current: float) -> list[PancakeLayer]:
        """Generate list of pancake layers."""
        layers = []
        d = sol.cable.conduit_side
        for i in range(sol.n_radial):
            r = sol.r_inner + (i + 0.5) * d
            for j in range(sol.n_axial):
                z = -sol.height / 2 + (j + 0.5) * d
                layers.append(PancakeLayer(
                    cable=sol.cable,
                    r_center=round(r, 8),
                    z_center=round(z, 8),
                    current=current,
                ))
        return layers