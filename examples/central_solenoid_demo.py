"""Demo script for central solenoid model."""

import warnings
import numpy as np
from ..data.materials import HTSTape
from ..data.conductors import TapeStack, StackedSlotCable
from ..geometry.solenoid_builder import SolenoidBuilder
from ..solvers.magnetic_field import BFieldSolver
from ..solvers.inductance import InductanceSolver
from ..visualization.geometry_plots import plot_geometry_overview
from ..visualization.field_plots import plot_bfield

warnings.filterwarnings("ignore")


def main():
    # Create conductor
    tape = HTSTape(
        width=4e-3,
        thickness=0.1e-3,
        t_rebco=1e-6,
        t_copper=20e-6,
        t_substrate=50e-6,
    )

    stack = TapeStack(
        tape=tape,
        N_tapes=40,
        gap=0.0,
    )

    cable = StackedSlotCable(
        stack=stack,
        N_slots=4,
        jacket_outer_diameter=27.7e-3,
        jacket_thick=2.0e-3,
        former_thick=2e-3,
        former_material="Copper",
        helium_channel_diameter=7.0e-3,
    )

    # Build geometry
    sol = SolenoidBuilder.from_target_field(
        cable=cable,
        r_inner=0.50,
        height=1.50,
        B_target=15.0,
        current=50e3,
    )

    sol.summary()

    # Run inductance solver
    inductance_solver = InductanceSolver(sol)
    inductance_sol = inductance_solver.compute()
    print(f"Inductance matrix shape: {inductance_sol.matrix.shape}")
    print(f"Total series inductance: {inductance_sol.total_inductance:.6f} H")
    print(f"Positive definite: {inductance_sol.positive_definite}")
    print(f"Eigenvalues: min={inductance_sol.min_eigenvalue:.3e}, "
          f"max={inductance_sol.max_eigenvalue:.3e}")

    # Save matrix
    np.savetxt("inductance_matrix.csv", inductance_sol.matrix, delimiter=",")
    print("Saved: inductance_matrix.csv")

    # Preview matrix
    preview_size = min(8, inductance_sol.matrix.shape[0])
    print(f"Showing top-left {preview_size}×{preview_size} block:")
    with np.printoptions(precision=3, suppress=False,
                         formatter={'float_kind':'{:.3e}'.format}, threshold=1000):
        print(inductance_sol.matrix[:preview_size, :preview_size])

    # Plot geometry
    plot_geometry_overview(sol, cable)

    # Run magnetic field solver
    bfield_solver = BFieldSolver(sol)
    bfield_sol = bfield_solver.compute(
        r_max=sol.r_outer * 2.0,
        z_max=sol.height * 0.75,
        nr=60,
        nz=80,
    )
    print(f"Peak |B|: {bfield_sol.Bmag.max():.2f} T")
    print(f"Bz on axis (z=0): {abs(bfield_sol.Bz[0, bfield_sol.z.shape[0]//2]):.2f} T")

    # Plot field
    plot_bfield(bfield_sol, sol)


if __name__ == "__main__":
    main()