"""Geometry builders for solenoid coils."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.coils import Solenoid, PancakeLayer
from data.conductors import StackedSlotCable
from utils.constants import MU0


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