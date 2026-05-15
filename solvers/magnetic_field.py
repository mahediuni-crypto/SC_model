"""Magnetic field solver for solenoid coils."""

import numpy as np
from scipy.special import ellipk, ellipe
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.coils import Solenoid
from data.solutions import MagneticFieldSolution
from utils.constants import MU0


class BFieldSolver:
    """Computes B(r,z) using elliptic integrals for circular loops."""

    def __init__(self, solenoid: Solenoid):
        self.sol = solenoid

    def compute(
        self,
        r_max: float = None,
        z_max: float = None,
        nr: int = 60,
        nz: int = 80
    ) -> MagneticFieldSolution:
        """Compute magnetic field solution."""
        if r_max is None:
            r_max = self.sol.r_outer * 2.0
        if z_max is None:
            z_max = self.sol.height * 0.75

        r_1d = np.linspace(0, r_max, nr)
        z_1d = np.linspace(-z_max, z_max, nz)
        R, Z = np.meshgrid(r_1d, z_1d, indexing='ij')

        Br_tot = np.zeros((nr, nz))
        Bz_tot = np.zeros((nr, nz))

        n = len(self.sol.layers)
        for lay in self.sol.layers:
            br, bz = self._loop_field(lay.r_center, lay.z_center,
                                       lay.current, R, Z)
            Br_tot += br
            Bz_tot += bz

        B_mag = np.sqrt(Br_tot**2 + Bz_tot**2)

        return MagneticFieldSolution(
            r=r_1d,
            z=z_1d,
            Br=Br_tot,
            Bz=Bz_tot,
            Bmag=B_mag
        )

    def _loop_field(self, a: float, z0: float, I: float,
                    R: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Field from single loop."""
        r = np.where(R < 1e-9, 1e-9, R)
        dz = Z - z0
        Q = (a + r)**2 + dz**2
        D = (a - r)**2 + dz**2
        k2 = np.clip(4 * a * r / Q, 0, 1 - 1e-10)
        K = ellipk(k2)
        E = ellipe(k2)
        sqQ = np.sqrt(Q)
        pre = MU0 * I / (2 * np.pi)
        Bz = pre / sqQ * (K + (a**2 - r**2 - dz**2) / D * E)
        Br = pre * dz / (r * sqQ) * (-K + (a**2 + r**2 + dz**2) / D * E)
        Br = np.where(R < 1e-9, 0.0, Br)
        return Br, Bz