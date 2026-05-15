"""Inductance matrix solver for solenoid coils."""

import numpy as np
from scipy.special import ellipk, ellipe
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.coils import Solenoid
from data.solutions import InductanceSolution
from utils.constants import MU0


class InductanceSolver:
    """Computes inductance matrix from solenoid geometry."""

    def __init__(self, solenoid: Solenoid):
        self.sol = solenoid

    def compute(self) -> InductanceSolution:
        """Compute inductance solution."""
        n = len(self.sol.layers)
        L = np.zeros((n, n), dtype=float)
        for i, lay_i in enumerate(self.sol.layers):
            for j in range(i, n):
                lay_j = self.sol.layers[j]
                if i == j:
                    L_val = self._loop_self_inductance(lay_i.r_center)
                else:
                    dz = abs(lay_i.z_center - lay_j.z_center)
                    L_val = self._loop_mutual_inductance(
                        lay_i.r_center, lay_j.r_center, dz)
                L[i, j] = L_val
                L[j, i] = L_val

        total_inductance = np.sum(L)
        eig = np.linalg.eigvalsh(L)
        min_ev, max_ev = eig[0], eig[-1]
        positive_definite = min_ev > 1e-15

        return InductanceSolution(
            matrix=L,
            total_inductance=total_inductance,
            min_eigenvalue=min_ev,
            max_eigenvalue=max_ev,
            positive_definite=positive_definite
        )

    def _loop_self_inductance(self, a: float) -> float:
        """Self inductance of circular loop."""
        cable = self.sol.cable
        if hasattr(cable, "stack"):
            tape = cable.stack.tape
            n_tapes = cable.stack.N_tapes
        else:
            tape = cable.tape
            n_tapes = cable.N_tapes

        area = tape.width * tape.thickness * n_tapes
        r_eq = np.sqrt(area / np.pi)
        return MU0 * a * (np.log(8.0 * a / r_eq) - 2.0)

    def _loop_mutual_inductance(self, a: float, b: float, dz: float) -> float:
        """Mutual inductance between two loops."""
        Q = (a + b)**2 + dz**2
        k2 = np.clip(4.0 * a * b / Q, 0.0, 1.0 - 1e-12)
        k = np.sqrt(k2)
        K = ellipk(k2)
        E = ellipe(k2)
        pre = MU0 / np.pi * np.sqrt(a * b)
        return pre * ((2.0 / k - k) * K - 2.0 / k * E)