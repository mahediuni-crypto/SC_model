"""Inductance matrix solver for solenoid coils using the filament method."""

import numpy as np
from scipy.special import ellipk, ellipe
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.coils import Solenoid
from data.solutions import InductanceSolution
from utils.constants import MU0
from magnetic_field import BFieldSolver #We use the premade geometry


class InductanceSolver:
    """Computes inductance matrix from solenoid geometry using distributed filaments."""

    def __init__(self, solenoid: Solenoid, filament_grid_size: int = 4):
        self.sol = solenoid
        self.n_div = filament_grid_size
        
        # Instantiation of the other solver to get its geometry  
        self.geo_solver = BFieldSolver(solenoid)

    def compute(self) -> InductanceSolution:
        n = len(self.sol.layers)
        L = np.zeros((n, n), dtype=float)
        
        # Compute the filaments for each turn in the solenoid
        turn_filaments = []
        for lay in self.sol.layers:
            # self.geo_solver._turn_filament_grid returns a list of tuples (r, z, I)
            coils = self.geo_solver._turn_filament_grid(lay, n_div=self.n_div)
            R_array = np.array([f[0] for f in coils])
            Z_array = np.array([f[1] for f in coils])
            turn_filaments.append((R_array, Z_array, lay))

        # Inductance matrix filling
        for idx in range(n):
            R_i, Z_i, lay_i = turn_filaments[idx]
            
            for jdx in range(idx, n):
                R_j, Z_j, lay_j = turn_filaments[jdx]
                
                # Cross grid to vectorize the calculation
                r_a, r_b = np.meshgrid(R_i, R_j, indexing='ij')
                z_a, z_b = np.meshgrid(Z_i, Z_j, indexing='ij')
                dz = np.abs(z_a - z_b)
                
                # Mutual inductances matrix  
                M_matrix = self._loop_mutual_inductance(r_a, r_b, dz)
                
                if idx == jdx:
                    # Formula to avoid the singularity on the diagonal  
                    w_cell = lay_i.cable.stack.width / self.n_div
                    h_cell = lay_i.cable.stack.height / self.n_div
                    area_f = w_cell * h_cell
                    r_eq_f = np.sqrt(area_f / np.pi)
                    
                    L_self_filaments = MU0 * R_i * (np.log(8.0 * R_i / r_eq_f) - 2.0)
                    
                    np.fill_diagonal(M_matrix, L_self_filaments)

                # Macro value  
                L_val = np.mean(M_matrix)
                
                L[idx, jdx] = L_val
                L[jdx, idx] = L_val

        total_inductance = np.sum(L)
        eig = np.linalg.eigvalsh(L)
        min_ev, max_ev = eig[0], eig[-1]

        return InductanceSolution(
            matrix=L,
            total_inductance=total_inductance,
            min_eigenvalue=min_ev,
            max_eigenvalue=max_ev,
            positive_definite=(min_ev > 1e-15)
        )

    def _loop_mutual_inductance(self, a, b, dz):
        """
        Mutual inductance between two loops. 
        for scalars or NumPy arrays (vectorized).
        """
        Q = (a + b)**2 + dz**2
        k2 = np.clip(4.0 * a * b / Q, 0.0, 1.0 - 1e-12)
        k = np.sqrt(k2)
        K = ellipk(k2)
        E = ellipe(k2)
        pre = (MU0 / np.pi) * np.sqrt(a * b)
        return pre * ((2.0 / k - k) * K - 2.0 / k * E)