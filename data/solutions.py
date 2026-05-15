"""Solution data structures for physics solvers."""

import numpy as np
from dataclasses import dataclass


@dataclass
class MagneticFieldSolution:
    """Solution from magnetic field solver."""
    r: np.ndarray  # radial grid
    z: np.ndarray  # axial grid
    Br: np.ndarray  # radial field component
    Bz: np.ndarray  # axial field component
    Bmag: np.ndarray  # field magnitude


@dataclass
class InductanceSolution:
    """Solution from inductance solver."""
    matrix: np.ndarray  # inductance matrix
    total_inductance: float  # sum of all elements
    min_eigenvalue: float  # smallest eigenvalue
    max_eigenvalue: float  # largest eigenvalue
    positive_definite: bool  # whether matrix is PD