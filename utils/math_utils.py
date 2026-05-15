"""Mathematical utilities for physics solvers."""

import numpy as np
from scipy.special import ellipk, ellipe


def elliptic_integrals(k2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute complete elliptic integrals K and E."""
    return ellipk(k2), ellipe(k2)