"""Material properties for superconducting components."""

from dataclasses import dataclass


@dataclass
class HTSTape:
    """High-temperature superconducting tape properties."""
    width: float  # m
    thickness: float  # m
    t_rebco: float  # m, REBCO layer thickness
    t_copper: float  # m, copper stabilizer thickness
    t_substrate: float  # m, substrate thickness

    @property
    def A_sc(self) -> float:
        """Superconducting cross-sectional area."""
        return self.width * self.t_rebco

    @property
    def A_total(self) -> float:
        """Total tape cross-sectional area."""
        return self.width * self.thickness

    def __repr__(self) -> str:
        return (f"HTSTape(w={self.width*1e3:.1f} mm, "
                f"t={self.thickness*1e3:.2f} mm)")