"""Conductor definitions for superconducting cables."""

from dataclasses import dataclass
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from .materials import HTSTape


@dataclass
class TapeStack:
    """A block of HTS tapes stacked into a square/rectangular block."""
    tape: HTSTape
    N_tapes: int = 40
    gap: float = 0.0  # m, spacing between adjacent tapes

    @property
    def width(self) -> float:
        """Width of the tape stack, equal to the tape width."""
        return self.tape.width

    @property
    def height(self) -> float:
        """Height of the tape stack from tape stacking."""
        return self.N_tapes * self.tape.thickness + max(0.0, self.N_tapes - 1) * self.gap

    @property
    def block_side(self) -> float:
        """Equivalent square side for packing the stacked block."""
        return max(self.width, self.height)

    @property
    def A_stack(self) -> float:
        """Cross-sectional area of the stacked tapes."""
        return self.width * self.height

    def __repr__(self) -> str:
        return (f"TapeStack(N={self.N_tapes}, "
                f"{self.width*1e3:.1f}×{self.height*1e3:.1f} mm)")


@dataclass
class StackedSlotCable:
    """A stacked-slot conductor assembly with circular jacket, copper former, and central helium channel."""
    stack: TapeStack
    N_slots: int = 4  # Fixed to 4 for VIPER architecture
    jacket_outer_diameter: float = 27.7e-3  # m, outer diameter of circular SS jacket
    jacket_thick: float = 2.0e-3  # m, SS jacket thickness
    former_thick: float = 2e-3  # m, copper former thickness
    former_material: str = "Copper"
    helium_channel_diameter: float = 7.0e-3  # m, diameter of central helium cooling channel
    void_fraction: float = 0.0

    @property
    def jacket_outer_radius(self) -> float:
        """Outer radius of the circular SS jacket."""
        return self.jacket_outer_diameter / 2

    @property
    def jacket_inner_radius(self) -> float:
        """Inner radius of the SS jacket."""
        return self.jacket_outer_radius - self.jacket_thick

    @property
    def former_outer_radius(self) -> float:
        """Outer radius of the copper former."""
        return self.jacket_inner_radius

    @property
    def former_inner_radius(self) -> float:
        """Inner radius of the copper former (same as helium radius)."""
        return self.helium_radius

    @property
    def helium_radius(self) -> float:
        """Radius of the central helium cooling channel."""
        return self.helium_channel_diameter / 2

    @property
    def stack_positions(self) -> list[tuple[float, float]]:
        """(x, y) positions of the 4 HTS stacks in the former slots."""
        # Stacks at 0°, 90°, 180°, 270°, centers at 8.5 mm from origin
        angles = [0, np.pi/2, np.pi, 3*np.pi/2]
        r_stack = 8.5e-3  # 8.5 mm
        positions = []
        for theta in angles:
            x = r_stack * np.cos(theta)
            y = r_stack * np.sin(theta)
            positions.append((x, y))
        return positions

    @property
    def conduit_side(self) -> float:
        """Effective conductor side for solenoid packing (outer jacket diameter)."""
        return self.jacket_outer_diameter

    @property
    def A_conduit(self) -> float:
        """Cross-sectional area of the circular steel jacket."""
        return np.pi * self.jacket_outer_radius**2

    @property
    def A_sc_total(self) -> float:
        """Total superconducting area in the slotted assembly."""
        return self.N_slots * self.stack.N_tapes * self.stack.tape.A_sc

    @property
    def A_he(self) -> float:
        """Helium area in the central channel."""
        return np.pi * self.helium_radius**2

    def __repr__(self) -> str:
        return (f"StackedSlotCable({self.N_slots} stacks, "
                f"{self.stack.N_tapes} tapes/stack, "
                f"jacket OD={self.jacket_outer_diameter*1e3:.1f} mm)")
