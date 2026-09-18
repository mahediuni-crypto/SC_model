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
        """(x, y) positions of the HTS stacks in the former slots."""
        return [(x, y) for x, y, _ in self.stack_geometry]

    @property
    def stack_geometry(self) -> list[tuple[float, float, float]]:
        """Return stack centers and radial orientation angles in radians."""
        angles = np.arange(self.N_slots) * 2.0 * np.pi / self.N_slots
        r_stack = self.slot_radius
        geometry = []
        for theta in angles:
            x = r_stack * np.cos(theta)
            y = r_stack * np.sin(theta)
            geometry.append((x, y, theta))
        return geometry

    @property
    def slot_radius(self) -> float:
        """Approximate slot-center radius for the selected slot count."""
        return 8.5e-3 if self.N_slots <= 4 else 7.5e-3

    @property
    def stack_clearance_ok(self) -> bool:
        """Check radial and tangential clearance of the oriented stacks."""
        half_radial = 0.5 * self.stack.height
        half_tangential = 0.5 * self.stack.width
        for cx, cy, theta in self.stack_geometry:
            radial_center = np.hypot(cx, cy)
            if radial_center - half_radial < self.helium_radius - 1e-12:
                return False
            if radial_center + half_radial > self.jacket_inner_radius + 1e-12:
                return False
            if half_tangential > radial_center * np.sin(np.pi / self.N_slots) + 1e-12:
                return False
        return True

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

    def critical_current(self, temperature: float, field: float) -> float:
        """Estimate total cable Ic from the Hermes tape model."""
        tape_ic = self.stack.tape.critical_current(temperature, field)
        return self.N_slots * self.stack.N_tapes * tape_ic

    @classmethod
    def sized_for_current(
        cls,
        tape: HTSTape,
        target_current: float,
        temperature: float,
        field: float,
        margin: float = 0.20,
        slot_candidates: tuple[int, ...] = (4, 6, 8),
        **kwargs,
    ) -> "StackedSlotCable":
        """Select the smallest geometrically valid cable meeting a current margin."""
        required_ic = target_current * (1.0 + margin)
        candidates = []
        for n_slots in slot_candidates:
            tapes_needed = int(np.ceil(required_ic / tape.critical_current(temperature, field) / n_slots))
            cable = cls(
                stack=TapeStack(tape=tape, N_tapes=tapes_needed),
                N_slots=n_slots,
                **kwargs,
            )
            if cable.stack_clearance_ok and cable.critical_current(temperature, field) >= required_ic:
                candidates.append(cable)
        if not candidates:
            raise ValueError("No candidate slot layout fits the jacket and current requirement")
        return min(candidates, key=lambda candidate: (candidate.N_slots * candidate.stack.N_tapes, candidate.N_slots))

    def __repr__(self) -> str:
        return (f"StackedSlotCable({self.N_slots} stacks, "
                f"{self.stack.N_tapes} tapes/stack, "
                f"jacket OD={self.jacket_outer_diameter*1e3:.1f} mm)")
