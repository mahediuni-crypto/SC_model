"""Coil and solenoid data structures."""

from dataclasses import dataclass, field
from typing import List
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from .conductors import StackedSlotCable


@dataclass
class PancakeLayer:
    """A single circular current loop in the solenoid."""
    cable: StackedSlotCable
    r_center: float  # m, radial center position
    z_center: float  # m, axial center position
    current: float = 50e3  # A, operating current

    def __repr__(self) -> str:
        return (f"PancakeLayer(r={self.r_center*100:.1f} cm, "
                f"z={self.z_center*100:.1f} cm)")


@dataclass
class Solenoid:
    """Full solenoid geometry with layers."""
    cable: StackedSlotCable
    r_inner: float  # m, inner radius
    height: float  # m, total height
    layers: List[PancakeLayer] = field(default_factory=list)
    r_outer: float = field(init=False)  # m, computed outer radius
    n_radial: int = field(init=False)  # number of radial layers
    n_axial: int = field(init=False)  # number of axial layers

    def __post_init__(self):
        # Initialize computed fields if not set
        if not hasattr(self, '_computed'):
            self.r_outer = self.r_inner
            self.n_radial = 0
            self.n_axial = 0

    @property
    def total_turns(self) -> int:
        """Total number of turns."""
        return len(self.layers)

    @property
    def NI(self) -> float:
        """Total ampere-turns."""
        return self.total_turns * self.layers[0].current if self.layers else 0

    def summary(self) -> None:
        """Print solenoid summary."""
        print("=" * 55)
        print("  SOLENOID SUMMARY")
        print("=" * 55)
        print(f"  Inner radius       : {self.r_inner*100:.1f} cm")
        print(f"  Outer radius       : {self.r_outer*100:.1f} cm")
        print(f"  Winding thickness  : {(self.r_outer-self.r_inner)*100:.1f} cm")
        print(f"  Height             : {self.height*100:.1f} cm")
        print(f"  Cable conduit      : {self.cable.conduit_side*1e3:.0f} mm")
        print(f"  Radial layers      : {self.n_radial}")
        print(f"  Axial layers       : {self.n_axial}")
        print(f"  Total turns        : {self.total_turns}")
        print(f"  Total NI           : {self.NI/1e6:.2f} MA·turns")
        print(f"  Cable              : {self.cable}")
        print("=" * 55)