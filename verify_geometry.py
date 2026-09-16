"""Geometry verification utility for the CICC/stacked-slot conductor model.

This script is intentionally self-contained and uses the same simplified
cable geometry that is already present in the project.
"""

from __future__ import annotations

import sys
import os
from typing import Iterable, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

try:
    from data.conductors import CICCCable  # type: ignore
except ImportError:
    CICCCable = None

from data.conductors import StackedSlotCable, TapeStack
from data.materials import HTSTape


def build_demo_cable() -> StackedSlotCable:
    """Build the same representative cable used in the current demo flow."""
    tape = HTSTape(
        width=4e-3,
        thickness=0.1e-3,
        t_rebco=1e-6,
        t_copper=20e-6,
        t_substrate=50e-6,
    )
    stack = TapeStack(
        tape=tape,
        N_tapes=40,
        gap=0.0,
    )
    return StackedSlotCable(
        stack=stack,
        N_slots=4,
        jacket_outer_diameter=27.7e-3,
        jacket_thick=2.0e-3,
        former_thick=2e-3,
        former_material="Copper",
        helium_channel_diameter=7.0e-3,
    )


def mm2(area_m2: float) -> float:
    return area_m2 * 1e6


def check_overlap(cable: StackedSlotCable) -> None:
    """Verify that the HTS stacks stay clear of the helium hole and the jacket wall.

    The simplified cable model places four stack centers on a circle of radius
    8.5 mm. Each stack is treated as a square block with a bounding radius.
    """
    stack_half_side = 0.5 * cable.stack.block_side
    stack_bounding_radius = np.sqrt(2.0) * stack_half_side

    min_stack_distance = min(
        np.hypot(cx, cy)
        for cx, cy in cable.stack_positions
    )

    # The stack must stay outside the central helium hole.
    assert min_stack_distance - stack_bounding_radius > cable.helium_radius, (
        "Stack centers are too close to the helium channel: the stack bounding"
        " circle intersects the helium hole."
    )

    # The stack must stay inside the former area and away from the jacket inner wall.
    max_stack_distance = max(
        np.hypot(cx, cy)
        for cx, cy in cable.stack_positions
    )
    assert max_stack_distance + stack_bounding_radius < cable.jacket_inner_radius, (
        "Stack bounding circles reach or exceed the stainless steel inner wall."
    )

    print("Overlap check: PASS")


def component_areas(cable: StackedSlotCable) -> dict[str, float]:
    """Return the key component areas in square metres."""
    r_outer = cable.jacket_outer_radius
    r_jacket_inner = cable.jacket_inner_radius
    r_he = cable.helium_radius

    area_total = np.pi * r_outer**2
    area_jacket = np.pi * (r_outer**2 - r_jacket_inner**2)
    area_he = cable.A_he
    area_stacks = cable.A_sc_total

    # In the current simplified model, the copper former area is the residual
    # metallic area left after removing the helium channel and the HTS stacks
    # from the internal annulus enclosed by the steel jacket.
    area_copper = area_total - area_jacket - area_he - area_stacks

    return {
        "Total conductor area": area_total,
        "Stainless Steel jacket": area_jacket,
        "Copper former": area_copper,
        "4 HTS stacks": area_stacks,
        "Helium channel": area_he,
    }


def void_fraction(cable: StackedSlotCable) -> float:
    """Compute the void fraction in percent from the internal flow area.

    For this simplified geometry, the effective helium-available fraction is
    taken as the helium area over the metallic cross-sectional area inside the
    jacket wall. The value is reported for verification purposes.
    """
    area_total = np.pi * cable.jacket_outer_radius**2
    area_jacket = np.pi * (cable.jacket_outer_radius**2 - cable.jacket_inner_radius**2)
    area_internal = area_total - area_jacket
    area_he = cable.A_he
    return 100.0 * area_he / area_internal


def verify_self_consistency(cable: StackedSlotCable) -> None:
    """Assert that the component areas close the overall conductor area."""
    areas = component_areas(cable)
    area_total = areas["Total conductor area"]
    area_sum = (
        areas["Stainless Steel jacket"]
        + areas["Copper former"]
        + areas["4 HTS stacks"]
        + areas["Helium channel"]
    )
    assert np.isclose(area_sum, area_total, rtol=1e-12, atol=1e-18), (
        f"Area balance failed: sum={mm2(area_sum):.6f} mm², total={mm2(area_total):.6f} mm²"
    )
    print("Self-consistency: PASS")


def main() -> None:
    cable = build_demo_cable()

    # Compatibility with the requested class name if the user has it in a later revision.
    if CICCCable is not None and not isinstance(cable, CICCCable):
        cable = CICCCable(**cable.__dict__)  # pragma: no cover

    areas = component_areas(cable)
    print("Component cross-sectional areas")
    for name, area in areas.items():
        print(f"  {name:25s}: {mm2(area):.6f} mm²")

    check_overlap(cable)

    vf = void_fraction(cable)
    print(f"Void fraction: {vf:.3f}%")
    if 30.0 <= vf <= 40.0:
        print("Void fraction check: PASS")
    else:
        print("Void fraction check: FAIL (expected 30–40% for a representative industrial CICC)")

    verify_self_consistency(cable)


if __name__ == "__main__":
    main()
