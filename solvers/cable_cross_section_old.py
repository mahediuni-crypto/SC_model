"""Two-dimensional magnetic field model of a radial stacked-slot cable."""

from dataclasses import dataclass

import numpy as np

from data.conductors import StackedSlotCable
from utils.constants import MU0


@dataclass(frozen=True)
class StackField:
    """Local field at one HTS stack in the cable cross-section."""

    stack_index: int
    angle_deg: float
    center_field_T: float
    inner_edge_field_T: float
    inner_edge_radial_T: float
    inner_edge_tangential_T: float
    inner_edge_angle_deg: float


class CableCrossSectionFieldSolver:
    """Compute the self-field of the rectangular HTS petals in 2D."""

    def __init__(self, cable: StackedSlotCable, filament_grid_size: int = 9):
        if filament_grid_size < 2:
            raise ValueError("filament_grid_size must be at least 2")
        self.cable = cable
        self.filament_grid_size = int(filament_grid_size)

    def _filaments(self, current: float) -> list[tuple[float, float, float]]:
        """Return (x, y, current) filaments for all oriented rectangular stacks."""
        n = self.filament_grid_size
        stack_current = current / self.cable.N_slots
        half_tangential = 0.5 * self.cable.stack.width
        half_radial = 0.5 * self.cable.stack.height
        d_area = self.cable.stack.width * self.cable.stack.height / (n * n)
        filament_current = stack_current * d_area / (self.cable.stack.width * self.cable.stack.height)
        filaments = []

        for cx, cy, theta in self.cable.stack_geometry:
            radial = np.array([np.cos(theta), np.sin(theta)])
            tangential = np.array([-np.sin(theta), np.cos(theta)])
            for i in range(n):
                tangential_offset = -half_tangential + (i + 0.5) * self.cable.stack.width / n
                for j in range(n):
                    radial_offset = -half_radial + (j + 0.5) * self.cable.stack.height / n
                    point = np.array([cx, cy]) + tangential_offset * tangential + radial_offset * radial
                    filaments.append((point[0], point[1], filament_current))
        return filaments

    @staticmethod
    def _field_from_filaments(
        x: float,
        y: float,
        filaments: list[tuple[float, float, float]],
    ) -> tuple[float, float]:
        """Return Bx, By from z-directed current filaments."""
        bx = 0.0
        by = 0.0
        for xf, yf, current in filaments:
            dx = x - xf
            dy = y - yf
            rho2 = dx * dx + dy * dy
            if rho2 < 1e-20:
                continue
            factor = MU0 * current / (2.0 * np.pi * rho2)
            bx += -factor * dy
            by += factor * dx
        return bx, by

    def compute(self, current: float) -> list[StackField]:
        """Compute center and inner-edge fields for every radial stack."""
        filaments = self._filaments(current)
        results = []
        half_radial = 0.5 * self.cable.stack.height

        for index, (cx, cy, theta) in enumerate(self.cable.stack_geometry):
            radial = np.array([np.cos(theta), np.sin(theta)])
            tangential = np.array([-np.sin(theta), np.cos(theta)])
            edge = np.array([cx, cy]) - half_radial * radial

            center_bx, center_by = self._field_from_filaments(cx, cy, filaments)
            edge_bx, edge_by = self._field_from_filaments(edge[0], edge[1], filaments)
            edge_vector = np.array([edge_bx, edge_by])
            radial_field = float(np.dot(edge_vector, radial))
            tangential_field = float(np.dot(edge_vector, tangential))
            edge_magnitude = float(np.hypot(edge_bx, edge_by))
            angle = float(np.degrees(np.arctan2(abs(radial_field), abs(tangential_field))))

            results.append(StackField(
                stack_index=index,
                angle_deg=float(np.degrees(theta)),
                center_field_T=float(np.hypot(center_bx, center_by)),
                inner_edge_field_T=edge_magnitude,
                inner_edge_radial_T=radial_field,
                inner_edge_tangential_T=tangential_field,
                inner_edge_angle_deg=angle,
            ))
        return results
