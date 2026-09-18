"""Material properties for superconducting components."""

from dataclasses import dataclass

import numpy as np


# Hermes data-book lift factors: Ic(B, T) / Ic(77 K, self-field).
# The data are for the conservative minimum Ic and B parallel to c, which
# matches the perpendicular-to-tape-face orientation used in the lab.
HERMES_LIFT_FACTORS = {
    4.2: {0.0: 15.9, 0.5: 11.5, 1.0: 9.3, 2.0: 7.0, 3.0: 5.8, 4.0: 5.0, 5.0: 4.4, 6.0: 4.0, 7.0: 3.7, 8.0: 3.4, 9.0: 3.2, 10.0: 3.0, 12.0: 2.7, 14.0: 2.5, 16.0: 2.3, 18.0: 2.1, 20.0: 2.0},
    20.0: {0.0: 11.7, 0.01: 11.7, 0.03: 11.6, 0.1: 11.3, 0.3: 9.8, 0.5: 7.3, 1.0: 6.1, 2.0: 4.3, 3.0: 3.5, 4.0: 3.0, 5.0: 2.6, 6.0: 2.3, 7.0: 2.1, 8.0: 1.9, 9.0: 1.8, 10.0: 1.7, 12.0: 1.5, 14.0: 1.4, 16.0: 1.3, 18.0: 1.2, 20.0: 1.1},
    30.0: {0.0: 9.1, 0.1: 8.6, 0.3: 7.0, 0.5: 5.8, 1.0: 4.2, 2.0: 2.9, 3.0: 2.4, 4.0: 2.1, 5.0: 1.8, 6.0: 1.7, 7.0: 1.5, 8.0: 1.3},
    40.0: {0.0: 6.8, 0.1: 6.2, 0.3: 4.7, 0.5: 3.9, 1.0: 2.8, 2.0: 2.0, 3.0: 1.6, 4.0: 1.4, 5.0: 1.2, 6.0: 1.1, 7.0: 1.0, 8.0: 0.89},
    50.0: {0.0: 5.0, 0.1: 4.3, 0.3: 3.0, 0.5: 2.5, 1.0: 1.8, 2.0: 1.3, 3.0: 1.0, 4.0: 0.88, 5.0: 0.76, 6.0: 0.68, 7.0: 0.60, 8.0: 0.53},
    60.0: {0.0: 3.3, 0.1: 2.6, 0.3: 1.8, 0.5: 1.4, 1.0: 1.0, 2.0: 0.69, 3.0: 0.54, 4.0: 0.46, 5.0: 0.38, 6.0: 0.33, 7.0: 0.28, 8.0: 0.24},
    70.0: {0.0: 1.9, 0.1: 1.2, 0.3: 0.76, 0.5: 0.58, 1.0: 0.40, 2.0: 0.26, 3.0: 0.19, 4.0: 0.15},
    77.3: {0.0: 1.0, 0.01: 0.91, 0.03: 0.74, 0.1: 0.46, 0.3: 0.27, 0.5: 0.22, 1.0: 0.12},
}


@dataclass
class HTSTape:
    """High-temperature superconducting tape properties."""
    width: float  # m
    thickness: float  # m
    t_rebco: float  # m, REBCO layer thickness
    t_copper: float  # m, copper stabilizer thickness
    t_substrate: float  # m, substrate thickness
    hermes_ic77_sf: float = 120.0  # A, conservative 4 mm tape lower bound
    hermes_data_source: str = "Faraday Factory Hermes data book, July 2025"

    @property
    def A_sc(self) -> float:
        """Superconducting cross-sectional area."""
        return self.width * self.t_rebco

    @property
    def A_total(self) -> float:
        """Total tape cross-sectional area."""
        return self.width * self.thickness

    @property
    def width_scale(self) -> float:
        """Scale a 4 mm reference current to this tape width."""
        return self.width / 4e-3

    def hermes_lift_factor(self, temperature: float, field: float) -> float:
        """Interpolate the published conservative Hermes lift-factor data."""
        if temperature < 4.2 or temperature > 77.3:
            raise ValueError("Hermes lift factors are defined for 4.2 <= T <= 77.3 K")
        if field < 0.0:
            raise ValueError("field must be non-negative")

        temperatures = np.array(sorted(HERMES_LIFT_FACTORS), dtype=float)
        temperature = float(temperature)
        lower_t = temperatures[temperatures <= temperature].max(initial=temperatures[0])
        upper_t = temperatures[temperatures >= temperature].min(initial=temperatures[-1])

        def field_factor(table: dict[float, float]) -> float:
            fields = np.array(sorted(table), dtype=float)
            values = np.array([table[value] for value in fields], dtype=float)
            if field > fields[-1]:
                raise ValueError(
                    f"Hermes lift-factor data at {temperature:.1f} K only extend "
                    f"to {fields[-1]:g} T"
                )
            return float(np.interp(field, fields, values))

        lower_value = field_factor(HERMES_LIFT_FACTORS[float(lower_t)])
        upper_value = field_factor(HERMES_LIFT_FACTORS[float(upper_t)])
        if lower_t == upper_t:
            return lower_value
        return float(np.interp(temperature, [lower_t, upper_t], [lower_value, upper_value]))

    def critical_current(self, temperature: float, field: float) -> float:
        """Return conservative Hermes Ic for this tape width at T and B."""
        return self.hermes_ic77_sf * self.width_scale * self.hermes_lift_factor(temperature, field)

    def __repr__(self) -> str:
        return (f"HTSTape(w={self.width*1e3:.1f} mm, "
                f"t={self.thickness*1e3:.2f} mm)")