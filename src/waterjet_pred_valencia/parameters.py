#!/usr/bin/env python3

"""
Provides various parameters needed for the fire stream model, including:

- Physical constants
- model constants such as air entrainment and drop distribution
- empirical formulas (e.g., Weber number)
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SimParams:
    """
    Per-simulation parameters (given or computed once).
    """

    num_drop_classes: int = 0

    # Flag indicating if simulation is at s > s_brk
    _is_post_breakup: bool = False

    def __init__(
        self, injection_speed: float, injection_angle_deg: float, nozzle_diameter: float
    ) -> None:
        """
        Constructor.

        Args:
            injection_speed: Water's speed as it exits the nozzle [m/s]
            injection_angle_deg: Elevation angle of the nozzle [deg]
            nozzle_diameter: Nozzle diameter [m]
        """
        self.injection_speed = injection_speed
        self.injection_angle_deg = injection_angle_deg
        self.nozzle_diameter = nozzle_diameter
        self.num_drop_classes: int = len(d_drop)
        self.s_brk = get_breakup_distance(self.nozzle_diameter)
        self.weber = get_weber_number(self.injection_speed, self.nozzle_diameter)
        return


# ---------------------------------------------------------------------------
# PHYSICAL CONSTANTS
# ---------------------------------------------------------------------------

g = 9.81  # Gravitational acceleration [m/s²]
rho_a = 1.225  # Air density at 15°C, sea level [kg/m³]
rho_w = 999  # Water density at 15°C [kg/m³]
mu_w = 8.9e-4  # Dynamic viscosity of water [Pa·s]
sigma_w = 0.0728  # Surface tension of water [N/m]

# ---------------------------------------------------------------------------
# MODEL CONSTANTS
# ---------------------------------------------------------------------------

# Droplet diameters [m]
d_drop = np.array([0.0005, 0.002, 0.003, 0.004, 0.005])
num_drop_classes = len(d_drop)

# Volume/mass fractions for each drop class (sum to 1)
p_d = np.array([0.05, 0.25, 0.30, 0.25, 0.15])

# Air entrainment coefficient [-] (sensitive!)
alpha = 0.13

# Friction coefficient between the water jet's surface and the air [-]
F = 0.3

# ---------------------------------------------------------------------------
# WEBER, REYDNOLDS, DRAG COEFFICIENT
# ---------------------------------------------------------------------------


def get_weber_number(injection_speed: float, nozzle_diameter: float) -> float:
    """
    Calculates the Weber number We_0 according to the formula after Eq. 23.

    Note: Lowercase d_0 means nozzle diameter here, but droplet diameter elsewhere!

    Args:
        injection_speed: U_0 [m/s]
        nozzle_diameter: d_0 [m]

    Returns:
        Weber number [-]
    """
    assert injection_speed > 0.0, "Injection speed must be positive"
    assert nozzle_diameter > 0.0, "Nozzle diameter must be positive"
    return (rho_w * injection_speed**2 * nozzle_diameter) / sigma_w


def get_reynolds_number(Us: float, d: float) -> float:
    """
    Computes a drop's Reynold's number, according to the formula given after Eq. 26.

    Args:
        Us: drop speed [m/s]
        d: drop diameter [m]

    Returns:
        Reynolds number [-]
    """
    assert Us > 0.0, f"Spray velocity {Us=} m/s must be positive"
    assert d > 0.0, f"Drop diameter {d=} m must be positive"
    return Us * d * rho_w / mu_w


def get_drag_coefficient(Re_d: float) -> float:
    """
    Computes drag coefficient C_D according to Eq. 26.

    Args:
        Re_d: Reynolds number [-]

    Returns:
        Drag coefficient C_D [-]
    """
    assert Re_d > 0.0, "Reynolds number must be positive"
    if Re_d < 1000:
        return (24.0 / Re_d) * (1 + ((Re_d ** (2.0 / 3.0)) / 6.0))
    else:
        return 0.424


def get_breakup_distance(nozzle_diameter: float) -> float:
    """
    Computes s_brk according to eq. 27 (empirically found correlation)

    Args:
        nozzle_diameter: [m]

    Returns:
        s_brk: Break-up location along the streamwise axis s [m]
    """
    # Note: rho_g in the paper (g for gas?) is rho_a (air) in our case
    s_brk: float = nozzle_diameter * 11 * np.sqrt(rho_w / rho_a)
    assert s_brk > 0.0, f"{s_brk=} must be positive"
    return s_brk


#
