"""Provides various parameters needed for the fire stream model, including:

- Physical constants
- model constants such as air entrainment and drop distribution
- empirical formulas (e.g., Weber number)
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Default numpy datatype.
DTYPE = np.float32


@dataclass
class SimParams:
    """Dataclass to hold per-simulation parameters (given or computed once).

    Attributes:
        injection_speed (U_0): Speed of the water as it exits the nozzle [m/s].
        injection_angle_deg (theta_0): Elevation angle of the nozzle [deg].
        nozzle_diameter (D_0): Nozzle diameter [m].
    """

    # Flag indicating if simulation is past s_brk
    _is_post_breakup: bool = False

    def __init__(
        self, injection_speed: float, injection_angle_deg: float, nozzle_diameter: float
    ) -> None:
        """Initialize simulation parameters.

        Args:
            See class attributes.
        """
        self.injection_speed = injection_speed
        self.injection_angle_deg = injection_angle_deg
        self.nozzle_diameter = nozzle_diameter
        self.s_brk = get_breakup_distance(self.nozzle_diameter)
        self.weber = get_weber_number(self.injection_speed, self.nozzle_diameter)
        return


# ---------------------------------------------------------------------------
# PHYSICAL CONSTANTS
# ---------------------------------------------------------------------------

g: float = 9.81  # Gravitational acceleration [m/s²]
rho_a: float = 1.225  # Air density at 15°C, sea level [kg/m³]
rho_w: float = 999.0  # Water density at 15°C [kg/m³]
mu_w: float = 8.9e-4  # Dynamic viscosity of water [Pa·s]
sigma_w: float = 0.0728  # Surface tension of water [N/m]

# ---------------------------------------------------------------------------
# MODEL CONSTANTS
# ---------------------------------------------------------------------------

# Droplet diameters [m]
d_drop: NDArray = np.array([0.0005, 0.002, 0.003, 0.004, 0.005], dtype=DTYPE)
num_drop_classes: int = len(d_drop)

# Volume/mass fractions for each drop class (sum to 1)
p_d: NDArray = np.array([0.05, 0.25, 0.30, 0.25, 0.15], dtype=DTYPE)

# Air entrainment coefficient [-] (sensitive!)
alpha: float = 0.13

# Friction coefficient between the water jet's surface and the air [-]
F: float = 0.3

# ---------------------------------------------------------------------------
# WEBER, REYDNOLDS, DRAG COEFFICIENT
# ---------------------------------------------------------------------------


def get_weber_number(injection_speed: float, nozzle_diameter: float) -> float:
    """Calculate the Weber number We_0 according to the formula after Eq. 23.


    Args:
        injection_speed (U_0): Speed of the water exiting the nozzle [m/s].
        nozzle_diameter (d_0): Diameter of the nozzle [m]. Note: Lowercase d_0 means
            nozzle diameter here, but droplet diameter elsewhere!

    Returns:
        Weber number [-].
    """
    assert injection_speed > 0.0, "Injection speed U_0 must be positive"
    assert nozzle_diameter > 0.0, "Nozzle diameter must be positive"
    return (rho_w * injection_speed**2 * nozzle_diameter) / sigma_w


def get_reynolds_number(Us: float, d: float) -> float:
    """Compute a drop's Reynold's number, according to the formula given after Eq. 26.

    Args:
        Us: Drop speed [m/s].
        d: Drop diameter [m].

    Returns:
        Reynolds number [-].
    """
    assert Us > 0.0, f"Spray velocity {Us=} m/s must be positive"
    assert d > 0.0, f"Drop diameter {d=} m must be positive"
    return Us * d * rho_w / mu_w


def get_drag_coefficient(Re_d: float) -> float:
    """Compute the drag coefficient C_D according to Eq. 26.

    Args:
        Re_d: Reynolds number [-].

    Returns:
        Drag coefficient C_D [-].
    """
    assert Re_d > 0.0, "Reynolds number must be positive"
    if Re_d < 1_000:
        return (24.0 / Re_d) * (1 + ((Re_d ** (2.0 / 3.0)) / 6.0))
    else:
        return 0.424


def get_breakup_distance(nozzle_diameter: float) -> float:
    """Compute 's_brk' according to Eq. 27 (empirically found correlation).

    Args:
        nozzle_diameter (D_0): Diameter of the nozzle [m].

    Returns:
        s_brk: Break-up location along the streamwise axis s [m].
    """
    # Note: rho_g in the paper (g for gas?) means rho_a (air) here.
    s_brk: float = nozzle_diameter * 11 * np.sqrt(rho_w / rho_a)
    assert s_brk > 0.0, f"{s_brk=} must be > 0"
    return s_brk
