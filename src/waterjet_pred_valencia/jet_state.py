#!/usr/bin/env python3

from .parameters import *

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import DTypeLike, NDArray
from typing import ClassVar, Optional

DTYPE: DTypeLike = np.float32


@dataclass(slots=True)
class JetState:
    """
    Dataclass representing the state vector (or its derivative).
    Improves readability when getting/setting values, in exchange for a very small
    performance hit.
    """

    BASE_VARS: ClassVar[list[str]] = [
        "Uc",
        "Dc",
        "Ua",
        "Da",
        "theta_a",
        "Uf",
        "Df",
        "theta_f",
        "rho_f",
        "x",
        "y",
    ]

    # can't use DTYPE on left-hand side, unfortunately
    Uc: np.float32 = DTYPE(0.0)  # Core phase speed [m/s]
    Dc: np.float32 = DTYPE(0.0)  # Core phase diameter [m]
    Ua: np.float32 = DTYPE(0.0)  # Air phase speed [m/s]
    Da: np.float32 = DTYPE(0.0)  # Air phase diameter [m]
    theta_a: np.float32 = DTYPE(0.0)  # Air phase angle (to vertical) [rad]
    Uf: np.float32 = DTYPE(0.0)  # Stream speed [m]
    Df: np.float32 = DTYPE(0.0)  # Stream diameter [m]
    theta_f: np.float32 = DTYPE(0.0)  # Stream angle (to vertical) [rad]
    rho_f: np.float32 = DTYPE(0.0)  # Stream density [kg/m3]
    x: np.float32 = DTYPE(0.0)  # Horizontal position [m]
    y: np.float32 = DTYPE(0.0)  # Vertical position [m]

    SPRAY_VARS: ClassVar[list[str]] = ["ND", "Us", "theta_s"]

    # Spray generation [drops/s]
    ND: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(num_drop_classes, dtype=DTYPE)
    )
    # Spray phase speed [m/s]
    Us: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(num_drop_classes, dtype=DTYPE)
    )
    # Spray phase angle (to vertical) [rad]
    theta_s: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(num_drop_classes, dtype=DTYPE)
    )

    def to_array(self) -> NDArray[np.float32]:
        """Convert to flat array for solve_ivp()."""
        core = np.array(
            [
                self.Uc,
                self.Dc,
                self.Ua,
                self.Da,
                self.theta_a,
                self.Uf,
                self.Df,
                self.theta_f,
                self.rho_f,
                self.x,
                self.y,
            ],
            dtype=DTYPE,
        )
        spray = np.concatenate([self.ND, self.Us, self.theta_s], dtype=DTYPE)
        return np.concatenate([core, spray], dtype=DTYPE)

    @staticmethod
    def from_array(y: NDArray[np.float32]) -> "JetState":
        """Reconstruct state from flat ndarray."""
        assert np.all(np.isfinite(y))
        base = 11
        return JetState(
            Uc=y[0],
            Dc=y[1],
            Ua=y[2],
            Da=y[3],
            theta_a=y[4],
            Uf=y[5],
            Df=y[6],
            theta_f=y[7],
            rho_f=y[8],
            x=y[9],
            y=y[10],
            ND=y[base : base + num_drop_classes],
            Us=y[base + num_drop_classes : base + 2 * num_drop_classes],
            theta_s=y[base + 2 * num_drop_classes : base + 3 * num_drop_classes],
        )

    @staticmethod
    def get_idx(name: str, s_class: Optional[int] = None) -> int:
        """
        Compute the flat array index for a given state variable.

        Args:
            name: variable name (e.g. "Uc", "theta_a", "ND")
            s_class: spray class index (only for spray variables)

        Returns:
            int: index of the variable in the flattened state vector returned by
                to_array().

        Raises:
            KeyError if variable name is unknown
        """
        # scalar variables
        if name in JetState.BASE_VARS:
            return JetState.BASE_VARS.index(name)

        # spray variables
        if name in JetState.SPRAY_VARS:
            assert s_class is not None
            base_offset = len(JetState.BASE_VARS)
            var_offset = JetState.SPRAY_VARS.index(name) * num_drop_classes
            return base_offset + var_offset + s_class

        raise KeyError(f"Unknown variable name '{name}'")

    @staticmethod
    def get_initial(
        injection_speed: float,
        injection_angle_deg: float,
        nozzle_diameter: float,
    ) -> "JetState":
        """
        Returns the initial state vector of the fire stream (see appendix of paper).

        Args:
            injection_speed: U_0, water's speed as it exits the nozzle [m/s]
            injection_angle_deg: theta_0 (nozzle angle relative to horizon) [deg]
            nozzle_diameter: D_0 [m]

        Returns:
            JetState representing the initial state of the simulation
        """

        init = JetState()

        # NOTE: Author confirmed that only the injection angle is measured relative to
        # horizon. Phase angles are measured relative to vertical axis.
        injection_angle_rad: float = np.deg2rad(injection_angle_deg)

        # TODO: Clarify Andres intention for outcommenting this line in 25.09.25 version
        # injection_angle_rad: float = np.pi / 2.0 - injection_angle_rad

        # position
        init.x = np.float32(0.0)
        init.y = np.float32(0.0)  # TODO: check if y > 0 works too

        # Eq 31-33 core phase
        init.Uc = np.float32(injection_speed)
        init.Dc = np.float32(nozzle_diameter)

        # Eq 34-36 spray phase (one set per droplet class)
        for i in range(num_drop_classes):
            # NOTE: Not 0.0 to prevent singularities, e.g. in dyds[theta_s] calculation
            # since ND will reach > 1e4, this starting value seems ok
            init.ND[i] = np.float32(1.0)

            init.Us[i] = np.float32(injection_speed)
            init.theta_s[i] = np.float32(injection_angle_rad)

        # Eq 38-40 air phase
        # NOTE: -1e-6 from notebook, not paper
        init.Ua = np.float32(injection_speed - 1e-6)
        # NOTE: not 0.0 as in paper to prevent singularity in dyds.Ua calculation
        init.Da = np.float32(nozzle_diameter)
        init.theta_a = np.float32(injection_angle_rad)

        # Eq 41-44 fire stream phase
        init.Uf = np.float32(injection_speed)
        init.Df = np.float32(nozzle_diameter)
        init.theta_f = np.float32(injection_angle_rad)
        init.rho_f = np.float32(rho_w)

        return init

    # TODO: Check if limits need refinement
    def assert_physically_plausible(self, params: SimParams) -> None:
        """
        Asserts all state variables are (physically) plausible.

        Some checks are relatively trivial, e.g. "diameters can't be negative".
        Others are derived from the plots in Dr. Valencia's Mathematica notebook.

        Args:
            params: simulation parameters (e.g., injection angle)

        Raises:
            AssertionError
        """

        # Speeds
        U_max: float = 2 * params.injection_speed
        assert (
            0 <= self.Uc <= U_max
        ), f"core speed {self.Uc=} must be in range (0, {U_max})"
        assert (
            0 <= self.Ua <= U_max
        ), f"air phase speed {self.Ua=} must be in range (0, {U_max})"
        assert (
            0 <= self.Uf <= U_max
        ), f"stream speed {self.Uf=} must be in range (0, {U_max})"

        # Diameters
        D0: float = params.nozzle_diameter
        assert (
            0 <= self.Dc <= (Dc_max := 2 * D0)
        ), f"{self.Dc=} must be in range (0, {Dc_max})"
        assert (
            0 <= self.Da <= (Da_max := 10.0)
        ), f"{self.Da=} must be in range (0, {Da_max})"
        assert (
            0 <= self.Df <= (Df_max := 10.0)
        ), f"{self.Df=} must be in range (0, {Df_max})"

        # Angles
        # NOTE: author clarified that all phase angles are relative to vertical axis.
        # Only the injection angle theta_0 is relative to the horizontal.
        th_low: float = 0.0  # upwards [deg]
        th_high: float = 180.0  # downwards [deg]
        assert (
            th_low <= (th_a_deg := np.rad2deg(self.theta_a)) <= th_high
        ), f"theta_a={th_a_deg}° must be in range {th_low, th_high}°"
        assert (
            th_low <= (th_f_deg := np.rad2deg(self.theta_f)) <= th_high
        ), f"theta_f={th_f_deg}° must be in range {th_low, th_high}°"
        for i in range(num_drop_classes):
            assert (
                th_low <= (th_f_deg := np.rad2deg(self.theta_s[i])) <= th_high
            ), f"theta_s[{i}]={th_f_deg}° must be in range {th_low, th_high}°"

        # Spray generation
        ND_max = [1e8, 1e7, 1e6, 1e6, 1e6]
        assert len(ND_max) == num_drop_classes
        for i in range(num_drop_classes):
            assert (
                0.0 <= self.ND[i] <= ND_max[i]
            ), f"ND{i}={self.ND[i]:.2g} must be in range (0, {ND_max[i]:.2g})"

        # Density
        assert (
            rho_a <= self.rho_f <= rho_w
        ), f"Stream density {self.rho_f:.2f} kg/m³ must be between air and water ({rho_a}, {rho_w})"

        return
