"""Defines JetState dataclass."""

import logging
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional

import numpy as np
from numpy.typing import NDArray

from .parameters import DTYPE, SimParams, num_drop_classes, rho_a, rho_w

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class JetState:
    """Dataclass representing the state vector (or its derivative).

    Improves readability when getting/setting values. However, the performance cost is
    not negligible. Consider going back to flat vector later on.

    Attributes:
        Uc: Core phase speed [m/s].
        Dc: Core phase diameter [m].
        Ua: Air phase speed [m/s].
        Da: Air phase diameter [m].
        theta_a: Air phase angle (to vertical)  [rad].
        Uf: Stream speed [m/s].
        Df: Stream diameter [m].
        theta_f: Stream angle (to vertical) [rad].
        rho_f: Stream density [kg/m³].
        x: Horizontal position [m].
        y: Vertical position [m].
        ND: (K,) Drop generation rate per droplet class [drops/s].
        Us: (K,) Spray phase speed per droplet class [m/s].
        theta_s: (K,) Spray phase angle per droplet class [rad].
    """

    # Order of variables in state vector.
    BASE_VARS: ClassVar[List[str]] = [
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
    SPRAY_VARS: ClassVar[List[str]] = ["ND", "Us", "theta_s"]

    Uc: float = 0.0
    Dc: float = 0.0
    Ua: float = 0.0
    Da: float = 0.0
    theta_a: float = 0.0
    Uf: float = 0.0
    Df: float = 0.0
    theta_f: float = 0.0
    rho_f: float = 0.0
    x: float = 0.0
    y: float = 0.0
    ND: NDArray[DTYPE] = field(
        default_factory=lambda: np.zeros(num_drop_classes, dtype=DTYPE)
    )
    Us: NDArray[DTYPE] = field(
        default_factory=lambda: np.zeros(num_drop_classes, dtype=DTYPE)
    )
    theta_s: NDArray[DTYPE] = field(
        default_factory=lambda: np.zeros(num_drop_classes, dtype=DTYPE)
    )

    def to_array(self) -> NDArray[DTYPE]:
        """Convert to flat array for solve_ivp().

        Returns:
            (N,) array representing the state vector.
        """
        core: NDArray[DTYPE] = np.array(
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
        spray: NDArray[DTYPE] = np.concatenate(
            [self.ND, self.Us, self.theta_s], dtype=DTYPE
        )
        return np.concatenate([core, spray], dtype=DTYPE)

    @staticmethod
    def from_array(y: NDArray[DTYPE]) -> "JetState":
        """Reconstruct state from flat numpy array.

        Args:
            y: (N,) flat state vector.

        Returns:
            JetState object.
        """

        y = np.asarray(y, dtype=DTYPE)
        assert np.all(np.isfinite(y))
        base: int = 11
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
        """Compute the flat array index for a given state variable.

        Args:
            name: Variable name (e.g. "Uc", "theta_a", "ND").
            s_class: Spray class index (only for spray variables).

        Returns:
            Index of the variable in the flattened state vector returned by to_array().

        Raises:
            KeyError: if variable name is unknown.
        """
        # Scalar variables
        if name in JetState.BASE_VARS:
            return JetState.BASE_VARS.index(name)
        # Spray variables
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
        injection_height: float = 2.0,
    ) -> "JetState":
        """Return the initial state vector of the fire stream (see appendix of paper).

        Args:
            injection_speed: U_0, water's speed as it exits the nozzle [m/s].
            injection_angle_deg: theta_0 (nozzle angle relative to horizon) [deg].
            nozzle_diameter: D_0 [m].
            injection_height: y_0, height of nozzle above ground [m] (default: 0.0).

        Returns:
            JetState representing the initial state of the simulation
        """

        init = JetState()

        # NOTE: Author confirmed that only the injection angle is measured relative to
        # horizon. Phase angles are measured relative to vertical axis.
        injection_angle_rad = float(np.deg2rad(injection_angle_deg))
        # Change injection angle reference to the vertical axis.
        # Andres outcommented this line in his 25.09.2025 version.
        # Reincluded it due to the fact stated above. Plots confirm that this only
        # changes where the angles start, crash behaviour is unchanged.
        injection_angle_rad: float = np.pi / 2.0 - injection_angle_rad

        # Position
        init.x = 0.0
        init.y = injection_height

        # Eq 31-33 core phase
        init.Uc = injection_speed
        init.Dc = nozzle_diameter

        # Eq 34-36 spray phase (one set per droplet class)
        for i in range(num_drop_classes):
            # Note: not 0.0 to prevent singularities, e.g. in dyds[theta_s] calculation.
            # Since ND will grow rapidly, a small starting value doesn't hurt.
            init.ND[i] = 1.0

            init.Us[i] = injection_speed
            init.theta_s[i] = injection_angle_rad

        # Eq 38-40 air phase
        # Note: -1e-6 from notebook, not paper.
        init.Ua = injection_speed - 1e-6
        # Note: not 0.0 as in paper to prevent singularity in dyds.Ua calculation
        init.Da = nozzle_diameter
        init.theta_a = injection_angle_rad

        # Eq 41-44 fire stream phase
        init.Uf = injection_speed
        init.Df = nozzle_diameter
        init.theta_f = injection_angle_rad
        init.rho_f = rho_w

        logger.debug(f"Created initial state vector:\n{init}")
        return init

    # TODO: Check if limits need refinement
    def assert_physically_plausible(self, params: SimParams) -> None:
        """Ensure all state variables are physically plausible.

        Some checks are relatively trivial, e.g. "diameters can't be negative".
        Others are derived from the plots in Dr. Valencia's Mathematica notebook.

        Args:
            params: simulation parameters (e.g., injection angle).

        Raises:
            AssertionError: if a physical plausibility check failed.
        """

        # Speeds.
        U_max: float = 2 * params.injection_speed
        assert 0 <= self.Uc <= U_max, (
            f"Core speed {self.Uc=} must be in range (0, {U_max})"
        )
        assert 0 <= self.Ua <= U_max, (
            f"Air phase speed {self.Ua=} must be in range (0, {U_max})"
        )
        assert 0 <= self.Uf <= U_max, (
            f"Stream speed {self.Uf=} must be in range (0, {U_max})"
        )

        # Diameters.
        D0: float = params.nozzle_diameter
        assert 0 <= self.Dc <= (Dc_max := 2 * D0), (
            f"Core phase diameter {self.Dc=} must be in range (0, {Dc_max})"
        )
        assert 0 <= self.Da <= (Da_max := 10.0), (
            f"Air phase diameter {self.Da=} must be in range (0, {Da_max})"
        )
        assert 0 <= self.Df <= (Df_max := 10.0), (
            f"Stream phase diameter {self.Df=} must be in range (0, {Df_max})"
        )

        # Angles
        # NOTE: author clarified that all phase angles are relative to vertical axis.
        # Only the injection angle theta_0 is relative to the horizontal.
        th_low: float = 0.0  # upwards [deg]
        th_high: float = 180.0  # downwards [deg]
        assert th_low <= (th_a_deg := np.rad2deg(self.theta_a)) <= th_high, (
            f"theta_a={th_a_deg:.4f}° must be in range {th_low, th_high}°"
        )
        assert th_low <= (th_f_deg := np.rad2deg(self.theta_f)) <= th_high, (
            f"theta_f={th_f_deg:.4f}° must be in range {th_low, th_high}°"
        )
        for i in range(num_drop_classes):
            assert th_low <= (th_f_deg := np.rad2deg(self.theta_s[i])) <= th_high, (
                f"theta_s[{i}]={th_f_deg:.4f}° must be in range {th_low, th_high}°"
            )

        # Spray generation.
        ND_max = [1e8, 1e7, 1e6, 1e6, 1e6]  # max drop generation rates
        assert len(ND_max) == num_drop_classes
        for i in range(num_drop_classes):
            assert 0.0 <= self.ND[i] <= ND_max[i], (
                f"ND{i}={self.ND[i]:.2g} must be in range (0, {ND_max[i]:.2g})"
            )

        # Density.
        assert rho_a <= self.rho_f <= rho_w, (
            f"Density {self.rho_f=:.2f}kg/m³ must be in range ({rho_a=}, {rho_w=})"
        )

        return
