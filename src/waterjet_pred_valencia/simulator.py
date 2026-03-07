"""Importable core logic for the fire stream trajectory model.

Uses SciPy's solve_ivp() to solve a ODE system that models the stream's evolution along
the streamwise axis "s".

Requires parameters:
- injection speed
- injection angle
- nozzle diameter
- simulation span, typically (0, <upper bound of expected arc length>)

Physical and model constants are imported from parameters.py.
Parameter "max_step" can be adjusted to trade accuracy for performance.
"""

import logging
from dataclasses import dataclass
from functools import partial
from sys import exit
from typing import Dict, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

from .jet_state import DTYPE, JetState
from .parameters import (
    F,
    SimParams,
    alpha,
    d_drop,
    g,
    get_drag_coefficient,
    get_reynolds_number,
    num_drop_classes,
    p_d,
    rho_a,
    rho_w,
)
from .tracer import Tracer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SimulationResult:
    """Structured simulation output for success and failure handling.

    Attributes:
        status: Execution outcome ("completed" or "failed").
        state_idx: Mapping of state variable names to positions in state vectors.
        sol: ODE solver output, available on successful runs.
        error: Captured exception, available on failed runs.
    """

    status: Literal["completed", "failed"]
    state_idx: Dict[str, int]
    sol: Optional[OdeResult] = None
    error: Optional[Exception] = None


def get_state_index_map() -> Dict[str, int]:
    """Build a state-name -> flat-index map for plotting and post-processing."""

    idx: Dict[str, int] = {}
    for name in JetState.BASE_VARS:
        idx[name] = JetState.get_idx(name)
    for name in JetState.SPRAY_VARS:
        for i in range(num_drop_classes):
            idx[f"{name}_{i}"] = JetState.get_idx(name, i)
    return idx


def simulate(
    injection_speed: float,
    injection_angle_deg: float,
    nozzle_diameter: float,
    s_span: Tuple[float, float] = (0, 100),
    max_step: float = 1e-3,
    method: str = "Radau",
    debug: bool = False,
    bypass: Optional[Dict[str, float]] = None,
    tracer: Optional[Tracer] = None,
) -> SimulationResult:
    """Simulate a fire stream trajectory.

    Args:
        injection_speed: U_0 Speed of the water as it exists the nozzle [m/s].
        injection_angle_deg: theta_0 Nozzle elevation angle (relative to horizon) [deg].
        nozzle_diameter: D_0 Diameter of the nozzle [m].
        s_span: (start, end) in streamwise domain `s` [m].
        max_step: Max integration step size [m].
        method: See documentation for `scipy.integrate.solve_ivp`.

    Optional args for debugging:
        debug: Enable debug mode (console printouts, auto-drop into PDB).
        bypass: Map of derivatives to override with provided values.
        tracer: Records variables (not only state vector) per simulation step.

    Returns:
        Structured simulation output including solver results and index map.

    Raises:
        AssertionError: if a physical plausibility check failed.
    """
    if debug:
        logger.setLevel(logging.DEBUG)

    # Validate input parameters.
    assert injection_speed > 0.0, f"{injection_speed=} must be > 0"
    assert 0 <= injection_angle_deg <= 90, (
        f"{injection_angle_deg=} must be in range (0, 90)°"
    )
    assert nozzle_diameter > 0.0, f"{nozzle_diameter} must be > 0"
    assert isinstance(s_span, tuple), f"{s_span=} must be a tuple"
    assert len(s_span) == 2, f"{s_span=} must be in the form (start, end)"
    assert all(isinstance(x, (float, int)) and x >= 0 for x in s_span), (
        f"Values of {s_span=} must all be >= 0"
    )
    assert s_span[1] > s_span[0], f"{s_span[1]=} end must be > start {s_span[0]=}"
    assert max_step > 0.0, f"{max_step=} must be > 0"

    # Store given and calculate some derived parameters (e.g., s_brk=breakup distance).
    params = SimParams(
        injection_speed=injection_speed,
        injection_angle_deg=injection_angle_deg,
        nozzle_diameter=nozzle_diameter,
    )
    if bypass is not None:
        logger.warning(f"Active bypasses: {bypass}")

    # Construct initial state vector
    initial_state_vec = JetState.get_initial(
        injection_speed, injection_angle_deg, nozzle_diameter
    )

    # Event to stop the simulation when water hits the ground (y < 0).
    def hit_ground_event(_, y) -> float:
        idx = JetState.get_idx("y")
        return y[idx]

    ev_ground: partial[float] = partial(hit_ground_event)
    ev_ground.terminal = True  # pyright: ignore
    ev_ground.direction = -1  # pyright: ignore

    # Event to stop the simulation when the stream's density is very close to air.
    def mass_depleted_event(_, y, tol=1e-3) -> float:
        idx = JetState.get_idx("rho_f")
        return y[idx] - (rho_a + tol)

    ev_mass: partial[float] = partial(mass_depleted_event)
    ev_mass.terminal = True  # pyright: ignore
    ev_mass.direction = -1  # pyright: ignore

    # This is where da number magic happens!
    state_idx: Dict[str, int] = get_state_index_map()
    sol: OdeResult
    try:
        sol = solve_ivp(
            partial(ode_right_hand_side, params=params, bypass=bypass, tracer=tracer),
            s_span,
            y0=initial_state_vec.to_array(),
            method=method,
            max_step=max_step,
            dense_output=True,
            events=[ev_ground, ev_mass],
        )
    except Exception as e:
        logger.error(f"{e}")
        if debug:
            # Automatically drop into debugger.
            import pdb

            pdb.post_mortem()
            exit(1)
        else:
            raise e

    return SimulationResult(status="completed", state_idx=state_idx, sol=sol)


def ode_right_hand_side(
    s: float,
    y: NDArray[DTYPE],
    params: SimParams,
    bypass: Optional[Dict[str, float]] = None,
    tracer: Optional[Tracer] = None,
) -> NDArray[DTYPE]:
    """Compute the derivative of the state vector for a single simulation step.

    Equations are derived from the originals in the paper using `rearrange.py`.
    Physical and models constants (g, rho_w, rho_a, etc.) are assumed to be accessible
    globally (imported from `parameters.py`).

    Args:
        s: Current position along streamwise axis [m].
        y: Current state vector (internally converted to dataclass `JetState`).
        params: User and computed-once parameters.

    Optional args for debugging:
        bypass: Map of derivatives to override with provided values.
        tracer: Records all variables (not only the state vector) per simulation step.

    Returns:
        Derivatives of the state vector at s.
    """
    # Convert state vector to dataclass for better readability and type safety.
    yc: JetState = JetState.from_array(y)
    print_debug_state(s, yc, params)
    yc.assert_physically_plausible(params)
    dyds = JetState()

    # Define helper vars.
    pi_2 = DTYPE(np.pi / 2.0)
    pi_4 = DTYPE(np.pi / 4.0)
    TOL = DTYPE(1e-6)

    # "The local direction of the core is assumed to be equal to the fire stream"
    theta_c: DTYPE = yc.theta_f

    # --- PRECOMPUTE SINES, COSINES, VECTORS, ... --- #

    # sin and cos of stream, core, and air phase angles.
    # FIX: 1e-9 added in Andres' version from 25.09.25. Check why.
    sin_f, cos_f = np.sin(yc.theta_f) + 1e-9, np.cos(yc.theta_f)
    sin_c, cos_c = np.sin(theta_c) + 1e-9, np.cos(theta_c)
    sin_a, cos_a = np.sin(yc.theta_a) + 1e-9, np.cos(yc.theta_a)

    # Safe denominators for ODEs to prevent division by ~0.
    den_f: DTYPE = sin_f
    den_a: DTYPE = sin_a
    if abs(den_f) < TOL:
        logger.warning(f"sin(theta_f)≈0 at {s=:.3f}; applying floor={TOL:g}")
        den_f = np.sign(sin_f) * max(abs(sin_f), TOL)
    if abs(den_a) < TOL:
        logger.warning(f"sin(theta_a)≈0 at {s=:.3f}; applying floor={TOL:g}")
        den_a = np.sign(sin_a) * max(abs(sin_a), TOL)

    # Unit vectors of phase directions (angles are w.r.t. vertical axis).
    e_c: NDArray[DTYPE] = np.array([sin_c, cos_c], dtype=DTYPE)  # core streamwise
    e_a: NDArray[DTYPE] = np.array([sin_a, cos_a], dtype=DTYPE)  # air streamwise

    n_c: NDArray[DTYPE] = rotate90_cw(e_c)  # core radial (90° clockwise)
    core_dotprod: DTYPE = np.dot(e_c, n_c)
    assert np.isclose(core_dotprod, 0.0, atol=1e-6), (
        f"Core vectors {e_c=}, {n_c=} must be orthogonal but {core_dotprod=} != 0"
    )

    # Relative vector from core to air phase.
    U_ca: NDArray = (yc.Ua * e_a) - (yc.Uc * e_c)

    # --- MASS AND MOMENTUM TRANSFER --- #

    # TODO: Mathematica notebook used abs for a number of calculations, article didn't.
    # Double-check where they are actually required.

    # Eq. 15 mass flow surroundings -> jet (air entrainment).
    m_sur2f = alpha * np.pi * rho_a * yc.Ua * yc.Df
    assert m_sur2f >= 0.0, f"{m_sur2f=} must be >= 0 (air is entrained)"

    # Eq. 16 mass flow air -> surroundings.
    m_a2sur = np.pi * rho_a * yc.Df * yc.Ua * abs(np.sin(yc.theta_f - yc.theta_a))
    assert m_a2sur >= 0.0, f"{m_a2sur=} must be >= 0 (air leaves the stream)"

    # Eq. 18 momentum exchange air -> surroundings.
    f_a2sur = m_a2sur * yc.Ua * np.cos(yc.theta_f - yc.theta_a)
    f_ra2sur = m_a2sur * yc.Ua * abs(np.sin(yc.theta_f - yc.theta_a))
    assert f_a2sur >= 0.0, f"{f_a2sur=} (momentum air->spray streamwise) must be >= 0"
    assert f_ra2sur >= 0.0, f"{f_ra2sur=} (momentum air->spray radial) must be >= 0"

    # Eq. 20 drag momentum.
    f_c2a_common = pi_2 * F * yc.Dc * np.linalg.norm(U_ca)
    f_c2a: float = f_c2a_common * abs(np.dot(U_ca, e_c))
    f_rc2a: float = sin_a * f_c2a_common * abs(np.dot(U_ca, n_c))
    assert f_c2a >= 0.0, f"{f_c2a=} (momentum core->air streamwise) must be >= 0"
    assert f_rc2a >= 0.0, f"{f_rc2a=} (momentum core->air radial) must be >= 0"

    # Eq. 22 liquid surface break-up efficiency factor.
    Delta = yc.Df  # radial integral scale of the jet, assumed to be core diameter
    epsilon: float = 0.012 * (s / (Delta * np.sqrt(params.weber)))

    # Mass and momentum transfer terms for each spray class.
    sin_s, cos_s, den_s = (np.zeros(num_drop_classes) for _ in range(3))
    m_c2s, m_s2sur, f_c2s, f_rc2s, f_s2a, f_rs2a, f_s2sur, f_rs2sur, u_rc2s = (
        np.zeros(num_drop_classes) for _ in range(9)
    )
    for i in range(num_drop_classes):
        sin_s[i], cos_s[i] = np.sin(yc.theta_s[i]), np.cos(yc.theta_s[i])

        # Safeguard against div-by-zero if sin(theta_s[i]) is used in denominator.
        den_s[i] = sin_s[i]
        if abs(den_s[i]) < TOL:
            logger.warning(
                f"sin(theta_s[{i}]) near 0 at {s=:.3f}; applying floor={TOL:g}"
            )
            den_s[i] = np.sign(sin_s[i]) * max(abs(sin_s[i]), TOL)

        # Construct stream phase unit vectors.
        e_s: NDArray = np.array([sin_s[i], cos_s[i]])  # streamwise
        n_s: NDArray = rotate90_cw(e_s)  # radial
        spray_dotprod: DTYPE = np.dot(e_s, n_s)
        assert np.isclose(spray_dotprod, 0.0, atol=1e-6), (
            f"Spray vectors {e_s=}, {n_s=} must be orthogonal but {spray_dotprod=} != 0"
        )

        # Relative velocity spray -> air.
        U_sa: NDArray = (yc.Ua * e_a) - (yc.Us[i] * e_s)

        # Eq. 17 mass flow spray -> surroundings per unit s [kg/(m*s)].
        # NOTE: Typos in research article, confirmed by the author. Factors rho_w, Pi,
        # Df were missing, and it used the wrong diameter index (Ds instead of Df).
        m_s2sur[i] = (
            (2.0 / 3.0)
            * yc.ND[i]
            * rho_w
            * (d_drop[i] ** 3)
            * (np.pi / yc.Df)
            * abs(np.sin(yc.theta_s[i] - yc.theta_f))
        )
        assert m_s2sur[i] >= 0.0, (
            f"{m_s2sur[i]=} kg/(m*s) (mass spray->surroungins) must be >= 0"
        )

        # Eq. 19 momentum exchange spray -> surroundings.
        f_s2sur[i] = m_s2sur[i] * yc.Us[i] * abs(np.cos(yc.theta_f - yc.theta_s[i]))
        f_rs2sur[i] = m_s2sur[i] * yc.Us[i] * abs(np.sin(yc.theta_f - yc.theta_s[i]))
        assert f_s2sur[i] >= 0, (
            f"{f_s2sur[i]=} N/m (momentum spray->surroundings streamwise) must be >= 0"
        )
        assert f_rs2sur[i] >= 0, (
            f"{f_rs2sur[i]=} N/m (momentum spray->surroundings radial) must be >= 0"
        )

        if params._is_post_breakup:
            # Core does not exist anymore.
            m_c2s[i] = 0.0
            f_c2s[i] = 0.0
            f_rc2s[i] = 0.0
        else:
            # Eq. 23 mass-averaged radial velocity of drops relative to core surface.
            u_rc2s[i] = 0.05 * yc.Us[i]

            # Eq. 21 radial mass flow core -> spray.
            m_c2s[i] = epsilon * u_rc2s[i] * np.pi * rho_w * yc.Dc
            assert m_c2s[i] >= -TOL, (
                f"{m_c2s[i]=} kg/(m*s) (mass core->spray) must be >= 0"
            )

            # Eq. 24 momentum transfer core -> spray.
            f_c2s[i] = m_c2s[i] * yc.Uc * abs(np.cos(yc.theta_s[i] - theta_c))
            f_rc2s[i] = m_c2s[i] * yc.Uc * abs(np.sin(yc.theta_s[i] - theta_c))
            assert f_c2s[i] >= 0, (
                f"{f_c2s[i]=} N/m (momentum core->spray streamwise) must be >= 0"
            )
            assert f_rc2s[i] >= 0, (
                f"{f_rc2s[i]=} N/m (momentum core->spray radial) must be >= 0"
            )

        # Eq. 25 drag force between spray and air.
        Re_d: float = get_reynolds_number(yc.Us[i], d_drop[i])
        C_d: float = get_drag_coefficient(Re_d)
        f_s2a_common: float = (
            pi_4
            * rho_a
            * C_d
            * (d_drop[i] ** 2)
            * (yc.ND[i] / yc.Us[i])
            * np.linalg.norm(U_sa)
        )
        f_s2a[i] = f_s2a_common * abs(np.dot(U_sa, e_s))
        f_rs2a[i] = sin_s[i] * f_s2a_common * abs(np.dot(U_sa, n_s))
        assert f_s2a[i] >= 0, (
            f"{f_s2a[i]=} N/m (momentum spray->air streamwise) must be >= 0"
        )
        assert f_rs2a[i] >= 0, (
            f"{f_rs2a[i]=} N/m (momentum spray->air radial) must be >= 0"
        )

    m_c2s_total: float = np.sum(m_c2s)  # eq 21
    assert m_c2s_total <= (
        m_in := pi_4 * rho_w * (params.nozzle_diameter**2) * params.injection_speed
    ), f"Mass transfer core->spray {m_c2s_total=} must be <= intake {m_in=} kg/(m*s)"
    m_s2sur_total: float = np.sum(m_s2sur)  # eq 17
    f_s2a_total: float = np.sum(f_s2a)  # eq 25 top
    f_rs2a_total: float = np.sum(f_rs2a)  # eq 25 bottom
    f_s2sur_total: float = np.sum(f_s2sur)  # eq 19 top
    f_rs2sur_total: float = np.sum(f_rs2sur)  # eq 19 bottom

    # --- GOVERNING EQUATIONS --- #
    # Rearranged from printed versions using rearrange.py helper script.

    # The core phase only exists until s_brk.
    if params._is_post_breakup:
        dyds.Uc = DTYPE(0.0)
        dyds.Dc = DTYPE(0.0)
    elif s < params.s_brk:
        # Eq 1, 2 core phass mass and momentum conservation.
        dyds.Uc = -(np.pi * yc.Dc**2 * g * rho_w * cos_c + 4 * f_c2a) / (
            np.pi * yc.Dc**2 * yc.Uc * rho_w
        )
        assert dyds.Uc <= TOL, f"{dyds.Uc=:.3f}m/s² Uc can not accelerate"

        dyds.Dc = (
            np.pi * yc.Dc**2 * g * rho_w * cos_c - 4 * yc.Uc * m_c2s_total + 4 * f_c2a
        ) / (2 * np.pi * yc.Dc * yc.Uc**2 * rho_w)
    else:
        logger.debug("--- CORE PHASE BREAKUP ---")
        params._is_post_breakup = True

        # Create droplets at breakup point.
        for i in range(num_drop_classes):
            yc.ND[i] = p_d[i] * (3.0 / 2.0) * (yc.Uc * yc.Dc**2) / (d_drop[i] ** 3)
            dyds.ND[i] += yc.ND[i]

    # Eq 3, 4, 5 air phase mass, streamwise momentum, radial momentum conservation.
    # Can temporarily increase (Mathematice plots show "upward bumps").
    dyds.Ua = (
        4
        * (yc.Ua * m_a2sur - yc.Ua * m_sur2f - f_a2sur + f_c2a + f_s2a_total)
        / (np.pi * yc.Da**2 * yc.Ua * rho_a)
    )

    dyds.Da = (
        -2
        * (2 * yc.Ua * m_a2sur - 2 * yc.Ua * m_sur2f - f_a2sur + f_c2a + f_s2a_total)
        / (np.pi * yc.Da * yc.Ua**2 * rho_a)
    )
    assert dyds.Da >= -TOL, f"{dyds.Da=} air phase diameter can't decrease."

    # FIX: My version on top, Andres' from 25.09.25 6:58 CET below
    dyds.theta_a = (
        4
        * (f_ra2sur - f_rc2a - f_rs2a_total)
        / (np.pi * yc.Da**2 * yc.Ua**2 * rho_a * den_a)
    )
    # dyds.theta_a = (
    #    -1
    #    * (
    #        np.pi * yc.Df**2 * g * rho_a * sin_f
    #        - np.pi * yc.Df**2 * g * yc.rho_f * sin_f
    #        - 4 * f_ra2sur
    #        - 4 * f_rs2sur_total
    #    )
    #    / (np.pi * yc.Df**2 * yc.Uf**2 * yc.rho_f * sin_f)
    # )

    # Eq 6, 7, 8 spray phase.
    for i in range(num_drop_classes):
        dyds.ND[i] = 6 * (m_c2s[i] - m_s2sur[i]) / (np.pi * (d_drop[i] ** 3) * rho_w)

        dyds.Us[i] = -(
            np.pi * yc.ND[i] * (d_drop[i] ** 3) * g * rho_w * cos_c
            + 6 * (yc.Us[i] ** 2) * m_c2s[i]
            - 6 * (yc.Us[i] ** 2) * m_s2sur[i]
            - 6 * yc.Us[i] * f_c2s[i]
            + 6 * yc.Us[i] * f_s2a[i]
            + 6 * yc.Us[i] * f_s2sur[i]
        ) / (np.pi * yc.ND[i] * yc.Us[i] * d_drop[i] ** 3 * rho_w)

        # FIX: Double check this calculation and its compontents. Current physics
        # violation occurs due to theta_s (classes 0 and 1) racing to -inf.
        dyds.theta_s[i] = (
            np.pi * yc.ND[i] * (d_drop[i] ** 3) * g * rho_w * sin_s[i]
            - 6 * yc.Us[i] * f_rc2s[i]
            + 6 * yc.Us[i] * f_rs2a[i]
            + 6 * yc.Us[i] * f_rs2sur[i]
        ) / (np.pi * yc.ND[i] * (yc.Us[i] ** 2) * (d_drop[i] ** 3) * rho_w * den_s[i])

        # if i == 1 and np.rad2deg(yc.theta_s[i]) < 40.0:
        #     logger.debug("Breakpoint")
        #     breakpoint()

    # Eq 9, 10, 11, 12 stream phase.
    dyds.Uf = (
        np.pi * yc.Df**2 * g * rho_a * cos_f
        - np.pi * yc.Df**2 * g * yc.rho_f * cos_f
        + 4 * yc.Uf * m_a2sur * yc.rho_f
        + 4 * yc.Uf * m_s2sur_total * yc.rho_f
        - 4 * yc.Uf * m_sur2f * yc.rho_f
        - 4 * f_a2sur
        - 4 * f_s2sur_total
    ) / (np.pi * yc.Df**2 * yc.Uf * yc.rho_f**2)

    dyds.Df = -(
        np.pi * yc.Df**2 * g * rho_a**2 * rho_w * cos_f
        - np.pi * yc.Df**2 * g * rho_a * yc.rho_f * rho_w * cos_f
        + 4 * yc.Uf * m_a2sur * rho_a * yc.rho_f * rho_w
        + 4 * yc.Uf * m_a2sur * yc.rho_f**2 * rho_w
        + 4 * yc.Uf * m_s2sur_total * rho_a * yc.rho_f**2
        + 4 * yc.Uf * m_s2sur_total * rho_a * yc.rho_f * rho_w
        - 4 * yc.Uf * m_sur2f * rho_a * yc.rho_f * rho_w
        - 4 * yc.Uf * m_sur2f * yc.rho_f**2 * rho_w
        - 4 * f_a2sur * rho_a * rho_w
        - 4 * f_s2sur_total * rho_a * rho_w
    ) / (2 * np.pi * yc.Df * yc.Uf**2 * rho_a * yc.rho_f**2 * rho_w)
    assert dyds.Df >= -TOL, f"{dyds.Df=} m stream diameter can't decrease."

    dyds.theta_f = -(
        np.pi * yc.Df**2 * g * rho_a * sin_f
        - np.pi * yc.Df**2 * g * yc.rho_f * sin_f
        - 4 * f_ra2sur
        - 4 * f_rs2sur_total
    ) / (np.pi * yc.Df**2 * yc.Uf**2 * yc.rho_f * den_f)

    dyds.rho_f = (
        -4
        * (
            m_a2sur * rho_w * (rho_a - yc.rho_f)
            + m_s2sur_total * rho_a * (rho_w - yc.rho_f)
            + m_sur2f * rho_w * (yc.rho_f - rho_a)
        )
        / (np.pi * yc.Df**2 * yc.Uf * rho_a * rho_w)
    )
    assert dyds.rho_f <= TOL, f"{dyds.rho_f=:g} kg/m³/s stream can't gain density."

    # Eq. 13 & 14 Cartesian coordinates of the trajectory.
    # NOTE: Typo in the original article: cos and sin must be swapped, since theta_f is
    # measured relative to vertical axis.
    dyds.x = sin_f
    dyds.y = cos_f
    assert dyds.x >= 0, f"{dyds.x=} stream can't move backwards."
    if s < params.s_brk:
        assert dyds.y >= 0.0, f"{dyds.y=} stream can't fall before core break-up"

    # DEBUG TOOL: Bypass potentially erroneous equations by overwriting their results
    if bypass:
        for name, value in bypass.items():
            if name in JetState.BASE_VARS:
                setattr(dyds, name, value)
            elif name in JetState.SPRAY_VARS:
                values = np.full((num_drop_classes,), value, dtype=DTYPE)
                setattr(dyds, name, values)
            else:
                raise KeyError(
                    f"Can not bypass {name}"
                    + f" because it's not a member of {type(dyds).__name__}"
                )

    # Record all variables with Tracer if available.
    # Angles will be printed in degrees for readability.
    if tracer is not None:
        scalars = {
            "x": yc.x,
            "y": yc.y,
            "Uc": yc.Uc,
            "Ua": yc.Ua,
            "Uf": yc.Uf,
            "Dc": yc.Dc,
            "Da": yc.Da,
            "Df": yc.Df,
            "theta_f_deg": np.rad2deg(yc.theta_f),
            "theta_a_deg": np.rad2deg(yc.theta_a),
            "rho_f": yc.rho_f,
            "m_sur2f": m_sur2f,
            "m_a2sur": m_a2sur,
            "f_a2sur": f_a2sur,
            "f_ra2sur": f_ra2sur,
            "f_c2a": f_c2a,
            "f_rc2a": f_rc2a,
            "f_s2a_total": f_s2a_total,
            "f_rs2a_total": f_rs2a_total,
            "f_s2sur_total": f_s2sur_total,
            "f_rs2sur_total": f_rs2sur_total,
            # "dyds_Uf": float(dyds.Uf),
            # "dyds_theta_f": float(dyds.theta_f),
            # "dyds_Ua": float(dyds.Ua),
            # "dyds_theta_a": float(dyds.theta_a),
            # "dyds_Df": float(dyds.Df),
            # "dyds_Da": float(dyds.Da),
        }
        vectors = {
            "theta_s_deg": np.rad2deg(yc.theta_s),
            "Us": yc.Us,
            "ND": yc.ND,
            "m_c2s": m_c2s,
            "m_s2sur": m_s2sur,
            "f_c2s": f_c2s,
            "f_rc2s": f_rc2s,
            "f_s2a": f_s2a,
            "f_rs2a": f_rs2a,
            "f_s2sur": f_s2sur,
            "f_rs2sur": f_rs2sur,
        }
        tracer.snapshot(s=s, scalars=scalars, vectors=vectors)

    # On to the next simulation step!
    return dyds.to_array()


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------


def rotate90_cw(v: NDArray[np.floating]) -> NDArray[DTYPE]:
    """Rotate 2D vector by 90° clockwise.

    Args:
        v: (2,) vector.

    Returns:
        (2,) rotated vector.
    """
    x, y = v
    return np.array([y, -x], dtype=DTYPE)


def print_debug_state(s: float, state: JetState, params: SimParams) -> None:
    """Print the full state vector using the debug logger.

    Args:
        s: position along streamwise axis.
        state: the JetState at s.
        params: the simulation parameters.
    """
    theta_c: DTYPE = state.theta_f

    th_c_deg, th_a_deg, th_f_deg = np.rad2deg([theta_c, state.theta_a, state.theta_f])
    debug_str = ("post" if params._is_post_breakup else "pre") + " break-up"

    debug_str += f"\n{s=:>7.4f} m"
    debug_str += f"\t x={state.x:>7.4f} m"
    debug_str += f"\t y={state.y:>7.4f} m"
    debug_str += f"\trho_f={state.rho_f:>7.4f} kg/m³"

    debug_str += "\nCore"
    debug_str += f"\t\tUc={state.Uc:>7.4f} m/s "
    debug_str += f"\tDc={state.Dc:>7.4f} m"
    debug_str += f"\ttheta_c={th_c_deg:>7.4f}°"

    debug_str += "\nAir"
    debug_str += f"\t\tUa={state.Ua:>7.4f} m/s"
    debug_str += f"\tDa={state.Da:>7.4f} m "
    debug_str += f"\ttheta_a={th_a_deg:>7.4f}°"

    debug_str += "\nStream"
    debug_str += f"\t\tUf={state.Uf:>7.4f} m/s "
    debug_str += f"\tDf={state.Df:>7.4f} m "
    debug_str += f"\ttheta_f={th_f_deg:>7.4f}°"

    for i in range(num_drop_classes):
        debug_str += f"\nDrop class {i} ({d_drop[i] * 1000:.2f} mm)"
        debug_str += f"\tUs={state.Us[i]:>7.4f} m/s"
        debug_str += f"\tND={state.ND[i]:.2g} /sec"
        debug_str += f"\ttheta_s={np.rad2deg(state.theta_s[i]):>7.4f}°"

    debug_str += "\n"
    logger.debug(debug_str)
    return
