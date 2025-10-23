#!/usr/bin/env python3

"""
Importable core logic for the fire stream trajectory model.

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

from .jet_state import JetState
from .parameters import *

from functools import partial
import logging
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from sys import exit
from typing import Dict, Optional, Tuple

simlogger = logging.getLogger("simulator")


def simulate(
    injection_speed: float,
    injection_angle_deg: float,
    nozzle_diameter: float,
    s_span: Tuple = (0, 100),
    max_step: float = 1e-3,
    method: str = "Radau",
    debug: bool = False,
    bypass: Optional[Dict[str, float]] = None,
) -> OdeResult:
    """
    Simulates a fire stream with the given injection parameters.

    Args:
        injection_speed: U_0 [m/s]
        injection_angle_deg: theta_0, relative to horizon [deg]
        nozzle_diameter: D_0 [m]
        s_span: (start, end) in streamwise domain s [m]
        max_step: Max integration step size [m]
        method: See documentation for scipy.integrate.solve_ivp
        debug: enable debug mode (with console printouts and auto-activation of PDB)
        bypass: for debugging purposes, map of ODEs to disable

    Returns:
        sol (OdeResult): the solution
    """

    # setup logging
    log_level = logging.DEBUG if debug else logging.INFO
    simlogger.setLevel(log_level)
    if not simlogger.handlers:
        h = logging.StreamHandler()
        log_fmt = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
        h.setFormatter(logging.Formatter(log_fmt))
        simlogger.addHandler(h)
        simlogger.propagate = False  # don't let root override us
    for h in simlogger.handlers:
        h.setLevel(log_level)

    # ensure parameter correctness
    assert injection_speed > 0.0, "Injection speed must be > 0"
    assert 0 <= injection_angle_deg <= 90, "Injection angle must be in range (0, 90)°"
    assert nozzle_diameter > 0.0, "Nozzle diameter must be > 0"
    assert isinstance(s_span, tuple), "s_span must be a tuple"
    assert len(s_span) == 2, "s_span must be in the form (start, end)"
    assert all(
        isinstance(x, (float, int)) and x >= 0 for x in s_span
    ), "Values of s_span must be >= 0"
    assert s_span[1] > s_span[0], "s_span end must be > start"
    assert max_step > 0.0, "max_step must be > 0"

    # store given and calculate some derived parameters (e.g., breakup distance)
    params = SimParams(
        injection_speed=injection_speed,
        injection_angle_deg=injection_angle_deg,
        nozzle_diameter=nozzle_diameter,
    )

    # construct initial state vector
    state_vec = JetState.get_initial(
        injection_speed, injection_angle_deg, nozzle_diameter
    )

    # this event stops the simulation when trajectory hits ground
    def hit_ground_event(_, y) -> float:
        idx = JetState.get_idx("y")
        return y[idx]  # vertical coordinate

    ev_ground: partial[float] = partial(hit_ground_event)
    ev_ground.terminal = True  # pyright: ignore
    ev_ground.direction = -1  # pyright: ignore

    # this one stops the sim when there's no water left (stream density close to air)
    def mass_depleted_event(_, y, tol=1e-3) -> float:
        idx = JetState.get_idx("rho_f")
        return y[idx] - (rho_a + tol)

    ev_mass: partial[float] = partial(mass_depleted_event)
    ev_mass.terminal = True  # pyright: ignore
    ev_mass.direction = -1  # pyright: ignore

    # let the number magic begin!
    sol: OdeResult
    try:
        sol = solve_ivp(
            partial(ode_right_hand_side, params=params, bypass=bypass),
            s_span,
            y0=state_vec.to_array(),
            method=method,
            max_step=max_step,
            dense_output=True,
            events=[ev_ground, ev_mass],
        )
    except Exception as e:
        simlogger.error(f"{e}")
        if debug:
            import pdb

            pdb.post_mortem()
            exit(1)
        else:
            raise e

    return sol


def ode_right_hand_side(
    s: float,
    y: NDArray[np.floating],
    params: SimParams,
    bypass: Optional[Dict[str, float]] = None,
) -> NDArray[np.floating]:
    """
    Encapsulates the right-hand side of the ODE system for the fire stream model. I.e.,
    it computes the derivative (dyds) of every element in the state vector at the
    streamwise coordinate 's'.

    The equations are derived with rearrange.py from the originals found in the paper.
    Physical and models constants (g, rho_w, rho_a, etc.) are assumed to be accessible
    globally (imported from parameters.py)

    Args:
        s: current streamwise position [m]
        y: current state vector. Will be converted to and from JetState internally.
        params: given and computed-once parameters
        bypass: for debugging purposes, map of ODEs to disable

    Returns:
        dyds: vector of state derivatives
    """
    # state vector as dataclass for better readability and type safety
    yc = JetState.from_array(y)
    debug_printout(s, yc, params)
    yc.assert_physically_plausible(params)
    dyds = JetState()

    # helper vars for performance
    pi_2, pi_4 = np.pi / 2.0, np.pi / 4.0
    TOL = 1e-6

    # "the local direction of the core is assumed to be equal to the fire stream"
    theta_c = yc.theta_f

    # --- PRECOMPUTE SINES, COSINES, VECTORS, ... --- #

    # stream, core, air
    # FIX: 1e-9 added in Andres' version 25.09.25 06:58 CET
    sin_f, cos_f = np.sin(yc.theta_f) + 1e-9, np.cos(yc.theta_f)
    sin_c, cos_c = np.sin(theta_c) + 1e-9, np.cos(theta_c)
    sin_a, cos_a = np.sin(yc.theta_a) + 1e-9, np.cos(yc.theta_a)

    # safe denominator for angle ODEs
    den_a: float = sin_a
    if abs(den_a) < TOL:
        simlogger.warning(f"sin(theta_a)≈0 at {s=:.3f}; applying floor={TOL:g}")
        den_a = np.sign(sin_a) * max(abs(sin_a), TOL)

    # unit vectors of phase directions (angles are w.r.t. vertical axis)
    e_c: NDArray = np.array([sin_c, cos_c])  # core streamwise
    e_a: NDArray = np.array([sin_a, cos_a])  # air streamwise

    n_c: NDArray = rotate90_cw(e_c)  # core radial (90° clockwise)
    assert np.isclose(np.dot(e_c, n_c), 0.0), "Core vectors must be orthogonal"

    # relative vector from core to air phase
    U_ca: NDArray = (yc.Ua * e_a) - (yc.Uc * e_c)

    # --- MASS AND MOMENTUM TRANSFER --- #
    # NOTE: Mathematica notebook used abs() for a number of calculations, article didn't

    # Eq. 15 mass flow surroundings -> jet (air entrainment)
    m_sur2f = alpha * np.pi * rho_a * yc.Ua * yc.Df
    assert m_sur2f >= -1e-6, f"{m_sur2f=} must be >= 0 (air is entrained)"

    # Eq. 16 mass flow air -> surroundings
    m_a2sur = abs(yc.Ua * (sin_f * cos_a - sin_a * cos_f)) * yc.Df * rho_a
    assert m_a2sur >= 0.0, f"{m_a2sur=} must be >= 0 (air leaves the stream)"

    # Eq. 18 momentum exchange air -> surroundings
    f_a2sur = m_a2sur * yc.Ua * np.cos(yc.theta_f - yc.theta_a)
    f_ra2sur = m_a2sur * yc.Ua * abs(np.sin(yc.theta_f - yc.theta_a))
    assert f_a2sur >= 0.0, f"{f_a2sur=} (momentum air->spray streamwise) must be >= 0"
    assert f_ra2sur >= 0.0, f"{f_ra2sur=} (momentum air->spray radial) must be >= 0"

    # Eq. 20 drag momentum
    f_c2a_common = pi_2 * F * yc.Dc * np.linalg.norm(U_ca)
    f_c2a: float = f_c2a_common * abs(np.dot(U_ca, e_c))
    f_rc2a: float = sin_a * f_c2a_common * abs(np.dot(U_ca, n_c))
    assert f_c2a >= 0.0, f"{f_c2a=} (momentum core->air streamwise) must be >= 0"
    assert f_rc2a >= 0.0, f"{f_rc2a=} (momentum core->air radial) must be >= 0"

    # Eq. 22 liquid surface break-up efficiency factor
    Delta = yc.Df  # radial integral scale of the jet, assumed to be core diameter
    epsilon: float = 0.012 * (s / (Delta * np.sqrt(params.weber)))

    # mass and momentum transfer vars for each spray class
    sin_s, cos_s, den_s = (np.zeros(params.num_drop_classes) for _ in range(3))
    m_c2s, m_s2sur, f_c2s, f_rc2s, f_s2a, f_rs2a, f_s2sur, f_rs2sur, u_rc2s = (
        np.zeros(params.num_drop_classes) for _ in range(9)
    )
    for i in range(params.num_drop_classes):

        sin_s[i], cos_s[i] = np.sin(yc.theta_s[i]), np.cos(yc.theta_s[i])
        # safeguard against div-by-zero if sin(theta_s[i]) is used in denominator
        den_s[i] = sin_s[i]
        if abs(den_s[i]) < TOL:
            simlogger.warning(
                f"sin(theta_s[{i}]) near 0 at {s=:.3f}; applying floor={TOL:g}"
            )
            den_s[i] = np.sign(sin_s[i]) * max(abs(sin_s[i]), TOL)

        # unit vectors
        e_s: NDArray = np.array([sin_s[i], cos_s[i]])  # streamwise
        n_s: NDArray = rotate90_cw(e_s)  # radial
        assert np.isclose(np.dot(e_s, n_s), 0.0), f"Spray vectors must be orthogonal"

        # relative velocity spray -> air
        U_sa: NDArray = (yc.Ua * e_a) - (yc.Us[i] * e_s)

        # Eq. 17 mass flow spray -> surroundings per unit s [kg/(m*s)]
        # NOTE: Typos in research article, confirmed by the author.
        # Factors rho_w, Pi, Df were missing, and it used the wrong diameter
        # (Ds instead of Df).
        m_s2sur[i] = (
            (2.0 / 3.0)
            * yc.ND[i]
            * rho_w
            * (d_drop[i] ** 3)
            * (np.pi / yc.Df)
            * abs(np.sin(yc.theta_s[i] - yc.theta_f))
        )
        assert (
            m_s2sur[i] >= 0.0
        ), f"{m_s2sur[i]=} kg/(m*s) (mass spray->surroungins) must be >= 0"

        # Eq. 19 momentum exchange spray -> surroundings
        f_s2sur[i] = m_s2sur[i] * yc.Us[i] * abs(np.cos(yc.theta_s[i] - yc.theta_f))
        f_rs2sur[i] = m_s2sur[i] * yc.Us[i] * abs(np.sin(yc.theta_s[i] - yc.theta_f))
        assert (
            f_s2sur[i] >= 0
        ), f"{f_s2sur[i]=} N/m (momentum spray->surroundings streamwise) must be >= 0"
        assert (
            f_rs2sur[i] >= 0
        ), f"{f_rs2sur[i]=} N/m (momentum spray->surroundings radial) must be >= 0"

        if params._is_post_breakup:
            # core does not exist anymore
            m_c2s[i] = f_c2s[i] = f_rc2s[i] = 0.0
        else:
            # Eq. 23 mass-averaged radial velocity of drops relative to core surface
            u_rc2s[i] = 0.05 * yc.Us[i]

            # Eq. 21 radial mass flow core -> spray
            m_c2s[i] = epsilon * u_rc2s[i] * np.pi * rho_w * yc.Dc
            assert (
                m_c2s[i] >= -TOL
            ), f"{m_c2s[i]=} kg/(m*s) (mass core->spray) must be >= 0"

            # Eq. 24 momentum transfer core -> spray
            f_c2s[i] = m_c2s[i] * yc.Uc * abs(np.cos(yc.theta_s[i] - theta_c))
            f_rc2s[i] = m_c2s[i] * yc.Uc * abs(np.sin(yc.theta_s[i] - theta_c))
            assert (
                f_c2s[i] >= 0
            ), f"{f_c2s[i]=} N/m (momentum core->spray streamwise) must be >= 0"
            assert (
                f_rc2s[i] >= 0
            ), f"{f_rc2s[i]=} N/m (momentum core->spray radial) must be >= 0"

        # Eq. 25 drag force between spray and air
        Re_d: float = get_reynolds_number(yc.Us[i], d_drop[i])
        C_d: float = get_drag_coefficient(Re_d)  # FIX: scalable by 0.01 for debugging
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
        assert (
            f_s2a[i] >= 0
        ), f"{f_s2a[i]=} N/m (momentum spray->air streamwise) must be >= 0"
        assert (
            f_rs2a[i] >= 0
        ), f"{f_rs2a[i]=} N/m (momentum spray->air radial) must be >= 0"

    m_c2s_total: float = np.sum(m_c2s)  # eq 21
    assert m_c2s_total <= (
        m_in := pi_4 * rho_w * (params.nozzle_diameter**2) * params.injection_speed
    ), f"Mass transfer core->spray {m_c2s_total} must be <= intake {m_in} kg/(m*s)"
    m_s2sur_total: float = np.sum(m_s2sur)  # eq 17
    f_s2a_total: float = np.sum(f_s2a)  # eq 25 top
    f_rs2a_total: float = np.sum(f_rs2a)  # eq 25 bottom
    f_s2sur_total: float = np.sum(f_s2sur)  # eq 19 top
    f_rs2sur_total: float = np.sum(f_rs2sur)  # eq 19 bottom

    # --- GOVERNING EQUATIONS --- #
    # rearranged using SymPy and helper script

    # core phase only exists until s_brk
    if params._is_post_breakup:
        dyds.Uc = np.float32(0.0)
        dyds.Dc = np.float32(0.0)

    elif s < params.s_brk:

        # Eq 1, 2 core phass mass and momentum conservation
        dyds.Uc = -(np.pi * yc.Dc**2 * g * rho_w * cos_c + 4 * f_c2a) / (
            np.pi * yc.Dc**2 * yc.Uc * rho_w
        )
        assert dyds.Uc <= TOL, f"{dyds.Uc=:.3f}m/s² Uc can not accelerate"

        dyds.Dc = (
            np.pi * yc.Dc**2 * g * rho_w * cos_c - 4 * yc.Uc * m_c2s_total + 4 * f_c2a
        ) / (2 * np.pi * yc.Dc * yc.Uc**2 * rho_w)

    else:
        simlogger.debug("--- CORE PHASE BREAKUP ---")
        params._is_post_breakup = True

        # Create droplets at breakup point
        for i in range(params.num_drop_classes):
            yc.ND[i] = p_d[i] * (3.0 / 2.0) * (yc.Uc * yc.Dc**2) / (d_drop[i] ** 3)
            dyds.ND[i] += yc.ND[i]

    # Eq 3, 4, 5 air phase mass, streamwise momentum, radial momentum conservation
    # can temporarily increase (Mathematice plots show "bumps")
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
    assert dyds.Da >= -TOL, f"{dyds.Da=} Air diameter can't decrease"

    # FIX: My version on top, Andres' from 25.09.25 6:58 CET below
    # dyds[idx["theta_a"]] = (
    #    4 * (f_ra2sur - f_rc2a - f_rs2a_total) / (np.pi * Da**2 * Ua**2 * rho_a * den_a)
    # )
    dyds.theta_a = (
        -1
        * (
            np.pi * yc.Df**2 * g * rho_a * sin_f
            - np.pi * yc.Df**2 * g * yc.rho_f * sin_f
            - 4 * f_ra2sur
            - 4 * f_rs2sur_total
        )
        / (np.pi * yc.Df**2 * yc.Uf**2 * yc.rho_f * sin_f)
    )

    # Eq 6, 7, 8 spray phase
    for i in range(params.num_drop_classes):

        dyds.ND[i] = 6 * (m_c2s[i] - m_s2sur[i]) / (np.pi * (d_drop[i] ** 3) * rho_w)

        dyds.Us[i] = -(
            np.pi * yc.ND[i] * (d_drop[i] ** 3) * g * rho_w * cos_c
            + 6 * (yc.Us[i] ** 2) * m_c2s[i]
            - 6 * (yc.Us[i] ** 2) * m_s2sur[i]
            - 6 * yc.Us[i] * f_c2s[i]
            + 6 * yc.Us[i] * f_s2a[i]
            + 6 * yc.Us[i] * f_s2sur[i]
        ) / (np.pi * yc.ND[i] * yc.Us[i] * d_drop[i] ** 3 * rho_w)

        dyds.theta_s[i] = (
            np.pi * yc.ND[i] * (d_drop[i] ** 3) * g * rho_w * sin_s[i]
            - 6 * yc.Us[i] * f_rc2s[i]
            + 6 * yc.Us[i] * f_rs2a[i]
            + 6 * yc.Us[i] * f_rs2sur[i]
        ) / (np.pi * yc.ND[i] * (yc.Us[i] ** 2) * (d_drop[i] ** 3) * rho_w * den_s[i])

    # Eq 9, 10, 11, 12 stream phase
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
    assert dyds.Df >= -TOL, f"{dyds.Df=} m Stream diameter can't decrease"

    dyds.theta_f = -(
        np.pi * yc.Df**2 * g * rho_a * sin_f
        - np.pi * yc.Df**2 * g * yc.rho_f * sin_f
        - 4 * f_ra2sur
        - 4 * f_rs2sur_total
    ) / (np.pi * yc.Df**2 * yc.Uf**2 * yc.rho_f * sin_f)

    dyds.rho_f = (
        -4
        * (
            m_a2sur * rho_w * (rho_a - yc.rho_f)
            + m_s2sur_total * rho_a * (rho_w - yc.rho_f)
            + m_sur2f * rho_w * (yc.rho_f - rho_a)
        )
        / (np.pi * yc.Df**2 * yc.Uf * rho_a * rho_w)
    )
    assert dyds.rho_f <= TOL, f"{dyds.rho_f=:g} kg/m³/s Stream can't gain density"

    # Eq. 13 & 14 Cartesian coordinates of the trajectory
    # NOTE: Typo in the original article, cos and sin must be swapped since theta_f is
    # measured relative to vertical axis.
    dyds.x = sin_f
    dyds.y = cos_f
    assert dyds.x >= 0, f"Stream can't move backwards"
    if s < params.s_brk:
        assert dyds.y >= 0.0, f"{dyds.y=} Stream can't fall before core break-up"

    # DEBUG TOOL: Bypass potentially erroneous equations by overwriting their results
    if bypass:
        pass
        # TODO: Re-implement
        # for name, value in bypass.items():
        #    dyds[idx[name]] = value

    # On to the next simulation step!
    return dyds.to_array()


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------


def debug_printout(s: float, state: JetState, params: SimParams) -> None:
    """ """
    theta_c = state.theta_f

    th_c_deg, th_a_deg, th_f_deg = np.rad2deg([theta_c, state.theta_a, state.theta_f])
    debug_str = ("post" if params._is_post_breakup else "pre") + " break-up"

    debug_str += f"\n{s=:>6.3f} m"
    debug_str += f"\t x={state.x:>6.3f} m"
    debug_str += f"\t y={state.y:>6.3f} m"
    debug_str += f"\trho_f={state.rho_f:>6.3f} kg/m³"

    debug_str += f"\nCore"
    debug_str += f"\t\tUc={state.Uc:>6.3f} m/s "
    debug_str += f"\tDc={state.Dc:>6.3f} m"
    debug_str += f"\ttheta_c={th_c_deg:>6.3f}°"

    debug_str += f"\nAir"
    debug_str += f"\t\tUa={state.Ua:>6.3f} m/s"
    debug_str += f"\tDa={state.Da:>6.3f} m "
    debug_str += f"\ttheta_a={th_a_deg:>6.3f}°"

    debug_str += f"\nStream"
    debug_str += f"\t\tUf={state.Uf:>6.3f} m/s "
    debug_str += f"\tDf={state.Df:>6.3f} m "
    debug_str += f"\ttheta_f={th_f_deg:>6.3f}°"

    for i in range(params.num_drop_classes):
        debug_str += f"\nDrop class {i} ({d_drop[i]} m)"
        debug_str += f"\tUs={state.Us[i]:>6.3f} m/s"
        debug_str += f"\tND={state.ND[i]:.2g} /sec"
        debug_str += f"\ttheta_s={np.rad2deg(state.theta_s[i]):>6.3f}°"

    debug_str += "\n"
    simlogger.debug(debug_str)
    return


def rotate90_cw(v: NDArray) -> NDArray:
    """Returns 2D vector v rotated by 90° clockwise"""
    x, y = v
    return np.array([y, -x])
