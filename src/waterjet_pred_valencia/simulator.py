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
    max_step: float = 1e-2,  # 1cm
    method: str = "Radau",
    debug: bool = False,
    bypass: Optional[Dict[str, float]] = None,
) -> Tuple[OdeResult, Dict[str, int]]:
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
        idx (dict): Maps variable names (e.g., "Uc") to indices in solution vector.
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

    # construct state vector and an index for it
    idx = build_state_idx(params.num_drop_classes)
    state_vec = get_initial_state_vector(
        injection_speed, injection_angle_deg, nozzle_diameter, idx
    )

    # this event stops the simulation when trajectory hits ground
    def hit_ground_event(_, y, idx) -> float:
        return y[idx["y"]]  # vertical coordinate

    ev_ground: partial[float] = partial(hit_ground_event, idx=idx)
    ev_ground.terminal = True  # pyright: ignore
    ev_ground.direction = -1  # pyright: ignore

    # this one stops the sim when there's no water left (stream density close to air)
    def mass_depleted_event(_, y, idx, tol=1e-3) -> float:
        return y[idx["rho_f"]] - (rho_a + tol)

    ev_mass: partial[float] = partial(mass_depleted_event, idx=idx)
    ev_mass.terminal = True  # pyright: ignore
    ev_mass.direction = -1  # pyright: ignore

    # let the number magic begin!
    sol = OdeResult()
    rhs: partial[NDArray] = partial(
        ode_right_hand_side, idx=idx, params=params, bypass=bypass
    )
    try:
        sol = solve_ivp(
            rhs,
            s_span,
            y0=state_vec,
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

    return sol, idx


def ode_right_hand_side(
    s: float,
    y: NDArray[np.floating],
    idx: Dict[str, int],
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
        y: current state vector (1D numpy array)
        idx: dictionary mapping variable names to indices in the state vector
        params: given and computed-once parameters
        bypass: for debugging purposes, map of ODEs to disable

    Returns:
        dyds: array of derivatives matching the shape and ordering of idx
    """

    assert len(y) == len(idx), "state vector and its index must have the same length"
    assert np.all(np.isfinite(y)), "Variables must be finite numbers"

    # return vector
    dyds = np.zeros_like(y)

    # helper vars for performance
    pi_2, pi_4 = np.pi / 2.0, np.pi / 4.0
    TOL = 1e-6  # tolerance

    # extract variables from state vector and clamp to lower bounds if appropriate
    Uc = max(y[idx["Uc"]], TOL)
    Ua = max(y[idx["Ua"]], TOL)
    Uf = max(y[idx["Uf"]], TOL)
    Dc = max(y[idx["Dc"]], TOL)
    Da = max(y[idx["Da"]], TOL)
    Df = max(y[idx["Df"]], TOL)
    theta_a = y[idx["theta_a"]]
    ND, Us, theta_s = (np.zeros(params.num_drop_classes) for _ in range(3))
    for i in range(params.num_drop_classes):
        ND[i] = max(y[idx[f"ND_{i}"]], TOL)
        Us[i] = max(y[idx[f"Us_{i}"]], TOL)
        theta_s[i] = y[idx[f"theta_s_{i}"]]

    theta_f = y[idx["theta_f"]]
    rho_f = max(y[idx["rho_f"]], rho_a)
    x_pos = y[idx["x"]]
    y_pos = y[idx["y"]]

    # "the local direction of the core is assumed to be equal to the fire stream"
    theta_c: float = theta_f

    th_c_deg, th_a_deg, th_f_deg = np.rad2deg([theta_c, theta_a, theta_f])
    debug_str = ("post" if params._is_post_breakup else "pre") + " break-up"
    debug_str += f"\n{s=:>6.3f} m\t{x_pos=:.3f} m\t{y_pos=:.3f} m\t{rho_f=:>6.3f} kg/m³"
    debug_str += f"\nCore\t\t{Uc=:>6.3f} m/s \t{Dc=:>6.3f} m\ttheta_c={th_c_deg:>6.3f}°"
    debug_str += f"\nAir\t\t{Ua=:>6.3f} m/s \t{Da=:>6.3f} m\ttheta_a={th_a_deg:>6.3f}°"
    debug_str += (
        f"\nStream\t\t{Uf=:>6.3f} m/s \t{Df=:>6.3f} m\ttheta_f={th_f_deg:>6.3f}°"
    )
    for i in range(params.num_drop_classes):
        debug_str += f"\nDrop class {i} ({d_drop[i]} m)"
        debug_str += f"\tUs={Us[i]:>6.3f} m/s\tND={ND[i]:.2g} /sec"
        debug_str += f"\ttheta_s={np.rad2deg(theta_s[i]):>6.3f}°"

    simlogger.debug(debug_str)

    # --- PHYSICAL FEASIBILITY CHECKS --- #
    # helps with debugging
    # upper limits taken from Mathematice notebook, might need refinement

    # Speeds [m/s]
    U0: float = params.injection_speed
    U_max: float = 2 * U0
    assert 0 <= Uc <= U_max, f"core speed {Uc=} must be in range (0, {U_max})"
    assert 0 <= Ua <= U_max, f"air phase speed {Ua=} must be in range (0, {U_max})"
    assert 0 <= Uf <= U_max, f"stream speed {Uf=} must be in range (0, {U_max})"

    # Diameters [m]
    D0: float = params.nozzle_diameter
    assert 0 <= Dc <= (Dc_max := 2 * D0), f"{Dc=} must be in range (0, {Dc_max})"
    assert 0 <= Da <= (Da_max := 10.0), f"{Da=} must be in range (0, {Da_max})"
    assert 0 <= Df <= (Df_max := 10.0), f"{Df=} must be in range (0, {Df_max})"

    # Angles [rad, deg]
    # NOTE: author clarified that all phase angles are relative to vertical axis.
    # Only the injection angle theta_0 is relative to the horizontal.
    th_low: float = 0.0  # upwards [deg]
    th_high: float = 180.0  # downwards [deg]
    assert (
        th_low <= (th_a_deg := np.rad2deg(theta_a)) <= th_high
    ), f"theta_a={th_a_deg}° must be in range {th_low, th_high}°"
    assert (
        th_low <= (th_f_deg := np.rad2deg(theta_f)) <= th_high
    ), f"theta_f={th_f_deg}° must be in range {th_low, th_high}°"
    for i in range(params.num_drop_classes):
        assert (
            th_low <= (th_f_deg := np.rad2deg(theta_s[i])) <= th_high
        ), f"theta_s[{i}]={th_f_deg}° must be in range {th_low, th_high}°"

    # Spray generation [drops/s]
    ND_max_pre = [3e5, 1e6, 1e6, 1e6, 5e4]
    ND_max_post = [1e8, 1e6, 5e5, 2.5e5, 1e5]
    assert len(ND_max_post) == params.num_drop_classes
    for i in range(params.num_drop_classes):
        if not params._is_post_breakup:
            pass
            # assert (
            #    0.0 <= ND[i] <= ND_max_pre[i]
            # ), f"ND{i}={ND[i]:.2g} must be in range (0, {ND_max_pre[i]:.2g}) pre break-up"
        else:
            assert (
                0.0 <= ND[i] <= ND_max_post[i]
            ), f"ND{i}={ND[i]:.2g} must be in range (0, {ND_max_post[i]:.2g}) post break-up"

    # Density [kg/m³]
    assert (
        rho_a - TOL <= rho_f <= rho_w + TOL
    ), f"Stream density {rho_f=:.2f} kg/m³ must be between air and water ({rho_a}, {rho_w})"

    # --- PRECOMPUTE SINES, COSINES, VECTORS, ... --- #

    # stream, core, air
    sin_f, cos_f = np.sin(theta_f), np.cos(theta_f)
    sin_c, cos_c = np.sin(theta_c), np.cos(theta_c)
    sin_a, cos_a = np.sin(theta_a), np.cos(theta_a)

    # safe denominator for angle ODEs
    den_a: float = sin_a
    if abs(den_a) < TOL:
        simlogger.warning(f"sin(theta_a)≈0 at {s=:.3f}; applying floor={TOL:g}")
        den_a = np.sign(sin_a) * max(abs(sin_a), TOL)

    # unit vectors of phase directions (angles are w.r.t. vertical axis)
    e_c: NDArray = np.array([sin_c, cos_c])  # core streamwise
    e_a: NDArray = np.array([sin_a, cos_a])  # air streamwise

    def rotate90_cw(v: NDArray) -> NDArray:
        """Returns 2D vector v rotated by 90° clockwise"""
        x, y = v
        return np.array([y, -x])

    n_c: NDArray = rotate90_cw(e_c)  # core radial (90° clockwise)
    assert np.isclose(np.dot(e_c, n_c), 0.0), "Core vectors must be orthogonal"

    # relative vector from core to air phase
    U_ca: NDArray = (Ua * e_a) - (Uc * e_c)

    # --- MASS AND MOMENTUM TRANSFER --- #

    # Eq. 15 mass flow surroundings -> jet (air entrainment)
    m_sur2f: float = alpha * np.pi * rho_a * Ua * Df
    assert m_sur2f >= -1e-6, f"{m_sur2f=} must be >= 0 (air is entrained)"

    # Eq. 16 mass flow air -> surroundings
    # NOTE: Mathematica notebook used abs(), article didn't
    m_a2sur = abs(Ua * (sin_f * cos_a - sin_a * cos_f)) * Df * rho_a
    assert m_a2sur >= 0.0, f"{m_a2sur=} must be >= 0 (air leaves the stream)"

    # Eq. 18 momentum exchange air -> surroundings
    f_a2sur: float = m_a2sur * Ua * np.cos(theta_f - theta_a)
    f_ra2sur: float = m_a2sur * Ua * abs(np.sin(theta_f - theta_a))
    assert f_a2sur >= 0.0, f"{f_a2sur=} (momentum air->spray streamwise) must be >= 0"
    assert f_ra2sur >= 0.0, f"{f_ra2sur=} (momentum air->spray radial) must be >= 0"

    # Eq. 20 drag momentum
    # TODO: abs() could be necessary here as well
    f_c2a_common: float = pi_2 * F * Dc * np.linalg.norm(U_ca)
    f_c2a: float = f_c2a_common * abs(np.dot(U_ca, e_c))
    f_rc2a: float = sin_a * f_c2a_common * abs(np.dot(U_ca, n_c))
    assert f_c2a >= 0.0, f"{f_c2a=} (momentum core->air streamwise) must be >= 0"
    assert f_rc2a >= 0.0, f"{f_rc2a=} (momentum core->air radial) must be >= 0"

    # Eq. 22 liquid surface break-up efficiency factor
    Delta: float = Df  # radial integral scale of the jet, assumed to be core diameter
    epsilon: float = 0.012 * (s / (Delta * np.sqrt(params.weber)))

    # mass and momentum transfer vars for each spray class
    sin_s, cos_s, den_s = (np.zeros(params.num_drop_classes) for _ in range(3))
    m_c2s, m_s2sur, f_c2s, f_rc2s, f_s2a, f_rs2a, f_s2sur, f_rs2sur, u_rc2s = (
        np.zeros(params.num_drop_classes) for _ in range(9)
    )
    for i in range(params.num_drop_classes):

        sin_s[i], cos_s[i] = np.sin(theta_s[i]), np.cos(theta_s[i])
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
        U_sa: NDArray = (Ua * e_a) - (Us[i] * e_s)

        # Eq. 17 mass flow spray -> surroundings per unit s [kg/(m*s)]
        # NOTE: Typos in research article, confirmed by the author.
        # Factors rho_w, Pi, Df were missing, and it used the wrong diameter
        # (Ds instead of Df).
        m_s2sur[i] = (
            (2.0 / 3.0)
            * ND[i]
            * rho_w
            * (d_drop[i] ** 3)
            * (np.pi / Df)
            * abs(np.sin(theta_s[i] - theta_f))
        )
        assert (
            m_s2sur[i] >= 0.0
        ), f"{m_s2sur[i]=} kg/(m*s) (mass spray->surroungins) must be >= 0"

        # Eq. 19 momentum exchange spray -> surroundings
        f_s2sur[i] = m_s2sur[i] * Us[i] * abs(np.cos(theta_s[i] - theta_f))
        f_rs2sur[i] = m_s2sur[i] * Us[i] * abs(np.sin(theta_s[i] - theta_f))
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
            u_rc2s[i] = 0.05 * Us[i]

            # Eq. 21 radial mass flow core -> spray
            m_c2s[i] = epsilon * u_rc2s[i] * np.pi * rho_w * Dc
            assert (
                m_c2s[i] >= -TOL
            ), f"{m_c2s[i]=} kg/(m*s) (mass core->spray) must be >= 0"

            # Eq. 24 momentum transfer core -> spray
            f_c2s[i] = m_c2s[i] * Uc * abs(np.cos(theta_s[i] - theta_c))
            f_rc2s[i] = m_c2s[i] * Uc * abs(np.sin(theta_s[i] - theta_c))
            assert (
                f_c2s[i] >= 0
            ), f"{f_c2s[i]=} N/m (momentum core->spray streamwise) must be >= 0"
            assert (
                f_rc2s[i] >= 0
            ), f"{f_rc2s[i]=} N/m (momentum core->spray radial) must be >= 0"

        # Eq. 25 drag force between spray and air
        Re_d: float = get_reynolds_number(Us[i], d_drop[i])
        C_d: float = get_drag_coefficient(Re_d)
        f_s2a_common: float = (
            pi_4
            * rho_a
            * C_d
            * (d_drop[i] ** 2)
            * (ND[i] / Us[i])
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
        m_in := pi_4 * rho_w * (D0**2) * U0
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
        dyds[idx["Uc"]] = 0.0
        dyds[idx["Dc"]] = 0.0

    elif s < params.s_brk:

        # Eq 1, 2 core phass mass and momentum conservation
        dyds[idx["Uc"]] = -(np.pi * Dc**2 * g * rho_w * cos_c + 4 * f_c2a) / (
            np.pi * Dc**2 * Uc * rho_w
        )
        assert (
            dyds[idx["Uc"]] <= TOL
        ), f"{dyds[idx["Uc"]]=:.3f}m/s² Uc must not accelerate"

        dyds[idx["Dc"]] = (
            np.pi * Dc**2 * g * rho_w * cos_c - 4 * Uc * m_c2s_total + 4 * f_c2a
        ) / (2 * np.pi * Dc * Uc**2 * rho_w)

    else:
        simlogger.debug("--- CORE PHASE BREAKUP ---")
        params._is_post_breakup = True

        # Create droplets at breakup point
        for i in range(params.num_drop_classes):
            ND[i] = p_d[i] * (3.0 / 2.0) * (Uc * Dc**2) / (d_drop[i] ** 3)
            dyds[idx[f"ND_{i}"]] += ND[i]

    # Eq 3, 4, 5 air phase mass, streamwise momentum, radial momentum conservation
    # can temporarily increase (Mathematice plots show "bumps")
    dyds[idx["Ua"]] = (
        4
        * (Ua * m_a2sur - Ua * m_sur2f - f_a2sur + f_c2a + f_s2a_total)
        / (np.pi * Da**2 * Ua * rho_a)
    )

    dyds[idx["Da"]] = (
        -2
        * (2 * Ua * m_a2sur - 2 * Ua * m_sur2f - f_a2sur + f_c2a + f_s2a_total)
        / (np.pi * Da * Ua**2 * rho_a)
    )
    assert dyds[idx["Da"]] >= -TOL, f"{dyds[idx["Da"]]=} Air diameter can't decrease"

    dyds[idx["theta_a"]] = (
        4 * (f_ra2sur - f_rc2a - f_rs2a_total) / (np.pi * Da**2 * Ua**2 * rho_a * den_a)
    )

    # Eq 6, 7, 8 spray phase
    for i in range(params.num_drop_classes):

        dyds[idx[f"ND_{i}"]] = (
            6 * (m_c2s[i] - m_s2sur[i]) / (np.pi * (d_drop[i] ** 3) * rho_w)
        )

        dyds[idx[f"Us_{i}"]] = -(
            np.pi * ND[i] * (d_drop[i] ** 3) * g * rho_w * cos_c
            + 6 * (Us[i] ** 2) * m_c2s[i]
            - 6 * (Us[i] ** 2) * m_s2sur[i]
            - 6 * Us[i] * f_c2s[i]
            + 6 * Us[i] * f_s2a[i]
            + 6 * Us[i] * f_s2sur[i]
        ) / (np.pi * ND[i] * Us[i] * d_drop[i] ** 3 * rho_w)

        dyds[idx[f"theta_s_{i}"]] = (
            np.pi * ND[i] * (d_drop[i] ** 3) * g * rho_w * sin_s[i]
            - 6 * Us[i] * f_rc2s[i]
            + 6 * Us[i] * f_rs2a[i]
            + 6 * Us[i] * f_rs2sur[i]
        ) / (np.pi * ND[i] * (Us[i] ** 2) * (d_drop[i] ** 3) * rho_w * den_s[i])

    # Eq 9, 10, 11, 12 stream phase
    dyds[idx["Uf"]] = (
        np.pi * Df**2 * g * rho_a * cos_f
        - np.pi * Df**2 * g * rho_f * cos_f
        + 4 * Uf * m_a2sur * rho_f
        + 4 * Uf * m_s2sur_total * rho_f
        - 4 * Uf * m_sur2f * rho_f
        - 4 * f_a2sur
        - 4 * f_s2sur_total
    ) / (np.pi * Df**2 * Uf * rho_f**2)

    dyds[idx["Df"]] = -(
        np.pi * Df**2 * g * rho_a**2 * rho_w * cos_f
        - np.pi * Df**2 * g * rho_a * rho_f * rho_w * cos_f
        + 4 * Uf * m_a2sur * rho_a * rho_f * rho_w
        + 4 * Uf * m_a2sur * rho_f**2 * rho_w
        + 4 * Uf * m_s2sur_total * rho_a * rho_f**2
        + 4 * Uf * m_s2sur_total * rho_a * rho_f * rho_w
        - 4 * Uf * m_sur2f * rho_a * rho_f * rho_w
        - 4 * Uf * m_sur2f * rho_f**2 * rho_w
        - 4 * f_a2sur * rho_a * rho_w
        - 4 * f_s2sur_total * rho_a * rho_w
    ) / (2 * np.pi * Df * Uf**2 * rho_a * rho_f**2 * rho_w)
    assert dyds[idx["Df"]] >= -TOL, f"Stream diameter can't decrease"

    dyds[idx["theta_f"]] = -(
        np.pi * Df**2 * g * rho_a * sin_f
        - np.pi * Df**2 * g * rho_f * sin_f
        - 4 * f_ra2sur
        - 4 * f_rs2sur_total
    ) / (np.pi * Df**2 * Uf**2 * rho_f * sin_f)

    dyds[idx["rho_f"]] = (
        -4
        * (
            m_a2sur * rho_a * rho_w
            - m_a2sur * rho_f * rho_w
            - m_s2sur_total * rho_a * rho_f
            + m_s2sur_total * rho_a * rho_w
            - m_sur2f * rho_a * rho_w
            + m_sur2f * rho_f * rho_w
        )
        / (np.pi * Df**2 * Uf * rho_a * rho_w)
    )
    dyds[idx["rho_f"]] = (
        -4
        * (
            m_a2sur * rho_w * (rho_a - rho_f)
            + m_s2sur_total * rho_a * (rho_w - rho_f)
            + m_sur2f * rho_w * (rho_f - rho_a)
        )
        / (np.pi * Df**2 * Uf * rho_a * rho_w)
    )
    assert (
        dyds[idx["rho_f"]] <= TOL
    ), f"{dyds[idx["rho_f"]]=:g} kg/m³/s Stream can't gain density!"

    # Eq. 13 & 14 Cartesian coordinates of the trajectory
    # NOTE: Typo in the original article, cos and sin must be swapped since theta_f is
    # measured relative to vertical axis.
    dyds[idx["x"]] = sin_f
    assert dyds[idx["x"]] >= 0.0, f"{dyds[idx["x"]]=} Stream can't move backwards"

    dyds[idx["y"]] = cos_f
    if s < params.s_brk:
        assert (
            dyds[idx["y"]] >= 0.0
        ), f"{dyds[idx["y"]]=} Stream can't fall before core break-up"

    # DEBUG TOOL: Bypass potentially erroneous equations by overwriting their results
    if bypass:
        for name, value in bypass.items():
            dyds[idx[name]] = value

    # On to the next simulation step!
    return dyds


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------


def build_state_idx(n_classes: int) -> Dict[str, int]:
    """
    Builds a dictionary, mapping state variable names to their corresponding index in
    the state vector, following a structured layout. Helps to use state vector consistenly.

    Args:
        n_classes: number of droplet classes

    Returns:
        idx (dict): mapping from variable names to indices
    """

    assert n_classes > 0
    counter = np.uint8(0)
    idx: dict = {}

    # Add these variables to the state vector index
    var_names = [
        "Uc",  # core-phase velocity [m/s]
        "Dc",  # core-phase diameter [m]
        "Ua",  # air phase velocity [m/s]
        "Da",  # air phase diameter [m]
        "theta_a",  # air phase angle [rad]
        "Uf",  # stream velocity [m/s]
        "Df",  # stream diameter [m]
        "theta_f",  # stream angle [rad]
        "rho_f",  # stream density [kg/m³]
        "x",  # x-position [m]
        "y",  # y-position [m]
    ]
    for name in var_names:
        idx[name] = counter
        counter += 1

    # Add one set of these for each droplet class
    var_names_spray: list = [
        "ND",  # Number of drops [drops/s]
        "Us",  # Spray phase velocity [m/s]
        "theta_s",  # Droplet angle [rad]
    ]
    for i_class in range(n_classes):
        for name in var_names_spray:
            idx[f"{name}_{i_class}"] = counter
            counter += 1

    return idx


def assert_physically_plausible(y: NDArray, idx: Dict[str, int]) -> None:

    return


def get_initial_state_vector(
    injection_speed: float,
    injection_angle_deg: float,
    nozzle_diameter: float,
    idx: Dict[str, int],
) -> NDArray[np.floating]:
    """
    Prepare the initial state vector of the fire stream (see appendix of paper).

    Args:
        injection_speed: U_0 [m/s]
        injection_angle_deg: theta_0 (relative to horizon) [deg]
        nozzle_diameter: D_0 [m]
        idx: Mapping of variable names to indices in state vector

    Returns:
        y0: (N,) Initial state vector
    """

    # TODO: Check if notebook has notable differences

    injection_angle_rad: float = np.deg2rad(injection_angle_deg)
    # NOTE: Author confirmed that only the injection angle is measured relative to horizon.
    # All phase angles are measured relative to vertical axis.
    injection_angle_rad: float = np.pi / 2.0 - injection_angle_rad

    # initialize state vector with all zeros
    state_vec: NDArray = np.zeros(len(idx), dtype=np.float32)

    # position
    state_vec[idx["x"]] = 0.0
    state_vec[idx["y"]] = 0.0

    # Eq 31-33 core phase
    state_vec[idx["Uc"]] = injection_speed
    state_vec[idx["Dc"]] = nozzle_diameter

    # Eq 34-36 spray phase (one set per droplet class)
    for i in range(len(d_drop)):
        state_vec[idx[f"ND_{i}"]] = 1e-12
        state_vec[idx[f"Us_{i}"]] = injection_speed
        state_vec[idx[f"theta_s_{i}"]] = injection_angle_rad

    # Eq 38-40 air phase
    # NOTE: -1e-6 from notebook, not paper
    state_vec[idx["Ua"]] = injection_speed - 1e-6
    # NOTE: not 0.0 as in paper to prevent singularity in dyds[Ua] calculation
    state_vec[idx["Da"]] = nozzle_diameter
    state_vec[idx["theta_a"]] = injection_angle_rad

    # Eq 41-44 fire stream phase
    state_vec[idx["Uf"]] = injection_speed
    state_vec[idx["Df"]] = nozzle_diameter
    state_vec[idx["theta_f"]] = injection_angle_rad
    state_vec[idx["rho_f"]] = rho_w

    return state_vec
