#!/usr/bin/env python3

"""
Importable core logic for the fire stream trajectory model.

Uses SciPy's solve_ivp() to solve a ODE system that models the stream's evolution along
the streamwise axis "s".

Requires initial parameters: injection speed, injection angle, nozzle diameter, maximum
simulation length (s_span).

Parameter "max_step" can be adjusted to trade accuracy for performance.
"""

from .parameters import *

from functools import partial
import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult


def simulate(
    injection_speed: float,
    injection_angle_deg: float,
    nozzle_diameter: float,
    s_span: tuple = (0, 100),
    max_step: float = 1.0,
    debug: bool = False,
) -> tuple[OdeResult, dict[str, int]]:
    """
    Run the fire stream simulation for given injection parameters.

    Parameters:
    - injection_speed: U_0 [m/s]
    - injection_angle_deg: theta_0 [deg]
    - nozzle_diameter: D_0 [m]
    - s_span: (start, end) in streamwise domain s [m]
    - max_step: Max integration step size [m]
    - debug: enable debug mode (with console printouts and auto-activation of PDB)

    Returns:
    - sol: ODE solution object
    - idx: Maps variable names to indices in solution vector.
    """

    # ensure parameter correctness
    assert injection_speed > 0.0, "Injection speed must be positive"
    assert 0 <= injection_angle_deg <= 90, "Injection angle must be in range (0, 90)°"
    assert nozzle_diameter > 0.0, "Nozzle diameter must be greater than 0"
    assert isinstance(s_span, tuple), "s_span must be a tuple"
    assert len(s_span) == 2, "s_span must be in the form (start, end)"
    assert all(
        isinstance(x, (float, int)) and x >= 0 for x in s_span
    ), "Values of s_span must be non-negative"
    assert s_span[1] > s_span[0], "s_span end must be greater than start"
    assert max_step > 0.0, "max_step must be positive"

    # construct state vector and an index for it
    num_drop_classes = len(d_drop)
    idx = build_state_idx(num_drop_classes)
    state_vec = get_initial_state_vector(
        injection_speed, injection_angle_deg, nozzle_diameter, idx
    )
    # parameters (either given or only computed once)
    params = {
        "num_drop_classes": num_drop_classes,
        "s_brk": get_breakup_distance(nozzle_diameter),
        "We_0": get_weber_number(injection_speed, nozzle_diameter),
        "injection_speed": injection_speed,
        "injection_angle_deg": injection_angle_deg,
        "nozzle_diameter": nozzle_diameter,
        # flag used inside function to trigger droplet generation at end of core phase
        "has_breakup_happened": False,
    }
    assert params["s_brk"] > 0.0, "f{params['s_brk']} must be positive"

    # stop simulation when trajectory hits ground
    def hit_ground_event(_, y, idx):
        return y[idx["y"]]  # vertical coordinate

    event_func = partial(hit_ground_event, idx=idx)
    event_func.terminal = True  # pyright: ignore
    event_func.direction = -1  # pyright: ignore

    sol = OdeResult()
    try:
        # this is where the number magic happens
        sol = solve_ivp(
            partial(ode_right_hand_side, idx=idx, params=params, debug=debug),
            s_span,
            y0=state_vec,
            method="RK45",
            max_step=max_step,
            dense_output=True,
            events=[event_func],
        )
    except AssertionError as e:
        if debug:
            print(f"\nAssertionError: {e}\nDropping into debugger...\n")
            import pdb

            pdb.post_mortem()
        raise

    return sol, idx


def ode_right_hand_side(
    s: float, y: npt.NDArray, idx: dict[str, int], params: dict, debug: bool = False
) -> npt.NDArray:
    """
    Encapsulates the right-hand side of the ODE system for the fire stream model. I.e.,
    it computes the derivative (dyds) of every element in the state vector at the
    streamwise coordinate 's'.

    The equations are derived with rearrange.py from the originals in the paper.

    Parameters:
    - s: current streamwise position [m]
    - y: current state vector (1D numpy array)
    - idx: dictionary mapping variable names to indices in the state vector
    - params: dict of given and computed-once parameters
    - debug: enable for printouts

    Returns:
    - dyds: array of derivatives matching the shape of state_vec

    Notes:
    - Physical and models constants (g, rho_w, rho_a, etc.) are assumed to be accessible globally.
    """

    assert len(y) == len(idx), "idx must match state vector y"
    assert np.all(np.isfinite(y)), "Variables must be finite numbers"
    dyds = np.zeros_like(y)

    # extract parameters
    num_drop_classes = params["num_drop_classes"]
    s_brk = params["s_brk"]
    We_0 = params["We_0"]
    has_breakup_happened = params["has_breakup_happened"]

    # extract variables from state vector and clamp to lower bounds if appropriate
    Uc = max(y[idx["Uc"]], 1e-3)
    Ua = max(y[idx["Ua"]], 1e-3)
    Uf = max(y[idx["Uf"]], 1e-3)
    Dc = max(y[idx["Dc"]], 1e-3)
    Da = max(y[idx["Da"]], 1e-3)
    Df = max(y[idx["Df"]], 1e-3)
    theta_a = y[idx["theta_a"]]
    ND, Us, theta_s = (np.zeros(num_drop_classes) for _ in range(3))
    for i in range(num_drop_classes):
        ND[i] = max(y[idx[f"ND_{i}"]], 1e-3)
        Us[i] = max(y[idx[f"Us_{i}"]], 1e-3)
        theta_s[i] = y[idx[f"theta_s_{i}"]]

    theta_f = y[idx["theta_f"]]
    rho_f = max(y[idx["rho_f"]], rho_a)
    x_pos = y[idx["x"]]
    y_pos = y[idx["y"]]

    # "the local direction of the core is assumed to be equal to the fire stream"
    theta_c = theta_f

    # spray phase diameter = sqrt of (stream phase area - core area - air area)
    # rounding errors can lead to Df² - Dc² - Da² becoming negative, clamp to 1e-6!
    # TODO: Check, seemingly never stated explicitly
    Ds = np.sqrt(max(Df**2 - Dc**2 - Da**2, 1e-6))

    # --- DEBUG PRINTOUTS --- #
    if debug:
        print(
            f"\ns={s:>6.3f} m\t{x_pos=:.3f}\t{y_pos=:.3f}\t{rho_f=: >6.3f} kg/m³"
            f"\n Core\t\t{Uc=: >6.3f} m/s\t{Dc=: >6.3f} m"
            f"\n Air\t\t{Ua=: >6.3f} m/s\t{Da=: >6.3f} m\ttheta_a={np.rad2deg(theta_a): >6.3f}°"
            f"\n Stream\t\t{Uf=: >6.3f} m/s\t{Df=: >6.3f} m\ttheta_f={np.rad2deg(theta_f): >6.3f}°"
        )
        for i in range(num_drop_classes):
            print(
                f" Drops[{i}]\tUs={Us[i]: >6.3f} m/s\tND={ND[i]:.2g} /sec\ttheta_s={np.rad2deg(theta_s[i]): >5.3f}°"
            )

    # --- PHYSICAL FEASIBILITY CHECKS --- #
    # helps with debugging

    U0 = params["injection_speed"]
    D0 = params["nozzle_diameter"]
    pi_2 = np.pi / 2.0
    pi_4 = np.pi / 4.0
    # input mass flow from the nozzle [kg / ms]
    m_in = pi_4 * rho_w * (D0**2) * U0
    max_diameter = 50.0

    # Speeds [m/s]
    assert 0.0 <= Uc <= 2 * U0, f"core speed {Uc=} must be in range (0.0, {2*U0})"
    assert 0.0 <= Ua <= 2 * U0, f"air phase speed {Ua=} must be in range (0.0, {2*U0})"
    assert 0.0 <= Uf <= 2 * U0, f"stream speed {Uf=} must be in range (0.0, {2*U0})"

    # Diameters [m]
    assert 0.0 <= Dc <= max_diameter, f"{Dc=} must be in range (0.0, {max_diameter})"
    assert 0.0 <= Da <= max_diameter, f"{Da=} must be in range (0.0, {max_diameter})"
    assert 0.0 <= Ds <= max_diameter, f"{Ds=} must be in range (0.0, {max_diameter})"
    assert 0.0 <= Df <= max_diameter, f"{Df=} must be in range (0.0, {max_diameter})"

    # Angles [rad]
    # TODO: Clarify angle reference (horizontal or vertical)
    # If vertical, the range goes from 0 (vertical up) to 180° (vertical down)
    angle_range = (0, np.pi)
    assert (
        angle_range[0] <= theta_a <= angle_range[1]
    ), f"theta_a={np.rad2deg(theta_a)}° must be in range {np.rad2deg(angle_range)}°"
    assert (
        angle_range[0] <= theta_f <= angle_range[1]
    ), f"theta_f={np.rad2deg(theta_f)=}° must be in range {np.rad2deg(angle_range)}°"
    for i in range(num_drop_classes):
        assert (
            angle_range[0] <= theta_s[i] <= angle_range[1]
        ), f"theta_s[{i}]={np.rad2deg(theta_s[i])}° must be in range {np.rad2deg(angle_range)}°"

    # Spray generation [drops/s]
    for i in range(num_drop_classes):
        assert 0.0 <= ND[i] <= 1e12, f"ND{i}={ND[i]:.2g} must be in range (0, 1e12)"

    # Density [kg/m³]
    assert (
        rho_a - 1e-6 <= rho_f <= rho_w + 1e-6
    ), f"Stream density rho_f={rho_f:.2f} kg/m³ must be between air and water ({rho_a}, {rho_w})"

    # TODO: Ask author for more checks, e.g. magnitude of mass and momentum transfers.

    # --- PRECOMPUTE SINES, COSINES, VECTORS --- #
    # TODO: Ask about angle usage, this seems to be the main difference between the
    # paper and Mathematica notebook.

    sin_f, cos_f = np.sin(theta_f), np.cos(theta_f)
    sin_c, cos_c = np.sin(theta_c), np.cos(theta_c)
    sin_a, cos_a = np.sin(theta_a), np.cos(theta_a)
    sin_fa, cos_fa = np.sin(theta_f - theta_a), np.cos(theta_f - theta_a)

    sin_s, cos_s, cos_cs, sin_fs, cos_fs = (
        np.zeros(num_drop_classes) for _ in range(5)
    )
    for i in range(num_drop_classes):
        sin_s[i], cos_s[i] = np.sin(theta_s[i]), np.cos(theta_s[i])
        sin_fs[i], cos_fs[i] = np.sin(theta_f - theta_s[i]), np.cos(
            theta_f - theta_s[i]
        )
        cos_cs[i] = np.cos(theta_c - theta_s[i])

    ## streamwise unit vector of core phase
    # e_c = np.array([cos_c, sin_c])
    ## radial unit vector of core phase
    # n_c = np.array([-sin_c, cos_c])
    ## streamwise unit vector of air phase
    # e_a = np.array([cos_a, sin_a])

    # NOTE: These are reversed since theta_c,theta_a are relative to vertical
    # streamwise unit vector of core phase
    e_c = np.array([sin_c, cos_c])
    # radial unit vector of core phase
    n_c = np.array([sin_c, -cos_c])  # TODO: Check this
    # streamwise unit vector of air phase
    e_a = np.array([sin_a, cos_a])
    # relative velocity vector core -> air
    U_ca = (Uc * e_c) - (Ua * e_a)

    # --- MASS AND MOMENTUM TRANSFER --- #

    # Eq. 15 mass flow surroundings -> jet (air entrainment)
    m_sur2f = alpha * np.pi * rho_a * Ua * Df
    assert m_sur2f >= -1e-6, f"{m_sur2f=} must be >= 0.0 (air is entrained)"

    # Eq. 16 mass flow air -> surroundings
    # NOTE: Mathematica notebook seems to actually compute -sin_fa but compensates with abs()
    m_a2sur = abs(np.pi * rho_a * Df * Ua * sin_fa)

    # Eq. 18 momentum exchange air -> surroundings
    # NOTE: cos_fa and sin_fa can become negative, so let's use abs again to be safe.
    f_a2sur = m_a2sur * Ua * abs(cos_fa)
    f_ra2sur = m_a2sur * Ua * abs(sin_fa)

    # Eq. 20 drag momentum
    f_c2a_common = pi_2 * F * Dc * np.linalg.norm(U_ca)
    f_c2a = f_c2a_common * np.dot(U_ca, e_c)
    f_rc2a = sin_a * f_c2a_common * np.dot(U_ca, n_c)

    # Eq. 22 liquid surface break-up efficiency factor
    Delta = Df  # radial integral scale of the jet, assumed to be core diameter
    epsilon = 0.012 * (s / (Delta * np.sqrt(We_0)))

    # prepare mass and momentum transfer vars for each spray class
    m_c2s, m_s2sur, f_c2s, f_rc2s, f_s2a, f_rs2a, f_s2sur, f_rs2sur, u_rc2s = (
        np.zeros(num_drop_classes) for _ in range(9)
    )
    # actually compute them
    for i in range(num_drop_classes):

        e_s = np.array([cos_s[i], sin_s[i]])  # streamwise unit vector
        n_s = np.array([-sin_s[i], cos_s[i]])  # radial unit vector
        U_sa = (Us[i] * e_s) - (Ua * e_a)  # relative velocity spray -> air

        # Eq. 17 mass flow spray -> surroundings
        # TODO: Ask author about this equation, there's likely multiple errors.
        # Unit analysis and Mathematica notebook seem to confirm that factors
        # rho_w, Pi, and Df are missing from the printed version.
        # Also, notebook uses Df (stream diameter) while paper uses Ds.
        # Last but not least, it seems like we need to clamp to 0.0 before s_brk to not
        # mess up the evolution of ND. Not explicitly stated in paper, but "significant
        # spray formation only occurs after breakup" is physically plausible.
        m_s2sur[i] = (
            (
                (2.0 / 3.0)
                * ((ND[i] * (d_drop[i] ** 3)) / (Df**2))
                * abs(sin_fs[i])
                * rho_w
                * np.pi
                * Df
            )
            if has_breakup_happened
            else 1e-6
        )
        assert m_s2sur[i] >= -1e-6, f"{m_s2sur[i]=} must be positive"

        # upper bound: total mass of this spray phase
        m_s = ND[i] * rho_w * (np.pi / 6.0) * (d_drop[i] ** 3)
        assert (
            m_s2sur[i] <= m_s + 1e-3
        ), f"{m_s2sur[i]=} must be smaller than spray[{i}] total mass {m_s=}"

        # Eq. 19 momentum exchange spray -> surroundings
        # NOTE: Again using abs to be safe
        f_s2sur[i] = m_s2sur[i] * Us[i] * abs(cos_fs[i])
        f_rs2sur[i] = m_s2sur[i] * Us[i] * abs(sin_fs[i])

        # Eq. 23 mass-averaged radial velocity of drops relative to core surface
        u_rc2s[i] = 0.05 * Us[i]

        # Eq. 21 radial mass flow core -> spray
        m_c2s[i] = epsilon * u_rc2s[i] * np.pi * rho_w * Dc
        assert m_c2s[i] >= -1e-6, f"{m_c2s[i]=} must be positive"

        # Eq. 24 momentum transfer core -> spray
        f_c2s[i] = m_c2s[i] * Uc * cos_cs[i]
        assert f_c2s[i] >= -1e-6, f"{f_c2s[i]=} must be positive"

        # Eq. 25 drag force between spray and air
        Re_d = get_reynolds_number(Us[i], d_drop[i])
        C_d = get_drag_coefficient(Re_d)
        f_s2a_common = (
            pi_4
            * rho_a
            * C_d
            * (d_drop[i] ** 2)
            * (ND[i] / Us[i])
            * np.linalg.norm(U_sa)
        )
        f_s2a[i] = f_s2a_common * abs(np.dot(U_sa, e_s))
        f_rs2a[i] = sin_s[i] * f_s2a_common * abs(np.dot(U_sa, n_s))

    m_c2s_total = np.sum(m_c2s)  # eq 21
    assert (
        m_c2s_total <= m_in
    ), f"Mass transfer core->spray {m_c2s_total} must be <= intake {m_in}"
    m_s2sur_total = np.sum(m_s2sur)  # eq 17
    f_s2a_total = np.sum(f_s2a)  # eq 25 top
    f_rs2a_total = np.sum(f_s2a)  # eq 25 bottom
    f_s2sur_total = np.sum(f_s2sur)  # eq 19 top
    f_rs2sur_total = np.sum(f_rs2sur)  # eq 19 bottom

    # --- GOVERNING EQUATIONS --- #
    # rearranged using SymPy and helper script

    # core phase only exists until s_brk
    if has_breakup_happened:
        dyds[idx["Uc"]] = 0.0
        dyds[idx["Dc"]] = 0.0

    elif s < s_brk:

        # Eq 1, 2 core phass mass and momentum conservation
        dyds[idx["Uc"]] = -(np.pi * Dc**2 * g * rho_w * cos_c + 4 * f_c2a) / (
            np.pi * Dc**2 * Uc * rho_w
        )
        assert dyds[idx["Uc"]] <= 1e-6, f"{Uc=} can't accelerate!"

        dyds[idx["Dc"]] = (
            np.pi * Dc**2 * g * rho_w * cos_c - 4 * Uc * m_c2s_total + 4 * f_c2a
        ) / (2 * np.pi * Dc * Uc**2 * rho_w)
        # assert dyds[idx["Dc"]] <= 1e-6, f"{Dc=} can't increase!"
    else:
        if debug:
            print("--- BREAKUP HAPPENED ---")
        params["has_breakup_happened"] = True

        # Create droplets at breakup point
        for i in range(num_drop_classes):
            ND[i] = p_d[i] * (3.0 / 2.0) * (Uc * Dc**2) / (d_drop[i] ** 3)
            dyds[idx[f"ND_{i}"]] += ND[i]

    # Eq 3, 4, 5 air phase mass, streamwise momentum, radial momentum conservation
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

    dyds[idx["theta_a"]] = (
        4 * (f_ra2sur - f_rc2a - f_rs2a_total) / (np.pi * Da**2 * Ua**2 * rho_a * sin_a)
    )

    # Eq 6, 7, 8
    for i in range(num_drop_classes):
        dyds[idx[f"ND_{i}"]] = (
            6 * (m_c2s[i] - m_s2sur[i]) / (np.pi * d_drop[i] ** 3 * rho_w)
        )

        dyds[idx[f"Us_{i}"]] = -(
            np.pi * ND[i] * d_drop[i] ** 3 * g * rho_w * cos_c
            + 6 * Us[i] ** 2 * m_c2s[i]
            - 6 * Us[i] ** 2 * m_s2sur[i]
            - 6 * Us[i] * f_c2s[i]
            + 6 * Us[i] * f_s2a[i]
            + 6 * Us[i] * f_s2sur[i]
        ) / (np.pi * ND[i] * Us[i] * d_drop[i] ** 3 * rho_w)

        dyds[idx[f"theta_s_{i}"]] = (
            np.pi * ND[i] * d_drop[i] ** 3 * g * rho_w * sin_s[i]
            - 6 * Us[i] * f_rc2s[i]
            + 6 * Us[i] * f_rs2a[i]
            + 6 * Us[i] * f_rs2sur[i]
        ) / (np.pi * ND[i] * Us[i] ** 2 * d_drop[i] ** 3 * rho_w * sin_s[i])

    # Eq 9, 10, 11, 12
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
    assert (
        dyds[idx["rho_f"]] <= 1e-3
    ), f"{dyds[idx["rho_f"]]=} Stream can't gain density!"

    # Eq. 13 & 14 Cartesian coordinates of the trajectory
    # TODO: Clarify angles. Should theta_f be relative to horizon or vertical?
    # In the latter case, cos and sin must probably be swapped.
    dyds[idx["x"]] = sin_f  # cos_f
    assert dyds[idx["x"]] >= 0.0, "Trajectory can't move backwards"

    dyds[idx["y"]] = cos_f  # sin_f
    if s < s_brk:
        assert dyds[idx["y"]] >= 0.0, "Trajectory can't fall before core break-up"

    # --- TEMPORARY WORKAROUND FOR BUGGY ODE --- #
    # By outcommenting these lines, set derivatives to a fixed value.
    # This effectively disables faulty calculations until the root cause can be identified.

    # dyds[idx["Uc"]] = 0.0
    # dyds[idx["Dc"]] = 0.0
    # dyds[idx["Ua"]] = -0.01
    # dyds[idx["Da"]] = 0.01
    dyds[idx["theta_a"]] = dyds[idx["theta_f"]]
    for i in range(num_drop_classes):
        # dyds[idx[f"ND_{i}"]] = 0.0
        dyds[idx[f"Us_{i}"]] = 0.0
        dyds[idx[f"theta_s_{i}"]] = dyds[idx["theta_f"]]
        ...

    # dyds[idx["Uf"]] = 0.0
    # dyds[idx["Df"]] = 0.0
    # dyds[idx["theta_f"]] = -0.01
    dyds[idx["rho_f"]] = -rho_f * 0.05

    # On to the next simulation step!
    return dyds


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------


def build_state_idx(n_classes: int) -> dict[str, int]:
    """
    Builds a dictionary, mapping state variable names to their corresponding index in
    the state vector, following a structured layout. Helps to use state vector consistenly.

    Parameters:
    - n_classes: number of droplet classes

    Returns:
    - idx (dict): mapping from variable names to indices
    """

    assert n_classes > 0
    counter = 0
    idx = {}

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
    var_names_spray = [
        "ND",  # Number of drops [drops/s]
        "Us",  # Spray phase velocity [m/s]
        "theta_s",  # Droplet angle [rad]
    ]
    for i_class in range(n_classes):
        for name in var_names_spray:
            idx[f"{name}_{i_class}"] = counter
            counter += 1

    return idx


def get_initial_state_vector(
    injection_speed: float,
    injection_angle_deg: float,
    nozzle_diameter: float,
    idx: dict,
) -> npt.NDArray:
    """
    Prepare the initial state vector of the fire stream (see appendix of paper).

    Parameters:
    - injection_speed: U_0 [m/s]
    - injection_angle_deg: theta_0 [deg]
    - nozzle_diameter: D_0 [m]
    - idx: Index of state vector (has to be consistent with the simulation)

    Returns:
    - y0: Initial state vector
    """

    # TODO: Check if notebook has notable differences

    injection_angle_rad: float = np.deg2rad(injection_angle_deg)
    # NOTE: Injection angle seems to be above horizon, but others (theta_c, theta_a,
    # theta_f) as angle relative to vertical axis (see first sentence after eq. 2).
    injection_angle_rad = np.pi / 2 - injection_angle_rad

    # initialize state vector with all zeros
    state_vec = np.zeros(len(idx))

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
    # NOTE: different from paper (0.0) to prevent singularity in dyds[Ua] calculation
    state_vec[idx["Da"]] = nozzle_diameter
    state_vec[idx["theta_a"]] = injection_angle_rad

    # Eq 41-44 fire stream phase
    state_vec[idx["Uf"]] = injection_speed
    state_vec[idx["Df"]] = nozzle_diameter
    state_vec[idx["theta_f"]] = injection_angle_rad
    state_vec[idx["rho_f"]] = rho_w

    return state_vec
