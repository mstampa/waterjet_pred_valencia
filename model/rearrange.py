#!/usr/bin/env python

"""
Helper script to rearrange the equations from the research article into the form
dy/ds = f(s, y) expected by the SciPi ODE solver (solve_ivp). Manual labor is reduced
to expanding the derivates with product and chain rules.
"""

from sympy import Derivative, factor, symbols, solve, simplify, pi, sin, cos
from sympy.abc import g, s
from scipy.integrate._ivp.ivp import OdeResult


def print_solution(name: str, deriv: Derivative, sol: OdeResult) -> None:
    """Simplifies a solution for a given symbol and replaces strings such that the
    expression can be copy-pasted directly from the console printout into simulator.py
    with minimal changes.

    Parameters:
    - name: Name of the variable for which the derivative solution should be printed.
    - deriv: SymPy symbol of the derivative.
    - sol: ODE solution object.
    """

    expr = factor(simplify(sol[deriv]))
    txt = str(expr)
    txt = txt.replace("pi", "np.pi")
    for x in ["f", "c", "a"]:
        txt = txt.replace(f"sin(theta_{x})", f"sin_{x}")
        txt = txt.replace(f"cos(theta_{x})", f"cos_{x}")

    txt = txt.replace("sin(theta_s_i)", "sin_s[i]")
    txt = txt.replace("cos(theta_s_i)", "cos_s[i]")
    txt = txt.replace("_i", "[i]")
    print(f'dyds[state_idx["{name}"]] = {txt}')


# --- DEFINE SYMBOLS --- #

# Water and air density, drop diameters
rho_w, rho_a, d_drop_i = symbols("rho_w rho_a d_drop_i")
# Core phase speed, diameter, angle
Uc, Dc, theta_c = symbols("Uc Dc theta_c")
# Air phase speed, diameter, angle
Ua, Da, theta_a = symbols("Ua Da theta_a")
# Spray phase drop formation rate, speed, angle
ND_i, Us_i, theta_s_i = symbols("ND_i Us_i theta_s_i")
# Stream phase speed, diameter, angle, density
Uf, Df, theta_f, rho_f = symbols("Uf Df theta_f rho_f")

# mass transfer from core phase
m_c2s_i, m_c2s_total = symbols("m_c2s_i m_c2s_total")  # to spray phase
# momentum transfer from core phase
f_c2a, f_rc2a = symbols("f_c2a, f_rc2a")  # to air phase
f_c2s_i, f_rc2s_i = symbols("f_c2s_i f_rc2s_i")  # to spray phase
# mass transfer from air phase
m_a2sur = symbols("m_a2sur")  # to surroundings
# momentum transfer from air phase
f_a2sur, f_ra2sur = symbols("f_a2sur f_ra2sur")  # to surroundings
# momentum transfer from spray phase
m_s2sur_i, m_s2sur_total = symbols("m_s2sur_i m_s2sur_total")  # to surroundings
# momentum transfer from spray phase
f_s2a_i, f_s2a_total = symbols("f_s2a_i, f_s2a_total")  # streamwise to air phase
f_rs2a_i, f_rs2a_total = symbols("f_rs2a_i, f_rs2a_total")  # radial to air phase
f_s2sur_i, f_s2sur_total = symbols("f_s2sur_i f_s2sur_total")  # s-wise to surroundings
f_rs2sur_i, f_rs2sur_total = symbols("f_rs2sur_i f_rs2sur_total")  # radial to surr
# mass transfer from surroundings
m_sur2f = symbols("m_sur2f")  # to stream phase (air entrainment)

# --- DEFINE DERIVATIVES --- #
Uc_, Ua_, Uf_ = Derivative(Uc, s), Derivative(Ua, s), Derivative(Uf, s)
Us_i_ = Derivative(Us_i, s)
Dc_, Da_, Df_ = Derivative(Dc, s), Derivative(Da, s), Derivative(Df, s)
theta_c_, theta_a_ = Derivative(theta_c, s), Derivative(theta_a, s)
theta_f_, theta_s_i_ = Derivative(theta_f, s), Derivative(theta_s_i, s)
ND_i_, rho_f_ = Derivative(ND_i, s), Derivative(rho_f, s)

# --- DEFINE EQUATIONS --- #

# Core phase mass conservation
eq1 = (pi / 4) * rho_w * (2 * Dc * Uc * Dc_ + Dc**2 * Uc_) + m_c2s_total

# Core phase momentum conservation
eq2 = (
    (pi / 4) * rho_w * (2 * Dc * (Uc**2) * Dc_ + 2 * Uc * (Dc**2) * Uc_)
    + Uc * m_c2s_total
    + (pi / 4) * rho_w * (Dc**2) * g * cos(theta_c)
    + f_c2a
)

# Air phase mass conservation
eq3 = (pi / 4) * rho_a * (2 * Da * Ua * Da_ + (Da**2) * Ua_) - m_sur2f + m_a2sur

# Air phase streamwise momentum
eq4 = (
    (pi / 4) * rho_a * (2 * Da * (Ua**2) * Da_ + 2 * Ua * (Da**2) * Ua_)
    - f_s2a_total
    + f_a2sur
    - f_c2a
)

# Air phase radial momentum
eq5 = (
    (pi / 4) * rho_a * (Da**2) * (Ua**2) * -(sin(theta_a) * theta_a_)  # pyright: ignore
    - f_rs2a_total
    + f_ra2sur
    - f_rc2a
)

# Spray phase mass conservation
eq6 = (pi / 6) * rho_w * (d_drop_i**3) * ND_i_ - m_c2s_i + m_s2sur_i

# Spray phase streamwise momentum
eq7 = (
    (pi / 6) * rho_w * (d_drop_i**3) * (Us_i * ND_i_ + ND_i * Us_i_)
    + (pi / 6) * rho_w * (d_drop_i**3) * g * (ND_i / Us_i) * cos(theta_c)
    - f_c2s_i
    + f_s2sur_i
    + f_s2a_i
)

# Spray phase radial momentum
eq8 = (
    (pi / 6)
    * rho_w
    * (d_drop_i**3)
    * ND_i
    * Us_i
    * (-(sin(theta_s_i) * theta_s_i_))  # pyright: ignore
    + (pi / 6) * rho_w * (d_drop_i**3) * g * (ND_i / Us_i) * sin(theta_s_i)
    - f_rc2s_i
    + f_rs2sur_i
    + f_rs2a_i
)

# Stream phase mass conservation
eq9 = (
    (pi / 4)
    * ((Df**2) * Uf * rho_f_ + 2 * Df * rho_f * Uf * Df_ + rho_f * (Df**2) * Uf_)
    - m_sur2f
    + m_a2sur
    + m_s2sur_total
)

# Spray streamwise momentum
# TODO: Clarify angle definitions.
# If theta were to mean angle above horizon, the projection of the vertical gravity
# force in the first term on the RHS would require sin(), not cos().
eq10 = (
    (pi / 4)
    * rho_f
    * (
        (Df**2) * (Uf**2) * rho_f_
        + 2 * Df * rho_f * (Uf**2) * Df_
        + 2 * Uf * rho_f * (Df**2) * Uf_
    )
    + (pi / 4) * (rho_f - rho_a) * (Df**2) * g * cos(theta_f)  # here!
    + f_a2sur
    + f_s2sur_total
)

# Spray radial momentum
eq11 = (
    (pi / 4)
    * rho_f
    * (Df**2)
    * (Uf**2)
    * (-(sin(theta_f) * theta_f_))  # pyright: ignore
    + (pi / 4) * (rho_f - rho_a) * (Df**2) * g * sin(theta_f)
    + f_ra2sur
    + f_rs2sur_total
)

# Volume conservation
# NOTE: Likely typo in paper, maybe a leftover from an earlier draft where subscript
# "s" refered to "stream" and not "spray" (seems to be the case in Mathematica notebook).
# Assuming Uf and Df are to be used here instead of Us and Ds.
eq12 = (
    (pi / 4) * (2 * Df * Uf * Df_ + (Df**2) * Uf_)
    - (m_sur2f / rho_a)
    + (m_a2sur / rho_a)
    + (m_s2sur_total / rho_w)
)

# --- REARRANGE --- #

sol = solve(
    (eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12),
    (
        Uc_,
        Dc_,
        Ua_,
        Da_,
        theta_a_,
        ND_i_,
        Us_i_,
        theta_s_i_,
        Uf_,
        Df_,
        theta_f_,
        rho_f_,
    ),
)

# --- PRETTY-PRINT THE SOLUTIONS --- #
# Manually paste output into simulator.py

print_solution("Uc", Uc_, sol)
print_solution("Dc", Dc_, sol)
print_solution("Ua", Ua_, sol)
print_solution("Da", Da_, sol)
print_solution("theta_a", theta_a_, sol)
print_solution("ND_{i}", ND_i_, sol)
print_solution("Us_{i}", Us_i_, sol)
print_solution("theta_s_{i}", theta_s_i_, sol)
print_solution("Uf", Uf_, sol)
print_solution("Df", Df_, sol)
print_solution("theta_f", theta_f_, sol)
print_solution("rho_f", rho_f_, sol)

# All done!
exit()
