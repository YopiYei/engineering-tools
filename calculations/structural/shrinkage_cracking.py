"""Restrained shrinkage cracking in a fully restrained member (Gilbert, 1992).

Gilbert, R. I. (1992). Shrinkage cracking in fully restrained concrete
members. ACI Structural Journal, 89(2), 141-149.

Units: N, mm, MPa (per metre width of slab). Use ``kgf_cm2_to_mpa`` to
convert from kgf/cm2.
"""

import math

import numpy as np


KGF_CM2_TO_MPA = 0.0980665
ES_DEFAULT_MPA = 200_000.0
PHI_DEFAULT = 2.5
EPS_SH_DEFAULT = -600e-6


def kgf_cm2_to_mpa(value):
    return value * KGF_CM2_TO_MPA


def concrete_ec(fc_mpa):
    """Elastic modulus, Ec = 4700 sqrt(f'c) [MPa]."""
    return 4700.0 * math.sqrt(fc_mpa)


def concrete_ft(fc_mpa):
    """Direct tensile strength, ft = 0.4 sqrt(f'c) [MPa] (value used in the paper)."""
    return 0.4 * math.sqrt(fc_mpa)


def australian_min_ratios(fy):
    """Minimum ratios of the Australian code: 14/fy and 21/fy (fy in the same
    units as the constant, i.e. kgf/cm2)."""
    return 14.0 / fy, 21.0 / fy


def gilbert_crack_width(
    length_mm,
    thickness_mm,
    rho,
    bar_diameter_mm,
    fy_mpa,
    fc_mpa=None,
    ft_mpa=None,
    ec_mpa=None,
    es_mpa=ES_DEFAULT_MPA,
    creep_coefficient=PHI_DEFAULT,
    shrinkage_strain=EPS_SH_DEFAULT,
):
    """Average crack width of a fully restrained slab strip (1 m wide).

    Either ``fc_mpa`` or both ``ft_mpa`` and ``ec_mpa`` must be given.

    Returns a dict with the result (``w_mm``, ``n_cracks``, ``spacing_mm``,
    ``yielded``, forces and stresses) and ``steps``: an ordered list of
    ``(label, value, unit)`` used to show the worked calculation.
    """
    if rho <= 0 or length_mm <= 0 or thickness_mm <= 0 or bar_diameter_mm <= 0:
        raise ValueError("L, thickness, rho and bar diameter must be positive.")
    if ec_mpa is None or ft_mpa is None:
        if fc_mpa is None or fc_mpa <= 0:
            raise ValueError("Provide f'c, or both ft and Ec.")
        ec_mpa = concrete_ec(fc_mpa) if ec_mpa is None else ec_mpa
        ft_mpa = concrete_ft(fc_mpa) if ft_mpa is None else ft_mpa

    L = length_mm
    Ac = 1000.0 * thickness_mm
    As = rho * Ac
    n = es_mpa / ec_mpa
    ec_eff = ec_mpa / (1.0 + creep_coefficient)
    n_eff = es_mpa / ec_eff
    s0 = bar_diameter_mm / (10.0 * rho)
    if 3.0 * L <= 2.0 * s0:
        raise ValueError("Transfer length S0 is too large for this length L.")

    steps = [
        ("Ec", ec_mpa, "MPa"),
        ("ft", ft_mpa, "MPa"),
        ("n = Es/Ec", n, ""),
        ("E'e = Ec/(1+phi*)", ec_eff, "MPa"),
        ("n* = Es/E'e", n_eff, ""),
        ("Ac", Ac, "mm2/m"),
        ("As = rho*Ac", As, "mm2/m"),
        ("S0 = db/(10 rho)", s0, "mm"),
    ]

    # First crack
    c1 = 2.0 * s0 / (3.0 * L - 2.0 * s0)
    n_cr = n * rho * ft_mpa * Ac / (c1 + n * rho * (1.0 + c1))
    sigma_c1 = n_cr * (1.0 + c1) / Ac
    sigma_av = 0.5 * (sigma_c1 + ft_mpa)
    steps += [
        ("C1 = 2S0/(3L-2S0)", c1, ""),
        ("Ncr", n_cr / 1000.0, "kN/m"),
        ("sigma_c1 = Ncr(1+C1)/Ac", sigma_c1, "MPa"),
        ("sigma_av = (sigma_c1+ft)/2", sigma_av, "MPa"),
    ]

    # Elastic trial: number of cracks
    a_term = -n_eff * rho * (sigma_av + shrinkage_strain * ec_eff)
    if 0.0 < a_term < ft_mpa:
        psi = a_term / (ft_mpa - a_term)
        s_max = 2.0 * s0 * (1.0 + psi) / (3.0 * psi)
        m = max(1, math.ceil(L / s_max - 1e-9))
    else:
        psi, s_max, m = None, None, 1
    spacing = L / m
    c2 = 2.0 * s0 / (3.0 * spacing - 2.0 * s0)
    n_inf = a_term * Ac / c2 if a_term > 0 else 0.0
    sigma_s2 = n_inf / As
    steps += [
        ("A = -n* rho (sigma_av + eps_sh E'e)", a_term, "MPa"),
        ("psi = A/(ft-A)", psi, ""),
        ("s_max = 2S0(1+psi)/(3 psi)", s_max, "mm"),
        ("Elastic trial: m, s = L/m", f"{m}, {spacing:.0f}", "mm"),
        ("C2 = 2S0/(3s-2S0)", c2, ""),
        ("N(inf) elastic trial = A Ac/C2", n_inf / 1000.0, "kN/m"),
        ("sigma_s2 = N(inf)/As", sigma_s2, "MPa"),
        ("fy", fy_mpa, "MPa"),
    ]

    yielded = sigma_s2 > fy_mpa or n_cr / As > fy_mpa
    if yielded:
        # Steel yields at the crack: single crack, N = fy As (Eq. 32-34)
        sigma_s1 = (n_eff * rho * fy_mpa + shrinkage_strain * es_mpa) / (1.0 + n_eff * rho)
        w = -(sigma_s1 * (3.0 * L - 2.0 * s0) + 2.0 * s0 * fy_mpa) / (3.0 * es_mpa)
        m_final, s_final, n_final = 1, L, fy_mpa * As
        steps += [
            ("Steel yields: single crack, N = fy As", n_final / 1000.0, "kN/m"),
            ("sigma*_s1 = (n* rho fy + eps_sh Es)/(1+n* rho)", sigma_s1, "MPa"),
            ("w = -[sigma*_s1 (3L-2S0) + 2 S0 fy]/(3 Es)", w, "mm"),
        ]
    else:
        sigma_c1_final = n_inf * (1.0 + c2) / Ac
        w = -(
            (sigma_c1_final / ec_eff) * (spacing - 2.0 * s0 / 3.0)
            + shrinkage_strain * spacing
        )
        m_final, s_final, n_final = m, spacing, n_inf
        steps += [
            ("sigma*_c1 = N(inf)(1+C2)/Ac", sigma_c1_final, "MPa"),
            ("w = -[(sigma*_c1/E'e)(s-2S0/3) + eps_sh s]", w, "mm"),
        ]

    return {
        "w_mm": w,
        "n_cracks": m_final,
        "spacing_mm": s_final,
        "yielded": yielded,
        "n_inf_n": n_final,
        "n_cr_n": n_cr,
        "sigma_s2_trial_mpa": sigma_s2,
        "yield_force_n": fy_mpa * As,
        "s0_mm": s0,
        "steps": steps,
    }


def crack_width_vs_rho(
    length_mm,
    thickness_mm,
    bar_diameter_mm,
    fy_mpa,
    fc_mpa,
    rho_min=0.001,
    rho_max=0.012,
    num_points=221,
    **kwargs,
):
    """Crack width for a range of reinforcement ratios (for plotting)."""
    rho = np.linspace(rho_min, rho_max, num_points)
    width = np.array(
        [
            gilbert_crack_width(
                length_mm, thickness_mm, r, bar_diameter_mm, fy_mpa, fc_mpa=fc_mpa, **kwargs
            )["w_mm"]
            for r in rho
        ]
    )
    return rho, width
