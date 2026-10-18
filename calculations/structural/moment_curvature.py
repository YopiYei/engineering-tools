"""Moment-curvature analysis of rectangular reinforced concrete sections.

Fibre analysis for pure bending (P = 0): for each extreme-fibre strain the
neutral axis depth is found from force equilibrium, then M and phi follow.

Constitutive models:
    - Concrete: Park & Paulay (parabola + linear descent) and Mander et al.
      (1988) for unconfined concrete. Tension in concrete is neglected.
    - Steel: elastic-perfectly plastic and Mander (yield plateau + hardening).

Park, R., & Paulay, T. (1975). Reinforced Concrete Structures, Fig. 6.6.
Mander, J. B., Priestley, M. J. N., & Park, R. (1988). Theoretical
stress-strain model for confined concrete. J. Struct. Eng., 114(8).

Units: N, mm, MPa.
"""

import numpy as np


PSI_TO_MPA = 0.00689476

# Park & Paulay Fig. 6.6 data (converted from lb/in2)
PARK_FC = 4000.0 * PSI_TO_MPA
PARK_FY = 40000.0 * PSI_TO_MPA
PARK_ES = 30.0e6 * PSI_TO_MPA
PARK_E0 = 0.002
PARK_ECU = 0.004

# rho, rho' of beams 1-7 in Fig. 6.6 (h = 1.1 d, d' = 0.1 d)
PARK_BEAMS = {
    1: (0.0375, 0.0250),
    2: (0.0375, 0.0125),
    3: (0.0375, 0.0),
    4: (0.0250, 0.0125),
    5: (0.0250, 0.0),
    6: (0.0125, 0.0125),
    7: (0.0125, 0.0),
}


# =========================================================================
# Concrete models (compression positive, strain in mm/mm)
# =========================================================================

def concrete_park_paulay(strain, fc, e0=PARK_E0):
    """Parabola up to e0, then f = fc [1 - 100 (e - e0)] (valid to 0.004)."""
    e = np.asarray(strain, dtype=float)
    f = np.where(
        e <= e0,
        fc * (2.0 * e / e0 - (e / e0) ** 2),
        fc * (1.0 - 100.0 * (e - e0)),
    )
    return np.where(e > 0.0, np.maximum(f, 0.0), 0.0)


def concrete_mander_unconfined(strain, fc, e0=0.002, e_sp=0.006):
    """Mander unconfined concrete: Popovics curve up to 2 e0, then linear
    descent to zero at the spalling strain e_sp."""
    e = np.asarray(strain, dtype=float)
    ec = 4700.0 * np.sqrt(fc)
    esec = fc / e0
    r = ec / (ec - esec)
    x = np.maximum(e, 0.0) / e0
    f = fc * x * r / (r - 1.0 + x**r)
    f2 = fc * 2.0 * r / (r - 1.0 + 2.0**r)
    f = np.where(e > 2.0 * e0, np.maximum(f2 * (e_sp - e) / (e_sp - 2.0 * e0), 0.0), f)
    return np.where(e > 0.0, f, 0.0)


# =========================================================================
# Steel models (compression positive like the concrete, strain in mm/mm)
# =========================================================================

def steel_elastoplastic(strain, fy, es):
    return np.clip(es * np.asarray(strain, dtype=float), -fy, fy)


def steel_mander(strain, fy, es, e_sh=0.01, e_su=0.12, fsu_ratio=1.5, esh_modulus_ratio=None):
    """Mander steel: elastic, yield plateau, strain hardening, rupture at e_su.

    ``esh_modulus_ratio`` is Esh/Es (default 500 ksi / 30000 ksi = 1/60).
    """
    e = np.asarray(strain, dtype=float)
    a = np.abs(e)
    fsu = fsu_ratio * fy
    esh_mod = es / 60.0 if esh_modulus_ratio is None else esh_modulus_ratio * es
    p = esh_mod * (e_su - e_sh) / (fsu - fy)
    ey = fy / es
    f = np.where(a < ey, es * a, fy)
    ratio = np.clip((e_su - a) / (e_su - e_sh), 0.0, None)
    hard = fsu + (fy - fsu) * ratio**p
    f = np.where(a > e_sh, np.where(a < e_su, hard, 0.0), f)
    return np.sign(e) * f


# =========================================================================
# Fibre analysis
# =========================================================================

def _neutral_axis(force_fn, c_min, c_max):
    """Largest sign change of ``force_fn`` on a log grid, refined by bisection."""
    grid = np.geomspace(c_min, c_max, 200)
    values = [force_fn(c) for c in grid]
    for i in range(len(grid) - 1, 0, -1):
        if values[i] * values[i - 1] <= 0.0:
            lo, hi, flo = grid[i - 1], grid[i], values[i - 1]
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                fmid = force_fn(mid)
                if flo * fmid <= 0.0:
                    hi = mid
                else:
                    lo, flo = mid, fmid
            return 0.5 * (lo + hi)
    return None


def moment_curvature(
    width_mm,
    height_mm,
    steel_layers,
    concrete_fn,
    steel_fn,
    strain_max,
    num_steps=120,
    num_fibres=200,
):
    """Moment-curvature curve of a rectangular section under pure bending.

    steel_layers: list of (depth_from_compression_face_mm, area_mm2).
    concrete_fn(strain) -> stress [MPa]; steel_fn(strain) -> stress [MPa];
    both use compression-positive strain, so tension steel has negative values.
    Returns dict with ``phi`` [1/mm], ``moment`` [N*mm], ``ecm``, ``c`` [mm].
    """
    h = height_mm
    dy = h / num_fibres
    y = (np.arange(num_fibres) + 0.5) * dy
    depths = np.array([d for d, _ in steel_layers], dtype=float)
    areas = np.array([a for _, a in steel_layers], dtype=float)

    phi, moment, ecm_out, c_out = [], [], [], []
    for ecm in np.linspace(strain_max / num_steps, strain_max, num_steps):

        def net_force(c):
            fc = np.sum(concrete_fn(ecm * (c - y) / c)) * dy * width_mm
            fs = np.sum(steel_fn(ecm * (c - depths) / c) * areas)
            return fc + fs  # tension steel has negative strain -> negative force

        c = _neutral_axis(net_force, 0.02 * h, 5.0 * h)
        if c is None:
            continue
        eps_c = ecm * (c - y) / c
        arm_c = h / 2.0 - y
        m_conc = np.sum(concrete_fn(eps_c) * arm_c) * dy * width_mm
        eps_s = ecm * (c - depths) / c
        m_steel = np.sum(steel_fn(eps_s) * areas * (h / 2.0 - depths))
        phi.append(ecm / c)
        moment.append(abs(m_conc + m_steel))
        ecm_out.append(ecm)
        c_out.append(c)

    return {
        "phi": np.array(phi),
        "moment": np.array(moment),
        "ecm": np.array(ecm_out),
        "c": np.array(c_out),
    }


def park_beam_layers(rho, rho_prime, width_mm, d_mm):
    """Steel layers of the Fig. 6.6 section: h = 1.1 d, d' = 0.1 d."""
    layers = [(d_mm, rho * width_mm * d_mm)]
    if rho_prime > 0:
        layers.append((0.1 * d_mm, rho_prime * width_mm * d_mm))
    return layers


def park_beam_curve(
    beam,
    concrete="park",
    steel="elastoplastic",
    width_mm=300.0,
    d_mm=500.0,
    strain_max=None,
    fc=PARK_FC,
    fy=PARK_FY,
    es=PARK_ES,
    **kwargs,
):
    """Curve of beam 1-7 of Fig. 6.6. Returns phi*d and M/(b d^2) [MPa]."""
    rho, rho_p = PARK_BEAMS[beam]
    concrete_fn, steel_fn, default_max = material_functions(concrete, steel, fc, fy, es)
    result = moment_curvature(
        width_mm,
        1.1 * d_mm,
        park_beam_layers(rho, rho_p, width_mm, d_mm),
        concrete_fn,
        steel_fn,
        strain_max or default_max,
        **kwargs,
    )
    result["phi_d"] = result["phi"] * d_mm
    result["m_norm"] = result["moment"] / (width_mm * d_mm**2)
    return result


def material_functions(concrete, steel, fc, fy, es):
    """Return (concrete_fn, steel_fn, default_strain_max) for the model names."""
    if concrete == "park":
        concrete_fn = lambda e: concrete_park_paulay(e, fc)
        strain_max = PARK_ECU
    elif concrete == "mander":
        concrete_fn = lambda e: concrete_mander_unconfined(e, fc)
        strain_max = 0.006
    else:
        raise ValueError(f"Unknown concrete model: {concrete}")
    if steel == "elastoplastic":
        steel_fn = lambda e: steel_elastoplastic(e, fy, es)
    elif steel == "mander":
        steel_fn = lambda e: steel_mander(e, fy, es)
    else:
        raise ValueError(f"Unknown steel model: {steel}")
    return concrete_fn, steel_fn, strain_max


def curvature_ductility(phi, moment):
    """Approximate mu = phi_u / phi_y.

    phi_y: secant through 75 % of M_max extrapolated to M_max.
    phi_u: last point of the curve.
    """
    m_max = float(np.max(moment))
    i = int(np.argmax(moment >= 0.75 * m_max))
    phi_y = phi[i] * m_max / moment[i]
    return phi_y, float(phi[-1] / phi_y)
