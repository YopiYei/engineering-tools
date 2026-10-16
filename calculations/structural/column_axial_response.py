import numpy as np

from calculations.structural.concrete_properties import (
    concrete_collins_mitchell,
    generate_compression_curve_collins_mitchell,
    generate_compression_curve_de_nicolo,
    generate_compression_curve_kent_park,
    generate_compression_curve_desayi_krishnan,
    generate_tension_curve_collins_mitchell,
    generate_tension_curve_restrepo_rodriguez,
    generate_tension_curve_fields_bischoff,
)


# =========================================================================
# Model registries
#
# Reuse the same stress-strain generators as the concrete curves tool, so
# the constitutive models have a single source of truth.
# =========================================================================

COMPRESSION_MODELS = {
    "collins_mitchell": generate_compression_curve_collins_mitchell,
    "de_nicolo": generate_compression_curve_de_nicolo,
    "kent_park": generate_compression_curve_kent_park,
    "desayi_krishnan": generate_compression_curve_desayi_krishnan,
}

TENSION_MODELS = {
    "collins_mitchell": generate_tension_curve_collins_mitchell,
    "restrepo_rodriguez": generate_tension_curve_restrepo_rodriguez,
    "fields_bischoff": generate_tension_curve_fields_bischoff,
}


def rectangular_column_area(width_mm, depth_mm):
    """
    Cross-sectional area of a solid rectangular column.

    Parameters
    ----------
    width_mm : float
        Column width, b (mm).
    depth_mm : float
        Column depth, d (mm).

    Returns
    -------
    float
        Cross-sectional area, A (mm^2).
    """
    if width_mm <= 0 or depth_mm <= 0:
        raise ValueError("Width and depth must be greater than zero.")

    return width_mm * depth_mm


def axial_stiffness(fc_prime, area_mm2, height_mm):
    """
    Initial (tangent-at-origin) axial stiffness of a prismatic concrete
    column, K0 = Ec * A / L.

    Parameters
    ----------
    fc_prime : float
        Concrete compressive strength, f'c (MPa).
    area_mm2 : float
        Cross-sectional area, A (mm^2).
    height_mm : float
        Column height, L (mm).

    Returns
    -------
    float
        Axial stiffness, K0 (N/mm).
    """
    if height_mm <= 0:
        raise ValueError("Height must be greater than zero.")

    Ec = concrete_elastic_modulus(fc_prime)

    return Ec * area_mm2 / height_mm


def concrete_elastic_modulus(fc_prime):
    """
    Concrete modulus of elasticity, Ec = 4700 * sqrt(f'c) (MPa), as used by
    every model in concrete_properties.py.

    Delegates to concrete_collins_mitchell so the formula is defined in a
    single place.
    """
    return concrete_collins_mitchell(fc_prime)["Ec"]


def generate_column_axial_response(
    fc_prime,
    width_mm,
    depth_mm,
    height_mm,
    action,
    model,
    num_points=2000,
):
    """
    Convert a concrete stress-strain curve into an axial force-displacement
    curve for a prismatic column subjected to a uniform normal stress
    (pure compression or pure tension along its axis, no bending).

    The column is assumed to shorten/elongate uniformly over its height,
    so strain and stress are constant across the cross-section at every
    load step:

        displacement (mm) = strain * height
        force        (N)  = stress (MPa) * area (mm^2)

    Parameters
    ----------
    fc_prime : float
        Concrete compressive strength, f'c (MPa).
    width_mm, depth_mm : float
        Cross-sectional dimensions of the column (mm).
    height_mm : float
        Column height, L (mm).
    action : {"compression", "tension"}
        Loading direction.
    model : str
        Constitutive model key. See COMPRESSION_MODELS / TENSION_MODELS.
    num_points : int
        Number of points along the curve.

    Returns
    -------
    dict
        - displacement_mm, force_n, strain, stress_mpa: full curves.
        - area_mm2, fc, height_mm, action, model: echoed inputs.
        - force_peak_n, displacement_at_peak_mm: coordinates of the
          maximum force reached (crushing in compression, cracking in
          tension).
        - displacement_ultimate_mm: displacement at the end of the
          model's defined domain (failure / softening limit).
        - initial_stiffness_n_per_mm: K0 = Ec * A / L.
    """
    if height_mm <= 0:
        raise ValueError("Height must be greater than zero.")

    if action == "compression":
        generators = COMPRESSION_MODELS
    elif action == "tension":
        generators = TENSION_MODELS
    else:
        raise ValueError(f"Unknown action: {action!r}. Use 'compression' or 'tension'.")

    generator = generators.get(model)
    if generator is None:
        raise ValueError(f"Unknown {action} model: {model!r}.")

    area_mm2 = rectangular_column_area(width_mm, depth_mm)

    strain, stress = generator(fc_prime, num_points)

    displacement_mm = strain * height_mm
    force_n = stress * area_mm2

    peak_idx = int(np.argmax(force_n))

    return {
        "displacement_mm": displacement_mm,
        "force_n": force_n,
        "strain": strain,
        "stress_mpa": stress,
        "area_mm2": area_mm2,
        "fc": fc_prime,
        "height_mm": height_mm,
        "action": action,
        "model": model,
        "force_peak_n": float(force_n[peak_idx]),
        "displacement_at_peak_mm": float(displacement_mm[peak_idx]),
        "displacement_ultimate_mm": float(displacement_mm[-1]),
        "initial_stiffness_n_per_mm": axial_stiffness(fc_prime, area_mm2, height_mm),
    }
