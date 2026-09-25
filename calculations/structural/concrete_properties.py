import math
import numpy as np


# =========================================================================
# Compression and Tension Models - Parameters Calculation
# =========================================================================


def concrete_collins_mitchell(fc_prime):
    """
    Calculate concrete stress-strain parameters using Collins & Mitchell (1991) model.

    For normal weight concrete with fc' in MPa.

    Parameters
    ----------
    fc_prime : float
        Concrete compressive strength in MPa (f'c).

    Returns
    -------
    dict
        Dictionary containing:
        - Ec: Modulus of elasticity (MPa)
        - n: Strain exponent parameter
        - strain_c: Peak compressive strain
        - ft: Tensile strength (MPa)
        - fc: Concrete compressive strength (MPa)
    """
    if fc_prime <= 0:
        raise ValueError("Concrete strength fc' must be greater than zero.")

    # Calculate modulus of elasticity (Ec)
    # Formula: Ec = 4700 * sqrt(f'c)  [ACI 318]
    Ec = 4700 * math.sqrt(fc_prime)

    # Calculate strain exponent (n)
    # Formula: n = 0.8 + f'c / 17
    n = 0.8 + fc_prime / 17

    # Calculate peak compressive strain (ε'c)
    # Formula: ε'c = (f'c / Ec) * n / (n - 1)
    strain_peak = (fc_prime / Ec) * n / (n - 1)

    # Calculate tensile strength (ft) - Simplified ACI 318
    # Formula: ft = 0.1 * sqrt(f'c)
    ft = 0.1 * math.sqrt(fc_prime)

    return {
        "Ec": Ec,
        "n": n,
        "strain_c": strain_peak,
        "ft": ft,
        "fc": fc_prime,
    }


def concrete_de_nicolo_pani_pozzo(fc_prime):
    """
    Calculate concrete stress-strain parameters using De Nicolo, Pani and Pozzo (1994) model.

    For normal weight concrete with 10 MPa ≤ f'c ≤ 100 MPa.

    Parameters
    ----------
    fc_prime : float
        Concrete compressive strength in MPa (f'c).

    Returns
    -------
    dict
        Dictionary containing:
        - Ec: Modulus of elasticity (MPa)
        - strain_c: Peak compressive strain
        - ft: Tensile strength (MPa)
        - fc: Concrete compressive strength (MPa)
    """
    if fc_prime < 10 or fc_prime > 100:
        raise ValueError(
            "De Nicolo, Pani & Pozzo model is valid for 10 ≤ f'c ≤ 100 MPa."
        )

    # Calculate peak compressive strain (ε'c)
    # Formula: ε'c = 0.00076 + sqrt[(0.626*f'c - 4.33) × 10^-7]
    term = (0.626 * fc_prime - 4.33) * 1e-7
    if term < 0:
        raise ValueError(
            "Invalid parameters for De Nicolo model. f'c might be too low."
        )
    strain_peak = 0.00076 + math.sqrt(term)

    # Calculate modulus of elasticity (Ec)
    Ec = 4700 * math.sqrt(fc_prime)

    # Calculate tensile strength (ft)
    ft = 0.1 * math.sqrt(fc_prime)

    return {
        "Ec": Ec,
        "strain_c": strain_peak,
        "ft": ft,
        "fc": fc_prime,
    }


def concrete_kent_park(fc_prime):
    """
    Calculate concrete stress-strain parameters using Kent & Park (1971) model.

    For unconfined concrete in compression.

    Parameters
    ----------
    fc_prime : float
        Concrete compressive strength in MPa (f'c).

    Returns
    -------
    dict
        Dictionary containing:
        - Ec: Modulus of elasticity (MPa)
        - strain_0: Initial strain parameter (ε0 = 0.002)
        - strain_50u: Strain at 50% of peak stress after peak (ε50u)
        - Z: Descending branch parameter
        - fc: Concrete compressive strength (MPa)
    """
    if fc_prime <= 0:
        raise ValueError("Concrete strength f'c must be greater than zero.")

    # Calculate modulus of elasticity (Ec)
    Ec = 4700 * math.sqrt(fc_prime)

    # Initial strain (at which peak stress is reached)
    # ε0 = 0.002
    strain_0 = 0.002

    # Convert f'c from MPa to psi for Kent & Park formula
    # 1 MPa = 145.038 psi
    fc_psi = fc_prime * 145.038

    # Calculate strain at 50% of peak stress in descending branch
    # Formula: ε50u = (3 + 0.29*f'c) / (145*f'c - 1000)  [f'c in psi]
    numerator = 3 + 0.29 * fc_psi
    denominator = 145 * fc_psi - 1000
    if denominator <= 0:
        raise ValueError("Invalid concrete strength for Kent & Park model.")
    strain_50u = numerator / denominator

    # Calculate Z parameter for descending branch
    # Formula: Z = 0.5 / (ε50u - ε0)
    Z = 0.5 / (strain_50u - strain_0)

    return {
        "Ec": Ec,
        "strain_0": strain_0,
        "strain_50u": strain_50u,
        "Z": Z,
        "fc": fc_prime,
    }


def concrete_desayi_krishnan(fc_prime):
    """
    Calculate concrete stress-strain parameters using Desayi & Krishnan (1964) model.

    For unconfined concrete in compression.

    Parameters
    ----------
    fc_prime : float
        Concrete compressive strength in MPa (f'c).

    Returns
    -------
    dict
        Dictionary containing:
        - Ec: Modulus of elasticity (MPa)
        - strain_0: Peak strain parameter (ε0)
        - fc: Concrete compressive strength (MPa)
    """
    if fc_prime <= 0:
        raise ValueError("Concrete strength f'c must be greater than zero.")

    # Calculate modulus of elasticity (Ec)
    Ec = 4700 * math.sqrt(fc_prime)

    # Calculate peak strain (ε0)
    # Approximation: ε0 = 2*f'c / Ec
    strain_0 = 2 * fc_prime / Ec

    return {
        "Ec": Ec,
        "strain_0": strain_0,
        "fc": fc_prime,
    }


# =========================================================================
# Compression Curve Generators
# =========================================================================


def generate_compression_curve_collins_mitchell(fc_prime, num_points=100):
    """
    Generate compression curve using Collins & Mitchell (1991) model.

    Stress-strain relationship:
    f_c = (n * (ε/ε'c)) / (n - 1 + (ε/ε'c)^n) * f'c

    Parameters
    ----------
    fc_prime : float
        Concrete compressive strength in MPa (f'c).
    num_points : int
        Number of points in the curve.

    Returns
    -------
    tuple
        (strains, stresses) arrays
    """
    params = concrete_collins_mitchell(fc_prime)
    strain_peak = params["strain_c"]
    n = params["n"]
    fc = params["fc"]

    strains = np.linspace(0, strain_peak * 1.2, num_points)
    stresses = np.zeros_like(strains)

    for i, strain in enumerate(strains):
        strain_ratio = strain / strain_peak
        if strain_ratio > 0:
            numerator = n * strain_ratio
            denominator = n - 1 + strain_ratio**n
            stress = (numerator / denominator) * fc
        else:
            stress = 0
        stresses[i] = max(0, min(stress, fc))

    return strains, stresses


def generate_compression_curve_de_nicolo(fc_prime, num_points=100):
    """
    Generate compression curve using De Nicolo, Pani y Pozzo (1994) model.

    Parabolic ascending and linear descending branches.

    Parameters
    ----------
    fc_prime : float
        Concrete compressive strength in MPa (f'c).
    num_points : int
        Number of points in the curve.

    Returns
    -------
    tuple
        (strains, stresses) arrays
    """
    params = concrete_de_nicolo_pani_pozzo(fc_prime)
    strain_peak = params["strain_c"]
    fc = params["fc"]

    strains = np.linspace(0, strain_peak * 1.2, num_points)
    stresses = np.zeros_like(strains)

    for i, strain in enumerate(strains):
        if strain <= strain_peak:
            strain_ratio = strain / strain_peak
            # Parabolic relationship: f_c = f'c * (2*ε/ε'c - (ε/ε'c)^2)
            stress = fc * (2 * strain_ratio - strain_ratio**2)
        else:
            # Linear descending
            stress = fc * (1 - (strain - strain_peak) / (strain_peak * 0.5))
            stress = max(0, stress)
        stresses[i] = max(0, stress)

    return strains, stresses


def generate_compression_curve_kent_park(fc_prime, num_points=100):
    """
    Generate compression curve using Kent & Park (1971) model.

    Ascending branch: f_c = f'c * [2*(ε/ε0) - (ε/ε0)^2]
    Descending branch: f_c = f'c * [1 - Z*(ε - ε0)]

    Parameters
    ----------
    fc_prime : float
        Concrete compressive strength in MPa (f'c).
    num_points : int
        Number of points in the curve.

    Returns
    -------
    tuple
        (strains, stresses) arrays
    """
    params = concrete_kent_park(fc_prime)
    strain_0 = params["strain_0"]
    strain_50u = params["strain_50u"]
    Z = params["Z"]
    fc = params["fc"]

    # Generate strain range up to descending branch end
    max_strain = strain_50u * 1.5
    strains = np.linspace(0, max_strain, num_points)
    stresses = np.zeros_like(strains)

    for i, strain in enumerate(strains):
        strain_ratio = strain / strain_0

        if strain <= strain_0:
            # Ascending branch: f_c = f'c * [2*(ε/ε0) - (ε/ε0)^2]
            stress = fc * (2 * strain_ratio - strain_ratio**2)
        else:
            # Descending branch: f_c = f'c * [1 - Z*(ε - ε0)]
            stress = fc * (1 - Z * (strain - strain_0))
            stress = max(0, stress)

        stresses[i] = max(0, min(stress, fc))

    return strains, stresses


def generate_compression_curve_desayi_krishnan(fc_prime, num_points=100):
    """
    Generate compression curve using Desayi & Krishnan (1964) model.

    Stress-strain relationship:
    f_c = Ec * ε / (1 + (ε/ε0)^2)

    Parameters
    ----------
    fc_prime : float
        Concrete compressive strength in MPa (f'c).
    num_points : int
        Number of points in the curve.

    Returns
    -------
    tuple
        (strains, stresses) arrays
    """
    params = concrete_desayi_krishnan(fc_prime)
    strain_0 = params["strain_0"]
    Ec = params["Ec"]
    fc = params["fc"]

    max_strain = strain_0 * 1.5
    strains = np.linspace(0, max_strain, num_points)
    stresses = np.zeros_like(strains)

    for i, strain in enumerate(strains):
        if strain > 0:
            strain_ratio = strain / strain_0
            # f_c = Ec * ε / (1 + (ε/ε0)^2)
            denominator = 1 + strain_ratio**2
            stress = Ec * strain / denominator
        else:
            stress = 0

        stresses[i] = max(0, min(stress, fc))

    return strains, stresses


# =========================================================================
# Tension Curve Generators
# =========================================================================


def generate_tension_curve_collins_mitchell(fc_prime, num_points=100):
    """
    Generate tension curve - Linear elastic until failure.

    Formula: f_t = 0.1 * sqrt(f'c)

    Parameters
    ----------
    fc_prime : float
        Concrete compressive strength in MPa (f'c).
    num_points : int
        Number of points in the curve.

    Returns
    -------
    tuple
        (strains, stresses) arrays
    """
    ft = 0.1 * math.sqrt(fc_prime)
    Ec = 4700 * math.sqrt(fc_prime)
    strain_failure = ft / Ec

    strains = np.linspace(0, strain_failure * 1.2, num_points)
    stresses = np.zeros_like(strains)

    for i, strain in enumerate(strains):
        if strain <= strain_failure:
            stresses[i] = Ec * strain
        else:
            stresses[i] = 0

    return strains, stresses


def generate_tension_curve_restrepo_rodriguez(fc_prime, num_points=100):
    """
    Generate tension curve using Restrepo-Rodríguez (2019) model.

    Elastic branch (ε_a ≤ ε_cr):
        f_ct(ε_a) = (f_cr / ε_cr) * ε_a

    Softening branch (ε_cr ≤ ε_a ≤ ε_cto):
        f_ct2(ε_a) = β * [1 - (ε_ct - ε_cr) / ε_cto] * f_a

    Parameters
    ----------
    fc_prime : float
        Concrete compressive strength in MPa (f'c).
    num_points : int
        Number of points in the curve.

    Returns
    -------
    tuple
        (strains, stresses) arrays
    """
    # Calculate tensile strength at cracking
    f_cr = 0.1 * math.sqrt(fc_prime)  # Cracking stress
    Ec = 4700 * math.sqrt(fc_prime)
    strain_cr = f_cr / Ec  # Cracking strain (ε_cr)

    # Ultimate strain in tension (ε_cto)
    # ε_cto = 2500 microstrain = 2500e-6
    strain_cto = 2500e-6

    # Softening parameter
    beta = 0.9
    f_a = 0  # Residual stress

    strains = np.linspace(0, strain_cto * 1.2, num_points)
    stresses = np.zeros_like(strains)

    for i, strain in enumerate(strains):
        if strain <= strain_cr:
            # Elastic branch: f_ct = (f_cr / ε_cr) * ε_a
            stresses[i] = (f_cr / strain_cr) * strain
        elif strain <= strain_cto:
            # Softening branch: f_ct2 = β * [1 - (ε_a - ε_cr) / ε_cto] * f_cr
            factor = 1 - (strain - strain_cr) / (strain_cto - strain_cr)
            stresses[i] = beta * factor * f_cr
        else:
            stresses[i] = 0

    return strains, stresses


def generate_tension_curve_fields_bischoff(fc_prime, num_points=100):
    """
    Generate tension curve using Fields & Bischoff (2004) model.

    Elastic branch (ε ≤ ε_cr):
        f_c = Ec * ε

    Exponential softening branch (ε > ε_cr):
        f_c / f_cr = exp[-0.8 * (ε - ε_cr) * 10³]

    Parameters
    ----------
    fc_prime : float
        Concrete compressive strength in MPa (f'c).
    num_points : int
        Number of points in the curve.

    Returns
    -------
    tuple
        (strains, stresses) arrays
    """
    f_t = 0.1 * math.sqrt(fc_prime)
    Ec = 4700 * math.sqrt(fc_prime)
    strain_cr = f_t / Ec  # Cracking strain

    strains = np.linspace(0, strain_cr * 3, num_points)
    stresses = np.zeros_like(strains)

    for i, strain in enumerate(strains):
        if strain <= strain_cr:
            # Elastic branch: f_c = Ec * ε
            stresses[i] = Ec * strain
        else:
            # Exponential softening: f_c / f_cr = exp[-0.8 * (ε - ε_cr) * 10³]
            exponent = -0.8 * (strain - strain_cr) * 1000
            stresses[i] = f_t * math.exp(exponent)

    return strains, stresses


# =========================================================================
# Combined Generation Function
# =========================================================================


def generate_curves(fc_prime, compression_model="collins_mitchell",
                    tension_model="collins_mitchell", num_points=100):
    """
    Generate both compression and tension curves.

    Parameters
    ----------
    fc_prime : float
        Concrete compressive strength in MPa (f'c).
    compression_model : str
        Model for compression:
        - "collins_mitchell": Collins & Mitchell (1991)
        - "de_nicolo": De Nicolo, Pani y Pozzo (1994)
        - "kent_park": Kent & Park (1971)
        - "desayi_krishnan": Desayi & Krishnan (1964)
    tension_model : str
        Model for tension:
        - "collins_mitchell": Linear elastic
        - "restrepo_rodriguez": Restrepo-Rodríguez (2019)
        - "fields_bischoff": Fields & Bischoff (2004)
    num_points : int
        Number of points in curves.

    Returns
    -------
    dict
        Dictionary with compression and tension curves and parameters.
    """

    # Generate compression curve
    if compression_model == "collins_mitchell":
        strain_comp, stress_comp = generate_compression_curve_collins_mitchell(
            fc_prime, num_points
        )
    elif compression_model == "de_nicolo":
        strain_comp, stress_comp = generate_compression_curve_de_nicolo(
            fc_prime, num_points
        )
    elif compression_model == "kent_park":
        strain_comp, stress_comp = generate_compression_curve_kent_park(
            fc_prime, num_points
        )
    elif compression_model == "desayi_krishnan":
        strain_comp, stress_comp = generate_compression_curve_desayi_krishnan(
            fc_prime, num_points
        )
    else:
        raise ValueError(f"Unknown compression model: {compression_model}")

    # Generate tension curve
    if tension_model == "collins_mitchell":
        strain_tens, stress_tens = generate_tension_curve_collins_mitchell(
            fc_prime, num_points
        )
    elif tension_model == "restrepo_rodriguez":
        strain_tens, stress_tens = generate_tension_curve_restrepo_rodriguez(
            fc_prime, num_points
        )
    elif tension_model == "fields_bischoff":
        strain_tens, stress_tens = generate_tension_curve_fields_bischoff(
            fc_prime, num_points
        )
    else:
        raise ValueError(f"Unknown tension model: {tension_model}")

    return {
        "strain_comp": strain_comp,
        "stress_comp": stress_comp,
        "strain_tens": strain_tens,
        "stress_tens": stress_tens,
        "fc": fc_prime,
    }


# =========================================================================
# Backward Compatibility Functions
# =========================================================================


def generate_compression_curve(fc, model="collins_mitchell", num_points=100):
    """Generate compression curve (backward compatibility)."""
    if model == "collins_mitchell":
        return generate_compression_curve_collins_mitchell(fc, num_points)
    elif model == "de_nicolo_pani_pozzo":
        return generate_compression_curve_de_nicolo(fc, num_points)
    else:
        raise ValueError(f"Unknown model: {model}")


def generate_tension_curve(ft, num_points=100):
    """Generate simple linear tension curve (backward compatibility)."""
    fc_estimate = 100 * ft**2
    Ec = 4700 * math.sqrt(max(1, fc_estimate))
    strain_failure = ft / Ec

    strains = np.linspace(0, strain_failure * 1.2, num_points)
    stresses = np.zeros_like(strains)

    for i, strain in enumerate(strains):
        if strain <= strain_failure:
            stresses[i] = Ec * strain
        else:
            stresses[i] = 0

    return strains, stresses
