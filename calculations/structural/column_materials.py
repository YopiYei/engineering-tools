"""Modelos constitutivos del concreto y del acero usados por el análisis tipo COLUMN.

Fuente: Mander, J. B. (1984), *Seismic Design of Bridge Piers*, Research Report
84-2, Univ. of Canterbury (tesis doctoral). Solo se implementa la rama
monotónica ("skeleton curve"); las ramas de descarga y recarga cíclicas de la
tesis (secciones 2.3 y 3.6.3) no están implementadas.

Convención de signos de ESTE módulo: compresión POSITIVA (la tesis usa tensión
positiva). Unidades: MPa y deformación unitaria (mm/mm).

    Concreto (Cap. 3, Ec. 3.33-3.34):
        compresión  f = f'cc x r / (r - 1 + x^r),  x = e/ecc,
                    r = Ec / (Ec - Esec),  Esec = f'cc / ecc
        tracción    f = Ec e  hasta  f't; luego 0 (fisurado)
        recubrimiento: Popovics hasta 2 eco y recta hasta 0 en e_spall.

    Acero (Cap. 2, Ec. 2.1-2.10):
        elástico, meseta de fluencia, endurecimiento
        f = fsu + (fy - fsu) |(esu - e)/(esu - esh)|^P,  P = Esh (esu - esh)/(fsu - fy)
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PopovicsConcrete:
    """Curva esfuerzo-deformación del concreto (confinado o no confinado).

    peak_stress:     f'cc (núcleo confinado) o f'co (concreto no confinado) [MPa].
    peak_strain:     deformación en el esfuerzo máximo (ecc o eco).
    elastic_modulus: Ec inicial [MPa]; la tesis usa Ec ~ 5000 sqrt(f'co).
    tensile_strength: f't [MPa]; 0 desprecia la tracción.
    spall_strain:    solo para el recubrimiento. A partir de 2 peak_strain la
                     curva desciende linealmente hasta cero en esta deformación.
    """

    peak_stress: float
    peak_strain: float
    elastic_modulus: float
    tensile_strength: float = 0.0
    spall_strain: float | None = None

    def __post_init__(self):
        if self.peak_stress <= 0 or self.peak_strain <= 0:
            raise ValueError("El esfuerzo y la deformación pico del concreto deben ser positivos.")
        if self.elastic_modulus <= self.secant_modulus:
            raise ValueError(
                "Ec debe ser mayor que f'cc/ecc para que la curva de Popovics esté definida "
                f"(Ec = {self.elastic_modulus:.0f} MPa, f'cc/ecc = {self.secant_modulus:.0f} MPa)."
            )
        if self.tensile_strength < 0:
            raise ValueError("La resistencia a tracción no puede ser negativa.")
        if self.spall_strain is not None and self.spall_strain <= 2.0 * self.peak_strain:
            raise ValueError("La deformación de descascaramiento debe ser mayor que 2 eco.")

    @property
    def secant_modulus(self) -> float:
        """Esec = f'cc / ecc, Ec. 3.33d."""
        return self.peak_stress / self.peak_strain

    @property
    def power_r(self) -> float:
        """Parámetro r = Ec / (Ec - Esec), Ec. 3.33c."""
        return self.elastic_modulus / (self.elastic_modulus - self.secant_modulus)

    @property
    def cracking_strain(self) -> float:
        """Deformación de tracción a la que se alcanza f't."""
        return self.tensile_strength / self.elastic_modulus

    def _popovics(self, strain):
        """Esfuerzo y módulo tangente de Popovics para deformaciones de compresión (>= 0)."""
        r = self.power_r
        x = strain / self.peak_strain
        denominator = r - 1.0 + x**r
        stress = self.peak_stress * x * r / denominator
        # derivada: df/dx = f'cc r (r-1)(1 - x^r) / (r - 1 + x^r)^2
        tangent = self.peak_stress * r * (r - 1.0) * (1.0 - x**r) / denominator**2 / self.peak_strain
        return stress, tangent

    def stress_and_tangent(self, strain, cracked=None, spalled=None):
        """Esfuerzo [MPa] y módulo tangente [MPa] para un arreglo de deformaciones.

        cracked: máscara booleana (misma forma que strain) de fibras que ya
                 superaron f't en tracción: no resisten tracción (hipótesis 4,
                 sección 7.4.2).
        spalled: máscara de fibras de recubrimiento que ya superaron e_spall:
                 esfuerzo cero (hipótesis 3, sección 7.4.2).
        """
        strain = np.asarray(strain, dtype=float)
        compression = np.maximum(strain, 0.0)
        stress, tangent = self._popovics(compression)

        if self.spall_strain is not None:
            # Recubrimiento: tramo descendente lineal a partir de 2 eco
            e2 = 2.0 * self.peak_strain
            f2, _ = self._popovics(np.array(e2))
            slope = -float(f2) / (self.spall_strain - e2)
            descending = compression > e2
            linear = np.maximum(float(f2) + slope * (compression - e2), 0.0)
            stress = np.where(descending, linear, stress)
            tangent = np.where(descending & (linear > 0.0), slope, np.where(descending, 0.0, tangent))
            if spalled is not None:
                stress = np.where(spalled, 0.0, stress)
                tangent = np.where(spalled, 0.0, tangent)

        # Tracción: lineal hasta f't y luego fisurado
        in_tension = strain < 0.0
        uncracked = in_tension & (-strain <= self.cracking_strain)
        if cracked is not None:
            uncracked = uncracked & ~cracked
        stress = np.where(in_tension, np.where(uncracked, self.elastic_modulus * strain, 0.0), stress)
        tangent = np.where(in_tension, np.where(uncracked, self.elastic_modulus, 0.0), tangent)
        return stress, tangent


@dataclass(frozen=True)
class ReinforcingSteel:
    """Curva monotónica del acero (Ec. 2.1-2.10), igual en tracción y compresión.

    yield_stress (fy), ultimate_stress (fsu) [MPa]; elastic_modulus (Es) y
    hardening_modulus (Esh) [MPa]; hardening_strain (esh) y ultimate_strain
    (esu) son deformaciones unitarias. esu es la deformación en fsu (no la de
    fractura, que ocurre después a menor esfuerzo).
    """

    yield_stress: float
    ultimate_stress: float
    elastic_modulus: float
    hardening_strain: float
    ultimate_strain: float
    hardening_modulus: float

    def __post_init__(self):
        if self.yield_stress <= 0 or self.elastic_modulus <= 0:
            raise ValueError("fy y Es del acero deben ser positivos.")
        if self.ultimate_stress <= self.yield_stress:
            raise ValueError("fsu debe ser mayor que fy.")
        if not self.yield_strain < self.hardening_strain < self.ultimate_strain:
            raise ValueError("Se requiere ey < esh < esu en el acero.")
        if self.hardening_modulus <= 0:
            raise ValueError("Esh debe ser positivo.")

    @property
    def yield_strain(self) -> float:
        return self.yield_stress / self.elastic_modulus

    @property
    def power_p(self) -> float:
        """Potencia de endurecimiento P = Esh (esu - esh) / (fsu - fy), Ec. 2.9b."""
        return (
            self.hardening_modulus
            * (self.ultimate_strain - self.hardening_strain)
            / (self.ultimate_stress - self.yield_stress)
        )

    def stress_and_tangent(self, strain):
        """Esfuerzo [MPa] y módulo tangente [MPa]; función impar de la deformación."""
        strain = np.asarray(strain, dtype=float)
        magnitude = np.abs(strain)
        p = self.power_p

        ratio = np.clip(
            (self.ultimate_strain - magnitude) / (self.ultimate_strain - self.hardening_strain), 0.0, 1.0
        )
        hardening_stress = self.ultimate_stress + (self.yield_stress - self.ultimate_stress) * ratio**p  # Ec. 2.6
        # Ec. 2.8: dfs/de = P (fsu - fy)/(esu - esh) |.|^(P-1)
        hardening_tangent = (
            p * (self.ultimate_stress - self.yield_stress) / (self.ultimate_strain - self.hardening_strain)
        ) * ratio ** (p - 1.0)

        stress = np.where(
            magnitude <= self.yield_strain,
            self.elastic_modulus * magnitude,  # Ec. 2.1
            np.where(magnitude <= self.hardening_strain, self.yield_stress, hardening_stress),  # Ec. 2.3 y 2.6
        )
        tangent = np.where(
            magnitude <= self.yield_strain,
            self.elastic_modulus,  # Ec. 2.2
            np.where(
                magnitude <= self.hardening_strain,
                0.0,  # Ec. 2.4
                np.where(magnitude < self.ultimate_strain, hardening_tangent, 0.0),
            ),
        )
        return np.sign(strain) * stress, tangent
