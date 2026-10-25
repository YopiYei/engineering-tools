"""Búsqueda de raíces y de máximos de una variable, sin dependencias externas.

Se usan en el análisis momento-curvatura (COLUMN). Son deliberadamente simples:
bisección y búsqueda de la sección áurea, ambas robustas aunque la función
tenga cambios bruscos de pendiente (fluencia, descascaramiento).
"""

from math import sqrt


def bisect(function, lower, upper, xtol=1e-12, max_iterations=200):
    """Raíz de `function` entre `lower` y `upper` (en cualquier orden) por bisección; exige cambio de signo."""
    f_lower, f_upper = function(lower), function(upper)
    if f_lower == 0.0:
        return lower
    if f_upper == 0.0:
        return upper
    if f_lower * f_upper > 0.0:
        raise ValueError("La función no cambia de signo en el intervalo dado.")
    for _ in range(max_iterations):
        middle = 0.5 * (lower + upper)
        f_middle = function(middle)
        if f_middle == 0.0 or 0.5 * abs(upper - lower) < xtol:
            return middle
        if f_lower * f_middle < 0.0:
            upper = middle
        else:
            lower, f_lower = middle, f_middle
    return 0.5 * (lower + upper)


def maximize_bounded(function, lower, upper, xtol=1e-12, max_iterations=200):
    """Abscisa del máximo de una función unimodal en [lower, upper] (sección áurea)."""
    lower, upper = min(lower, upper), max(lower, upper)
    ratio = (sqrt(5.0) - 1.0) / 2.0
    left, right = upper - ratio * (upper - lower), lower + ratio * (upper - lower)
    f_left, f_right = function(left), function(right)
    for _ in range(max_iterations):
        if upper - lower < xtol:
            break
        if f_left < f_right:
            lower, left, f_left = left, right, f_right
            right = lower + ratio * (upper - lower)
            f_right = function(right)
        else:
            upper, right, f_right = right, left, f_left
            left = upper - ratio * (upper - lower)
            f_left = function(left)
    return 0.5 * (lower + upper)
