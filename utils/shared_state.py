"""Datos compartidos entre paneles de la app (vía `st.session_state`).

Un panel publica lo que el usuario eligió y otro lo lee, sin que el usuario copie valores a mano:
    «Concrete Stress-Strain» publica f'c y los modelos de compresión y tracción.
    «Section Properties» publica el ancho y el alto de un rectángulo con su unidad.
    COLUMN lee ambos y los usa como valores iniciales (editables).

Las funciones reciben el mapa de estado (`state`); por defecto es `st.session_state`. Así se pueden
probar con un diccionario.
"""

CONCRETE_KEY = "shared_concrete"
RECTANGLE_KEY = "shared_rectangle"
UNITS_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4}


def _default_state():
    import streamlit as st

    return st.session_state


def publish_concrete(fc_mpa, compression_model, tension_model, state=None):
    """Publica el f'c y los modelos elegidos en el panel de curvas del concreto."""
    state = _default_state() if state is None else state
    state[CONCRETE_KEY] = {"fc": float(fc_mpa), "compression_model": compression_model, "tension_model": tension_model}


def read_concrete(state=None):
    """Concreto publicado por el panel de curvas, o None si el usuario aún no lo usó."""
    state = _default_state() if state is None else state
    return state.get(CONCRETE_KEY)


def publish_rectangle(width, height, unit, state=None):
    """Publica el rectángulo ingresado en Section Properties (dimensiones en `unit`)."""
    if unit not in UNITS_TO_MM:
        raise ValueError(f"Unidad desconocida: {unit}")
    state = _default_state() if state is None else state
    state[RECTANGLE_KEY] = {"width": float(width), "height": float(height), "unit": unit}


def clear_rectangle(state=None):
    """Quita el rectángulo publicado (cuando Section Properties ya no muestra un rectángulo)."""
    state = _default_state() if state is None else state
    state.pop(RECTANGLE_KEY, None)


def read_rectangle(state=None):
    """Rectángulo publicado: dict con width, height, unit y las mismas dimensiones en mm (width_mm, height_mm)."""
    state = _default_state() if state is None else state
    shared = state.get(RECTANGLE_KEY)
    if shared is None:
        return None
    factor = UNITS_TO_MM[shared["unit"]]
    return {**shared, "width_mm": shared["width"] * factor, "height_mm": shared["height"] * factor}
