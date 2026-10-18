# Engineering Tools

Aplicación Streamlit con herramientas de ingeniería estructural y métodos numéricos.

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
pip install -r requirements-dev.txt && pytest      # tests
```

## Estructura

```
streamlit_app.py        # navegación (se arma desde tools_registry.py)
tools_registry.py       # registro único de herramientas: grupo, título, ruta, ícono, referencia
pages/                  # interfaz: una página por herramienta
calculations/           # lógica de cálculo pura (sin Streamlit), reutilizable y testeable
visualization/          # dibujos de secciones
utils/ data/ styles/    # carga de datos AISC, estilos globales
tests/                  # tests de la lógica de cálculo
```

## Herramientas

| Grupo | Herramienta | Referencia |
|---|---|---|
| Numerical Methods | Least Squares, Gaussian Elimination | |
| Steel Sections | Section Properties, Section Classification | AISC Shapes Database; AISC 360-16 Tabla B4.1 |
| Reinforced Concrete | Concrete Stress-Strain, Column Axial Response | |
| Reinforced Concrete | Shrinkage Cracking | Gilbert (1992), ACI Structural Journal 89(2) |
| Reinforced Concrete | Moment-Curvature | Park & Paulay, Fig. 6.6; Mander et al. (1988) |

## Cómo agregar una herramienta

1. Lógica en `calculations/<área>/<herramienta>.py`: funciones puras con unidades documentadas (N, mm, MPa).
2. Interfaz en `pages/<área>/<herramienta>.py`: entradas, resultado y gráfico. Importar y llamar `apply_global_styles()`.
3. Una entrada `Tool(...)` en `tools_registry.py`. El menú y la Home se actualizan solos. Un grupo nuevo se agrega también a `GROUP_ORDER`.
4. Un test en `tests/` que reproduzca un ejemplo de la referencia (valor publicado), de modo que el conocimiento quede verificado.

## Convenciones

- Cálculo y presentación separados: las páginas no contienen fórmulas, solo llaman a `calculations/`.
- Las curvas constitutivas del concreto se reutilizan desde `calculations/structural/concrete_properties.py` cuando existen; los modelos nuevos (Park-Paulay, Mander no confinado) viven en `moment_curvature.py`.
- La teoría extensa vive en Obsidian (`03_Masters_Degree/Cycle_2/Concrete_Design/Tarea 2/`); la app muestra solo planteamiento, ecuaciones y resultado.
