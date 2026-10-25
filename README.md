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
| Reinforced Concrete | COLUMN — Análisis Momento–Curvatura | Mander (1984), Research Report 84-2, §7.4.2; Soesianawati (1986) |

## Cómo agregar una herramienta

1. Lógica en `calculations/<área>/<herramienta>.py`: funciones puras con unidades documentadas (N, mm, MPa).
2. Interfaz en `pages/<área>/<herramienta>.py`: entradas, resultado y gráfico. Importar y llamar `apply_global_styles()`.
3. Una entrada `Tool(...)` en `tools_registry.py`. El menú y la Home se actualizan solos. Un grupo nuevo se agrega también a `GROUP_ORDER`.
4. Un test en `tests/` que reproduzca un ejemplo de la referencia (valor publicado), de modo que el conocimiento quede verificado.

## Convenciones

- Cálculo y presentación separados: las páginas no contienen fórmulas, solo llaman a `calculations/`.
- Las curvas constitutivas del concreto se reutilizan desde `calculations/structural/concrete_properties.py` cuando existen; los modelos nuevos (Park-Paulay, Mander no confinado) viven en `moment_curvature.py`.
- La teoría extensa vive en Obsidian (`03_Masters_Degree/Cycle_2/Concrete_Design/Tarea 2/`); la app muestra solo planteamiento, ecuaciones y resultado.

## COLUMN — Análisis Momento–Curvatura

Análisis por fibras, monotónico, de secciones rectangulares de concreto armado con concreto confinado
(Mander), acero con endurecimiento y carga axial constante. Es una **reconstrucción** del procedimiento de
COLUMN descrito en la tesis de Mander (1984), no una traducción de su FORTRAN, y **no incluye la parte cíclica**.

```
calculations/structural/column_materials.py      concreto (Popovics) y acero (Ec. 2.1-2.10)
calculations/structural/column_section.py        geometría, barras, estribos y láminas
calculations/structural/column_confinement.py    ke, f'cc (procedimiento de 6 pasos), ecc
calculations/structural/column_analysis.py       algoritmo de §7.4.2, Mi, phi_y, estados límite
calculations/structural/column_examples.py       Unidades 1-4 de Soesianawati (ejemplo y validación)
calculations/structural/column_interaction.py    diagrama P-M (ideal y del análisis) y ductilidad vs. carga axial
calculations/structural/column_variants.py       comparación de configuraciones (estribos, P, f'co)
calculations/structural/column_response.py       de la sección a la columna: rótula plástica, V-Δ y ductilidad de desplazamiento
calculations/structural/concrete_link.py         concreto tomado del panel «Concrete Stress-Strain»
calculations/structural/root_finding.py          bisección y búsqueda áurea (sin SciPy)
utils/shared_state.py                            datos compartidos entre paneles (st.session_state)
utils/column_report.py                           informe en Word (python-docx) y PDF (fpdf2)
visualization/column_plots.py                    gráficos de Plotly (incluye el corte acotado de la sección)
visualization/column_report_plots.py             figuras PNG (matplotlib) para el informe
pages/structural/column_mphi.py                  interfaz
tests/test_column_*.py                           pruebas y validación con datos publicados
```

El panel incluye, además del análisis principal: la tabla editable de capas de barras con el corte de la sección,
el **diagrama de interacción P–M** y la **ductilidad vs. carga axial**, la **comparación de configuraciones** y la
descarga de un **informe Word/PDF**. Las librerías `python-docx` y `fpdf2` solo se necesitan para el informe.

**De la sección a la columna** (sección 10 del panel): con la altura L de la columna en voladizo equivalente y la longitud de
penetración de la fluencia, calcula la longitud de rótula plástica `Lp = Lpy + 0.06 L`, la rotación plástica, la curva fuerza
lateral–desplazamiento (flexión elástica + cortante opcional + plástico) y la ductilidad de desplazamiento, siguiendo Mander (1984),
§7.4.3 y §7.4.4. Incluye el criterio de curvatura última de diseño de Mander (Ec. 7.70).

**Conexión entre paneles:** «Concrete Stress-Strain» (vista *Single fc' value*) publica el f'c y los modelos elegidos, y «Section
Properties» (Custom Dimensions → Rectangle) publica b y h con su unidad. COLUMN puede tomarlos como valores iniciales (editables); del
panel de concreto toma f'c, la deformación del pico, Ec y f't de las curvas, y superpone la curva del panel a la de COLUMN.

Convenciones: compresión positiva; N, mm, MPa; curvatura en µrad/mm en la interfaz (1/mm internamente).
Las pruebas comparan con Soesianawati (1986), Tablas 5.1 y 8.3, con el ejemplo numérico de Mander (p. 65)
y con el módulo `moment_curvature.py` de esta app (verificación cruzada con P = 0).

