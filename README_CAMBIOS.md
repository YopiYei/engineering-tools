# Cambios y actualizaciones de Engineering Tools

Registro de lo que se agregó o cambió en la app Streamlit, cómo se validó y qué falta. Última actualización: 25/10/2026.
El `README.md` describe la estructura general del proyecto; este archivo describe la evolución reciente.

## Resumen

- **Panel nuevo `COLUMN — Análisis Momento–Curvatura`** (Reinforced Concrete): análisis momento–curvatura monotónico de secciones rectangulares con concreto confinado, carga axial, diagrama de interacción, comparación de configuraciones, respuesta de la columna (fuerza–desplazamiento) e informe en Word y PDF.
- **Introducción breve** (qué calcula, para qué sirve, qué resultados da) al inicio de los 8 paneles existentes.
- **Conexión entre paneles:** `Concrete Stress-Strain` y `Section Properties` alimentan a COLUMN.
- 165 pruebas automatizadas (eran 30).

COLUMN es una **reconstrucción** del programa de Mander (1984), *Seismic Design of Bridge Piers*, Research Report 84-2, Univ. of Canterbury, y **no es una traducción** de su FORTRAN. **Solo cubre carga monotónica**; la parte cíclica no está implementada.

## 1. Panel COLUMN

### Datos de entrada
- **Origen de los datos:** columnas de ensayo de Soesianawati (1986, Unidades 1 a 4) o datos propios. Los ejemplos se pueden editar.
- **Geometría:** ancho y peralte, recubrimiento libre hasta el estribo. Se pueden tomar de `Section Properties`.
- **Concreto no confinado:** f'co, εco, εspall, f't y Ec. Se pueden tomar de `Concrete Stress-Strain`.
- **Acero longitudinal:** curva con meseta y endurecimiento (fy, fsu, Es, εsh, εsu, Esh).
- **Barras longitudinales:** **tabla editable de capas** (profundidad desde la cara comprimida, número de barras, diámetro). Admite diámetros distintos y armado asimétrico. La tabla por defecto se adapta a la geometría actual.
- **Estribos:** diámetro, espaciamiento, fyh, configuración (cuadrado + octagonal superpuestos, o estribo y ganchos por número de ramas) y barras apoyadas.
- **Corte de la sección** (a escala, acotado): concreto, núcleo confinado, estribos, barras y cotas.
- **Carga axial P** y **criterios de terminación:** fractura del estribo por balance de energía, εcu del núcleo, ductilidad máxima, caída del momento y curvatura última de diseño de Mander (§7.4.3, Ec. 7.70). Siempre rige εsu del acero.

### Análisis y resultados
- **Algoritmo:** curvatura impuesta y deformación axial corregida por Newton para cumplir N = P (Mander §7.4.2, "Algorithm 2"), con respaldo por búsqueda en intervalo.
- **Confinamiento:** coeficiente ke y resistencia confinada f'cc por el procedimiento de 6 pasos de la superficie de resistencia (§3.4.5). εcc = εco[1 + 5(f'cc/f'co − 1)].
- **Resultados:** curva M–φ con estados límite y su criterio, Mmax, Mi (método de la norma), φ'y, My, φy (Ec. 7.60), eje neutro, deformaciones, perfil de deformaciones, convergencia por incremento y tabla con CSV.
- **Estados límite:** fisuración, primera fluencia, deformación 0.003, recubrimiento en su pico y descascarado, núcleo en su pico, endurecimiento, momento máximo y los de terminación. La curvatura última siempre se informa junto al criterio que terminó el análisis.
- **Interpretación** automática de los resultados.

### Ampliaciones
- **Diagrama de interacción P–M y ductilidad vs. carga axial:** curva ideal (tracción pura, punto balanceado, compresión pura) y Mmax del análisis completo para un barrido de cargas. La ductilidad se colorea según el criterio que termina el análisis.
- **Comparación de configuraciones:** variantes de espaciamiento, diámetro y fyh de los estribos, carga axial y f'co. Una celda vacía usa el valor de la base. Al cambiar el diámetro del estribo, las capas de barras se desplazan para conservar su distancia al estribo.
- **De la sección a la columna** (Mander §7.4.3 y §7.4.4): con la altura L y la penetración de fluencia calcula la longitud de rótula Lp = Lpy + 0.06 L (Ec. 7.71–7.72), la rotación plástica (Ec. 7.67 y 7.87), la curva fuerza lateral–desplazamiento Δ = Δe + Δs + Δp (Ec. 7.74) con cortante opcional, y la ductilidad de desplazamiento μΔ = Δ/Δy.
- **Informe en Word (.docx) y PDF** con datos, corte de la sección, resultados, estados límite, interpretación y, si se calcularon, columna, interacción y comparación.

## 2. Mejoras en los paneles existentes

- **Introducción** de 3 líneas en: Gaussian Elimination, Least Squares, Section Properties, Section Classification, Concrete Stress-Strain, Column Axial Response, Shrinkage Cracking y Moment-Curvature. Se quitaron tres párrafos que ahora se repetían. No cambió ningún cálculo.
- **Componente reutilizable** `panel_intro()` en `styles/main.py`.

## 3. Conexión entre paneles

Los datos pasan por `st.session_state` (`utils/shared_state.py`):

| Panel de origen | Publica | COLUMN lo usa como |
|---|---|---|
| Concrete Stress-Strain (vista *Single fc' value*) | f'c y modelos de compresión y tracción | f'co; εco, Ec y f't tomados de las curvas del modelo; la curva del panel se superpone a la de COLUMN |
| Section Properties (Custom Dimensions → Rectangle) | b y h con su unidad (selector nuevo: mm, cm, m, in; por defecto cm) | ancho y peralte, convertidos a mm |

Lo tomado es el valor inicial y se puede editar. Si el otro panel aún no publicó nada, COLUMN lo avisa y usa los valores manuales.

## 4. Cambios técnicos

**Módulos nuevos**

| Archivo | Contenido |
|---|---|
| `calculations/structural/column_materials.py` | concreto (Popovics) y acero (Mander Cap. 2) |
| `calculations/structural/column_section.py` | capas de barras, estribos y mallado en láminas |
| `calculations/structural/column_confinement.py` | ke, f'cc y εcc |
| `calculations/structural/column_analysis.py` | algoritmo §7.4.2, Mi, φy y estados límite |
| `calculations/structural/column_interaction.py` | interacción P–M y ductilidad vs. carga axial |
| `calculations/structural/column_variants.py` | comparación de configuraciones |
| `calculations/structural/column_response.py` | rótula plástica y curva fuerza–desplazamiento |
| `calculations/structural/column_examples.py` | Unidades 1 a 4 de Soesianawati (ejemplo y validación) |
| `calculations/structural/concrete_link.py` | concreto tomado del panel de curvas |
| `calculations/structural/root_finding.py` | bisección y búsqueda áurea |
| `utils/column_report.py` | informe en Word y PDF |
| `utils/shared_state.py` | datos compartidos entre paneles |
| `visualization/column_plots.py`, `column_report_plots.py` | gráficos interactivos y figuras del informe |
| `pages/structural/column_mphi.py` | interfaz |

**Dependencias**
- Se agregaron `python-docx` y `fpdf2>=2.7.6` (solo para el informe).
- SciPy ya no lo usa ningún módulo nuevo. Se quitó porque fallaba en entornos con SciPy y NumPy desparejados (`numpy._core.multiarray failed to import`). Sigue listado en `requirements.txt` por si otra parte lo necesita.

**Otros**
- `ideal_flexural_strength` se refactorizó en `code_forces` (reutilizada por la interacción P–M).
- El mallado reparte el área de cada barra entre las láminas que cubre su diámetro.

## 5. Validación

Todo está en `tests/test_column_*.py` (165 pruebas en total). Comparaciones con datos publicados:

| Verificación | Resultado |
|---|---|
| Mmax contra Soesianawati, Tabla 8.3 (Unidades 1 a 4) | −0.5 % a −0.8 % |
| Mi contra Tabla 8.3 (con f'c de ensayo) | −1.0 % |
| f'cc contra Tabla 5.1 | −0.4 % a −2.1 % |
| εcc contra Tabla 5.1 | −4 % a −8 % |
| Ejemplo numérico de Mander (p. 65) | 50.4 MPa; la tesis lee 49.5 de un gráfico |
| Fórmula cerrada de Soesianawati (Ec. 5.17) | coincide con el procedimiento de 6 pasos al 4.º dígito |
| Longitud de rótula Lp contra Mander, Tabla 7.1 | 8 filas reproducidas |
| Desplazamiento último contra ensayos (Unidades 2, 3 y 4) | −19 %, −9 % y +9 % (contraste, no validación exacta) |
| Módulo `moment_curvature.py` existente (P = 0) | 1 % |
| Sección asimétrica contra cálculo a mano | Mi: +0.7 %; rigidez inicial: 0.001 % |
| Convergencia (láminas de 120 a 480; incremento de curvatura) | < 0.03 %; sin efecto |

## 6. Historial de commits

| Commit | Contenido |
|---|---|
| `6cbb871` | Panel COLUMN base e introducciones de los paneles |
| `2497927` | Se quita la dependencia de SciPy |
| `84475fd` | Tabla editable de capas de barras y corte acotado de la sección |
| `76ff56f` | Interacción P–M, ductilidad vs. P, variantes e informe Word/PDF |
| `c51531b` | Respuesta de la columna, criterio de diseño de Mander y conexión entre paneles |

## 7. Limitaciones conocidas

- **Solo carga monotónica:** la respuesta se evalúa sobre la curva esqueleto, sin descarga de fibras. Sin cortante en el análisis de la sección ni deslizamiento de barras.
- **φy de la Tabla 8.1 de Soesianawati:** mi φy difiere (−14 % a +9 % según la unidad) y no se logró explicar. La tendencia publicada entre las Unidades 1 y 2 es contraria a la física con carga axial creciente.
- **εcc:** queda 4–8 % por debajo de la Tabla 5.1. Probablemente la tesis contó las barras apoyadas de otra forma.
- **Balance de energía** (fractura del estribo): sus constantes son empíricas y no hay un valor publicado contra el cual comprobar *cuándo* ocurre.
- **Carga axial baja:** el análisis termina por εsu del acero y da desplazamientos muy grandes (Unidad 1: 192 mm calculado frente a 91 mm medido). El panel lo avisa y sugiere el criterio de diseño de Mander.
- **Cerca de la capacidad axial** (más de 80 %) el equilibrio puede no converger. Esos puntos se marcan en la interacción.
- **f't = 0.6√f'c** y Wc = Ws = 1 en el balance de energía son supuestos propios (en COLUMN son datos). Se pueden cambiar.
- **Rigidez a cortante:** la Ec. 7.83c impresa usa Kve en la zona fisurada exterior; se siguió la Fig. 7.14 (Kvc). Va anotado en el código.
- **Conexión entre paneles:** el panel de concreto solo publica en la vista de un f'c; si está en otra vista, COLUMN no recibe datos nuevos.
- **Streamlit:** `use_container_width` muestra avisos de obsolescencia (se mantiene por compatibilidad con `streamlit>=1.42`). Al abrir `/column_mphi` escribiendo la URL aparece un aviso "Page not found" (con el menú lateral no ocurre).

## 8. Pendientes (TODO)

### Trabajo grande

8. **Carga cíclica: lazos de histéresis.** Es lo que hace único al COLUMN original, y la tesis de Mander da las ecuaciones.
9. **Flexión biaxial,** y efectos como **velocidad de carga** o **pandeo de barras.**

### Otros pendientes detectados

- Secciones circulares (con zunchos) y huecas.
- Penetración de fluencia y deformación por cortante dentro del análisis cíclico (hoy solo se usan en la curva fuerza–desplazamiento).
- Explicar la diferencia de φy con la Tabla 8.1 y la de εcc con la Tabla 5.1.
- Encontrar un valor publicado para validar el balance de energía.
- Comparar con curvas de ensayo completas (hoy solo hay desplazamientos máximos).
- Publicar datos desde otras vistas del panel de concreto y enviar la geometría de COLUMN hacia Section Properties.
- Pasar a `width="stretch"` cuando se suba la versión mínima de Streamlit; revisar el aviso "Page not found".
- Revisar visualmente la sección 10 de COLUMN en el navegador (se probó solo con pruebas automatizadas).

## 9. Cómo ejecutar y probar

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
pip install -r requirements-dev.txt && pytest
```
