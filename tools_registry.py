"""Single source of truth for the tools shown in the app.

To add a tool: create ``calculations/<area>/<tool>.py`` (logic) and
``pages/<area>/<tool>.py`` (UI), then add one ``Tool`` entry below.
The sidebar menu and the Home page are built from this list.
"""

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class Tool:
    group: str
    title: str
    path: str
    icon: str
    description: str
    reference: str = ""  # source paper / book / note
    default: bool = False


HOME = Tool(
    group="Engineering Tools",
    title="Home",
    path="pages/home.py",
    icon="🏠",
    description="Index of tools.",
    default=True,
)

TOOLS = [
    # --- Numerical methods ---------------------------------------------
    Tool(
        "Numerical Methods",
        "Least Squares",
        "pages/numerical_methods/least_squares.py",
        "📈",
        "Curve fitting by least squares.",
    ),
    Tool(
        "Numerical Methods",
        "Gaussian Elimination",
        "pages/numerical_methods/gaussian_elimination.py",
        "🧮",
        "Solution of linear systems step by step.",
    ),
    # --- Steel ---------------------------------------------------------
    Tool(
        "Steel Sections",
        "Section Properties",
        "pages/structural/section_properties.py",
        "📐",
        "AISC shape properties and section drawing.",
        "AISC Shapes Database",
    ),
    Tool(
        "Steel Sections",
        "Section Classification",
        "pages/structural/section_classification.py",
        "🔍",
        "Compact, noncompact and slender elements.",
        "AISC 360-16, Table B4.1",
    ),
    # --- Reinforced concrete -------------------------------------------
    Tool(
        "Reinforced Concrete",
        "Concrete Stress-Strain",
        "pages/structural/concrete_curves.py",
        "🏗️",
        "Compression and tension curves of concrete.",
    ),
    Tool(
        "Reinforced Concrete",
        "Column Axial Response",
        "pages/structural/column_axial_response.py",
        "🧱",
        "Force-displacement curve of a plain concrete column.",
    ),
    Tool(
        "Reinforced Concrete",
        "Shrinkage Cracking",
        "pages/structural/shrinkage_cracking.py",
        ":material/vertical_split:",
        "Crack width of a fully restrained slab versus reinforcement ratio.",
        "Gilbert (1992), ACI Structural Journal 89(2)",
    ),
    Tool(
        "Reinforced Concrete",
        "Moment-Curvature",
        "pages/structural/moment_curvature.py",
        ":material/moving:",
        "Moment-curvature of rectangular sections: Park-Paulay and Mander models.",
        "Park & Paulay, Fig. 6.6; Mander et al. (1988)",
    ),
    Tool(
        "Reinforced Concrete",
        "COLUMN — Análisis Momento–Curvatura",
        "pages/structural/column_mphi.py",
        ":material/query_stats:",
        "Moment-curvature of confined rectangular sections under axial load (monotonic, COLUMN-type fibre analysis).",
        "Mander (1984), Research Report 84-2, §7.4.2; Soesianawati (1986)",
    ),
]

GROUP_ORDER = ["Numerical Methods", "Steel Sections", "Reinforced Concrete"]


def tools_by_group():
    """Ordered mapping group -> list of tools (Home excluded)."""
    groups = {name: [] for name in GROUP_ORDER}
    for tool in TOOLS:
        groups.setdefault(tool.group, []).append(tool)
    return groups


def _page(tool):
    return st.Page(tool.path, title=tool.title, icon=tool.icon, default=tool.default)


def build_navigation():
    """Dictionary for ``st.navigation``."""
    pages = {HOME.group: [_page(HOME)]}
    for group, tools in tools_by_group().items():
        pages[group] = [_page(t) for t in tools]
    return pages
