import streamlit as st

from styles.main import apply_global_styles
from tools_registry import tools_by_group

apply_global_styles()

st.title("Engineering Tools")

st.write(
    """
    Herramientas interactivas de ingeniería estructural y métodos numéricos.
    """
)

for group, tools in tools_by_group().items():
    st.subheader(group)
    columns = st.columns(2)
    for i, tool in enumerate(tools):
        with columns[i % 2].container(border=True):
            st.page_link(tool.path, label=f"**{tool.title}**", icon=tool.icon)
            st.caption(tool.description)
            if tool.reference:
                st.caption(f"Referencia: {tool.reference}")
