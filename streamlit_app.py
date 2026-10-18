import streamlit as st

from tools_registry import build_navigation


st.set_page_config(
    page_title="Engineering Tools",
    page_icon="🧮",
    layout="wide",
)

navigation = st.navigation(build_navigation())

navigation.run()
