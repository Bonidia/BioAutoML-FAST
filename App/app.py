import streamlit as st
from streamlit_option_menu import option_menu
import utils, modules
import subprocess, os, sys
from utils.tasks import manager
from datetime import datetime

def clear_cache():
    keys = list(st.session_state.keys())
    for key in keys:
        if key != "cookie":
            st.session_state.pop(key)

@st.dialog("🍪 Cookie notice", width="large")
def cookie_dialog():
    st.markdown(
        """
        This web server uses **session cookies solely** to ensure proper functionality.

        No personal tracking or persistent cookies are employed.
        """
    )

    st.session_state["cookie"] = True

def runUI():
    st.set_page_config(page_title = "BioAutoML-FAST", page_icon = "imgs/icon.png", initial_sidebar_state = "expanded", layout="wide")

    utils.inject_css()

    manager._create_db()

    page = option_menu(None, ["Home", "Jobs", "Model Repository", "Share", "AI Help & Tutorials", "About"], 
    icons=["house", "gear-wide", "diagram-2", "link", "book", "info-circle"],
    menu_icon="cast", default_index=0, orientation="horizontal")

    if page == "Home":
        modules.home.runUI()
        clear_cache()
    elif page == "Jobs":
        modules.jobs.runUI()
    elif page == "Model Repository":
        modules.repo.runUI()
        clear_cache()
    elif page == "Share":
        modules.share.runUI()
        clear_cache()
    elif page == "AI Help & Tutorials":
        modules.help.runUI()
        clear_cache()
    elif page == "About":
        modules.about.runUI()
        clear_cache()

    # Show dialog once per session
    if "cookie" not in st.session_state:
        cookie_dialog()

    st.markdown(
        f"""
        <hr>
        <div style="text-align:center; font-size: 0.9em;">
        © {datetime.now().year} BioAutoML-FAST — Released under the 
        <a href="https://opensource.org/licenses/MIT" target="_blank">MIT License</a>. 
        </div>
        <br>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    runUI()