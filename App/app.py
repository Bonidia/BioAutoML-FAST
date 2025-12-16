import streamlit as st
from streamlit_option_menu import option_menu
import utils, modules
import subprocess, os, sys
from utils.tasks import manager

def clear_cache():
    keys = list(st.session_state.keys())
    for key in keys:
        st.session_state.pop(key)

def runUI():
    st.set_page_config(page_title = "BioAutoML-FAST", page_icon = "imgs/icon.png", initial_sidebar_state = "expanded", layout="wide")

    utils.inject_css()

    manager._create_db()

    page = option_menu(None, ["Home", "Jobs", "Model Repository", "Share", "About"], 
    icons=["house", "gear-wide", "diagram-2", "link", "info-circle"],
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
    elif page == "About":
        clear_cache()

    # Example usage when your app starts (guard to avoid spawning multiple workers):
    # if "worker_proc" not in st.session_state:
    #     st.session_state["worker_proc"] = start_worker_subprocess()

if __name__ == "__main__":
    runUI()