import streamlit as st
from streamlit_option_menu import option_menu
import utils, modules
import subprocess, os

def runUI():
    st.set_page_config(page_title = "BioAutoML-FAST", page_icon = "imgs/icon.png", initial_sidebar_state = "expanded", layout="wide")
    
    curr_dir = os.getcwd()

    root_dir = curr_dir.split("BioAutoML-Fast")[0] + "BioAutoML-Fast"

    os.chdir(os.path.join(root_dir, "App"))

    utils.inject_css()

    page = option_menu(None, ["Home", "Jobs", "Model Repository", "About"], 
    icons=["house", "gear-wide", "diagram-2", "info-circle"], 
    menu_icon="cast", default_index=0, orientation="horizontal")

    # Initialize session state for thread management
    if "queue_started" not in st.session_state:
        st.session_state.queue_started = False

    if page == "Home":
        modules.home.runUI()
        if "job_path" in st.session_state:
            del st.session_state["job_path"]
    elif page == "Jobs":
        modules.jobs.runUI()
    elif page == "Model Repository":
        modules.repo.runUI()
        if "job_path" in st.session_state:
            del st.session_state["job_path"]
    elif page == "Model Repository":
        if "job_path" in st.session_state:
            del st.session_state["job_path"]
    elif page == "About":
        if "job_path" in st.session_state:
            del st.session_state["job_path"]

if __name__ == "__main__":
    runUI()