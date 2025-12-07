import streamlit as st
from streamlit_option_menu import option_menu
import utils, modules
import subprocess, os, sys

def start_worker_subprocess():
    # Run worker in background, redirect logs to a file or leave them alone
    log = open(os.path.join(os.path.dirname(__file__), "rq_worker.log"), "a")

    p = subprocess.Popen(["rq", "worker"], stdout=log, stderr=log, close_fds=True)

    return p  

def runUI():
    st.set_page_config(page_title = "BioAutoML-FAST", page_icon = "imgs/icon.png", initial_sidebar_state = "expanded", layout="wide")

    utils.inject_css()

    page = option_menu(None, ["Home", "Jobs", "Model Repository", "About"], 
    icons=["house", "gear-wide", "diagram-2", "info-circle"], 
    menu_icon="cast", default_index=0, orientation="horizontal")

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

    # Example usage when your app starts (guard to avoid spawning multiple workers):
    if "worker_proc" not in st.session_state:
        st.session_state["worker_proc"] = start_worker_subprocess()

if __name__ == "__main__":
    runUI()