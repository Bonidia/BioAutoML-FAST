import streamlit as st
import base64

@st.cache_data
def get_base64(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def inject_css():
    with open("css/style.css", "r") as f:
        st.markdown(f'''<style>
                    [data-testid="stSidebar"] {{
                    background-image: url("data:image/png;base64,{get_base64('imgs/logo.png')}");
                    padding-top: 0px;
                    background-repeat: no-repeat;
                    background-position: 50% 2%;
                    margin-top: -0.2%;
                    background-size: 275px;
                    }}
                    {f.read()}</style>''', unsafe_allow_html=True)