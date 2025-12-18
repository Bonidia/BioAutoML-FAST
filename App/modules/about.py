import streamlit as st

def runUI():
    with st.container(border=True):
        st.markdown("**Authors**")
        st.markdown(
            """
            This platform was developed by:

            - **Breno L. S. de Almeida** (corresponding author; brenoslivio@usp.br)
            - **Robson P. Bonidia** (corresponding author; bonidia@utfpr.edu.br)
            - Martin Bole
            - Anderson P. Avila-Santos
            - Peter F. Stadler
            - **Ulisses Rocha** (corresponding author; ulisses.rocha@ufz.de)
            - André C. P. L. F. de Carvalho

            Please cite the associated publication when using this platform in academic work.
            """
        )

    with st.container(border=True):
        st.markdown("**Acknowledgements**")
        st.markdown(
            """
            This work has been funded by the Canadian International Development Research Centre (IDRC) under the Grant Agreement 109981,
            and the UK government’s Foreign, Commonwealth and Development Office. The views expressed here do not necessarily reflect 
            those of the UK government’s Foreign, Commonwealth and Development Office, IDRC, or IDRC’s Board of Governors. 
            Breno L. S. de Almeida has been funded by the São Paulo Research Foundation (FAPESP), grant #2024/10958-1, and the 
            Google PhD Fellowship.

            We also acknowledge open-source libraries and tools that made this work possible.
            """
        )

    # The authors would like to thank all people that contributed with this platform in some form. 
    # These people include: 

    with st.container(border=True):
        st.markdown("**Data Availability**")
        st.markdown(
            """
            The source code of the platform, along with all datasets used to build the model repository, is available at: https://github.com/Bonidia/BioAutoML-FAST

            All trained models can be downloaded directly from the platform.
            """
        )

if __name__ == "__main__":
    runUI()