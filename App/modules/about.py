import streamlit as st

def runUI():
    with st.container(border=True):
        st.markdown("**Publication**")
        st.markdown(
            """
            Silva de Almeida, B. L., Bonidia, R., Bole, M., Avila-Santos, A., Stadler, P. F.,
            Nunes da Rocha, U., & de Carvalho, A. C. L. F. (2026).
            *BioAutoML-FAST: an automated machine-learning platform for reusable and benchmarked biological sequence models.*
            **bioRxiv**, 2026-04.
            """
        )

        col_doi, col_zenodo, col_github = st.columns(3)
        with col_doi:
            st.link_button(
                "Paper (DOI)",
                "https://doi.org/10.64898/2026.04.18.719383",
                use_container_width=True,
            )
        with col_zenodo:
            st.link_button(
                "Zenodo",
                "https://doi.org/10.5281/zenodo.20349210",
                use_container_width=True,
            )
        with col_github:
            st.link_button(
                "GitHub",
                "https://github.com/Bonidia/BioAutoML-FAST",
                use_container_width=True,
            )

        with st.expander("BibTeX citation"):
            st.code(
                """@article{silva2026bioautoml,
  title={BioAutoML-FAST: an automated machine-learning platform for reusable and benchmarked biological sequence models},
  author={Silva de Almeida, Breno Livio and Bonidia, Robson and Bole, Martin and Avila-Santos, Anderson and Stadler, Peter F and Nunes da Rocha, Ulisses and de Carvalho, Andre CP L F},
  journal={bioRxiv},
  pages={2026--04},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}""",
                language="bibtex",
            )

    with st.container(border=True):
        st.markdown("**Authors**")
        st.markdown(
            """
            This platform was developed by:

            - **Breno L. S. de Almeida** (corresponding author; [brenoslivio@usp.br](mailto:brenoslivio@usp.br))
            - **Robson P. Bonidia** (corresponding author; [bonidia@utfpr.edu.br](mailto:bonidia@utfpr.edu.br))
            - Martin Bole
            - Anderson P. Avila-Santos
            - Peter F. Stadler
            - **Ulisses Rocha** (corresponding author; [ulisses.rocha@ufz.de](mailto:ulisses.rocha@ufz.de))
            - André C. P. L. F. de Carvalho

            Please cite the associated publication when using this platform in academic work.
            """
        )

    with st.container(border=True):
        st.markdown("**Acknowledgements**")
        st.markdown(
            """
            This work has been funded by the Canadian International Development Research Centre (IDRC) under the Grant Agreement 109981,
            and the UK government's Foreign, Commonwealth and Development Office. The views expressed here do not necessarily reflect
            those of the UK government's Foreign, Commonwealth and Development Office, IDRC, or IDRC's Board of Governors.
            Breno L. S. de Almeida has been funded by the São Paulo Research Foundation (FAPESP), grant #2024/10958-1, and the
            Google PhD Fellowship. This project (ZT-I-PF-3-108) was funded by the Initiative and Networking Fund of the Helmholtz
            Association in the framework of the Helmholtz Metadata Collaboration project call.

            The authors thank the following individuals for their valuable suggestions and assistance with testing: Sanchita Kamath, Asaad Ataa,
            Faith Oni, Jamile Souza, Jana Schor, Matthias Bernt, and Bianca Pessa.

            We also acknowledge open-source libraries and tools that made this work possible.
            """
        )

    with st.container(border=True):
        st.markdown("**Data Availability**")
        st.markdown(
            """
            The source code of the platform, along with all datasets used to build the model repository, is available at [github.com/Bonidia/BioAutoML-FAST](https://github.com/Bonidia/BioAutoML-FAST).

            All trained models can be downloaded directly from the platform.
            """
        )

if __name__ == "__main__":
    runUI()
