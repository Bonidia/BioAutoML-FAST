import streamlit as st

def runUI():
    st.markdown("### Frequently Asked Questions")

    st.markdown(
        """
        **Q1: What type of data can I upload?**  
        A: Description here.

        **Q2: How long are results stored?**  
        A: Description here.

        **Q3: Can I use BioAutoML-FAST for commercial purposes?**  
        A: Yes. BioAutoML-FAST is released under the MIT License.
        """
    )

    st.markdown("### Video Tutorials")

    # Placeholder for videos – replace with your own URLs
    st.video("https://www.youtube.com/watch?v=VIDEO_ID_1")
    st.video("https://www.youtube.com/watch?v=VIDEO_ID_2")


if __name__ == "__main__":
    runUI()