import streamlit as st
import requests
from utils.tasks import manager
import re

def runUI():
    with st.expander("Sharing your model"):
        st.info("""
            Here you can **share a model generated within the platform** to be considered for inclusion in the **model repository**.  
            Shared models allow other users to benefit from task-specific predictors trained on new datasets and biological problems.

            To submit a model, provide the **Job ID** of a completed, **non-encrypted** job, along with a brief **description of the data** used and the **DOI of the corresponding publication** to be cited. This information ensures proper attribution, transparency, and reproducibility.

            All submissions undergo a **manual review process** to verify data quality, documentation, and relevance. If approved, the model may be added to the curated repository and made available for community use.

            You may be contacted by the our team if additional details or clarification are required during the review process.
            """
        )

    with st.form("share_submit", border=True, clear_on_submit=True):

        checkcol1, checkcol2 = st.columns(2)

        with checkcol1:
            job_id = st.text_input("Enter Job ID", key="job_input", help="Job ID of a valid non-encrypted submission")
            
        with checkcol2:
            email = st.text_input("Your email address (Optional)", help="We may enter in contact if more details are required")

        text_area = st.text_area(
            "Provide a description of your data and the DOI of the paper we should reference",
        )

        submitted = st.form_submit_button("Submit", use_container_width=True, type="primary")

    if submitted:
        if job_id and text_area:

            job = manager.get_result(job_id)

            if job:
                api_key = st.secrets["api_key"]

                subject = f"[BioAutoML-FAST] User wants to share their model"

                if email:
                    body = f"""Dear administrator,\n\nUser ({email}) wants to share their model.\n\nJob ID: {job_id}\nText from the user:\n{text_area}\n\nPlease verify if the model is suitable to be used in the model repository."""
                else:
                    body = f"""Dear administrator,\n\nUser wants to share their model.\n\nJob ID: {job_id}\nText from the user:\n{text_area}\n\nPlease verify if the model is suitable to be used in the model repository."""

                response =  requests.post(
                            "https://api.mailgun.net/v3/bioauto.inteligentehub.com.br/messages",
                            auth=("api", api_key),
                            data={"from": "BioAutoML-FAST <sharing@bioauto.inteligentehub.com.br>",
                                "to": "brenoslivio@usp.br",
                                "subject": subject,
                                "text": body})
                
                if response.status_code == 200:
                    # Print a message to console after successfully sending the email.
                    st.success("Email sent to administrator.")
                else:
                    st.error("Email failed to be sent to administrator.")
            else:
                st.error("Please enter a valid Job ID.")
        else:
            st.error("Please fill all the fields.")

if __name__ == "__main__":
    runUI()