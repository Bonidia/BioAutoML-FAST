import streamlit as st
import pandas as pd
import os

def runUI():
    if not st.session_state["queue"]:
        st.session_state["queue"] = True

    predict_path = "jobs"
    with st.form("sequences_submit"):
        job_id = st.text_input("Enter Job ID")

        submitted = st.form_submit_button("Submit")
    
    st.divider()

    if submitted:

        if job_id:
            job_path = os.path.join(predict_path, job_id)
            if os.path.exists(job_path):
                predictions = os.path.join(job_path, "model_predictions.csv")
                if os.path.exists(predictions):
                    st.success("Job was completed with the following results.")

                    df_results = pd.read_csv(predictions)

                    df_results["Probability"] = df_results.apply(lambda x: max(x[["Cis-reg", "coding", "rRNA", "sRNA", "tRNA", "unknown"]])*100, axis=1)

                    df_results = df_results[["nameseq", "prediction", "Probability"]]
                    
                    df_results.columns = ["Name", "Prediction", "Probability"]

                    st.dataframe(df_results,
                                column_config = {"Probability": st.column_config.ProgressColumn(
                                    help="Prediction probability",
                                    format="%.2f%%",
                                    min_value=0,
                                    max_value=100
                                )},
                                hide_index=True, use_container_width=True)
                else:
                    st.info("Job is still in progress. Come back later.")
            else:
                st.error("Job does not exist!")