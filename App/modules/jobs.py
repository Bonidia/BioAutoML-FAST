import streamlit as st
import polars as pl
import pandas as pd
import plotly.graph_objects as go
import os

def runUI():
    if not st.session_state["queue"]:
        st.session_state["queue"] = True

    predict_path = "jobs"
    with st.form("sequences_submit"):
        job_id = st.text_input("Enter Job ID")

        submitted = st.form_submit_button("Submit")
    
    if submitted:

        if job_id:
            job_path = os.path.join(predict_path, job_id)
            if os.path.exists(job_path):
                test_fold = os.path.join(job_path, "test")

                test_set = True if os.path.exists(test_fold) else False

                if test_set:
                    predictions = os.path.join(job_path, "test_confusion_matrix.csv")
                else:
                    predictions = os.path.join(job_path, "training_confusion_matrix.csv")
                
                if os.path.exists(predictions):
                    st.session_state["job_path"] = job_path
                else:
                    del st.session_state["job_path"]
                    st.info("Job is still in progress. Come back later.")
            else:
                del st.session_state["job_path"]
                st.error("Job does not exist!")

    if "job_path" in st.session_state:
        st.divider()

        st.success("Job was completed with the following results.")

        st.markdown("Sequence statistics")

        tab1, tab2, tab3 = st.tabs(['Performance Metrics', 'Predictions', 'Feature Importance'])
        
        with tab1:
            evaluation = st.selectbox(":mag_right: Evaluation set", ["Training set", "Test set"],
                        help="Training set evaluated with 10-fold cross-validation") #index=None

            col1, col2 = st.columns(2)

            with col1:
                if evaluation == "Training set":
                    df_cv = pl.read_csv(os.path.join(st.session_state["job_path"], "training_kfold(10)_metrics.csv"))

                    st.markdown(f"""**Accuracy:** {df_cv['ACC'].item()} ± {df_cv['std_ACC'].item()}""")
                    st.markdown(f"""**MCC:** {df_cv['MCC'].item()} ± {df_cv['std_MCC'].item()}""")
                    st.markdown(f"""**F1-score (micro avg.):** {df_cv['F1_micro'].item()} ± {df_cv['std_F1_micro'].item()}""")
                    st.markdown(f"""**F1-score (macro avg.):** {df_cv['F1_macro'].item()} ± {df_cv['std_F1_macro'].item()}""")
                    st.markdown(f"""**F1-score (weighted avg.):** {df_cv['F1_w'].item()} ± {df_cv['std_F1_w'].item()}""")
                    st.markdown(f"""**Kappa:** {df_cv['kappa'].item()} ± {df_cv['std_kappa'].item()}""")
                else:
                    df_report = pl.read_csv(os.path.join(st.session_state["job_path"], "metrics_test.csv"))
                    st.dataframe(df_report, hide_index=True, use_container_width=True)
            with col2:
                if evaluation == "Training set":
                    df = pd.read_csv(os.path.join(st.session_state["job_path"], "training_confusion_matrix.csv"))

                    # Extract labels and confusion matrix values
                    labels = df.columns[1:-1].tolist()
                    
                    values = df.iloc[0:-1, 1:-1].values.tolist()

                    fig = go.Figure(data=go.Heatmap(
                        z=values,
                        x=labels,
                        y=labels,
                        colorscale='Blues'
                    ))

                    fig.update_layout(
                        title='Confusion Matrix',
                        xaxis_title='Predicted label',
                        yaxis_title='True label'
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    df = pd.read_csv(os.path.join(st.session_state["job_path"], "test_confusion_matrix.csv"))

                    # Extract labels and confusion matrix values
                    labels = df.columns[1:-1].tolist()
                    
                    values = df.iloc[0:-1, 1:-1].values.tolist()

                    fig = go.Figure(data=go.Heatmap(
                        z=values,
                        x=labels,
                        y=labels,
                        colorscale='Blues'
                    ))

                    fig.update_layout(
                        title='Confusion Matrix',
                        xaxis_title='Predicted label',
                        yaxis_title='True label'
                    )

                    st.plotly_chart(fig, use_container_width=True)

        # df_results = pd.read_csv(predictions)

        # df_results["Probability"] = df_results.apply(lambda x: max(x[["Cis-reg", "coding", "rRNA", "sRNA", "tRNA", "unknown"]])*100, axis=1)

        # df_results = df_results[["nameseq", "prediction", "Probability"]]
        
        # df_results.columns = ["Name", "Prediction", "Probability"]

        # st.dataframe(df_results,
        #             column_config = {"Probability": st.column_config.ProgressColumn(
        #                 help="Prediction probability",
        #                 format="%.2f%%",
        #                 min_value=0,
        #                 max_value=100
        #             )},
        #             hide_index=True, use_container_width=True)