import streamlit as st
import polars as pl
from queue import Queue
from threading import Thread
from streamlit.runtime.scriptrunner import add_script_run_ctx
import joblib
import os

def queue_listener():
    while True:
        if not job_queue.empty():
            train_files, test_files, job_path, data_type, training, testing, classifier, imbalance, fselection = job_queue.get()
            submit_job(train_files, test_files, job_path, data_type, training, testing, classifier, imbalance, fselection)

def submit_job(train_files, test_files, job_path, data_type, training, testing, classifier, imbalance, fselection):

    if training == "Load model":
        save_path = os.path.join(job_path, "trained_model.sav")
        with open(save_path, mode="wb") as f:
            f.write(train_files.getvalue())

        model = joblib.load(save_path)

        command = [
            "python",
            "BioAutoML-multiclass.py" if len(model["label_encoder"].classes_) > 2 else "BioAutoML-binary.py",
            "-path_model", save_path,
            "-nf", "True",
        ]

        if test_files:
            if data_type == "Structured data":
                test_path = os.path.join(job_path, "test")
                os.makedirs(test_path)

                feat_path = os.path.join(job_path, "feat_extraction")
                os.makedirs(feat_path)

                if testing == "Test set":
                    save_path = os.path.join(test_path, "test.csv")
                    with open(save_path, mode="wb") as f:
                        f.write(test_files.getvalue())
                    
                    df_test = pl.from_pandas(pd.read_csv(save_path).reset_index())
                    df_test = df_test.rename({"index": "nameseq"})
                    df_labels = df_test.select(["label"])
                    df_index = df_test.select(["nameseq"])
                    df_test = df_test.drop(["nameseq", "label"])

                    df_index.write_csv(os.path.join(feat_path, "fnameseqtest.csv"))
                    df_test.write_csv(os.path.join(feat_path, "test.csv"))
                    df_labels.write_csv(os.path.join(feat_path, "test_labels.csv"))
                    
                    command.append("--test")
                    command.append(os.path.join(feat_path, "test.csv"))
                    command.append("--test_label")
                    command.append(os.path.join(feat_path, "test_labels.csv"))
                    command.append("--test_nameseq")
                    command.append(os.path.join(feat_path, "fnameseqtest.csv"))
                else:
                    save_path = os.path.join(test_path, "predicted.csv")
                    with open(save_path, mode="wb") as f:
                        f.write(test_files.getvalue())
                    
                    df_test = pl.from_pandas(pd.read_csv(save_path).reset_index())
                    df_test = df_test.rename({"index": "nameseq"})
                    df_test = df_test.with_columns(pl.lit("Predicted").alias("label"))
                    df_index = df_test.select(["nameseq"])
                    df_labels = df_test.select(["label"])
                    df_test = df_test.drop(["nameseq", "label"])

                    df_index.write_csv(os.path.join(feat_path, "fnameseqtest.csv"))
                    df_test.write_csv(os.path.join(feat_path, "test.csv"))
                    df_labels.write_csv(os.path.join(feat_path, "test_labels.csv"))

                    command.append("--test")
                    command.append(os.path.join(feat_path, "test.csv"))
                    command.append("--test_label")
                    command.append(os.path.join(feat_path, "test_labels.csv"))
                    command.append("--test_nameseq")
                    command.append(os.path.join(feat_path, "fnameseqtest.csv"))

                utils.summary_stats(os.path.join(job_path, "test"), data_type, job_path, True)
            else:
                # test_path = os.path.join(job_path, "test")
                # os.makedirs(test_path)

                test_path = os.path.join(job_path, "test")
                os.makedirs(test_path)

                if testing == "Test set":
                    for file in test_files:
                        save_path = os.path.join(test_path, file.name)
                        with open(save_path, mode="wb") as f:
                            f.write(file.getvalue())

                    test_fasta = {os.path.splitext(f)[0] : os.path.join(test_path, f) for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))}

                    test_extraction(job_path, test_fasta, model, data_type)

                    utils.summary_stats(os.path.join(job_path, "feat_extraction/test"), data_type, job_path, False)

                    command.extend(["--test", os.path.join(job_path, "best_descriptors/best_test.csv")])
                    command.extend(["--test_label", os.path.join(job_path, "feat_extraction/flabeltest.csv")])
                    command.extend(["--test_nameseq", os.path.join(job_path, "feat_extraction/fnameseqtest.csv")])
                else:
                    save_path = os.path.join(test_path, "predicted.fasta")
                    with open(save_path, mode="wb") as f:
                        f.write(test_files.getvalue())
                    
                    test_fasta = {"Predicted" : os.path.join(test_path, f) for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))}

                    test_extraction(job_path, test_fasta, model, data_type)

                    utils.summary_stats(os.path.join(job_path, "feat_extraction/test"), data_type, job_path, False)

                    command.extend(["--test", os.path.join(job_path, "best_descriptors/best_test.csv")])
                    command.extend(["--test_label", os.path.join(job_path, "feat_extraction/flabeltest.csv")])
                    command.extend(["--test_nameseq", os.path.join(job_path, "feat_extraction/fnameseqtest.csv")])

        command.extend(["--n_cpu", "-1"])
        command.extend(["--output", job_path])

        subprocess.run(command, cwd="..")

def runUI():
    global job_queue

    job_queue = Queue()

    if not st.session_state["queue"]:
        queue_thread = Thread(target=queue_listener)
        add_script_run_ctx(queue_thread)
        queue_thread.start()
        st.session_state["queue"] = True

    model = st.selectbox("Select trained model", [
                "Non-classical secreted proteins"])
    job_id = ""
    
    if model == "Non-classical secreted proteins":
        st.info("""
                **Data set from the following paper:** Zhang, Y., Yu, S., 
                Xie, R., Li, J., Leier, A., Marquez-Lago, T. T., ... & 
                Song, J. (2020). PeNGaRoo, a combined gradient boosting 
                and ensemble learning framework for predicting non-classical 
                secreted proteins. Bioinformatics, 36(3), 704-712.
                """)
        dataset_id = "dataset1_zhang_protein"
    else:
        st.info("test")

    st.success(f"You can consult experiments done with this data set in \"Jobs\" using the following ID: **{dataset_id}**")

    with st.form("repo_submit", clear_on_submit=True):

        test_files = st.file_uploader("FASTA file for prediction", accept_multiple_files=False, help="Single file for prediction (e.g. predict.fasta)")

        submitted = st.form_submit_button("Submit", use_container_width=True, type="primary")

    predict_path = os.path.abspath("jobs")

    if submitted:
        if not test_files:
            #with queue_info:
            st.error("Please upload the required test or prediction file(s).")
        else:
            job_id = ''.join([choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16)])
            job_path = os.path.join(predict_path, job_id)

            os.makedirs(job_path)

            dtype_str = dataset_id.split('_')[-1]

            training = "Load model"
            testing = "Prediction set"
            classifier, imbalance, fselection = False, False, False

            if dtype_str == "protein":
                data_type = "Protein"
            elif dtype_str == "dnarna":
                data_type = "DNA/RNA"

            job_data = {
                "data_type": [data_type],
                "training_set": [training == "Training set"],
                "testing_set": [testing],
                "classifier_selected": [classifier], 
                "imbalance_methods": [imbalance],  
                "feature_selection": [fselection],  
            }

            df_job_data = pl.DataFrame(job_data)
            tsv_path = os.path.join(job_path, "job_info.tsv")
            df_job_data.write_csv(tsv_path, separator='\t')

            job_queue.put((train_files, test_files, job_path, data_type, training, testing, classifier, imbalance, fselection))

            with queue_info:
                st.success(f"Job submitted to the queue. You can consult the results in \"Jobs\" using the following ID: **{job_id}**")