import streamlit as st
import polars as pl
import pandas as pd
from io import StringIO
from Bio import SeqIO
import subprocess
from subprocess import Popen
import streamlit.components.v1 as components
import os
from secrets import choice
import string
from queue import Queue
from threading import Thread
from streamlit.runtime.scriptrunner import add_script_run_ctx
import utils
import base64
import joblib
import shutil

def test_extraction(job_path, test_data):
    datasets = []

    path = os.path.join(job_path, "feat_extraction", "test")
    feat_path = os.path.join(job_path, "feat_extraction")

    try:
        shutil.rmtree(path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    print("Creating Directory...")
    os.makedirs(path)

    for label in test_data:
        subprocess.run(["python", "MathFeature/preprocessing/preprocessing.py",
                    "-i", test_data[label], 
                    "-o", os.path.join(path, f"pre_{label}.fasta")],
                    cwd="..",
                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        datasets.append(feat_path + "/NAC.csv")
        datasets.append(feat_path + "/DNC.csv")
        datasets.append(feat_path + "/TNC.csv")
        datasets.append(feat_path + "/kGap_di.csv")
        datasets.append(feat_path + "/kGap_tri.csv")
        datasets.append(feat_path + "/ORF.csv")
        datasets.append(feat_path + "/Fickett.csv")
        datasets.append(feat_path + "/Shannon.csv")
        datasets.append(feat_path + "/FourierBinary.csv")
        datasets.append(feat_path + "/FourierComplex.csv")
        datasets.append(feat_path + "/Tsallis.csv")
        datasets.append(feat_path + "/repDNA.csv")

        commands = [["python", "MathFeature/methods/ExtractionTechniques.py",
                            "-i", os.path.join(path, f"pre_{label}.fasta"), "-o", feat_path + "/NAC.csv", "-l", label,
                            "-t", "NAC", "-seq", "1"],

                    ["python", "MathFeature/methods/ExtractionTechniques.py", "-i",
                            os.path.join(path, f"pre_{label}.fasta"), "-o", feat_path + "/DNC.csv", "-l", label,
                            "-t", "DNC", "-seq", "1"],

                    ["python", "MathFeature/methods/ExtractionTechniques.py", "-i",
                            os.path.join(path, f"pre_{label}.fasta"), "-o", feat_path + "/TNC.csv", "-l", label,
                            "-t", "TNC", "-seq", "1"],
                    ["python", "MathFeature/methods/Kgap.py", "-i",
                            os.path.join(path, f"pre_{label}.fasta"), "-o", feat_path + "/kGap_di.csv", "-l",
                            label, "-k", "1", "-bef", "1",
                            "-aft", "2", "-seq", "1"],
                    ["python", "MathFeature/methods/Kgap.py", "-i",
                            os.path.join(path, f"pre_{label}.fasta"), "-o", feat_path + "/kGap_tri.csv", "-l",
                            label, "-k", "1", "-bef", "1",
                            "-aft", "3", "-seq", "1"],
                    ["python", "MathFeature/methods/CodingClass.py", "-i",
                            os.path.join(path, f"pre_{label}.fasta"), "-o", feat_path + "/ORF.csv", "-l", label],
                    ["python", "MathFeature/methods/FickettScore.py", "-i",
                            os.path.join(path, f"pre_{label}.fasta"), "-o", feat_path + "/Fickett.csv", "-l", label,
                            "-seq", "1"],
                    ["python", "MathFeature/methods/EntropyClass.py", "-i",
                            os.path.join(path, f"pre_{label}.fasta"), "-o", feat_path + "/Shannon.csv", "-l", label,
                            "-k", "5", "-e", "Shannon"],
                    ["python", "MathFeature/methods/FourierClass.py", "-i",
                            os.path.join(path, f"pre_{label}.fasta"), "-o", feat_path + "/FourierBinary.csv", "-l", label,
                            "-r", "1"],
                    ["python", "other-methods/FourierClass.py", "-i",
                            os.path.join(path, f"pre_{label}.fasta"), "-o", feat_path + "/FourierComplex.csv", "-l", label,
                            "-r", "6"],
                    ["python", "other-methods/TsallisEntropy.py", "-i",
                            os.path.join(path, f"pre_{label}.fasta"), "-o", feat_path + "/Tsallis.csv", "-l", label,
                            "-k", "5", "-q", "2.3"],
                    ["python", "other-methods/repDNA/repDNA-feat.py", "--file",
                            os.path.join(path, f"pre_{label}.fasta"), "--output", feat_path + "/repDNA.csv", "--label", label]
        ]

        processes = [Popen(cmd, cwd="..", stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT) for cmd in commands]
        for p in processes: p.wait()

    if datasets:
        datasets = list(dict.fromkeys(datasets))
        dataframes = pd.concat([pd.read_csv(f) for f in datasets], axis=1)
        dataframes = dataframes.loc[:, ~dataframes.columns.duplicated()]
        dataframes = dataframes[~dataframes.nameseq.str.contains("nameseq")]

    y_test = dataframes.pop("label")
    nameseq_test = dataframes.pop("nameseq")
    flabeltest = feat_path + '/flabeltest.csv'
    fnameseqtest = feat_path + '/fnameseqtest.csv'
    nameseq_test.to_csv(fnameseqtest, index=False, header=True)
    y_test.to_csv(flabeltest, index=False, header=True)

    path_bio = os.path.join(job_path, "best_descriptors")
    if not os.path.exists(path_bio):
        os.mkdir(path_bio)

    df_train = joblib.load(os.path.join(job_path, "trained_model.sav"))["train"]

    common_columns = dataframes.columns.intersection(df_train.columns)
    df_predict = dataframes[common_columns]

    df_predict.to_csv(os.path.join(path_bio, "best_test.csv"), index=False)

def submit_job(train_files, test_files, job_path, data_type, training, testing):

    if training == "Training set":
        train_path = os.path.join(job_path, "train")
        os.makedirs(train_path)

        for file in train_files:
            save_path = os.path.join(train_path, file.name)
            with open(save_path, mode="wb") as f:
                f.write(file.getvalue())
        
        train_fasta = {os.path.splitext(f)[0] : os.path.join(train_path, f) for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, f))}
    
        command = [
            "python",
            "BioAutoML-feature.py" if data_type == "DNA/RNA" else "BioAutoML-feature-protein.py",
            "--fasta_train",
        ]

        command.extend(train_fasta.values())
        command.append("--fasta_label_train")
        command.extend(train_fasta.keys())

        if test_files:
            test_path = os.path.join(job_path, "test")
            os.makedirs(test_path)

            for file in test_files:
                save_path = os.path.join(test_path, file.name)
                with open(save_path, mode="wb") as f:
                    f.write(file.getvalue())

            test_fasta = {os.path.splitext(f)[0] : os.path.join(test_path, f) for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))}

            command.append("--fasta_test")
            command.extend(test_fasta.values())
            command.append("--fasta_label_test")
            command.extend(test_fasta.keys())

        command.extend(["--n_cpu", "-1"])
        command.extend(["--output", job_path])

        subprocess.run(command, cwd="..")

        utils.summary_stats(os.path.join(job_path, "feat_extraction/train"), job_path)

        if test_files:
            utils.summary_stats(os.path.join(job_path, "feat_extraction/test"), job_path)
  
        model = joblib.load(os.path.join(job_path, "trained_model.sav"))
        model["train_stats"] = pd.read_csv(os.path.join(job_path, "train_stats.csv"))
        joblib.dump(model, os.path.join(job_path, "trained_model.sav"))

    elif training == "Load model":
        save_path = os.path.join(job_path, "trained_model.sav")
        with open(save_path, mode="wb") as f:
            f.write(train_files.getvalue())

        command = [
            "python",
            "BioAutoML-multiclass.py",
            "-path_model", save_path,
            "-nf", "True",
        ]

        if test_files:
            test_path = os.path.join(job_path, "test")
            os.makedirs(test_path)

            for file in test_files:
                save_path = os.path.join(test_path, file.name)
                with open(save_path, mode="wb") as f:
                    f.write(file.getvalue())

            test_fasta = {os.path.splitext(f)[0] : os.path.join(test_path, f) for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))}

            test_extraction(job_path, test_fasta)

            utils.summary_stats(os.path.join(job_path, "feat_extraction/test"), job_path)

            command.extend(["--test", os.path.join(job_path, "best_descriptors/best_test.csv")])
            command.extend(["--test_label", os.path.join(job_path, "feat_extraction/flabeltest.csv")])
            command.extend(["--test_nameseq", os.path.join(job_path, "feat_extraction/fnameseqtest.csv")])

        command.extend(["--n_cpu", "-1"])
        command.extend(["--output", job_path])

        subprocess.run(command, cwd="..")

        #model = joblib.load(os.path.join(job_path, "trained_model.sav"))
        #model["train_stats"].to_csv(os.path.join(job_path, "train_stats.csv"), index=False)

        # [1python', 'BioAutoML-multiclass.py', 
        # '-train', '/home/brenoslivio/Documents/🥼 Research/git/BioAutoML-Fast/App/jobs/nXCE6KZC690Gl2kk/best_descriptors/best_train.csv', 
        # '-train_label', '/home/brenoslivio/Documents/🥼 Research/git/BioAutoML-Fast/App/jobs/nXCE6KZC690Gl2kk/feat_extraction/flabeltrain.csv', 
        # '-test', '', '-test_label', '', '-test_nameseq', '', '-nf', 'True', '-n_cpu', '-1', '-classifier', '1', '-output', 
        # '/home/brenoslivio/Documents/🥼 Research/git/BioAutoML-Fast/App/jobs/nXCE6KZC690Gl2kk']

def queue_listener():
    while True:
        if not job_queue.empty():
            train_files, test_files, job_path, data_type, training, testing = job_queue.get()
            submit_job(train_files, test_files, job_path, data_type, training, testing)
            
def runUI():
    global job_queue

    job_queue = Queue()

    if not st.session_state["queue"]:
        queue_thread = Thread(target=queue_listener)
        add_script_run_ctx(queue_thread)
        queue_thread.start()
        st.session_state["queue"] = True
    
    file_ = open("imgs/logo.png", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(f"""
        <div style='text-align: center;'>
            <img src="data:image/gif;base64,{data_url}" alt="logo" width="400">
            <h5 style="color:gray">Empowering researchers with machine learning</h5>
        </div>
    """, unsafe_allow_html=True)

    st.info("""**BioAutoML-FAST**, a **F**eature-based **A**utomated **S**ys**T**em, is an advanced web server implementation of
     BioAutoML, optimized for speed and enhanced functionality. It allows
     users to input their sequences or structured data for classification, 
     generating models that can be saved for future use. The application 
     features a repository of around 50 trained models for various problems,
     such as cancer, COVID-19, and other diseases. By automating feature extraction, 
     selection, and algorithm tuning, BioAutoML-FAST makes powerful machine 
     learning tools accessible to researchers, biologists, and physicians, 
     even those with limited ML expertise. This user-friendly interface 
     supports innovative solutions to critical health challenges, advancing 
     the field of bioinformatics and enabling the scientific community to 
     develop new treatments and interventions, ultimately improving health 
     outcomes and benefiting society.""")

    st.divider()

    st.markdown("""##### Classification""", unsafe_allow_html=True)

    queue_info = st.container()

    col1, col2, col3 = st.columns(3)

    with col1:
        training = st.selectbox(":brain: Training", ["Training set", "Load model"],
                                    help="Training set evaluated with 10-fold cross-validation") #index=None
    with col2:
        testing = st.selectbox(":mag_right: Testing", ["No test set", "Test set", "Prediction set"],
                                    help="Test set ")
    with col3:
        data_type = st.selectbox(":dna: Data type", ["DNA/RNA", "Protein", "Structured data"], 
                                    help="Only sequences without ambiguous nucleotides or amino acids are supported") #index=None

    with st.form("sequences_submit", clear_on_submit=True):
        if training == "Training set":
            if testing == "No test set":
                if data_type == "Structured data":
                    train_files = st.file_uploader("Training set file", accept_multiple_files=False, help='CSV file with the column "label" to indicate the row labels.')
                else:
                    train_files = st.file_uploader("Training set FASTA files", accept_multiple_files=True, help="Separated by class (e.g. sRNA.fasta, tRNA.fasta)")
            elif testing == "Test set":
                set1, set2 = st.columns(2)

                with set1:
                    if data_type == "Structured data":
                        train_files = st.file_uploader("Training set file", accept_multiple_files=False, help='CSV file with the column "label" to indicate the row labels.')
                    else:
                        train_files = st.file_uploader("Training set FASTA files", accept_multiple_files=True, help="Separated by class (e.g. sRNA.fasta, tRNA.fasta)")
                with set2:
                    if data_type == "Structured data":
                        test_files = st.file_uploader("Test set file", accept_multiple_files=False, help='CSV file with the column "label" to indicate the row labels.')
                    else:
                        test_files = st.file_uploader("Test set FASTA files", accept_multiple_files=True, help="Separated by class (e.g. sRNA.fasta, tRNA.fasta)")
            elif testing == "Prediction set":
                set1, set2 = st.columns(2)

                with set1:
                    if data_type == "Structured data":
                        train_files = st.file_uploader("Training set file", accept_multiple_files=False, help='CSV file with the column "label" to indicate the row labels.')
                    else:
                        train_files = st.file_uploader("Training set FASTA files", accept_multiple_files=True, help="Separated by class (e.g. sRNA.fasta, tRNA.fasta)")
                with set2:
                    if data_type == "Structured data":
                        test_files = st.file_uploader("Test set file", accept_multiple_files=False, help='CSV file without column to indicate row labels.')
                    else:
                        test_files = st.file_uploader("FASTA file for prediction", accept_multiple_files=False, help="Single file for prediction (e.g. predict.fasta)")
        else:
            if testing == "No test set":
                # st.warning("You need a set for using against the trained model.")
                train_files = st.file_uploader("Trained model file", accept_multiple_files=False, help="Only models generated by BioAutoML-FAST are accepted (e.g. trained_model.sav)")
            elif testing == "Test set":
                set1, set2 = st.columns(2)

                with set1:
                    train_files = st.file_uploader("Trained model file", accept_multiple_files=False, help="Only models generated by BioAutoML-FAST are accepted (e.g. trained_model.sav)")
                with set2:
                    if data_type == "Structured data":
                        test_files = st.file_uploader("Test set file", accept_multiple_files=False, help='CSV file with the column "label" to indicate the row labels.')
                    else:
                        test_files = st.file_uploader("Test set FASTA files", accept_multiple_files=True, help="Separated by class (e.g. sRNA.fasta, tRNA.fasta)")
            elif testing == "Prediction set":
                set1, set2 = st.columns(2)

                with set1:
                    train_files = st.file_uploader("Trained model file", accept_multiple_files=False, help="Only models generated by BioAutoML-FAST are accepted (e.g. trained_model.sav)")
                with set2:
                    if data_type == "Structured data":
                        test_files = st.file_uploader("Test set file", accept_multiple_files=False, help='CSV file without column to indicate row labels.')
                    else:
                        test_files = st.file_uploader("FASTA file for prediction", accept_multiple_files=False, help="Single file for prediction (e.g. predict.fasta)")

        submitted = st.form_submit_button("Submit", use_container_width=True, type="primary")

    predict_path = os.path.abspath("jobs")

    if submitted:
        if train_files is None:
            st.error("Please upload the required training file(s).")
        elif (testing != "No test set" and not test_files):
            st.error("Please upload the required test or prediction file(s).")
        else:
            job_id = ''.join([choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16)])
            job_path = os.path.join(predict_path, job_id)

            os.makedirs(job_path)
            if testing == "No test set":
                job_queue.put((train_files, None, job_path, data_type, training, testing))
            else:
                job_queue.put((train_files, test_files, job_path, data_type, training, testing))

            with queue_info:
                st.success(f"Job submitted to the queue. You can consult the results in \"Jobs\" using the following ID: **{job_id}**")