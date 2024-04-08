import streamlit as st
import polars as pl
import pandas as pd
from io import StringIO
from Bio import SeqIO
import subprocess
import streamlit.components.v1 as components
import os
from secrets import choice
import string
from queue import Queue
from threading import Thread
from streamlit.runtime.scriptrunner import add_script_run_ctx
import utils

def submit_job(train_files, test_files, job_path, seq_type):
    train_path = os.path.join(job_path, 'train')
    os.makedirs(train_path)

    for file in train_files:
        save_path = os.path.join(train_path, file.name)
        with open(save_path, mode='wb') as f:
            f.write(file.getvalue())

    train_fasta = {os.path.splitext(f)[0] : os.path.join(train_path, f) for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, f))}
    
    command = [
        "python",
        "BioAutoML-BigData-DNA.py" if seq_type == "DNA/RNA" else "BioAutoML-feature-protein.py",
        "--fasta_train",
    ]

    command.extend(train_fasta.values())
    command.append("--fasta_label_train")
    command.extend(train_fasta.keys())

    if test_files:
        test_path = os.path.join(job_path, 'test')
        os.makedirs(test_path)

        for file in test_files:
            save_path = os.path.join(test_path, file.name)
            with open(save_path, mode='wb') as f:
                f.write(file.getvalue())

        test_fasta = {os.path.splitext(f)[0] : os.path.join(test_path, f) for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))}

        command.append("--fasta_test")
        command.extend(test_fasta.values())
        command.append("--fasta_label_test")
        command.extend(test_fasta.keys())

    command.extend(["--n_cpu", "-1"])
    command.extend(["--output", job_path])

    subprocess.run(command, cwd="..")
    # with open(os.path.join(job_path, "predict.fasta"), "w") as f: 
    #     for record in SeqIO.parse(fasta_sequences, "fasta"):
    #         f.write(record.format("fasta"))

    # subprocess.run(["python", "../Classification/main.py", "--test", job_path] +
    #                 ["--path_model", "../Classification/results/enc1_cnn_bilstm_4conv_k1_concat2_bio/model.h5"] +
    #                 ["--encoding", "1", "--k", "1", "--concat", "1", "--feat_extraction", "1", "--features_exist", "1", "--output", job_path])

def queue_listener():
    while True:
        if not job_queue.empty():
            train_files, test_files, job_path, seq_type = job_queue.get()
            submit_job(train_files, test_files, job_path, seq_type)
            
def runUI():
    global job_queue

    job_queue = Queue()

    if not st.session_state["queue"]:
        queue_thread = Thread(target=queue_listener)
        add_script_run_ctx(queue_thread)
        queue_thread.start()
        st.session_state["queue"] = True

    img_cols = st.columns([3, 2, 3])

    with img_cols[1]:
        st.image("imgs/logo.png")

    st.markdown("""
        <div style='text-align: center;'>
            <h5 style="color:gray">Democratizing Machine Learning in Life Sciences</h5>
        </div>
    """, unsafe_allow_html=True)

    st.info("""BioAutoML is ...""")

    st.divider()

    st.markdown("""##### Classification""", unsafe_allow_html=True)

    queue_info = st.container()

    col1, col2 = st.columns(2)

    with col1:
        evaluation = st.selectbox(":mag_right: Dataset", ["Training set", "Training and test set"],
                                    help="Training set evaluated with 10-fold cross-validation") #index=None
    with col2:
        seq_type = st.selectbox(":dna: Sequence type", ["DNA/RNA", "Protein"], 
                                    help="Only sequences without ambiguous nucleotides or amino acids are supported") #index=None

    with st.form("sequences_submit", clear_on_submit=True):
        
        if evaluation == "Training and test set":
            set1, set2 = st.columns(2)

            with set1:
                train_files = st.file_uploader("Training set FASTA files", accept_multiple_files=True, help="Separated by class (e.g. sRNA.fasta, tRNA.fasta)")

            with set2:
                test_files = st.file_uploader("Test set FASTA files", accept_multiple_files=True, help="Separated by class (e.g. sRNA.fasta, tRNA.fasta)")
        else:
            train_files = st.file_uploader("Training set FASTA files", accept_multiple_files=True, help="Separated by class (e.g. sRNA.fasta, tRNA.fasta)")

        submitted = st.form_submit_button("Submit")

    predict_path = os.path.abspath("jobs")

    if submitted:
        job_id = ''.join([choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16)])
        job_path = os.path.join(predict_path, job_id)

        os.makedirs(job_path)
        if evaluation == "Training and test set":
            job_queue.put((train_files, test_files, job_path, seq_type))
            train_files, test_files
        else:
            job_queue.put((train_files, None, job_path, seq_type))

        with queue_info:
            st.success(f"Job submitted to the queue. You can consult the results in \"Jobs\" using the following ID: **{job_id}**")
        
    #     if fasta_text:
    #         job_id = ''.join([choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16)])
    #         job_path = os.path.join(predict_path, job_id)
    #         fasta_sequences = StringIO(fasta_text)

    #         os.makedirs(job_path)
    #         job_queue.put((fasta_sequences, job_path))
    #         st.success(f"Job submitted to the queue. You can consult the results in \"Jobs\" using the following ID: **{job_id}**")

    #     elif fasta_file:
    #         job_id = ''.join([choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16)])
    #         job_path = os.path.join(predict_path, job_id)
    #         fasta_sequences = StringIO(fasta_file.getvalue().decode("utf-8"))

    #         os.makedirs(job_path)
    #         job_queue.put((fasta_sequences, job_path))
    #         st.success(f"Job submitted to the queue. You can consult the results in \"Jobs\" using the following ID: **{job_id}**")
    #     else:
    #         st.error("No sequences submitted!")