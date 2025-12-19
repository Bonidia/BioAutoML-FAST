import streamlit as st
import polars as pl
import pandas as pd
import string
from secrets import choice
import subprocess
from subprocess import Popen
import bibtexparser
import utils
import base64
import joblib
import shutil
import os
import time
from utils import tasks
from rq import get_current_job
from utils.tasks import manager
from utils.db import TaskResultManager, TaskStatus
import re
import tarfile
import tempfile
import io
import secrets
import base64
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet

def test_extraction(job_path, test_data, model, data_type):
    datasets = []

    path = os.path.join(job_path, "feat_extraction", "test")
    feat_path = os.path.join(job_path, "feat_extraction")

    try:
        shutil.rmtree(path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    print("Creating Directory...")
    os.makedirs(path)

    if data_type == "DNA/RNA":
        for label in test_data:
            subprocess.run(["python", "other-methods/preprocessing.py",
            "-i", test_data[label],
            "-d", data_type,
            "-o", os.path.join(path, f"pre_{label}.fasta")],
            cwd="..", stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
            )

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
                        ["python", "other-methods/EntropyClass.py", "-i",
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
    elif data_type == "Protein":
        for label in test_data:
            subprocess.run(["python", "other-methods/preprocessing.py",
            "-i", test_data[label],
            "-d", data_type,
            "-o", os.path.join(path, f"pre_{label}.fasta")],
            cwd="..", stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
            )

            datasets.append(feat_path + "/Shannon.csv")
            datasets.append(feat_path + "/Tsallis_23.csv")
            datasets.append(feat_path + "/Tsallis_30.csv")
            datasets.append(feat_path + "/Tsallis_40.csv")
            datasets.append(feat_path + "/ComplexNetworks.csv")
            datasets.append(feat_path + "/kGap_di.csv")
            datasets.append(feat_path + "/AAC.csv")
            datasets.append(feat_path + "/DPC.csv")
            datasets.append(feat_path + "/iFeature-features.csv")
            datasets.append(feat_path + "/Global.csv")
            datasets.append(feat_path + "/Peptide.csv")
            
            commands = [["python", "other-methods/EntropyClass.py",
                                "-i", os.path.join(path, f"pre_{label}.fasta"), "-o", feat_path + "/Shannon.csv", "-l", label,
                                "-k", "5", "-e", "Shannon"],
                        ["python", "other-methods/TsallisEntropy.py",
                                "-i", os.path.join(path, f"pre_{label}.fasta"), "-o", feat_path + "/Tsallis_23.csv", "-l", label,
                                "-k", "5", "-q", "2.3"],
                        ["python", "other-methods/TsallisEntropy.py",
                                "-i", os.path.join(path, f"pre_{label}.fasta"), "-o", feat_path + "/Tsallis_30.csv", "-l", label,
                                "-k", "5", "-q", "3.0"],
                        ["python", "other-methods/TsallisEntropy.py",
                                "-i", os.path.join(path, f"pre_{label}.fasta"), "-o", feat_path + "/Tsallis_40.csv", "-l", label,
                                "-k", "5", "-q", "4.0"],
                        ["python", "MathFeature/methods/ComplexNetworksClass-v2.py", "-i",
                                os.path.join(path, f"pre_{label}.fasta"), "-o", feat_path + "/ComplexNetworks.csv", "-l", label,
                                "-k", "3"],
                        ["python", "MathFeature/methods/Kgap.py", "-i",
                                os.path.join(path, f"pre_{label}.fasta"), "-o", feat_path + "/kGap_di.csv", "-l",
                                label, "-k", "1", "-bef", "1",
                                "-aft", "1", "-seq", "3"],
                        ["python", "other-methods/ExtractionTechniques-Protein.py", "-i",
                                os.path.join(path, f"pre_{label}.fasta"), "-o", feat_path + "/AAC.csv", "-l", label,
                                "-t", "AAC"],
                        ["python", "other-methods/ExtractionTechniques-Protein.py", "-i",
                                os.path.join(path, f"pre_{label}.fasta"), "-o", feat_path + "/DPC.csv", "-l", label,
                                "-t", "DPC"],
                        ["python", "other-methods/iFeature-modified/iFeature.py", "--file",
                                os.path.join(path, f"pre_{label}.fasta"), "--type", "All", "--label", label, 
                                "--out", feat_path + "/iFeature-features.csv"],
                        ["python", "other-methods/modlAMP-modified/descriptors.py", "-option",
                                "global", "-label", label, "-input", os.path.join(path, f"pre_{label}.fasta"), 
                                "-output", feat_path + "/Global.csv"],
                        ["python", "other-methods/modlAMP-modified/descriptors.py", "-option",
                                "peptide", "-label", label, "-input", os.path.join(path, f"pre_{label}.fasta"), 
                                "-output", feat_path + "/Peptide.csv"],
            ]

            processes = [Popen(cmd, cwd="..", stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT) for cmd in commands]
            for p in processes: p.wait()

        text_input = ''
        for label in test_data:
            text_input += os.path.join(path, f"pre_{label}.fasta") + '\n' + label + '\n'

        dataset = feat_path + '/Fourier_Integer.csv'

        subprocess.run(['python', 'MathFeature/methods/Mappings-Protein.py',
                        '-n', str(len(test_data)), '-o',
                        dataset, '-r', '6'], cwd="..", text=True, input=text_input,
                        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        with open(dataset, 'r') as temp_f:
            col_count = [len(l.split(",")) for l in temp_f.readlines()]

        colnames = ['Integer_Fourier_' + str(i) for i in range(0, max(col_count))]

        df = pd.read_csv(dataset, names=colnames, header=0)
        df.rename(columns={df.columns[0]: 'nameseq', df.columns[-1]: 'label'}, inplace=True)
        df.to_csv(dataset, index=False)
        datasets.append(dataset)

        dataset = feat_path + '/Fourier_EIIP.csv'

        subprocess.run(['python', 'MathFeature/methods/Mappings-Protein.py',
                        '-n', str(len(test_data)), '-o',
                        dataset, '-r', '8'], cwd="..", text=True, input=text_input,
                        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        with open(dataset, 'r') as temp_f:
            col_count = [len(l.split(",")) for l in temp_f.readlines()]

        colnames = ['EIIP_Fourier_' + str(i) for i in range(0, max(col_count))]

        df = pd.read_csv(dataset, names=colnames, header=0)
        df.rename(columns={df.columns[0]: 'nameseq', df.columns[-1]: 'label'}, inplace=True)
        df.to_csv(dataset, index=False)
        datasets.append(dataset)

        # dataset = feat_path + '/EIIP.csv'

        # subprocess.run(['python', 'MathFeature/methods/Mappings-Protein.py',
        #                 '-n', str(len(test_data)), '-o',
        #                 dataset, '-r', '7'], cwd="..", text=True, input=text_input,
        #                 stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        # with open(dataset, 'r') as temp_f:
        #     col_count = [len(l.split(",")) for l in temp_f.readlines()]

        # colnames = ['EIIP_' + str(i) for i in range(0, max(col_count))]

        # df = pd.read_csv(dataset, names=colnames, header=None)
        # df.rename(columns={df.columns[0]: 'nameseq', df.columns[-1]: 'label'}, inplace=True)
        # df.to_csv(dataset, index=False)
        # datasets.append(dataset)

        # dataset = feat_path + '/AAAF.csv'

        # subprocess.run(['python', 'MathFeature/methods/Mappings-Protein.py',
        #                 '-n', str(len(test_data)), '-o',
        #                 dataset, '-r', '1'], cwd="..", text=True, input=text_input,
        #                 stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        # with open(dataset, 'r') as temp_f:
        #     col_count = [len(l.split(",")) for l in temp_f.readlines()]

        # colnames = ['AccumulatedFrequency_' + str(i) for i in range(0, max(col_count))]

        # df = pd.read_csv(dataset, names=colnames, header=None)
        # df.rename(columns={df.columns[0]: 'nameseq', df.columns[-1]: 'label'}, inplace=True)
        # df.to_csv(dataset, index=False)
        # datasets.append(dataset)

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

    df_train = model["train"]

    common_columns = dataframes.columns.intersection(df_train.columns)
    df_predict = dataframes[common_columns]

    df_predict.to_csv(os.path.join(path_bio, "best_test.csv"), index=False)

# Derive a URL-safe Base64 key for Fernet from a password + salt
def derive_key_from_password(password: str, salt: bytes, iterations: int = 390000) -> bytes:
    password_bytes = password.encode("utf-8")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
    return key

# Create a tar archive in memory from a directory path and return bytes
def make_tar_bytes_from_dir(folder_path: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # Add all files and subdirectories
        tar.add(folder_path, arcname=".")
    buf.seek(0)
    return buf.read()

def encrypt_job_folder(job_path: str, password: str) -> None:
    # 1) Create tar.gz bytes of folder
    tar_bytes = make_tar_bytes_from_dir(job_path)

    # 2) Generate salt and derive key
    salt = secrets.token_bytes(16)
    key = derive_key_from_password(password, salt)
    fernet = Fernet(key)

    # 3) Encrypt the tar bytes
    encrypted = fernet.encrypt(tar_bytes)

    # 4) Write encrypted archive and salt into job_path
    enc_path = os.path.join(job_path, "job_archive.enc")
    salt_path = os.path.join(job_path, "job_salt.bin")

    with open(enc_path, "wb") as f:
        f.write(encrypted)
    with open(salt_path, "wb") as f:
        f.write(salt)

    # 5) Remove everything else in job_path except the newly created files
    for root, dirs, files in os.walk(job_path):
        for name in files:
            full = os.path.join(root, name)
            if full not in {enc_path, salt_path}:
                try:
                    os.remove(full)
                except Exception:
                    pass
        # remove empty directories (except the top job_path)
        for d in dirs:
            dirfull = os.path.join(root, d)
            try:
                # attempt rmdir (will only remove if empty)
                os.rmdir(dirfull)
            except Exception:
                pass

def submit_job(dataset_path, test_files, predict_path, data_type, training, testing, email=None, password=None):
    """Process a single job - modified to be thread-safe."""

    job = get_current_job()
    job_id = job.get_id()
    manager.store_result(job_id, TaskStatus.RUNNING)

    job_path = os.path.join(predict_path, job_id)
    os.makedirs(job_path, exist_ok=True)

    try:
        if training == "Load model":
            save_path = os.path.join(dataset_path, "trained_model.sav")
            link_path = os.path.join(job_path, "trained_model.sav")

            model = joblib.load(save_path)

            # Create symbolic link
            os.symlink(save_path, link_path)

            if "label_encoder" in model:
                task = "Classification"
                command = [
                    "python",
                    "BioAutoML-multiclass.py" if len(model["label_encoder"].classes_) > 2 else "BioAutoML-binary.py",
                    "-path_model", save_path,
                    "-nf", "True",
                ]
            else:
                task = "Regression"
                command = [
                    "python",
                    "BioAutoML-regression.py",
                    "-path_model", save_path,
                    "-nf", "True",
                ]

            
            if test_files:
                data_type = "Structured data"

                if "descriptors" in model:
                    df_descriptors = model["descriptors"]

                    if "NAC" in df_descriptors.columns:
                        data_type = "DNA/RNA"
                    else:
                        data_type = "Protein"

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
                    test_path = os.path.join(job_path, "test")
                    os.makedirs(test_path)

                    if testing == "Test set":
                        if task == "Classification":
                            for file in test_files:
                                save_path = os.path.join(test_path, file.name)
                                with open(save_path, mode="wb") as f:
                                    f.write(file.getvalue())
                        elif task == "Regression":
                            for file in test_files:
                                save_path = os.path.join(test_path, file.name)
                                with open(save_path, mode="wb") as f:
                                    f.write(file.getvalue())
                            # save_path = os.path.join(test_path, test_files.name)
                            # with open(save_path, mode="wb") as f:
                            #     f.write(test_files.getvalue())

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

        try:
            if password:
                encrypt_job_folder(job_path, password)
        except Exception as e:
            print(f"Error encrypting job {job_id}: {e}")
    except Exception as e:
        print(f"Error in job processing: {e}")

@st.cache_resource
def bibtex_to_dict(bib_file="references.bib"):
    bib_path = os.path.join(os.path.abspath("datasets"), bib_file)
    with open(bib_path) as bibtex_file:
        bib = bibtexparser.load(bibtex_file)

    citation_dict = {}

    for entry in bib.entries:
        key = entry.get("ID")
        author = entry.get("author", "").replace("\n", " ")
        title = entry.get("title", "")
        journal = entry.get("journal", entry.get("booktitle", ""))
        year = entry.get("year", "")
        doi = entry.get("doi", "")

        citation = f"{author} ({year}). *{title}*. {journal}. {doi}"

        citation_dict[key] = citation

    return citation_dict

@st.dialog("Job submitted")
def job_submitted_dialog(job_id):
    st.success(
        f'Job submitted to the queue.\n\n'
        f'You can consult the results in **Jobs** using the following ID:\n\n'
        f'**{job_id}**'
    )

def count_fasta_sequences(uploaded_files):
    """Counts total FASTA records across one or more uploaded files."""
    if not uploaded_files:
        return 0

    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    total = 0
    for f in uploaded_files:
        f.seek(0)
        for line in f:
            if line.startswith(b">"):
                total += 1
        f.seek(0)

    return total

def runUI():

    with st.expander("Predicting new data"):
        st.info("""
            Here you can **apply one of 60 curated, trained models** to perform **classification or regression** on biological sequences, without training a model from scratch.  
            These models cover a wide range of tasks, including peptide activity prediction, RNA annotation, protein function identification, and regulatory sequence analysis.

            Simply select a model from the repository and upload a **FASTA file for prediction**. Each model is linked to a **published dataset**, and the corresponding references are shown to ensure transparency and reproducibility.
            
            **Important limits:** You can upload at most **5,000 prediction sequences** per job.    

            The **Examples button** provides concrete submission examples to help you get started.

            Jobs are executed asynchronously and queued for processing. Once completed, results can be accessed in the **Jobs** module using the generated job ID. Optional email notification and submission encryption are available.
            """
        )

    MAX_SEQS = 5_000

    _, excol2 = st.columns([9, 1])

    with excol2:
        zip_path = "repo_examples.zip"
        with open(zip_path, "rb") as f:
            st.download_button(
                label="Examples",
                data=f,
                file_name="repo_examples.zip",
                mime="application/zip",
                use_container_width=True
            )

    citation_dict = bibtex_to_dict()

    models = [
        "Model 1: antibody sequences",
        "Model 2: anticancer peptides",
        "Model 3: anticancer peptides",
        "Model 4: anticancer peptides",
        "Model 5: anticancer peptides",
        "Model 6: anti-coronavirus",
        "Model 7: anti-coronavirus and random non-secretory proteins",
        "Model 8: antifungal peptides",
        "Model 9: anti-hypertensive peptides",
        "Model 10: antimalarial peptides",
        "Model 11: antimalarial peptides",
        "Model 12: antimicrobial peptides",
        "Model 13: antimicrobial peptides",
        "Model 14: antimicrobial peptides",
        "Model 15: anti-MRSA strains peptides",
        "Model 16: antioxidant proteins",
        "Model 17: anti-parasitic peptides",
        "Model 18: antiviral",
        "Model 19: antiviral and random non-secretory proteins",
        "Model 20: antiviral peptides",
        "Model 21: bitter peptides",
        "Model 22: blood-brain barrier peptides",
        "Model 23: DNA-binding proteins",
        "Model 24: DNA-binding proteins",
        "Model 25: DNase I hypersensitive sites",
        "Model 26: DPP IV inhibitory peptides",
        "Model 27: hemolytic peptides",
        "Model 28: identification of hotspots",
        "Model 29: lncRNA (Homo sapiens)",
        "Model 30: lncRNA (Mus musculus)",
        "Model 31: lncRNA (Triticum aestivum)",
        "Model 32: lncRNA (Zea mays)",
        "Model 33: m5C sites (Arabidopsis thaliana)",
        "Model 34: m5C sites (Homo sapiens)",
        "Model 35: m5C sites (Mus musculus)",
        "Model 36: m5C sites (Saccharomyces cerevisiae)",
        "Model 37: neuropeptides",
        "Model 38: non-classical secreted proteins",
        "Model 39: peptide toxicity",
        "Model 40: phage virion proteins",
        "Model 41: proinflammatory peptides",
        "Model 42: protein lysine crotonylation sites",
        "Model 43: quorum-sensing peptides",
        "Model 44: real microRNA precursors",
        "Model 45: ribosome binding site sequences",
        "Model 46: sigma70 promoters",
        "Model 47: small non-coding RNA and shuffled sequences",
        "Model 48: toehold switch sequences",
        "Model 49: Tumor T cell antigens",
        "Model 50: umami peptides",
        "Model 51: lncRNAs subcellular localization - training set with 4 classes",
        "Model 52: lncRNAs subcellular localization - training set with 5 classes",
        "Model 53: mRNA subcellular localization",
        "Model 54: non-coding RNA - 4 classes",
        "Model 55: non-coding RNA - 8 classes",
        "Model 56: non-coding RNA - E. coli K12",
        "Model 57: non-coding RNA - Multiple bacterial phyla",
        "Model 58: antibody sequences",
        "Model 59: ribosome binding site sequences",
        "Model 60: toehold switch sequences"
    ]

    datasets = [
        "dataset1_liu_protein_0",
        "dataset2_yu_protein_0",
        "dataset3_li_protein_0",
        "dataset4_charoenkwan_protein_0",
        "dataset5_agrawal_protein_0",
        "dataset6_timmons_protein_0",
        "dataset7_timmons_protein_0",
        "dataset8_pinacho_protein_0",
        "dataset9_manavalan_protein_0",
        "dataset10_charoenkwan_protein_0",
        "dataset11_charoenkwan_protein_0",
        "dataset12_chung_protein_0",
        "dataset13_xiao_protein_0",
        "dataset14_pang_protein_0",
        "dataset15_charoenkwan_protein_0",
        "dataset16_lam_protein_0",
        "dataset17_zhang_protein_0",
        "dataset18_timmons_protein_0",
        "dataset19_timmons_protein_0",
        "dataset20_pinacho_protein_0",
        "dataset21_charoenkwan_protein_0",
        "dataset22_dai_protein_0",
        "dataset23_chowdhury_protein_0",
        "dataset24_li_protein_0",
        "dataset25_liu_dnarna_0",
        "dataset26_charoenkwan_protein_0",
        "dataset27_chaudhary_protein_0",
        "dataset28_khan_dnarna_0",
        "dataset29_han_dnarna_0",
        "dataset30_han_dnarna_0",
        "dataset31_han_dnarna_0",
        "dataset32_meng_dnarna_0",
        "dataset33_lv_dnarna_0",
        "dataset34_lv_dnarna_0",
        "dataset35_lv_dnarna_0",
        "dataset36_lv_dnarna_0",
        "dataset37_bin_protein_0",
        "dataset38_zhang_protein_0",
        "dataset39_wei_protein_0",
        "dataset40_charoenkwan_protein_0",
        "dataset41_khatun_protein_0",
        "dataset42_zhao_protein_0",
        "dataset43_wei_protein_0",
        "dataset44_liu_dnarna_0",
        "dataset45_hoellerer_dnarna_0",
        "dataset46_lin_dnarna_0",
        "dataset47_barman_dnarna_0",
        "dataset48_valeri_dnarna_0",
        "dataset49_charoenkwan_protein_0",
        "dataset50_charoenkwan_protein_0",
        "dataset51_cai_dnarna_0",
        "dataset52_cai_dnarna_0",
        "dataset53_musleh_dnarna_0",
        "dataset54_avila_dnarna_0",
        "dataset55_bonidia_dnarna_0",
        "dataset56_bonidia_dnarna_0",
        "dataset57_bonidia_dnarna_0",
        "dataset58_liu_protein_1",
        "dataset59_hoellerer_dnarna_1",
        "dataset60_valeri_dnarna_1"
    ]

    cites = [
        "liu2020antibody",
        "wei2018acpred",
        "hajisharifi2014predicting",
        "agrawal2021anticp",
        "agrawal2021anticp",
        "timmons2021ennavia",
        "timmons2021ennavia",
        "pinacho2021alignment",
        "manavalan2019mahtpred",
        "charoenkwan2022iamap",
        "charoenkwan2022iamap",
        "chung2020characterization",
        "xiao2013iamp",
        "pang2022integrating",
        "charoenkwan2022scmrsa",
        "zhang2016sequence, butt2019prediction",
        "zhang2022predapp",
        "timmons2021ennavia",
        "timmons2021ennavia",
        "pinacho2021alignment",
        "charoenkwan2021bert4bitter",
        "dai2021bbppred",
        "liu2014idna, lou2014sequence",
        "liu2015psedna",
        "noble2005predicting",
        "charoenkwan2020idppiv",
        "chaudhary2016web",
        "jiang2007rf, khan2020prediction",
        "han2019lncfinder",
        "han2019lncfinder",
        "han2019lncfinder",
        "meng2021plncrna",
        "lv2020evaluation",
        "feng2016identifying",
        "lv2020evaluation",
        "lv2020evaluation",
        "chen2022neuropred, bin2020prediction",
        "zhang2020pengaroo",
        "wei2021atse",
        "charoenkwan2020meta",
        "khatun2020proin",
        "zhao2020identification",
        "wei2020comparative",
        "liu2015identification",
        "hollerer2020large",
        "lin2017identifying",
        "barman2017improved",
        "valeri2020sequence",
        "charoenkwan2020ittca",
        "charoenkwan2020iumami",
        "cai2023gm",
        "cai2023gm",
        "musleh2023mslp",
        "avila2024biodeepfuse",
        "bonidia2022bioautoml",
        "bonidia2022bioautoml",
        "bonidia2022bioautoml",
        "liu2020antibody",
        "hollerer2020large",
        "valeri2020sequence"
    ]

    # Build mapping dict programmatically to avoid 60 if/else branches
    model_map = {
        models[i]: {"dataset": datasets[i], "cite": cites[i]}
        for i in range(len(models))
    }

    model = st.selectbox("Select trained model", models)

    # Use mapping to set dataset_id and show info
    if model in model_map:
        dataset_id = model_map[model]["dataset"]
        cite_key = model_map[model]["cite"]

        df_train_stats = pd.read_csv(os.path.join("datasets", dataset_id, "runs/run_6/train_stats.csv"))

        task = int(dataset_id.split('_')[-1])

        # Split into individual cite keys
        keys = [k.strip() for k in cite_key.split(",")]

        # Retrieve each citation (or a not-found warning)
        citation_text = " ; ".join(
            citation_dict.get(k, f"[Citation not found for key {k}]")
            for k in keys
        )

        st.info(
            f"""
            **Model chosen:** {model}

            **Task:** {'Classification' if task == 0 else 'Regression'}
  
            **Data type:** {'Nucleotide' if "gc_content" in df_train_stats.columns else 'Amino acid'}

            **Possible labels:** {", ".join(df_train_stats["class"].tolist())}

            **Dataset from the following paper(s):** {citation_text}

            You can consult experiments done with this dataset in "Jobs" using the following ID: **{dataset_id}**
            """
        )
    else:
        st.warning("Selected model not found in mapping. Please contact the admin.")
        
    job_id = ""

    queue_info = st.container()

    with st.form("repo_submit", clear_on_submit=True):
        test_files = st.file_uploader("FASTA file for prediction", 
                                    accept_multiple_files=False, 
                                    help="Single file for prediction (e.g. predict.fasta)")

        col1, col2 = st.columns(2)
    
        with col1:
            email = st.text_input("Email to notify when job finishes (Optional)", help="We will send a completion notification to this address.")

            # Simple validation (not strict): show warning if looks invalid
            if email:
                if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                    st.warning("That doesn't look like a valid email address.")
        with col2:
            password = st.text_input("Password to encrypt submission (Optional)", type='password', help="Only with this password can the job be accessed. Not even the administrators can view encrypted submissions.")

        submitted = st.form_submit_button("Submit", 
                                    use_container_width=True, 
                                    type="primary")

    predict_path = os.path.abspath("jobs")

    if submitted:
        if not test_files:
            with queue_info:
                st.error("Please upload the required prediction file.")
        else:
            prediction_seq_count = count_fasta_sequences(test_files)
            if prediction_seq_count > MAX_SEQS:
                with queue_info:
                    st.error(
                        f"Prediction set exceeds the maximum allowed size "
                        f"({prediction_seq_count:,} sequences uploaded, limit is {MAX_SEQS})."
                    )
                st.stop()

            training = "Load model"
            testing = "Prediction set"
            classifier, imbalance = False, False

            dtype_str = dataset_id.split('_')[-2]

            if dtype_str == "protein":
                data_type = "Protein"
            elif dtype_str == "dnarna":
                data_type = "DNA/RNA"

            dataset_path = os.path.join(os.path.abspath("datasets"), dataset_id, "runs/run_6")

            fn_kwargs = {
                "dataset_path":  dataset_path,
                "test_files": test_files,
                "predict_path":  predict_path,
                "data_type": data_type,
                "training":  training,
                "testing":   testing,
                "email":     email,
                "password":  password
            }
                
            # Add job to the queue
            job_id = tasks.enqueue_task(submit_job, fn_kwargs=fn_kwargs)
            job_path = os.path.join(predict_path, job_id)

            os.makedirs(job_path, exist_ok=True)

            task = int(dataset_id.split('_')[-1])

            job_data = {
                "data_type": [data_type],
                "task": ["Classification" if task == 0 else "Regression"],
                "training_set": [training == "Training set"],
                "testing_set": [testing],
                "classifier_selected": [classifier], 
                "imbalance_methods": [imbalance]
            }

            df_job_data = pl.DataFrame(job_data)
            tsv_path = os.path.join(job_path, "job_info.tsv")
            df_job_data.write_csv(tsv_path, separator='\t')

            job_submitted_dialog(job_id)

# Run the Streamlit app
if __name__ == "__main__":
    runUI()