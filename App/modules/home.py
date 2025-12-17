import streamlit as st
import polars as pl
import pandas as pd
from io import StringIO
from Bio import SeqIO
import subprocess
from subprocess import Popen
import streamlit.components.v1 as components
import os
import string
import utils
import base64
import joblib
import shutil
import time
import re
from pathlib import Path
from functools import partial
from utils import tasks
from rq import get_current_job
from utils.tasks import manager
from utils.db import TaskResultManager, TaskStatus
import tarfile
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
                        "-d", "DNA/RNA",
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
            "-d", "Protein",
            "-i", test_data[label], 
            "-o", os.path.join(path, f"pre_{label}.fasta")],
            cwd="..",
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

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

# Encrypt the entire job folder into job_archive.enc and save salt (job_salt.bin).
# Removes the original files once encrypted.
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

def submit_job(train_files, test_files, predict_path, data_type, task, training, testing, classifier, imbalance, email=None, password=None):
    """Process a single job - modified to be thread-safe."""

    job = get_current_job()
    job_id = job.get_id()
    manager.store_result(job_id, TaskStatus.RUNNING)

    job_path = os.path.join(predict_path, job_id)
    os.makedirs(job_path, exist_ok=True)

    try:
        if training == "Training set":
            train_path = os.path.join(job_path, "train")
            os.makedirs(train_path)

            if data_type == "Structured data":
                save_path = os.path.join(train_path, "train.csv")
                with open(save_path, mode="wb") as f:
                    f.write(train_files.getvalue())

                df_train = pl.from_pandas(pd.read_csv(save_path).reset_index())
                df_train = df_train.rename({"index": "nameseq"})
                df_labels = df_train.select(["label"])
                df_index = df_train.select(["nameseq"])
                df_train = df_train.drop(["nameseq", "label"])

                feat_path = os.path.join(job_path, "feat_extraction")
                os.makedirs(feat_path)
                
                df_train.write_csv(os.path.join(feat_path, "train.csv"))
                df_labels.write_csv(os.path.join(feat_path, "train_labels.csv"))
                df_index.write_csv(os.path.join(feat_path, "fnameseqtrain.csv"))
                
                if classifier == "CatBoost":
                    classifier_option = 0
                elif classifier == "Random Forest":
                    classifier_option = 1
                elif classifier == "LightGBM":
                    classifier_option = 2
                elif classifier == "XGBoost":
                    classifier_option = 3

                command = [
                    "python",
                    "BioAutoML-multiclass.py" if df_labels.n_unique() > 2 else "BioAutoML-binary.py",
                    "--imbalance",
                    "1" if imbalance else "0",
                    "--train", os.path.join(feat_path, "train.csv"),
                    "--train_label", os.path.join(feat_path, "train_labels.csv"),
                    "--train_nameseq", os.path.join(feat_path, "fnameseqtrain.csv"),
                    "--classifier", str(classifier_option),
                    "-nf", "True",
                ]

                if test_files:
                    test_path = os.path.join(job_path, "test")
                    os.makedirs(test_path)

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
                        
                        df_test = pd.read_csv(save_path).reset_index().rename(columns={"index": "nameseq"})
                        df_test["label"] = "Predicted"
                        df_test = pl.from_pandas(df_test)
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

                command.extend(["--n_cpu", "-1"])
                command.extend(["--output", job_path])

                subprocess.run(command, cwd="..")

                utils.summary_stats(os.path.join(job_path, "train"), data_type, job_path, True)

                if test_files:
                    utils.summary_stats(os.path.join(job_path, "test"), data_type, job_path, True)
            
                model = joblib.load(os.path.join(job_path, "trained_model.sav"))
                model["train_stats"] = pd.read_csv(os.path.join(job_path, "train_stats.csv"))
                joblib.dump(model, os.path.join(job_path, "trained_model.sav"))
            else:
                if task == "Classification":
                    for file in train_files:
                        save_path = os.path.join(train_path, file.name)
                        with open(save_path, mode="wb") as f:
                            f.write(file.getvalue())
                elif task == "Regression":
                    save_path = os.path.join(train_path, train_files.name)
                    with open(save_path, mode="wb") as f:
                        f.write(train_files.getvalue())
                
                train_fasta = {os.path.splitext(f)[0] : os.path.join(train_path, f) for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, f))}
            
                command = [
                    "python",
                    "BioAutoML-feature.py" if data_type == "DNA/RNA" else "BioAutoML-protein.py",
                    "--task",
                    "1" if task == "Regression" else "0",
                    "--imbalance",
                    "1" if imbalance else "0",
                    "--fasta_train",
                ]

                command.extend(train_fasta.values())
                command.append("--fasta_label_train")
                command.extend(train_fasta.keys())

                if test_files:
                    test_path = os.path.join(job_path, "test")
                    os.makedirs(test_path)

                    if testing == "Test set":
                        if task == "Classification":
                            for file in test_files:
                                save_path = os.path.join(test_path, file.name)
                                with open(save_path, mode="wb") as f:
                                    f.write(file.getvalue())
                        elif task == "Regression":
                            save_path = os.path.join(test_path, test_files.name)
                            with open(save_path, mode="wb") as f:
                                f.write(test_files.getvalue())
                        
                        test_fasta = {os.path.splitext(f)[0] : os.path.join(test_path, f) for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))}

                        command.append("--fasta_test")
                        command.extend(test_fasta.values())
                        command.append("--fasta_label_test")
                        command.extend(test_fasta.keys())
                    else:
                        save_path = os.path.join(test_path, "predicted.fasta")
                        with open(save_path, mode="wb") as f:
                            f.write(test_files.getvalue())
                        
                        command.append("--fasta_test")
                        command.append(save_path)
                        command.append("--fasta_label_test")
                        command.append("Predicted")

                command.extend(["--n_cpu", "-1"])
                command.extend(["--output", job_path])

                subprocess.run(command, cwd="..")

                utils.summary_stats(os.path.join(job_path, "feat_extraction/train"), data_type, job_path, False)

                if test_files:
                    utils.summary_stats(os.path.join(job_path, "feat_extraction/test"), data_type, job_path, False)
            
                model = joblib.load(os.path.join(job_path, "trained_model.sav"))
                model["train_stats"] = pd.read_csv(os.path.join(job_path, "train_stats.csv"))
                joblib.dump(model, os.path.join(job_path, "trained_model.sav"))

        elif training == "Load model":
            save_path = os.path.join(job_path, "trained_model.sav")
            with open(save_path, mode="wb") as f:
                f.write(train_files.getvalue())

            model = joblib.load(save_path)

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

def runUI():
    """Main Streamlit UI function with thread management."""

    with open("imgs/logo.png", "rb") as file_:
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")

    st.markdown(f"""
        <div style='text-align: center;'>
            <img src="data:image/png;base64,{data_url}" alt="logo" width="400">
            <h5 style="color:gray">Empowering Researchers with Machine Learning</h5>
        </div>
    """, unsafe_allow_html=True)

    st.info("""**BioAutoML-FAST**, a **F**eature-based **A**utomated **S**ys**T**em, is a platform that enables users to upload raw 
            biological sequences and automatically build customised classification models for sequence annotation, or regression 
            models to predict quantitative biological activity, such as expression strength and binding affinity, with optional 
            external validation. The platform summarises datasets through statistical metrics and dimensionality-reduction 
            visualisations. It also includes an extensive repository of 60 pretrained models spanning diverse biological problems, 
            such as anticancer and antimicrobial peptide prediction, non-coding RNA classification, and even taste prediction.""")

    st.divider()

    with st.expander("Preparing your submission"):
        st.info("""
            Here you can **train a new model or load an existing model** to perform **classification or regression** on biological sequences.  
            You may optionally evaluate the model using a **labeled test set** or apply it to **unlabeled data for prediction**.

            Each option and file uploader includes a **tooltip** with instructions about the **required file formats, labels, and submission rules**.
                
            The **Examples button** provides concrete submission examples to help you get started.

            Jobs are executed asynchronously and queued for processing. Once completed, results can be accessed in the **Jobs** module using the generated job ID. Optional email notification and submission encryption are available.
            """
        )

    queue_info = st.container()

    _, excol2 = st.columns([9, 1])

    with excol2:
        zip_path = "home_examples.zip"
        with open(zip_path, "rb") as f:
            st.download_button(
                label="Examples",
                data=f,
                file_name="home_examples.zip",
                mime="application/zip",
                use_container_width=True
            )

    col1, col2 = st.columns(2)

    with col1:
        training = st.selectbox(":brain: Training", ["Training set", "Load model"],
                                help="Training set evaluated with 10-fold cross-validation.")

        if training == "Training set":
            task = st.selectbox(":hammer_and_wrench: Task", ["Classification", "Regression"],
                                help="Choose your machine learning predictive task.")
        else:
            task = None

    data_type_map = {
        "Nucleotide": "DNA/RNA",
        "Amino acid": "Protein",
    }

    with col2:
        testing = st.selectbox(":mag_right: Testing", ["No test set", "Test set", "Prediction set"],
                                help="Whether to use a labeled testing set to evaluate the model, or alternatively, an unlabeled prediction set.")
        
        if training == "Training set":
            data_type_label = st.selectbox(":dna: Data type", list(data_type_map.keys()), # "Structured data" (TO BE DONE)
                                    help="Any sequence that includes ambiguous nucleotides or amino acids will be preprocessed, with all ambiguous characters removed.")

            data_type = data_type_map[data_type_label]
        else:
            data_type = None

    if training == "Training set" and task == "Classification":
        checkcol1, checkcol2, checkcol3 = st.columns(3)

        with checkcol1:
            imbalance = st.checkbox("Oversampling/Undersampling", help="Whether to use imbalanced techniques for the datasets.")
            
        with checkcol2:
            email = st.text_input("Email to notify when job finishes (Optional)", help="We will send a completion notification to this address.")

            # Simple validation (not strict): show warning if looks invalid
            if email:
                if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                    st.warning("That doesn't look like a valid email address.")

        with checkcol3:
            password = st.text_input("Password to encrypt submission (Optional)", type='password', help="Only with this password can the job be accessed. Not even the administrators can view encrypted submissions.")
    elif training == "Load model" or task == "Regression":
        imbalance = False

        checkcol1, checkcol2 = st.columns(2)

        with checkcol1:
            email = st.text_input("Email to notify when job finishes (Optional)", help="We will send a completion notification to this address.")

            # Simple validation (not strict): show warning if looks invalid
            if email:
                if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                    st.warning("That doesn't look like a valid email address.")
        
        with checkcol2:
            password = st.text_input("Password to encrypt submission (Optional)", type='password', help="Only with this password can the job be accessed. Not even the administrators can view encrypted submissions.")

    if training == "Training set" and data_type == "Structured data":
        # Show algorithm choices depending on the selected task
        if task == "Classification":
            classifier = st.selectbox(":wrench: Algorithm for structured data",
                                      ["Random Forest", "XGBoost", "LightGBM", "CatBoost"],
                                      help="Classification algorithms for structured data.")
        elif task == "Regression":
            classifier = st.selectbox(":wrench: Algorithm for structured data",
                                      ["Random Forest Regressor", "XGBoost Regressor", "LightGBM Regressor", "CatBoost Regressor"],
                                      help="Regression algorithms for structured data.")
        else:
            classifier = st.selectbox(":wrench: Algorithm for structured data",
                                      ["Random Forest", "XGBoost", "LightGBM", "CatBoost"],
                                      help="Choose algorithm.")
            
    with st.form("sequences_submit", clear_on_submit=True):
        if training == "Training set":
            if testing == "No test set":
                if data_type == "Structured data":
                    train_files = st.file_uploader("Training set CSV file", accept_multiple_files=False, help='CSV file with the column "label" to indicate the row labels.')
                else:
                    if task == "Classification":
                        train_files = st.file_uploader("Training set FASTA files", accept_multiple_files=True, 
                                                       help="Separated by class (e.g. sRNA.fasta, tRNA.fasta). Upload one FASTA file per class.")
                    elif task == "Regression":
                        train_files = st.file_uploader("Training set FASTA file", accept_multiple_files=False, 
                                                       help="Single FASTA file with continuous target values provided sequence headers.")
            elif testing == "Test set":
                set1, set2 = st.columns(2)

                with set1:
                    if data_type == "Structured data":
                        train_files = st.file_uploader("Training set CSV file", accept_multiple_files=False, help='CSV file with the column "label" to indicate the row labels.')
                    else:
                        if task == "Classification":
                            train_files = st.file_uploader("Training set FASTA files", accept_multiple_files=True, 
                                                        help="Separated by class (e.g. sRNA.fasta, tRNA.fasta). Upload one FASTA file per class.")
                        elif task == "Regression":
                            train_files = st.file_uploader("Training set FASTA file", accept_multiple_files=False, 
                                                        help="Single FASTA file with continuous target values provided sequence headers.")
                with set2:
                    if data_type == "Structured data":
                        test_files = st.file_uploader("Test set CSV file", accept_multiple_files=False, help='CSV file with the column "label" to indicate the row labels.')
                    else:
                        if task == "Classification":
                            test_files = st.file_uploader("Test set FASTA files", accept_multiple_files=True, 
                                                        help="Separated by class (e.g. sRNA.fasta, tRNA.fasta). Upload one FASTA file per class.")
                        elif task == "Regression":
                            test_files = st.file_uploader("Test set FASTA file", accept_multiple_files=False, 
                                                        help="Single FASTA file with continuous target values provided sequence headers.")
            elif testing == "Prediction set":
                set1, set2 = st.columns(2)

                with set1:
                    if data_type == "Structured data":
                        train_files = st.file_uploader("Training set CSV file", accept_multiple_files=False, help='CSV file with the column "label" to indicate the row labels.')
                    else:
                        if task == "Classification":
                            train_files = st.file_uploader("Training set FASTA files", accept_multiple_files=True, 
                                                        help="Separated by class (e.g. sRNA.fasta, tRNA.fasta). Upload one FASTA file per class.")
                        elif task == "Regression":
                            train_files = st.file_uploader("Training set FASTA file", accept_multiple_files=False, 
                                                        help="Single FASTA file with continuous target values provided sequence headers.")
                with set2:
                    if data_type == "Structured data":
                        test_files = st.file_uploader("CSV file for prediction", accept_multiple_files=False, help='CSV file without column to indicate row labels.')
                    else:
                        test_files = st.file_uploader("FASTA file for prediction", accept_multiple_files=False, help="Single file for prediction (e.g. predict.fasta)")
        else:
            if testing == "No test set":
                train_files = st.file_uploader("Trained model file", accept_multiple_files=False, help="Only models generated by BioAutoML-FAST are accepted (e.g. trained_model.sav)")
            elif testing == "Test set":
                set1, set2 = st.columns(2)

                with set1:
                    train_files = st.file_uploader("Trained model file", accept_multiple_files=False, help="Only models generated by BioAutoML-FAST are accepted (e.g. trained_model.sav)")
                with set2:
                    test_files = st.file_uploader("Test set files", accept_multiple_files=True, 
                                                    help="Files accordingly to the loaded model (e.g., files separated by class if classification).")
            elif testing == "Prediction set":
                set1, set2 = st.columns(2)

                with set1:
                    train_files = st.file_uploader("Trained model file", accept_multiple_files=False, help="Only models generated by BioAutoML-FAST are accepted (e.g. trained_model.sav)")
                with set2:
                    test_files = st.file_uploader("Test set files", accept_multiple_files=False, 
                                                    help="File accordingly to the loaded model (e.g., continuous numerical value appended to the end of the header after the | character).")

        submitted = st.form_submit_button("Submit", use_container_width=True, type="primary")

    predict_path = os.path.abspath("jobs")

    if submitted:
        # For non-structured sequence classification, require >= 2 class files
        if task:
            if training == "Training set" and task == "Classification":
                if not train_files or len(train_files) < 2:
                    with queue_info:
                        st.error("Training set (classification) requires at least 2 classes (one FASTA per class).")
                    st.stop()

            # For non-structured sequence test set, require >= 2 class files for classification
            if testing == "Test set" and task == "Classification":
                if not test_files or len(test_files) < 2:
                    with queue_info:
                        st.error("Test set (classification) requires at least 2 classes (one FASTA per class).")
                    st.stop()

        if data_type:
            # For structured data training, require a single CSV for both tasks
            if training == "Training set" and data_type == "Structured data" and train_files is None:
                with queue_info:
                    st.error("Training set requires 1 file with the column for labels (or continuous target for regression).")
                st.stop()

            # For structured data test set, require single CSV
            if testing == "Test set" and data_type == "Structured data" and test_files is None:
                with queue_info:
                    st.error("Test set requires 1 file with the column for labels (or continuous target for regression).")
                st.stop()

        # Test/prediction files required unless "No test set"
        if testing != "No test set" and not test_files:
            with queue_info:
                st.error("Please upload the required test or prediction file(s).")
            st.stop()

        if training == "Training set" or training == "Load model":
            classifier = False

        if testing == "No test set":
            test_files = None

        fn_kwargs = {
            "train_files": train_files,
            "test_files": test_files,
            "predict_path":  predict_path,
            "data_type": data_type,
            "task":      task,
            "training":  training,
            "testing":   testing,
            "classifier":classifier,
            "imbalance": imbalance,
            "email":     email,
            "password":  password
        }

        job_id = tasks.enqueue_task(submit_job, fn_kwargs=fn_kwargs)

        job_path = os.path.join(predict_path, job_id)
        os.makedirs(job_path, exist_ok=True)

        job_data = {
            "data_type": [data_type],
            "task": [task],
            "training_set": [training == "Training set"],
            "testing_set": [testing],
            "classifier_selected": [classifier], 
            "imbalance_methods": [imbalance],
        }

        df_job_data = pl.DataFrame(job_data)
        tsv_path = os.path.join(job_path, "job_info.tsv")
        df_job_data.write_csv(tsv_path, separator='\t')

        with queue_info:
            st.success(f"Job submitted to the queue. You can consult the results in \"Jobs\" using the following ID: **{job_id}**")

if __name__ == "__main__":
    runUI()
