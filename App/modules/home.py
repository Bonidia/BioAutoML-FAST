import streamlit as st
import polars as pl
import pandas as pd
from io import StringIO
from Bio import SeqIO
import subprocess
from subprocess import Popen
import streamlit.components.v1 as components
import os
import csv
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

def submit_job(train_files, test_files, predict_path, data_type, task, training, testing, tuning, email=None, password=None):
    """Process a single job - modified to be thread-safe."""

    job = get_current_job()
    job_id = job.get_id()
    manager.store_start(job_id, TaskStatus.RUNNING)

    job_path = os.path.join(predict_path, job_id)
    os.makedirs(job_path, exist_ok=True)

    log_path = os.path.join(job_path, "subprocess.log")

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
                
                if task == "Regression":
                    df_train = df_train.with_columns(
                        pl.concat_str(["nameseq", "label"], separator="|").alias("nameseq")
                    )
                    
                    df_train = df_train.with_columns(pl.lit(train_files.name.split(".csv")[0]).alias("label"))

                df_labels = df_train.select(["label"])
                df_index = df_train.select(["nameseq"])
                df_train = df_train.drop(["nameseq", "label"])

                feat_path = os.path.join(job_path, "feat_extraction")
                os.makedirs(feat_path)
                
                df_train.write_csv(os.path.join(feat_path, "train.csv"))
                df_labels.write_csv(os.path.join(feat_path, "train_labels.csv"))
                df_index.write_csv(os.path.join(feat_path, "fnameseqtrain.csv"))

                if task == "Regression":
                    df_train = pl.read_csv(save_path).with_columns(pl.lit(train_files.name.split(".csv")[0]).alias("label"))
                    df_train.write_csv(save_path)

                command = [
                    "python",
                    "generation.py",
                    "--task",
                    "1" if task == "Regression" else "0",
                    "--train", os.path.join(feat_path, "train.csv"),
                    "--train_label", os.path.join(feat_path, "train_labels.csv"),
                    "--train_nameseq", os.path.join(feat_path, "fnameseqtrain.csv"),
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
                        
                        if task == "Regression":
                            df_test = df_test.with_columns(
                                pl.concat_str(["nameseq", "label"], separator="|").alias("nameseq")
                            )
                            
                            df_test = df_test.with_columns(pl.lit(test_files.name.split(".csv")[0]).alias("label"))

                        df_labels = df_test.select(["label"])
                        df_index = df_test.select(["nameseq"])
                        df_test = df_test.drop(["nameseq", "label"])

                        df_index.write_csv(os.path.join(feat_path, "fnameseqtest.csv"))
                        df_test.write_csv(os.path.join(feat_path, "test.csv"))
                        df_labels.write_csv(os.path.join(feat_path, "test_labels.csv"))

                        if task == "Regression":
                            df_test = pl.read_csv(save_path).with_columns(pl.lit(test_files.name.split(".csv")[0]).alias("label"))
                            df_test.write_csv(save_path)
                            
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

                        df_test = pl.read_csv(save_path).with_columns(pl.lit("Predicted").alias("label"))
                        df_test.write_csv(save_path)

                        command.append("--test")
                        command.append(os.path.join(feat_path, "test.csv"))
                        command.append("--test_label")
                        command.append(os.path.join(feat_path, "test_labels.csv"))
                        command.append("--test_nameseq")
                        command.append(os.path.join(feat_path, "fnameseqtest.csv"))

                command.extend(["--n_cpu", "-1"])
                command.extend(["--output", job_path])

                with open(log_path, "w") as log_file:
                    subprocess.run(command, cwd="..", stdout=log_file, stderr=subprocess.STDOUT, text=True, check=True)

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
                    "engineering.py",
                    "--dtype", 
                    data_type,
                    "--task",
                    "1" if task == "Regression" else "0",
                    "--tuning",
                    "150" if tuning else "0",
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

                with open(log_path, "w") as log_file:
                    subprocess.run(command, cwd="..", stdout=log_file, stderr=subprocess.STDOUT, text=True, check=True)

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

            command = [
                "python",
                "generation.py", 
                "--task",
                "0" if "label_encoder" in model else "1",
                "-path_model", save_path,
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
                            f.write(test_files[0].getvalue())
                        
                        df_test = pl.from_pandas(pd.read_csv(save_path).reset_index())
                        df_test = df_test.rename({"index": "nameseq"})

                        if "label_encoder" not in model:
                            df_test = df_test.with_columns(
                                pl.concat_str(["nameseq", "label"], separator="|").alias("nameseq")
                            )
                            
                            df_test = df_test.with_columns(pl.lit(test_files[0].name.split(".csv")[0]).alias("label"))

                        df_labels = df_test.select(["label"])
                        df_index = df_test.select(["nameseq"])
                        df_test = df_test.drop(["nameseq", "label"])

                        df_index.write_csv(os.path.join(feat_path, "fnameseqtest.csv"))
                        df_test.write_csv(os.path.join(feat_path, "test.csv"))
                        df_labels.write_csv(os.path.join(feat_path, "test_labels.csv"))

                        if "label_encoder" not in model:
                            df_test = pl.read_csv(save_path).with_columns(pl.lit(test_files[0].name.split(".csv")[0]).alias("label"))
                            df_test.write_csv(save_path)
                        
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

                        df_test = pl.read_csv(save_path).with_columns(pl.lit("Predicted").alias("label"))
                        df_test.write_csv(save_path)

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
                        if "label_encoder" in model:
                            for file in test_files:
                                save_path = os.path.join(test_path, file.name)
                                with open(save_path, mode="wb") as f:
                                    f.write(file.getvalue())
                        else:
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

            with open(log_path, "w") as log_file:
                subprocess.run(command, cwd="..", stdout=log_file, stderr=subprocess.STDOUT, text=True, check=True)
        try:
            if password:
                encrypt_job_folder(job_path, password)
        except Exception as e:
            print(f"Error encrypting job {job_id}: {e}")
    except Exception as e:
        print(f"Error in job processing: {e}")

@st.dialog("Job submitted")
def job_submitted_dialog(job_id):
    st.success(
        f'Job submitted to the queue.\n\n'
        f'You can consult the results in **Jobs** using the following link:\n\n'
        f'**https://bioautoml.icmc.usp.br/?id={job_id}**\n\n'
        f'Save this link securely.'
    )
    st.markdown("**Job ID**")
    st.code(job_id, language=None)

def count_samples(uploaded_files, data_type):
    """Counts total records across one or more uploaded files."""
    if not uploaded_files:
        return 0

    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    total = 0
    if data_type != "Structured data":
        for f in uploaded_files:
            f.seek(0)
            for line in f:
                if line.startswith(b">"):
                    total += 1
            f.seek(0)
    else:
        for f in uploaded_files:
            f.seek(0)
            for line in f:
                total += 1
            total -= 1
            f.seek(0)

    return total

import csv

def check_structured_data(uploaded_files, task):
    """
    Check whether each uploaded CSV file:
    1. Contains a column named 'label'
    2. The 'label' column contains only numerical values
    
    Returns:
        True if all conditions are satisfied for all files
        False otherwise
    """
    if not uploaded_files:
        return None

    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    for f in uploaded_files:
        f.seek(0)

        try:
            reader = csv.reader(
                (line.decode("utf-8") if isinstance(line, bytes) else line)
                for line in f
            )

            header = next(reader)

            if "label" not in header:
                return False

            label_idx = header.index("label")

            for row in reader:
                # Skip empty rows
                if not row or len(row) <= label_idx:
                    return False

                value = row[label_idx].strip()

                # Reject empty values
                if value == "":
                    return False

                if task == "Regression":
                    # Check numeric (int or float)
                    try:
                        float(value)
                    except ValueError:
                        return False

        except Exception:
            return False

    return True

# Valid IUPAC nucleotide codes (DNA + RNA, including ambiguous)
_FASTA_NT_CHARS = frozenset("ACGTURYSWKMBDHVNacgturyswkmbdhvn")
# Valid IUPAC amino acid single-letter codes (including ambiguous/special)
_FASTA_AA_CHARS = frozenset("ACDEFGHIKLMNPQRSTVWYUOBZXJacdefghiklmnpqrstvwyuobzxj*-")

def validate_fasta(uploaded_file, data_type, task=None):
    """
    Validates a FASTA file's format and biological content.
    Returns (is_valid, error_message_or_None).
    """
    name = uploaded_file.name

    uploaded_file.seek(0)
    raw = uploaded_file.read()
    uploaded_file.seek(0)

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1")
        except Exception:
            return False, f"**{name}**: could not be read as a text file. Make sure it is a plain-text FASTA file."

    lines = text.splitlines()
    non_empty = [l for l in lines if l.strip()]

    if not non_empty:
        return False, f"**{name}**: file is empty."

    if not non_empty[0].startswith(">"):
        return False, (
            f"**{name}**: invalid FASTA format — the file must start with a header line "
            f"beginning with '>'. Make sure you are uploading a FASTA file and not a CSV or other format."
        )

    valid_chars = _FASTA_NT_CHARS if data_type == "DNA/RNA" else _FASTA_AA_CHARS

    seq_count = 0
    current_header = None
    header_line_no = 0
    has_sequence = False

    for line_no, raw_line in enumerate(lines, 1):
        line = raw_line.rstrip()
        if not line:
            continue

        if line.startswith(">"):
            if current_header is not None and not has_sequence:
                return False, (
                    f"**{name}**: header '>{current_header}' at line {header_line_no} "
                    f"has no sequence data."
                )

            header = line[1:].strip()
            if not header:
                return False, f"**{name}**: empty sequence header at line {line_no}."

            if task == "Regression":
                if "|" not in header:
                    return False, (
                        f"**{name}**: header '>{header}' at line {line_no} is missing the "
                        f"required '|value' label. Regression FASTA headers must follow the "
                        f"format '>sequence_name|numeric_value' (e.g. '>seq1|0.85')."
                    )
                value_str = header.rsplit("|", 1)[-1].strip()
                try:
                    float(value_str)
                except ValueError:
                    return False, (
                        f"**{name}**: header '>{header}' at line {line_no} has "
                        f"'|{value_str}' which is not a valid number."
                    )

            current_header = header
            header_line_no = line_no
            has_sequence = False
            seq_count += 1

        else:
            if current_header is None:
                return False, (
                    f"**{name}**: sequence data at line {line_no} appears before any header."
                )

            invalid = set(line) - valid_chars
            if invalid:
                inv_str = ", ".join(f"'{c}'" for c in sorted(invalid))
                dtype_label = "DNA/RNA" if data_type == "DNA/RNA" else "Protein"
                return False, (
                    f"**{name}**: unexpected character(s) {inv_str} in sequence at line {line_no} "
                    f"— not valid for {dtype_label}."
                )

            has_sequence = True

    if current_header is not None and not has_sequence:
        return False, (
            f"**{name}**: last header '>{current_header}' (line {header_line_no}) "
            f"has no sequence data."
        )

    if seq_count == 0:
        return False, f"**{name}**: no sequences found in file."

    return True, None

class InternalUploadedFile:
    def __init__(self, path: str):
        self.path = path
        self.name = os.path.basename(path)

        with open(path, "rb") as f:
            self._data = f.read()

    def getvalue(self):
        return self._data

    def read(self):
        return self._data

def runUI():
    """Main Streamlit UI function with thread management."""

    with open("imgs/logo.png", "rb") as file_:
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")

    st.markdown(f"""
        <div style='text-align: center;'>
            <img src="data:image/png;base64,{data_url}" alt="logo" width="400">
            <p class="hero-subtitle">Empowering Breakthroughs in Life Sciences with End-to-End Machine Learning</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    _cards = [
        ("🧬", "Multi-Omics Ready", "Nucleotide, amino acid, and structured biological data — all supported out of the box"),
        ("🤖", "End-to-End AutoML", "Train classifiers or regressors with automatic feature extraction and model selection"),
        ("📊", "Deep Explainability", "SHAP values, confusion matrices, feature distributions, and 3D dimensionality reduction"),
        ("🗂️", "60+ Trained Models", "Ready-to-use models for anticancer peptides, non-coding RNAs, taste prediction, and more"),
    ]
    _c1, _c2, _c3, _c4 = st.columns(4)
    for _col, (_icon, _title, _text) in zip([_c1, _c2, _c3, _c4], _cards):
        with _col:
            st.markdown(
                f'<div class="feature-card">'
                f'<span class="feature-icon">{_icon}</span>'
                f'<div class="feature-title">{_title}</div>'
                f'<div class="feature-text">{_text}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    with st.expander("Preparing your submission"):
        st.info("""
            Here you can **train a new model or load an existing model** to perform **classification or regression** on biological sequences.  
            You may optionally evaluate the model using a **labeled test set** or apply it to **unlabeled data for prediction**.

            **Important limits:** You can upload at most **5,000 training sequences** or **5,000 testing/prediction sequences** per job.

            Each option and file uploader includes a **tooltip** with instructions about the **required file formats, labels, and submission rules**.
                
            The **Examples button** provides concrete submission examples to help you get started.

            Jobs are executed asynchronously and queued for processing. Once completed, results can be accessed in the **Jobs** module using the generated job ID. Optional email notification and submission encryption are available.
            """
        )

    MAX_SEQS = 5_000

    queue_info = st.container()

    _, excol2, excol3 = st.columns([7, 1.2, 1.2])

    with excol2:
        sample_data = st.toggle("Use example", help="Use example data instead of submitted files.")

    with excol3:
        zip_path = "examples/home_examples.zip"
        with open(zip_path, "rb") as f:
            st.download_button(
                label="Examples",
                data=f,
                file_name="home_examples.zip",
                mime="application/zip",
                use_container_width=True,
                help="Download examples"
            )

    st.markdown('<div class="section-label">Configuration</div>', unsafe_allow_html=True)

    tuning = False  # default; overridden by the checkbox widget below when applicable

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
        "Structured data": "Structured data",
    }

    with col2:
        testing = st.selectbox(":mag_right: Testing", ["No test set", "Test set", "Prediction set"],
                                help="Whether to use a labeled testing set to evaluate the model, or alternatively, an unlabeled prediction set.")
        
        if training == "Training set":
            data_type_label = st.selectbox(":dna: Data type", list(data_type_map.keys()),
                                    help="Any sequence that includes ambiguous nucleotides or amino acids will be preprocessed, with all ambiguous characters removed.")

            data_type = data_type_map[data_type_label]
        else:
            data_type = None

    if training == "Training set" and data_type != "Structured data":
        checkcol1, checkcol2, checkcol3 = st.columns(3)

        with checkcol1:
            tuning = st.checkbox("Hyperparameter tuning", help="Whether to use hyperparameter tuning for the model (this can make the training take longer).")
        
        with checkcol2:
            email = st.text_input("Email to notify when job finishes (Optional)", help="We will send a completion notification to this address.")

            # Simple validation (not strict): show warning if looks invalid
            if email:
                if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                    st.warning("That doesn't look like a valid email address.")

        with checkcol3:
            password = st.text_input("Password to encrypt submission (Optional)", type='password', help="Only with this password can the job be accessed. Not even the administrators can view encrypted submissions.")
    elif training == "Training set" and data_type == "Structured data":
        checkcol1, checkcol2 = st.columns(2)

        with checkcol1:
            email = st.text_input("Email to notify when job finishes (Optional)", help="We will send a completion notification to this address.")

            # Simple validation (not strict): show warning if looks invalid
            if email:
                if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                    st.warning("That doesn't look like a valid email address.")

        with checkcol2:
            password = st.text_input("Password to encrypt submission (Optional)", type='password', help="Only with this password can the job be accessed. Not even the administrators can view encrypted submissions.")
    elif training == "Load model":
        tuning = False

        checkcol1, checkcol2 = st.columns(2)

        with checkcol1:
            email = st.text_input("Email to notify when job finishes (Optional)", help="We will send a completion notification to this address.")

            # Simple validation (not strict): show warning if looks invalid
            if email:
                if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                    st.warning("That doesn't look like a valid email address.")
        
        with checkcol2:
            password = st.text_input("Password to encrypt submission (Optional)", type='password', help="Only with this password can the job be accessed. Not even the administrators can view encrypted submissions.")

    # ── Configuration summary ──────────────────────────────────────────────
    _summary = []
    if training == "Training set":
        _summary.append(f"🎯 <b>Mode:</b> Training a new model")
        if task:
            _summary.append(f"📋 <b>Task:</b> {task}")
        if data_type:
            _summary.append(f"🧬 <b>Data type:</b> {data_type_label}")
        if data_type and data_type != "Structured data":
            _summary.append(f"⚙️ <b>Tuning:</b> {'Yes' if tuning else 'No'}")
    else:
        _summary.append("🎯 <b>Mode:</b> Loading an existing model")
    _summary.append(f"🔬 <b>Validation:</b> {testing}")
    if email:
        _summary.append(f"📧 <b>Notify:</b> {email}")
    if password:
        _summary.append(" 🔒 <b>Encrypted</b>")
    st.markdown(
        '<div class="config-summary-card">'
        '<strong class="card-title">Current Configuration</strong>'
        + " &nbsp;·&nbsp; ".join(_summary)
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Upload files</div>', unsafe_allow_html=True)

    with st.form("sequences_submit", clear_on_submit=True):
        if training == "Training set":
            if testing == "No test set":
                if data_type == "Structured data":
                    train_files = st.file_uploader("Training set CSV file", accept_multiple_files=False, help='CSV file with the column "label" to indicate the row labels.')
                else:
                    if task == "Classification":
                        train_files = st.file_uploader("Training set FASTA files", accept_multiple_files=True, 
                                                       help="Separated by class (e.g. sRNA.fasta, rRNA.fasta, tRNA.fasta). Upload one FASTA file per class. If it is only two classes, name them as positive.fasta and negative.fasta.")
                    elif task == "Regression":
                        train_files = st.file_uploader("Training set FASTA file", accept_multiple_files=False, 
                                                       help="Single FASTA file with continuous target values appended to the end of the headers after the | character.")
            elif testing == "Test set":
                set1, set2 = st.columns(2)

                with set1:
                    if data_type == "Structured data":
                        train_files = st.file_uploader("Training set CSV file", accept_multiple_files=False, help='CSV file with the column "label" to indicate the row labels.')
                    else:
                        if task == "Classification":
                            train_files = st.file_uploader("Training set FASTA files", accept_multiple_files=True, 
                                                        help="Separated by class (e.g. sRNA.fasta, rRNA.fasta, tRNA.fasta). Upload one FASTA file per class. If it is only two classes, name them as positive.fasta and negative.fasta.")
                        elif task == "Regression":
                            train_files = st.file_uploader("Training set FASTA file", accept_multiple_files=False, 
                                                        help="Single FASTA file with continuous target values appended to the end of the headers after the | character.")
                with set2:
                    if data_type == "Structured data":
                        test_files = st.file_uploader("Test set CSV file", accept_multiple_files=False, help='CSV file with the column "label" to indicate the row labels.')
                    else:
                        if task == "Classification":
                            test_files = st.file_uploader("Test set FASTA files", accept_multiple_files=True, 
                                                        help="Separated by class (e.g. sRNA.fasta, rRNA.fasta, tRNA.fasta). Upload one FASTA file per class. If it is only two classes, name them as positive.fasta and negative.fasta.")
                        elif task == "Regression":
                            test_files = st.file_uploader("Test set FASTA file", accept_multiple_files=False, 
                                                        help="Single FASTA file with continuous target values appended to the end of the headers after the | character.")
            elif testing == "Prediction set":
                set1, set2 = st.columns(2)

                with set1:
                    if data_type == "Structured data":
                        train_files = st.file_uploader("Training set CSV file", accept_multiple_files=False, help='CSV file with the column "label" to indicate the row labels.')
                    else:
                        if task == "Classification":
                            train_files = st.file_uploader("Training set FASTA files", accept_multiple_files=True, 
                                                        help="Separated by class (e.g. sRNA.fasta, rRNA.fasta, tRNA.fasta). Upload one FASTA file per class. If it is only two classes, name them as positive.fasta and negative.fasta.")
                        elif task == "Regression":
                            train_files = st.file_uploader("Training set FASTA file", accept_multiple_files=False, 
                                                        help="Single FASTA file with continuous target values appended to the end of the headers after the | character.")
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
        if sample_data:
            train_files = InternalUploadedFile("examples/home_examples/classification/aminoacid/trained_model.sav")
            test_files = [
                InternalUploadedFile("examples/home_examples/classification/aminoacid/test/positive.fasta"),
                InternalUploadedFile("examples/home_examples/classification/aminoacid/test/negative.fasta"),
            ]
            training = "Load model"
            testing = "Test set"
            data_type = "Nucleotide"
            task = "Classification"
            tuning = False
        else:
            # For non-structured sequence classification, require >= 2 class files
            if task and data_type != "Structured data":
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

                if training == "Training set" and task == "Regression":
                    if not train_files:
                        with queue_info:
                            st.error("Training set (Regression) requires one FASTA file.")
                        st.stop()

                if testing == "Test set" and task == "Regression":
                    if not test_files:
                        with queue_info:
                            st.error("Test set (Regression) requires one FASTA file.")
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
            
            if training == "Load model":
                if not train_files:
                    with queue_info:
                        st.error("Please provide the trained model file.")
                    st.stop()

            # Test/prediction files required unless "No test set"
            if testing != "No test set" and not test_files:
                with queue_info:
                    st.error("Please upload the required test or prediction file(s).")
                st.stop()

            if testing == "No test set":
                test_files = None

            if training == "Training set":
                train_seq_count = count_samples(train_files, data_type)
                if train_seq_count > MAX_SEQS:
                    with queue_info:
                        st.error(
                            f"Training set exceeds the maximum allowed size "
                            f"({train_seq_count:,} samples uploaded, limit is {MAX_SEQS})."
                        )
                    st.stop()

            if testing in ["Test set", "Prediction set"]:
                test_seq_count = count_samples(test_files, data_type)
                if test_seq_count > MAX_SEQS:
                    with queue_info:
                        st.error(
                            f"Testing/Prediction set exceeds the maximum allowed size "
                            f"({test_seq_count:,} samples uploaded, limit is {MAX_SEQS})."
                        )
                    st.stop()

            # FASTA format and content validation
            if training == "Training set" and data_type in ("DNA/RNA", "Protein"):
                _regression_task = task if task == "Regression" else None

                # Duplicate class filename check for classification
                if task == "Classification" and isinstance(train_files, list):
                    _train_names = [f.name for f in train_files]
                    if len(_train_names) != len(set(_train_names)):
                        with queue_info:
                            st.error(
                                "Training set has duplicate file names. "
                                "Each class must have a unique filename."
                            )
                        st.stop()
                if task == "Classification" and testing == "Test set" and isinstance(test_files, list):
                    _test_names = [f.name for f in test_files]
                    if len(_test_names) != len(set(_test_names)):
                        with queue_info:
                            st.error(
                                "Test set has duplicate file names. "
                                "Each class must have a unique filename."
                            )
                        st.stop()

                # Validate each training FASTA file
                _train_list = train_files if isinstance(train_files, list) else [train_files]
                for _f in _train_list:
                    _ok, _err = validate_fasta(_f, data_type, _regression_task)
                    if not _ok:
                        with queue_info:
                            st.error(_err)
                        st.stop()

                # Validate test/prediction FASTA files
                if testing == "Test set":
                    _test_list = test_files if isinstance(test_files, list) else [test_files]
                    for _f in _test_list:
                        _ok, _err = validate_fasta(_f, data_type, _regression_task)
                        if not _ok:
                            with queue_info:
                                st.error(_err)
                            st.stop()
                elif testing == "Prediction set":
                    _ok, _err = validate_fasta(test_files, data_type, task=None)
                    if not _ok:
                        with queue_info:
                            st.error(_err)
                        st.stop()

            if data_type == "Structured data":
                if training == "Training set":
                    if not check_structured_data(train_files, task):
                        with queue_info:
                            st.error(
                                "Training set doesn't have a column named 'label' with numerical values only."
                            )
                        st.stop()

                if testing == "Test set":
                    if not check_structured_data(test_files, task):
                        with queue_info:
                            st.error(
                                "Test set doesn't have a column named 'label' with numerical values only."
                            )
                        st.stop()

                tuning = True

        fn_kwargs = {
            "train_files": train_files,
            "test_files": test_files,
            "predict_path":  predict_path,
            "data_type": data_type,
            "task":      task,
            "training":  training,
            "testing":   testing,
            "tuning": tuning,
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
            "tuning": [tuning],
        }

        df_job_data = pl.DataFrame(job_data)
        tsv_path = os.path.join(job_path, "job_info.tsv")
        df_job_data.write_csv(tsv_path, separator='\t')

        job_submitted_dialog(job_id)

if __name__ == "__main__":
    runUI()
