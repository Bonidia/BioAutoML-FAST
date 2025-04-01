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
from threading import Thread, Lock
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
import utils
import base64
import joblib
import shutil
import time

# Global variables for thread management
job_queue = Queue()
queue_lock = Lock()
queue_thread = None

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
    elif data_type == "Protein":
        for label in test_data:
            subprocess.run(["python", "other-methods/preprocessing.py",
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
            
            commands = [["python", "MathFeature/methods/EntropyClass.py",
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

def worker():
    """Worker thread that processes jobs from the queue."""
    ctx = get_script_run_ctx()
    if ctx is None:
        return
    
    while True:
        with queue_lock:
            if not job_queue.empty():
                job_data = job_queue.get()
                
                try:
                    # Unpack job data
                    (train_files, test_files, job_path, data_type, 
                     training, testing, classifier, imbalance, fselection) = job_data
                    
                    # Process the job
                    submit_job(train_files, test_files, job_path, data_type, 
                              training, testing, classifier, imbalance, fselection)
                
                except Exception as e:
                    print(f"Error processing job: {e}")
                
                finally:
                    job_queue.task_done()
        
        time.sleep(1)  # Prevent busy waiting

def submit_job(train_files, test_files, job_path, data_type, training, testing, classifier, imbalance, fselection):
    """Process a single job - modified to be thread-safe."""
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
                    "--fselection",
                    "1" if fselection else "0",
                    "--train", os.path.join(feat_path, "train.csv"),
                    "--train_label", os.path.join(feat_path, "train_labels.csv"),
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
                        
                        df_test = pd.read_csv(save_path).reset_index()
                        df_test["label"] = "Predicted"
                        df_test = pl.from_pandas(df_test)
                        df_index = df_test.select(["index"])
                        df_labels = df_test.select(["label"])
                        df_test = df_test.drop(["index", "label"])

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
                for file in train_files:
                    save_path = os.path.join(train_path, file.name)
                    with open(save_path, mode="wb") as f:
                        f.write(file.getvalue())
                
                train_fasta = {os.path.splitext(f)[0] : os.path.join(train_path, f) for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, f))}
            
                command = [
                    "python",
                    "BioAutoML-feature.py" if data_type == "DNA/RNA" else "BioAutoML-protein.py",
                    "--imbalance",
                    "1" if imbalance else "0",
                    "--fselection",
                    "1" if fselection else "0",
                    "--fasta_train",
                ]

                command.extend(train_fasta.values())
                command.append("--fasta_label_train")
                command.extend(train_fasta.keys())

                if test_files:
                    test_path = os.path.join(job_path, "test")
                    os.makedirs(test_path)

                    if testing == "Test set":
                        for file in test_files:
                            save_path = os.path.join(test_path, file.name)
                            with open(save_path, mode="wb") as f:
                                f.write(file.getvalue())

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

    except Exception as e:
        print(f"Error in job processing: {e}")

def runUI():
    """Main Streamlit UI function with thread management."""
    global job_queue, queue_thread
    
    if "queue_started" in st.session_state:
        st.session_state.queue_started = False
        
    # Start the worker thread if not already running
    if not st.session_state.queue_started:
        queue_thread = Thread(target=worker, daemon=True)
        add_script_run_ctx(queue_thread)
        queue_thread.start()
        st.session_state.queue_started = True
    
    with open("imgs/logo.png", "rb") as file_:
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")

    st.markdown(f"""
        <div style='text-align: center;'>
            <img src="data:image/png;base64,{data_url}" alt="logo" width="400">
            <h5 style="color:gray">Empowering Researchers with Machine Learning</h5>
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

    st.markdown("""##### Prediction""", unsafe_allow_html=True)

    queue_info = st.container()

    col1, col2, col3 = st.columns(3)

    with col1:
        training = st.selectbox(":brain: Training", ["Training set", "Load model"],
                                    help="Training set evaluated with 10-fold cross-validation.")
    with col2:
        testing = st.selectbox(":mag_right: Testing", ["No test set", "Test set", "Prediction set"],
                                    help="Whether to use a labeled testing set to evaluate the model, or alternatively, an unlabeled prediction set.")
    with col3:
        data_type = st.selectbox(":dna: Data type", ["DNA/RNA", "Protein", "Structured data"], 
                                    help="Only sequences without ambiguous nucleotides or amino acids are supported, as well as structured data with categorical variables.")
    
    if training == "Training set":
        _, checkcol1, checkcol2, _ = st.columns([2, 3, 3, 2])

        with checkcol1:
            fselection = st.checkbox("Feature Selection", help="Whether to use feature selection methods.")
        with checkcol2:
            imbalance = st.checkbox("Oversampling/Undersampling", help="Whether to use imbalanced techniques for the data sets.")

    if training == "Training set" and data_type == "Structured data":
        classifier = st.selectbox(":wrench: Algorithm for structured data", ["Random Forest", "XGBoost", "LightGBM", "CatBoost"],
                                    help="Algorithm to be used for prediction.")

    with st.form("sequences_submit", clear_on_submit=True):
        if training == "Training set":
            if testing == "No test set":
                if data_type == "Structured data":
                    train_files = st.file_uploader("Training set CSV file", accept_multiple_files=False, help='CSV file with the column "label" to indicate the row labels.')
                else:
                    train_files = st.file_uploader("Training set FASTA files", accept_multiple_files=True, help="Separated by class (e.g. sRNA.fasta, tRNA.fasta)")
            elif testing == "Test set":
                set1, set2 = st.columns(2)

                with set1:
                    if data_type == "Structured data":
                        train_files = st.file_uploader("Training set CSV file", accept_multiple_files=False, help='CSV file with the column "label" to indicate the row labels.')
                    else:
                        train_files = st.file_uploader("Training set FASTA files", accept_multiple_files=True, help="Separated by class (e.g. sRNA.fasta, tRNA.fasta)")
                with set2:
                    if data_type == "Structured data":
                        test_files = st.file_uploader("Test set CSV file", accept_multiple_files=False, help='CSV file with the column "label" to indicate the row labels.')
                    else:
                        test_files = st.file_uploader("Test set FASTA files", accept_multiple_files=True, help="Separated by class (e.g. sRNA.fasta, tRNA.fasta)")
            elif testing == "Prediction set":
                set1, set2 = st.columns(2)

                with set1:
                    if data_type == "Structured data":
                        train_files = st.file_uploader("Training set CSV file", accept_multiple_files=False, help='CSV file with the column "label" to indicate the row labels.')
                    else:
                        train_files = st.file_uploader("Training set FASTA files", accept_multiple_files=True, help="Separated by class (e.g. sRNA.fasta, tRNA.fasta)")
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
                    if data_type == "Structured data":
                        test_files = st.file_uploader("Test set CSV file", accept_multiple_files=False, help='CSV file with the column "label" to indicate the row labels.')
                    else:
                        test_files = st.file_uploader("Test set FASTA files", accept_multiple_files=True, help="Separated by class (e.g. sRNA.fasta, tRNA.fasta)")
            elif testing == "Prediction set":
                set1, set2 = st.columns(2)

                with set1:
                    train_files = st.file_uploader("Trained model file", accept_multiple_files=False, help="Only models generated by BioAutoML-FAST are accepted (e.g. trained_model.sav)")
                with set2:
                    if data_type == "Structured data":
                        test_files = st.file_uploader("CSV file for prediction", accept_multiple_files=False, help='CSV file without column to indicate row labels.')
                    else:
                        test_files = st.file_uploader("FASTA file for prediction", accept_multiple_files=False, help="Single file for prediction (e.g. predict.fasta)")

        submitted = st.form_submit_button("Submit", use_container_width=True, type="primary")

    predict_path = os.path.abspath("jobs")

    if submitted:
        if training == "Training set" and data_type != "Structured data" and len(train_files) < 2:
            with queue_info:
                st.error("Training set requires at least 2 classes.")
        if training == "Training set" and data_type == "Structured data" and train_files == None:
            with queue_info:
                st.error("Training set requires 1 file with the column for labels.")
        elif testing != "No test set" and not test_files:
            with queue_info:
                st.error("Please upload the required test or prediction file(s).")
        elif testing == "Test set" and data_type != "Structured data" and len(test_files) < 2:
            with queue_info:
                st.error("Test set requires at least 2 classes.")
        elif testing == "Test set" and data_type == "Structured data" and test_files == None:
            with queue_info:
                st.error("Test set requires 1 file with the column for labels.")
        else:
            job_id = ''.join([choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16)])
            job_path = os.path.join(predict_path, job_id)

            os.makedirs(job_path)

            if training == "Load model":
                classifier, imbalance, fselection = False, False, False
            elif training == "Training set" and data_type != "Structured data":
                classifier = False

            if testing == "No test set":
                test_files = None

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

            # Add job to the queue with thread safety
            with queue_lock:
                job_queue.put((train_files, test_files, job_path, data_type, training, testing, classifier, imbalance, fselection))
            
            with queue_info:
                st.success(f"Job submitted to the queue. You can consult the results in \"Jobs\" using the following ID: **{job_id}**")

# Run the Streamlit app
if __name__ == "__main__":
    runUI()