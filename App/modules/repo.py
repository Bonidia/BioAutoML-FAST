import streamlit as st
import polars as pl
import pandas as pd
import string
from secrets import choice
import subprocess
from subprocess import Popen
from queue import Queue
from threading import Thread, Lock
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
import utils
import base64
import joblib
import shutil
import os
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
            subprocess.run(["python", "other-methods/preprocessing.py",
            "-i", test_data[label],
            "--data", data_type,
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
            "--data", data_type,
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

def submit_job(dataset_path, test_files, job_path, data_type, training, testing, classifier, imbalance, fselection):
    """Process a single job - modified to be thread-safe."""
    try:
        if training == "Load model":
            save_path = os.path.join(dataset_path, "trained_model.sav")

            model = joblib.load(save_path)
            joblib.dump(model, os.path.join(job_path, "trained_model.sav"))

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
    
    # Initialize session state for thread management
    if "queue_started" in st.session_state:
        st.session_state.queue_started = False
    
    # Start the worker thread if not already running
    if not st.session_state.queue_started:
        queue_thread = Thread(target=worker, daemon=True)
        add_script_run_ctx(queue_thread)
        queue_thread.start()
        st.session_state.queue_started = True
    
    model = st.selectbox("Select trained model", [
        "Model 1: non-classical secreted proteins",
        "Model 2: phage virion proteins",
        "Model 3: sigma70 promoters",
        "Model 4: anticancer peptides",
        "Model 5: protein lysine crotonylation sites",
        "Model 6: long non-coding RNA - human",
        "Model 7: long non-coding RNA - wheat",
        "Model 8: plant long non-coding RNA",
        "Model 9: 5-methylcytosine sites - H. sapiens",
        "Model 10: 5-methylcytosine sites - M. musculus",
        "Model 11: 5-methylcytosine sites - S. cerevisiae",
        "Model 12: 5-methylcytosine sites - A. thaliana",
        "Model 13: non-coding RNA - E. coli K12",
        "Model 14: non-coding RNA - Multiple bacterial phyla",
        "Model 15: non-coding RNA - 8 classes",
        "Model 16: antimicrobial peptides",
        "Model 17: antiviral",
        "Model 18: antiviral using random sequences",
        "Model 19: anti-coronavirus",
        "Model 20: anti-coronavirus using random sequences",
        "Model 21: antimicrobial peptides",
        "Model 22: antimicrobial peptides",
        "Model 23: anticancer peptides",
        "Model 24: circRNA vs lncRNA",
        "Model 25: mRNA subcellular localization",
        "Model 26: lncRNAs subcellular localization - training set with 5 classes",
        "Model 27: lncRNAs subcellular localization - training set with 4 classes",
        "Model 28: antioxidant proteins",
        "Model 29: proinflammatory peptides",
        "Model 30: recombination spots",
        "Model 31: DNA-binding proteins"
    ])
    job_id = ""
    
    if model == "Model 1: non-classical secreted proteins":
        dataset_id = "dataset1_zhang_protein"
        st.info(f"""
                **Dataset from the following paper:** Zhang, Y., Yu, S., 
                Xie, R., Li, J., Leier, A., Marquez-Lago, T. T., ... & 
                Song, J. (2020). PeNGaRoo, a combined gradient boosting 
                and ensemble learning framework for predicting non-classical 
                secreted proteins. Bioinformatics, 36(3), 704-712.

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 2: phage virion proteins":
        dataset_id = "dataset2_phasit_protein"
        st.info(f"""
                **Dataset from the following paper:** Charoenkwan, P., 
                Nantasenamat, C., Hasan, M. M., & Shoombuatong, W. (2020). 
                Meta-iPVP: a sequence-based meta-predictor for improving 
                the prediction of phage virion proteins using effective 
                feature representation. Journal of Computer-Aided Molecular 
                Design, 34(10), 1105-1116.

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 3: sigma70 promoters":
        dataset_id = "dataset3_lin_dnarna"
        st.info(f"""
                **Dataset from the following paper:** Lin, H., Liang, Z. Y., 
                Tang, H., & Chen, W. (2017). Identifying sigma70 promoters 
                with novel pseudo nucleotide composition. IEEE/ACM transactions 
                on computational biology and bioinformatics, 16(4), 1316-1321.

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 4: anticancer peptides":
        dataset_id = "dataset4_li_protein"
        st.info(f"""
                **Dataset from the following paper:** Li, Q., Zhou, W., Wang, 
                D., Wang, S., & Li, Q. (2020). Prediction of anticancer peptides 
                using a low-dimensional feature model. Frontiers in Bioengineering 
                and Biotechnology, 8, 892.

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 5: protein lysine crotonylation sites":
        dataset_id = "dataset5_zhao_protein"
        st.info(f"""
                **Dataset from the following paper:** Zhao, Y., He, N., Chen, Z., 
                & Li, L. (2020). Identification of protein lysine crotonylation sites 
                by a deep learning framework with convolutional neural networks. Ieee 
                Access, 8, 14244-14252.

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 6: long non-coding RNA - human":
        dataset_id = "dataset6_han_dnarna"
        st.info(f"""
                **Dataset from the following paper:** Han, S., Liang, Y., Ma, Q., 
                Xu, Y., Zhang, Y., Du, W., ... & Li, Y. (2019). LncFinder: an integrated 
                platform for long non-coding RNA identification utilizing sequence 
                intrinsic composition, structural information and physicochemical property. 
                Briefings in bioinformatics, 20(6), 2009-2027. 

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 7: long non-coding RNA - wheat":
        dataset_id = "dataset7_han_dnarna"
        st.info(f"""
                **Dataset from the following paper:** Han, S., Liang, Y., Ma, Q., 
                Xu, Y., Zhang, Y., Du, W., ... & Li, Y. (2019). LncFinder: an integrated 
                platform for long non-coding RNA identification utilizing sequence 
                intrinsic composition, structural information and physicochemical property. 
                Briefings in bioinformatics, 20(6), 2009-2027. 

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 8: plant long non-coding RNA":
        dataset_id = "dataset8_meng_dnarna"
        st.info(f"""
                **Dataset from the following paper:** Meng, J., Kang, Q., Chang, Z., 
                & Luan, Y. (2021). PlncRNA-HDeep: plant long noncoding RNA prediction 
                using hybrid deep learning based on two encoding styles. BMC bioinformatics, 
                22(Suppl 3), 242.

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 9: 5-methylcytosine sites - H. sapiens":
        dataset_id = "dataset9_lv_dnarna"
        st.info(f"""
                **Dataset from the following paper:** Lv, H., Zhang, Z. M., Li, S. H., 
                Tan, J. X., Chen, W., & Lin, H. (2020). Evaluation of different computational
                methods on 5-methylcytosine sites identification. Briefings in bioinformatics, 
                21(3), 982-995.

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 10: 5-methylcytosine sites - M. musculus":
        dataset_id = "dataset10_lv_dnarna"
        st.info(f"""
                **Dataset from the following paper:** Lv, H., Zhang, Z. M., Li, S. H., 
                Tan, J. X., Chen, W., & Lin, H. (2020). Evaluation of different computational
                methods on 5-methylcytosine sites identification. Briefings in bioinformatics, 
                21(3), 982-995.

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 11: 5-methylcytosine sites - S. cerevisiae":
        dataset_id = "dataset11_lv_dnarna"
        st.info(f"""
                **Dataset from the following paper:** Lv, H., Zhang, Z. M., Li, S. H., 
                Tan, J. X., Chen, W., & Lin, H. (2020). Evaluation of different computational
                methods on 5-methylcytosine sites identification. Briefings in bioinformatics, 
                21(3), 982-995.

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 12: 5-methylcytosine sites - A. thaliana":
        dataset_id = "dataset12_lv_dnarna"
        st.info(f"""
                **Dataset from the following paper:** Lv, H., Zhang, Z. M., Li, S. H., 
                Tan, J. X., Chen, W., & Lin, H. (2020). Evaluation of different computational
                methods on 5-methylcytosine sites identification. Briefings in bioinformatics, 
                21(3), 982-995.

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 13: non-coding RNA - E. coli K12":
        dataset_id = "dataset13_bonidia_dnarna"
        st.info(f"""
                **Dataset from the following paper:** Bonidia, R. P., Santos, A. P. A., 
                de Almeida, B. L., Stadler, P. F., da Rocha, U. N., Sanches, D. S., & de Carvalho, 
                A. C. (2022). BioAutoML: automated feature engineering and metalearning to predict 
                noncoding RNAs in bacteria. Briefings in Bioinformatics, 23(4), bbac218.

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 14: non-coding RNA - Multiple bacterial phyla":
        dataset_id = "dataset14_bonidia_dnarna"
        st.info(f"""
                **Dataset from the following paper:** Bonidia, R. P., Santos, A. P. A., 
                de Almeida, B. L., Stadler, P. F., da Rocha, U. N., Sanches, D. S., & de Carvalho, 
                A. C. (2022). BioAutoML: automated feature engineering and metalearning to predict 
                noncoding RNAs in bacteria. Briefings in Bioinformatics, 23(4), bbac218.

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 15: non-coding RNA - 8 classes":
        dataset_id = "dataset15_bonidia_dnarna"
        st.info(f"""
                **Dataset from the following paper:** Bonidia, R. P., Santos, A. P. A., 
                de Almeida, B. L., Stadler, P. F., da Rocha, U. N., Sanches, D. S., & de Carvalho, 
                A. C. (2022). BioAutoML: automated feature engineering and metalearning to predict 
                noncoding RNAs in bacteria. Briefings in Bioinformatics, 23(4), bbac218.

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 16: antimicrobial peptides":
        dataset_id = "dataset16_chung_protein"
        st.info(f"""
                **Dataset from the following paper:** Chung, C. R., Kuo, T. R., Wu, L. C., 
                Lee, T. Y., & Horng, J. T. (2020). Characterization and identification of 
                antimicrobial peptides with different functional activities. Briefings in 
                bioinformatics, 21(3), 1098-1114.

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 17: antiviral":
        dataset_id = "dataset17_timmons_protein"
        st.info(f"""
                **Dataset from the following paper:** Timmons, P. B., & Hewage, C. M. (2021). 
                ENNAVIA is a novel method which employs neural networks for antiviral and 
                anti-coronavirus activity prediction for therapeutic peptides. Briefings in 
                bioinformatics, 22(6).

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 18: antiviral using random sequences":
        dataset_id = "dataset18_timmons_protein"
        st.info(f"""
                **Dataset from the following paper:** Timmons, P. B., & Hewage, C. M. (2021). 
                ENNAVIA is a novel method which employs neural networks for antiviral and 
                anti-coronavirus activity prediction for therapeutic peptides. Briefings in 
                bioinformatics, 22(6).

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 19: anti-coronavirus":
        dataset_id = "dataset19_timmons_protein"
        st.info(f"""
                **Dataset from the following paper:** Timmons, P. B., & Hewage, C. M. (2021). 
                ENNAVIA is a novel method which employs neural networks for antiviral and 
                anti-coronavirus activity prediction for therapeutic peptides. Briefings in 
                bioinformatics, 22(6).

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 20: anti-coronavirus using random sequences":
        dataset_id = "dataset20_timmons_protein"
        st.info(f"""
                **Dataset from the following paper:** Timmons, P. B., & Hewage, C. M. (2021). 
                ENNAVIA is a novel method which employs neural networks for antiviral and 
                anti-coronavirus activity prediction for therapeutic peptides. Briefings in 
                bioinformatics, 22(6).

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 21: antimicrobial peptides":
        dataset_id = "dataset21_xing_protein"
        st.info(f"""
                **Dataset from the following paper:** Xing, W., Zhang, J., Li, C., Huo, Y., 
                & Dong, G. (2023). iAMP-Attenpred: a novel antimicrobial peptide predictor based 
                on BERT feature extraction method and CNN-BiLSTM-Attention combination model. 
                Briefings in bioinformatics, 25(1).

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 22: antimicrobial peptides":
        dataset_id = "dataset22_xing_protein"
        st.info(f"""
                **Dataset from the following paper:** Xing, W., Zhang, J., Li, C., Huo, Y., 
                & Dong, G. (2023). iAMP-Attenpred: a novel antimicrobial peptide predictor based 
                on BERT feature extraction method and CNN-BiLSTM-Attention combination model. 
                Briefings in bioinformatics, 25(1).

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 23: anticancer peptides":
        dataset_id = "dataset23_yu_protein"
        st.info(f"""
                **Dataset from the following paper:** Yu, L., Jing, R., Liu, F., Luo, J., & Li, 
                Y. (2020). DeepACP: a novel computational approach for accurate identification 
                of anticancer peptides by deep learning algorithm. Molecular Therapy Nucleic Acids, 
                22, 862-870.

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 24: circRNA vs lncRNA":
        dataset_id = "dataset24_bonidia_dnarna"
        st.info(f"""
                **Dataset from the following paper:** Bonidia, R. P., Sampaio, L. D., Domingues, 
                D. S., Paschoal, A. R., Lopes, F. M., de Carvalho, A. C., & Sanches, D. S. (2021). 
                Feature extraction approaches for biological sequences: a comparative study of 
                mathematical features. Briefings in Bioinformatics, 22(5), bbab011.

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 25: mRNA subcellular localization":
        dataset_id = "dataset25_musleh_dnarna"
        st.info(f"""
                **Dataset from the following paper:** Musleh, S., Islam, M. T., Qureshi, R., Alajez, 
                N. M., & Alam, T. (2023). MSLP: mRNA subcellular localization predictor based on 
                machine learning techniques. BMC bioinformatics, 24(1), 109.

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 26: lncRNAs subcellular localization - training set with 5 classes":
        dataset_id = "dataset26_cai_dnarna"
        st.info(f"""
                **Dataset from the following paper:** Cai, J., Wang, T., Deng, X., Tang, L., & Liu, L. 
                (2023). GM-lncLoc: LncRNAs subcellular localization prediction based on graph neural 
                network with meta-learning. BMC genomics, 24(1), 52.

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 27: lncRNAs subcellular localization - training set with 4 classes":
        dataset_id = "dataset27_cai_dnarna"
        st.info(f"""
                **Dataset from the following paper:** Cai, J., Wang, T., Deng, X., Tang, L., & Liu, L. 
                (2023). GM-lncLoc: LncRNAs subcellular localization prediction based on graph neural 
                network with meta-learning. BMC genomics, 24(1), 52.

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 28: antioxidant proteins":
        dataset_id = "dataset28_1_lam_protein"
        st.info(f"""
                **Dataset from the following paper:** Ho Thanh Lam, L., Le, N. H., Van Tuan, L., 
                Tran Ban, H., Nguyen Khanh Hung, T., Nguyen, N. T. K., ... & Le, N. Q. K. (2020). 
                Machine learning model for identifying antioxidant proteins using features calculated 
                from primary sequences. Biology, 9(10), 325.

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 29: proinflammatory peptides":
        dataset_id = "dataset29_khatun_protein"
        st.info(f"""
                **Dataset from the following paper:** Khatun, M. S., Hasan, M. M., Shoombuatong, 
                W., & Kurata, H. (2020). ProIn-Fuse: improved and robust prediction of proinflammatory 
                peptides by fusing of multiple feature representations. Journal of Computer-Aided 
                Molecular Design, 34(12), 1229-1236.

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 30: recombination spots":
        dataset_id = "dataset30_khan_dnarna"
        st.info(f"""
                **Dataset from the following paper:** Khan, F., Khan, M., Iqbal, N., Khan, S., Muhammad Khan, 
                D., Khan, A., & Wei, D. Q. (2020). Prediction of recombination spots using novel hybrid feature 
                extraction method via deep learning approach. Frontiers in Genetics, 11, 539227.

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 31: DNA-binding proteins":
        dataset_id = "dataset31_chowdhury_protein"
        st.info(f"""
                **Dataset from the following paper:** Chowdhury, S. Y., Shatabda, S., & Dehzangi, A. (2017). 
                iDNAProt-ES: identification of DNA-binding proteins using evolutionary and structural features. 
                Scientific reports, 7(1), 14938.

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 32: DNase I hypersensitive sites":
        dataset_id = "dataset32_liu_dnarna"
        st.info(f"""
                **Dataset from the following paper:** Liu, B., Long, R., & Chou, K. C. (2016). iDHS-EL: identifying 
                DNase I hypersensitive sites by fusing three different modes of pseudo nucleotide composition into 
                an ensemble learning framework. Bioinformatics, 32(16), 2411-2418.

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 33: real microRNA precursors":
        dataset_id = "dataset33_liu_dnarna"
        st.info(f"""
                **Dataset from the following paper:** Liu, B., Fang, L., Liu, F., Wang, X., Chen, J., & Chou, K. C. (2015). 
                Identification of real microRNA precursors with a pseudo structure status composition approach. PloS one, 
                10(3), e0121501.

                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 34: small non-coding RNAs in bacteria":
        dataset_id = "dataset34_barman_dnarna"
        st.info(f"""
                **Dataset from the following paper:** Barman, R. K., Mukhopadhyay, A., & Das, S. (2017). An improved method 
                for identification of small non-coding RNAs in bacteria using support vector machine. Scientific reports, 7(1), 46070.
                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 35: non-coding RNA - 4 classes":
        dataset_id = "dataset35_avila_dnarna"
        st.info(f"""
                **Dataset from the following paper:** Avila Santos, A. P., de Almeida, B. L., Bonidia, R. P., Stadler, P. F., Stefanic, 
                P., Mandic-Mulec, I., ... & de Carvalho, A. C. (2024). BioDeepfuse: a hybrid deep learning approach with integrated 
                feature extraction techniques for enhanced non-coding RNA classification. RNA biology, 21(1), 410-421.
                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 36: toehold switch sequences":
        dataset_id = "dataset36_valeri_dnarna"
        st.info(f"""
                **Dataset from the following paper:** Valeri, J. A., Collins, K. M., Ramesh, P., Alcantar, M. A., Lepe, B. A., Lu, T. K.,
                & Camacho, D. M. (2020). Sequence-to-function deep learning frameworks for engineered riboregulators. Nature communications, 
                11(1), 5058.
                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 37: ribosome binding site sequences":
        dataset_id = "dataset37_hoellerer_dnarna"
        st.info(f"""
                **Dataset from the following paper:** Höllerer, S., Papaxanthos, L., Gumpinger, A. C., Fischer, K., Beisel, C., Borgwardt, 
                K., ... & Jeschek, M. (2020). Large-scale DNA-based phenotypic recording and deep learning enable highly accurate 
                sequence-function mapping. Nature communications, 11(1), 3551.
                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)
    elif model == "Model 38: antibody sequences":
        dataset_id = "dataset38_liu_protein"
        st.info(f"""
                **Dataset from the following paper:** Liu, G., Zeng, H., Mueller, J., Carter, B., Wang, Z., Schilz, J., ... & Gifford, D. K. 
                (2020). Antibody complementarity determining region design using high-capacity machine learning. Bioinformatics, 36(7), 2126-2133.
                You can consult experiments done with this dataset in \"Jobs\" using the following ID: **{dataset_id}**
                """)

    queue_info = st.container()

    with st.form("repo_submit", clear_on_submit=True):
        test_files = st.file_uploader("FASTA file for prediction", 
                                    accept_multiple_files=False, 
                                    help="Single file for prediction (e.g. predict.fasta)")

        submitted = st.form_submit_button("Submit", 
                                    use_container_width=True, 
                                    type="primary")

    predict_path = os.path.abspath("jobs")

    if submitted:
        if not test_files:
            with queue_info:
                st.error("Please upload the required prediction file.")
        else:
            job_id = ''.join([choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16)])
            job_path = os.path.join(predict_path, job_id)
            dataset_path = os.path.join(os.path.abspath("datasets"), dataset_id, "runs/run_1")

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

            # Add job to the queue
            with queue_lock:
                job_queue.put((dataset_path, test_files, job_path, data_type, 
                             training, testing, classifier, imbalance, fselection))
            
            with queue_info:
                st.success(f"Job submitted to the queue. You can consult the results in \"Jobs\" using the following ID: **{job_id}**")

# Run the Streamlit app
if __name__ == "__main__":
    runUI()