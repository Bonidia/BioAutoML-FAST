import subprocess
import polars as pl
import pandas as pd
import os
import joblib
from App.utils.stats import summary_stats

def main():
    full_datasets_path = "App/datasets"

    datasets_list = os.listdir(full_datasets_path)

    for dataset in datasets_list:
        dataset_path = os.path.join(full_datasets_path, dataset)

        dtype_str = dataset.split('_')[-1]

        if dtype_str == "protein":
            data_type = "Protein"
        if dtype_str == "dnarna":
            data_type = "DNA/RNA"

        train_path = os.path.join(dataset_path, "train")
        train_files = [os.path.join(train_path, file) for file in os.listdir(train_path)]
        train_labels = [os.path.splitext(os.path.basename(file))[0] for file in train_files]

        test_path = os.path.join(dataset_path, "test")
        test_files = [os.path.join(test_path, file) for file in os.listdir(test_path)]
        test_labels = [os.path.splitext(os.path.basename(file))[0] for file in test_files]

        command = [
            "python",
            "BioAutoML-protein.py" if data_type == "Protein" else "BioAutoML-feature.py",
            "--imbalance",
            "0", # "1" if imbalance else "0",
            "--fselection",
            "0", # "1" if fselection else "0",
            "--fasta_train",
        ]

        command.extend(train_files)

        command.append("--fasta_label_train")
        command.extend(train_labels)

        command.append("--fasta_test")
        command.extend(test_files)

        command.append("--fasta_label_test")
        command.extend(test_labels)

        command.extend(["--n_cpu", "-1"])
        command.extend(["--output", dataset_path])

        subprocess.run(command)

        classifier, imbalance, fselection = False, False, False

        job_data = {
            "data_type": [data_type],
            "training_set": ["Training set"],
            "testing_set": ["Test set"],
            "classifier_selected": [classifier], 
            "imbalance_methods": [imbalance],  
            "feature_selection": [fselection],  
        }

        df_job_data = pl.DataFrame(job_data)
        tsv_path = os.path.join(dataset_path, "job_info.tsv")
        df_job_data.write_csv(tsv_path, separator='\t')

        summary_stats(os.path.join(dataset_path, "feat_extraction", "train"), data_type, dataset_path, False)
        summary_stats(os.path.join(dataset_path, "feat_extraction", "test"), data_type, dataset_path, False)

        model = joblib.load(os.path.join(dataset_path, "trained_model.sav"))
        model["train_stats"] = pd.read_csv(os.path.join(dataset_path, "train_stats.csv"))
        joblib.dump(model, os.path.join(dataset_path, "trained_model.sav"))

if __name__ == "__main__":
    main()