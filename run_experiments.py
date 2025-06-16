import subprocess
import polars as pl
import pandas as pd
import os
import joblib
from App.utils.stats import summary_stats

def main():
    full_datasets_path = "App/datasets"
    num_runs = 1  # Number of times to run each dataset

    datasets_list = os.listdir(full_datasets_path)

    for dataset in datasets_list:
        dataset_path = os.path.join(full_datasets_path, dataset)

        # Skip this dataset if it already has a "runs" folder
        runs_folder = os.path.join(dataset_path, "runs")
        if os.path.exists(runs_folder):
            continue

        dtype_str = dataset.split('_')[-1]

        if dtype_str == "protein":
            data_type = "Protein"
        if dtype_str == "dnarna":
            data_type = "DNA/RNA"

        train_path = os.path.join(dataset_path, "train")
        train_files = [os.path.join(train_path, file) for file in os.listdir(train_path)]
        train_labels = [os.path.splitext(os.path.basename(file))[0] for file in train_files]

        test_path = os.path.join(dataset_path, "test")

        if os.path.exists(test_path):
            test_files = [os.path.join(test_path, file) for file in os.listdir(test_path)]
            test_labels = [os.path.splitext(os.path.basename(file))[0] for file in test_files]

        # Create a runs folder for this dataset
        os.makedirs(runs_folder, exist_ok=True)

        for run_num in range(1, num_runs + 1):
            # Create a folder for this run inside the runs folder
            run_folder = os.path.join(runs_folder, f"run_{run_num}")
            os.makedirs(run_folder, exist_ok=True)

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

            if os.path.exists(test_path):
                command.append("--fasta_test")
                command.extend(test_files)

                command.append("--fasta_label_test")
                command.extend(test_labels)

            command.extend(["--n_cpu", "-1"])
            command.extend(["--output", run_folder])  # Output to the run-specific folder

            print(f"Running dataset {dataset}, iteration {run_num}")
            subprocess.run(command)

            classifier, imbalance, fselection = False, False, False

            job_data = {
                "data_type": [data_type],
                "training_set": ["Training set"],
                "testing_set": ["Test set"],
                "classifier_selected": [classifier], 
                "imbalance_methods": [imbalance],  
                "feature_selection": [fselection],  
                "run_number": [run_num],
            }

            df_job_data = pl.DataFrame(job_data)
            tsv_path = os.path.join(run_folder, "job_info.tsv")
            df_job_data.write_csv(tsv_path, separator='\t')

            # Update paths for summary stats to use the run folder
            summary_stats(os.path.join(run_folder, "feat_extraction", "train"), data_type, run_folder, False)
            summary_stats(os.path.join(run_folder, "feat_extraction", "test"), data_type, run_folder, False)

            model_path = os.path.join(run_folder, "trained_model.sav")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                model["train_stats"] = pd.read_csv(os.path.join(run_folder, "train_stats.csv"))
                joblib.dump(model, model_path)

if __name__ == "__main__":
    main()