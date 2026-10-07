import subprocess
import sys
import polars as pl
import pandas as pd
import os
import time
import joblib
import argparse
import math
from pathlib import Path

# Import the statistics helper without initializing the web queue/database.
PROJECT_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_PATH / 'App' / 'utils'))
from stats import summary_stats


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run five sequential, seeded searches per dataset.')
    parser.add_argument('--stage2_gate', action='store_true', help='Enable the optional default LightGBM fallback')
    parser.add_argument('--stage2_gate_margin_sd', type=float, default=0.5)
    parser.add_argument('--n_cpu', type=int, default=8)
    parser.add_argument('--seed', type=int, default=63)
    args = parser.parse_args()
    if not math.isfinite(args.stage2_gate_margin_sd) or args.stage2_gate_margin_sd < 0:
        parser.error('--stage2_gate_margin_sd must be finite and nonnegative')
    return args

def main():
    args = parse_arguments()
    os.chdir(PROJECT_PATH)
    start_all = time.time()  # Start measuring total time of main()

    full_datasets_path = "App/datasets"
    num_runs = 5  # Number of times to run each dataset

    datasets_list = sorted(item for item in os.listdir(full_datasets_path)
                           if os.path.isdir(os.path.join(full_datasets_path, item)))

    for dataset in datasets_list:
        dataset_path = os.path.join(full_datasets_path, dataset)

        # Skip this dataset if it already has a "runs" folder
        experiments_folder = os.path.join(dataset_path, "runs")
        # if os.path.exists(experiments_folder):
        #     continue

        dtype_str, task = dataset.split('_')[-2:]

        if dtype_str == "protein":
            data_type = "Protein"
        if dtype_str == "dnarna":
            data_type = "DNA/RNA"

        train_path = os.path.join(dataset_path, "train")
        train_files = [os.path.join(train_path, file) for file in sorted(os.listdir(train_path))
                       if file.lower().endswith(('.fasta', '.fa', '.fna', '.faa')) and os.path.isfile(os.path.join(train_path, file))]
        train_labels = [os.path.splitext(os.path.basename(file))[0] for file in train_files]

        test_path = os.path.join(dataset_path, "test")

        if os.path.exists(test_path):
            test_files = [os.path.join(test_path, file) for file in sorted(os.listdir(test_path))
                          if file.lower().endswith(('.fasta', '.fa', '.fna', '.faa')) and os.path.isfile(os.path.join(test_path, file))]
            test_labels = [os.path.splitext(os.path.basename(file))[0] for file in test_files]

        # Create a runs folder for this dataset
        os.makedirs(experiments_folder, exist_ok=True)

        for run_num in range(1, num_runs + 1):
            # Create a folder for this run inside the runs folder
            run_folder = os.path.join(experiments_folder, f"run_{run_num}")
            if not os.path.exists(run_folder):
                os.makedirs(run_folder, exist_ok=True)

                classifier = False

                command = [
                    sys.executable,
                    "engineering.py",
                    "--dtype",
                    dtype_str,
                    "--estimations",
                    "200", # 200
                    "--patience",
                    "80",
                    "--tuning",
                    "150", # 150
                    "--difference",
                    "0.001",
                    "--task",
                    task,
                    "--fasta_train",
                ]

                command.extend(train_files)

                command.append("--fasta_label_train")
                command.extend(train_labels)

                testing_set = True if os.path.exists(test_path) else False

                if testing_set:
                    command.append("--fasta_test")
                    command.extend(test_files)

                    command.append("--fasta_label_test")
                    command.extend(test_labels)

                command.extend(["--n_cpu", str(args.n_cpu)])
                command.extend(["--seed", str(args.seed)])
                command.extend(["--search_seed", str(6300 + run_num)])
                command.extend(["--search_jobs", "1"])
                command.extend(["--stage2_gate_margin_sd", str(args.stage2_gate_margin_sd)])
                if args.stage2_gate:
                    command.append('--stage2_gate')
                command.extend(["--output", run_folder])  # Output to the run-specific folder

                print(f"Running dataset {dataset}, iteration {run_num}")

                # === START TIMING THIS RUN ===
                start_run = time.time()

                subprocess.run(command)

                # === END TIMING THIS RUN ===
                run_time_seconds = round(time.time() - start_run, 2)

                # Save time spent for this run
                df_time = pl.DataFrame({
                    "dataset": [dataset],
                    "run": [run_num],
                    "time_seconds": [run_time_seconds],
                })

                time_csv_path = os.path.join(run_folder, "time_spent.csv")
                df_time.write_csv(time_csv_path)

                job_data = {
                    "data_type": [data_type],
                    "task": ["Classification" if task == "0" else "Regression"],
                    "training_set": ["Training set"],
                    "testing_set": ["Test set" if testing_set else "No test set"],
                    "classifier_selected": [classifier]
                }

                df_job_data = pl.DataFrame(job_data)
                tsv_path = os.path.join(run_folder, "job_info.tsv")
                df_job_data.write_csv(tsv_path, separator='\t')

                # Update paths for summary stats to use the run folder
                
                summary_stats(os.path.join(run_folder if run_num == 1 else os.path.join(experiments_folder, "run_1"), "feat_extraction", "train"), data_type, run_folder, False)

                if testing_set:
                    summary_stats(os.path.join(run_folder if run_num == 1 else os.path.join(experiments_folder, "run_1"), "feat_extraction", "test"), data_type, run_folder, False)

                model_path = os.path.join(run_folder, "trained_model.sav")
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    model["train_stats"] = pd.read_csv(os.path.join(run_folder, "train_stats.csv"))
                    joblib.dump(model, model_path)

    # End total time
    total_time = round(time.time() - start_all, 2)
    print(f"\n⏱ Total execution time for all datasets: {total_time} seconds")

if __name__ == "__main__":
    main()
