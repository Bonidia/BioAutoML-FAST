import subprocess
import sys
import polars as pl
import pandas as pd
import os
import time
import argparse
from pathlib import Path

# Import the statistics helper without initializing the web queue/database.
PROJECT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_PATH))
from bioautoml.execution import Execution, run_path
from bioautoml.model_artifacts import update_model_summary
sys.path.insert(0, str(PROJECT_PATH / 'App' / 'utils'))
from stats import summary_stats


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run five sequential, seeded searches per dataset.')
    parser.add_argument('--n_cpu', type=int, default=8)
    parser.add_argument('--seed', type=int, default=63)
    parser.add_argument('--output', required=True, help='New directory for benchmark results; existing directories are not reused')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    os.chdir(PROJECT_PATH)
    output_path = Path(args.output).resolve()
    start_all = time.perf_counter()

    full_datasets_path = "App/datasets"
    num_runs = 5  # Number of times to run each dataset

    if output_path.is_relative_to((PROJECT_PATH / full_datasets_path).resolve()):
        raise ValueError('Keep benchmark outputs outside App/datasets to preserve the source datasets.')
    output_path.mkdir(parents=True, exist_ok=False)

    datasets_list = sorted(item for item in os.listdir(full_datasets_path)
                           if item.startswith('dataset') and os.path.isdir(os.path.join(full_datasets_path, item)))

    for dataset in datasets_list:
        dataset_path = os.path.join(full_datasets_path, dataset)

        experiments_folder = os.path.join(output_path, dataset, "runs")

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
        test_files, test_labels = [], []

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

                testing_set = bool(test_files)

                if testing_set:
                    command.append("--fasta_test")
                    command.extend(test_files)

                    command.append("--fasta_label_test")
                    command.extend(test_labels)

                command.extend(["--n_cpu", str(args.n_cpu)])
                command.extend(["--seed", str(args.seed)])
                command.extend(["--search_seed", str(6300 + run_num)])
                command.extend(["--search_jobs", "1"])
                command.extend(["--output", run_folder])  # Output to the run-specific folder

                print(f"Running dataset {dataset}, iteration {run_num}")

                execution = Execution(run_folder, 'training', settings={
                    'dataset': dataset, 'run': run_num, 'command': command,
                    'seed': args.seed, 'search_seed': 6300 + run_num,
                    'n_cpu': args.n_cpu,
                })
                try:
                    execution.add_inputs(train_files + test_files)
                    subprocess.run(command, check=True)

                    with execution.phase('benchmark_summary'):
                        job_data = {
                            "data_type": [data_type],
                            "task": ["Classification" if task == "0" else "Regression"],
                            "training_set": ["Training set"],
                            "testing_set": ["Test set" if testing_set else "No test set"],
                            "classifier_selected": [classifier]
                        }
                        pl.DataFrame(job_data).write_csv(run_path(run_folder, "job_info.tsv"), separator='\t')
                        summary_stats(run_path(run_folder, "feat_extraction", "train"), data_type, run_folder, False)
                        if testing_set:
                            summary_stats(run_path(run_folder, "feat_extraction", "test"), data_type, run_folder, False)

                        # This artifact was produced locally by the checked subprocess,
                        # not uploaded to the web platform. Benchmark models stay unsigned.
                        update_model_summary(
                            run_path(run_folder, "trained_model.sav"), trust_unsigned=True,
                            train_stats=pd.read_csv(run_path(run_folder, "train_stats.csv")))
                except BaseException:
                    execution.finish('failed')
                    raise
                else:
                    execution.finish()

    # End total time
    total_time = (time.perf_counter() - start_all) / 60
    print(f"\n⏱ Total execution time for all datasets: {total_time:.2f} minutes")

if __name__ == "__main__":
    main()
