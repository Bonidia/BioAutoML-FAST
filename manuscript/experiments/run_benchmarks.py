import subprocess
import sys
import polars as pl
import pandas as pd
import os
import time
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from Bio import SeqIO

# Import the statistics helper without initializing the web queue/database.
PROJECT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_PATH))
from bioautoml.execution import Execution, run_path, atomic_json, file_hash
from bioautoml.model_artifacts import update_model_summary
sys.path.insert(0, str(PROJECT_PATH / 'App' / 'utils'))
from stats import summary_stats


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run sequential, seeded searches per dataset.')
    parser.add_argument('--n_cpu', type=int, default=8)
    parser.add_argument('--seed', type=int, default=63)
    parser.add_argument('--output', required=True, help='New directory for benchmark results; existing directories are not reused')
    parser.add_argument('--runs', type=int, choices=range(1, 6), default=5)
    parser.add_argument('--reuse_ablation', type=Path, help='Reuse verified, matching selected_tuned results and their shared features')
    parser.add_argument('--resume', action='store_true', help='Skip verified completed runs; never overwrite partial runs')
    parser.add_argument('--dry_run', action='store_true', help='Validate inputs and reuse without creating outputs or training')
    args = parser.parse_args()
    if args.n_cpu < 1 or not 0 <= args.seed < 2**32:
        parser.error('Use a positive CPU count and a seed in [0, 2**32).')
    return args


def input_inventory():
    datasets, rows = [], []
    directories = sorted((PROJECT_PATH / 'App/datasets').glob('dataset*'),
                         key=lambda path: int(path.name.split('_')[0][7:]))
    for directory in directories:
        if not directory.is_dir():
            continue
        dtype, task = directory.name.split('_')[-2:]
        dataset = dict(dataset=directory.name, dataset_id=int(directory.name.split('_')[0][7:]),
                       dtype=dtype, task=int(task))
        for split in ('train', 'test'):
            files = sorted(path for path in (directory / split).glob('*') if path.is_file()
                           and path.suffix.lower() in ('.fasta', '.fa', '.fna', '.faa'))
            if split == 'train' and not files:
                raise ValueError(f'Missing training FASTA files: {directory}')
            dataset[f'{split}_files'] = list(map(str, files))
            dataset[f'{split}_labels'] = [path.stem for path in files]
            count = 0
            for path in files:
                number = 0
                for record in SeqIO.parse(path, 'fasta'):
                    if not record.seq:
                        raise ValueError(f'Empty sequence in {path}: {record.id}')
                    number += 1
                if not number:
                    raise ValueError(f'Empty FASTA: {path}')
                count += number
                rows.append(dict(dataset=directory.name, split=split, path=str(path),
                                 records=number, sha256=file_hash(path)))
            dataset[f'{split}_count'] = count
        dataset['input_hashes'] = {row['path']: row['sha256'] for row in rows
                                   if row['dataset'] == directory.name}
        datasets.append(dataset)
    return datasets, pd.DataFrame(rows)


def reusable_results(args, datasets):
    if not args.reuse_ablation:
        return {}
    import run_ablation as ablation
    from run_learner_aware import baseline_configs
    root = args.reuse_ablation.resolve()
    protocol = ablation.read_json(root / 'protocol.json')
    if protocol['status'] != 'completed':
        raise ValueError('Only a completed ablation may be reused.')
    settings = SimpleNamespace(ablation_root=root, seed=args.seed, n_cpu=args.n_cpu,
                               estimations=200, tuning=150, smoke_test=False)
    reused = {}
    for dataset in datasets:
        if dataset['dataset_id'] not in ablation.DATASETS:
            continue
        for run in range(1, args.runs + 1):
            if 6300 + run not in protocol['search_seeds']:
                continue
            configs = baseline_configs(settings, dataset, run)
            references = []
            for config in configs:
                status = ablation.validate_previous_step(config, protocol.get('compatible_runner_hashes', []),
                                                         protocol.get('compatible_code_changes', []))
                if status != 'completed':
                    raise ValueError(f'Incomplete cached run: {config["output"]}')
                path = Path(config['output'])
                references.append(dict(output=str(path), run_sha256=file_hash(path / 'run.json')))
            row = ablation.collect_result(configs[-1])
            reused[(dataset['dataset'], run)] = dict(
                output=configs[-1]['output'], references=references,
                primary_metric=row['primary_metric'], primary_score=row['primary_score'],
                timing_scope='Reused ablation result; extraction and Stage 1 are shared, not standalone timings.')
    return reused


def verify_sources(dataset, hashes):
    import run_ablation as ablation
    if ablation.code_hashes() != hashes:
        raise ValueError('Training source changed after benchmark protocol freeze.')
    for path, expected in dataset['input_hashes'].items():
        if file_hash(path) != expected:
            raise ValueError(f'Input changed: {path}')

def main():
    args = parse_arguments()
    os.chdir(PROJECT_PATH)
    output_path = Path(args.output).resolve()
    start_all = time.perf_counter()

    full_datasets_path = "App/datasets"
    num_runs = args.runs

    if output_path.is_relative_to((PROJECT_PATH / full_datasets_path).resolve()):
        raise ValueError('Keep benchmark outputs outside App/datasets to preserve the source datasets.')
    if args.reuse_ablation and (output_path.is_relative_to(args.reuse_ablation.resolve())
                               or args.reuse_ablation.resolve().is_relative_to(output_path)):
        raise ValueError('Benchmark and ablation output directories must not overlap.')
    import run_ablation as ablation
    datasets, inventory = input_inventory()
    reused = reusable_results(args, datasets)
    settings = dict(n_cpu=args.n_cpu, seed=args.seed, search_seeds=list(range(6301, 6301 + num_runs)),
                    estimations=200, tuning=150, patience=80, difference=0.001, search_jobs=1,
                    homology_aware=False, datasets=datasets, code_sha256=ablation.code_hashes(),
                    runner_sha256=file_hash(__file__),
                    reuse_ablation=str(args.reuse_ablation.resolve()) if args.reuse_ablation else None,
                    reused_results=[dict(dataset=name, run=run, **value) for (name, run), value in reused.items()])
    print(f'{len(datasets)} datasets x {num_runs} run(s): {len(reused)} reused, '
          f'{len(datasets) * num_runs - len(reused)} new pipelines.', flush=True)
    print(inventory.groupby(['dataset', 'split'], sort=False).records.sum().unstack().to_string(), flush=True)
    if args.dry_run:
        print('Dry run: no outputs created and no training launched.', flush=True)
        return
    protocol_path = output_path / 'protocol.json'
    if args.resume:
        protocol = json.loads(protocol_path.read_text())
        if protocol['settings'] != settings:
            raise ValueError('Resume settings, inputs, source code or reuse provenance changed.')
    else:
        output_path.mkdir(parents=True, exist_ok=False)
        protocol = dict(settings=settings, status='running', completed_results=0)
        inventory.to_csv(output_path / 'input_inventory.csv', index=False)
    protocol['status'] = 'running'
    atomic_json(protocol_path, protocol)
    rows = []

    def checkpoint(row):
        rows.append(row)
        ablation.save_csv(pd.DataFrame(rows), output_path / 'results_index.csv')
        protocol['completed_results'] = len(rows)
        atomic_json(protocol_path, protocol)

    for dataset_info in datasets:
        dataset = dataset_info['dataset']
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
            verify_sources(dataset_info, settings['code_sha256'])
            row = dict(dataset=dataset, run=run_num, seed=args.seed, search_seed=6300 + run_num)
            if (dataset, run_num) in reused:
                reference = reused[(dataset, run_num)]
                reference_path = Path(experiments_folder) / f'run_{run_num}' / 'reuse.json'
                reference_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(reference_path, reference)
                checkpoint(dict(row, status='completed', reused=True, output=reference['output'],
                                primary_metric=reference['primary_metric'], primary_score=reference['primary_score']))
                print(f'Reusing {dataset}, iteration {run_num}: {reference["output"]}', flush=True)
                continue
            # Create a folder for this run inside the runs folder
            run_folder = os.path.join(experiments_folder, f"run_{run_num}")
            if os.path.exists(run_folder):
                record_path = Path(run_folder) / 'run.json'
                record = json.loads(record_path.read_text()) if record_path.is_file() else {}
                if args.resume and record.get('status') == 'completed':
                    if record['settings'].get('search_seed') != 6300 + run_num:
                        raise ValueError(f'Resume seed mismatch: {run_folder}')
                    if not Path(run_path(run_folder, 'trained_model.sav')).is_file():
                        raise ValueError(f'Missing completed model: {run_folder}')
                    checkpoint(dict(row, status='completed', reused=False, output=run_folder))
                    continue
                raise ValueError(f'Incomplete output preserved; inspect before retrying: {run_folder}')
            if not os.path.exists(run_folder):
                os.makedirs(run_folder, exist_ok=True)

                classifier = False

                command = [
                    sys.executable,
                    "engineering.py",
                    "--dtype",
                    data_type,
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
                    environment = dict(os.environ, PYTHONHASHSEED=str(args.seed))
                    with open(Path(run_folder) / 'logs/benchmark.log', 'w') as log:
                        subprocess.run(command, check=True, env=environment, stdout=log, stderr=subprocess.STDOUT)

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
                    verify_sources(dataset_info, settings['code_sha256'])
                except BaseException:
                    execution.finish('failed')
                    protocol.update(status='failed', failed_dataset=dataset, failed_run=run_num)
                    atomic_json(protocol_path, protocol)
                    raise
                else:
                    execution.finish()
                    checkpoint(dict(row, status='completed', reused=False, output=run_folder))

    # End total time
    total_time = (time.perf_counter() - start_all) / 60
    protocol.update(status='completed', total_minutes=total_time)
    atomic_json(protocol_path, protocol)
    print(f"\n⏱ Total execution time for all datasets: {total_time:.2f} minutes")

if __name__ == "__main__":
    main()
