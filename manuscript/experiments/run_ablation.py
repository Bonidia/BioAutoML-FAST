"""Six-condition, paired ablation using the production extraction/search code.

Run --dry_run first. Reduced budgets or alternate inputs require --smoke_test.
Cached timings describe this experiment, NOT standalone pipeline runtimes.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from Bio import SeqIO

PROJECT_PATH = Path(__file__).resolve().parents[2]
EXPERIMENTS_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_PATH))
from bioautoml.execution import Execution, atomic_json, file_hash, run_path


DATASETS = {
    2: 'dataset2_yu_protein_0',
    10: 'dataset10_charoenkwan_protein_0',
    13: 'dataset13_xiao_protein_0',
    20: 'dataset20_pinacho_protein_0',
    23: 'dataset23_chowdhury_protein_0',
    29: 'dataset29_han_dnarna_0',
    31: 'dataset31_han_dnarna_0',
    33: 'dataset33_lv_dnarna_0',
    39: 'dataset39_wei_protein_0',
    51: 'dataset51_cai_dnarna_0',
    53: 'dataset53_musleh_dnarna_0',
    54: 'dataset54_avila_dnarna_0',
    55: 'dataset55_bonidia_dnarna_0',
    62: 'dataset62_qiu_protein_1',
    63: 'dataset63_leuenberger_protein_1',
}
CONDITIONS = ('canonical_default', 'canonical_tuned', 'all_default',
              'all_tuned', 'selected_default', 'selected_tuned')


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_cpu', type=int, default=8)
    parser.add_argument('--seed', type=int, default=63)
    parser.add_argument('--datasets', type=int, nargs='+', choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument('--runs', type=int, choices=range(1, 6), default=3)
    parser.add_argument('--estimations', type=int, default=200)
    parser.add_argument('--tuning', type=int, default=150)
    parser.add_argument('--datasets_root', type=Path, default=PROJECT_PATH / 'App/datasets')
    parser.add_argument('--output', type=Path, default=EXPERIMENTS_PATH / 'results/ablation',
                        help='Fresh directory under manuscript/experiments/results; never overwritten')
    parser.add_argument('--dry_run', action='store_true', help='Validate inputs and print inventory without training or writing outputs')
    parser.add_argument('--resume', action='store_true', help='Reuse verified completed steps; archive and restart incomplete steps')
    parser.add_argument('--smoke_test', action='store_true', help='Explicitly label reduced-budget or alternate-input validation runs')
    parser.add_argument('--worker_config', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_config:
        return args
    if args.n_cpu < 1 or not 0 <= args.seed < 2**32:
        parser.error('--n_cpu must be positive and --seed must be in [0, 2**32).')
    if args.estimations < 1 or args.tuning < 1:
        parser.error('Both trial caps must be positive; default arms internally use tuning=0.')
    if len(args.datasets) != len(set(args.datasets)):
        parser.error('Dataset IDs must not be repeated.')
    args.datasets_root = args.datasets_root.resolve()
    alternate = args.datasets_root != (PROJECT_PATH / 'App/datasets').resolve()
    if not args.smoke_test and (args.runs not in (3, 5) or args.estimations != 200 or args.tuning != 150 or alternate):
        parser.error('Reduced budgets/repeats or alternate input roots require --smoke_test.')
    args.output = args.output.resolve()
    results_path = (EXPERIMENTS_PATH / 'results').resolve()
    if args.output == results_path or not args.output.is_relative_to(results_path):
        parser.error('--output must be a subdirectory of manuscript/experiments/results.')
    return args


def read_json(path):
    return json.loads(Path(path).read_text())


def save_csv(frame, path):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def input_inventory(args):
    datasets, rows = [], []
    for dataset_id in sorted(args.datasets):
        name = DATASETS[dataset_id]
        dtype, task = name.split('_')[-2:]
        dataset = dict(dataset=name, dataset_id=dataset_id, dtype=dtype, task=int(task))
        for split in ('train', 'test'):
            directory = args.datasets_root.resolve() / name / split
            files = sorted(path for path in directory.glob('*') if path.is_file()
                           and path.suffix.lower() in ('.fasta', '.fa', '.fna', '.faa'))
            if not files:
                raise ValueError(f'Missing FASTA inputs: {directory}')
            labels = [path.stem for path in files]
            if len(labels) != len(set(labels)):
                raise ValueError(f'Duplicate FASTA filename labels: {directory}')
            dataset[f'{split}_files'] = list(map(str, files))
            dataset[f'{split}_labels'] = labels
            targets, total = set(), 0
            for path in files:
                count, maximum = 0, 0
                with path.open() as handle:
                    for record in SeqIO.parse(handle, 'fasta'):
                        if not record.seq:
                            raise ValueError(f'Empty sequence in {path}: {record.id}')
                        count += 1
                        maximum = max(maximum, len(record.seq))
                        if dataset['task'] == 1:
                            target = float(record.id.split('|')[-1])
                            if not math.isfinite(target):
                                raise ValueError(f'Nonfinite regression target in {path}: {record.id}')
                            targets.add(target)
                if count == 0 or (dataset['task'] == 0 and split == 'train' and count < 10):
                    raise ValueError(f'Empty FASTA or insufficient records for ten-fold classification CV: {path}')
                if maximum > 100_000:
                    print(f'WARNING: {name}/{split}/{path.name} contains a {maximum:,}-residue sequence; verify source protocol before a full run.')
                rows.append(dict(dataset=name, split=split, label=path.stem, path=str(path),
                                 records=count, max_length=maximum, sha256=file_hash(path)))
                total += count
            dataset[f'{split}_count'] = total
            if dataset['task'] == 1 and (len(targets) < 2 or (split == 'train' and total < 20)):
                raise ValueError(f'Regression needs variable targets and enough rows for ten-fold R²: {directory}')
        if dataset['task'] == 0:
            if len(dataset['train_labels']) < 2 or len(dataset['test_labels']) < 2:
                raise ValueError(f'Classification requires at least two classes: {name}')
            if not set(dataset['test_labels']).issubset(dataset['train_labels']):
                raise ValueError(f'Test labels absent from training: {name}')
        dataset['input_hashes'] = {row['path']: row['sha256'] for row in rows if row['dataset'] == name}
        datasets.append(dataset)
    return datasets, pd.DataFrame(rows)


def code_hashes():
    # Include descriptor implementations, including the checked-out MathFeature code.
    paths = [PROJECT_PATH / name for name in ('engineering.py', 'generation.py')]
    for directory in ('bioautoml', 'other-methods', 'MathFeature'):
        paths.extend((PROJECT_PATH / directory).rglob('*.py'))
    paths.append(Path(__file__).resolve())
    return {str(path.relative_to(PROJECT_PATH)): file_hash(path) for path in sorted(paths)}


def copy_reports(source, output):
    for relative in ('reports/sequence_names', 'reports/sequence_sources.json', 'reports/homology', 'reports/cv_folds.json'):
        path = Path(source) / relative
        destination = Path(output) / relative
        if path.is_dir():
            shutil.copytree(path, destination)
        elif path.is_file():
            shutil.copy2(path, destination)


def extract_features(config):
    import engineering
    from sklearn.model_selection import KFold, StratifiedKFold
    from bioautoml.sequence_names import register_sources
    from bioautoml.homology import prepare_homology

    dataset, output = config['dataset'], config['output']
    protein = dataset['dtype'] == 'protein'
    descriptors = engineering.AMINOACID_DESCRIPTORS if protein else engineering.NUCLEOTIDE_DESCRIPTORS
    for split in ('train', 'test'):
        register_sources(output, dataset[f'{split}_files'], dataset[f'{split}_labels'], split)
    prepare_homology(dataset['train_files'], dataset['train_labels'], dataset['test_files'],
                     dataset['test_labels'], 'Protein' if protein else 'DNA/RNA', dataset['task'],
                     config['seed'], run_path(output, 'homology'), enabled=False, n_cpu=config['n_cpu'])
    extract = engineering.feature_extraction_aminoacid if protein else engineering.feature_extraction_nucleotide
    names, _, train, labels, _, _ = extract(dataset['train_files'], dataset['train_labels'],
                                           None, None, output, config['n_cpu'])
    columns = pd.read_csv(train, nrows=0).columns.tolist()
    engineering.get_descriptor_indices(columns, descriptors, require_all=True)
    mask = run_path(output, 'best_descriptors', 'selected_descriptors.csv')
    pd.DataFrame([{name: 1 for name in descriptors}]).to_csv(mask, index=False)
    extract_test = engineering.feature_extraction_aminoacid_test if protein else engineering.feature_extraction_nucleotide_test
    test_names, test, test_labels, _ = extract_test(dataset['test_files'], dataset['test_labels'],
                                                  mask, columns, output, config['n_cpu'])
    inputs = dict(train=train, train_label=labels, train_nameseq=names,
                  test=test, test_label=test_labels, test_nameseq=test_names,
                  columns=columns, descriptors=descriptors)
    atomic_json(Path(output) / 'feature_inputs.json', inputs)
    # Record actual row-index folds used by production splitters, including row IDs.
    ids = pd.read_csv(names)['nameseq'].tolist()
    if len(ids) != dataset['train_count'] or len(pd.read_csv(test_names)) != dataset['test_count']:
        raise ValueError('Feature extraction changed the number of input records; inspect preprocessing before proceeding.')
    y = pd.read_csv(labels).iloc[:, 0].to_numpy() if dataset['task'] == 0 else np.array([float(value.split('|')[-1]) for value in ids])
    folds = {'sequence_ids': ids, 'seed': config['seed'], 'reporting_scope': 'post-selection optimization CV'}
    for count in (5, 10):
        splitter = StratifiedKFold if dataset['task'] == 0 else KFold
        folds[str(count)] = [{'train': train_idx.tolist(), 'validation': valid_idx.tolist()}
                             for train_idx, valid_idx in splitter(count, shuffle=True, random_state=config['seed']).split(ids, y)]
    atomic_json(Path(output) / 'reports/cv_folds.json', folds)


def select_features(config):
    import engineering
    dataset = config['dataset']
    inputs = read_json(Path(config['shared']) / 'feature_inputs.json')
    engineering.random_seed = config['seed']
    engineering.optimization_seed = config['search_seed']
    engineering.patience, engineering.difference = 80, 0.001
    select = engineering.feature_engineering_aminoacid if dataset['dtype'] == 'protein' else engineering.feature_engineering_nucleotide
    select(dataset['task'], config['estimations'], inputs['train_nameseq'], inputs['train'],
           inputs['train_label'], '', config['output'], config['n_cpu'], 1)


def train_condition(config, execution):
    import engineering
    dataset, output = config['dataset'], config['output']
    inputs = read_json(Path(config['shared']) / 'feature_inputs.json')
    representation, learner = config['condition'].split('_')
    if representation == 'selected':
        source_mask = run_path(config['stage_1'], 'best_descriptors', 'selected_descriptors.csv')
        mask = pd.read_csv(source_mask)
        columns = pd.read_csv(run_path(config['stage_1'], 'best_descriptors', 'best_train.csv'), nrows=0).columns.tolist()
        shutil.copytree(Path(config['stage_1']) / 'reports/optimization', Path(output) / 'reports/optimization')
    else:
        canonical = 'AAC' if dataset['dtype'] == 'protein' else 'TNC'
        mask = pd.DataFrame([{name: int(representation == 'all' or name == canonical) for name in inputs['descriptors']}])
        columns = inputs['columns'] if representation == 'all' else [name for name in inputs['columns'] if name.startswith(engineering.descriptor_prefix(canonical))]
    if not columns:
        raise ValueError('The selected representation has no features.')
    mask.to_csv(run_path(output, 'best_descriptors', 'selected_descriptors.csv'), index=False)
    copy_reports(config['shared'], output)
    with execution.phase('cached_feature_subset'):
        for split in ('train', 'test'):
            path = run_path(output, 'best_descriptors', f'best_{split}.csv')
            pd.read_csv(inputs[split], usecols=columns).loc[:, columns].to_csv(path, index=False)
            inputs[split] = path
    command = [sys.executable, str(PROJECT_PATH / 'generation.py'), '--task', str(dataset['task']),
               '--tuning', str(config['tuning'] if learner == 'tuned' else 0), '--n_cpu', str(config['n_cpu']),
               '--seed', str(config['seed']), '--search_seed', str(config['search_seed']),
               '--search_jobs', '1', '--output', output]
    for name in ('train', 'train_label', 'train_nameseq', 'test', 'test_label', 'test_nameseq'):
        command.extend([f'--{name}', inputs[name]])
    atomic_json(Path(output) / 'reports/ablation_condition.json', dict(
        condition=config['condition'], command=command, descriptors=int(mask.iloc[0].sum()), features=len(columns),
        shared_features=config['shared'], stage_1=config['stage_1'] if representation == 'selected' else None,
        timing_scope='Cached-condition execution; excludes shared extraction and separately measured Stage 1.'))
    subprocess.run(command, check=True)


def worker(config):
    if code_hashes() != config['code_sha256']:
        raise RuntimeError('Source code changed after protocol freeze; use a new experiment directory.')
    for path, expected in config['dataset']['input_hashes'].items():
        if file_hash(path) != expected:
            raise RuntimeError(f'Input changed after protocol freeze: {path}')
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    execution = Execution(config['output'], config['operation'], settings=config)
    try:
        execution.add_inputs(config['dataset']['train_files'] + config['dataset']['test_files'])
        if config['operation'] == 'extraction':
            extract_features(config)
        elif config['operation'] == 'selection':
            select_features(config)
        elif config['operation'] == 'condition':
            train_condition(config, execution)
        else:
            raise ValueError('Unknown worker operation.')
        for path, expected in config['dataset']['input_hashes'].items():
            if file_hash(path) != expected:
                raise RuntimeError(f'Input changed during execution: {path}')
    except BaseException:
        execution.finish('failed')
        raise
    else:
        execution.finish()


def validate_previous_step(config, compatible_runner_hashes, compatible_code_changes=()):
    output = Path(config['output'])
    record = read_json(output / 'run.json')
    previous = dict(record['settings'])
    expected = dict(config)
    previous_code = dict(previous.pop('code_sha256'))
    expected_code = dict(expected.pop('code_sha256'))
    runner = str(Path(__file__).resolve().relative_to(PROJECT_PATH))
    old_runner = previous_code.pop(runner)
    current_runner = expected_code.pop(runner)
    # Explicit, dataset-scoped migrations only; retain original run provenance.
    for change in compatible_code_changes:
        name = change['path']
        if (config.get('dataset', {}).get('dataset') in change['datasets']
                and previous_code.get(name) == change['before']
                and expected_code.get(name) == change['after']):
            previous_code[name] = change['after']
    if previous != expected or previous_code != expected_code:
        raise ValueError(f'Existing step has different inputs, settings or production code: {output}')
    if old_runner not in {current_runner, *compatible_runner_hashes}:
        raise ValueError(f'Existing step uses an unapproved runner version: {output}')
    if record['status'] == 'completed':
        missing = [name for name in record['outputs'] if not (output / name).is_file()]
        if missing:
            raise ValueError(f'Completed step has missing outputs: {output}: {missing[:3]}')
    return record['status']


def run_worker(root, config, resume=False, compatible_runner_hashes=(), compatible_code_changes=()):
    relative = Path(config['output']).relative_to(root)
    name = '__'.join(relative.parts)
    config_path = root / 'configs' / f'{name}.json'
    log_path = root / 'logs' / f'{name}.log'
    if Path(config['output']).exists():
        if not resume:
            raise FileExistsError(f'Existing step is preserved; use --resume: {config["output"]}')
        status = validate_previous_step(config, compatible_runner_hashes, compatible_code_changes)
        if status == 'completed':
            print(f'Reusing completed {relative}', flush=True)
            return
        # Preserve interrupted work and its log/config, but never continue a partial study.
        archive_root = root / 'interrupted_attempts' / relative
        archive_root.mkdir(parents=True, exist_ok=True)
        number = 1
        while (archive_root / f'attempt_{number}').exists():
            number += 1
        archive = archive_root / f'attempt_{number}'
        shutil.move(config['output'], archive)
        if config_path.exists():
            shutil.move(config_path, archive / 'worker_config.json')
        if log_path.exists():
            shutil.move(log_path, archive / 'worker.log')
        print(f'Archived incomplete {relative}; restarting this step with the same seed.', flush=True)
    atomic_json(config_path, config)
    print(f'Running {relative}; log: {log_path}', flush=True)
    start = time.perf_counter()
    environment = dict(os.environ, PYTHONHASHSEED=str(config['seed']))
    # These are local, unsigned experiments, never children of web jobs.
    environment.pop('BIOAUTOML_RUN_ROOT', None)
    with log_path.open('x') as log:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker_config', str(config_path)],
                       stdout=log, stderr=subprocess.STDOUT, env=environment, check=True)
    print(f'Completed {relative}: {(time.perf_counter() - start) / 60:.2f} minutes', flush=True)


def collect_result(config):
    output = Path(config['output'])
    dataset = config['dataset']
    record = read_json(output / 'run.json')
    if record['status'] != 'completed':
        raise RuntimeError(f'Incomplete condition: {output}')
    details = read_json(output / 'reports/ablation_condition.json')
    # Read only locally generated JSON metadata, never deserialize a model to summarize it.
    with zipfile.ZipFile(run_path(output, 'trained_model.sav')) as archive:
        model_info = json.loads(archive.read('manifest.json'))['info']
    performance = record['performance']
    shared = read_json(Path(config['shared']) / 'run.json')['performance']
    row = dict(dataset=dataset['dataset'], task='classification' if dataset['task'] == 0 else 'regression',
               dtype=dataset['dtype'], run=config['run'], search_seed=config['search_seed'], seed=config['seed'],
               condition=config['condition'], smoke_test=config['smoke_test'],
               train_records=dataset['train_count'], test_records=dataset['test_count'],
               descriptors=details['descriptors'], features=details['features'],
               primary_metric='MCC' if dataset['task'] == 0 else 'R2',
               estimator=model_info['estimator_name'], estimator_params=json.dumps(model_info['params'], sort_keys=True),
               condition_minutes=performance['total_seconds'] / 60,
               condition_peak_gib=performance['peak_memory_bytes'] / 1024**3,
               stage_2_minutes=(performance['stage_2_seconds'] or 0) / 60,
               shared_extraction_minutes=shared['descriptor_extraction_seconds'] / 60,
               shared_total_minutes=shared['total_seconds'] / 60,
               shared_peak_gib=shared['peak_memory_bytes'] / 1024**3,
               stage_1_minutes=0.0, output=str(output))
    if config['condition'].startswith('selected_'):
        selection = read_json(Path(config['stage_1']) / 'run.json')['performance']
        row.update(stage_1_minutes=selection['stage_1_seconds'] / 60,
                   stage_1_total_minutes=selection['total_seconds'] / 60,
                   stage_1_peak_gib=selection['peak_memory_bytes'] / 1024**3)
    filename = 'metrics_other.csv' if dataset['task'] == 0 else 'metrics_test.csv'
    metrics = pd.read_csv(run_path(output, filename))
    row.update({f'test_{item.Metric}': item.Value for item in metrics.itertuples()})
    cv = pd.read_csv(run_path(output, 'training_kfold(10)_metrics.csv')).iloc[0]
    row.update({f'optimization_cv_{name}': value for name, value in cv.items()})
    row['primary_score'] = row[f'test_{row["primary_metric"]}']
    if not math.isfinite(row['primary_score']):
        raise ValueError(f'Nonfinite primary test metric: {output}')
    if not math.isfinite(row[f'optimization_cv_{row["primary_metric"]}']):
        raise ValueError(f'Nonfinite primary optimization CV metric: {output}')
    for stage in ('stage_1', 'stage_2'):
        path = output / 'reports/optimization' / f'{stage}_study.json'
        if path.exists():
            history = read_json(path)
            row[f'{stage}_best_params'] = json.dumps(history['best_params'], sort_keys=True)
            row[f'{stage}_stop_reason'] = history['stop_reason']
            for state, count in history['trial_counts'].items():
                row[f'{stage}_{state.lower()}'] = count
    return row


def summarize(rows, root):
    frame = pd.DataFrame(rows)
    save_csv(frame, root / 'results.csv')
    keys = ['dataset', 'task', 'dtype', 'condition', 'primary_metric']
    additional = {}
    for name in frame.columns:
        if name.startswith('test_') and name != 'test_records':
            additional[f'{name}_mean'] = (name, 'mean')
            additional[f'{name}_sd'] = (name, 'std')
    summary = frame.groupby(keys, sort=False).agg(
        repeats=('primary_score', 'size'), primary_mean=('primary_score', 'mean'),
        primary_sd=('primary_score', 'std'), mean_features=('features', 'mean'),
        mean_condition_minutes=('condition_minutes', 'mean'), mean_stage_1_minutes=('stage_1_minutes', 'mean'),
        mean_stage_2_minutes=('stage_2_minutes', 'mean'), max_condition_peak_gib=('condition_peak_gib', 'max'),
        **additional).reset_index()
    save_csv(summary, root / 'summary.csv')
    paired_keys = ['dataset', 'task', 'dtype', 'run', 'search_seed', 'primary_metric']
    full = frame[frame.condition == 'selected_tuned'][paired_keys + ['primary_score']]
    paired = frame[frame.condition != 'selected_tuned'].merge(full, on=paired_keys, suffixes=('', '_full'))
    paired['full_minus_condition'] = paired['primary_score_full'] - paired['primary_score']
    save_csv(paired[paired_keys + ['condition', 'full_minus_condition']], root / 'paired_results.csv')
    summary = paired.groupby(keys, sort=False).agg(
        paired_repeats=('full_minus_condition', 'size'), mean_difference=('full_minus_condition', 'mean'),
        sd_difference=('full_minus_condition', 'std')).reset_index()
    save_csv(summary, root / 'paired_summary.csv')


def main():
    args = parse_arguments()
    os.chdir(PROJECT_PATH)
    if args.worker_config:
        worker(read_json(args.worker_config))
        return
    if os.environ.get('BIOAUTOML_WEB_EXECUTION') == '1':
        raise RuntimeError('Run manuscript experiments outside the web job queue.')
    datasets, inventory = input_inventory(args)
    print(inventory.groupby(['dataset', 'split'], sort=False).records.sum().unstack().to_string())
    print(f'{len(datasets)} datasets × 6 conditions × {args.runs} repeats = {len(datasets) * 6 * args.runs} results')
    if args.dry_run:
        print('Dry run only: no outputs written and no experiments launched.')
        return
    if not args.resume and args.output.exists():
        raise FileExistsError('Existing results are preserved; use --resume or a fresh output directory.')
    settings = dict(seed=args.seed, n_cpu=args.n_cpu, estimations=args.estimations, tuning=args.tuning,
                    smoke_test=args.smoke_test, code_sha256=code_hashes())
    protocol = dict(settings, datasets=[dataset['dataset'] for dataset in datasets], conditions=list(CONDITIONS),
                    search_seeds=list(range(6301, 6301 + args.runs)), search_jobs=1, stage_1_patience=80,
                    stage_1_min_delta=0.001, homology_aware=False, canonical={'protein': 'AAC', 'dnarna': 'TNC'},
                    note='Test labels never enter selection. Ten-fold results are post-selection optimization CV. '
                         'Default canonical/all repeats are deterministic controls, not independent searches. '
                         'Shared all-descriptor train/test extraction is measured once per dataset. Condition times '
                         'exclude shared work and Stage 1; do not interpret them as standalone end-to-end costs.',
                    status='running')
    previous_minutes = 0
    if args.resume:
        previous = read_json(args.output / 'protocol.json')
        if previous['status'] == 'running':
            raise ValueError('Stop the active experiment before resuming; running protocols cannot be resumed.')
        for key in protocol:
            if key not in ('status', 'note') and previous.get(key) != protocol[key]:
                raise ValueError(f'Resume protocol mismatch: {key}. Do not mix different experiments.')
        pd.testing.assert_frame_equal(pd.read_csv(args.output / 'input_inventory.csv'), inventory, check_dtype=False)
        previous_minutes = previous.get('total_minutes') or 0
        protocol = dict(previous, status='running')
        protocol['resume_count'] = previous.get('resume_count', 0) + 1
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        for directory in ('configs', 'logs'):
            (args.output / directory).mkdir()
        save_csv(inventory, args.output / 'input_inventory.csv')
    atomic_json(args.output / 'protocol.json', protocol)
    worker_options = dict(resume=args.resume, compatible_runner_hashes=protocol.get('compatible_runner_hashes', []),
                          compatible_code_changes=protocol.get('compatible_code_changes', []))
    rows, started = [], time.perf_counter()
    try:
        for dataset in datasets:
            dataset_path = args.output / dataset['dataset']
            shared = str(dataset_path / 'shared_features')
            common = dict(settings, dataset=dataset, shared=shared)
            run_worker(args.output, dict(common, operation='extraction', output=shared), **worker_options)
            for run_num in range(1, args.runs + 1):
                run_folder = dataset_path / 'runs' / f'run_{run_num}'
                stage_1 = str(run_folder / 'stage_1')
                config = dict(common, run=run_num, search_seed=6300 + run_num, stage_1=stage_1)
                run_worker(args.output, dict(config, operation='selection', output=stage_1), **worker_options)
                for condition in CONDITIONS:
                    current = dict(config, operation='condition', condition=condition, output=str(run_folder / condition))
                    run_worker(args.output, current, **worker_options)
                    rows.append(collect_result(current))
                    summarize(rows, args.output)
                    protocol['completed_results'] = len(rows)
                    atomic_json(args.output / 'protocol.json', protocol)
    except BaseException:
        protocol['status'] = 'failed'
        raise
    else:
        protocol['status'] = 'completed'
    finally:
        protocol.update(completed_results=len(rows), total_minutes=previous_minutes + (time.perf_counter() - started) / 60)
        atomic_json(args.output / 'protocol.json', protocol)
    print(f'Ablation finished: {len(rows)} results in {protocol["total_minutes"]:.2f} minutes. Output: {args.output}')


if __name__ == '__main__':
    main()
