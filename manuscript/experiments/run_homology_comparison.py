"""Paired ordinary versus homology-aware CV using the production two-stage search.

Five datasets, five search seeds, unchanged supplied test sets. Cache extraction
and reuse verified ordinary-CV ablation runs; never use test scores for selection.
"""
import argparse
import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ablation as ablation
from run_learner_aware import baseline_configs
from bioautoml.execution import Execution, atomic_json, file_hash, run_path, utc_now
from bioautoml.homology import get_homology_folds, get_mmseqs, prepare_homology

PROJECT_PATH = ablation.PROJECT_PATH
EXPERIMENTS_PATH = ablation.EXPERIMENTS_PATH
DATASETS = (2, 13, 29, 55, 63)
CONDITIONS = ('ordinary', 'homology')


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_cpu', type=int, default=32)
    parser.add_argument('--seed', type=int, default=63)
    parser.add_argument('--runs', type=int, choices=range(1, 6), default=5)
    parser.add_argument('--datasets', type=int, nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--estimations', type=int, default=200)
    parser.add_argument('--tuning', type=int, default=150)
    parser.add_argument('--datasets_root', type=Path, default=PROJECT_PATH / 'App/datasets')
    parser.add_argument('--ablation_root', type=Path, default=EXPERIMENTS_PATH / 'results/ablation')
    parser.add_argument('--output', type=Path, default=EXPERIMENTS_PATH / 'results/homology_comparison')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry_run', action='store_true', help='Check source compatibility; no output or MMseqs2 searches')
    parser.add_argument('--prepare_only', action='store_true', help='Validate grouped folds and audit overlap, without training')
    parser.add_argument('--smoke_test', action='store_true')
    parser.add_argument('--worker_config', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_config:
        return args
    if min(args.n_cpu, args.estimations, args.tuning) < 1 or not 0 <= args.seed < 2**32:
        parser.error('Use positive CPU/trial counts and a valid seed.')
    if len(set(args.datasets)) != len(args.datasets):
        parser.error('Dataset IDs must not repeat.')
    for name in ('datasets_root', 'ablation_root', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    results = EXPERIMENTS_PATH / 'results'
    if args.output == results or not args.output.is_relative_to(results):
        parser.error('Output must be below manuscript/experiments/results.')
    if args.output.is_relative_to(args.ablation_root) or args.ablation_root.is_relative_to(args.output):
        parser.error('Output and source directories must not overlap.')
    if not args.smoke_test and (tuple(args.datasets) != DATASETS or args.runs != 5
            or args.seed != 63 or args.n_cpu != 32 or args.estimations != 200 or args.tuning != 150
            or args.datasets_root != PROJECT_PATH / 'App/datasets'):
        parser.error('Changes to the full protocol require --smoke_test and a separate output.')
    if args.smoke_test and args.output == results / 'homology_comparison':
        parser.error('Smoke tests require a separate output directory.')
    return args


def code_hashes():
    paths = (Path(__file__).resolve(), EXPERIMENTS_PATH / 'run_learner_aware.py')
    return dict(ablation.code_hashes(), **{str(p.relative_to(PROJECT_PATH)): file_hash(p) for p in paths})


def package_versions():
    return {name: version(name) for name in
            ('scikit-learn', 'lightgbm', 'xgboost', 'optuna', 'numpy', 'joblib', 'pandas')}


def prepare_sources(args, datasets):
    protocol = ablation.read_json(args.ablation_root / 'protocol.json')
    if protocol['status'] != 'completed':
        raise ValueError('Source ablation must be completed.')
    source_args = SimpleNamespace(ablation_root=args.ablation_root, seed=args.seed, n_cpu=32,
                                  estimations=200, tuning=150, smoke_test=False)
    sources, assets, reusable = {}, {}, {}
    for dataset in datasets:
        configs = baseline_configs(source_args, dataset, 1)
        checked = [configs[0]]
        if not args.smoke_test:
            for run in range(1, args.runs + 1):
                if 6300 + run in protocol['search_seeds']:
                    selection, condition = baseline_configs(source_args, dataset, run)[1:]
                    checked.extend((selection, condition))
                    reusable[f'{dataset["dataset"]}/run_{run}'] = condition
        for config in checked:
            if ablation.validate_previous_step(config, protocol.get('compatible_runner_hashes', []),
                    protocol.get('compatible_code_changes', [])) != 'completed':
                raise ValueError(f'Incomplete source: {config["output"]}')
            record_path = Path(config['output']) / 'run.json'
            record = ablation.read_json(record_path)
            if any(package_versions().get(name) != value for name, value in record['packages'].items()):
                raise ValueError(f'Package mismatch: {record_path}')
            assets[str(record_path)] = file_hash(record_path)
            # Freeze locally generated results used by summaries and reference metadata.
            for relative in record['outputs']:
                path = Path(config['output']) / relative
                if path.suffix in ('.json', '.csv') and path.is_file():
                    assets[str(path)] = file_hash(path)
        shared = Path(configs[0]['output'])
        inputs_path = shared / 'feature_inputs.json'
        inputs = ablation.read_json(inputs_path)
        for path in [str(inputs_path)] + [inputs[key] for key in
                ('train', 'train_label', 'train_nameseq', 'test', 'test_label', 'test_nameseq')]:
            assets[path] = file_hash(path)
        sources[dataset['dataset']] = str(shared)
    return sources, assets, reusable


def verify_config(config):
    if config['code_sha256'] != code_hashes() or config['packages'] != package_versions():
        raise ValueError('Source code or package versions changed after protocol freeze.')
    if config['mmseqs'] != list(get_mmseqs()):
        raise ValueError('MMseqs2 executable/version changed.')
    for path, expected in {**config['dataset']['input_hashes'], **config['assets']}.items():
        if file_hash(path) != expected:
            raise ValueError(f'Input or cached asset changed: {path}')


def prepare_groups(config):
    dataset, output = config['dataset'], config['output']
    report = prepare_homology(dataset['train_files'], dataset['train_labels'], dataset['test_files'],
        dataset['test_labels'], 'Protein' if dataset['dtype'] == 'protein' else 'DNA/RNA',
        dataset['task'], config['seed'], run_path(output, 'homology'), enabled=True,
        n_cpu=config['n_cpu'], identity=90, coverage=80)
    inputs = ablation.read_json(Path(config['shared']) / 'feature_inputs.json')
    ids = pd.read_csv(inputs['train_nameseq'])['nameseq'].tolist()
    folds = dict(sequence_ids=ids, seed=config['seed'], reporting_scope='post-selection optimization CV')
    for count in (5, 10):
        folds[str(count)] = [dict(train=train.tolist(), validation=valid.tolist())
            for train, valid in get_homology_folds(report, ids, dataset['task'], config['seed'], count)]
    atomic_json(Path(output) / 'reports/cv_folds.json', folds)


def select_features(config):
    import engineering
    inputs = ablation.read_json(Path(config['shared']) / 'feature_inputs.json')
    engineering.random_seed = config['seed']
    engineering.optimization_seed = config['search_seed']
    engineering.patience, engineering.difference = 80, 0.001
    select = (engineering.feature_engineering_aminoacid if config['dataset']['dtype'] == 'protein'
              else engineering.feature_engineering_nucleotide)
    report = ablation.read_json(Path(config['preparation']) / 'reports/homology/homology_report.json') if config['cv'] == 'homology' else None
    select(config['dataset']['task'], config['estimations'], inputs['train_nameseq'], inputs['train'],
           inputs['train_label'], '', config['output'], config['n_cpu'], 1, homology_report=report)


def generation_command(config, inputs):
    command = [sys.executable, str(PROJECT_PATH / 'generation.py'), '--task', str(config['dataset']['task']),
        '--tuning', str(config['tuning']), '--n_cpu', str(config['n_cpu']), '--seed', str(config['seed']),
        '--search_seed', str(config['search_seed']), '--search_jobs', '1', '--output', config['output']]
    for name in ('train', 'train_label', 'train_nameseq', 'test', 'test_label', 'test_nameseq'):
        command.extend([f'--{name}', inputs[name]])
    if config['cv'] == 'homology':
        command.extend(['--homology_aware', '--homology_identity', '90', '--homology_coverage', '80'])
    return command


def train_condition(config, execution):
    output = Path(config['output'])
    inputs = ablation.read_json(Path(config['shared']) / 'feature_inputs.json')
    mask = pd.read_csv(run_path(config['stage_1'], 'best_descriptors', 'selected_descriptors.csv'))
    columns = pd.read_csv(run_path(config['stage_1'], 'best_descriptors', 'best_train.csv'), nrows=0).columns.tolist()
    if not columns:
        raise ValueError('No selected features.')
    shutil.copytree(Path(config['stage_1']) / 'reports/optimization', output / 'reports/optimization')
    mask.to_csv(run_path(output, 'best_descriptors', 'selected_descriptors.csv'), index=False)
    ablation.copy_reports(config['shared'], output)
    if config['cv'] == 'homology':
        # Replace cached ordinary-fold metadata with the frozen grouped assignments.
        shutil.copytree(Path(config['preparation']) / 'reports/homology', output / 'reports/homology', dirs_exist_ok=True)
        shutil.copy2(Path(config['preparation']) / 'reports/cv_folds.json', output / 'reports/cv_folds.json')
    with execution.phase('cached_feature_subset'):
        for split in ('train', 'test'):
            path = run_path(output, 'best_descriptors', f'best_{split}.csv')
            pd.read_csv(inputs[split], usecols=columns).loc[:, columns].to_csv(path, index=False)
            inputs[split] = path
    command = generation_command(config, inputs)
    atomic_json(output / 'reports/ablation_condition.json', dict(condition='selected_tuned', cv=config['cv'],
        command=command, descriptors=int(mask.iloc[0].sum()), features=len(columns), shared_features=config['shared'],
        stage_1=config['stage_1'], timing_scope='Cached condition, excluding extraction, grouping and Stage 1.'))
    subprocess.run(command, check=True)


def worker(config):
    verify_config(config)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    execution = Execution(config['output'], config['operation'], settings=config)
    try:
        execution.add_inputs(config['dataset']['train_files'] + config['dataset']['test_files'])
        if config['operation'] == 'preparation':
            prepare_groups(config)
        elif config['operation'] == 'selection':
            select_features(config)
        elif config['operation'] == 'condition':
            train_condition(config, execution)
        else:
            raise ValueError('Unknown worker operation.')
        verify_config(config)
    except BaseException:
        execution.finish('failed')
        raise
    else:
        execution.finish()


def run_worker(root, config, resume):
    output = Path(config['output'])
    name = '__'.join(output.relative_to(root).parts)
    log_path, config_path = root / 'logs' / f'{name}.log', root / 'configs' / f'{name}.json'
    if output.exists():
        if not resume:
            raise FileExistsError(f'Existing output preserved: {output}')
        record = ablation.read_json(output / 'run.json')
        if record['settings'] != config:
            raise ValueError(f'Resume configuration mismatch: {output}')
        if record['status'] == 'completed':
            if any(not (output / item).is_file() for item in record['outputs']):
                raise ValueError(f'Missing completed outputs: {output}')
            print(f'Reusing {output.relative_to(root)}', flush=True)
            return
        archive = root / 'interrupted_attempts' / name
        archive.mkdir(parents=True, exist_ok=True)
        number = 1
        while (archive / f'attempt_{number}').exists():
            number += 1
        target = archive / f'attempt_{number}'
        shutil.move(output, target)
        for source, filename in ((log_path, 'worker.log'), (config_path, 'worker_config.json')):
            if source.exists():
                shutil.move(source, target / filename)
    atomic_json(config_path, config)
    print(f'Running {output.relative_to(root)}; log: {log_path}', flush=True)
    environment = dict(os.environ, PYTHONHASHSEED=str(config['seed']))
    environment.pop('BIOAUTOML_RUN_ROOT', None)
    with log_path.open('x') as log:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker_config', str(config_path)],
                       stdout=log, stderr=subprocess.STDOUT, env=environment, check=True)


def summarize(rows, root):
    frame = pd.DataFrame(rows)
    ablation.save_csv(frame, root / 'results.csv')
    keys = ['dataset', 'task', 'dtype', 'cv', 'primary_metric']
    metrics = {f'{name}_{stat}': (name, stat) for name in frame.columns
               if name.startswith('test_') and name != 'test_records' for stat in ('mean', 'std')}
    summary = frame.groupby(keys, sort=False).agg(repeats=('primary_score', 'size'),
        primary_mean=('primary_score', 'mean'), primary_sd=('primary_score', 'std'),
        mean_features=('features', 'mean'), mean_pipeline_minutes=('pipeline_minutes', 'mean'),
        max_peak_gib=('pipeline_peak_gib', 'max'), **metrics).reset_index()
    ablation.save_csv(summary, root / 'summary.csv')
    keys = ['dataset', 'task', 'dtype', 'run', 'search_seed', 'primary_metric']
    ordinary = frame[frame.cv == 'ordinary'][keys + ['primary_score', 'pipeline_minutes']]
    pairs = frame[frame.cv == 'homology'].merge(ordinary, on=keys, suffixes=('', '_ordinary'), validate='one_to_one')
    pairs['homology_minus_ordinary'] = pairs.primary_score - pairs.primary_score_ordinary
    pairs['extra_minutes'] = pairs.pipeline_minutes - pairs.pipeline_minutes_ordinary
    ablation.save_csv(pairs[keys + ['homology_minus_ordinary', 'extra_minutes']], root / 'paired_results.csv')
    grouped = pairs.groupby(['dataset', 'task', 'dtype', 'primary_metric'], sort=False).agg(
        paired_repeats=('homology_minus_ordinary', 'size'), mean_difference=('homology_minus_ordinary', 'mean'),
        sd_difference=('homology_minus_ordinary', 'std'), mean_extra_minutes=('extra_minutes', 'mean')).reset_index()
    ablation.save_csv(grouped, root / 'paired_summary.csv')


def execute(args, datasets, inventory, sources, assets, reusable):
    settings = dict(seed=args.seed, n_cpu=args.n_cpu, estimations=args.estimations, tuning=args.tuning,
        packages=package_versions(), code_sha256=code_hashes(), mmseqs=list(get_mmseqs()), smoke_test=args.smoke_test)
    protocol = dict(settings, datasets=datasets, search_seeds=list(range(6301, 6301 + args.runs)),
        identity=90, coverage=80, search_jobs=1, assets=assets, sources=sources, reused_baselines=reusable,
        timing_scope='Additive extraction + grouping/audit + Stage 1 + condition times, not standalone wall time. '
                     'Reused results retain original execution conditions; grouping/audit shared across seeds.',
        evaluation='Unchanged supplied test sets. Optimization CV is post-selection, not nested. '
                   '90% identity screens high-identity redundancy, not remote homology.')
    protocol_path = args.output / 'protocol.json'
    if protocol_path.exists():
        if not args.resume or ablation.read_json(protocol_path)['configuration'] != protocol:
            raise ValueError('Existing experiment requires --resume with an identical protocol.')
    elif args.resume:
        raise FileNotFoundError('No experiment exists to resume.')
    for name in ('configs', 'logs'):
        (args.output / name).mkdir(exist_ok=True)
    ablation.save_csv(inventory, args.output / 'input_inventory.csv')
    state = dict(configuration=protocol, status='preparing', completed_results=0, started_at=utc_now())
    atomic_json(protocol_path, state)
    rows, started, preparations = [], time.perf_counter(), {}
    try:
        overlaps = []
        # Check every dataset before spending any budget on model searches.
        for dataset in datasets:
            shared = sources[dataset['dataset']]
            preparation = str(args.output / dataset['dataset'] / 'preparation')
            relevant = {path: digest for path, digest in assets.items() if dataset['dataset'] in Path(path).parts}
            common = dict(settings, dataset=dataset, shared=shared, assets=relevant, preparation=preparation)
            run_worker(args.output, dict(common, operation='preparation', output=preparation), args.resume)
            for path in (Path(preparation) / 'reports').rglob('*'):
                if path.is_file():
                    relevant[str(path)] = file_hash(path)
            # Copy, so the preparation's original configuration remains unchanged.
            preparations[dataset['dataset']] = dict(common, assets=dict(relevant))
            overlap = ablation.read_json(Path(preparation) / 'reports/homology/overlap_report.json')
            report = ablation.read_json(Path(preparation) / 'reports/homology/homology_report.json')
            overlaps.append(dict(dataset=dataset['dataset'], groups=report['groups'], **overlap))
            ablation.save_csv(pd.DataFrame(overlaps), args.output / 'overlap_summary.csv')
        state['status'] = 'prepared' if args.prepare_only else 'running'
        atomic_json(protocol_path, state)
        if args.prepare_only:
            return
        for dataset in datasets:
            common = preparations[dataset['dataset']]
            preparation_record = ablation.read_json(Path(common['preparation']) / 'run.json')['performance']
            for run in range(1, args.runs + 1):
                for cv in CONDITIONS:
                    reused = cv == 'ordinary' and f'{dataset["dataset"]}/run_{run}' in reusable
                    if reused:
                        current = reusable[f'{dataset["dataset"]}/run_{run}']
                    else:
                        folder = args.output / dataset['dataset'] / 'runs' / f'run_{run}' / cv
                        config = dict(common, cv=cv, run=run, search_seed=6300 + run, stage_1=str(folder / 'stage_1'))
                        run_worker(args.output, dict(config, operation='selection', output=config['stage_1']), args.resume)
                        current = dict(config, operation='condition', condition='selected_tuned', output=str(folder / 'selected_tuned'))
                        run_worker(args.output, current, args.resume)
                    verify_config(common)
                    row = ablation.collect_result(current)
                    preparation_minutes = preparation_record['total_seconds'] / 60 if cv == 'homology' else 0
                    row.update(cv=cv, reused_from_ablation=reused, preparation_minutes=preparation_minutes,
                        pipeline_minutes=row['shared_total_minutes'] + row['stage_1_total_minutes'] + row['condition_minutes'] + preparation_minutes,
                        pipeline_peak_gib=max(row['shared_peak_gib'], row['stage_1_peak_gib'], row['condition_peak_gib'],
                            preparation_record['peak_memory_bytes'] / 1024**3 if cv == 'homology' else 0))
                    rows.append(row)
                    summarize(rows, args.output)
                    state['completed_results'] = len(rows)
                    atomic_json(protocol_path, state)
    except BaseException:
        state['status'] = 'failed'
        raise
    else:
        state['status'] = 'completed'
    finally:
        state.update(completed_results=len(rows), invocation_minutes=(time.perf_counter() - started) / 60, updated_at=utc_now())
        atomic_json(protocol_path, state)


def main():
    args = parse_arguments()
    os.chdir(PROJECT_PATH)
    if args.worker_config:
        worker(ablation.read_json(args.worker_config))
        return
    if os.environ.get('BIOAUTOML_WEB_EXECUTION') == '1':
        raise RuntimeError('Run manuscript experiments outside the web queue.')
    datasets, inventory = ablation.input_inventory(args)
    sources, assets, reusable = prepare_sources(args, datasets)
    print(inventory.groupby(['dataset', 'split'], sort=False).records.sum().unstack().to_string(), flush=True)
    print(f'MMseqs2: {get_mmseqs()}; identity 90%, bidirectional coverage 80%.', flush=True)
    print(f'{len(datasets) * args.runs * 2} results; {len(reusable)} reused ordinary results; '
          f'{len(datasets) * args.runs * 2 - len(reusable)} new pipelines.', flush=True)
    if args.dry_run:
        print('Dry run: source compatibility verified; no writes or searches. Group feasibility checked during preparation.')
        return
    if args.output.exists() and not args.resume:
        raise FileExistsError('Existing output preserved; use --resume or a fresh output path.')
    if args.resume and not args.output.exists():
        raise FileNotFoundError('No output directory exists to resume.')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args, datasets, inventory, sources, assets, reusable)


if __name__ == '__main__':
    main()
