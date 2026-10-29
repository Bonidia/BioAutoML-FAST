"""Paired, stage-specific trial-budget diagnostics using production objectives.

Stage 1 patience is disabled only in isolated diagnostic workers. Stage 2 uses
each dataset's production seed-6301 descriptor set for every search seed.
Scores are internal optimization CV, not external or unbiased performance.
"""
import argparse
import fcntl
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ablation as ablation
from run_learner_aware import baseline_configs
from bioautoml.execution import Execution, atomic_json, file_hash, run_path, utc_now

PROJECT_PATH = ablation.PROJECT_PATH
EXPERIMENTS_PATH = ablation.EXPERIMENTS_PATH
DATASETS = (2, 13, 29, 55, 63)
BUDGETS = {'stage_1': (25, 50, 100, 200), 'stage_2': (25, 50, 100, 150, 200)}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_cpu', type=int, default=32)
    parser.add_argument('--seed', type=int, default=63)
    parser.add_argument('--runs', type=int, choices=range(1, 6), default=5)
    parser.add_argument('--datasets', type=int, nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--max_trials', type=int, default=200)
    parser.add_argument('--datasets_root', type=Path, default=PROJECT_PATH / 'App/datasets')
    parser.add_argument('--ablation_root', type=Path, default=EXPERIMENTS_PATH / 'results/ablation')
    parser.add_argument('--output', type=Path, default=EXPERIMENTS_PATH / 'results/trial_budget_sweep')
    parser.add_argument('--wait_for_benchmark', type=Path,
                        help='Wait for this benchmark protocol.json to become completed; abort if it fails')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke_test', action='store_true')
    parser.add_argument('--worker_config', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_config:
        return args
    if args.n_cpu < 1 or args.max_trials < 1 or not 0 <= args.seed < 2**32:
        parser.error('Use positive CPU/trial counts and a seed in [0, 2**32).')
    if len(set(args.datasets)) != len(args.datasets):
        parser.error('Dataset IDs must not repeat.')
    for name in ('datasets_root', 'ablation_root', 'output', 'wait_for_benchmark'):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    results = EXPERIMENTS_PATH / 'results'
    if args.output == results or not args.output.is_relative_to(results):
        parser.error('Output must be a subdirectory of manuscript/experiments/results.')
    if args.output.is_relative_to(args.ablation_root) or args.ablation_root.is_relative_to(args.output):
        parser.error('Sweep and ablation directories must not overlap.')
    if args.wait_for_benchmark and args.wait_for_benchmark.is_relative_to(args.output):
        parser.error('Cannot wait for a benchmark inside the sweep output.')
    if not args.smoke_test and (args.runs != 5 or args.max_trials != 200 or tuple(args.datasets) != DATASETS
                               or args.seed != 63 or args.n_cpu != 32):
        parser.error('The full protocol uses five datasets/seeds, 200 trials, seed 63 and 32 CPUs; otherwise use --smoke_test.')
    if args.smoke_test and args.output == results / 'trial_budget_sweep':
        parser.error('Smoke tests require a separate output directory.')
    return args


def code_hashes():
    paths = [Path(__file__).resolve(), EXPERIMENTS_PATH / 'run_learner_aware.py']
    return dict(ablation.code_hashes(), **{str(p.relative_to(PROJECT_PATH)): file_hash(p) for p in paths})


def package_versions():
    return {name: version(name) for name in
            ('scikit-learn', 'lightgbm', 'xgboost', 'optuna', 'numpy', 'joblib', 'pandas')}


def reusable_history(history, maximum):
    trials = history['trials']
    return (history['status'] == 'completed' and history['stop_reason'] == 'trial_budget'
            and len(trials) == maximum and [t['number'] for t in trials] == list(range(maximum))
            and all(t['state'] in ('COMPLETE', 'PRUNED') for t in trials))


def prepare_sources(args, datasets):
    protocol = ablation.read_json(args.ablation_root / 'protocol.json')
    if protocol['status'] != 'completed':
        raise ValueError('The source ablation must be complete.')
    source_args = SimpleNamespace(ablation_root=args.ablation_root, seed=protocol['seed'],
        n_cpu=protocol['n_cpu'], estimations=200, tuning=150, smoke_test=False)
    sources, assets, reusable = {}, {}, {}
    for dataset in datasets:
        configs = baseline_configs(source_args, dataset, 1)
        # Validate extraction and Stage 1 only; no external test scores are read.
        for config in configs[:2]:
            status = ablation.validate_previous_step(config, protocol.get('compatible_runner_hashes', []),
                                                     protocol.get('compatible_code_changes', []))
            if status != 'completed':
                raise ValueError(f'Incomplete source: {config["output"]}')
            record_path = Path(config['output']) / 'run.json'
            record = ablation.read_json(record_path)
            if any(package_versions().get(name) != value for name, value in record['packages'].items()):
                raise ValueError(f'Package versions differ from source: {record_path}')
            assets[str(record_path)] = file_hash(record_path)
        shared, selection = Path(configs[0]['output']), Path(configs[1]['output'])
        inputs_path = shared / 'feature_inputs.json'
        inputs = ablation.read_json(inputs_path)
        source = dict(shared=str(shared), train=inputs['train'], labels=inputs['train_label'],
                      names=inputs['train_nameseq'], fixed_train=run_path(selection, 'best_descriptors', 'best_train.csv'),
                      fixed_mask=run_path(selection, 'best_descriptors', 'selected_descriptors.csv'),
                      reference_search_seed=6301)
        for path in [str(inputs_path), source['train'], source['labels'], source['names'],
                     source['fixed_train'], source['fixed_mask']]:
            assets[path] = file_hash(path)
        sources[dataset['dataset']] = source
        if args.smoke_test:
            continue
        for run in range(1, args.runs + 1):
            if 6300 + run not in protocol['search_seeds']:
                continue
            config = baseline_configs(source_args, dataset, run)[1]
            if ablation.validate_previous_step(config, protocol.get('compatible_runner_hashes', []),
                    protocol.get('compatible_code_changes', [])) != 'completed':
                continue
            path = Path(config['output']) / 'reports/optimization/stage_1_study.json'
            history = ablation.read_json(path)
            if reusable_history(history, args.max_trials):
                assets[str(path)] = file_hash(path)
                record_path = Path(config['output']) / 'run.json'
                assets[str(record_path)] = file_hash(record_path)
                reusable[f'{dataset["dataset"]}/run_{run}/stage_1'] = str(path)
    return sources, assets, reusable


def verify_config(config):
    if config['code_sha256'] != code_hashes() or config['packages'] != package_versions():
        raise ValueError('Source code or environment changed after protocol freeze.')
    for path, expected in dict(config['assets'], **config['dataset']['input_hashes']).items():
        if file_hash(path) != expected:
            raise ValueError(f'Input or cache changed after freeze: {path}')


def diagnostic_optimize(original, *args, **kwargs):
    settings = dict(kwargs['settings'], patience=None, production_patience=80,
                    diagnostic='Patience disabled for the budget sweep; production objective and pruning unchanged.')
    kwargs.update(settings=settings, early_stopping=None)
    return original(*args, **kwargs)


def worker(config):
    verify_config(config)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    dataset, source = config['dataset'], config['source']
    execution = Execution(config['output'], 'trial_budget_sweep', settings=config)
    try:
        execution.add_inputs(dataset['train_files'])
        if config['stage'] == 'stage_1':
            import engineering
            original = engineering.optimize_study
            with patch.object(engineering, 'optimize_study',
                              side_effect=lambda *a, **kw: diagnostic_optimize(original, *a, **kw)):
                ablation.select_features(dict(config, shared=source['shared'], estimations=config['max_trials']))
        else:
            import generation
            from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
            generation.random_seed, generation.optimization_seed = config['seed'], config['search_seed']
            train = pd.read_csv(source['fixed_train'])
            string_columns = train.select_dtypes(include=['object']).columns
            if len(string_columns):
                train[string_columns] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit_transform(train[string_columns])
            train = train.replace([np.inf, -np.inf], np.nan)
            if dataset['task'] == 0:
                y = LabelEncoder().fit_transform(pd.read_csv(source['labels']).iloc[:, 0])
            else:
                y = np.array([float(name.split('|')[-1]) for name in pd.read_csv(source['names']).nameseq])
            if len(train) != len(y):
                raise ValueError('Fixed features and training targets have different row counts.')
            with execution.phase('stage_2'):
                generation.get_best_model_optuna(train, y, dataset['task'], config['max_trials'],
                    config['n_cpu'], 1, output=config['output'])
        verify_config(config)
    except BaseException:
        execution.finish('failed')
        raise
    else:
        execution.finish()


def checkpoint_rows(history, budgets, metadata):
    trials = sorted(history['trials'], key=lambda trial: trial['number'])
    if [trial['number'] for trial in trials] != list(range(len(trials))):
        raise ValueError('History trial numbers are not a contiguous zero-based trajectory.')
    rows, previous = [], None
    maximize = history['direction'] == 'MAXIMIZE'
    for budget in budgets:
        prefix = trials[:budget]
        reached = len(prefix) == budget and all(t['state'] in ('COMPLETE', 'PRUNED', 'FAIL') for t in prefix)
        complete = [t for t in prefix if t['state'] == 'COMPLETE'
                    and t['value'] is not None and math.isfinite(t['value'])]
        best = (max if maximize else min)(complete, key=lambda t: t['value']) if reached and complete else None
        score = best['value'] if best else None
        gain = None if previous is None or score is None else (score - previous if maximize else previous - score)
        elapsed = prefix[-1]['completed_elapsed_seconds'] if reached else None
        rows.append(dict(metadata, budget=budget, reached=reached,
            metric=history['settings']['metric'], direction=history['direction'],
            best_value=score, gain_from_previous=gain,
            best_trial=best['number'] if best else None,
            best_params=json.dumps(best['params'], sort_keys=True) if best else None,
            optimization_minutes=elapsed / 60 if elapsed is not None else None,
            complete=sum(t['state'] == 'COMPLETE' for t in prefix),
            pruned=sum(t['state'] == 'PRUNED' for t in prefix),
            failed=sum(t['state'] == 'FAIL' for t in prefix),
            actual_trials=len(prefix), trajectory_trials=len(trials), stop_reason=history['stop_reason']))
        previous = score
    return rows


def summarize(rows, output, plots=False):
    frame = pd.DataFrame(rows)
    ablation.save_csv(frame, output / 'checkpoints.csv')
    summary = frame.groupby(['dataset', 'stage', 'metric', 'budget'], sort=False).agg(
        trajectories=('run', 'size'), reached_seeds=('reached', 'sum'),
        score_mean=('best_value', 'mean'), score_sd=('best_value', 'std'),
        optimization_minutes_mean=('optimization_minutes', 'mean'),
        optimization_minutes_sd=('optimization_minutes', 'std'),
        gain_mean=('gain_from_previous', 'mean')).reset_index()
    ablation.save_csv(summary, output / 'summary.csv')
    if not plots:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    datasets = frame.dataset.unique()
    for value, spread, title in [('score_mean', 'score_sd', 'Internal optimization CV (mean ± SD)'),
                                  ('optimization_minutes_mean', 'optimization_minutes_sd', 'Optimization time (minutes; mean ± SD)')]:
        fig, axes = plt.subplots(len(datasets), 2, figsize=(11, 3 * len(datasets)), squeeze=False)
        for i, dataset in enumerate(datasets):
            for j, stage in enumerate(BUDGETS):
                data = summary[(summary.dataset == dataset) & (summary.stage == stage)].sort_values('budget')
                ax = axes[i, j]
                ax.errorbar(data.budget, data[value], yerr=data[spread].fillna(0), marker='o', capsize=3)
                ax.set(title=f'{dataset.split("_")[0]} · {stage}', xlabel='Attempted trials',
                       ylabel=(data.metric.iloc[0] if len(data) else '') if value == 'score_mean' else 'Minutes')
                ax.grid(alpha=0.2)
        fig.suptitle(title + '\nStage 2: fixed seed-6301 descriptors; missing checkpoints are not extrapolated')
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        for suffix in ('pdf', 'png'):
            fig.savefig(output / f'{value}.{suffix}')
        plt.close(fig)


def run_worker(root, config, resume):
    output = Path(config['output'])
    name = '__'.join(output.relative_to(root).parts)
    log_path, config_path = root / 'logs' / f'{name}.log', root / 'configs' / f'{name}.json'
    if output.exists():
        if not resume:
            raise FileExistsError(f'Existing result preserved: {output}')
        record = ablation.read_json(output / 'run.json')
        if record['settings'] != config:
            raise ValueError(f'Resume configuration changed: {output}')
        if record['status'] == 'completed':
            history = output / 'reports/optimization' / f'{config["stage"]}_study.json'
            if ablation.read_json(history)['status'] != 'completed':
                raise ValueError(f'Missing/incomplete history: {history}')
            return
        archive = root / 'interrupted_attempts' / name
        archive.mkdir(parents=True, exist_ok=True)
        number = 1
        while (archive / f'attempt_{number}').exists():
            number += 1
        target = archive / f'attempt_{number}'
        shutil.move(output, target)
        for path in (log_path, config_path):
            if path.exists():
                shutil.move(path, target / path.name)
    atomic_json(config_path, config)
    print(f'Running {output.relative_to(root)}; log: {log_path}', flush=True)
    environment = dict(os.environ, PYTHONHASHSEED=str(config['seed']))
    environment.pop('BIOAUTOML_RUN_ROOT', None)
    with log_path.open('x') as log:
        subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--worker_config', str(config_path)],
                       stdout=log, stderr=subprocess.STDOUT, env=environment, check=True)


def wait_for_benchmark(path, state, protocol_path):
    if path is None:
        return
    print(f'Waiting for benchmark completion: {path}', flush=True)
    while True:
        benchmark = ablation.read_json(path)
        status = benchmark['status']
        state.update(status='queued', last_checked_at=utc_now(), benchmark_status=status,
                     benchmark_completed=benchmark.get('completed_results'))
        atomic_json(protocol_path, state)
        if status == 'completed':
            return
        if status != 'running':
            raise RuntimeError(f'Benchmark is {status}; sweep was not started.')
        time.sleep(45)


def main():
    args = parse_arguments()
    os.chdir(PROJECT_PATH)
    if args.worker_config:
        worker(ablation.read_json(args.worker_config))
        return
    datasets, inventory = ablation.input_inventory(args)
    sources, assets, reusable = prepare_sources(args, datasets)
    settings = dict(seed=args.seed, n_cpu=args.n_cpu, max_trials=args.max_trials,
                    packages=package_versions(), code_sha256=code_hashes(), smoke_test=args.smoke_test)
    budgets = {stage: [1, args.max_trials] if args.max_trials > 1 else [1] for stage in BUDGETS} if args.smoke_test else {k: list(v) for k, v in BUDGETS.items()}
    configuration = dict(settings, datasets=datasets, sources=sources, assets=assets,
        reusable_histories=reusable, search_seeds=list(range(6301, 6301 + args.runs)), budgets=budgets,
        wait_for_benchmark=str(args.wait_for_benchmark) if args.wait_for_benchmark else None,
        note='Stage-level sensitivity, not a grid of full pipelines. Stage 1 patience disabled for new diagnostic '
             'searches. Reused 200-trial histories never triggered patience. Stage 2 holds production seed-6301 '
             'descriptors fixed across seeds. Original startup/pruning/timeouts retained. Scores are optimization '
             'CV only. COMPLETE/PRUNED/FAIL count against caps; only COMPLETE trials compete. Extraction is '
             'cached and excluded from optimization timings. Reused histories retain original timing conditions.')
    total = len(datasets) * args.runs * 2
    print(f'{total} trajectories: {len(reusable)} reusable Stage 1 histories; {total - len(reusable)} new searches.', flush=True)
    if args.dry_run:
        print('Dry run: validation complete; nothing written or launched.', flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=args.resume)
    # A process-held lock permits safe restart after a crash, but not duplicate runners.
    with (args.output / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol_path = args.output / 'protocol.json'
        if args.resume:
            state = ablation.read_json(protocol_path)
            if state['configuration'] != configuration:
                raise ValueError('Resume protocol differs; existing outputs preserved.')
        else:
            state = dict(configuration=configuration, status='queued', completed_trajectories=0,
                         total_trajectories=total, created_at=utc_now())
            for name in ('configs', 'logs'):
                (args.output / name).mkdir()
            ablation.save_csv(inventory, args.output / 'input_inventory.csv')
        atomic_json(protocol_path, state)
        rows, started = [], None
        try:
            wait_for_benchmark(args.wait_for_benchmark, state, protocol_path)
            started = time.perf_counter()
            state.update(status='running', started_at=utc_now())
            atomic_json(protocol_path, state)
            for dataset in datasets:
                source = sources[dataset['dataset']]
                dataset_assets = {p: h for p, h in assets.items() if f'/{dataset["dataset"]}/' in p}
                for run in range(1, args.runs + 1):
                    for stage in BUDGETS:
                        key = f'{dataset["dataset"]}/run_{run}/{stage}'
                        output = args.output / dataset['dataset'] / 'runs' / f'run_{run}' / stage
                        config = dict(settings, dataset=dataset, source=source, assets=dataset_assets,
                                      run=run, search_seed=6300 + run, stage=stage, output=str(output))
                        verify_config(config)
                        reused = key in reusable
                        if reused:
                            history_path = Path(reusable[key])
                            output.mkdir(parents=True, exist_ok=args.resume)
                            atomic_json(output / 'reuse.json', dict(history=str(history_path), sha256=assets[str(history_path)],
                                note='Production patience never triggered; this is the complete 200-trial trajectory.'))
                        else:
                            run_worker(args.output, config, args.resume)
                            history_path = output / 'reports/optimization' / f'{stage}_study.json'
                        metadata = dict(dataset=dataset['dataset'], task=dataset['task'], run=run,
                            search_seed=6300 + run, stage=stage, reused=reused, history=str(history_path))
                        rows.extend(checkpoint_rows(ablation.read_json(history_path), budgets[stage], metadata))
                        summarize(rows, args.output)
                        state['completed_trajectories'] = len({(r['dataset'], r['run'], r['stage']) for r in rows})
                        atomic_json(protocol_path, state)
            summarize(rows, args.output, plots=True)
        except BaseException as error:
            state.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        else:
            state['status'] = 'completed'
        finally:
            state.update(ended_at=utc_now(), invocation_minutes=(time.perf_counter() - started) / 60 if started else None)
            atomic_json(protocol_path, state)


if __name__ == '__main__':
    main()
