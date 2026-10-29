"""Compare fixed Stage 1 learners, followed by the unchanged production Stage 2.

Default: three paired seeds on five datasets; reuse verified LightGBM ablation
results and cached extraction. Run --dry_run first. Use --fresh to run all arms.
"""
import argparse
from contextlib import nullcontext
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
from bioautoml.execution import Execution, atomic_json, file_hash

PROJECT_PATH = ablation.PROJECT_PATH
EXPERIMENTS_PATH = ablation.EXPERIMENTS_PATH
DATASETS = (2, 13, 29, 55, 63)
LEARNERS = ('lightgbm', 'random_forest', 'xgboost')


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_cpu', type=int, default=32)
    parser.add_argument('--seed', type=int, default=63)
    parser.add_argument('--datasets', type=int, nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--runs', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--estimations', type=int, default=200)
    parser.add_argument('--tuning', type=int, default=150)
    parser.add_argument('--datasets_root', type=Path, default=PROJECT_PATH / 'App/datasets')
    parser.add_argument('--ablation_root', type=Path, default=EXPERIMENTS_PATH / 'results/ablation')
    parser.add_argument('--output', type=Path, default=EXPERIMENTS_PATH / 'results/learner_aware')
    parser.add_argument('--fresh', action='store_true', help='Extract features and run all three learners without ablation reuse')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry_run', action='store_true', help='Validate inputs and reuse readiness; write nothing')
    parser.add_argument('--smoke_test', action='store_true')
    parser.add_argument('--worker_config', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_config:
        return args
    if args.n_cpu < 1 or not 0 <= args.seed < 2**32 or min(args.estimations, args.tuning) < 1:
        parser.error('Use positive CPU/trial counts and a seed in [0, 2**32).')
    if len(set(args.datasets)) != len(args.datasets):
        parser.error('Dataset IDs must not repeat.')
    for name in ('datasets_root', 'ablation_root', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    if not args.smoke_test and (args.runs != 3 or args.estimations != 200 or args.tuning != 150
                               or args.datasets_root != PROJECT_PATH / 'App/datasets'):
        parser.error('Reduced budgets/repeats or alternate inputs require --smoke_test.')
    results = (EXPERIMENTS_PATH / 'results').resolve()
    if args.output == results or not args.output.is_relative_to(results):
        parser.error('--output must be below manuscript/experiments/results.')
    if args.output.is_relative_to(args.ablation_root) or args.ablation_root.is_relative_to(args.output):
        parser.error('Output and ablation directories must not overlap.')
    if args.smoke_test and args.output == results / 'learner_aware':
        parser.error('Smoke tests require a separate --output directory.')
    return args


def code_hashes():
    return dict(ablation.code_hashes(), **{str(Path(__file__).resolve().relative_to(PROJECT_PATH)): file_hash(__file__)})


def make_learner(learner, task, seed, n_cpu):
    """Fixed, untuned candidates. Resolved library defaults are recorded before search."""
    if learner == 'lightgbm':
        from lightgbm import LGBMClassifier, LGBMRegressor
        cls = LGBMClassifier if task == 0 else LGBMRegressor
        return cls(random_state=seed, n_jobs=n_cpu, verbosity=-1)
    if learner == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        cls = RandomForestClassifier if task == 0 else RandomForestRegressor
        return cls(n_estimators=100, random_state=seed, n_jobs=n_cpu)
    if learner == 'xgboost':
        from xgboost import XGBClassifier, XGBRegressor
        cls = XGBClassifier if task == 0 else XGBRegressor
        return cls(n_estimators=100, tree_method='hist', device='cpu', random_state=seed,
                   n_jobs=n_cpu, verbosity=0)
    raise ValueError(f'Unknown learner: {learner}')


def learner_parameters(model):
    return {key: str(value) if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in model.get_params().items()}


def select_features(config):
    import engineering
    learner = config['learner']
    model = make_learner(learner, config['dataset']['task'], config['seed'], config['n_cpu'])
    atomic_json(Path(config['output']) / 'reports/selection_learner.json',
                dict(learner=learner, estimator=type(model).__name__, params=learner_parameters(model),
                     note='Fixed Stage 1 learner only; production Stage 2 is unchanged.'))
    # Scoped to this isolated worker. Reuse the exact production objective, folds,
    # cache, sampler, pruning and stopping; never patch the installed LightGBM module.
    replacement = SimpleNamespace(
        LGBMClassifier=lambda **kw: make_learner(learner, 0, kw['random_state'], kw['n_jobs']),
        LGBMRegressor=lambda **kw: make_learner(learner, 1, kw['random_state'], kw['n_jobs']))
    context = nullcontext() if learner == 'lightgbm' else patch.object(engineering, 'lgb', replacement)
    with context:
        ablation.select_features(config)


def verify_inputs(config):
    if config['code_sha256'] != code_hashes():
        raise ValueError('Source changed after protocol freeze.')
    for path, expected in config['dataset']['input_hashes'].items():
        if file_hash(path) != expected:
            raise ValueError(f'Input changed: {path}')


def worker(config):
    verify_inputs(config)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    execution = Execution(config['output'], config['operation'], settings=config)
    try:
        execution.add_inputs(config['dataset']['train_files'] + config['dataset']['test_files'])
        if config['operation'] == 'extraction':
            ablation.extract_features(config)
        elif config['operation'] == 'selection':
            select_features(config)
        elif config['operation'] == 'condition':
            ablation.train_condition(config, execution)
        else:
            raise ValueError('Unknown operation.')
        verify_inputs(config)
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
            raise ValueError(f'Resume settings/code mismatch: {output}')
        if record['status'] == 'completed':
            if any(not (output / name).is_file() for name in record['outputs']):
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


def baseline_configs(args, dataset, run):
    shared = str(args.ablation_root / dataset['dataset'] / 'shared_features')
    common = dict(seed=args.seed, n_cpu=args.n_cpu, estimations=args.estimations, tuning=args.tuning,
                  smoke_test=args.smoke_test, code_sha256=ablation.code_hashes(), dataset=dataset, shared=shared)
    stage_1 = str(args.ablation_root / dataset['dataset'] / 'runs' / f'run_{run}' / 'stage_1')
    selection = dict(common, run=run, search_seed=6300 + run, stage_1=stage_1)
    return (dict(common, operation='extraction', output=shared),
            dict(selection, operation='selection', output=stage_1),
            dict(selection, operation='condition', condition='selected_tuned',
                 output=str(Path(stage_1).parent / 'selected_tuned')))


def validate_baselines(args, datasets):
    protocol = ablation.read_json(args.ablation_root / 'protocol.json')
    if protocol['status'] == 'running':
        raise ValueError('Ablation is still running. Wait until it stops before starting this experiment.')
    references = []
    for dataset in datasets:
        for run in range(1, args.runs + 1):
            for config in baseline_configs(args, dataset, run):
                status = ablation.validate_previous_step(config, protocol.get('compatible_runner_hashes', []),
                                                        protocol.get('compatible_code_changes', []))
                if status != 'completed':
                    raise ValueError(f'Incomplete baseline: {config["output"]}')
                references.append(dict(output=config['output'], run_sha256=file_hash(Path(config['output']) / 'run.json')))
    return references


def summarize(rows, root):
    frame = pd.DataFrame(rows)
    ablation.save_csv(frame, root / 'results.csv')
    keys = ['dataset', 'task', 'dtype', 'learner', 'primary_metric']
    summary = frame.groupby(keys, sort=False).agg(
        repeats=('primary_score', 'size'), primary_mean=('primary_score', 'mean'),
        primary_sd=('primary_score', 'std'), mean_features=('features', 'mean'),
        mean_pipeline_minutes=('pipeline_minutes', 'mean'), max_peak_gib=('pipeline_peak_gib', 'max')).reset_index()
    ablation.save_csv(summary, root / 'summary.csv')
    paired_keys = ['dataset', 'task', 'dtype', 'run', 'search_seed', 'primary_metric']
    baseline = frame[frame.learner == 'lightgbm'][paired_keys + ['primary_score', 'pipeline_minutes']]
    pairs = frame[frame.learner != 'lightgbm'].merge(baseline, on=paired_keys, suffixes=('', '_lightgbm'))
    pairs['alternative_minus_lightgbm'] = pairs.primary_score - pairs.primary_score_lightgbm
    pairs['extra_minutes'] = pairs.pipeline_minutes - pairs.pipeline_minutes_lightgbm
    ablation.save_csv(pairs[paired_keys + ['learner', 'alternative_minus_lightgbm', 'extra_minutes']], root / 'paired_results.csv')
    paired = pairs.groupby(keys, sort=False).agg(paired_repeats=('alternative_minus_lightgbm', 'size'),
        mean_difference=('alternative_minus_lightgbm', 'mean'), sd_difference=('alternative_minus_lightgbm', 'std'),
        mean_extra_minutes=('extra_minutes', 'mean')).reset_index()
    ablation.save_csv(paired, root / 'paired_summary.csv')


def main():
    args = parse_arguments()
    os.chdir(PROJECT_PATH)
    if args.worker_config:
        worker(ablation.read_json(args.worker_config))
        return
    if os.environ.get('BIOAUTOML_WEB_EXECUTION') == '1':
        raise RuntimeError('Run manuscript experiments outside the web queue.')
    datasets, inventory = ablation.input_inventory(args)
    print(inventory.groupby(['dataset', 'split'], sort=False).records.sum().unstack().to_string())
    count = len(datasets) * args.runs
    print(f'{count * 3} results: {count * (3 if args.fresh else 2)} new pipelines, {0 if args.fresh else count} reused.')
    references = []
    if not args.fresh:
        try:
            references = validate_baselines(args, datasets)
        except (ValueError, FileNotFoundError) as error:
            if not args.dry_run:
                raise
            print(f'Reuse not ready: {error}')
    if args.dry_run:
        print('Dry run: nothing written or launched.')
        return
    settings = dict(seed=args.seed, n_cpu=args.n_cpu, estimations=args.estimations, tuning=args.tuning,
                    smoke_test=args.smoke_test, code_sha256=code_hashes())
    protocol = dict(settings, datasets=datasets, learners=list(LEARNERS),
                    search_seeds=list(range(6301, 6301 + args.runs)), fresh=args.fresh,
                    baseline_references=references, learner_parameters={name: {
                        str(task): learner_parameters(make_learner(name, task, args.seed, args.n_cpu))
                        for task in (0, 1)} for name in LEARNERS},
                    search_jobs=1, homology_aware=False,
                    timing_scope='Pipeline minutes = shared extraction + Stage 1 + condition wall times; '
                                 'an additive estimate, not a standalone end-to-end benchmark. '
                                 'Do not compare cached wall time alone; reused runs retain original timings.')
    # Normalize estimator parameters to their persisted JSON representation.
    protocol = json.loads(json.dumps(protocol, default=str, allow_nan=False))
    if args.output.exists():
        if not args.resume:
            raise FileExistsError('Existing results preserved; use --resume or a fresh output path.')
        previous = ablation.read_json(args.output / 'protocol.json')
        if previous['status'] == 'running' or previous['configuration'] != protocol:
            raise ValueError('Active experiment or mismatched resume protocol.')
    else:
        if args.resume:
            raise FileNotFoundError('No experiment exists to resume.')
        args.output.mkdir(parents=True)
        for name in ('configs', 'logs'):
            (args.output / name).mkdir()
        ablation.save_csv(inventory, args.output / 'input_inventory.csv')
    state = dict(configuration=protocol, status='running', completed_results=0)
    atomic_json(args.output / 'protocol.json', state)
    rows, started = [], time.perf_counter()
    try:
        for dataset in datasets:
            shared = str((args.output if args.fresh else args.ablation_root) / dataset['dataset'] / 'shared_features')
            common = dict(settings, dataset=dataset, shared=shared)
            if args.fresh:
                run_worker(args.output, dict(common, operation='extraction', output=shared), args.resume)
            for run in range(1, args.runs + 1):
                for learner in LEARNERS:
                    reused = learner == 'lightgbm' and not args.fresh
                    if reused:
                        current = baseline_configs(args, dataset, run)[2]
                    else:
                        folder = args.output / dataset['dataset'] / 'runs' / f'run_{run}' / learner
                        stage_1 = str(folder / 'stage_1')
                        config = dict(common, run=run, search_seed=6300 + run, learner=learner, stage_1=stage_1)
                        run_worker(args.output, dict(config, operation='selection', output=stage_1), args.resume)
                        current = dict(config, operation='condition', condition='selected_tuned', output=str(folder / 'selected_tuned'))
                        run_worker(args.output, current, args.resume)
                    row = ablation.collect_result(current)
                    row.update(learner=learner, reused_from_ablation=reused,
                        pipeline_minutes=row['shared_total_minutes'] + row['stage_1_total_minutes'] + row['condition_minutes'],
                        pipeline_peak_gib=max(row['shared_peak_gib'], row['stage_1_peak_gib'], row['condition_peak_gib']))
                    rows.append(row)
                    summarize(rows, args.output)
                    state['completed_results'] = len(rows)
                    atomic_json(args.output / 'protocol.json', state)
    except BaseException:
        state['status'] = 'failed'
        raise
    else:
        state['status'] = 'completed'
    finally:
        state.update(completed_results=len(rows), invocation_minutes=(time.perf_counter() - started) / 60)
        atomic_json(args.output / 'protocol.json', state)


if __name__ == '__main__':
    main()
