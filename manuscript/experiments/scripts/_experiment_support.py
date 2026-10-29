"""Shared validation and isolated production workers for the remaining experiments."""
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import random
import runpy
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

SCRIPTS_PATH = Path(__file__).resolve().parent
EXPERIMENTS_PATH = SCRIPTS_PATH.parent
PROJECT_PATH = EXPERIMENTS_PATH.parents[1]
RESULTS_PATH = EXPERIMENTS_PATH / 'results'
sys.path.insert(0, str(EXPERIMENTS_PATH))
sys.path.insert(0, str(PROJECT_PATH))
import run_ablation as ablation
import run_benchmarks as benchmarks
import run_homology_comparison as homology
from run_learner_aware import baseline_configs
from bioautoml.execution import Execution, atomic_json, file_hash, run_path, utc_now


def code_hashes():
    paths = list(SCRIPTS_PATH.glob('*.py')) + [EXPERIMENTS_PATH / name for name in
             ('run_benchmarks.py', 'run_learner_aware.py', 'run_homology_comparison.py')]
    return dict(ablation.code_hashes(), **{str(p.relative_to(PROJECT_PATH)): file_hash(p) for p in paths})


def settings(args):
    return dict(seed=63, n_cpu=args.n_cpu, estimations=args.estimations, tuning=args.tuning,
                search_jobs=1, smoke_test=args.smoke_test, packages=homology.package_versions(), code_sha256=code_hashes())


def validate_record(path):
    path = Path(path)
    record = ablation.read_json(path / 'run.json')
    if record['status'] != 'completed':
        raise ValueError(f'Incomplete source: {path}')
    if any(not (path / item).is_file() for item in record['outputs']):
        raise ValueError(f'Missing source outputs: {path}')
    if any(homology.package_versions().get(k) != v for k, v in record['packages'].items()):
        raise ValueError(f'Package mismatch: {path}')
    return record


def check_history(path, run):
    for stage, cap in (('stage_1', 200), ('stage_2', 150)):
        history = ablation.read_json(Path(path) / 'reports/optimization' / f'{stage}_study.json')
        values = history['settings']
        if (history['status'] != 'completed' or history['pruner'] != 'MedianPruner'
                or values['seed'] != 63 or values['search_seed'] != 6300 + run
                or values['requested_trials'] != cap or values['folds'] != 5
                or values['search_jobs'] != 1 or values['model_jobs'] != 32):
            raise ValueError(f'Incompatible ordinary-CV search: {path}/{stage}')


def prepare_sources(datasets, include_reuse=True):
    """Validate source protocols and fingerprint all tables used by new workers."""
    first_root, ablation_root = RESULTS_PATH / 'benchmarks', RESULTS_PATH / 'ablation'
    first = ablation.read_json(first_root / 'protocol.json')
    old = first['settings']
    if (first['status'] != 'completed' or old['code_sha256'] != ablation.code_hashes()
            or old['n_cpu'] != 32 or old['seed'] != 63 or old['homology_aware']
            or old['estimations'] != 200 or old['tuning'] != 150):
        raise ValueError('First benchmark protocol is not compatible with current production code/settings.')
    source_datasets = {d['dataset']: d for d in old['datasets']}
    index = pd.read_csv(first_root / 'results_index.csv').set_index('dataset')
    ap = ablation.read_json(ablation_root / 'protocol.json')
    if ap['status'] != 'completed':
        raise ValueError('Source ablation must be complete.')
    source_args = SimpleNamespace(ablation_root=ablation_root, seed=63, n_cpu=32,
                                  estimations=200, tuning=150, smoke_test=False)
    sources, reused, assets = {}, {}, {}
    for dataset in datasets:
        name = dataset['dataset']
        if dataset != source_datasets[name]:
            raise ValueError(f'Dataset inventory changed: {name}')
        print(f'Validating source/cache: {name}', flush=True)
        baseline = Path(index.loc[name, 'output'])
        baseline_record = validate_record(baseline)
        check_history(baseline, 1)
        if name in (ablation.DATASETS.values()):
            source_configs = baseline_configs(source_args, dataset, 1)
            for config in source_configs:
                if ablation.validate_previous_step(config, ap.get('compatible_runner_hashes', []),
                        ap.get('compatible_code_changes', [])) != 'completed':
                    raise ValueError(f'Invalid ablation baseline: {name}')
            shared = Path(source_configs[0]['output'])
            inputs = ablation.read_json(shared / 'feature_inputs.json')
            if include_reuse:
                for run in (1, 2, 3):
                    configs = baseline_configs(source_args, dataset, run)
                    for config in configs[1:]:
                        ablation.validate_previous_step(config, ap.get('compatible_runner_hashes', []),
                                                        ap.get('compatible_code_changes', []))
                    path = configs[-1]['output']
                    validate_record(path)
                    check_history(path, run)
                    reused[f'{name}/run_{run}'] = dict(output=path, source='ablation')
        else:
            shared = baseline
            inputs = {key: run_path(shared, 'feat_extraction', filename) for key, filename in
                (('train', 'ftrain.csv'), ('train_label', 'flabeltrain.csv'), ('train_nameseq', 'fnameseqtrain.csv'))}
            # First benchmark extracted only its selected test descriptors. Do not reuse that test matrix.
            if baseline_record['code_sha256'] != {k: ablation.code_hashes()[k] for k in baseline_record['code_sha256']}:
                raise ValueError(f'Production code mismatch: {baseline}')
            if baseline_record['settings']['search_seed'] != 6301 or baseline_record['settings']['n_cpu'] != 32:
                raise ValueError(f'First-run settings mismatch: {baseline}')
        validate_record(shared)
        names = pd.read_csv(inputs['train_nameseq'])['nameseq']
        if len(names) != dataset['train_count'] or names.duplicated().any():
            raise ValueError(f'Cached training row/ID mismatch: {name}')
        if len(pd.read_csv(inputs['train_label'])) != len(names):
            raise ValueError(f'Cached labels do not align: {name}')
        columns = pd.read_csv(inputs['train'], nrows=0).columns.tolist()
        import engineering
        descriptors = engineering.AMINOACID_DESCRIPTORS if dataset['dtype'] == 'protein' else engineering.NUCLEOTIDE_DESCRIPTORS
        engineering.get_descriptor_indices(columns, descriptors, require_all=True)
        if 'test' in inputs:
            if len(pd.read_csv(inputs['test_nameseq'])) != dataset['test_count']:
                raise ValueError(f'Cached test row count mismatch: {name}')
            if pd.read_csv(inputs['test'], nrows=0).columns.tolist() != columns:
                raise ValueError(f'Cached full test columns mismatch: {name}')
        paths = [shared / 'run.json', shared / 'timings.csv']
        paths += [Path(inputs[k]) for k in ('train', 'train_label', 'train_nameseq', 'test', 'test_label', 'test_nameseq') if k in inputs]
        paths += [p for p in (shared / 'reports').rglob('*') if p.is_file()]
        assets[name] = {str(path): file_hash(path) for path in paths}
        times = pd.read_csv(shared / 'timings.csv')
        cache_minutes = float(times.loc[times.phase.isin(['training_features', 'test_features'] if 'test' in inputs else ['training_features']), 'elapsed_seconds'].sum()) / 60
        sources[name] = dict(shared=str(shared), inputs=inputs, cached_extraction_minutes=cache_minutes,
                             full_test_cache='test' in inputs)
        if include_reuse:
            reused.setdefault(f'{name}/run_1', dict(output=str(baseline), source='first_benchmark'))
    if include_reuse:
        hp = ablation.read_json(RESULTS_PATH / 'homology_comparison/protocol.json')
        if hp['status'] != 'completed':
            raise ValueError('Homology comparison is not complete.')
        for dataset in datasets:
            if dataset['dataset_id'] not in homology.DATASETS:
                continue
            for run in (4, 5):
                path = RESULTS_PATH / 'homology_comparison' / dataset['dataset'] / 'runs' / f'run_{run}' / 'ordinary/selected_tuned'
                record = validate_record(path)
                config = record['settings']
                homology.verify_config(config)
                if config['cv'] != 'ordinary' or config['dataset'] != dataset or config['search_seed'] != 6300 + run:
                    raise ValueError(f'Incompatible homology-study baseline: {path}')
                check_history(path, run)
                reused[f'{dataset["dataset"]}/run_{run}'] = dict(output=str(path), source='homology_ordinary')
    # Preserve output provenance separately from feature-cache provenance.
    for reference in reused.values():
        path = Path(reference['output'])
        reference['sha256'] = {str(p): file_hash(p) for p in
            [path / 'run.json', Path(run_path(path, 'trained_model.sav'))] + list((path / 'results/metrics').glob('*.csv'))}
    return sources, assets, reused


def verify_config(config):
    if config['code_sha256'] != code_hashes() or config['packages'] != homology.package_versions():
        raise ValueError('Code or environment changed after protocol freeze.')
    for path, expected in {**config['dataset']['input_hashes'], **config['assets']}.items():
        if file_hash(path) != expected:
            raise ValueError(f'Input/cache changed: {path}')


@contextmanager
def pruning_context(enabled):
    import optuna
    original = optuna.create_study

    def create_study(*args, **kwargs):
        kwargs['pruner'] = (optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0,
                           interval_steps=1, n_min_trials=1) if enabled else optuna.pruners.NopPruner())
        return original(*args, **kwargs)

    with patch.object(optuna, 'create_study', create_study):
        yield


def generation_arguments(config, inputs):
    args = ['--task', str(config['dataset']['task']), '--tuning', str(config['tuning']),
            '--n_cpu', str(config['n_cpu']), '--seed', '63', '--search_seed', str(config['search_seed']),
            '--search_jobs', '1', '--output', config['output']]
    for name in ('train', 'train_label', 'train_nameseq', 'test', 'test_label', 'test_nameseq'):
        if inputs.get(name):
            args.extend([f'--{name}', inputs[name]])
    return args


def worker(config):
    verify_config(config)
    random.seed(63)
    np.random.seed(63)
    output, dataset = Path(config['output']), config['dataset']
    execution = Execution(output, 'training', settings=config)
    try:
        import engineering
        source = config['source']
        inputs = dict(source['inputs'])
        ablation.copy_reports(source['shared'], output)
        execution.add_inputs(dataset['train_files'] + dataset['test_files'])
        engineering.random_seed = 63
        engineering.optimization_seed = config['search_seed']
        engineering.patience, engineering.difference = 80, 0.001
        select = engineering.feature_engineering_aminoacid if dataset['dtype'] == 'protein' else engineering.feature_engineering_nucleotide
        with pruning_context(config['pruning']):
            select(dataset['task'], config['estimations'], inputs['train_nameseq'], inputs['train'],
                   inputs['train_label'], '', str(output), config['n_cpu'], 1)
        inputs['train'] = run_path(output, 'best_descriptors', 'best_train.csv')
        columns = pd.read_csv(inputs['train'], nrows=0).columns.tolist()
        mask = run_path(output, 'best_descriptors', 'selected_descriptors.csv')
        if dataset['test_count']:
            if source['full_test_cache']:
                with execution.phase('cached_test_subset'):
                    target = run_path(output, 'best_descriptors', 'best_test.csv')
                    pd.read_csv(inputs['test'], usecols=columns).loc[:, columns].to_csv(target, index=False)
                    inputs['test'] = target
            else:
                extract = engineering.feature_extraction_aminoacid_test if dataset['dtype'] == 'protein' else engineering.feature_extraction_nucleotide_test
                inputs['test_nameseq'], inputs['test'], inputs['test_label'], _ = extract(
                    dataset['test_files'], dataset['test_labels'], mask, columns, str(output), config['n_cpu'])
        command = [sys.executable, str(Path(__file__).resolve()), '--generation',
                   'on' if config['pruning'] else 'off', *generation_arguments(config, inputs)]
        atomic_json(output / 'reports/pipeline.json', dict(command=command, cached_extraction_minutes=source['cached_extraction_minutes'],
            features=len(columns), descriptors=int(pd.read_csv(mask).iloc[0].sum()), pruning=config['pruning'],
            timing_scope='Measured cached pipeline plus separately recorded original extraction cost; not standalone end-to-end.'))
        subprocess.run(command, check=True)
        # Carry available descriptive statistics without loading source estimators.
        stats_path = Path(run_path(source['shared'], 'train_stats.csv'))
        if stats_path.exists():
            from bioautoml.model_artifacts import update_model_summary
            with execution.phase('benchmark_summary'):
                shutil.copy2(stats_path, run_path(output, 'train_stats.csv'))
                update_model_summary(run_path(output, 'trained_model.sav'), trust_unsigned=True,
                                     train_stats=pd.read_csv(stats_path))
        verify_config(config)
    except BaseException:
        execution.finish('failed')
        raise
    else:
        execution.finish()


def run_worker(root, config, resume):
    output = Path(config['output'])
    name = '__'.join(output.relative_to(root).parts)
    config_path, log_path = root / 'configs' / f'{name}.json', root / 'logs' / f'{name}.log'
    if output.exists():
        if not resume:
            raise FileExistsError(f'Existing output preserved: {output}')
        record = ablation.read_json(output / 'run.json')
        if record['settings'] != config:
            raise ValueError(f'Resume mismatch: {output}')
        if record['status'] == 'completed':
            validate_record(output)
            print(f'Reusing {output.relative_to(root)}', flush=True)
            return
        # Do not restart a still-running worker left behind by an interrupted controller.
        pid_path = output.parent / f'{output.name}.pid'
        if pid_path.exists():
            import psutil
            try:
                process = psutil.Process(int(pid_path.read_text()))
                if str(config_path) in process.cmdline():
                    raise RuntimeError(f'Worker still active: {output}')
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                pass
        archive = root / 'interrupted_attempts' / name
        archive.mkdir(parents=True, exist_ok=True)
        number = 1
        while (archive / f'attempt_{number}').exists():
            number += 1
        target = archive / f'attempt_{number}'
        shutil.move(output, target)
        for path, filename in ((log_path, 'worker.log'), (config_path, 'config.json')):
            if path.exists():
                shutil.move(path, target / filename)
    atomic_json(config_path, config)
    environment = dict(os.environ, PYTHONHASHSEED='63')
    environment.pop('BIOAUTOML_RUN_ROOT', None)
    print(f'Running {output.relative_to(root)}; log: {log_path}', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('x') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(config_path)],
                                   env=environment, stdout=log, stderr=subprocess.STDOUT)
        (output.parent / f'{output.name}.pid').write_text(str(process.pid))
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, process.args)


def collect_result(dataset, run, output, source='new', pruning=True):
    output = Path(output)
    record = validate_record(output)
    performance = record['performance']
    row = dict(dataset=dataset['dataset'], task=dataset['task'], dtype=dataset['dtype'], run=run,
        search_seed=6300 + run, source=source, pruning=pruning, output=str(output),
        primary_metric='MCC' if dataset['task'] == 0 else 'R2', test_available=bool(dataset['test_count']),
        measured_minutes=performance['total_seconds'] / 60, peak_gib=performance['peak_memory_bytes'] / 1024**3)
    for phase in ('descriptor_extraction', 'stage_1', 'stage_2', 'optimisation'):
        value = performance.get(f'{phase}_seconds')
        row[f'{phase}_minutes'] = value / 60 if value is not None else None
    details_path = output / 'reports/pipeline.json'
    if details_path.exists():
        details = ablation.read_json(details_path)
        row.update(features=details['features'], descriptors=details['descriptors'],
                   cached_extraction_minutes=details['cached_extraction_minutes'],
                   additive_pipeline_minutes=row['measured_minutes'] + details['cached_extraction_minutes'])
    cv = pd.read_csv(run_path(output, 'training_kfold(10)_metrics.csv')).iloc[0]
    row.update({f'optimization_cv_{k}': v for k, v in cv.items()})
    if dataset['test_count']:
        metrics = pd.read_csv(run_path(output, 'metrics_other.csv' if dataset['task'] == 0 else 'metrics_test.csv'))
        row.update({f'test_{item.Metric}': item.Value for item in metrics.itertuples()})
        if not math.isfinite(row[f'test_{row["primary_metric"]}']):
            raise ValueError(f'Nonfinite test primary metric: {output}')
    if not math.isfinite(row[f'optimization_cv_{row["primary_metric"]}']):
        raise ValueError(f'Nonfinite optimization CV metric: {output}')
    for stage in ('stage_1', 'stage_2'):
        history = ablation.read_json(output / 'reports/optimization' / f'{stage}_study.json')
        row[f'{stage}_pruner'] = history['pruner']
        row[f'{stage}_stop_reason'] = history['stop_reason']
        row.update({f'{stage}_{k.lower()}': v for k, v in history['trial_counts'].items()})
    return row


def save_results(rows, root, pruning=False):
    frame = pd.DataFrame(rows)
    ablation.save_csv(frame, root / 'results.csv')
    keys = ['dataset', 'task', 'dtype', 'primary_metric'] + (['pruning'] if pruning else [])
    metrics = [c for c in frame if c.startswith('test_') and c != 'test_available']
    metrics += [c for c in frame if c.startswith('optimization_cv_') and '_std_' not in c]
    summary = frame.groupby(keys, sort=False).agg(repeats=('run', 'size'),
        **{f'{c}_{stat}': (c, stat) for c in metrics for stat in ('mean', 'std')}).reset_index()
    ablation.save_csv(summary, root / 'summary.csv')
    if pruning:
        paired = ['dataset', 'task', 'run', 'search_seed', 'primary_metric']
        frame['primary_score'] = [r[f'test_{r["primary_metric"]}'] for r in rows]
        pairs = frame[~frame.pruning].merge(frame[frame.pruning], on=paired, suffixes=('_off', '_on'), validate='one_to_one')
        pairs['score_off_minus_on'] = pairs.primary_score_off - pairs.primary_score_on
        pairs['minutes_off_minus_on'] = pairs.measured_minutes_off - pairs.measured_minutes_on
        ablation.save_csv(pairs, root / 'paired_results.csv')
        grouped = pairs.groupby(['dataset', 'task', 'primary_metric'], sort=False).agg(
            paired_repeats=('run', 'size'), mean_score_difference=('score_off_minus_on', 'mean'),
            sd_score_difference=('score_off_minus_on', 'std'), mean_extra_minutes=('minutes_off_minus_on', 'mean')).reset_index()
        ablation.save_csv(grouped, root / 'paired_summary.csv')


@contextmanager
def experiment(root, configuration, resume):
    root = Path(root)
    if root.exists() and not resume:
        raise FileExistsError(f'Existing output preserved: {root}')
    if resume and not (root / 'protocol.json').exists():
        raise FileNotFoundError(f'No protocol to resume: {root}')
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root / 'protocol.json'
        if path.exists() and ablation.read_json(path)['configuration'] != configuration:
            raise ValueError('Resume configuration/input/code mismatch.')
        for name in ('configs', 'logs'):
            (root / name).mkdir(exist_ok=True)
        state = dict(configuration=configuration, status='running', completed_results=0, started_at=utc_now())
        atomic_json(path, state)
        started = time.perf_counter()
        try:
            yield state
        except BaseException:
            state['status'] = 'failed'
            raise
        else:
            state['status'] = 'completed'
        finally:
            state.update(invocation_minutes=(time.perf_counter() - started) / 60, updated_at=utc_now())
            atomic_json(path, state)


def validate_output(path, default, smoke=False):
    path = path.resolve()
    if path == RESULTS_PATH or not path.is_relative_to(RESULTS_PATH):
        raise ValueError('Output must be below manuscript/experiments/results.')
    protected = ('ablation', 'benchmarks', 'learner_aware', 'trial_budget_sweep', 'homology_comparison')
    if any(path.is_relative_to(RESULTS_PATH / name) for name in protected):
        raise ValueError('Existing experiment directories are protected.')
    if smoke and path == RESULTS_PATH / default:
        raise ValueError('Smoke tests require a separate output directory.')
    return path


if __name__ == '__main__':
    os.chdir(PROJECT_PATH)
    if sys.argv[1] == '--worker':
        worker(ablation.read_json(sys.argv[2]))
    elif sys.argv[1] == '--generation':
        enabled = sys.argv[2] == 'on'
        sys.argv = [str(PROJECT_PATH / 'generation.py'), *sys.argv[3:]]
        with pruning_context(enabled):
            runpy.run_path(str(PROJECT_PATH / 'generation.py'), run_name='__main__')
    else:
        raise ValueError('Internal worker arguments required.')
