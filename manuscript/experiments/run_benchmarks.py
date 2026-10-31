"""Run or resume seeded benchmarks, sharing verified descriptor caches."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import pandas as pd
from Bio import SeqIO

PROJECT_PATH = Path(__file__).resolve().parents[2]
EXPERIMENTS_PATH = Path(__file__).resolve().parent
RESULTS_PATH = EXPERIMENTS_PATH / 'results'
DATASETS_PATH = PROJECT_PATH / 'App/datasets'
sys.path.insert(0, str(PROJECT_PATH))
from bioautoml.execution import Execution, run_path, atomic_json, file_hash, utc_now

# Reviewed transition: explicit MedianPruner defaults preserve the previous searches.
PRUNING_COMPATIBILITY = {
    'engineering.py': (
        'b8d115124515e5e29fcaf514f8fa513c84d868cf72c4c3a41640a203e6865467',
        '7b77b1912f5c434fc95a3899107070724ee5d7aede77397c58f0d3397fc9c321'),
    'generation.py': (
        '6996a0c3c2e2b798c10aa6ba161d330e0bdcbc112433ab489d0d5febde0255b3',
        '9735c633fef92ffa51200fbd37b5f6799cc9f1d25881af3647560ccbdfc51d6f'),
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_cpu', type=int, default=32)
    parser.add_argument('--seed', type=int, default=63)
    parser.add_argument('--runs', type=int, choices=range(1, 6), default=5)
    parser.add_argument('--datasets', type=int, nargs='+')
    parser.add_argument('--estimations', type=int, default=200)
    parser.add_argument('--tuning', type=int, default=150)
    parser.add_argument('--pruning', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--output', type=Path, default=DATASETS_PATH)
    parser.add_argument('--reuse_ablation', type=Path, default=RESULTS_PATH / 'ablation')
    parser.add_argument('--reuse_from', type=Path, nargs='*', default=[], help='Additional benchmark directories to check for compatible results/features')
    parser.add_argument('--dry_run', action='store_true', help='Validate and report without writing or training')
    parser.add_argument('--resume', action='store_true', help='Accepted for existing commands; resume is automatic')
    args = parser.parse_args()
    if min(args.n_cpu, args.estimations, args.tuning) < 1 or not 0 <= args.seed * 100 + args.runs < 2**32 or args.seed < 0:
        parser.error('Use positive CPU/trial counts and a nonnegative seed with seed * 100 + runs < 2**32.')
    args.output = args.output.resolve()
    inputs = DATASETS_PATH.resolve()
    protected = [args.reuse_ablation.resolve()]
    if args.output != inputs:
        protected.append(inputs)
    if args.output == RESULTS_PATH or any(args.output == p or args.output in p.parents
                                          or p in args.output.parents for p in protected):
        parser.error('Output must not overlap source datasets or the ablation.')
    return args


def input_inventory():
    datasets, rows = [], []
    directories = sorted(DATASETS_PATH.glob('dataset*'),
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



def runtime_hashes():
    import run_ablation
    return {key: value for key, value in run_ablation.code_hashes().items()
            if not key.startswith('manuscript/')}


def check_code(previous, current):
    for name, value in current.items():
        old = previous.get(name)
        if old != value and (old, value) != PRUNING_COMPATIBILITY.get(name):
            raise ValueError(f'Incompatible source code: {name}')


def check_hashes(values):
    for filename, expected in values.items():
        if not Path(filename).is_file() or file_hash(filename) != expected:
            raise ValueError(f'Missing or changed input/artifact: {filename}')


def check_source_code(record, output, dataset, current):
    previous = dict(record['settings'].get('code_sha256', {}))
    context = source_context(output)
    previous = dict(context['runtime_sha256'], **previous)
    previous.update(record['code_sha256'])
    for change in context['compatible_code_changes']:
        name = change['path']
        if (dataset['dataset'] in change['datasets'] and previous.get(name) == change['before']
                and current.get(name) == change['after']):
            previous[name] = change['after']
    check_code(previous, current)


def source_context(output):
    saved = Path(output) / 'reports/benchmark_source.json'
    context = read_json(saved) if saved.exists() else {}
    if 'runtime_sha256' in context:
        return context
    for parent in Path(output).parents:
        path = source_metadata(parent) / 'protocol.json'
        if path.exists():
            protocol = read_json(path)
            config = protocol.get('configuration', protocol.get('settings', protocol))
            context.update(output=str(output), protocol=str(path), protocol_sha256=file_hash(path),
                           runtime_sha256=config.get('code_sha256', config.get('runtime_sha256', {})),
                           compatible_code_changes=protocol.get('compatible_code_changes', []))
            return context
    context.update(output=str(output), runtime_sha256={}, compatible_code_changes=[])
    return context


def read_json(path):
    return json.loads(Path(path).read_text())


def source_metadata(root):
    root = Path(root)
    metadata = root / '.benchmark'
    return metadata if (metadata / 'protocol.json').is_file() else root


def training_settings(args, support):
    return dict(seed=args.seed, n_cpu=args.n_cpu, estimations=args.estimations,
                tuning=args.tuning, search_jobs=1, pruning=args.pruning,
                homology_aware=False, packages=support.homology.package_versions(),
                runtime_sha256=runtime_hashes(),
                orchestration_sha256={str(Path(__file__).relative_to(PROJECT_PATH)): file_hash(__file__),
                                      str(Path(support.__file__).relative_to(PROJECT_PATH)): file_hash(support.__file__)})


def check_settings(previous, current):
    for key, value in current.items():
        if key == 'runtime_sha256':
            check_code(previous.get(key, previous.get('code_sha256', {})), value)
        elif key == 'orchestration_sha256' and 'runtime_sha256' not in previous:
            # The pre-merge protocol is preserved in protocol_history on migration.
            continue
        elif key == 'orchestration_sha256':
            from run_ablation import same_configuration
            if not same_configuration(previous.get(key), value):
                raise ValueError('Benchmark runner changed; existing outputs preserved.')
        elif key == 'packages' and key not in previous:
            # First-run protocols recorded packages in each run.json instead.
            continue
        elif previous.get(key, True if key == 'pruning' else None) != value:
            raise ValueError(f'Incompatible benchmark setting: {key}; use a different output directory.')


def check_result(dataset, run, output, settings, support):
    record = support.validate_record(output)
    values = record['settings']
    if isinstance(values.get('dataset'), dict) and values['dataset'] != dataset:
        raise ValueError(f'Result dataset mismatch: {output}')
    check_source_code(record, output, dataset, settings['runtime_sha256'])
    expected = sorted((Path(p).name, digest) for p, digest in dataset['input_hashes'].items())
    actual = sorted((item['name'], item['sha256']) for item in record['inputs'])
    if actual != expected:
        raise ValueError(f'Result input mismatch: {output}')
    homology = Path(output) / 'reports/homology/homology_report.json'
    if values.get('homology_aware') or values.get('cv', 'ordinary') != 'ordinary':
        raise ValueError(f'Result uses different CV: {output}')
    if values.get('condition', 'selected_tuned') != 'selected_tuned':
        raise ValueError(f'Result is not the full two-stage pipeline: {output}')
    if homology.exists() and read_json(homology).get('enabled'):
        raise ValueError(f'Result uses homology-aware CV: {output}')
    for stage, cap in (('stage_1', settings['estimations']), ('stage_2', settings['tuning'])):
        history = read_json(Path(output) / 'reports/optimization' / f'{stage}_study.json')
        wanted = dict(seed=settings['seed'], search_seed=settings['seed'] * 100 + run,
                      requested_trials=cap, folds=5, search_jobs=1, model_jobs=settings['n_cpu'],
                      startup_trials=30 if stage == 'stage_1' else 10)
        if stage == 'stage_1':
            wanted.update(patience=80, min_delta=0.001)
        if (history['status'] != 'completed'
                or history['pruner'] != ('MedianPruner' if settings['pruning'] else 'NopPruner')
                or any(history['settings'].get(k) != v for k, v in wanted.items())):
            raise ValueError(f'Incompatible search history: {output}/{stage}')
    return record


def result_reference(output, source):
    output = Path(output)
    paths = [output / 'run.json', Path(run_path(output, 'trained_model.sav'))]
    paths += list((output / 'results/metrics').glob('*.csv'))
    paths += list((output / 'reports/optimization').glob('*'))
    paths += list((output / 'reports').glob('benchmark_source.json'))
    return dict(output=str(output), source=source,
                sha256={str(p): file_hash(p) for p in paths if p.is_file()})


def check_cache(dataset, source, assets, settings, support):
    check_hashes(assets)
    record = support.validate_record(source['shared'])
    check_source_code(record, source['shared'], dataset, settings['runtime_sha256'])
    values = record['settings'].get('dataset')
    if isinstance(values, dict) and values != dataset:
        raise ValueError(f'Cached dataset changed: {dataset["dataset"]}')
    expected = sorted((Path(p).name, h) for p, h in dataset['input_hashes'].items())
    if sorted((item['name'], item['sha256']) for item in record['inputs']) != expected:
        raise ValueError(f'Cache input mismatch: {source["shared"]}')
    import engineering
    inputs = source['inputs']
    columns = pd.read_csv(inputs['train'], nrows=0).columns.tolist()
    descriptors = engineering.AMINOACID_DESCRIPTORS if dataset['dtype'] == 'protein' else engineering.NUCLEOTIDE_DESCRIPTORS
    engineering.get_descriptor_indices(columns, descriptors, require_all=True)
    for split in ('train', 'test') if source['full_test_cache'] else ('train',):
        names = pd.read_csv(inputs[f'{split}_nameseq'])['nameseq']
        count = sum(len(chunk) for chunk in pd.read_csv(inputs[split], chunksize=10000, usecols=[0]))
        if (len(names) != dataset[f'{split}_count'] or names.duplicated().any()
                or count != len(names) or len(pd.read_csv(inputs[f'{split}_label'])) != len(names)):
            raise ValueError(f'Cached row alignment mismatch: {dataset["dataset"]}/{split}')
        if pd.read_csv(inputs[split], nrows=0).columns.tolist() != columns:
            raise ValueError(f'Cached column mismatch: {dataset["dataset"]}/{split}')


def cache_reference(shared):
    shared = Path(shared)
    manifest = shared / 'feature_inputs.json'
    if manifest.exists():
        inputs = read_json(manifest)
    else:
        inputs = {key: run_path(shared, 'feat_extraction', filename) for key, filename in
                  (('train', 'ftrain.csv'), ('train_label', 'flabeltrain.csv'),
                   ('train_nameseq', 'fnameseqtrain.csv'))}
    paths = [shared / 'run.json', shared / 'timings.csv']
    paths += [Path(inputs[k]) for k in ('train', 'train_label', 'train_nameseq',
                                       'test', 'test_label', 'test_nameseq') if k in inputs]
    paths += [p for p in (shared / 'reports').rglob('*') if p.is_file()]
    if manifest.exists():
        paths.append(manifest)
    times = pd.read_csv(shared / 'timings.csv')
    phases = ['training_features', 'test_features'] if 'test' in inputs else ['training_features']
    minutes = float(times.loc[times.phase.isin(phases), 'elapsed_seconds'].sum()) / 60
    return (dict(shared=str(shared), inputs=inputs, cached_extraction_minutes=minutes,
                 full_test_cache='test' in inputs),
            {str(p): file_hash(p) for p in paths})


def discover_sources(args, datasets, settings, support):
    sources, assets, references = {}, {}, {}
    # Existing manifests take precedence; never select a source by its score.
    roots = [args.output, *getattr(args, 'reuse_from', []), DATASETS_PATH]
    seen = set()
    for root in roots:
        root = root.resolve()
        metadata = source_metadata(root)
        if root in seen or not (metadata / 'protocol.json').exists():
            continue
        seen.add(root)
        protocol = read_json(metadata / 'protocol.json')
        config = protocol.get('configuration', protocol.get('settings', {}))
        inventory = {d['dataset']: d for d in config.get('datasets', [])}
        for dataset in datasets:
            name = dataset['dataset']
            if inventory.get(name) != dataset:
                if root == args.output and name in inventory:
                    raise ValueError(f'Input inventory changed: {name}')
                continue
            if name in config.get('sources', {}) and name not in sources:
                print(f'Validating descriptor cache: {name}', flush=True)
                source, hashes = config['sources'][name], config['assets'][name]
                check_cache(dataset, source, hashes, settings, support)
                sources[name], assets[name] = source, hashes
        candidates = dict(config.get('reused', {}))
        ledger = metadata / 'result_sources.json'
        if ledger.exists():
            for name, reference in read_json(ledger).items():
                candidates.setdefault(f'{name}/run_1', reference)
        index = metadata / 'results.csv'
        if not index.exists():
            index = metadata / 'results_index.csv'
        if index.exists():
            for row in pd.read_csv(index).to_dict('records'):
                candidates.setdefault(f'{row["dataset"]}/run_{int(row["run"])}', dict(
                    output=row['output'], source=row.get('source', 'first_benchmark')))
        for dataset in datasets:
            name = dataset['dataset']
            if inventory.get(name) != dataset:
                continue
            for run in range(1, args.runs + 1):
                key = f'{name}/run_{run}'
                if key in references or key not in candidates:
                    continue
                ref = candidates[key]
                try:
                    check_result(dataset, run, ref['output'], settings, support)
                except ValueError:
                    if root == args.output:
                        raise
                    continue
                check_hashes(ref.get('sha256', {}))
                references[key] = result_reference(ref['output'], ref['source'])
                cache_root = Path(ref['output'])
                if name not in sources and ((cache_root / 'feature_inputs.json').exists()
                                             or Path(run_path(cache_root, 'feat_extraction', 'ftrain.csv')).is_file()):
                    source, hashes = cache_reference(ref['output'])
                    # Selected-only training matrices are not raw feature caches.
                    if Path(source['inputs']['train']).exists():
                        check_cache(dataset, source, hashes, settings, support)
                        sources[name], assets[name] = source, hashes
    ablation_root = args.reuse_ablation.resolve()
    for dataset in datasets:
        name = dataset['dataset']
        shared = ablation_root / name / 'shared_features'
        if name not in sources and (shared / 'run.json').exists():
            source, hashes = cache_reference(shared)
            check_cache(dataset, source, hashes, settings, support)
            sources[name], assets[name] = source, hashes
        for run in range(1, args.runs + 1):
            key = f'{name}/run_{run}'
            path = ablation_root / name / 'runs' / f'run_{run}' / 'selected_tuned'
            if key not in references and (path / 'run.json').exists():
                try:
                    check_result(dataset, run, path, settings, support)
                except ValueError:
                    continue
                references[key] = result_reference(path, 'ablation')
    for dataset in datasets:
        for run in range(4, args.runs + 1):
            key = f'{dataset["dataset"]}/run_{run}'
            path = RESULTS_PATH / 'homology_comparison' / dataset['dataset'] / 'runs' / f'run_{run}' / 'ordinary/selected_tuned'
            if key not in references and (path / 'run.json').exists():
                try:
                    check_result(dataset, run, path, settings, support)
                except ValueError:
                    continue
                references[key] = result_reference(path, 'homology_ordinary')
    if not args.pruning:
        for dataset in datasets:
            for run in range(1, args.runs + 1):
                key = f'{dataset["dataset"]}/run_{run}'
                path = RESULTS_PATH / 'pruning_comparison' / dataset['dataset'] / 'runs' / f'run_{run}' / 'pruning_off'
                if key not in references and (path / 'run.json').exists():
                    try:
                        check_result(dataset, run, path, settings, support)
                    except ValueError:
                        continue
                    references[key] = result_reference(path, 'pruning_comparison')
    return sources, assets, references


def copy_artifacts(source, destination):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Existing output preserved: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{destination.name}-copy-', dir=destination.parent))
    # Failed copies remain under a temporary name for inspection, never as completed runs.
    shutil.copytree(source, temporary, dirs_exist_ok=True)
    for original in source.rglob('*'):
        if original.is_file() and file_hash(original) != file_hash(temporary / original.relative_to(source)):
            raise ValueError(f'Copied artifact differs: {original}')
    atomic_json(temporary / 'reports/benchmark_source.json', source_context(source))
    manifest = temporary / 'feature_inputs.json'
    if manifest.exists():
        inputs = read_json(manifest)
        for key, value in inputs.items():
            if isinstance(value, str) and Path(value).is_absolute():
                relative = None
                for base in (source, source.resolve()):
                    try:
                        relative = Path(value).relative_to(base)
                        break
                    except ValueError:
                        pass
                if relative is not None:
                    inputs[key] = str(destination / relative)
        atomic_json(manifest, inputs)
    temporary.rename(destination)


def prepare_layout(root, datasets, sources, assets, references, settings, support):
    for dataset in datasets:
        name = dataset['dataset']
        for split in ('train', 'test'):
            for filename in dataset[f'{split}_files']:
                source = Path(filename)
                target = root / name / split / source.name
                if target.resolve() == source.resolve():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if file_hash(target) != file_hash(source):
                        raise ValueError(f'Different input already exists: {target}')
                else:
                    shutil.copy2(source, target)
        if name in sources:
            target = root / name / 'shared'
            copy_artifacts(sources[name]['shared'], target)
            source, hashes = cache_reference(target)
            check_cache(dataset, source, hashes, settings, support)
            sources[name], assets[name] = source, hashes
        for run in range(1, 6):
            key = f'{name}/run_{run}'
            if key in references:
                ref = references[key]
                target = root / name / 'runs' / f'run_{run}'
                copy_artifacts(ref['output'], target)
                check_result(dataset, run, target, settings, support)
                references[key] = result_reference(target, ref['source'])
    bibliography = DATASETS_PATH / 'references.bib'
    target = root / 'references.bib'
    if bibliography.resolve() != target.resolve():
        if target.exists():
            if file_hash(target) != file_hash(bibliography):
                raise ValueError(f'Different bibliography already exists: {target}')
        else:
            shutil.copy2(bibliography, target)


def inputs_only(root):
    for path in root.iterdir():
        if path.name == 'references.bib' and path.is_file():
            continue
        if not path.name.startswith('dataset') or not path.is_dir() or path.is_symlink():
            return False
        for split in path.iterdir():
            if split.name not in ('train', 'test') or not split.is_dir() or split.is_symlink():
                return False
            if any(not p.is_file() or p.is_symlink() for p in split.iterdir()):
                return False
    return True


def archive_attempt(root, output):
    import psutil
    metadata = root / '.benchmark'
    name = '__'.join(output.relative_to(root).parts)
    config_path = metadata / 'configs' / f'{name}.json'
    pid_path = output.parent / f'{output.name}.pid'
    if pid_path.exists():
        try:
            process = psutil.Process(int(pid_path.read_text()))
            if str(config_path) in process.cmdline():
                raise RuntimeError(f'Worker still active: {output}')
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
    if not output.exists() and not config_path.exists():
        return
    archive = metadata / 'interrupted_attempts' / name
    number = 1
    while (archive / f'attempt_{number}').exists():
        number += 1
    target = archive / f'attempt_{number}'
    target.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        shutil.move(output, target)
    else:
        target.mkdir()
    for path, filename in ((config_path, 'worker_config.json'),
                           (metadata / 'logs' / f'{name}.log', 'worker.log'), (pid_path, 'worker.pid')):
        if path.exists():
            shutil.move(path, target / filename)


def extract_cache(dataset, output, settings, support):
    import engineering
    from bioautoml.sequence_names import register_sources
    from bioautoml.homology import prepare_homology
    config = dict(settings, dataset=dataset, code_sha256=support.code_hashes())
    execution = Execution(output, 'extraction', settings=config)
    try:
        execution.add_inputs(dataset['train_files'] + dataset['test_files'])
        for split in ('train', 'test'):
            register_sources(output, dataset[f'{split}_files'], dataset[f'{split}_labels'], split)
        protein = dataset['dtype'] == 'protein'
        prepare_homology(dataset['train_files'], dataset['train_labels'], dataset['test_files'],
                         dataset['test_labels'], 'Protein' if protein else 'DNA/RNA', dataset['task'],
                         settings['seed'], run_path(output, 'homology'), enabled=False, n_cpu=settings['n_cpu'])
        extract = engineering.feature_extraction_aminoacid if protein else engineering.feature_extraction_nucleotide
        names, _, train, labels, _, _ = extract(dataset['train_files'], dataset['train_labels'],
                                               None, None, str(output), settings['n_cpu'])
        atomic_json(output / 'feature_inputs.json', dict(train=train, train_label=labels, train_nameseq=names))
        sys.path.insert(0, str(PROJECT_PATH / 'App/utils'))
        from stats import summary_stats
        with execution.phase('benchmark_summary'):
            summary_stats(run_path(output, 'feat_extraction', 'train'),
                          'Protein' if protein else 'DNA/RNA', str(output), False)
        check_hashes(dataset['input_hashes'])
        if config['code_sha256'] != support.code_hashes():
            raise ValueError('Code changed during descriptor extraction.')
    except BaseException:
        execution.finish('failed')
        raise
    else:
        execution.finish()


def write_folds(inputs, dataset, seed, output):
    from sklearn.model_selection import KFold, StratifiedKFold
    ids = pd.read_csv(inputs['train_nameseq'])['nameseq'].tolist()
    y = pd.read_csv(inputs['train_label']).iloc[:, 0].to_numpy()
    folds = dict(sequence_ids=ids, seed=seed, reporting_scope='post-selection optimization CV')
    for count in (5, 10):
        splitter = StratifiedKFold if dataset['task'] == 0 else KFold
        folds[str(count)] = [dict(train=train.tolist(), validation=test.tolist())
                             for train, test in splitter(count, shuffle=True, random_state=seed).split(ids, y)]
    atomic_json(Path(output) / 'reports/cv_folds.json', folds)


@contextmanager
def benchmark_state(args, settings, datasets, previous):
    root = args.output
    metadata = root / '.benchmark'
    metadata.mkdir(parents=True, exist_ok=True)
    with (metadata / '.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = metadata / 'protocol.json'
        latest = read_json(path) if path.exists() else {}
        if latest != previous:
            raise RuntimeError('Another runner updated this benchmark during validation; retry when it stops.')
        if previous:
            check_settings(previous.get('configuration', previous.get('settings', {})), settings)
            archive = metadata / 'protocol_history'
            archive.mkdir(exist_ok=True)
            snapshot = archive / f'{file_hash(path)}.json'
            if not snapshot.exists():
                shutil.copy2(path, snapshot)
        for directory in ('configs', 'logs'):
            (metadata / directory).mkdir(exist_ok=True)
        old = previous.get('configuration', previous.get('settings', {}))
        inventory = {d['dataset']: d for d in old.get('datasets', [])}
        inventory.update({d['dataset']: d for d in datasets})
        state = dict(schema_version=2, configuration=dict(settings, datasets=list(inventory.values())),
                     status='running', started_at=previous.get('started_at', utc_now()),
                     completed_results=previous.get('completed_results', 0))
        if 'compatible_code_changes' in previous:
            state['compatible_code_changes'] = previous['compatible_code_changes']
        started = time.perf_counter()
        try:
            yield state, previous
        except BaseException:
            state['status'] = 'failed'
            raise
        else:
            target = len(state['configuration']['datasets']) * len(state['configuration']['search_seeds'])
            state['status'] = 'completed' if state['completed_results'] >= target else 'partial'
        finally:
            state.update(updated_at=utc_now(), invocation_minutes=(time.perf_counter() - started) / 60)
            atomic_json(path, state)


def main():
    args = parse_arguments()
    os.chdir(PROJECT_PATH)
    sys.path.insert(0, str(EXPERIMENTS_PATH))
    import _experiment_support as support
    datasets, inventory = input_inventory()
    if args.datasets:
        if len(set(args.datasets)) != len(args.datasets) or not set(args.datasets) <= {d['dataset_id'] for d in datasets}:
            raise ValueError('Use valid unique dataset IDs.')
        datasets = [d for d in datasets if d['dataset_id'] in args.datasets]
    settings = training_settings(args, support)
    metadata = args.output / '.benchmark'
    lock_path = metadata / '.runner.lock'
    if lock_path.exists():
        with lock_path.open('r') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = read_json(metadata / 'protocol.json') if (metadata / 'protocol.json').exists() else {}
    if previous:
        check_settings(previous.get('configuration', previous.get('settings', {})), settings)
    elif args.output.exists() and not inputs_only(args.output):
        raise ValueError('Nonempty output directory has no protocol; inspect before proceeding.')
    print('Checking input inventories, cached descriptors and completed searches...', flush=True)
    sources, assets, references = discover_sources(args, datasets, settings, support)
    for dataset in datasets:
        for run in range(1, args.runs + 1):
            path = args.output / dataset['dataset'] / 'runs' / f'run_{run}'
            if (path / 'run.json').exists() and read_json(path / 'run.json')['status'] == 'completed':
                check_result(dataset, run, path, settings, support)
                references.setdefault(f'{dataset["dataset"]}/run_{run}', result_reference(path, 'completed'))
    total = len(datasets) * args.runs
    print(f'{total} results: {len(references)} reusable, {total - len(references)} remaining; '
          f'{len(sources)}/{len(datasets)} descriptor caches available.', flush=True)
    if args.dry_run:
        print('Dry run: no files changed and no workers launched.', flush=True)
        return
    with benchmark_state(args, settings, datasets, previous) as (state, previous):
        old = previous.get('configuration', previous.get('settings', {}))
        sources = dict(old.get('sources', {}), **sources)
        assets = dict(old.get('assets', {}), **assets)
        references = dict(old.get('reused', {}), **references)
        seeds = sorted(set(old.get('search_seeds', [])) | {args.seed * 100 + run for run in range(1, args.runs + 1)})
        state['configuration'].update(sources=sources, assets=assets, reused=references, search_seeds=seeds)
        prepare_layout(args.output, datasets, sources, assets, references, settings, support)
        support.ablation.save_csv(inventory[inventory.dataset.isin([d['dataset'] for d in state['configuration']['datasets']])],
                                 metadata / 'input_inventory.csv')
        atomic_json(metadata / 'protocol.json', state)
        result_path = metadata / 'results.csv'
        rows = pd.read_csv(result_path).to_dict('records') if result_path.exists() else []
        old_index = metadata / 'results_index.csv'
        if not result_path.exists() and old_index.exists():
            by_name = {d['dataset']: d for d in state['configuration']['datasets']}
            for item in pd.read_csv(old_index).to_dict('records'):
                dataset = by_name[item['dataset']]
                check_result(dataset, int(item['run']), item['output'], settings, support)
                rows.append(support.collect_result(dataset, int(item['run']), item['output'],
                                                   source='first_benchmark', pruning=args.pruning))
        for dataset in datasets:
            name = dataset['dataset']
            for run in range(1, args.runs + 1):
                key = f'{name}/run_{run}'
                reference = references.get(key)
                if reference:
                    check_hashes(reference['sha256'])
                    row = support.collect_result(dataset, run, reference['output'],
                                                 source=reference['source'], pruning=args.pruning)
                else:
                    if name not in sources:
                        cache = args.output / name / 'shared'
                        if (cache / 'run.json').exists() and read_json(cache / 'run.json')['status'] == 'completed':
                            source, hashes = cache_reference(cache)
                            check_cache(dataset, source, hashes, settings, support)
                        else:
                            archive_attempt(args.output, cache)
                            config = dict(settings, dataset=dataset, code_sha256=support.code_hashes(),
                                          assets={}, operation='benchmark_extraction', output=str(cache))
                            support.run_worker(args.output, config, resume=False, metadata_root=metadata)
                            source, hashes = cache_reference(cache)
                            check_cache(dataset, source, hashes, settings, support)
                        sources[name], assets[name] = source, hashes
                        atomic_json(metadata / 'protocol.json', state)
                    output = args.output / name / 'runs' / f'run_{run}'
                    archive_attempt(args.output, output)
                    config = dict(settings, smoke_test=False, code_sha256=support.code_hashes(),
                                  dataset=dataset, source=sources[name], assets=assets[name], run=run,
                                  search_seed=args.seed * 100 + run, output=str(output), benchmark_runner=True)
                    support.run_worker(args.output, config, resume=False, metadata_root=metadata)
                    row = support.collect_result(dataset, run, output, pruning=args.pruning)
                    references[key] = result_reference(output, 'new')
                row['search_seed'] = args.seed * 100 + run
                rows = [r for r in rows if (r['dataset'], r['run']) != (name, run)] + [row]
                rows.sort(key=lambda r: (int(r['dataset'].split('_')[0][7:]), r['run']))
                support.save_results(rows, metadata)
                # Retain the first-benchmark index consumed by supporting experiments.
                index = pd.DataFrame(rows)
                index['seed'], index['status'] = args.seed, 'completed'
                index['reused'] = index.source != 'new'
                index['primary_score'] = [r.get(f'test_{r["primary_metric"]}',
                                               r[f'optimization_cv_{r["primary_metric"]}']) for r in rows]
                support.ablation.save_csv(index, metadata / 'results_index.csv')
                state['completed_results'] = len(rows)
                atomic_json(metadata / 'protocol.json', state)


if __name__ == '__main__':
    main()
