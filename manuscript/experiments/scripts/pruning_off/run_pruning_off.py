"""Single-seed, 64-dataset pruning-off benchmark, isolated from the active queue.

Keep this runner in its own subdirectory: the running queue freezes the Python
files directly under scripts/. This runner and its tests are independently
fingerprinted as worker assets, without changing that existing frozen code.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT_PATH.parent.parent))
import _experiment_support as support
from _experiment_support import RESULTS_PATH, PROJECT_PATH, ablation, atomic_json, file_hash

BASE_KEYS = ('seed', 'n_cpu', 'estimations', 'tuning', 'search_jobs', 'smoke_test', 'packages', 'code_sha256')


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_cpu', type=int, default=32)
    parser.add_argument('--estimations', type=int, default=200)
    parser.add_argument('--tuning', type=int, default=150)
    parser.add_argument('--datasets', type=int, nargs='+')
    parser.add_argument('--output', type=Path, default=RESULTS_PATH / 'benchmark_pruning_off')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke_test', action='store_true')
    args = parser.parse_args()
    if min(args.n_cpu, args.estimations, args.tuning) < 1:
        parser.error('Use positive CPU/trial counts.')
    if not args.smoke_test and (args.n_cpu != 32 or args.estimations != 200
                               or args.tuning != 150 or args.datasets is not None):
        parser.error('Changes to the full protocol require --smoke_test and separate outputs.')
    args.output = support.validate_output(args.output, 'benchmark_pruning_off', args.smoke_test)
    if any(args.output.is_relative_to(RESULTS_PATH / name) for name in
           ('remaining_experiments', 'sequence_overlap', 'pruning_comparison', 'benchmark_repeats')):
        parser.error('Existing queue outputs are protected.')
    return args


def runner_assets():
    return {str(p): file_hash(p) for p in sorted(SCRIPT_PATH.parent.glob('*.py'))}


def check_assets(assets):
    for path, expected in assets.items():
        if file_hash(path) != expected:
            raise ValueError(f'Frozen asset changed: {path}')


def reference_assets(output):
    output = Path(output)
    paths = [output / 'run.json', Path(support.run_path(output, 'trained_model.sav'))]
    paths += list((output / 'results/metrics').glob('*.csv'))
    return {str(path): file_hash(path) for path in paths}


def reuse_candidates(datasets, sources, assets, base):
    """Freeze eligible paths/settings, including the not-yet-completed third pair.

    Eligibility never depends on test scores. Use a candidate if complete when
    reached; otherwise run a fresh pipeline without waiting on the other queue.
    """
    protocol = RESULTS_PATH / 'pruning_comparison/protocol.json'
    frozen = ablation.read_json(protocol)['configuration']
    if any(frozen[k] != base[k] for k in BASE_KEYS):
        raise ValueError('Pruning comparison does not match this benchmark protocol.')
    candidates = {}
    available = {d['dataset']: d for d in frozen['datasets']}
    for dataset in datasets:
        name = dataset['dataset']
        if name not in available:
            continue
        if (dataset != available[name] or sources[name] != frozen['sources'][name]
                or assets[name] != frozen['assets'][name]):
            raise ValueError(f'Pruning comparison source mismatch: {name}')
        output = protocol.parent / name / 'runs/run_1/pruning_off'
        candidates[name] = dict(base, dataset=dataset, source=sources[name], assets=assets[name],
            run=1, search_seed=6301, pruning=False, output=str(output))
    return candidates


def check_off_history(output, n_cpu, estimations, tuning):
    for stage, cap in (('stage_1', estimations), ('stage_2', tuning)):
        history = ablation.read_json(Path(output) / 'reports/optimization' / f'{stage}_study.json')
        values = history['settings']
        if (history['status'] != 'completed' or history['pruner'] != 'NopPruner'
                or values['seed'] != 63 or values['search_seed'] != 6301
                or values['requested_trials'] != cap or values['folds'] != 5
                or values['search_jobs'] != 1 or values['model_jobs'] != n_cpu):
            raise ValueError(f'Incompatible pruning-off history: {output}/{stage}')


def completed_candidate(config):
    if config is None:
        return False
    path = Path(config['output']) / 'run.json'
    if not path.exists():
        return False
    record = ablation.read_json(path)
    if record['settings'] != config:
        raise ValueError(f'Candidate settings mismatch: {path}')
    if record['status'] != 'completed':
        return False
    support.verify_config(config)
    support.validate_record(config['output'])
    check_off_history(config['output'], config['n_cpu'], config['estimations'], config['tuning'])
    return True


def paired_results(rows, baselines):
    baseline = {r['dataset']: r for r in baselines}
    pairs = []
    for row in rows:
        other = baseline[row['dataset']]
        metric = row['primary_metric']
        if row['search_seed'] != other['search_seed'] or metric != other['primary_metric']:
            raise ValueError('Unmatched baseline seed/task.')
        result = dict(dataset=row['dataset'], task=row['task'], primary_metric=metric,
            search_seed=row['search_seed'], test_available=row['test_available'],
            pruning_off_output=row['output'], pruning_on_output=other['output'])
        for scope in ('test', 'optimization_cv'):
            key = f'{scope}_{metric}'
            off, on = row.get(key), other.get(key)
            result.update({f'{scope}_off': off, f'{scope}_on': on,
                           f'{scope}_off_minus_on': off - on if off is not None and on is not None else None})
        result['runtime_caveat'] = 'Concurrent workload; historical baseline timings are not a controlled speed comparison.'
        pairs.append(result)
    return pd.DataFrame(pairs)


def main():
    args = parse_arguments()
    os.chdir(PROJECT_PATH)
    datasets, inventory = support.benchmarks.input_inventory()
    if args.datasets is not None:
        if len(set(args.datasets)) != len(args.datasets) or not set(args.datasets) <= {d['dataset_id'] for d in datasets}:
            raise ValueError('Use unique valid dataset IDs.')
        datasets = [d for d in datasets if d['dataset_id'] in args.datasets]
    sources, assets, _ = support.prepare_sources(datasets, include_reuse=False)
    base = support.settings(args)
    scripts = runner_assets()
    candidates = {} if args.smoke_test else reuse_candidates(datasets, sources, assets, base)
    index = pd.read_csv(RESULTS_PATH / 'benchmarks/results_index.csv').set_index('dataset')
    baselines, baseline_assets = [], {}
    for dataset in datasets:
        output = index.loc[dataset['dataset'], 'output']
        baselines.append(support.collect_result(dataset, 1, output, source='first_benchmark', pruning=True))
        baseline_assets[dataset['dataset']] = reference_assets(output)
    configuration = dict(base, datasets=datasets, sources=sources, assets=assets, runner_assets=scripts,
        candidates=candidates, baseline_assets=baseline_assets, search_seeds=[6301],
        homology_aware=False, pruning=False, pruning_stages=['stage_1', 'stage_2'],
        note='Ordinary CV, tuning on; unchanged data, budgets, timeouts and patience. '
             'Only fold-score pruning disabled; invalid configurations remain rejected. '
             'One seed per dataset; do not infer seed robustness. Tests used to choose a repository '
             'configuration become development evidence. Artifacts are not automatically published. '
             'Runs share this machine with the other queue: record timings but do not treat them '
             'as isolated comparisons. Complete all-descriptor training caches are reused; '
             'new selected test features are extracted when no full test cache exists.')
    ready = [name for name, config in candidates.items() if completed_candidate(config)]
    print(f'{len(datasets)} pruning-off first-seed results; {len(ready)} already reusable; '
          f'{len(candidates) - len(ready)} additional candidate(s) may complete in the other queue.', flush=True)
    if not args.smoke_test and len(datasets) != 64:
        raise ValueError('Full protocol requires all 64 datasets.')
    if args.dry_run:
        return
    rows = []
    with support.experiment(args.output, configuration, args.resume) as state:
        ablation.save_csv(inventory[inventory.dataset.isin([d['dataset'] for d in datasets])], args.output / 'input_inventory.csv')
        ablation.save_csv(pd.DataFrame(baselines), args.output / 'baseline_results.csv')
        ledger_path = args.output / 'result_sources.json'
        ledger = ablation.read_json(ledger_path) if ledger_path.exists() else {}
        # Consume ready references first. This saves progress immediately; all
        # remaining new pipelines still follow the dataset inventory order.
        ordered = sorted(datasets, key=lambda d: (d['dataset'] not in ready, d['dataset_id']))
        for dataset in ordered:
            name = dataset['dataset']
            check_assets(scripts)
            check_assets(baseline_assets[name])
            output = args.output / name / 'runs/run_1'
            candidate = candidates.get(name)
            reference = ledger.get(name)
            if reference and reference['source'] == 'pruning_comparison':
                check_assets(reference['sha256'])
                if not completed_candidate(candidate):
                    raise ValueError(f'Reused result no longer complete: {name}')
                output = Path(reference['output'])
            elif not output.exists() and completed_candidate(candidate):
                output = Path(candidate['output'])
                reference = dict(output=str(output), source='pruning_comparison', sha256=reference_assets(output))
                ledger[name] = reference
                atomic_json(ledger_path, ledger)
            else:
                config = dict(base, dataset=dataset, source=sources[name], assets=dict(assets[name], **scripts),
                              run=1, search_seed=6301, pruning=False, output=str(output))
                support.run_worker(args.output, config, args.resume)
                check_off_history(output, args.n_cpu, args.estimations, args.tuning)
                reference = dict(output=str(output), source='new', sha256=reference_assets(output))
                ledger[name] = reference
                atomic_json(ledger_path, ledger)
            check_assets(scripts)
            row = support.collect_result(dataset, 1, output, source=reference['source'], pruning=False)
            rows.append(row)
            print(f'Completed {len(rows)}/{len(datasets)}: {name} ({reference["source"]})', flush=True)
            support.save_results(rows, args.output)
            ablation.save_csv(paired_results(rows, baselines), args.output / 'paired_results.csv')
            ablation.save_csv(pd.DataFrame([dict(dataset=r['dataset'], source=r['source'],
                model=support.run_path(r['output'], 'trained_model.sav')) for r in rows]), args.output / 'models_index.csv')
            state.update(completed_results=len(rows), reused_results=sum(r['source'] != 'new' for r in rows))
            atomic_json(args.output / 'protocol.json', state)


if __name__ == '__main__':
    main()
