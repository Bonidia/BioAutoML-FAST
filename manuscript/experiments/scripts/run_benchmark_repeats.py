"""Complete five-seed benchmark using verified first runs and subset baselines."""
import argparse
import os

import _experiment_support as support
from _experiment_support import RESULTS_PATH, PROJECT_PATH, ablation, atomic_json, file_hash


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_cpu', type=int, default=32)
    parser.add_argument('--runs', type=int, choices=range(1, 6), default=5)
    parser.add_argument('--datasets', type=int, nargs='+')
    parser.add_argument('--estimations', type=int, default=200)
    parser.add_argument('--tuning', type=int, default=150)
    parser.add_argument('--output', type=support.Path, default=RESULTS_PATH / 'benchmark_repeats')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke_test', action='store_true')
    args = parser.parse_args()
    if min(args.n_cpu, args.estimations, args.tuning) < 1:
        parser.error('Counts must be positive.')
    if not args.smoke_test and (args.n_cpu != 32 or args.runs != 5 or args.datasets is not None
                               or args.estimations != 200 or args.tuning != 150):
        parser.error('Protocol changes require --smoke_test.')
    args.output = support.validate_output(args.output, 'benchmark_repeats', args.smoke_test)
    return args


def main():
    args = parse_arguments()
    os.chdir(PROJECT_PATH)
    datasets, inventory = support.benchmarks.input_inventory()
    if args.datasets is not None:
        if len(set(args.datasets)) != len(args.datasets) or not set(args.datasets) <= {d['dataset_id'] for d in datasets}:
            raise ValueError('Use valid unique dataset IDs.')
        datasets = [d for d in datasets if d['dataset_id'] in args.datasets]
    sources, assets, reused = support.prepare_sources(datasets, include_reuse=not args.smoke_test)
    base_settings = support.settings(args)
    configuration = dict(base_settings, datasets=datasets, sources=sources, assets=assets, reused=reused,
        search_seeds=list(range(6301, 6301 + args.runs)), homology_aware=False,
        note='Ordinary CV; tuning on. Test sets unchanged. Post-selection optimization CV is not unbiased. '
             'First-seed and matched subset results are referenced, never overwritten. '
             'Training extraction reused; test extraction reused only when the full descriptor matrix is available.')
    total = len(datasets) * args.runs
    print(f'{total} results: {len(reused)} reused, {total - len(reused)} new pipelines.', flush=True)
    if not args.smoke_test and (total != 320 or len(reused) != 104):
        raise ValueError('Unexpected frozen inventory/reuse count; inspect before starting.')
    if args.dry_run:
        return
    rows = []
    with support.experiment(args.output, configuration, args.resume) as state:
        ablation.save_csv(inventory[inventory.dataset.isin([d['dataset'] for d in datasets])], args.output / 'input_inventory.csv')
        for dataset in datasets:
            name = dataset['dataset']
            for run in range(1, args.runs + 1):
                reference = reused.get(f'{name}/run_{run}')
                if reference:
                    for path, expected in reference['sha256'].items():
                        if file_hash(path) != expected:
                            raise ValueError(f'Reused result changed: {path}')
                    row = support.collect_result(dataset, run, reference['output'], source=reference['source'])
                    print(f'Reusing {name}/run_{run}: {reference["source"]}', flush=True)
                else:
                    output = args.output / name / 'runs' / f'run_{run}'
                    config = dict(base_settings, dataset=dataset, source=sources[name], assets=assets[name],
                                  run=run, search_seed=6300 + run, pruning=True, output=str(output))
                    support.run_worker(args.output, config, args.resume)
                    row = support.collect_result(dataset, run, output)
                rows.append(row)
                support.save_results(rows, args.output)
                state['completed_results'] = len(rows)
                atomic_json(args.output / 'protocol.json', state)


if __name__ == '__main__':
    main()
