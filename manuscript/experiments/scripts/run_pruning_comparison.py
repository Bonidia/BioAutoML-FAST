"""Fresh paired fold-pruning on/off comparison; extraction is shared equally."""
import argparse
import os

import _experiment_support as support
from _experiment_support import RESULTS_PATH, PROJECT_PATH, ablation, atomic_json


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_cpu', type=int, default=32)
    parser.add_argument('--datasets', type=int, nargs='+', default=[2, 55, 63], choices=[2, 55, 63])
    parser.add_argument('--runs', type=int, default=3, choices=[1, 2, 3])
    parser.add_argument('--estimations', type=int, default=200)
    parser.add_argument('--tuning', type=int, default=150)
    parser.add_argument('--output', type=support.Path, default=RESULTS_PATH / 'pruning_comparison')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke_test', action='store_true')
    args = parser.parse_args()
    if min(args.n_cpu, args.estimations, args.tuning) < 1 or len(set(args.datasets)) != len(args.datasets):
        parser.error('Use positive counts and unique datasets.')
    if not args.smoke_test and (args.n_cpu != 32 or args.runs != 3 or args.datasets != [2, 55, 63]
                               or args.estimations != 200 or args.tuning != 150):
        parser.error('Protocol changes require --smoke_test.')
    args.output = support.validate_output(args.output, 'pruning_comparison', args.smoke_test)
    return args


def main():
    args = parse_arguments()
    os.chdir(PROJECT_PATH)
    datasets, inventory = support.benchmarks.input_inventory()
    datasets = [d for d in datasets if d['dataset_id'] in args.datasets]
    sources, assets, _ = support.prepare_sources(datasets, include_reuse=False)
    base_settings = support.settings(args)
    configuration = dict(base_settings, datasets=datasets, sources=sources, assets=assets,
        search_seeds=list(range(6301, 6301 + args.runs)), pruning_stages=['stage_1', 'stage_2'],
        pruners={'on': 'MedianPruner(5, 0, 1, n_min_trials=1)', 'off': 'NopPruner'},
        note='Only performance-based fold pruning changes. Invalid configurations remain rejected. '
             'Equal caps/timeouts/patience do not imply equal completed trials or equal TPE proposals. '
             'Both arms freshly trained sequentially; original cached extraction time is reported separately.')
    print(f'{len(datasets) * args.runs * 2} fresh pipelines; 32 CPU full protocol; no model-result reuse.', flush=True)
    if args.dry_run:
        return
    rows = []
    with support.experiment(args.output, configuration, args.resume) as state:
        ablation.save_csv(inventory[inventory.dataset.isin([d['dataset'] for d in datasets])], args.output / 'input_inventory.csv')
        for dataset in datasets:
            name = dataset['dataset']
            for run in range(1, args.runs + 1):
                for pruning in ((True, False) if run % 2 else (False, True)):
                    output = args.output / name / 'runs' / f'run_{run}' / ('pruning_on' if pruning else 'pruning_off')
                    config = dict(base_settings, dataset=dataset, source=sources[name], assets=assets[name],
                                  run=run, search_seed=6300 + run, pruning=pruning, output=str(output))
                    support.run_worker(args.output, config, args.resume)
                    rows.append(support.collect_result(dataset, run, output, pruning=pruning))
                    support.save_results(rows, args.output, pruning=True)
                    state['completed_results'] = len(rows)
                    atomic_json(args.output / 'protocol.json', state)


if __name__ == '__main__':
    main()
