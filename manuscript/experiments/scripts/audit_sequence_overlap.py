"""Exact sequence overlap audit; never modifies sequences or train/test partitions."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import os

from Bio import SeqIO
import pandas as pd

import _experiment_support as support
from _experiment_support import RESULTS_PATH, PROJECT_PATH, ablation, atomic_json, file_hash


def records(dataset, split):
    for filename, label in zip(dataset[f'{split}_files'], dataset[f'{split}_labels']):
        with open(filename) as handle:
            for number, record in enumerate(SeqIO.parse(handle, 'fasta'), 1):
                sequence = str(record.seq).upper()
                if dataset['dtype'] == 'dnarna':
                    sequence = sequence.replace('U', 'T')
                target = str(float(record.id.split('|')[-1])) if dataset['task'] else label
                yield sequence, target, dict(source_file=filename, record_number=number, original_id=record.id)


def audit_dataset(dataset, output):
    train, test_counts = defaultdict(Counter), Counter()
    for sequence, target, _ in records(dataset, 'train'):
        train[sequence][target] += 1
    matches, conflicting_records, conflicting_pairs, matched_train_records = 0, 0, 0, 0
    overlapping = set()
    fields = ['source_file', 'record_number', 'original_id', 'target', 'sequence_sha256',
              'matching_train_records', 'conflicting_train_records', 'training_targets']
    with (output / 'overlapping_test_records.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sequence, target, info in records(dataset, 'test'):
            test_counts[sequence] += 1
            if sequence not in train:
                continue
            counts = train[sequence]
            number = sum(counts.values())
            conflicts = number - counts[target]
            matches += 1
            conflicting_records += bool(conflicts)
            conflicting_pairs += conflicts
            overlapping.add(sequence)
            writer.writerow(dict(info, target=target, sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
                matching_train_records=number, conflicting_train_records=conflicts,
                training_targets=support.json.dumps(dict(counts), sort_keys=True)))
    matched_train_records = sum(sum(train[s].values()) for s in overlapping)
    return dict(dataset=dataset['dataset'], task=dataset['task'], dtype=dataset['dtype'],
        train_records=dataset['train_count'], test_records=dataset['test_count'],
        status='assessed' if dataset['test_count'] else 'no test set',
        unique_train_sequences=len(train), unique_test_sequences=len(test_counts),
        duplicate_train_records=dataset['train_count'] - len(train),
        duplicate_test_records=dataset['test_count'] - len(test_counts),
        overlapping_test_records=matches if dataset['test_count'] else None,
        overlapping_test_percent=100 * matches / dataset['test_count'] if dataset['test_count'] else None,
        unique_overlapping_sequences=len(overlapping) if dataset['test_count'] else None,
        matched_train_records=matched_train_records if dataset['test_count'] else None,
        conflicting_test_records=conflicting_records if dataset['test_count'] else None,
        conflicting_pairs=conflicting_pairs if dataset['test_count'] else None,
        partitions_modified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=support.Path, default=RESULTS_PATH / 'sequence_overlap')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.chdir(PROJECT_PATH)
    args.output = support.validate_output(args.output, 'sequence_overlap')
    datasets, inventory = support.benchmarks.input_inventory()
    configuration = dict(datasets=datasets, code_sha256=support.code_hashes(),
        definition='Exact full-length uppercase sequence equality; nucleotide U -> T; forward orientation only. '
                   'No reverse-complement equivalence or similarity search. Targets compared as class labels or numeric regression values.',
        output_scope='One row per overlapping test record; match counts avoid a potentially enormous pairwise expansion.')
    print(f'{len(datasets)} datasets; {sum(bool(d["test_count"]) for d in datasets)} with test sets.', flush=True)
    if args.dry_run:
        return
    rows = []
    with support.experiment(args.output, configuration, args.resume) as state:
        ablation.save_csv(inventory, args.output / 'input_inventory.csv')
        for dataset in datasets:
            for path, expected in dataset['input_hashes'].items():
                if file_hash(path) != expected:
                    raise ValueError(f'Input changed: {path}')
            directory = args.output / dataset['dataset']
            directory.mkdir(exist_ok=True)
            summary = directory / 'summary.json'
            if summary.exists() and args.resume and (directory / 'overlapping_test_records.csv').exists():
                row = ablation.read_json(summary)
            else:
                row = audit_dataset(dataset, directory)
                for path, expected in dataset['input_hashes'].items():
                    if file_hash(path) != expected:
                        raise ValueError(f'Input changed during audit: {path}')
                atomic_json(summary, row)
            rows.append(row)
            ablation.save_csv(pd.DataFrame(rows), args.output / 'summary.csv')
            state['completed_results'] = len(rows)
            atomic_json(args.output / 'protocol.json', state)
            print(f'{dataset["dataset"]}: {row["overlapping_test_records"]} overlapping test records', flush=True)


if __name__ == '__main__':
    main()
