"""Automatic sequence-similarity groups, reusable CV folds and overlap audits.

No test sequence or test label participates in training group construction.
MMseqs2 is needed only for homology-aware training, never for inference.
"""
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
from Bio import SeqIO
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

from feature_execution import get_available_cpus

MMSEQS_VERSION = '15.6f452'
DEFAULT_IDENTITY = 90.0
DEFAULT_COVERAGE = 80.0
SHORT_SEQUENCE_LENGTH = 50


def validate_homology_settings(identity=DEFAULT_IDENTITY, coverage=DEFAULT_COVERAGE):
    """Validate public percentages and return MMseqs2 fractions (never guess units)."""
    values = []
    for name, value in (('identity', identity), ('coverage', coverage)):
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f'Homology {name} must be a numeric percentage from 1 to 100.') from None
        if isinstance(value, (bool, np.bool_)) or not math.isfinite(number) or not 1 <= number <= 100:
            raise ValueError(f'Homology {name} must be a finite percentage from 1 to 100; '
                             'use 90 for 90%, not 0.9.')
        values.append(number / 100)
    return tuple(values)


def homology_warnings(identity, coverage, data_type):
    threshold, minimum_coverage = validate_homology_settings(identity, coverage)
    messages = []
    if threshold < (0.3 if data_type.lower() == 'protein' else 0.7):
        messages.append('Low identity can connect weakly similar sequences into large groups; '
                        'this is not proof of biological homology.')
    if minimum_coverage < 0.5:
        messages.append('Coverage below 50% permits partial-sequence matches and may create large groups.')
    return messages


def add_homology_arguments(parser):
    # None distinguishes explicit overrides from absent options, including when
    # generation.py reuses assignments made with historical thresholds.
    parser.add_argument('--homology_identity', type=float, default=None,
                        help='Minimum identity in percent (1–100); default 90 for new sequence training')
    parser.add_argument('--homology_coverage', type=float, default=None,
                        help='Minimum coverage of EACH sequence in percent (1–100); default 80')


def resolve_homology_arguments(parser, args):
    if not args.homology_aware and (args.homology_identity is not None or args.homology_coverage is not None):
        parser.error('--homology_identity and --homology_coverage require --homology_aware')
    identity = DEFAULT_IDENTITY if args.homology_identity is None else args.homology_identity
    coverage = DEFAULT_COVERAGE if args.homology_coverage is None else args.homology_coverage
    try:
        validate_homology_settings(identity, coverage)
    except ValueError as error:
        parser.error(str(error))
    return identity, coverage


def validate_report_settings(report, identity=None, coverage=None):
    """Frozen reports are authoritative; reject only explicitly conflicting overrides."""
    validate_homology_settings(report['minimum_identity'] * 100,
                               report['minimum_query_and_target_coverage'] * 100)
    for value, key in ((identity, 'minimum_identity'), (coverage, 'minimum_query_and_target_coverage')):
        if value is not None:
            fraction, _ = validate_homology_settings(value, DEFAULT_COVERAGE)
            if not math.isclose(fraction, report[key], rel_tol=0, abs_tol=1e-12):
                raise ValueError('Requested homology thresholds conflict with frozen fold assignments. '
                                 'Rerun engineering.py in a new output directory to change thresholds.')


def get_cv_folds(y, task, n_splits, seed, groups=None):
    """Preserve historical random CV, or validate explicit group-disjoint CV."""
    y = np.asarray(y).ravel()
    X = np.zeros((len(y), 1))
    if groups is None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed) if task == 0 else KFold(
            n_splits=n_splits, shuffle=True, random_state=seed)
        return list(cv.split(X, y))
    groups = np.asarray(groups)
    if len(groups) != len(y) or len(np.unique(groups)) < n_splits:
        raise ValueError(f'Homology-aware {n_splits}-fold CV requires at least {n_splits} independent groups.')
    if task == 0:
        for label in np.unique(y):
            if len(np.unique(groups[y == label])) < n_splits:
                raise ValueError(f'Class {label!r} occurs in fewer than {n_splits} homology groups. '
                                 'Requested CV is infeasible; no random-split fallback was applied.')
        # Shuffle group tie-breaking, not their class-count rows. This avoids
        # poor stratification from shuffle=True in the locked sklearn 1.6.1.
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=False)
    else:
        cv = GroupKFold(n_splits=n_splits)
    unique_groups, group_indices = np.unique(groups, return_inverse=True)
    split_groups = np.random.default_rng(seed).permutation(len(unique_groups))[group_indices]
    folds = list(cv.split(X, y, split_groups))
    for train_idx, val_idx in folds:
        if set(groups[train_idx]) & set(groups[val_idx]):
            raise ValueError('A homology group crosses a fold boundary.')
        if len(val_idx) < 2 or len(train_idx) < 2:
            raise ValueError('Homology-aware CV produced a fold with fewer than two samples.')
        if task == 0 and (len(np.unique(y[train_idx])) != len(np.unique(y)) or
                          len(np.unique(y[val_idx])) != len(np.unique(y))):
            raise ValueError('Could not balance homology groups with every class in each fold. '
                             'No sequences were removed and no thresholds were relaxed.')
    return folds


def read_sequences(files, labels, split, data_type, task):
    """Map original FASTAs to the same source-qualified IDs as preprocessing.py.

    Similarity inputs are uppercased (U -> T for nucleotides). We do not remove
    ambiguity symbols, modify descriptor inputs, or collapse duplicate records.
    """
    if len(files) != len(labels):
        raise ValueError('Provide one label per FASTA for the sequence audit.')
    records = []
    seen = set()
    for filename, label in sorted(zip(files, labels), key=lambda pair: (str(pair[1]), str(Path(pair[0]).resolve()))):
        source = re.sub(r'[^A-Za-z0-9_.-]+', '_', Path(filename).stem)
        set_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', f'{split}_{label}').strip('_')
        for index, record in enumerate(SeqIO.parse(filename, 'fasta')):
            name = f'pre_{set_name}_{source}_{index}_{record.name}'
            if name in seen:
                raise ValueError(f'Ambiguous source-qualified sequence identifier: {name}')
            seen.add(name)
            sequence = str(record.seq).upper()
            if data_type == 'DNA/RNA':
                sequence = sequence.replace('U', 'T')
            if not sequence:
                raise ValueError(f'Empty sequence: {name}')
            target = None if label == 'Predicted' else str(label)
            if task == 1 and target is not None:
                target = str(float(record.id.split('|')[-1]))
            records.append({'sequence_id': name, 'original_id': record.id,
                            'source': Path(filename).name, 'sequence': sequence, 'label': target})
    return sorted(records, key=lambda record: record['sequence_id'])


def get_mmseqs():
    executable = shutil.which('mmseqs')
    if executable is None:
        raise RuntimeError('Homology-aware training requires MMseqs2. Use the locked Pixi or Docker environment.')
    version = subprocess.check_output([executable, 'version'], text=True).strip()
    if version != MMSEQS_VERSION:
        raise RuntimeError(f'Expected MMseqs2 {MMSEQS_VERSION}, found {version}. Use the locked environment.')
    return executable, version


def search_sequences(queries, targets, data_type, output, n_cpu, identity=DEFAULT_IDENTITY, coverage=DEFAULT_COVERAGE):
    """Yield qualifying matches; retain logs, but not temporary native databases.

    Long queries use sensitive heuristic search; short queries bypass the k-mer
    prefilter. Neither mode certifies the absence of all biological homology.
    """
    threshold, minimum_coverage = validate_homology_settings(identity, coverage)
    executable, _ = get_mmseqs()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mmseqs_', dir=output) as temporary:
        temporary = Path(temporary)
        target_file = temporary / 'target.fasta'
        target_file.write_text(''.join(f'>t{i}\n{record["sequence"]}\n' for i, record in enumerate(targets)))
        for mode in ('sensitive', 'short'):
            indices = [i for i, record in enumerate(queries)
                       if (len(record['sequence']) < SHORT_SEQUENCE_LENGTH) == (mode == 'short')]
            if not indices:
                continue
            query_file = temporary / f'{mode}.fasta'
            query_file.write_text(''.join(f'>q{i}\n{queries[i]["sequence"]}\n' for i in indices))
            result = temporary / f'{mode}.tsv'
            command = [executable, 'easy-search', str(query_file), str(target_file), str(result),
                       str(temporary / f'tmp_{mode}'), '--threads', str(get_available_cpus(n_cpu)),
                       '--search-type', '1' if data_type == 'Protein' else '3',
                       '--min-seq-id', str(threshold), '-c', str(minimum_coverage), '--cov-mode', '0',
                       '--alignment-mode', '3', '-a', '1', '--seq-id-mode', '0', '--max-seqs', str(len(targets)),
                       '-e', '1000000', '--mask', '0', '--comp-bias-corr', '0',
                       '--format-output', 'query,target,nident,alnlen,qstart,qend,qlen,tstart,tend,tlen', '-v', '2']
            if mode == 'short':
                command += ['--prefilter-mode', '2']
            else:
                command += ['-s', '7.5', '-k', '5' if data_type == 'Protein' else '8',
                            '--spaced-kmer-mode', '0', '--min-ungapped-score', '0']
            if data_type == 'DNA/RNA':
                command += ['--strand', '2']
            (output / f'{mode}_command.json').write_text(json.dumps(command, indent=2))
            with (output / f'{mode}.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            empty_alignments = 0
            with result.open() as handle:
                for row in csv.reader(handle, delimiter='\t'):
                    query, target = int(row[0][1:]), int(row[1][1:])
                    identical, length, qstart, qend, qlength, tstart, tend, tlength = map(int, row[2:])
                    # MMseqs2 can emit empty local alignments for very short
                    # sequences. These are not similarity evidence; exact
                    # sequence matches are handled independently upstream.
                    if length == 0:
                        empty_alignments += 1
                        continue
                    match_identity = identical / length
                    qcov = (abs(qend - qstart) + 1) / qlength
                    tcov = (abs(tend - tstart) + 1) / tlength
                    if match_identity >= threshold and min(qcov, tcov) >= minimum_coverage:
                        yield query, target, match_identity, qcov, tcov
            (output / f'{mode}_parser.json').write_text(json.dumps(
                {'empty_alignments_skipped': empty_alignments}, indent=2))
            if empty_alignments:
                print(f'MMseqs2 {mode}: ignored {empty_alignments} zero-length alignments.', flush=True)


def create_homology_report(train, data_type, task, seed, output, n_cpu, identity=DEFAULT_IDENTITY, coverage=DEFAULT_COVERAGE):
    """Create groups from training-only similarity links, and freeze both CVs."""
    started = time.monotonic()
    threshold, minimum_coverage = validate_homology_settings(identity, coverage)
    _, version = get_mmseqs()
    parents = list(range(len(train)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(first, second):
        first, second = root(first), root(second)
        parents[max(first, second)] = min(first, second)

    exact = {}
    for i, record in enumerate(train):
        if record['sequence'] in exact:
            join(i, exact[record['sequence']])
        else:
            exact[record['sequence']] = i
    matches = 0
    for first, second, _, _, _ in search_sequences(train, train, data_type, Path(output) / 'training_search', n_cpu, identity, coverage):
        if first != second:
            join(first, second)
            matches += 1
    groups = [f'group_{root(i):08d}' for i in range(len(train))]
    y = [record['label'] for record in train]
    if task == 1:
        y = np.asarray(y, dtype=float)
    rows = [{'sequence_id': record['sequence_id'], 'original_id': record['original_id'],
             'source': record['source'], 'label': record['label'], 'group_id': groups[i],
             'sequence_sha256': hashlib.sha256(record['sequence'].encode()).hexdigest()}
            for i, record in enumerate(train)]
    fold_summary = {}
    for count in (5, 10):
        folds = get_cv_folds(y, task, count, seed, groups)
        fold_summary[str(count)] = []
        for number, (_, val_idx) in enumerate(folds):
            for index in val_idx:
                rows[index][f'fold_{count}'] = number
            values, counts = np.unique(np.asarray(y)[val_idx], return_counts=True)
            fold_summary[str(count)].append({'fold': number, 'samples': len(val_idx),
                'groups': len(set(np.asarray(groups)[val_idx])),
                'classes': dict(zip(map(str, values), map(int, counts))) if task == 0 else None})
    report = {'enabled': True, 'schema_version': 1, 'task': task, 'seed': seed,
              'data_type': data_type, 'mmseqs_version': version,
              'minimum_identity': threshold,
              'minimum_query_and_target_coverage': minimum_coverage,
              'identity_definition': 'identical aligned residues / alignment length (including gaps)',
              'normalization': 'uppercase; nucleotide U -> T; descriptor inputs unchanged',
              'nucleotide_strands': 'both', 'short_query_nofilter_below_length': SHORT_SEQUENCE_LENGTH,
              'groups': len(set(groups)), 'training_samples': len(train),
              'detected_directed_similarity_links': matches,
              'detected_cross_fold_violations': 0, 'fold_summary': fold_summary,
              'evaluation': 'post-selection homology-aware CV, not nested AutoML evaluation',
              'limitation': 'Detected sequence similarity, not proof of absence of homology. '
                            'Domain-only relationships may not meet the bidirectional coverage criterion.',
              'rows': rows, 'minutes': (time.monotonic() - started) / 60}
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / 'homology_report.json').write_text(json.dumps(report, indent=2))
    with (Path(output) / 'fold_assignments.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Homology-aware CV: {report["groups"]} groups; grouping took {report["minutes"]:.3f} minutes.', flush=True)
    return report


def get_homology_folds(report, sequence_ids, task, seed, n_splits=5):
    """Resolve frozen fold assignments by ID, not by feature-table row order."""
    sequence_ids = np.asarray(sequence_ids).ravel().astype(str).tolist()
    rows = {row['sequence_id']: row for row in report['rows']}
    if (report['task'] != task or report['seed'] != seed or len(rows) != len(sequence_ids)
            or len(set(sequence_ids)) != len(sequence_ids) or set(rows) != set(sequence_ids)):
        raise ValueError('Homology assignments do not match the training IDs, task or seed.')
    assignment = np.array([rows[name][f'fold_{n_splits}'] for name in sequence_ids])
    return [(np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold)) for fold in range(n_splits)]


def audit_overlap(train, test, data_type, enabled, output, n_cpu, identity=DEFAULT_IDENTITY, coverage=DEFAULT_COVERAGE):
    """Audit original inputs without modifying groups, training or test membership."""
    started = time.monotonic()
    threshold, minimum_coverage = validate_homology_settings(identity, coverage)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    exact_index = {}
    for i, record in enumerate(train):
        exact_index.setdefault(record['sequence'], []).append(i)
    exact_queries, similar_queries, conflicts = set(), set(), 0
    exact_pairs = set()
    fields = ['test_sequence_id', 'train_sequence_id', 'exact', 'identity', 'test_coverage',
              'train_coverage', 'label_conflict']
    with (output / 'overlap_matches.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for query, record in enumerate(test):
            for target in exact_index.get(record['sequence'], []):
                known_labels = record['label'] is not None and train[target]['label'] is not None
                conflict = record['label'] != train[target]['label'] if known_labels else None
                conflicts += int(bool(conflict))
                exact_queries.add(query)
                exact_pairs.add((query, target))
                writer.writerow(dict(zip(fields, [record['sequence_id'], train[target]['sequence_id'],
                                                   True, 1.0, 1.0, 1.0, conflict])))
        if enabled and test:
            similar_queries.update(exact_queries)
            for query, target, match_identity, qcov, tcov in search_sequences(test, train, data_type, output / 'test_search', n_cpu, identity, coverage):
                similar_queries.add(query)
                if (query, target) not in exact_pairs:
                    writer.writerow(dict(zip(fields, [test[query]['sequence_id'], train[target]['sequence_id'],
                                                       False, match_identity, qcov, tcov, 'not assessed for nonidentical sequences'])))
    report = {'test_samples': len(test), 'training_samples': len(train),
              'exact_status': 'assessed' if test else 'not assessed: no external test sequences',
              'similarity_status': 'assessed' if enabled and test else 'not assessed',
              'exact_test_sequences': len(exact_queries),
              'exact_test_percent': 100 * len(exact_queries) / len(test) if test else None,
              'similar_test_sequences_including_exact': len(similar_queries) if enabled and test else None,
              'similar_test_percent_including_exact': 100 * len(similar_queries) / len(test) if enabled and test else None,
              'conflicting_exact_pairs': conflicts, 'original_partitions_preserved': True,
              'label_conflicts_status': ('assessed' if all(record['label'] is not None for record in train + test)
                                         else 'not assessed for pairs with unknown labels') if test else 'not assessed',
              'exact_definition': 'equal full-length uppercase sequences; nucleotide U normalized to T; forward orientation',
              'minimum_identity': threshold if enabled else None,
              'minimum_query_and_target_coverage': minimum_coverage if enabled else None,
              'mmseqs_version': MMSEQS_VERSION if enabled and test else None,
              'minutes': (time.monotonic() - started) / 60}
    (output / 'overlap_report.json').write_text(json.dumps(report, indent=2))
    print(f'Train-test overlap: {len(exact_queries)}/{len(test)} exact; similarity {report["similarity_status"]}. '
          f'Audit took {report["minutes"]:.3f} minutes.', flush=True)
    if exact_queries or similar_queries:
        print('WARNING: detected train-test sequence overlap; external scores may benefit from similarity.', flush=True)
    return report


def prepare_homology(fasta_train, labels_train, fasta_test, labels_test, data_type, task, seed, output, enabled=False, n_cpu=-1,
                     identity=DEFAULT_IDENTITY, coverage=DEFAULT_COVERAGE):
    """Run sequence-only preparation before feature extraction / AutoML."""
    validate_homology_settings(identity, coverage)
    if data_type.lower() not in ('protein', 'dna/rna', 'nucleotide'):
        raise ValueError('Sequence grouping requires protein or DNA/RNA FASTA inputs, not structured data.')
    data_type = 'Protein' if data_type.lower() == 'protein' else 'DNA/RNA'
    if enabled:
        for message in homology_warnings(identity, coverage, data_type):
            warnings.warn(message, UserWarning)
    train = read_sequences(fasta_train, labels_train, 'train', data_type, task)
    if not train:
        raise ValueError('No training sequences were found.')
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / 'homology_report.json').write_text(json.dumps({'enabled': False, 'status': 'preparing' if enabled else 'disabled'}))
    report = create_homology_report(train, data_type, task, seed, output, n_cpu, identity, coverage) if enabled else None
    # Deliberately read and search test inputs only after training groups are frozen.
    test = read_sequences(fasta_test or [], labels_test or [], 'test', data_type, task)
    audit_overlap(train, test, data_type, enabled, output, n_cpu, identity, coverage)
    return report
