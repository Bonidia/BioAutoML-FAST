"""Additional holdout evaluation after excluding detected train-test matches."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)

from bioautoml.execution import atomic_json, file_hash, run_path


def prepare_test_filter(output, train_files, train_labels, test_files, test_labels, data_type, task):
    from bioautoml.homology import read_sequences
    data_type = 'Protein' if data_type.lower() == 'protein' else 'DNA/RNA'
    path = Path(run_path(output, 'homology'))
    train = read_sequences(train_files, train_labels, 'train', data_type, task)
    test = read_sequences(test_files, test_labels, 'test', data_type, task)
    if not test or any(row['label'] is None for row in test):
        raise ValueError('Test-overlap filtering requires a labelled test set.')
    atomic_json(path / 'test_filter_inputs.json', {
        'train_ids': [row['sequence_id'] for row in train],
        'test_labels': {row['sequence_id']: row['label'] for row in test},
        'audit_sha256': file_hash(path / 'overlap_report.json'),
        'matches_sha256': file_hash(path / 'overlap_matches.csv'),
    })


def test_filter_mask(output, train_ids, test_ids, test_labels, task, homology_report):
    path = Path(run_path(output, 'homology'))
    with (path / 'test_filter_inputs.json').open() as handle:
        inputs = json.load(handle)
    with (path / 'overlap_report.json').open() as handle:
        audit = json.load(handle)
    if (inputs['audit_sha256'] != file_hash(path / 'overlap_report.json') or
            inputs['matches_sha256'] != file_hash(path / 'overlap_matches.csv')):
        raise ValueError('The overlap audit changed after test-filter preparation.')
    if audit.get('similarity_status') != 'assessed':
        raise ValueError('Test filtering requires a completed similarity audit.')
    for key in ('minimum_identity', 'minimum_query_and_target_coverage'):
        if audit[key] != homology_report[key]:
            raise ValueError('Test audit and CV grouping thresholds differ.')
    train_ids = np.asarray(train_ids).reshape(-1).astype(str)
    test_ids = np.asarray(test_ids).reshape(-1).astype(str)
    labels = np.asarray(test_labels).reshape(-1)
    if (len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids) or
            set(train_ids) != set(inputs['train_ids']) or
            set(test_ids) != set(inputs['test_labels']) or len(labels) != len(test_ids) or
            audit['training_samples'] != len(train_ids) or audit['test_samples'] != len(test_ids)):
        raise ValueError('Sequence IDs do not match the audited training/test sets.')
    expected = np.asarray([inputs['test_labels'][name] for name in test_ids])
    if task == 1:
        matches_labels = np.array_equal(labels.astype(float), expected.astype(float))
    else:
        matches_labels = np.array_equal(labels.astype(str), expected.astype(str))
    if not matches_labels:
        raise ValueError('Test labels do not match the audited sequence order.')
    matches = pd.read_csv(path / 'overlap_matches.csv', dtype=str, keep_default_na=False)
    if (not set(matches.test_sequence_id) <= set(test_ids) or
            not set(matches.train_sequence_id) <= set(train_ids)):
        raise ValueError('Overlap matches contain unknown sequence IDs.')
    excluded = set(matches.test_sequence_id)
    exact = set(matches.loc[matches.exact.str.lower() == 'true', 'test_sequence_id'])
    if (len(excluded) != audit['similar_test_sequences_including_exact'] or
            len(exact) != audit['exact_test_sequences']):
        raise ValueError('Overlap counts do not match the audit.')
    mask = np.asarray([name not in excluded for name in test_ids])
    membership = pd.DataFrame({
        'sequence_id': test_ids, 'retained': mask,
        'reason': ['exact training match' if name in exact else
                   'similar training match' if name in excluded else 'no detected training match'
                   for name in test_ids],
    })
    membership.to_csv(path / 'test_filter_membership.csv', index=False)
    return mask


def save_filtered_test(output, task, test_labels, predictions, mask, probabilities=None, classes=None):
    labels = np.asarray(test_labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    mask = np.asarray(mask, dtype=bool)
    if not (len(labels) == len(predictions) == len(mask)):
        raise ValueError('Filtered evaluation inputs have different lengths.')
    y, pred = labels[mask], predictions[mask]
    count = len(y)
    values, reasons = {}, {}
    if task == 0:
        metrics = ['Accuracy', 'Balanced ACC', 'MCC', 'F1 weighted', 'AUC']
        if count == 0 or len(np.unique(y)) < 2:
            reason = 'no retained sequences' if count == 0 else 'fewer than two retained classes'
            reasons = dict.fromkeys(metrics, reason)
        else:
            values = {'Accuracy': accuracy_score(y, pred),
                      'Balanced ACC': balanced_accuracy_score(y, pred),
                      'MCC': matthews_corrcoef(y, pred),
                      'F1 weighted': f1_score(y, pred, average='weighted', zero_division=0)}
            if probabilities is None or classes is None:
                reasons['AUC'] = 'class probabilities unavailable'
            elif set(np.unique(y)) != set(classes):
                reasons['AUC'] = 'not all training classes retained'
            else:
                probs = np.asarray(probabilities)[mask]
                values['AUC'] = (roc_auc_score(y, probs[:, 1], labels=classes) if len(classes) == 2
                                 else roc_auc_score(y, probs, labels=classes, multi_class='ovr', average='macro'))
    else:
        metrics = ['MAE', 'MSE', 'RMSE', 'R2', 'Pearson']
        if count == 0:
            reasons = dict.fromkeys(metrics, 'no retained sequences')
        else:
            values = {'MAE': mean_absolute_error(y, pred), 'MSE': mean_squared_error(y, pred)}
            values['RMSE'] = np.sqrt(values['MSE'])
            if count < 2 or np.ptp(y.astype(float)) == 0:
                reason = 'fewer than two retained observations' if count < 2 else 'constant retained targets'
                reasons.update(R2=reason, Pearson=reason)
            else:
                values['R2'] = r2_score(y, pred)
                if np.ptp(pred.astype(float)) == 0:
                    reasons['Pearson'] = 'constant predictions'
                else:
                    values['Pearson'] = float(np.corrcoef(y.astype(float), pred.astype(float))[0, 1])
    metrics_path = Path(output) / 'results/metrics/filtered_test_metrics.csv'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{'Metric': metric, 'Value': values.get(metric, np.nan),
                   'status': reasons.get(metric, 'evaluated'), 'n_observations': count}
                  for metric in metrics]).to_csv(metrics_path, index=False)
    original = pd.read_csv(run_path(output, 'test_predictions.csv'))
    if len(original) != len(mask):
        raise ValueError('Saved predictions do not match the evaluated test set.')
    original.loc[mask].to_csv(Path(output) / 'results/predictions/filtered_test_predictions.csv', index=False)
    with open(run_path(output, 'homology', 'overlap_report.json')) as handle:
        audit = json.load(handle)
    report = {'original_test_samples': len(mask), 'retained_test_samples': count,
              'excluded_test_samples': int((~mask).sum()), 'original_test_preserved': True,
              'minimum_identity': audit['minimum_identity'],
              'minimum_query_and_target_coverage': audit['minimum_query_and_target_coverage'],
              'metric_limitations': reasons}
    if task == 0:
        report['original_class_counts'] = {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
        report['retained_class_counts'] = {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    atomic_json(Path(run_path(output, 'homology', 'test_filter_report.json')), report)
    print(f'Filtered test evaluation: retained {count}/{len(mask)} sequences. Full-test results preserved.')
