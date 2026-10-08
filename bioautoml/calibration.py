"""Training-only probability calibration with explicit, group-safe folds."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from bioautoml.homology import get_cv_folds, get_homology_folds


def get_base_pipeline(model):
    """Return the fitted underlying pipeline for inspection, not prediction."""
    if isinstance(model, CalibratedClassifierCV):
        return model.calibrated_classifiers_[0].estimator
    return model


def get_calibration_groups(report, sequence_ids):
    if not report:
        return None
    rows = {row['sequence_id']: row['group_id'] for row in report['rows']}
    ids = np.asarray(sequence_ids).ravel().astype(str)
    if len(ids) != len(rows) or len(set(ids)) != len(ids) or set(ids) != set(rows):
        raise ValueError('Calibration group assignments do not match the training sequence IDs.')
    return np.asarray([rows[name] for name in ids])


def validate_folds(y, folds, groups=None):
    y = np.asarray(y)
    classes = set(y)
    if len(classes) < 2:
        raise ValueError('Probability calibration requires at least two classes.')
    validation = []
    for train, test in folds:
        if set(train) & set(test) or set(train) | set(test) != set(range(len(y))):
            raise ValueError('Calibration folds must partition the training rows without overlap.')
        if set(y[train]) != classes or set(y[test]) != classes:
            raise ValueError('Probability calibration requires every class in each training and validation fold.')
        if groups is not None and set(groups[train]) & set(groups[test]):
            raise ValueError('A similarity group crosses a calibration fold boundary.')
        validation.extend(test)
    if sorted(validation) != list(range(len(y))):
        raise ValueError('Each row must receive exactly one held-out calibration prediction.')


def prepare_calibration_folds(y, seed, report=None, sequence_ids=None):
    """Preflight final five-fold calibration and the inner folds of ten-fold CV."""
    y = np.asarray(y)
    groups = get_calibration_groups(report, sequence_ids) if report else None
    final = (get_homology_folds(report, sequence_ids, 0, seed, 5) if report
             else get_cv_folds(y, 0, 5, seed))
    reporting = (get_homology_folds(report, sequence_ids, 0, seed, 10) if report
                 else get_cv_folds(y, 0, 10, seed))
    validate_folds(y, final, groups)
    validate_folds(y, reporting, groups)
    inner = []
    for number, (train, _) in enumerate(reporting):
        subset_groups = groups[train] if groups is not None else None
        try:
            folds = get_cv_folds(y[train], 0, 5, seed, subset_groups)
            validate_folds(y[train], folds, subset_groups)
        except ValueError as error:
            raise ValueError(f'Calibration is infeasible inside reporting fold {number + 1}: {error}') from error
        inner.append(folds)
    return {'final': final, 'reporting': reporting, 'inner': inner}


def make_calibrated_model(model, X, folds):
    # Contiguous column blocks preserve feature order without one transformer
    # per descriptor feature (sequence tables can contain thousands of columns).
    # Encoding and imputation are fitted inside each calibration-training fold.
    categorical = set(X.select_dtypes(include=['object', 'category', 'string']).columns)
    blocks = []
    for column in X.columns:
        is_categorical = column in categorical
        if not blocks or blocks[-1][0] != is_categorical:
            blocks.append((is_categorical, []))
        blocks[-1][1].append(column)
    transformers = [(f'columns_{index}',
                     OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                     if is_categorical else 'passthrough', columns)
                    for index, (is_categorical, columns) in enumerate(blocks)]
    pipeline = Pipeline([
        ('encoder', ColumnTransformer(transformers, sparse_threshold=0)),
        ('imputer', SimpleImputer(strategy='mean', keep_empty_features=True)),
        ('clf', clone(get_base_pipeline(model).named_steps['clf'])),
    ])
    return CalibratedClassifierCV(pipeline, method='sigmoid', cv=folds, ensemble=False, n_jobs=1)


def probability_metrics(y, probabilities):
    """Encoded classes are 0..K-1 in the same order as probability columns."""
    y = np.asarray(y, dtype=int)
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2 or len(y) == 0:
        raise ValueError('Probabilities must be a nonempty matrix with one column per class.')
    classes = probabilities.shape[1]
    if (probabilities.shape[0] != len(y) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1)
            or (y < 0).any() or (y >= classes).any()):
        raise ValueError('Invalid probability array or class ordering.')
    if classes == 2:
        brier = np.mean((probabilities[:, 1] - (y == 1)) ** 2)
    else:
        brier = np.mean(np.sum((probabilities - np.eye(classes)[y]) ** 2, axis=1))
    return {'Log_loss': float(log_loss(y, probabilities, labels=np.arange(classes))),
            'Brier': float(brier)}


def save_probability_report(y, probabilities, labels, output, prefix, raw_probabilities=None):
    """Evaluation-only reports. Never fit or select a calibrator here."""
    from matplotlib.figure import Figure
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = probability_metrics(y, probabilities)
    report.update(samples=len(y), classes=list(map(str, labels)),
                  brier_definition='positive-class squared error' if len(labels) == 2 else
                  'mean sum of squared errors across classes, unscaled (range 0 to 2)',
                  evaluation='post-selection CV, not fully nested AutoML' if prefix == 'cv' else 'external evaluation only')
    predictions = {'calibrated': probabilities}
    if raw_probabilities is not None:
        report['uncalibrated_same_estimator'] = probability_metrics(y, raw_probabilities)
        predictions['uncalibrated'] = raw_probabilities
    rows = []
    figure = Figure(figsize=(7, 5))
    axes = figure.subplots()
    axes.plot([0, 1], [0, 1], '--', color='gray')
    y = np.asarray(y)
    for mode, values in predictions.items():
        for index in ([1] if len(labels) == 2 else range(len(labels))):
            bins = np.minimum((values[:, index] * 10).astype(int), 9)
            points = []
            for number in range(10):
                selected = bins == number
                count = int(selected.sum())
                mean = float(values[selected, index].mean()) if count else None
                observed = float((y[selected] == index).mean()) if count else None
                rows.append(dict(mode=mode, label=str(labels[index]), bin=number, count=count,
                                 mean_probability=mean, observed_frequency=observed))
                if count:
                    points.append((mean, observed))
            if points:
                axes.plot(*zip(*points), marker='o', linestyle='-' if mode == 'calibrated' else ':',
                          label=f'{labels[index]} ({mode})')
    axes.set(xlabel='Mean predicted probability', ylabel='Observed class frequency', xlim=(0, 1), ylim=(0, 1))
    axes.legend(fontsize='small')
    figure.tight_layout()
    figure.savefig(output / f'{prefix}_reliability.svg')
    pd.DataFrame(rows).to_csv(output / f'{prefix}_reliability.csv', index=False)
    (output / f'{prefix}_probability_metrics.json').write_text(json.dumps(report, indent=2))
    return report
