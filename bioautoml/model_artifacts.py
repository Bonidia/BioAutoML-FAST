"""Lazy, self-contained model artifacts. Only load files from trusted sources.

Version 1 bundles are uncompressed ZIP files with independently loadable joblib
sections. Legacy joblib dictionaries remain readable. No archive extraction or
process-global model cache is used; pickle/joblib is NOT an untrusted-data format.
"""
import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import joblib
import numpy as np


SUMMARY_KEYS = frozenset({
    'train_stats', 'cross_validation', 'confusion_matrix', 'descriptors',
    'feature_importance', 'calibration', 'pearson_folds',
})
EXPLORATION_KEYS = frozenset({'train', 'train_labels', 'nameseq_train', 'homology_report', 'overlap_report'})
PARTS = ('summary', 'prediction', 'exploration')


def model_fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def get_model_info(model):
    """Display metadata without loading an estimator from a new bundle."""
    if isinstance(model, ModelArtifact):
        return model.info
    from bioautoml.calibration import get_base_pipeline
    clf = get_base_pipeline(model['clf'])
    estimator = clf.named_steps['clf'] if hasattr(clf, 'named_steps') else clf
    descriptors = model.get('descriptors')
    data_type = ('Structured data' if descriptors is None else
                 'DNA/RNA' if 'NAC' in descriptors.columns else 'Protein')
    return {
        'task': 'Classification' if 'label_encoder' in model else 'Regression',
        'data_type': data_type,
        'estimator_name': type(estimator).__name__,
        'params': estimator.get_params(deep=False),
        'num_train': len(model['train']),
    }


def get_training_count(model):
    return model.info['num_train'] if isinstance(model, ModelArtifact) else len(model['train'])


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    # Display-only parameters (e.g. callable objectives); fitted objects are
    # preserved losslessly in prediction.joblib, never reconstructed from JSON.
    return str(value)


class ModelArtifact(Mapping):
    """Session-owned mapping; load a section only on its first key access."""
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.fingerprint = model_fingerprint(self.path)
        self._loaded = {}
        with zipfile.ZipFile(self.path) as archive:
            names = archive.namelist()
            expected = {'manifest.json', *(f'{part}.joblib' for part in PARTS)}
            if set(names) != expected or len(names) != len(expected):
                raise ValueError('Invalid model bundle members.')
            if archive.getinfo('manifest.json').file_size > 4 * 1024 * 1024:
                raise ValueError('Model manifest is too large.')
            manifest = json.loads(archive.read('manifest.json'))
        if manifest.get('format') != 'BioAutoML-FAST' or manifest.get('version') != 1:
            raise ValueError('Unsupported model artifact version.')
        sections = manifest['sections']
        if set(sections) != set(PARTS):
            raise ValueError('Invalid model sections.')
        self._keys = {}
        for part, keys in sections.items():
            if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
                raise ValueError('Invalid model keys.')
            for key in keys:
                if key in self._keys:
                    raise ValueError('Duplicate model key.')
                self._keys[key] = part
        self.info = manifest['info']

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def __contains__(self, key):
        return key in self._keys

    def __getitem__(self, key):
        part = self._keys[key]
        if model_fingerprint(self.path) != self.fingerprint:
            raise ValueError('Model artifact changed; reload it before continuing.')
        if part not in self._loaded:
            with zipfile.ZipFile(self.path) as archive, archive.open(f'{part}.joblib') as handle:
                values = joblib.load(handle)
            if not isinstance(values, dict) or set(values) != {k for k, p in self._keys.items() if p == part}:
                raise ValueError('Model section does not match its manifest.')
            self._loaded[part] = values
        return self._loaded[part][key]


def load_model(path, mmap_mode=None):
    """Load trusted new or legacy artifacts; new sections stay lazy."""
    if zipfile.is_zipfile(path):
        return ModelArtifact(path)
    model = joblib.load(path, mmap_mode=mmap_mode)
    if not isinstance(model, dict):
        raise ValueError('Expected a BioAutoML-FAST model dictionary.')
    return model


def _write_bundle(path, sections, info, source=None):
    """Publish atomically. Unchanged sections can be streamed from a bundle."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        manifest = {'format': 'BioAutoML-FAST', 'version': 1, 'info': info, 'sections': {}}
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for part in PARTS:
                values = sections[part]
                if values is None:
                    manifest['sections'][part] = [key for key, section in source._keys.items() if section == part]
                    with zipfile.ZipFile(source.path) as original, original.open(f'{part}.joblib') as src:
                        with archive.open(f'{part}.joblib', 'w', force_zip64=True) as dst:
                            shutil.copyfileobj(src, dst, length=1024 * 1024)
                else:
                    manifest['sections'][part] = list(values)
                    with archive.open(f'{part}.joblib', 'w', force_zip64=True) as dst:
                        joblib.dump(values, dst, compress=0)
            archive.writestr('manifest.json', json.dumps(manifest, default=_json_default))
        # Match ordinary model-file readability for bind-mounted repositories;
        # preserve explicit permissions when replacing an existing artifact.
        os.chmod(temporary, path.stat().st_mode & 0o777 if path.exists() else 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_model(model, path):
    """Keep all original values and dtypes, partitioning only their storage."""
    sections = {part: {} for part in PARTS}
    for key in model:
        part = 'summary' if key in SUMMARY_KEYS else 'exploration' if key in EXPLORATION_KEYS else 'prediction'
        sections[part][key] = model[key]
    _write_bundle(path, sections, get_model_info(model))


def update_model_summary(path, **updates):
    """Attach web statistics without deserializing the training matrix."""
    if not set(updates) <= SUMMARY_KEYS:
        raise ValueError('Only summary fields may be updated.')
    model = load_model(path)
    if isinstance(model, ModelArtifact):
        summary = {key: model[key] for key in model if model._keys[key] == 'summary'}
        summary.update(updates)
        _write_bundle(path, dict(summary=summary, prediction=None, exploration=None), model.info, source=model)
    else:
        model.update(updates)
        save_model(model, path)


def convert_model(source, destination):
    """Non-destructive, one-time conversion; never overwrite an existing file."""
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Destination already exists: {destination}')
    save_model(load_model(source), destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a trusted legacy model to the lazy, single-file format.')
    parser.add_argument('source')
    parser.add_argument('destination', help='New .sav path; the original is preserved')
    parser.add_argument('--trust-model', action='store_true', help='Acknowledge that joblib loading can execute code')
    args = parser.parse_args()
    if not args.trust_model:
        parser.error('Only convert files you trust; pass --trust-model to confirm.')
    convert_model(args.source, args.destination)
    print(f'Saved {args.destination}; original preserved.')
