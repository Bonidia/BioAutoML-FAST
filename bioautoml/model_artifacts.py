"""Lazy, self-contained model artifacts. Only load files from trusted sources.

Version 2 bundles authenticate independently loadable joblib sections. Unsigned
current-format artifacts require explicit local trust and are forbidden on the web.
"""
import hashlib
from importlib.metadata import version
from collections.abc import Mapping
import json
import os
import re
from pathlib import Path
import shutil
import tempfile
import zipfile
import uuid

import joblib
import numpy as np
from bioautoml.execution import timed
from bioautoml.model_security import PACKAGES, sign_manifest, verify_signature


SUMMARY_KEYS = frozenset({
    'train_stats', 'cross_validation', 'confusion_matrix', 'descriptors',
    'feature_importance', 'pearson_folds',
})
EXPLORATION_KEYS = frozenset({'train', 'train_labels', 'nameseq_train', 'homology_report', 'overlap_report', 'sequence_names'})
PARTS = ('summary', 'prediction', 'exploration')


def validate_model_support(model):
    """Reject retired model types without silently changing their predictions."""
    message = 'Probability-calibrated models are no longer supported. Retrain the model with the current application.'
    if 'calibration' in model:
        raise ValueError(message)
    # New bundles can be checked from their manifest without loading estimators.
    if not isinstance(model, ModelArtifact) and 'clf' in model:
        if any(cls.__module__ == 'sklearn.calibration' for cls in type(model['clf']).__mro__):
            raise ValueError(message)


def model_fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def get_model_info(model):
    """Display metadata without loading an estimator from a new bundle."""
    validate_model_support(model)
    if isinstance(model, ModelArtifact):
        return model.info
    clf = model['clf']
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
    def __init__(self, path, trust_unsigned=False):
        self.path = Path(path).resolve()
        self.fingerprint = model_fingerprint(self.path)
        self._loaded = {}
        self.signature = None
        with zipfile.ZipFile(self.path) as archive:
            names = archive.namelist()
            expected = {'manifest.json', *(f'{part}.joblib' for part in PARTS)}
            if 'signature.json' in names:
                expected.add('signature.json')
            if set(names) != expected or len(names) != len(expected):
                raise ValueError('Invalid model bundle members.')
            if archive.getinfo('manifest.json').file_size > 4 * 1024 * 1024:
                raise ValueError('Model manifest is too large.')
            limit = int(os.environ.get('BIOAUTOML_MAX_MODEL_BYTES', 32 * 1024**3))
            if sum(member.file_size for member in archive.infolist()) > limit:
                raise ValueError('Model exceeds the configured size limit.')
            if any(member.compress_type != zipfile.ZIP_STORED for member in archive.infolist()):
                raise ValueError('Compressed model members are not supported.')
            manifest = json.loads(archive.read('manifest.json'))
            if 'signature.json' in names:
                if archive.getinfo('signature.json').file_size > 4096:
                    raise ValueError('Invalid signature metadata size.')
                self.signature = json.loads(archive.read('signature.json'))
                verify_signature(manifest, self.signature)
            elif not trust_unsigned:
                raise ValueError('Unsigned model rejected. Web models must have a trusted signature.')
        if manifest.get('format') != 'BioAutoML-FAST' or manifest.get('version') != 2:
            raise ValueError('Unsupported model artifact version.')
        if set(manifest.get('hashes', {})) != set(PARTS):
            raise ValueError('Invalid model section hashes.')
        if any(not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value)
               for value in manifest['hashes'].values()):
            raise ValueError('Invalid model section digest.')
        self.manifest = manifest
        self.model_id = manifest.get('model_id')
        self.verified = self.signature is not None
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
        validate_model_support(self)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def __contains__(self, key):
        return key in self._keys

    @timed('model_section_loading')
    def __getitem__(self, key):
        part = self._keys[key]
        if self.signature:
            verify_signature(self.manifest, self.signature)
        if model_fingerprint(self.path) != self.fingerprint:
            raise ValueError('Model artifact changed; reload it before continuing.')
        if part not in self._loaded:
            with zipfile.ZipFile(self.path) as archive, archive.open(f'{part}.joblib') as handle:
                # Private snapshot: verify exactly the bytes later deserialized,
                # even if an attacker modifies/replaces the source file mid-read.
                with tempfile.TemporaryFile() as snapshot:
                    digest = hashlib.sha256()
                    limit = int(os.environ.get('BIOAUTOML_MAX_MODEL_BYTES', 32 * 1024**3))
                    size = 0
                    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                        size += len(chunk)
                        if size > limit:
                            raise ValueError('Model section exceeds size limit.')
                        digest.update(chunk)
                        snapshot.write(chunk)
                    expected = self.manifest['hashes'][part]
                    if digest.hexdigest() != expected:
                        raise ValueError('Model section hash mismatch; rejected before deserialization.')
                    snapshot.seek(0)
                    values = joblib.load(snapshot)
            if not isinstance(values, dict) or set(values) != {k for k, p in self._keys.items() if p == part}:
                raise ValueError('Model section does not match its manifest.')
            validate_model_support(values)
            self._loaded[part] = values
        return self._loaded[part][key]


@timed('model_loading')
def load_model(path, *, trust_unsigned=False):
    """Verify web models; unsigned loading is an explicit local-only decision."""
    if not zipfile.is_zipfile(path):
        raise ValueError('Unsupported model format; retrain with the current application.')
    return ModelArtifact(path, trust_unsigned=trust_unsigned)


def _write_bundle(path, sections, info, source=None):
    """Publish atomically. Unchanged sections can be streamed from a bundle."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        manifest = {'format': 'BioAutoML-FAST', 'version': 2, 'info': info, 'sections': {},
                    'model_id': str(uuid.uuid4()), 'packages': {name: version(name) for name in PACKAGES}, 'hashes': {}}
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
        with zipfile.ZipFile(temporary, 'r') as archive:
            for part in PARTS:
                with archive.open(f'{part}.joblib') as handle:
                    manifest['hashes'][part] = hashlib.file_digest(handle, 'sha256').hexdigest()
        with zipfile.ZipFile(temporary, 'a', compression=zipfile.ZIP_STORED) as archive:
            archive.writestr('manifest.json', json.dumps(manifest, default=_json_default))
        # Match ordinary model-file readability for bind-mounted repositories;
        # preserve explicit permissions when replacing an existing artifact.
        os.chmod(temporary, path.stat().st_mode & 0o777 if path.exists() else 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@timed('model_saving')
def save_model(model, path):
    """Keep all original values and dtypes, partitioning only their storage."""
    validate_model_support(model)
    sections = {part: {} for part in PARTS}
    for key in model:
        part = 'summary' if key in SUMMARY_KEYS else 'exploration' if key in EXPLORATION_KEYS else 'prediction'
        sections[part][key] = model[key]
    _write_bundle(path, sections, get_model_info(model))


def update_model_summary(path, *, trust_unsigned=False, **updates):
    """Attach web statistics without deserializing the training matrix."""
    if not set(updates) <= SUMMARY_KEYS:
        raise ValueError('Only summary fields may be updated.')
    model = load_model(path, trust_unsigned=trust_unsigned)
    if model.verified:
        raise ValueError('Signed models are immutable; finalize statistics before signing.')
    summary = {key: model[key] for key in model if model._keys[key] == 'summary'}
    summary.update(updates)
    _write_bundle(path, dict(summary=summary, prediction=None, exploration=None), model.info, source=model)


@timed('model_signing')
def sign_web_model(path):
    """Trusted training-completion hook, never called for uploads or inference."""
    source = ModelArtifact(path, trust_unsigned=True)
    if source.verified:
        raise ValueError('Already signed; models must not be re-signed on upload.')
    if source.manifest['version'] != 2:
        raise ValueError('Only freshly created version 2 models can be signed.')
    signature = sign_manifest(source.manifest)
    # Hash each section during copying. No unsigned payload is deserialized here.
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(path) as original, zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED) as archive:
            for part in PARTS:
                digest = hashlib.sha256()
                with original.open(f'{part}.joblib') as src, archive.open(f'{part}.joblib', 'w', force_zip64=True) as dst:
                    for chunk in iter(lambda: src.read(1024 * 1024), b''):
                        digest.update(chunk)
                        dst.write(chunk)
                if digest.hexdigest() != source.manifest['hashes'][part]:
                    raise ValueError('Training artifact changed before signing.')
            archive.writestr('manifest.json', json.dumps(source.manifest))
            archive.writestr('signature.json', json.dumps(signature))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
