"""Original sequence metadata, separate from unique computational identifiers."""
import csv
from collections import Counter
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re

from bioautoml.execution import atomic_json, run_root


FIELDS = ['internal_id', 'original_id', 'original_header', 'source_file', 'source_id', 'record_number', 'split']


def source_key(filename, set_name):
    return hashlib.sha256(f'{Path(filename).resolve()}\0{set_name}'.encode()).hexdigest()


def safe_token(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', value)


def register_sources(root, files, labels, split):
    """Resolve source-name collisions before any FASTA or descriptor is written."""
    root = Path(root)
    directory = root / 'reports'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'sequence_sources.json'
    sources = json.loads(path.read_text()) if path.exists() else {}
    pairs = sorted(zip(files or [], labels or []), key=lambda pair: (str(pair[1]), str(Path(pair[0]).resolve())))
    stems = [safe_token(Path(filename).stem) for filename, _ in pairs]
    basenames = [Path(filename).name for filename, _ in pairs]
    used_tokens = {stem for stem in stems if stems.count(stem) == 1}
    for index, (filename, label) in enumerate(pairs):
        key = source_key(filename, f'{split}_{label}')
        stem = stems[index]
        # Unambiguous inputs retain their historical IDs and sort order.
        collision = stems.count(stem) > 1 or basenames.count(Path(filename).name) > 1
        token = f'{stem}_source{index + 1:04d}' if collision else stem
        if collision:
            while token in used_tokens:
                token += '_'
            used_tokens.add(token)
        value = dict(token=token, source_id=f'{split}_{index + 1:04d}', label=str(label), split=split,
                     output_name=f'pre_{token}{Path(filename).suffix}' if collision else f'pre_{Path(filename).name}')
        if key in sources and sources[key] != value:
            raise ValueError('Sequence sources changed within the same execution.')
        if key in [source_key(other, f'{split}_{other_label}') for other, other_label in pairs[:index]]:
            raise ValueError('The same sequence file and label were submitted twice.')
        sources[key] = value
    atomic_json(path, sources)
    _sources.cache_clear()


@lru_cache(maxsize=32)
def _sources(path, modified):
    return json.loads(Path(path).read_text())


def source_info(filename, set_name, root=None):
    root = root or os.environ.get('BIOAUTOML_RUN_ROOT')
    if root:
        path = Path(root) / 'reports/sequence_sources.json'
        if path.exists():
            value = _sources(str(path), path.stat().st_mtime_ns).get(source_key(filename, set_name))
            if value:
                return value
    return dict(token=safe_token(Path(filename).stem), source_id=source_key(filename, set_name),
                output_name=f'pre_{Path(filename).name}')


def sequence_id(filename, set_name, index, original_id, root=None, source=None):
    source = (source if source is not None else source_info(filename, set_name, root))['token']
    set_name = safe_token(set_name).strip('_')
    return f"pre_{f'{set_name}_' if set_name else ''}{source}_{index}_{original_id}"


def write_names(filename, set_name, records, root):
    if not root:
        return
    directory = Path(root) / 'reports/sequence_names'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{source_key(filename, set_name)}.tsv'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter='\t')
        writer.writeheader()
        writer.writerows(records)


def read_names(root):
    rows = {}
    for path in sorted((Path(root) / 'reports/sequence_names').glob('*.tsv')):
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle, delimiter='\t'):
                key = row['internal_id']
                if key in rows and rows[key] != row:
                    raise ValueError(f'Conflicting sequence metadata for {key}')
                rows[key] = row
    return rows


def names_for_run(root, model=None):
    rows = dict(model['sequence_names']) if model is not None and 'sequence_names' in model else {}
    rows.update(read_names(root))
    return rows


def display_names(ids, rows, context=False):
    ids = list(ids)
    counts = Counter(rows.get(str(key), {}).get('original_id', str(key)) for key in ids)
    values = []
    for key in ids:
        row = rows.get(str(key))
        name = row['original_id'] if row else str(key)
        if context and row and counts[name] > 1:
            name += f" [{row['source_file']}; {row['source_id']}; record {row['record_number']}]"
        values.append(name)
    return values


def annotate_predictions(frame, root):
    """Keep computational CSVs intact; annotate only final prediction exports."""
    rows = read_names(root)
    if not rows or 'nameseq' not in frame:
        return frame
    frame = frame.copy()
    ids = frame['nameseq'].astype(str).tolist()
    frame['internal_id'] = ids
    frame['nameseq'] = display_names(ids, rows)
    for field in ('original_header', 'source_file', 'source_id', 'record_number'):
        frame[field] = [rows.get(key, {}).get(field, '') for key in ids]
    return frame
