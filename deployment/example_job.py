"""Verify, export, and install the queued example without a separate manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import uuid
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'App')]

from bioautoml.execution import run_path
from bioautoml.model_artifacts import ModelArtifact
from utils.db import TaskResultManager
from utils.retention import job_directory, job_lock, jobs_root


FILES = ('run.json', 'execution_summary.json', 'performance_summary.csv', 'timings.csv')
DIRECTORIES = ('inputs', 'model', 'results', 'reports', 'work', 'logs')


def verify_job(path):
    if path.is_symlink() or not path.is_dir() or str(uuid.UUID(path.name)) != path.name:
        raise ValueError('Example must be a UUID-named job directory.')
    for child in path.rglob('*'):
        if child.is_symlink() or not (child.is_file() or child.is_dir()):
            raise ValueError('Example contains a symlink or special file.')
    record = json.loads((path / 'run.json').read_text())
    if record['status'] != 'completed' or record['origin'] != 'web' or record['operation'] != 'training':
        raise ValueError('Example must be a completed web training job.')
    if not record.get('started_at') or not record.get('ended_at'):
        raise ValueError('Example timestamps are missing.')
    model = ModelArtifact(run_path(path, 'trained_model.sav'))
    if record.get('model_id') != model.model_id or record.get('signer') != model.signature['key_id']:
        raise ValueError('Example run record does not match its signed model.')
    with zipfile.ZipFile(model.path) as archive:
        for part, expected in model.manifest['hashes'].items():
            digest = hashlib.sha256()
            with archive.open(part + '.joblib') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError('Example model section checksum mismatch.')
    for name in ('training_kfold(10)_metrics.csv', 'metrics_test.csv', 'test_predictions.csv',
                 'train_stats.csv', 'test_stats.csv', 'job_info.tsv'):
        if not Path(run_path(path, name)).is_file():
            raise ValueError(f'Missing example output: {name}')
    return record


def export_job(source, destination):
    verify_job(source)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    target = destination / source.name
    target.mkdir()
    for name in FILES:
        if (source / name).is_file():
            shutil.copy2(source / name, target / name)
    for name in DIRECTORIES:
        if (source / name).is_dir():
            shutil.copytree(source / name, target / name)
    verify_job(target)
    for path in (destination, target, *target.rglob('*')):
        path.chmod(0o755 if path.is_dir() else 0o644)
    print(f'Exported example {source.name}', flush=True)


def install_example(bundle):
    if not bundle.exists():
        return None
    candidates = list(bundle.iterdir())
    if len(candidates) != 1:
        raise ValueError('Expected exactly one bundled example job.')
    source = candidates[0]
    record = verify_job(source)
    identifier = source.name
    target = job_directory(identifier)
    manager = TaskResultManager()
    with job_lock(identifier):
        existing = manager.get_result(identifier)
        if existing and manager.get_example_job_id() != identifier:
            raise ValueError('Example ID collides with an unrelated job; refusing to overwrite.')
        if target.exists():
            installed = verify_job(target)
            source_model = ModelArtifact(run_path(source, 'trained_model.sav'))
            installed_model = ModelArtifact(run_path(target, 'trained_model.sav'))
            if installed != record or installed_model.manifest != source_model.manifest or installed_model.signature != source_model.signature:
                raise ValueError('Existing job differs from bundled example; refusing to overwrite.')
        else:
            jobs_root().mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='.example-', dir=jobs_root()) as temporary:
                staged = Path(temporary) / identifier
                shutil.copytree(source, staged)
                verify_job(staged)
                staged.rename(target)
        manager.register_example(identifier, record)
    print(f'Example job available: {identifier}', flush=True)
    return identifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=PROJECT / 'bundled-example')
    parser.add_argument('--export-job', type=Path)
    args = parser.parse_args()
    if args.export_job:
        export_job(args.export_job, args.bundle)
    else:
        install_example(args.bundle)
