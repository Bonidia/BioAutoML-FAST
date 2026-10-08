"""Shared, append-only phase timings and versioned CLI/web run layouts."""
import atexit
from contextlib import contextmanager
from contextvars import ContextVar
import csv
from datetime import datetime, timezone
from functools import wraps
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import uuid
from bioautoml.performance import MemoryMonitor, performance_summary, write_performance


_current = ContextVar('bioautoml_execution', default=None)
_parent = ContextVar('bioautoml_phase', default=None)


class _LogStream:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.log.write(value)
        self.log.flush()
        return self.stream.write(value)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


LAYOUT = {
    'trained_model.sav': 'model/trained_model.sav',
    'train': 'inputs/train', 'test': 'inputs/test',
    'feat_extraction': 'work/features', 'best_descriptors': 'results/descriptors',
    'homology': 'reports/homology',
    'subprocess.log': 'logs/pipeline.log', 'job_info.tsv': 'reports/job_info.tsv',
    'training_kfold(10)_metrics.csv': 'results/metrics/optimization_cv_metrics.csv',
    'training_kfold(10)_metrics_pearson_folds.csv': 'results/metrics/optimization_cv_pearson_folds.csv',
    'training_confusion_matrix.csv': 'results/metrics/optimization_cv_confusion_matrix.csv',
    'metrics_test.csv': 'results/metrics/test_metrics.csv',
    'metrics_other.csv': 'results/metrics/test_additional_metrics.csv',
    'metrics_test_pearson.csv': 'results/metrics/test_pearson.csv',
    'test_confusion_matrix.csv': 'results/metrics/test_confusion_matrix.csv',
    'test_predictions.csv': 'results/predictions/test_predictions.csv',
    'feature_importance.tsv': 'results/descriptors/feature_importance.tsv',
    'train_stats.csv': 'reports/train_stats.csv', 'test_stats.csv': 'reports/test_stats.csv',
}


def run_path(root, *parts):
    """Resolve logical output names into the single CLI/web run layout."""
    root = Path(root)
    relative = Path(*parts)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Run paths must be relative and contained in the run.')
    if relative.parts:
        first, *rest = relative.parts
        relative = Path(LAYOUT.get(first, first), *rest)
    return str(root / relative)


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, default=str, allow_nan=False)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def run_root(path):
    path = Path(path)
    return str(next((parent for parent in (path, *path.parents) if (parent / 'run.json').is_file()), path))


@contextmanager
def phase(name, status='completed'):
    execution = _current.get()
    if execution is None:
        yield
    else:
        with execution.phase(name, status):
            yield


class Execution:
    def __init__(self, root, operation, settings=None, origin='cli', queued_at=None):
        self.root = Path(root).resolve()
        self.owner = os.environ.get('BIOAUTOML_RUN_ROOT') != str(self.root)
        self.started = time.perf_counter()
        self.done = False
        self.streams = None
        self.root.mkdir(parents=True, exist_ok=True)
        if self.owner:
            # Never overwrite a completed/partial run or old scientific output.
            allowed = {'reports'} if origin == 'web' else set()
            if any(path.name not in allowed for path in self.root.iterdir()):
                raise FileExistsError('Choose a fresh output directory; existing runs are preserved.')
            packages = {}
            for name in ('scikit-learn', 'lightgbm', 'xgboost', 'optuna', 'numpy', 'joblib'):
                try:
                    packages[name] = version(name)
                except PackageNotFoundError:
                    pass
            self.record = dict(schema_version=1, run_id=str(uuid.uuid4()), operation=operation,
                               origin=origin, status='running', started_at=utc_now(), ended_at=None,
                               elapsed_seconds=None, settings=settings or {}, inputs=[],
                               python=platform.python_version(), packages=packages,
                               code_sha256={name: file_hash(Path(__file__).resolve().parent.parent / name)
                                            for name in ('engineering.py', 'generation.py',
                                                         *(f'bioautoml/{path.name}' for path in sorted(Path(__file__).parent.glob('*.py'))))},
                               image_identity=os.environ.get('BIOAUTOML_IMAGE_ID'),
                               queue_seconds=None)
            if queued_at:
                if queued_at.tzinfo is None:
                    queued_at = queued_at.replace(tzinfo=timezone.utc)
                self.record['queue_seconds'] = max(0, (datetime.now(timezone.utc) - queued_at).total_seconds())
            # Exclusive claim prevents two processes racing into the same output.
            with (self.root / 'run.json').open('x') as handle:
                json.dump(self.record, handle, indent=2, default=str, allow_nan=False)
            for directory in ('inputs', 'model', 'results/metrics', 'results/predictions',
                              'results/descriptors', 'results/figures', 'reports', 'logs', 'work'):
                (self.root / directory).mkdir(parents=True, exist_ok=True)
            self.previous_root = os.environ.get('BIOAUTOML_RUN_ROOT')
            os.environ['BIOAUTOML_RUN_ROOT'] = str(self.root)
            self.memory = MemoryMonitor()
            atexit.register(self.interrupted)
        self.token = _current.set(self)

    def add_inputs(self, paths):
        if not self.owner:
            return
        with self.phase('input_fingerprints'):
            for path in paths:
                if path and Path(path).is_file():
                    self.record['inputs'].append(dict(name=Path(path).name, bytes=Path(path).stat().st_size,
                                                      sha256=file_hash(path)))
        atomic_json(self.root / 'run.json', self.record)

    @contextmanager
    def phase(self, name, status='completed'):
        phase_id = str(uuid.uuid4())
        parent = _parent.get()
        token = _parent.set(phase_id)
        start = time.perf_counter()
        started_at = utc_now()
        try:
            yield
        except BaseException:
            status = 'failed'
            raise
        finally:
            event = dict(phase_id=phase_id, parent_phase_id=parent, phase=name, status=status,
                         started_at=started_at, ended_at=utc_now(),
                         elapsed_seconds=time.perf_counter() - start, pid=os.getpid())
            # Each process has its own journal: no cross-process CSV overwrite.
            with open(self.root / 'logs' / f'timings-{os.getpid()}.jsonl', 'a') as handle:
                handle.write(json.dumps(event) + '\n')
            if self.owner:
                atomic_json(self.root / 'logs' / 'memory.json', self.memory.metadata())
            _parent.reset(token)

    def finish(self, status='completed', defer_resources=False):
        if self.done:
            return
        self.done = True
        if self.owner:
            if not defer_resources:
                self.memory.stop()
            self.record.update(status=status, ended_at=utc_now(), elapsed_seconds=time.perf_counter() - self.started)
            # Metadata only; never deserialize an artifact to finish a run record.
            import zipfile
            model_path = Path(run_path(self.root, 'trained_model.sav'))
            if status == 'completed' and model_path.is_file() and zipfile.is_zipfile(model_path):
                try:
                    with zipfile.ZipFile(model_path) as archive:
                        if archive.getinfo('manifest.json').file_size <= 4 * 1024 * 1024:
                            self.record.setdefault('model_id', json.loads(archive.read('manifest.json')).get('model_id'))
                except (KeyError, ValueError, zipfile.BadZipFile):
                    pass
            self.record['outputs'] = sorted(str(path.relative_to(self.root)) for path in self.root.rglob('*')
                                            if path.is_file() and path.name != 'run.json')
            atomic_json(self.root / 'run.json', self.record)
            events = []
            for path in sorted((self.root / 'logs').glob('timings-*.jsonl')):
                for line in path.read_text().splitlines():
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # A killed writer can leave a partial last event.
            with open(self.root / 'timings.csv', 'w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['phase_id', 'parent_phase_id', 'phase', 'status',
                                                            'started_at', 'ended_at', 'elapsed_seconds', 'pid'])
                writer.writeheader()
                writer.writerows(events)
            self.performance = performance_summary(events, time.perf_counter() - self.started, self.memory, status)
            self.performance['total_scope'] = 'before_archive_packaging' if defer_resources else 'end_to_end'
            write_performance(self.root / 'reports' / 'performance_summary.csv', self.performance)
            self.record['performance'] = self.performance
            self.record['elapsed_seconds'] = self.performance['total_seconds']
            atomic_json(self.root / 'run.json', self.record)
            if self.previous_root is None:
                os.environ.pop('BIOAUTOML_RUN_ROOT', None)
            else:
                os.environ['BIOAUTOML_RUN_ROOT'] = self.previous_root
            print(f'{self.record["operation"]}: {status}; {self.record["elapsed_seconds"] / 60:.3f} minutes')
            for label, key in [('Descriptor extraction', 'descriptor_extraction_seconds'),
                               ('Optimisation', 'optimisation_seconds')]:
                value = self.performance[key]
                print(f'{label}: ' + ('not applicable' if value is None else f'{value / 60:.3f} minutes'))
            print(f'Sampled peak memory: {self.performance["peak_memory_bytes"] / 1024**3:.3f} GiB')
        if self.streams:
            sys.stdout, sys.stderr, log = self.streams
            log.close()
        _current.reset(self.token)

    def interrupted(self):
        if not self.done:
            self.finish('interrupted')


def timed(name):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            execution = _current.get()
            if execution is None:
                return function(*args, **kwargs)
            with execution.phase(name):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def start_cli(args, operation):
    execution = Execution(args.output, operation, vars(args))
    # Preserve resolved stage defaults even when engineering/web owns run.json.
    atomic_json(execution.root / 'logs' / f'settings-{os.getpid()}.json',
                dict(operation=operation, settings=vars(args), pid=os.getpid()))
    if os.environ.get('BIOAUTOML_WEB_EXECUTION') != '1':
        log = open(execution.root / 'logs' / f'cli-{os.getpid()}.log', 'a')
        execution.streams = sys.stdout, sys.stderr, log
        sys.stdout, sys.stderr = _LogStream(sys.stdout, log), _LogStream(sys.stderr, log)
    inputs = []
    for key in ('fasta_train', 'fasta_test', 'train', 'train_label', 'train_nameseq',
                'test', 'test_label', 'test_nameseq', 'path_model'):
        value = getattr(args, key, None)
        inputs.extend(value if isinstance(value, list) else [value] if value else [])
    execution.add_inputs(inputs)
    original_hook = sys.excepthook
    def failed(kind, value, traceback):
        original_hook(kind, value, traceback)
        execution.finish('failed')
    sys.excepthook = failed
    return execution
