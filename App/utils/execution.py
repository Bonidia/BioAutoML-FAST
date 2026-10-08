"""Trusted queue boundary for run recording and web-only model signing."""
from functools import wraps
import inspect
import os
import time
from pathlib import Path

from rq import get_current_job

from bioautoml.execution import Execution, run_path, atomic_json, utc_now
from bioautoml.model_artifacts import sign_web_model, ModelArtifact
from bioautoml.model_security import require_web_signer
from bioautoml.performance import write_performance


def web_execution(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        arguments = inspect.signature(function).bind(*args, **kwargs)
        arguments.apply_defaults()
        values = arguments.arguments
        job = get_current_job()
        training = function.__module__.endswith('.home') and values.get('training') == 'Training set'
        expected_queue = 'bioautoml-training' if training else 'bioautoml'
        if job is None or job.origin != expected_queue:
            raise ValueError('Unexpected execution queue for this operation.')
        if training:
            require_web_signer()
        for field in ('train_files', 'test_files'):
            files = values.get(field)
            if isinstance(files, list) and len({Path(file.name).stem for file in files}) != len(files):
                raise ValueError('Uploaded files within a split must have distinct filenames/stems. Sequence names may repeat.')
            for file in files if isinstance(files, list) else [files] if files else []:
                if not file.name or Path(file.name).name != file.name or '\\' in file.name:
                    raise ValueError('Uploaded filenames must not contain path components.')
        root = Path(values['predict_path']) / job.get_id()
        settings = {key: value for key, value in values.items()
                    if key not in ('train_files', 'test_files', 'password', 'email', 'predict_path')}
        execution = Execution(root, 'training' if training else 'prediction', settings, 'web', job.enqueued_at)
        previous = os.environ.get('BIOAUTOML_WEB_EXECUTION')
        os.environ['BIOAUTOML_WEB_EXECUTION'] = '1'
        try:
            result = function(*args, **kwargs)
            execution.add_inputs(sorted((root / 'inputs').rglob('*')))
            model_path = run_path(root, 'trained_model.sav')
            if training:
                sign_web_model(model_path)
            model = ModelArtifact(model_path)
            execution.record.update(model_id=model.model_id, signer=model.signature['key_id'])
            # The job archive includes the completed timing record. Its separate
            # packaging time is retained in the database/outer queue timestamps.
            execution.finish(defer_resources=True)
            if values.get('password'):
                function.__globals__['encrypt_job_folder'](str(root), values['password'])
            execution.memory.stop()
            execution.performance.update(total_seconds=time.perf_counter() - execution.started,
                                         total_scope='end_to_end',
                                         **execution.memory.metadata())
            write_performance(root / 'performance_summary.csv', execution.performance)
            if (root / 'run.json').exists():
                execution.record.update(elapsed_seconds=execution.performance['total_seconds'], ended_at=utc_now())
                atomic_json(root / 'run.json', execution.record)
                write_performance(root / 'reports/performance_summary.csv', execution.performance)
            atomic_json(root / 'execution_summary.json', dict(run_id=execution.record['run_id'],
                        performance=execution.performance,
                        status='completed', ended_at=utc_now(), queue_seconds=execution.record['queue_seconds'],
                        execution_seconds=execution.record['elapsed_seconds'],
                        total_with_packaging_seconds=time.perf_counter() - execution.started))
            return result
        except BaseException:
            execution.finish('failed')
            execution.memory.stop()
            if hasattr(execution, 'performance'):
                execution.performance.update(status='failed', total_seconds=time.perf_counter() - execution.started,
                                             total_scope='end_to_end',
                                             **execution.memory.metadata())
                write_performance(root / 'performance_summary.csv', execution.performance)
            if (root / 'run.json').is_file():
                execution.record.update(status='failed', ended_at=utc_now())
                atomic_json(root / 'run.json', execution.record)
            atomic_json(root / 'execution_summary.json', dict(run_id=execution.record['run_id'], status='failed',
                        performance=getattr(execution, 'performance', {}),
                        ended_at=utc_now(), total_with_packaging_seconds=time.perf_counter() - execution.started))
            raise
        finally:
            if execution.owner:
                execution.memory.stop()
            if previous is None:
                os.environ.pop('BIOAUTOML_WEB_EXECUTION', None)
            else:
                os.environ['BIOAUTOML_WEB_EXECUTION'] = previous
    return wrapped
