"""Durable trial histories for the existing Optuna searches, without resuming them."""
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time

import optuna

from bioautoml.execution import atomic_json


def finite_value(value):
    return value if value is None or math.isfinite(value) else None


class TrialHistory:
    def __init__(self, output, stage, settings):
        self.directory = Path(output) / 'reports' / 'optimization' if output is not None else None
        self.stage = stage
        self.settings = settings
        self.started = time.perf_counter()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.completed_elapsed = {}
        self.lock = threading.Lock()
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, study, trial):
        with self.lock:
            self.completed_elapsed[trial.number] = time.perf_counter() - self.started
            self.save(study, 'running', None)

    def save(self, study, status, stop_reason):
        if self.directory is None:
            return
        trials = study.get_trials(deepcopy=True) if study is not None else []
        rows = []
        counts = {state.name: 0 for state in optuna.trial.TrialState}
        for trial in trials:
            counts[trial.state.name] += 1
            rows.append(dict(
                number=trial.number, state=trial.state.name, value=finite_value(trial.value),
                started_at=trial.datetime_start.astimezone(timezone.utc).isoformat() if trial.datetime_start else None,
                ended_at=trial.datetime_complete.astimezone(timezone.utc).isoformat() if trial.datetime_complete else None,
                duration_seconds=trial.duration.total_seconds() if trial.duration else None,
                completed_elapsed_seconds=self.completed_elapsed.get(trial.number),
                params=trial.params,
                intermediate_values={str(k): finite_value(v) for k, v in trial.intermediate_values.items()},
                user_attrs=trial.user_attrs,
            ))
        complete = [trial for trial in trials if trial.state == optuna.trial.TrialState.COMPLETE]
        best = study.best_trial if complete else None
        record = dict(
            schema_version=1, stage=self.stage, status=status, stop_reason=stop_reason,
            started_at=self.started_at, elapsed_seconds=time.perf_counter() - self.started,
            optuna_version=optuna.__version__, settings=self.settings,
            direction=study.direction.name if study is not None else self.settings['direction'],
            sampler=type(study.sampler).__name__ if study is not None else None,
            pruner=type(study.pruner).__name__ if study is not None else None,
            trial_counts=counts, best_trial=best.number if best else None,
            best_value=finite_value(best.value) if best else None,
            best_params=best.params if best else None, trials=rows,
            note='Only COMPLETE trials are eligible for selection. Intermediate values are running fold means. '
                 'Histories are evidence, not resumable studies. Interrupted files may contain RUNNING trials.',
        )
        atomic_json(self.directory / f'{self.stage}_study.json', record)
        fields = ['number', 'state', 'value', 'started_at', 'ended_at', 'duration_seconds',
                  'completed_elapsed_seconds', 'params', 'intermediate_values', 'user_attrs']
        with tempfile.NamedTemporaryFile(mode='w', newline='', dir=self.directory, delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, allow_nan=False) if isinstance(value, dict) else value
                                 for key, value in row.items()})
        try:
            os.replace(temporary, self.directory / f'{self.stage}_trials.csv')
        finally:
            temporary.unlink(missing_ok=True)


def optimize_study(study, objective, output, stage, n_trials, timeout, n_jobs, settings, early_stopping=None):
    settings = dict(settings, requested_trials=n_trials, timeout_seconds=timeout, search_jobs=n_jobs,
                    failure_policy='abort on unexpected exception',
                    budget_counts='COMPLETE, PRUNED and FAIL trials count toward the requested trial cap')
    history = TrialHistory(output, stage, settings)
    history.save(study, 'running', None)

    def recorded_objective(trial):
        try:
            return objective(trial)
        except optuna.TrialPruned as error:
            trial.set_user_attr('pruning_reason', str(error) or 'pruner decision')
            raise
        except Exception as error:
            trial.set_user_attr('error_type', type(error).__name__)
            trial.set_user_attr('error_message', str(error))
            raise

    callbacks = ([early_stopping] if early_stopping is not None else []) + [history]
    status, reason = 'completed', None
    try:
        study.optimize(recorded_objective, n_trials=n_trials, timeout=timeout,
                       n_jobs=n_jobs, show_progress_bar=True, callbacks=callbacks)
        if early_stopping is not None and early_stopping.stopped:
            reason = 'patience'
        elif len(study.trials) >= n_trials:
            reason = 'trial_budget'
        elif timeout is not None and time.perf_counter() - history.started >= timeout:
            reason = 'timeout'
        else:
            reason = 'study_stopped'
        if not any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials):
            raise ValueError('Optimization produced no completed trials; no model can be selected.')
    except BaseException as error:
        status = 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed'
        reason = reason or type(error).__name__
        raise
    finally:
        history.save(study, status, reason)


def save_skipped_history(output, stage, settings):
    history = TrialHistory(output, stage, dict(settings, requested_trials=0))
    history.save(None, 'skipped', 'tuning_disabled')
