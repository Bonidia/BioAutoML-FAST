"""Bounded, job-scoped cleanup. Never operate on datasets or benchmark outputs."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid

from redis import Redis
from rq.exceptions import NoSuchJobError
from rq.job import Job

from .db import TaskResultManager


RETENTION_DAYS = 30
TERMINAL = {'success', 'failure', 'expired', 'canceled', 'stopped'}
logger = logging.getLogger(__name__)


def jobs_root():
    app = Path(__file__).resolve().parents[1]
    root = Path(os.environ.get('BIOAUTOML_JOBS_DIR', app / 'jobs')).resolve()
    if root in {Path('/'), Path.home(), app, *app.parents} or root.is_relative_to(app / 'datasets') or root.is_relative_to(app.parent / 'manuscript'):
        raise ValueError('Jobs must use a dedicated directory outside datasets and manuscript outputs.')
    return root


def job_directory(job_id):
    identifier = str(uuid.UUID(job_id))
    if identifier != job_id:
        raise ValueError('Invalid job identifier.')
    path = jobs_root() / identifier
    if path.is_symlink() or path.resolve().parent != jobs_root():
        raise ValueError('Job path is not a direct, non-symlink child of the jobs directory.')
    return path


@contextmanager
def job_lock(job_id, blocking=True):
    job_directory(job_id)
    state = Path(os.environ.get('TASK_RESULTS_DB', 'task_results.db')).resolve().parent / 'job-locks'
    state.mkdir(parents=True, exist_ok=True)
    with open(state / (job_id + '.lock'), 'a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def is_expired(record, now=None):
    if not record or record.get('status') not in TERMINAL:
        return False
    if record['status'] == 'expired':
        return True
    try:
        ended = datetime.fromisoformat(record['end_time'])
        if ended.tzinfo is None:
            ended = ended.replace(tzinfo=timezone.utc)
    except (KeyError, TypeError, ValueError):
        return False  # Unknown age is not permission to delete.
    return (now or datetime.now(timezone.utc)) >= ended + timedelta(days=RETENTION_DAYS)


def remove_job_files(job_id):
    path = job_directory(job_id)
    if path.exists():
        shutil.rmtree(path)


def temporary_root():
    path = Path(tempfile.gettempdir()) / 'bioautoml-private-jobs'
    if path.is_symlink():
        raise ValueError('Private temporary directory must not be a symlink.')
    path.mkdir(mode=0o700, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def create_job_temp(job_id):
    job_directory(job_id)
    return tempfile.mkdtemp(prefix=job_id + '_', dir=temporary_root())


def remove_job_temp(path):
    path = Path(path)
    if path.is_symlink() or path.resolve().parent != temporary_root().resolve():
        raise ValueError('Unrecognized private temporary directory.')
    if path.exists():
        shutil.rmtree(path)


def cleanup_temporary(manager, now=None):
    now = now or datetime.now(timezone.utc)
    for path in temporary_root().iterdir():
        try:
            identifier = path.name.split('_', 1)[0]
            with job_lock(identifier, blocking=False) as acquired:
                if not acquired or not path.is_dir() or path.is_symlink():
                    continue
                record = manager.get_result(identifier)
                if not record or is_expired(record, now) or now.timestamp() - path.stat().st_mtime > 3600:
                    remove_job_temp(path)
        except (ValueError, OSError):
            logger.warning('Skipped an invalid or unavailable temporary job directory.')


def cleanup_jobs(manager, connection, now=None, dry_run=False):
    now = now or datetime.now(timezone.utc)
    rows = manager.connection.execute('SELECT id FROM task_results').fetchall()
    removed = []
    for (identifier,) in rows:
        try:
            with job_lock(identifier, blocking=False) as acquired:
                if not acquired:
                    continue
                record = manager.get_result(identifier)
                try:
                    job = Job.fetch(identifier, connection=connection)
                except NoSuchJobError:
                    job = None
                status = job.get_status(refresh=True) if job else None
                # Reconcile terminal RQ states after crashes/cancellations where
                # callbacks did not execute. Never guess that a live job is dead.
                if record and record['status'] in {'pending', 'running'} and status in {'failed', 'canceled', 'stopped'}:
                    if not dry_run:
                        if job.kwargs.get('_encrypted'):
                            remove_job_files(identifier)
                        from .privacy import discard_credentials
                        discard_credentials(job.kwargs.get('_private_ref'))
                        manager.connection.execute('UPDATE task_results SET status = ?, end_time = ? WHERE id = ?',
                                                   ('failure', now.isoformat(), identifier))
                        manager.connection.commit()
                    continue
                if not is_expired(record, now):
                    continue
                if job and status not in {'finished', 'failed', 'canceled', 'stopped'}:
                    continue
                if dry_run:
                    removed.append(identifier)
                    continue
                # Hide first, so a crash during deletion cannot expose partial results.
                manager.connection.execute('UPDATE task_results SET status = ? WHERE id = ?', ('expired', identifier))
                manager.connection.commit()
                remove_job_files(identifier)
                for path in temporary_root().glob(identifier + '_*'):
                    remove_job_temp(path)
                if job:
                    from .privacy import discard_credentials
                    discard_credentials(job.kwargs.get('_private_ref'))
                    job.delete()
                manager.connection.execute('DELETE FROM task_results WHERE id = ?', (identifier,))
                manager.connection.commit()
                removed.append(identifier)
        except (ValueError, OSError):
            logger.warning('Skipped an invalid or unavailable job directory.')
    if not dry_run:
        cleanup_temporary(manager, now)
    return removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loop', action='store_true', help='Clean at startup and every hour.')
    parser.add_argument('--dry-run', action='store_true', help='Report eligible job IDs without deleting.')
    args = parser.parse_args()
    if args.loop and args.dry_run:
        parser.error('--loop and --dry-run cannot be combined')
    logging.basicConfig(level=logging.INFO)
    manager = TaskResultManager(os.environ.get('TASK_RESULTS_DB', 'task_results.db'))
    connection = Redis.from_url(os.environ.get('REDIS_URL', 'redis://127.0.0.1:6379/0'),
                                socket_connect_timeout=5, socket_timeout=5)
    while True:
        try:
            removed = cleanup_jobs(manager, connection, dry_run=args.dry_run)
            logger.info('Retention cleanup: %s jobs %s.', len(removed), 'eligible' if args.dry_run else 'removed')
            if args.dry_run:
                for identifier in removed:
                    print(identifier)
        except Exception:
            # Do not include queue payloads or credential-bearing exception strings.
            logger.error('Retention cleanup unavailable; will retry without deleting unverified jobs.')
            if not args.loop:
                raise SystemExit(1)
        if not args.loop:
            break
        time.sleep(3600)


if __name__ == '__main__':
    main()
