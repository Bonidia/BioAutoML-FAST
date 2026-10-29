import time
import os
from enum import Enum
from redis import Redis
from rq import Queue
from .db import TaskResultManager, TaskStatus
import requests
import uuid
import csv
from .privacy import store_credentials, take_email, discard_credentials

redis_conn = Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
q = Queue("bioautoml", connection=redis_conn)
training_q = Queue("bioautoml-training", connection=redis_conn)

manager = TaskResultManager(os.environ.get("TASK_RESULTS_DB", "task_results.db"))

class JobStatus(Enum):
    PENDING = "pending"
    FINISHED = "finished"
    FAILED = "failed"
    INVALID = "invalid"

def send_job_email(email, job_id, status):
    """
    Send a notification without logging recipient addresses or provider responses.
    """

    api_key = os.environ.get("MAILGUN_API_KEY", "").strip()
    if not api_key:
        print('Job notification unavailable: email service is not configured.')
        return

    # Email content
    if status == "started":
        subject = f"[BioAutoML-FAST] Job submitted to queue"
        body = f"""Dear user,\n\nYour job was submitted to the queue.\nJob link: https://bioautoml.icmc.usp.br/?id={job_id}\n\nYou will receive an email when your job is finished."""
    elif status == "success":
        subject = f"[BioAutoML-FAST] Job has finished"
        body = f"""Dear user,\n\nYour job has completed successfully.\nJob link: https://bioautoml.icmc.usp.br/?id={job_id}\n\nConsult the output in the Jobs module using your Job ID and, if encrypted, password."""
    elif status == "failed":
        subject = f"[BioAutoML-FAST] Job has finished"
        body = f"""Dear user,\n\nYour job has failed.\n\nTry submitting again later."""

    try:
        response = requests.post(
                "https://api.mailgun.net/v3/bioauto.inteligentehub.com.br/messages",
                auth=("api", api_key),
                data={"from": "BioAutoML-FAST <submission@bioauto.inteligentehub.com.br>",
                    "to": email,
                    "subject": subject,
                    "text": body}, timeout=15)
        print('Job notification sent.' if response.status_code == 200 else 'Job notification delivery failed.')
    except requests.RequestException:
        print('Job notification delivery failed.')


def _notify_finished(job, status):
    try:
        email = take_email(job.kwargs.get('_private_ref'))
        if email:
            send_job_email(email, job.id, status)
    except Exception:
        print('Job notification unavailable; no recipient details were logged.')

def _on_success(job, connection, result, *args, **kwargs):
    """
    RQ success callback signature: (job, connection, result)
    You can access the job kwargs with `job.kwargs` (a dict).
    """

    manager.store_result(job.id, TaskStatus.SUCCESS)

    _notify_finished(job, 'success')

def _on_failure(job, connection, *args, **kwargs):
    """
    RQ failure callback signature can be (job, *args...) — safer to ignore 'result' param name.
    """
    manager.store_result(job.id, TaskStatus.FAILURE)
    from .retention import job_lock, remove_job_files

    # Also covers work-horse termination/timeouts, where the child finally block
    # may not run. Never leave partial plaintext for an encrypted submission.
    if job.kwargs.get('_encrypted'):
        with job_lock(job.id):
            remove_job_files(job.id)
    _notify_finished(job, 'failed')

def enqueue_task(fn, fn_kwargs=None, job_info=None):
    """
    Enqueue fn with fn_kwargs (dict). Always attach callbacks and
    always store a pending task entry in the DB.
    """
    fn_kwargs = dict(fn_kwargs or {})
    from .retention import job_directory, remove_job_files
    email = fn_kwargs.pop('email', None)
    password = fn_kwargs.pop('password', None)

    enqueue_kwargs = {"on_success": _on_success, "on_failure": _on_failure, "on_stopped": _on_failure}

    # create the job
    training = fn.__module__.endswith('.home') and fn_kwargs.get('training') == 'Training set'
    if training and os.environ.get('BIOAUTOML_TRAINING_ENABLED') != '1':
        raise ValueError('Web training requires the dedicated signing worker. Configure it before submitting training jobs.')
    queue = training_q if training else q
    identifier = str(uuid.uuid4())
    root = job_directory(identifier)
    reference = store_credentials(email, password)
    fn_kwargs.update(_private_ref=reference, _encrypted=bool(password))
    password = None
    manager.store_pending_task(identifier)
    try:
        # Create reports before enqueueing: otherwise a fast worker can encrypt
        # or delete its outputs while the UI is still writing job_info.tsv.
        if job_info:
            reports = root / 'reports'
            reports.mkdir(parents=True, exist_ok=True)
            with open(reports / 'job_info.tsv', 'w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(job_info), delimiter='\t')
                writer.writeheader()
                writer.writerow(job_info)
        queue.enqueue(fn, kwargs=fn_kwargs, **enqueue_kwargs, job_timeout=7200,
                      job_id=identifier, description='BioAutoML job ' + identifier,
                      result_ttl=86400, failure_ttl=30 * 86400)
    except Exception:
        discard_credentials(reference)
        remove_job_files(identifier)
        manager.store_result(identifier, TaskStatus.FAILURE)
        raise RuntimeError('Could not queue the submission; please try again.') from None
    if email:
        try:
            send_job_email(email, identifier, 'started')
        except Exception:
            print('Job notification unavailable; no recipient details were logged.')
    return identifier

def check_job_status(job_id):
    job = q.fetch_job(job_id) or training_q.fetch_job(job_id)
    if job is None:
        return JobStatus.INVALID, None
    
    if job.is_finished:
        return JobStatus.FINISHED, job.result
    elif job.is_failed:
        return JobStatus.FAILED, None
    else:
        return JobStatus.PENDING, None
