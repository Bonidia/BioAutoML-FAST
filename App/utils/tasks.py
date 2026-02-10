import time
import streamlit as st
from enum import Enum
from redis import Redis
from rq import Queue
from .db import TaskResultManager, TaskStatus
import requests

redis_conn = Redis(host="localhost", port=6379)
q = Queue("bioautoml", connection=redis_conn)

manager = TaskResultManager("task_results.db")

class JobStatus(Enum):
    PENDING = "pending"
    FINISHED = "finished"
    FAILED = "failed"
    INVALID = "invalid"

def send_job_email(email, job_id, status):
    """
    Send a job completion email using Gmail SMTP.
    """

    api_key = st.secrets["api_key"]

    # Email content
    if status == "started":
        subject = f"[BioAutoML-FAST] Job submitted to queue"
        body = f"""Dear user,\n\nYour job was submitted to the queue.\nJob ID: {job_id}\n\nYou will receive an email when your job is finished."""
    elif status == "success":
        subject = f"[BioAutoML-FAST] Job has finished"
        body = f"""Dear user,\n\nYour job has completed successfully.\nJob ID: {job_id}\n\nConsult the output in the Jobs module using your Job ID and, if encrypted, password."""
    elif status == "failed":
        subject = f"[BioAutoML-FAST] Job has finished"
        body = f"""Dear user,\n\nYour job has failed.\n\nTry submitting again later."""

    response =  requests.post(
                "https://api.mailgun.net/v3/bioauto.inteligentehub.com.br/messages",
                auth=("api", api_key),
                data={"from": "BioAutoML-FAST <submission@bioauto.inteligentehub.com.br>",
                    "to": email,
                    "subject": subject,
                    "text": body})
    
    if response.status_code == 200:
        # Print a message to console after successfully sending the email.
        print(f"Email sent to {email}.")
    else:
        print(f"Email failed to be sent to {email}.")

def _on_success(job, connection, result, *args, **kwargs):
    """
    RQ success callback signature: (job, connection, result)
    You can access the job kwargs with `job.kwargs` (a dict).
    """

    manager.store_result(job.id, TaskStatus.SUCCESS)

    try:
        email = job.kwargs.get("email")
    except Exception:
        email = None

    if email:
        send_job_email(email, job.id, "success")

def _on_failure(job, connection, *args, **kwargs):
    """
    RQ failure callback signature can be (job, *args...) — safer to ignore 'result' param name.
    """
    manager.store_result(job.id, TaskStatus.FAILURE)

    try:
        email = job.kwargs.get("email")
    except Exception:
        email = None

    if email:
        send_job_email(email, job.id, "failed")

def enqueue_task(fn, fn_kwargs=None):
    """
    Enqueue fn with fn_kwargs (dict). Always attach callbacks and
    always store a pending task entry in the DB.
    """
    if fn_kwargs is None:
        fn_kwargs = {}

    enqueue_kwargs = {"on_success": _on_success, "on_failure": _on_failure}

    # create the job
    job = q.enqueue(fn, kwargs=fn_kwargs, **enqueue_kwargs, job_timeout=7200)

    try:
        email = job.kwargs.get("email")
    except Exception:
        email = None

    if email:
        send_job_email(email, job.id, "started")
    
    id_ = job.id
    manager.store_pending_task(id_)

    return id_

def check_job_status(job_id):
    job = q.fetch_job(job_id)
    if job is None:
        return JobStatus.INVALID, None
    
    if job.is_finished:
        return JobStatus.FINISHED, job.result
    elif job.is_failed:
        return JobStatus.FAILED, None
    else:
        return JobStatus.PENDING, None