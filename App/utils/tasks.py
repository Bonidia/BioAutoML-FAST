import time
import streamlit as st
from enum import Enum
from redis import Redis
from rq import Queue
from utils.db import TaskResultManager, TaskStatus
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr

redis_conn = Redis(host="localhost", port=6379)
q = Queue(connection=redis_conn)

manager = TaskResultManager("task_results.db")

class JobStatus(Enum):
    PENDING = "pending"
    FINISHED = "finished"
    FAILED = "failed"
    INVALID = "invalid"

def send_job_finished_email(email, job_id, status):
    """
    Send a job completion email using Gmail SMTP.
    """

    sender = st.secrets["email_sender"]
    password = st.secrets["email_pass"]

    # Email content
    subject = f"[BioAutoML-FAST] Job ID {job_id} has finished"

    if status:
        body = (
            f"Dear user,\n\n"
            f"Your job has completed successfully.\n"
            f"Job ID: {job_id}\n\n"
            f"Consult the output in the Jobs module using your Job ID and, if encrypted, password."
        )
    else:
        body = (
            f"Dear user,\n\n"
            f"Your job has failed.\n\n"
            f"Try submitting again later."
        )

    # Create a MIMEText object with the body of the email.
    msg = MIMEText(body)
    # Set the subject of the email.
    msg["Subject"] = subject
    # Set the sender's email.
    msg["From"] = formataddr(("BioAutoML", sender))
    # Join the list of recipients into a single string separated by commas.
    msg["To"] = email

    # Connect to Gmail's SMTP server using SSL.
    with smtplib.SMTP("smtp.gmail.com", 587) as smtp_server:
        # Login to the SMTP server using the sender's credentials.
        smtp_server.starttls()  # Secure the connection
        smtp_server.login(sender, password)
        # Send the email. The sendmail function requires the sender's email, the list of recipients, and the email message as a string.
        smtp_server.sendmail(sender, email, msg.as_string())
    # Print a message to console after successfully sending the email.
    print(f"Email sent to {email}.")

def _on_success(job, connection, result, *args, **kwargs):
    """
    RQ success callback signature: (job, connection, result)
    You can access the job kwargs with `job.kwargs` (a dict).
    """

    manager.store_result(job.id, TaskStatus.SUCCESS)

    # Extract email if available and notify
    email = None
    try:
        email = job.kwargs.get("email")
    except Exception:
        email = None

    if email:
        send_job_finished_email(email, job.id, True)

def _on_failure(job, connection, *args, **kwargs):
    """
    RQ failure callback signature can be (job, *args...) — safer to ignore 'result' param name.
    """
    manager.store_result(job.id, TaskStatus.FAILURE)

    # Extract email if available and notify
    email = None
    try:
        email = job.kwargs.get("email")
    except Exception:
        email = None

    if email:
        send_job_finished_email(email, job.id, False)

def enqueue_task(fn, fn_kwargs=None):
    """
    Enqueue fn with fn_kwargs (dict). Always attach callbacks and
    always store a pending task entry in the DB.
    """
    if fn_kwargs is None:
        fn_kwargs = {}

    enqueue_kwargs = {"on_success": _on_success, "on_failure": _on_failure}

    # create the job
    job = q.enqueue(fn, kwargs=fn_kwargs, **enqueue_kwargs, job_timeout=3600)
    id_ = job.id

    # store pending task (store start_time)
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