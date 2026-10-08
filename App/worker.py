"""Start the BioAutoML queue worker from a stable application directory."""

import os
from pathlib import Path
import sys
from rq.job import Job


class TrainingJob(Job):
    def _execute(self):
        if self.func_name != 'modules.home.submit_job' or self.kwargs.get('training') != 'Training set':
            raise ValueError('The signing worker accepts only new web training jobs.')
        return super()._execute()


def main():
    app_path = Path(__file__).resolve().parent
    os.chdir(app_path)
    sys.path.insert(0, str(app_path))
    sys.path.insert(0, str(app_path.parent))

    from rq import Worker
    from rq.utils import import_attribute
    from utils.tasks import q, training_q, redis_conn
    if os.environ.get('BIOAUTOML_WORKER_ROLE') == 'training':
        from bioautoml.model_security import initialize_web_signer
        initialize_web_signer()
        queue = training_q
    else:
        if os.environ.get('BIOAUTOML_SIGNING_KEY'):
            raise ValueError('Never configure a private key on an inference worker.')
        queue = q

    for name in ("modules.home.submit_job", "modules.repo.submit_job"):
        if not callable(import_attribute(name)):
            raise TypeError(f"Queued function is not callable: {name}")

    redis_conn.ping()
    worker = Worker([queue], connection=redis_conn, name=os.environ.get("RQ_WORKER_NAME"),
                    job_class=TrainingJob if queue is training_q else Job)
    worker.work()


if __name__ == "__main__":
    main()
