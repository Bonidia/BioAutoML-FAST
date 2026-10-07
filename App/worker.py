"""Start the BioAutoML queue worker from a stable application directory."""

import os
from pathlib import Path
import sys


def main():
    app_path = Path(__file__).resolve().parent
    os.chdir(app_path)
    sys.path.insert(0, str(app_path))

    from rq import Worker
    from rq.utils import import_attribute
    from utils.tasks import q, redis_conn

    for name in ("modules.home.submit_job", "modules.repo.submit_job"):
        if not callable(import_attribute(name)):
            raise TypeError(f"Queued function is not callable: {name}")

    redis_conn.ping()
    worker = Worker([q], connection=redis_conn, name=os.environ.get("RQ_WORKER_NAME"))
    worker.work()


if __name__ == "__main__":
    main()
