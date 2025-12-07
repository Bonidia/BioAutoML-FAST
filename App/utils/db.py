import json
import sqlite3
import pandas as pd
from enum import Enum
from pathlib import Path
import threading
from datetime import datetime, timezone

class TaskStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"
    RUNNING = "running"

class TaskResultManager:
    def __init__(self, path_to_db):
        self._path_to_db = path_to_db
        self._local = threading.local()
        self._create_db()

    @property
    def connection(self):
        if not hasattr(self._local, "connection"):
            self._local.connection = sqlite3.connect(self._path_to_db)
        return self._local.connection

    def __del__(self):
        if hasattr(self._local, "connection"):
            self._local.connection.close()

    def _create_db(self):
        Path(self._path_to_db).parent.mkdir(parents=True, exist_ok=True)
        cursor = self.connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS task_results (
                id TEXT PRIMARY KEY,
                status TEXT,
                start_time TEXT,
                end_time TEXT
            )
            """
        )
        self.connection.commit()

    def store_pending_task(self, task_id):
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO task_results (id, status, start_time, end_time)
            VALUES (?, ?, ?, NULL)
            """,
            (task_id, TaskStatus.PENDING.value, datetime.now(timezone.utc))
        )
        self.connection.commit()

    def store_result(self, task_id, status):
        cursor = self.connection.cursor()
        cursor.execute(
            """
            UPDATE task_results
            SET status = ?, end_time = ?
            WHERE id = ?
            """,
            (status.value, datetime.now(timezone.utc), task_id)
        )
        self.connection.commit()

    def get_result(self, task_id):
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT status, start_time, end_time 
            FROM task_results WHERE id = ?
            """,
            (task_id,),
        )
        result = cursor.fetchone()
        if result:
            return {
                "status": result[0],
                "start_time": result[1],
                "end_time": result[2]
            }
        return None
    
    def get_job_position(self, task_id):
        """
        Return the 1-based queue position of task_id among jobs with status
        PENDING or RUNNING ordered by start_time ASC then id ASC.
        Returns a dict:
            {"task_id": task_id, "position": <int or None>, "total": <int>}
        If the task is not pending/running (or not found), position is None.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT id
            FROM task_results
            WHERE status IN (?, ?)
            ORDER BY start_time ASC, id ASC
            """,
            (TaskStatus.PENDING.value, TaskStatus.RUNNING.value)
        )
        rows = cursor.fetchall()
        ids = [r[0] for r in rows]
        total = len(ids)
        try:
            pos = ids.index(task_id) + 1
        except ValueError:
            pos = None
            
        return pos

    def get_pending_jobs(self):
        """
        Returns pending jobs as a DataFrame with columns:
        ['Job queue position', 'task_id', 'start_time', 'status']
        Ordered by start_time ascending (earliest pending = position 1).
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT id, start_time, status
            FROM task_results
            WHERE status IN (?, ?)
            ORDER BY start_time ASC
            LIMIT 5
            """,
            (TaskStatus.PENDING.value, TaskStatus.RUNNING.value)
        )
        results = cursor.fetchall()
        df = pd.DataFrame(results, columns=["Job ID", "Start", "Status"])

        # Ensure consistent column types and compute queue position
        if df.empty:
            return pd.DataFrame(columns=["Queue position", "Job ID", "Start", "Status"])

        df.reset_index(drop=True, inplace=True)
        # position is 1-based index
        df.insert(0, "Queue position", df.index + 1)

        return df[["Queue position", "Job ID", "Start", "Status"]]

    def get_recent_completed_jobs(self):
        """
        Returns completed jobs as a DataFrame with columns:
        ['task_id', 'end_time', 'duration', 'status']
        Ordered by end_time DESC (most recent first).
        Only rows with a non-null end_time are considered completed.
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT id, start_time, end_time, status
            FROM task_results
            WHERE status = ?
            ORDER BY end_time DESC
            LIMIT 5
            """, 
            (TaskStatus.SUCCESS.value,)
        )
        results = cursor.fetchall()
        df = pd.DataFrame(results, columns=["Job ID", "Start", "End", "Status"])

        if df.empty:
            return pd.DataFrame(columns=["Job ID", "End", "Duration", "Status"])

        def compute_duration(row):
            if row["Start"] and row["End"]:
                try:
                    start = datetime.fromisoformat(row["Start"])
                    end = datetime.fromisoformat(row["End"])
                    delta = end - start
                    total_seconds = int(delta.total_seconds())
                    hours = total_seconds // 3600
                    minutes = (total_seconds % 3600) // 60
                    seconds = total_seconds % 60
                    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                except Exception:
                    return None
            return None

        df["Duration"] = df.apply(compute_duration, axis=1)
        # reorder columns as requested
        df = df[["Job ID", "End", "Duration", "Status"]]

        return df