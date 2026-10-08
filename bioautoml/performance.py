"""Wall-time summaries and lightweight, job-local process-tree RSS sampling."""
import csv
from datetime import datetime
import os
from pathlib import Path
import sys
import threading
import time

import psutil


class MemoryMonitor:
    def __init__(self, interval=0.1):
        self.process = psutil.Process()
        self.interval = interval
        self.peak = 0
        self.samples = 0
        self.incomplete = False
        self.procfs = sys.platform.startswith('linux') and Path('/proc/self/task').is_dir()
        self.page_size = os.sysconf('SC_PAGE_SIZE') if self.procfs else None
        self.stopped = threading.Event()
        self.sample()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def sample(self):
        total = 0
        if self.procfs:
            # psutil.children(recursive=True) scans every process on the host.
            # Linux exposes direct child PIDs per thread: visit only this job's
            # subtree, including children launched by non-main Python threads.
            pending, seen = [self.process.pid], set()
            while pending:
                pid = pending.pop()
                if pid in seen:
                    continue
                seen.add(pid)
                try:
                    process = Path('/proc') / str(pid)
                    total += int((process / 'statm').read_text().split()[1]) * self.page_size
                    for thread in (process / 'task').iterdir():
                        try:
                            pending.extend(map(int, (thread / 'children').read_text().split()))
                        except FileNotFoundError:
                            pass
                except (FileNotFoundError, ProcessLookupError):
                    pass
                except (OSError, ValueError, IndexError):
                    self.incomplete = True
            self.peak = max(self.peak, total)
            self.samples += 1
            return
        try:
            processes = [self.process, *self.process.children(recursive=True)]
        except psutil.Error:
            self.incomplete = True
            return
        for process in processes:
            try:
                total += process.memory_info().rss
            except psutil.NoSuchProcess:
                pass
            except psutil.Error:
                self.incomplete = True
        self.peak = max(self.peak, total)
        self.samples += 1

    def _run(self):
        while not self.stopped.wait(self.interval):
            self.sample()

    def stop(self):
        if self.stopped.is_set():
            return
        self.stopped.set()
        self.thread.join()
        self.sample()

    def metadata(self):
        return dict(peak_memory_bytes=self.peak, memory_method='sampled_process_tree_rss',
                    memory_sampling_seconds=self.interval, memory_samples=self.samples,
                    memory_incomplete=self.incomplete)


def phase_seconds(events, names, exclude=()):
    """Union intervals instead of summing nested or parallel elapsed times."""
    intervals = []
    for event in events:
        if event['phase'] in names and event['status'] != 'skipped':
            start = datetime.fromisoformat(event['started_at']).timestamp()
            intervals.append((start, start + event['elapsed_seconds']))
    if not intervals:
        return None
    for event in events:
        if event['phase'] not in exclude or event['status'] == 'skipped':
            continue
        cut_start = datetime.fromisoformat(event['started_at']).timestamp()
        cut_end = cut_start + event['elapsed_seconds']
        remaining = []
        for start, end in intervals:
            if end <= cut_start or start >= cut_end:
                remaining.append((start, end))
            else:
                if start < cut_start:
                    remaining.append((start, cut_start))
                if end > cut_end:
                    remaining.append((cut_end, end))
        intervals = remaining
    if not intervals:
        return 0.0
    intervals.sort()
    start, end = intervals[0]
    total = 0.0
    for next_start, next_end in intervals[1:]:
        if next_start > end:
            total += end - start
            start = next_start
        end = max(end, next_end)
    return total + end - start


def performance_summary(events, elapsed, memory, status):
    return dict(
        status=status,
        descriptor_extraction_seconds=phase_seconds(events, {'training_features', 'test_features'}, {'training_preprocessing', 'test_preprocessing'}),
        training_descriptor_seconds=phase_seconds(events, {'training_features'}, {'training_preprocessing'}),
        test_descriptor_seconds=phase_seconds(events, {'test_features'}, {'test_preprocessing'}),
        optimisation_seconds=phase_seconds(events, {'stage_1', 'stage_2'}),
        stage_1_seconds=phase_seconds(events, {'stage_1'}),
        stage_2_seconds=phase_seconds(events, {'stage_2'}),
        total_seconds=elapsed, **memory.metadata())


def write_performance(path, summary):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
