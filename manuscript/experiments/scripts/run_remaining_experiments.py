"""Validate and run the remaining revision tasks sequentially in the background."""
import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import psutil

import _experiment_support as support
from _experiment_support import RESULTS_PATH, SCRIPTS_PATH, PROJECT_PATH, atomic_json, ablation

STAGES = [('sequence_overlap', 'audit_sequence_overlap.py'),
          ('pruning_comparison', 'run_pruning_comparison.py'),
          ('benchmark_repeats', 'run_benchmark_repeats.py')]
ROOT = RESULTS_PATH / 'remaining_experiments'
LOG = RESULTS_PATH / 'remaining_experiments.log'


def run_stage(name, script, root, state, dry_run=False):
    command = [sys.executable, str(SCRIPTS_PATH / script)]
    if dry_run:
        command.append('--dry_run')
    elif (RESULTS_PATH / name / 'protocol.json').exists():
        command.append('--resume')
    log_path = root / 'logs' / (name + ('_preflight' if dry_run else '') + '.log')
    state['current_stage'] = name + ('_preflight' if dry_run else '')
    state['command'] = command
    state['stage_log'] = str(log_path)
    print(f'{state["current_stage"]}: {log_path}', flush=True)
    with log_path.open('a') as log:
        process = subprocess.Popen(command, cwd=PROJECT_PATH, stdout=log, stderr=subprocess.STDOUT)
        state['child_pid'] = process.pid
        atomic_json(root / 'protocol.json', state)
        code = process.wait()
    state['child_pid'] = None
    if code:
        raise subprocess.CalledProcessError(code, command)


def active_child(state):
    pid = state.get('child_pid')
    if pid:
        try:
            return psutil.Process(pid).cmdline() == state.get('command')
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
    return False


def foreground(resume=False):
    if ROOT.exists() and (ROOT / 'protocol.json').exists():
        previous = ablation.read_json(ROOT / 'protocol.json')
        if active_child(previous):
            raise RuntimeError('A task from the previous controller is still running; do not start a duplicate.')
    configuration = dict(stages=STAGES, code_sha256=support.code_hashes(),
                         packages=support.homology.package_versions(), n_cpu=32,
                         seed=63, search_seeds=[6301, 6302, 6303, 6304, 6305])
    # JSON represents tuples as lists, including on resume.
    configuration = support.json.loads(support.json.dumps(configuration))
    with support.experiment(ROOT, configuration, resume) as state:
        if len(os.sched_getaffinity(0)) < 32:
            raise RuntimeError('Fewer than 32 CPUs are available.')
        if shutil.disk_usage(RESULTS_PATH).free < 500 * 1024**3:
            raise RuntimeError('Less than 500 GiB free; inspect storage before starting.')
        atomic_json(ROOT / 'hardware.json', dict(platform=platform.platform(), cpu_count=os.cpu_count(),
            available_cpus=len(os.sched_getaffinity(0)), ram_bytes=psutil.virtual_memory().total,
            free_disk_bytes=shutil.disk_usage(RESULTS_PATH).free, python=sys.version))
        state['pid'] = os.getpid()
        # Check every source and cache before spending time on training.
        for name, script in STAGES:
            run_stage(name, script, ROOT, state, dry_run=True)
        for name, script in STAGES:
            if configuration['code_sha256'] != support.code_hashes():
                raise RuntimeError('Code changed after workflow freeze.')
            run_stage(name, script, ROOT, state)
            state['completed_results'] += 1
            atomic_json(ROOT / 'protocol.json', state)
        state['current_stage'] = 'finished'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--foreground', action='store_true', help='Internal controller mode; otherwise detach.')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.foreground:
        foreground(args.resume)
        return
    if ROOT.exists() and not args.resume:
        raise FileExistsError(f'Existing workflow preserved: {ROOT}; inspect before using --resume.')
    command = [sys.executable, str(Path(__file__).resolve()), '--foreground']
    if args.resume:
        command.append('--resume')
    environment = dict(os.environ, PYTHONHASHSEED='63', PYTHONUNBUFFERED='1')
    environment.pop('BIOAUTOML_RUN_ROOT', None)
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    with LOG.open('a') as log:
        process = subprocess.Popen(command, cwd=PROJECT_PATH, env=environment,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(f'Background controller PID: {process.pid}\nLog: {LOG}\nStatus: {ROOT / "protocol.json"}')


if __name__ == '__main__':
    main()
