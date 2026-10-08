"""CPU-bounded execution of independent descriptor programs."""

import os
import subprocess
import time

from joblib import cpu_count


def get_available_cpus(n_cpu=-1):
    """Respect affinity and container CPU quotas, including with --n_cpu=-1."""
    available_cpus = cpu_count()
    return available_cpus if n_cpu < 1 else min(n_cpu, available_cpus)


def run_feature_commands(commands, log_dir, n_cpu=-1, cwd=None):
    """Bound both outer programs and repDNA's nested descriptor workers.

    repDNA receives up to all but one slot when other programs are present;
    otherwise it can use the whole budget. Each other program uses one slot.
    Its correlation descriptors dominate nucleotide extraction, so reserving
    half the CPUs for shorter programs unnecessarily delays the critical path.
    Start repDNA first so its longer-running work overlaps the other programs.
    Output files and their eventual concatenation order are unchanged.
    """
    os.makedirs(log_dir, exist_ok=True)
    total_cpus = get_available_cpus(n_cpu)
    pending = []
    for descriptor, command in commands:
        command = list(command)
        workers = 1
        if descriptor == 'repDNA':
            descriptor_count = len(command) - command.index('--descriptors') - 1 if '--descriptors' in command else 9
            workers = min(descriptor_count, max(1, total_cpus - 1) if len(commands) > 1 else total_cpus)
            command += ['--n_cpu', str(workers)]
        pending.append((descriptor, command, workers))
    pending.sort(key=lambda item: item[0] != 'repDNA')

    # These limits apply only to extractor subprocesses, never to model fits.
    environment = os.environ.copy()
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                     'POLARS_MAX_THREADS'):
        environment[variable] = '1'
    processes = []
    failed_command = None
    available_cpus = total_cpus
    try:
        while pending or processes:
            for item in pending[:]:
                descriptor, command, workers = item
                if workers > available_cpus:
                    continue
                log_file = open(os.path.join(log_dir, f'{descriptor}.log'), 'a')
                child_environment = environment.copy()
                # modlAMP also uses joblib workers internally. Give every
                # child only its reserved slots, including repDNA's pool.
                child_environment['LOKY_MAX_CPU_COUNT'] = str(workers)
                try:
                    process = subprocess.Popen(command, cwd=cwd, env=child_environment,
                                               stdout=log_file, stderr=subprocess.STDOUT)
                except BaseException:
                    log_file.close()
                    raise
                processes.append((command, process, log_file, workers))
                available_cpus -= workers
                pending.remove(item)
            for item in processes[:]:
                command, process, log_file, workers = item
                return_code = process.poll()
                if return_code is None:
                    continue
                log_file.close()
                processes.remove(item)
                available_cpus += workers
                if return_code != 0 and failed_command is None:
                    failed_command = (return_code, command)
            if processes:
                time.sleep(0.02)
        if failed_command is not None:
            raise subprocess.CalledProcessError(*failed_command)
    finally:
        for _, process, log_file, _ in processes:
            if process.poll() is None:
                process.terminate()
            process.wait()
            log_file.close()
