"""Smoke checks for the deployment started by ci.sh."""

import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen

from playwright.sync_api import sync_playwright


def docker(*args, **kwargs):
    return subprocess.run(
        ['docker', *args], check=True, capture_output=True, text=True,
        timeout=60, **kwargs,
    ).stdout.strip()


def main():
    container = os.environ['CONTAINER_NAME']
    address = f"http://127.0.0.1:{os.environ['HOST_PORT']}"
    artifacts = Path(os.environ['CI_ARTIFACTS_DIR'])
    deadline = time.monotonic() + 180
    last_error = None
    while time.monotonic() < deadline:
        try:
            for name in (container, f'{container}-training'):
                state = json.loads(docker('inspect', '--format', '{{json .State}}', name))
                if not state['Running']:
                    raise RuntimeError(f'{name} is not running: {state["Status"]}')
            with urlopen(f'{address}/_stcore/health', timeout=5) as response:
                assert response.status == 200
            docker('exec', container, 'python', '-c', '''
from redis import Redis
from rq import Worker
connection = Redis(host='127.0.0.1', port=6379, socket_timeout=5)
assert connection.ping()
assert Redis(host='127.0.0.1', port=6380, socket_timeout=5).ping()
queues = {queue for worker in Worker.all(connection=connection) for queue in worker.queue_names()}
assert {'bioautoml', 'bioautoml-training'} <= queues, queues
''')
            break
        except (OSError, AssertionError, RuntimeError, subprocess.SubprocessError) as error:
            last_error = error
            time.sleep(3)
    else:
        raise RuntimeError(f'Deployment did not become ready: {last_error}')
    print('Streamlit, Redis, and both workers are ready.', flush=True)

    for script in ('engineering.py', 'generation.py'):
        output = docker('exec', '--workdir', '/app', container, 'python', script, '--help')
        assert 'usage:' in output.lower(), output
        print(f'{script} --help passed.', flush=True)
    print(f"MMseqs2: {docker('exec', container, 'mmseqs', 'version')}", flush=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        try:
            page.goto(address, wait_until='domcontentloaded', timeout=60000)
            page.get_by_role('button', name='Got it', exact=True).click(timeout=90000)
            for label in ('Hyperparameter tuning', 'Homology-aware cross-validation', 'Trial pruning'):
                # Streamlit styles the label and hides the native checkbox input.
                page.get_by_role('checkbox', name=label, exact=True).wait_for(state='attached')
                page.get_by_text(label, exact=True).wait_for(state='visible')
            assert 'BioAutoML-FAST' in page.title()
            assert page.locator('[data-testid="stException"]').count() == 0
            assert not errors, errors
            print('Home page rendered successfully in Chromium.', flush=True)
        except Exception:
            page.screenshot(path=str(artifacts / 'home-failure.png'), full_page=True)
            (artifacts / 'browser-errors.txt').write_text('\n'.join(errors), encoding='utf-8')
            raise
        finally:
            browser.close()


if __name__ == '__main__':
    main()
