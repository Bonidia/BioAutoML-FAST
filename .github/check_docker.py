"""Smoke checks for the deployment started by ci.sh."""

import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

from playwright.sync_api import expect, sync_playwright


def docker(*args, **kwargs):
    if args[0] == 'exec':
        args = ('exec', '--env', 'PYTHONPATH=/app', *args[1:])
    try:
        return subprocess.run(
            ['docker', *args], check=True, capture_output=True, text=True,
            timeout=60, **kwargs,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        error.add_note(error.stderr[-5000:])
        raise


def wait_job(container, identifier):
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        result = json.loads(docker('exec', '--workdir', '/app/App', container, 'python', '-c',
                                  'import json; from utils.db import TaskResultManager; '
                                  f'print(json.dumps(TaskResultManager().get_result({identifier!r})))'))
        if result['status'] == 'success':
            return
        if result['status'] == 'failure':
            raise RuntimeError(f'Queued job failed: {identifier}; inspect the training/worker logs.')
        time.sleep(10)
    raise TimeoutError(f'Queued job did not finish within 30 minutes: {identifier}')


def train_example(container):
    identifier = docker('exec', '--workdir', '/app/App', container, 'python', '-c', '''
from io import BytesIO
from pathlib import Path
from modules.home import submit_job
from utils.tasks import enqueue_task
def uploads(split):
    files = []
    for path in sorted((Path('/app/example-input') / split).glob('*.fasta')):
        item = BytesIO(path.read_bytes())
        item.name = path.name
        files.append(item)
    return files
print(enqueue_task(submit_job, dict(
    train_files=uploads('train'), test_files=uploads('test'), predict_path='/app/App/jobs',
    data_type='Protein', task='Classification', training='Training set', testing='Test set',
    tuning=True, seed=63, search_seed=6301, n_cpu=2, search_jobs=1,
    homology_aware=False, pruning=True), job_info=dict(
    data_type='Protein', task='Classification', training_set=True,
    testing_set='Test set', tuning=True, pruning=True)))
''').splitlines()[-1]
    print(f'Example queued through the training worker: {identifier}', flush=True)
    wait_job(container, identifier)
    docker('exec', '--workdir', '/app/App', container, 'python', '/app/deployment/example_job.py',
           '--export-job', f'/app/App/jobs/{identifier}', '--bundle', '/tmp/ci-example')
    print('Completed example verified and exported.', flush=True)


def check_example(container, address, artifacts):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            with urlopen(f'{address}/_stcore/health', timeout=5) as response:
                assert response.status == 200
            identifier = docker('exec', '--workdir', '/app/App', container, 'python', '-c', '''
from pathlib import Path
from utils.db import TaskResultManager
from deployment.example_job import verify_job
identifier = TaskResultManager().get_example_job_id()
assert identifier
assert not Path('/run/bioautoml/signing.pem').exists()
verify_job(Path('/app/App/jobs') / identifier)
print(identifier)
''')
            break
        except (OSError, AssertionError, subprocess.SubprocessError):
            time.sleep(3)
    else:
        raise RuntimeError('Published image did not install its example.')

    docker('exec', '--workdir', '/app/App', container, 'python', '-c', '''
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os, shutil, tempfile, uuid, zipfile
from redis import Redis
from deployment.example_job import install_example, verify_job
from utils.db import TaskResultManager, TaskStatus
from utils.retention import cleanup_jobs, is_expired, job_directory
manager = TaskResultManager()
identifier = manager.get_example_job_id()
bundle = Path('/app/bundled-example')
assert install_example(bundle) == identifier
future = datetime.now(timezone.utc) + timedelta(days=31)
assert not is_expired(manager.get_result(identifier), future)
ordinary = str(uuid.uuid4())
manager.store_pending_task(ordinary)
manager.store_result(ordinary, TaskStatus.SUCCESS)
job_directory(ordinary).mkdir()
removed = cleanup_jobs(manager, Redis(host='127.0.0.1'), now=future)
assert ordinary in removed and identifier not in removed
assert not job_directory(ordinary).exists()
verify_job(job_directory(identifier))
# Reject a database collision without changing the user's record or files.
with manager.connection:
    manager.connection.execute("DELETE FROM app_settings WHERE name='example_job_id'")
try:
    install_example(bundle)
except ValueError:
    pass
else:
    raise AssertionError('Unrelated job collision accepted')
manager.register_example(identifier, verify_job(bundle / identifier))
# Reject changed payloads before any estimator is deserialized.
with tempfile.TemporaryDirectory() as temporary:
    damaged = Path(temporary) / identifier
    shutil.copytree(bundle / identifier, damaged)
    model = damaged / 'model' / 'trained_model.sav'
    with zipfile.ZipFile(model) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members['prediction.joblib'] += b'tampered'
    with zipfile.ZipFile(model, 'w') as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    try:
        verify_job(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError('Tampered model payload accepted')
    trust = os.environ['BIOAUTOML_TRUSTED_KEYS']
    untrusted = Path(temporary) / 'untrusted.json'
    untrusted.write_text('{"keys": {}, "revoked": []}')
    os.environ['BIOAUTOML_TRUSTED_KEYS'] = str(untrusted)
    try:
        verify_job(bundle / identifier)
    except ValueError:
        pass
    else:
        raise AssertionError('Untrusted example accepted')
    finally:
        os.environ['BIOAUTOML_TRUSTED_KEYS'] = trust
''')
    print('Example installation, idempotence, collision and 30-day cleanup checks passed.', flush=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        try:
            page.goto(address, wait_until='domcontentloaded')
            page.get_by_role('button', name='Got it', exact=True).click(timeout=90000)
            page.frame_locator('iframe[title="streamlit_option_menu.option_menu"]').get_by_text('Jobs', exact=True).click()
            page.get_by_role('button', name='Example', exact=True).click(timeout=60000)
            expect(page.get_by_role('textbox', name='Enter Job ID', exact=True)).to_have_value(identifier)
            page.get_by_role('button', name='Submit', exact=True).click()
            page.get_by_text('Summary Statistics', exact=True).wait_for(state='visible', timeout=90000)
            page.get_by_role('tab', name='Model Information', exact=True).wait_for(state='visible')
            page.get_by_role('tab', name='Predictions', exact=True).click()
            assert page.locator('[data-testid="stException"]').count() == 0
            assert page.get_by_text('An error occurred. Submit a new job.', exact=True).count() == 0
        except Exception:
            page.screenshot(path=str(artifacts / 'example-failure.png'), full_page=True)
            raise
        finally:
            browser.close()
    print('Jobs opens the bundled example through its Example button.', flush=True)

    prediction = docker('exec', '--workdir', '/app/App', container, 'python', '-c', '''
from io import BytesIO
from pathlib import Path
from bioautoml.execution import run_path
from modules.home import submit_job
from utils.db import TaskResultManager
from utils.tasks import enqueue_task
root = Path('/app/App/jobs') / TaskResultManager().get_example_job_id()
model = BytesIO(Path(run_path(root, 'trained_model.sav')).read_bytes())
model.name = 'trained_model.sav'
files = []
for path in sorted(Path('/app/example-input/test').glob('*.fasta')):
    item = BytesIO(path.read_bytes())
    item.name = path.name
    files.append(item)
print(enqueue_task(submit_job, dict(train_files=model, test_files=files,
    predict_path='/app/App/jobs', data_type='Protein', task='Classification',
    training='Load model', testing='Test set', tuning=False), job_info=dict(
    data_type='Protein', task='Classification', training_set=False,
    testing_set='Test set', tuning=False, pruning=None)))
''').splitlines()[-1]
    wait_job(container, prediction)
    docker('exec', '--workdir', '/app/App', container, 'python', '-c', f'''
from pathlib import Path
import pandas as pd
from bioautoml.execution import run_path
left = pd.read_csv(run_path(Path('/app/App/jobs') / {identifier!r}, 'test_predictions.csv'))
right = pd.read_csv(run_path(Path('/app/App/jobs') / {prediction!r}, 'test_predictions.csv'))
assert left['nameseq'].is_unique and right['nameseq'].is_unique
pd.testing.assert_frame_equal(left.set_index('nameseq').sort_index(),
                              right.set_index('nameseq').sort_index())
''')
    docker('restart', container)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            docker('exec', '--workdir', '/app/App', container, 'python', '-c', f'''
from utils.db import TaskResultManager
from deployment.example_job import verify_job
from pathlib import Path
manager = TaskResultManager()
assert manager.get_example_job_id() == {identifier!r}
assert manager.get_result({prediction!r})['status'] == 'success'
verify_job(Path('/app/App/jobs') / {identifier!r})
''')
            with urlopen(f'{address}/_stcore/health', timeout=5) as response:
                assert response.status == 200
            break
        except (OSError, subprocess.SubprocessError):
            time.sleep(3)
    else:
        raise RuntimeError('Example failed after restart.')
    print('Signed-model inference matches training predictions; restart preserves both jobs.', flush=True)


def main():
    container = os.environ['CONTAINER_NAME']
    address = f"http://127.0.0.1:{os.environ['HOST_PORT']}"
    artifacts = Path(os.environ['CI_ARTIFACTS_DIR'])
    if '--example' in sys.argv:
        check_example(container, address, artifacts)
        return
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
    assert {path.name for path in Path(os.environ['MODEL_KEYS_DIR']).iterdir()} == {'signing.pem'}

    docker('exec', container, 'python', '-c', '''
import json, os
from pathlib import Path
assert not Path('/app/.bioautoml').exists(), 'Private key store was included in the image'
assert not Path('/run/bioautoml/signing.pem').exists(), 'Private key exposed to web container'
official = json.loads(Path('/app/deployment/official_public_keys.json').read_text())
trust = json.loads(Path(os.environ['BIOAUTOML_TRUSTED_KEYS']).read_text())
assert official['keys'].items() <= trust['keys'].items()
''')
    docker('exec', f'{container}-training', 'python', '-c', '''
import os
from bioautoml.model_security import initialize_web_signer, sign_manifest, verify_signature
initialize_web_signer()
manifest = {'ci': 'signed'}
signature = sign_manifest(manifest)
verify_signature(manifest, signature, check_packages=False)
for changed, token in (({'ci': 'tampered'}, signature),
                       (manifest, dict(signature, key_id='untrusted-ci-key'))):
    try:
        verify_signature(changed, token, check_packages=False)
    except ValueError:
        pass
    else:
        raise AssertionError('Untrusted or tampered signature accepted')
''')
    if os.environ.get('CI_REQUIRE_OFFICIAL_KEY') == '1':
        docker('exec', f'{container}-training', 'python', '-c', '''
import os
from bioautoml.model_security import initialize_web_signer, sign_manifest, verify_signature
initialize_web_signer()
manifest = {'ci': 'official signer'}
signature = sign_manifest(manifest)
os.environ['BIOAUTOML_TRUSTED_KEYS'] = '/app/deployment/official_public_keys.json'
verify_signature(manifest, signature, check_packages=False)
''')
    print('Signing isolation, bundled trust and signature rejection checks passed.', flush=True)

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
            page.get_by_text('Homology-aware cross-validation', exact=True).click()
            page.get_by_role('combobox', name=re.compile('Testing')).click()
            page.get_by_role('option', name='Test set', exact=True).click()
            filter_label = 'Filter test sequences similar to training sequences'
            page.get_by_text(filter_label, exact=True).wait_for(state='visible')
            assert not page.get_by_role('checkbox', name=filter_label, exact=True).is_checked()
            page.get_by_text(filter_label, exact=True).click()
            expect(page.get_by_role('checkbox', name=filter_label, exact=True)).to_be_checked()
            page.get_by_role('combobox', name=re.compile('Testing')).click()
            page.get_by_role('option', name='Prediction set', exact=True).click()
            page.get_by_text(filter_label, exact=True).wait_for(state='detached')
            print('Optional test-overlap filter is available only for labelled test evaluation.', flush=True)
            menu = page.frame_locator('iframe[title="streamlit_option_menu.option_menu"]')
            menu.get_by_text('Model Repository', exact=True).click()
            page.get_by_text('Datasets were not found.', exact=False).wait_for(state='visible')
            assert page.locator('[data-testid="stException"]').count() == 0
            assert not errors, errors
            print('Model Repository handles missing datasets without an error.', flush=True)
        except Exception:
            page.screenshot(path=str(artifacts / 'home-failure.png'), full_page=True)
            (artifacts / 'browser-errors.txt').write_text('\n'.join(errors), encoding='utf-8')
            raise
        finally:
            browser.close()
    train_example(container)


if __name__ == '__main__':
    main()
