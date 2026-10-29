"""Ephemeral job credentials, kept separate from the durable RQ database."""
import base64
import io
import os
from pathlib import Path
import secrets
import shutil
import tarfile
import uuid

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from redis import Redis


SECRET_TTL = 30 * 24 * 3600


def private_store():
    connection = Redis.from_url(os.environ.get('PRIVATE_REDIS_URL', 'redis://127.0.0.1:6380/0'),
                                socket_connect_timeout=3, socket_timeout=3)
    # A different Redis database number is NOT sufficient: persistence is server-wide.
    configuration = {**connection.config_get('appendonly'), **connection.config_get('save')}
    if configuration.get('appendonly') != 'no' or configuration.get('save') != '':
        raise RuntimeError('The private Redis server must have AOF and snapshots disabled.')
    return connection


def private_key(reference):
    return 'bioautoml:private:' + str(uuid.UUID(reference))


def derive_key_from_password(password, salt):
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=390000)
    return base64.urlsafe_b64encode(kdf.derive(password.encode('utf-8')))


def store_credentials(email=None, password=None):
    if not email and not password:
        return None
    values = {}
    if email:
        values['email'] = email
    if password:
        salt = secrets.token_bytes(16)
        values.update(key=derive_key_from_password(password, salt), salt=salt)
    reference = str(uuid.uuid4())
    connection = private_store()
    try:
        with connection.pipeline(transaction=True) as pipeline:
            pipeline.hset(private_key(reference), mapping=values)
            pipeline.expire(private_key(reference), SECRET_TTL)
            pipeline.execute()
    except Exception:
        # Redis pipeline errors can include the complete HSET command. Never
        # propagate them to Streamlit/RQ, where arguments could reach logs.
        try:
            connection.delete(private_key(reference))
        except Exception:
            pass
        raise RuntimeError('Temporary credential storage unavailable; please try again.') from None
    return reference


def take_encryption_key(reference):
    if not reference:
        raise RuntimeError('Encryption credentials unavailable; please resubmit the job.')
    connection = private_store()
    with connection.pipeline(transaction=True) as pipeline:
        pipeline.hmget(private_key(reference), 'key', 'salt')
        pipeline.hdel(private_key(reference), 'key', 'salt')
        values, _ = pipeline.execute()
    if not all(values):
        raise RuntimeError('Encryption credentials expired or were lost after a restart; please resubmit.')
    return values


def take_email(reference):
    if not reference:
        return None
    connection = private_store()
    with connection.pipeline(transaction=True) as pipeline:
        pipeline.hget(private_key(reference), 'email')
        pipeline.delete(private_key(reference))
        value, _ = pipeline.execute()
    return value.decode('utf-8') if value else None


def discard_credentials(reference):
    if reference:
        private_store().delete(private_key(reference))


def encrypt_job_folder(job_path, key, salt):
    """Publish an archive only after encryption succeeds; do not ignore deletion errors."""
    root = Path(job_path)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz', dereference=False) as archive:
        archive.add(root, arcname='.')
    encrypted = Fernet(key).encrypt(buffer.getvalue())
    temporary = root / '.job_archive.enc.tmp'
    temporary.write_bytes(encrypted)
    os.chmod(temporary, 0o600)
    (root / 'job_salt.bin').write_bytes(salt)
    temporary.replace(root / 'job_archive.enc')
    for path in root.iterdir():
        if path.name in {'job_archive.enc', 'job_salt.bin'}:
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
