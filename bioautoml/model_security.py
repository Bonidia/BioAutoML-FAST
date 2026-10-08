"""Ed25519 bundle verification. Private keys belong only to training workers."""
import base64
from importlib.metadata import version
import json
import os
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


_signing_key = None
_key_id = None
PACKAGES = ('numpy', 'scikit-learn', 'lightgbm', 'xgboost', 'joblib')


def canonical_manifest(manifest):
    return json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def initialize_web_signer():
    """Called at trusted training-worker bootstrap, never by an upload handler."""
    global _signing_key, _key_id
    path = Path(os.environ.pop('BIOAUTOML_SIGNING_KEY'))
    if path.stat().st_mode & 0o077:
        raise ValueError('Signing key must be accessible only to its owner (chmod 600).')
    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError('An Ed25519 private key is required.')
    _key_id = os.environ.pop('BIOAUTOML_SIGNING_KEY_ID')
    _signing_key = key
    # Fail at startup, not after an expensive training job, on mismatched trust.
    probe = {'bootstrap': True}
    verify_signature(probe, sign_manifest(probe), check_packages=False)


def require_web_signer():
    if _signing_key is None:
        raise ValueError('Web training signing is not configured; use a dedicated training worker.')


def sign_manifest(manifest):
    require_web_signer()
    return {'key_id': _key_id,
            'signature': base64.b64encode(_signing_key.sign(canonical_manifest(manifest))).decode('ascii')}


def verify_signature(manifest, signature, check_packages=True):
    trust_path = os.environ.get('BIOAUTOML_TRUSTED_KEYS')
    if not trust_path:
        raise ValueError('No trusted model verification keys configured.')
    # Re-read on access so removing/revoking a key also affects existing sessions.
    trust = json.loads(Path(trust_path).read_text())
    key_id = signature.get('key_id')
    if key_id in trust.get('revoked', []) or key_id not in trust.get('keys', {}):
        raise ValueError('Unknown or revoked model signer.')
    try:
        key = Ed25519PublicKey.from_public_bytes(base64.b64decode(trust['keys'][key_id], validate=True))
        key.verify(base64.b64decode(signature['signature'], validate=True), canonical_manifest(manifest))
    except (InvalidSignature, ValueError, KeyError, TypeError) as error:
        raise ValueError('Invalid model signature.') from error
    if check_packages:
        if set(manifest.get('packages', {})) != set(PACKAGES):
            raise ValueError('Missing signed model compatibility metadata.')
        for name, expected in manifest['packages'].items():
            if version(name) != expected:
                raise ValueError(f'Incompatible model dependency: {name} requires {expected}.')
    return key_id


if __name__ == '__main__':
    import argparse
    import uuid
    parser = argparse.ArgumentParser(description='Create deployment signing keys; does not sign model files.')
    parser.add_argument('directory', help='New private directory outside the repository and Docker build context')
    args = parser.parse_args()
    directory = Path(args.directory).resolve()
    project = Path(__file__).resolve().parent.parent
    if directory == project or project in directory.parents:
        parser.error('Keep private keys outside the repository/build context.')
    directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    key = Ed25519PrivateKey.generate()
    key_id = str(uuid.uuid4())
    descriptor = os.open(directory / 'signing.pem', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as handle:
        handle.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                       serialization.NoEncryption()))
    public = base64.b64encode(key.public_key().public_bytes(serialization.Encoding.Raw,
                                                          serialization.PublicFormat.Raw)).decode('ascii')
    (directory / 'trusted_keys.json').write_text(json.dumps({'keys': {key_id: public}, 'revoked': []}, indent=2))
    (directory / 'key_id').write_text(key_id)
    print('Created signing.pem (private), trusted_keys.json (public), and key_id. Back up the private key securely.')
