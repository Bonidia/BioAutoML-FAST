"""Create and validate deployment keys without signing model files."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def read_private(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
        raise ValueError('Private key must be a regular, non-symlink file with owner-only permissions (chmod 600).')
    if info.st_size > 16384:
        raise ValueError('Private-key file is too large.')
    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError('An unencrypted Ed25519 PEM private key is required.')
    return key


def public_value(key):
    return base64.b64encode(key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode('ascii')


def read_trust(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Trust file must be a regular, non-symlink file: {path}')
    trust = json.loads(path.read_text())
    if not isinstance(trust.get('keys'), dict) or not isinstance(trust.get('revoked'), list):
        raise ValueError('Trust files must contain a keys object and a revoked list.')
    for key_id, value in trust['keys'].items():
        if not isinstance(key_id, str) or not key_id or not all(c.isascii() and (c.isalnum() or c in '_-') for c in key_id):
            raise ValueError('Invalid public-key ID.')
        Ed25519PublicKey.from_public_bytes(base64.b64decode(value, validate=True))
    if any(not isinstance(item, str) for item in trust['revoked']):
        raise ValueError('Invalid revoked-key list.')
    return trust


def merge_trust(documents):
    merged = {'keys': {}, 'revoked': []}
    for document in documents:
        for key_id, value in document['keys'].items():
            if key_id in merged['keys'] and merged['keys'][key_id] != value:
                raise ValueError(f'Conflicting public keys for signer {key_id}.')
            merged['keys'][key_id] = value
        merged['revoked'] = sorted(set(merged['revoked']) | set(document['revoked']))
    return merged


def write_new(path, data):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'wb') as handle:
        handle.write(data)


def write_trust(path, trust):
    if path.is_symlink():
        raise ValueError('Trust output must not be a symlink.')
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as handle:
        json.dump(trust, handle, indent=2)
        temporary = Path(handle.name)
    temporary.replace(path)


def confirm(message):
    if not sys.stdin.isatty():
        raise ValueError('Run this command in a terminal to confirm the change.')
    try:
        return input(message + ' [y/N] ').strip().lower() in ('y', 'yes')
    except EOFError:
        return False


def manage_key(output, official, delete=False):
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError('Key directories must not contain symlinks.')
    if output.exists() and (not output.is_dir() or output.stat().st_mode & 0o077):
        raise ValueError('Key directory must have owner-only permissions (chmod 700).')
    private_path = output / 'signing.pem'
    previous = None
    if private_path.exists() or private_path.is_symlink():
        read_private(private_path)
        previous = private_path.read_bytes()
    if delete and previous is None:
        print(f'No private key found: {private_path}. Public keys unchanged.')
        return
    if previous is not None:
        action = 'Delete' if delete else 'Replace'
        if not confirm(f'{action} {private_path}? Public keys remain trusted. Back up first; recovery requires a backup.'):
            print('Cancelled; keys unchanged.')
            return
        if previous is not None and (private_path.is_symlink() or private_path.read_bytes() != previous):
            raise ValueError('Private key changed during confirmation; retry.')
    if delete:
        private_path.unlink(missing_ok=True)
        print(f'Removed {private_path}. Private-key recovery requires a backup. Public keys unchanged.')
        return
    trust = read_trust(official)
    key = Ed25519PrivateKey.generate()
    public = public_value(key)
    key_id = 'ed25519-' + hashlib.sha256(base64.b64decode(public)).hexdigest()
    trust = merge_trust([trust, {'keys': {key_id: public}, 'revoked': []}])
    output.mkdir(parents=True, mode=0o700, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(key.private_bytes(serialization.Encoding.PEM,
                                      serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    try:
        # Publish verification material before replacing the private key.
        write_trust(official, trust)
        if previous is None:
            os.link(temporary, private_path)
        else:
            temporary.replace(private_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Private key: {private_path} (keep secret; back up securely)')
    print(f'Updated {official}; previous public keys and explicit revocations retained.')
    print('Publish the updated public-key file and rebuild the image to distribute trust. Restart deployments to use the new key.')


def configure(output, private_key=None, reuse=False, trust_files=(), official=None,
              verification_file=None, print_key_id=False):
    # Do not resolve symlinks before checking the operator-supplied destination.
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError('Key directories must not contain symlinks.')
    if verification_file:
        if any(path.is_symlink() for path in (verification_file, *verification_file.parents)):
            raise ValueError('Verification paths must not contain symlinks.')
        if verification_file.parent.stat().st_mode & 0o077:
            raise ValueError('Verification directory must have owner-only permissions (chmod 700).')
    additional = [read_trust(path) for path in trust_files]
    if verification_file and verification_file.exists():
        additional.append(read_trust(verification_file))
    if (output / 'trusted_keys.json').exists():
        additional.append(read_trust(output / 'trusted_keys.json'))
    imported = read_private(private_key) if private_key else None
    if output.exists():
        if not output.is_dir() or output.stat().st_mode & 0o077:
            raise ValueError('Key directory must have owner-only permissions (chmod 700).')
    private_path = output / 'signing.pem'
    if private_path.exists() or private_path.is_symlink():
        if not reuse:
            raise FileExistsError('Private key already exists; choose a new directory. Existing keys are never overwritten.')
        key = read_private(output / 'signing.pem')
        if imported and public_value(imported) != public_value(key):
            raise ValueError('The supplied key differs from this directory. Use a new MODEL_KEYS_DIR for rotation.')
    else:
        key = imported or Ed25519PrivateKey.generate()
    public = public_value(key)
    fingerprint = hashlib.sha256(base64.b64decode(public)).hexdigest()
    key_id = 'ed25519-' + fingerprint
    local = merge_trust([{'keys': {key_id: public}, 'revoked': []}, *additional])
    merged = merge_trust([local, read_trust(official)] if official else [local])
    if any(merged['keys'].get(revoked) == public for revoked in merged['revoked']):
        raise ValueError('The selected signing key is revoked.')
    if verification_file and verification_file.absolute() == private_path.absolute():
        raise ValueError('Verification output must not overwrite the private key.')
    if not output.exists():
        output.mkdir(parents=True, mode=0o700)
    if not private_path.exists():
        write_new(output / 'signing.pem', key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    if verification_file:
        write_trust(verification_file, merged)
    if print_key_id:
        print(key_id)
    else:
        print(f'Private key: {private_path} (keep secret; back up securely)')
        print(f'Public-key SHA256 fingerprint: {fingerprint}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path.cwd() / '.bioautoml')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--create', action='store_true', help='Create a private key and update bundled public trust; confirm replacement')
    actions.add_argument('--delete', action='store_true', help='Confirm deletion of only the private key; public trust is unchanged')
    parser.add_argument('--private-key', type=Path, help='Import an existing private key into a new protected directory')
    parser.add_argument('--reuse', action='store_true', help='Validate existing keys instead of replacing them')
    parser.add_argument('--trust-file', type=Path, action='append', default=[],
                        help='Additional public-key JSON trust, e.g. previous signing keys during rotation')
    parser.add_argument('--verification-file', type=Path, help='Write generated public trust outside the private-key directory')
    parser.add_argument('--print-key-id', action='store_true', help='Print only the derived public signer ID (for the launcher)')
    args = parser.parse_args()
    official = Path(__file__).with_name('official_public_keys.json')
    try:
        if args.create or args.delete:
            if args.private_key or args.reuse or args.trust_file or args.verification_file or args.print_key_id:
                parser.error('--create/--delete can only be combined with --output')
            manage_key(args.output.absolute(), official, delete=args.delete)
        else:
            configure(args.output.absolute(), args.private_key, args.reuse, args.trust_file, official,
                      args.verification_file, args.print_key_id)
    except (ValueError, TypeError, OSError) as error:
        parser.exit(1, f'Key configuration failed: {error}\n')


if __name__ == '__main__':
    main()
