"""Print environment and source fingerprints without reading data or secrets."""

import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import sys


def main():
    project_path = Path(__file__).resolve().parents[1]
    paths = [project_path / name for name in (
        "pyproject.toml", "uv.lock", "pixi.toml", "pixi.lock", ".python-version",
        "Dockerfile", "start.sh", "engineering.py", "generation.py", "App/index.html",
    )]
    for name in ("bioautoml", "App", "other-methods", "MathFeature", "manuscript"):
        paths.extend((project_path / name).rglob("*.py"))
    hashes = {
        str(path.relative_to(project_path)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(paths)) if path.is_file()
    }
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": dict(sorted(
            (package.metadata["Name"], package.version) for package in distributions()
            if package.metadata["Name"]
        )),
        "source_sha256": hashes,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
