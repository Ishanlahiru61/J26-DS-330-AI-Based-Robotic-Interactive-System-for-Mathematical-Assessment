"""Install and verify the untracked C2 runtime model from an approved source."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


RUNTIME_DIR = Path(__file__).resolve().parents[2] / "app" / "modules" / "adaptive_questions" / "models" / "transferable"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install(source: Path, destination: Path = RUNTIME_DIR) -> None:
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    for filename in ("model.json", "calibrator.joblib"):
        source_file = source / filename
        expected = manifest["files"][filename]
        if not source_file.is_file():
            raise FileNotFoundError(source_file)
        if source_file.stat().st_size != expected["bytes"] or sha256(source_file) != expected["sha256"]:
            raise ValueError(f"hash or size mismatch: {filename}")
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination / filename)
        if sha256(destination / filename) != expected["sha256"]:
            raise ValueError(f"post-copy verification failed: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="directory containing model.json and calibrator.joblib")
    parser.add_argument("--destination", type=Path, default=RUNTIME_DIR)
    args = parser.parse_args()
    install(args.source.resolve(), args.destination.resolve())
    print(f"Verified C2 model installed in {args.destination.resolve()}")
