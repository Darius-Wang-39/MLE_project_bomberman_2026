"""Create and verify a deterministic agent-only submission archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


INCLUDED_SUFFIXES = {".py", ".pt", ".npz", ".md", ".txt"}
EXCLUDED_PARTS = {"__pycache__", "logs"}
FIXED_ZIP_TIME = (2026, 8, 24, 0, 0, 0)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def submission_files(agent_directory: Path):
    files = []
    for path in agent_directory.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(agent_directory)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() not in INCLUDED_SUFFIXES:
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.as_posix())


def build_archive(agent_directory: Path, output: Path, manifest_path: Path):
    agent_directory = agent_directory.resolve()
    files = submission_files(agent_directory)
    names = {path.name for path in files}
    required = {"callbacks.py", "dueling_ddqn_model.pt", "features.py", "model.py"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"Missing required submission files: {missing}")

    output.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = Path(agent_directory.name) / path.relative_to(agent_directory)
            content = path.read_bytes()
            info = zipfile.ZipInfo(relative.as_posix(), FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content, compresslevel=9)
            entries.append(
                {
                    "path": relative.as_posix(),
                    "bytes": len(content),
                    "sha256": sha256_bytes(content),
                }
            )

    with zipfile.ZipFile(output, "r") as archive:
        archive.testzip()
        members = archive.namelist()
        if any("__pycache__" in name or "/logs/" in name or name.endswith(".pyc") for name in members):
            raise ValueError("Archive contains excluded cache or log files")
        for member in members:
            if member.endswith(".py"):
                compile(archive.read(member), member, "exec")

    archive_content = output.read_bytes()
    manifest = {
        "agent": agent_directory.name,
        "archive": output.name,
        "archive_bytes": len(archive_content),
        "archive_sha256": sha256_bytes(archive_content),
        "files": entries,
        "verification": {
            "zip_crc": "passed",
            "python_syntax": "passed",
            "cache_and_logs_excluded": True,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_archive(args.agent_directory, args.output, args.manifest), indent=2))


if __name__ == "__main__":
    main()

