#!/usr/bin/env python3
"""Download only the four collections specified in CHI3242 week 8.

Uses Git's shallow, blob-filtered sparse checkout in a temporary directory.
Existing files with different content are protected unless --force is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY = "https://github.com/mcjkurz/qhchina-data.git"
COLLECTIONS = (
    ("鲁迅", "呐喊", "corpora/authors/鲁迅/呐喊"),
    ("鲁迅", "彷徨", "corpora/authors/鲁迅/彷徨"),
    ("鲁迅", "故事新编", "corpora/authors/鲁迅/故事新编"),
    ("沈从文", "沈从文", "corpora/authors/沈从文/沈从文小说集"),
)


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, encoding="utf-8"
    ).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(destination: Path, *, force: bool = False, commit: str | None = None) -> dict:
    destination = destination.resolve()
    with tempfile.TemporaryDirectory(prefix="chi3242-corpus-") as temporary:
        checkout = Path(temporary) / "qhchina-data"
        git(
            "clone", "--filter=blob:none", "--depth=1", "--sparse",
            "--branch", "main", REPOSITORY, str(checkout),
        )
        if commit and git("rev-parse", "HEAD", cwd=checkout) != commit:
            git("fetch", "--depth=1", "origin", commit, cwd=checkout)
            git("checkout", "--detach", "FETCH_HEAD", cwd=checkout)
        git("sparse-checkout", "set", *(item[2] for item in COLLECTIONS), cwd=checkout)
        revision = git("rev-parse", "HEAD", cwd=checkout)
        source_files: list[tuple[Path, Path]] = []
        collections = []
        for author, label, relative_directory in COLLECTIONS:
            source_directory = checkout / relative_directory
            if not source_directory.is_dir():
                raise FileNotFoundError(f"Missing collection: {relative_directory}")
            files = sorted(source_directory.rglob("*.txt"))
            if not files:
                raise ValueError(f"No .txt files in {relative_directory}")
            file_records = []
            for source in files:
                relative_path = source.relative_to(checkout)
                target = destination / relative_path
                digest = sha256(source)
                if target.exists() and sha256(target) != digest and not force:
                    raise FileExistsError(
                        f"Refusing to overwrite different content: {target}. "
                        "Review it first, or use --force."
                    )
                # Validate all files before making any changes to the destination.
                source_files.append((source, target))
                file_records.append({
                    "path": relative_path.as_posix(),
                    "bytes": source.stat().st_size,
                    "sha256": digest,
                })
            collections.append({
                "author": author,
                "collection": label,
                "source_directory": relative_directory,
                "txt_count": len(files),
                "files": file_records,
            })
        for source, target in source_files:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or sha256(target) != sha256(source):
                shutil.copyfile(source, target)
        provenance = {
            "repository": REPOSITORY,
            "branch": "main",
            "commit": revision,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "method": "git clone --filter=blob:none --depth=1 --sparse; sparse-checkout four requested collection directories; copy .txt files only",
            "total_txt_files": len(source_files),
            "collections": collections,
        }
        manifest = destination / "data-provenance.json"
        manifest.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination", type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project directory (default: parent of scripts/)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite changed corpus files")
    parser.add_argument("--commit", help="Reproduce a particular upstream commit")
    args = parser.parse_args()
    manifest = download(args.destination, force=args.force, commit=args.commit)
    print(f"Upstream commit: {manifest['commit']}")
    for collection in manifest["collections"]:
        print(f"{collection['source_directory']}: {collection['txt_count']} .txt files")
    print(f"Total: {manifest['total_txt_files']} .txt files")


if __name__ == "__main__":
    main()
