"""Export an audited source snapshot without internal Git history or local data."""

import argparse
import shutil
import sys
import tarfile
from pathlib import Path

from audit_release import scan_source, source_files

TOP_LEVEL = {
    "README.md",
    "README.zh-CN.md",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    ".gitignore",
    ".env.example",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "LICENSE",
    "NOTICE",
}
DIRECTORIES = {"src", "config", "tools", "examples", "tests", "docs", ".github"}


def archive_metadata(info):
    # Do not disclose workstation user/group names through archive headers.
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("dist/edge-model-bench"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = args.destination.resolve()
    if destination.exists():
        parser.error("Destination already exists; choose a new empty path")
    files = []
    for path in source_files(root):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts[0] in DIRECTORIES or str(relative) in TOP_LEVEL:
            if path.is_symlink():
                parser.error(f"Review symlink before export: {relative}")
            files.append((path, relative))
    destination.mkdir(parents=True)
    for path, relative in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    if scan_source(destination):
        print(f"Export requires review: {destination}", file=sys.stderr)
        raise SystemExit(1)
    archive = destination.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(destination, arcname="edge-model-bench", filter=archive_metadata)
    print(f"Audited source: {destination}\nArchive: {archive}")
    if not (destination / "LICENSE").is_file():
        print(
            "Publication pending: copyright holder approval and an approved LICENSE are required."
        )


if __name__ == "__main__":
    main()
