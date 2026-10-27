"""Heuristic release scan. Reports locations/rules, never matching secret values."""

import argparse
import ipaddress
import re
import subprocess
from pathlib import Path

RULES = {
    "personal absolute path": re.compile(r"/(?:home|Users)/[\w.-]+/"),
    "embedded password": re.compile(r"""(?i)password\s*(?:=|:)\s*["'][^"']+["']"""),
    "private key": re.compile(r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "unverified SSH host key policy": re.compile(r"paramiko\.AutoAddPolicy\(\)"),
}
IP = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
BINARY_SUFFIXES = {
    ".onnx",
    ".hbm",
    ".mtnn",
    ".bin",
    ".dlc",
    ".whl",
    ".zip",
    ".pt",
    ".pth",
    ".pem",
    ".key",
    ".credentials",
}


def findings(data: bytes):
    text = data.decode("utf-8", errors="replace")
    for number, line in enumerate(text.splitlines(), 1):
        for label, pattern in RULES.items():
            if pattern.search(line):
                yield number, label
        for candidate in IP.findall(line):
            try:
                address = ipaddress.ip_address(candidate)
            except ValueError:
                continue
            if (
                address.is_private and not address.is_loopback and not address.is_unspecified
            ) or address in ipaddress.ip_network((0x64400000, 10)):
                yield number, "internal IP address"


def source_files(root: Path):
    if (root / ".git").exists():
        output = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "-cz", "--others", "--exclude-standard"]
        )
        return [root / name.decode() for name in output.split(b"\0") if name]
    return [p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]


def scan_source(root: Path) -> int:
    count = 0
    for path in sorted(set(source_files(root))):
        if not path.is_file():
            continue
        if path.suffix.lower() in BINARY_SUFFIXES:
            print(f"{path.relative_to(root)}: binary model, SDK, archive, or credential file")
            count += 1
        for number, label in findings(path.read_bytes()):
            print(f"{path.relative_to(root)}:{number}: {label}")
            count += 1
    return count


def scan_history(root: Path) -> int:
    """Scan every reachable blob, including old/deleted files, without printing content."""
    objects = subprocess.check_output(
        ["git", "-C", str(root), "rev-list", "--objects", "--all"]
    ).splitlines()
    names = {
        line.split(b" ", 1)[0].decode(): line.split(b" ", 1)[-1].decode(errors="replace")
        for line in objects
    }
    count = 0
    # Two batches avoid spawning a process for every historical object.
    query = "\n".join(names) + "\n"
    info = subprocess.check_output(
        ["git", "-C", str(root), "cat-file", "--batch-check"], input=query.encode()
    )
    blobs = [line.split()[0] for line in info.splitlines() if line.split()[1] == b"blob"]
    raw = subprocess.check_output(
        ["git", "-C", str(root), "cat-file", "--batch"], input=b"\n".join(blobs) + b"\n"
    )
    position = 0
    for _ in blobs:
        end = raw.index(b"\n", position)
        oid, _, size = raw[position:end].split()
        position = end + 1
        data = raw[position : position + int(size)]
        position += int(size) + 1
        labels = sorted({label for _, label in findings(data)})
        if labels:
            print(f"history {oid.decode()[:12]} {names[oid.decode()]}: {', '.join(labels)}")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    count = scan_history(root) if args.history else scan_source(root)
    print(f"{'Historical blobs' if args.history else 'Source findings'}: {count}")
    raise SystemExit(1 if count else 0)


if __name__ == "__main__":
    main()
