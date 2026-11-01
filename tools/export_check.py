"""Check release files for private content and generated artifacts before publishing."""

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_LIST = ROOT / "tools" / "release_files.txt"
BINARY = {
    ".pt",
    ".pth",
    ".safetensors",
    ".bin",
    ".gguf",
    ".pdf",
    ".sqlite",
    ".db",
    ".pem",
    ".key",
    ".npy",
    ".npz",
    ".parquet",
    ".onnx",
}
DATA_SUFFIXES = {".jsonl", ".log", ".csv", ".tsv"}
# Exact paths of reviewed binary presentation assets. Each entry must also be
# on the release list. Only the text-pattern scans are skipped for these
# paths, and only when the bytes carry genuine image magic.
REVIEWED_IMAGE_ASSETS = {
    "docs/assets/siliconshield-banner.png",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
TOKEN = re.compile(r"\b(?:hf_|gh[pousr]_|github_pat_|sk-)[A-Za-z0-9_-]{20,}")
GPU = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----\s+[A-Za-z0-9+/=]{32,}"
)
UUID = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
WORKSTATION = re.compile(r"(/(?:home|Users)/[^/\s]+/|\b[A-Za-z]:\\Users\\)")
IP_PORT = re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3}):\d{2,5}\b")
RESULT_MARKERS = (
    '"training_report"',
    '"development_acceptance"',
    "campaign-status",
    '"initial_development"',
    '"source_and_data"',
)


def release_files():
    return {
        line.strip()
        for line in RELEASE_LIST.read_text().splitlines()
        if line.strip()
    }


def _image_magic(suffix, content):
    if suffix == ".png":
        return content[:8] == b"\x89PNG\r\n\x1a\n"
    if suffix in {".jpg", ".jpeg"}:
        return content[:2] == b"\xff\xd8"
    return False


def check(name, content, private_words=(), reviewed=None):
    p = Path(name)
    issues = []
    if reviewed is not None and name not in reviewed:
        issues.append(
            "file is not on the reviewed release list (tools/release_files.txt)"
        )
    if p.suffix in BINARY or p.suffix in DATA_SUFFIXES or ".local." in name:
        issues.append("runtime/private artifact type")
    if p.suffix == ".json" and ".example." not in name:
        issues.append("JSON is only allowed as an .example. file")
    if len(content) > 512 * 1024:
        issues.append("unexpected large source file")
    if name in REVIEWED_IMAGE_ASSETS:
        if p.suffix not in IMAGE_SUFFIXES or not _image_magic(
            p.suffix, content
        ):
            issues.append("reviewed image asset has unexpected format")
        return issues
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return issues + ["binary/non-UTF-8 file"]
    for label, pattern in [
        ("credential-like value", TOKEN),
        ("machine GPU identity", GPU),
        ("private key material", PRIVATE_KEY),
        ("dataset case UUID", UUID),
        ("workstation path", WORKSTATION),
    ]:
        if pattern.search(text):
            issues.append(label)
    if any(
        not match.group(1).startswith("127.")
        for match in IP_PORT.finditer(text)
    ):
        issues.append("non-loopback IP endpoint")
    if p.suffix not in {".py"} and any(
        marker in text for marker in RESULT_MARKERS
    ):
        issues.append("experiment-result content in a non-code file")
    if any(word and word.lower() in text.lower() for word in private_words):
        issues.append("operator-supplied private marker")
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracked",
        action="store_true",
        help="Check the full index, including unchanged files",
    )
    parser.add_argument(
        "--private-word",
        action="append",
        default=[],
        help="Reject this private marker in file contents; may be repeated",
    )
    args = parser.parse_args()
    command = (
        ["git", "ls-files", "-z"]
        if args.tracked
        else [
            "git",
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=ACMR",
            "-z",
        ]
    )
    names = [
        name
        for name in subprocess.check_output(command, cwd=ROOT)
        .decode()
        .split("\0")
        if name
    ]
    reviewed = release_files()
    count, failed = 0, False
    for name in sorted(reviewed - set(names)):
        print(name + ": on the reviewed release list but not staged/tracked")
        failed = True
    for name in filter(None, names):
        # Inspect index bytes, not a potentially different working copy.
        mode = (
            subprocess.check_output(
                ["git", "ls-files", "-s", "--", name], cwd=ROOT
            )
            .decode()
            .split()[0]
        )
        content = subprocess.check_output(["git", "show", ":" + name], cwd=ROOT)
        issues = check(name, content, args.private_word, reviewed)
        if mode not in {"100644", "100755"}:
            issues.append("non-regular tracked file")
        if issues:
            print(name + ": " + "; ".join(issues))
            failed = True
        count += 1
    print(
        str(count)
        + " source files checked; "
        + ("FAILED" if failed else "passed")
    )
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()
