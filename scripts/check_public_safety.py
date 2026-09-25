#!/usr/bin/env python3
"""Guard the public repository against leaking private data.

Checks every file git tracks (including staged-but-uncommitted files):

1. Forbidden paths — private inventories, real-data exports, `.env` files,
   local AI-assistant guidance, internal docs.
2. metric_id references — every `XXX_0000`-style id in tracked text must
   exist in the committed example inventory; an unknown id usually means a
   row of a private inventory was pasted in.
3. Secrets — API keys, private-key blocks, non-empty `AD_LLM_API_KEY=`.
4. Private-inventory names — if a private inventory is present locally
   (`AD_REGISTRY_CSV` or `data/registry/metrics.csv`), none of its `name_cn`
   values that are absent from the example may appear in tracked files.
5. Local denylist — optional, gitignored `.public-safety-denylist`, one term
   per line (`#` comments allowed), for terms that must never be published.

Exit status is non-zero when any check fails. Standard library only, so it
runs in pre-commit and CI without installing the project.
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_INVENTORY = ROOT / "data/registry/metrics.example.csv"
PRIVATE_INVENTORY_DEFAULT = ROOT / "data/registry/metrics.csv"
DENYLIST = ROOT / ".public-safety-denylist"

FORBIDDEN_PATHS = [
    (re.compile(r"^data/registry/(?!metrics\.example\.csv$).+\.csv$"), "private inventory"),
    (re.compile(r"(^|/)(原子)?指标清单[^/]*\.csv$"), "private inventory"),
    (re.compile(r"(^|/)\.env(\.[^/]*)?$(?<!\.env\.example)"), "environment file"),
    (re.compile(r"(^|/)CLAUDE(\.local)?\.md$"), "local AI-assistant guidance"),
    (re.compile(r"^docs/"), "internal docs"),
    (re.compile(r"^data/(raw|private|prod|real)/"), "real business data"),
    (re.compile(r"\.(xlsx|xls|dta|sav|db|sqlite3?)$"), "data export"),
    (re.compile(r"^(?!tests/fixtures/).*\.parquet$"), "data export"),
]

METRIC_ID = re.compile(r"\b[A-Z]{2,5}_\d{4}\b")
SECRETS = [
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"), "an API key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "a private key"),
    (
        re.compile(r"^[ \t]*AD_LLM_API_KEY[ \t]*=[ \t]*\S+", re.MULTILINE),
        "a non-empty AD_LLM_API_KEY",
    ),
]
# Private-inventory names shorter than this are too generic to flag.
MIN_PRIVATE_NAME_LEN = 4


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "-z"],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout.decode("utf-8")
    return [p for p in out.split("\0") if p]


def read_text(rel: str) -> str | None:
    path = ROOT / rel
    if not path.is_file():  # staged deletion
        return None
    data = path.read_bytes()
    if b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", errors="ignore")


def load_inventory(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def private_inventory_path() -> Path | None:
    env = os.environ.get("AD_REGISTRY_CSV")
    candidate = Path(env) if env else PRIVATE_INVENTORY_DEFAULT
    if candidate.is_file() and candidate.resolve() != EXAMPLE_INVENTORY.resolve():
        return candidate
    return None


def load_denylist() -> list[str]:
    if not DENYLIST.is_file():
        return []
    terms = []
    for line in DENYLIST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    return terms


def main() -> int:
    files = tracked_files()
    example = load_inventory(EXAMPLE_INVENTORY)
    example_ids = {row["metric_id"] for row in example}
    example_names = {row["name_cn"] for row in example}

    private_names: list[str] = []
    private_path = private_inventory_path()
    if private_path is not None:
        private_names = sorted(
            {
                row["name_cn"]
                for row in load_inventory(private_path)
                if row.get("name_cn")
                and row["name_cn"] not in example_names
                and len(row["name_cn"]) >= MIN_PRIVATE_NAME_LEN
            },
            key=len,
            reverse=True,
        )
    denylist = load_denylist()

    problems: list[str] = []
    for rel in files:
        for pattern, why in FORBIDDEN_PATHS:
            if pattern.search(rel):
                problems.append(f"{rel}: forbidden path ({why})")
                break

        if rel == "scripts/check_public_safety.py":
            continue  # holds the patterns themselves
        text = read_text(rel)
        if text is None:
            continue

        for mid in sorted(set(METRIC_ID.findall(text)) - example_ids):
            problems.append(f"{rel}: metric_id {mid} is not in the example inventory")
        for pattern, why in SECRETS:
            if pattern.search(text):
                problems.append(f"{rel}: looks like {why}")
        # Report private-inventory hits without echoing the term itself, so the
        # check's own output is safe to paste into a CI log or issue.
        hits = sum(1 for name in private_names if name in text)
        if hits:
            problems.append(f"{rel}: contains {hits} field name(s) from the private inventory")
        hits = sum(1 for term in denylist if term in text)
        if hits:
            problems.append(f"{rel}: contains {hits} term(s) from .public-safety-denylist")

    if problems:
        print("Public-safety check FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    extras = []
    if private_path is not None:
        extras.append("private-inventory names")
    if denylist:
        extras.append("local denylist")
    suffix = f" (+ {', '.join(extras)})" if extras else ""
    print(f"Public-safety check passed: {len(files)} tracked files{suffix}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
