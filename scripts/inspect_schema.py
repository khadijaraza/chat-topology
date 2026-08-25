#!/usr/bin/env python3
"""Introspect raw chat export JSON without assuming a platform schema."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

MAX_STRING_LEN = 80
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Substrings that suggest a list holds conversations or chat-like records.
COLLECTION_HINTS = (
    "conversation",
    "conversations",
    "chat",
    "chats",
    "message",
    "messages",
    "thread",
    "threads",
    "activity",
    "activities",
    "session",
    "sessions",
    "dialog",
    "dialogs",
    "history",
    "items",
    "records",
    "data",
)


def truncate_string(value: str) -> str:
    if len(value) <= MAX_STRING_LEN:
        return value
    return value[: MAX_STRING_LEN - 3] + "..."


def summarize(value: Any, indent: int = 0) -> list[str]:
    """Return lines describing value shape (keys/types), truncating long strings."""
    prefix = "  " * indent
    lines: list[str] = []

    if value is None:
        lines.append(f"{prefix}(null)")
    elif isinstance(value, bool):
        lines.append(f"{prefix}(bool) {value}")
    elif isinstance(value, (int, float)):
        lines.append(f"{prefix}({type(value).__name__}) {value}")
    elif isinstance(value, str):
        lines.append(f"{prefix}(str) {truncate_string(value)!r}")
    elif isinstance(value, list):
        lines.append(f"{prefix}(list) len={len(value)}")
        if value:
            lines.append(f"{prefix}  [0]:")
            lines.extend(summarize(value[0], indent + 2))
        else:
            lines.append(f"{prefix}  (empty)")
    elif isinstance(value, dict):
        lines.append(f"{prefix}(dict) keys={list(value.keys())}")
        for key, nested in value.items():
            lines.append(f"{prefix}  {key!r}:")
            lines.extend(summarize(nested, indent + 2))
    else:
        lines.append(f"{prefix}({type(value).__name__}) {value!r}")

    return lines


def find_json_files(platform_dir: Path) -> list[Path]:
    return sorted(platform_dir.rglob("*.json"))


def collection_score(key: str | None, items: list[Any]) -> tuple[int, int]:
    """Higher is better: hint match in key name, then list length."""
    hint_score = 0
    if key is not None:
        lower = key.lower()
        for hint in COLLECTION_HINTS:
            if hint in lower:
                hint_score = max(hint_score, len(hint))
    return hint_score, len(items)


def find_collections(root: Any, path: str = "root") -> list[tuple[str, list[Any]]]:
    """Find list-valued fields that may hold conversation-like records."""
    collections: list[tuple[str, list[Any]]] = []

    if isinstance(root, list):
        collections.append((path, root))
        return collections

    if isinstance(root, dict):
        for key, value in root.items():
            child_path = f"{path}.{key}" if path != "root" else key
            if isinstance(value, list):
                collections.append((child_path, value))
            elif isinstance(value, dict):
                collections.extend(find_collections(value, child_path))

    return collections


def pick_primary_collection(collections: list[tuple[str, list[Any]]]) -> tuple[str, list[Any]] | None:
    if not collections:
        return None
    return max(collections, key=lambda item: collection_score(item[0], item[1]))


def print_top_level(root: Any) -> None:
    if isinstance(root, dict):
        print("Top-level keys:", list(root.keys()))
        print(f"Root type: dict ({len(root)} keys)")
    elif isinstance(root, list):
        print("Top-level keys: (root is a JSON array, not an object)")
        print(f"Root type: list (len={len(root)})")
    else:
        print(f"Root type: {type(root).__name__}")


def inspect_platform(platform: str, platform_dir: Path) -> None:
    print("=" * 72)
    print(f"Platform: {platform}")
    print(f"Directory: {platform_dir}")
    print("=" * 72)

    json_files = find_json_files(platform_dir)
    if not json_files:
        print("No JSON files found.\n")
        return

    json_path = json_files[0]
    if len(json_files) > 1:
        print(f"Using first JSON file ({len(json_files)} total): {json_path.name}")
    else:
        print(f"File: {json_path.name}")

    try:
        with json_path.open(encoding="utf-8") as handle:
            root = json.load(handle)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Failed to parse JSON: {exc}\n")
        return

    print()
    print_top_level(root)
    print()

    collections = find_collections(root)
    primary = pick_primary_collection(collections)

    if primary is None:
        print("No list-valued collections found in this file.")
        print()
        print("Root structure:")
        for line in summarize(root):
            print(line)
        print()
        return

    collection_path, items = primary
    print(f"Primary collection (heuristic): {collection_path!r}")
    print(f"Total entries in collection: {len(items)}")
    print()

    if not items:
        print("Collection is empty — no first entry to show.")
        print()
        return

    print("First entry structure:")
    for line in summarize(items[0]):
        print(line)
    print()

    if len(collections) > 1:
        print("Other list collections in file:")
        for path, coll in sorted(collections, key=lambda x: (-len(x[1]), x[0])):
            marker = "  (primary)" if path == collection_path else ""
            print(f"  - {path!r}: {len(coll)} items{marker}")
        print()


def main() -> int:
    if not RAW_DIR.is_dir():
        print(f"ERROR: Raw data directory not found: {RAW_DIR}", file=sys.stderr)
        return 1

    platform_dirs = sorted(
        path for path in RAW_DIR.iterdir() if path.is_dir() and not path.name.startswith(".")
    )

    if not platform_dirs:
        print(f"No platform folders found under {RAW_DIR}")
        return 0

    for platform_dir in platform_dirs:
        inspect_platform(platform_dir.name, platform_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
