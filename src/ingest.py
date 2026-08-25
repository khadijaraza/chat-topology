"""Ingest raw platform exports into a normalized conversation schema."""

from __future__ import annotations

import html
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

Platform = Literal["claude", "chatgpt", "gemini"]

PROMPTED_PREFIX = re.compile(r"^Prompted\s+", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class IngestStats:
    loaded_by_platform: dict[str, int] = field(default_factory=dict)
    skipped_by_platform: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_loaded(self) -> int:
        return sum(self.loaded_by_platform.values())

    @property
    def total_skipped(self) -> int:
        return sum(self.skipped_by_platform.values())


_last_ingest_stats: IngestStats | None = None


def get_last_ingest_stats() -> IngestStats:
    if _last_ingest_stats is None:
        return IngestStats()
    return _last_ingest_stats


def html_to_text(raw_html: str) -> str:
    """Strip HTML tags and decode entities to plain text."""
    text = HTML_TAG_RE.sub(" ", raw_html)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_timestamp(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_timestamp(value: Any) -> datetime | None:
    iso = normalize_timestamp(value)
    if iso is None:
        return None
    if iso.endswith("Z"):
        iso = iso[:-1] + "+00:00"
    return datetime.fromisoformat(iso)


def extract_chat_id(entry: dict[str, Any]) -> str | None:
    """Derive a session id for grouping turns into one conversation.

    The export's ``header`` field is always the product name (e.g. "Gemini Apps").
    Conversations are grouped by the Gemini app URL in ``details``.
    """
    details = entry.get("details")
    if not isinstance(details, list) or not details:
        return None
    first = details[0]
    if not isinstance(first, dict):
        return None
    url = first.get("url") or first.get("name")
    if not isinstance(url, str) or not url.strip():
        return None
    url = url.strip()
    if "/app/" in url:
        return url.rsplit("/app/", 1)[-1].split("?", 1)[0]
    return url


def extract_user_text(entry: dict[str, Any]) -> str | None:
    title = entry.get("title")
    if not isinstance(title, str) or not title.strip():
        return None
    text = PROMPTED_PREFIX.sub("", title.strip()).strip()
    return text or None


def extract_assistant_text(entry: dict[str, Any]) -> str | None:
    items = entry.get("safeHtmlItem")
    if not isinstance(items, list) or not items:
        return None
    chunks: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_html = item.get("html")
        if not isinstance(raw_html, str) or not raw_html.strip():
            continue
        text = html_to_text(raw_html)
        if text:
            chunks.append(text)
    if not chunks:
        return None
    return "\n\n".join(chunks)


def parse_gemini_entry(
    entry: Any,
    *,
    source_file: str,
    entry_index: int,
    stats: IngestStats,
) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        stats.skipped_by_platform["gemini"] = stats.skipped_by_platform.get("gemini", 0) + 1
        msg = f"gemini:{source_file}[{entry_index}] skipped: entry is not an object"
        stats.warnings.append(msg)
        logger.warning(msg)
        return None

    chat_id = extract_chat_id(entry)
    timestamp = normalize_timestamp(entry.get("time"))
    user_text = extract_user_text(entry)
    assistant_text = extract_assistant_text(entry)

    missing: list[str] = []
    if not chat_id:
        missing.append("chat_id (from details.url)")
    if not timestamp:
        missing.append("timestamp (time)")
    if not user_text:
        missing.append("user text (title)")
    if not assistant_text:
        missing.append("assistant text (safeHtmlItem)")

    if missing:
        stats.skipped_by_platform["gemini"] = stats.skipped_by_platform.get("gemini", 0) + 1
        msg = f"gemini:{source_file}[{entry_index}] skipped: missing {', '.join(missing)}"
        stats.warnings.append(msg)
        logger.warning(msg)
        return None

    parsed_time = parse_timestamp(entry.get("time"))
    return {
        "chat_id": chat_id,
        "timestamp": timestamp,
        "_sort_time": parsed_time,
        "user_text": user_text,
        "assistant_text": assistant_text,
    }


def build_gemini_conversations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["chat_id"]].append(record)

    conversations: list[dict[str, Any]] = []
    for chat_id, entries in grouped.items():
        entries.sort(
            key=lambda item: item["_sort_time"] or datetime.min.replace(tzinfo=timezone.utc)
        )

        turns: list[dict[str, Any]] = []
        turn_index = 0
        for entry in entries:
            turns.append(
                {"role": "user", "text": entry["user_text"], "turn_index": turn_index}
            )
            turn_index += 1
            turns.append(
                {
                    "role": "assistant",
                    "text": entry["assistant_text"],
                    "turn_index": turn_index,
                }
            )
            turn_index += 1

        conversations.append(
            {
                "chat_id": chat_id,
                "platform": "gemini",
                "timestamp": entries[0]["timestamp"],
                "turns": turns,
            }
        )

    conversations.sort(key=lambda conv: conv["timestamp"])
    return conversations


def parse_gemini_export(path: Path, stats: IngestStats) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        stats.skipped_by_platform["gemini"] = stats.skipped_by_platform.get("gemini", 0) + 1
        msg = f"gemini:{path.name} skipped: failed to load JSON ({exc})"
        stats.warnings.append(msg)
        logger.warning(msg)
        return []

    if not isinstance(payload, list):
        stats.skipped_by_platform["gemini"] = stats.skipped_by_platform.get("gemini", 0) + 1
        msg = f"gemini:{path.name} skipped: expected top-level JSON array"
        stats.warnings.append(msg)
        logger.warning(msg)
        return []

    parsed_records: list[dict[str, Any]] = []
    for index, entry in enumerate(payload):
        record = parse_gemini_entry(
            entry, source_file=path.name, entry_index=index, stats=stats
        )
        if record is not None:
            parsed_records.append(record)

    conversations = build_gemini_conversations(parsed_records)
    stats.loaded_by_platform["gemini"] = stats.loaded_by_platform.get("gemini", 0) + len(
        conversations
    )
    return conversations


def load_all_conversations(raw_dir: str = "data/raw") -> list[dict]:
    """Load and normalize conversations from all supported platform exports."""
    global _last_ingest_stats

    raw_path = Path(raw_dir)
    stats = IngestStats()
    conversations: list[dict[str, Any]] = []

    gemini_dir = raw_path / "gemini"
    if gemini_dir.is_dir():
        for json_path in sorted(gemini_dir.rglob("*.json")):
            conversations.extend(parse_gemini_export(json_path, stats))

    output_path = Path("data/processed/conversations.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(conversations, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    _last_ingest_stats = stats
    logger.info("Wrote %d conversations to %s", len(conversations), output_path)
    return conversations
