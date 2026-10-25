import json
from datetime import date
from typing import Any


def normalize_keywords(keywords: list[str] | None) -> list[str]:
    normalized: dict[str, str] = {}
    for keyword in keywords or []:
        cleaned = keyword.strip()
        if cleaned:
            normalized.setdefault(cleaned.casefold(), cleaned)
    return list(normalized.values())


def serialize_keywords(keywords: list[str] | None) -> str:
    return json.dumps(normalize_keywords(keywords), ensure_ascii=False)


def parse_keywords(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return normalize_keywords([str(item) for item in value])
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value.split(",")
    if isinstance(parsed, list):
        return normalize_keywords([str(item) for item in parsed])
    return []


def build_metadata_where(
    *,
    course_id: str | None = None,
    author: str | None = None
) -> dict | None:
    clauses: list[dict] = []
    if course_id is not None:
        clauses.append({"course_id": course_id})
    if author:
        clauses.append({"author": author})

    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def matches_publish_date(
    metadata: dict,
    publish_date_from: date | str | None = None,
    publish_date_to: date | str | None = None
) -> bool:
    if publish_date_from is None and publish_date_to is None:
        return True
    value = metadata.get("publish_date")
    if not value:
        return False
    value = str(value)[:10]
    if publish_date_from is not None:
        lower = publish_date_from.isoformat() if isinstance(publish_date_from, date) else str(publish_date_from)
        if value < lower[:10]:
            return False
    if publish_date_to is not None:
        upper = publish_date_to.isoformat() if isinstance(publish_date_to, date) else str(publish_date_to)
        if value > upper[:10]:
            return False
    return True


def matches_keywords(metadata: dict, requested: list[str] | None) -> bool:
    required = {keyword.casefold() for keyword in normalize_keywords(requested)}
    if not required:
        return True
    available = {keyword.casefold() for keyword in parse_keywords(metadata.get("keywords"))}
    # A document must have every requested keyword.
    return required.issubset(available)
