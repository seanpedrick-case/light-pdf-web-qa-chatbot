"""
LLM-generated Amazon Bedrock Knowledge Base RetrievalFilter helpers.

When USE_KB_METADATA_FILTER is enabled, proposes a filter from the user
question via Bedrock Converse, then validates and sanitizes before retrieval.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, FrozenSet, Literal, Optional

from tools.config import (
    KB_DATE_FILTER_STRATEGY,
    KB_DATE_TO_YEAR_MAP_DICT,
    KB_YEAR_COLUMNS_INDEXED,
    KB_YEAR_METADATA_KEY_SET,
)

DateFilterStrategy = Literal[
    "semantic_only", "iso_range", "numeric_yyyymmdd", "year_in"
]


ISO8601_OFFSET_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$"
)
_CALENDAR_YEAR_RE = re.compile(r"^\d{4}$")
_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_YYYYMMDD_RE = re.compile(r"^\d{8}$")

# S3 Vectors knowledge bases do not support startsWith — align with Lambda filter ops only.
_LEAF_OPS = frozenset(
    {
        "in",
        "notIn",
        "equals",
        "notEquals",
        "greaterThan",
        "greaterThanOrEquals",
        "lessThan",
        "lessThanOrEquals",
    }
)
_RANGE_OPS = frozenset(
    {
        "greaterThan",
        "greaterThanOrEquals",
        "lessThan",
        "lessThanOrEquals",
    }
)
_BOOL_OPS = frozenset({"andAll", "orAll"})


def _year_bounds_iso(year: int) -> tuple[str, str]:
    start = f"{year:04d}-01-01T00:00:00Z"
    end_exclusive = f"{year + 1:04d}-01-01T00:00:00Z"
    return start, end_exclusive


def _coerce_to_iso_datetime(
    metadata_key: str, val: Any, date_keys: FrozenSet[str]
) -> Any:
    """Normalize date-like filter scalars to ISO-8601 strings for Bedrock date keys."""
    if metadata_key not in date_keys:
        return val
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        year = int(val)
        if 1900 <= year <= 2100:
            return _year_bounds_iso(year)[0]
        return val
    if not isinstance(val, str) or not val.strip():
        return val
    text = val.strip()
    if _CALENDAR_YEAR_RE.match(text):
        return _year_bounds_iso(int(text))[0]
    if _DATE_ONLY_RE.match(text):
        return f"{text}T00:00:00Z"
    return val


def normalized_date_filter_strategy(override: str | None = None) -> DateFilterStrategy:
    raw = (override or KB_DATE_FILTER_STRATEGY or "semantic_only").strip().lower()
    if raw in ("iso_range", "iso", "lambda"):
        return "iso_range"
    if raw in ("numeric_yyyymmdd", "yyyymmdd", "numeric", "s3_vectors_number"):
        return "numeric_yyyymmdd"
    if raw in ("year_in", "year_only", "year", "calendar_year"):
        return "year_in"
    return "semantic_only"


def resolve_date_to_year_map(
    date_keys: FrozenSet[str],
    year_keys: FrozenSet[str],
) -> dict[str, str]:
    return {
        k: v
        for k, v in KB_DATE_TO_YEAR_MAP_DICT.items()
        if k in date_keys and v in year_keys
    }


def _year_string_from_date_value(val: Any) -> Optional[str]:
    year = _extract_calendar_year(val)
    if year is not None:
        return f"{year:04d}"
    if isinstance(val, str):
        text = val.strip()
        if _YYYYMMDD_RE.match(text):
            return text[:4]
    return None


def _is_valid_range_filter_value(
    metadata_key: str,
    val: Any,
    date_keys: FrozenSet[str],
    *,
    strategy: DateFilterStrategy | None = None,
) -> bool:
    strategy = strategy or normalized_date_filter_strategy()
    if metadata_key in date_keys:
        if strategy == "numeric_yyyymmdd":
            if isinstance(val, bool):
                return False
            if isinstance(val, int):
                return 19000101 <= val <= 21001231
            return False
        if strategy == "iso_range":
            if isinstance(val, str) and val:
                return bool(ISO8601_OFFSET_DATETIME_RE.match(val))
            return False
        if strategy == "year_in":
            return False
        return False
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return True
    return False


def _calendar_year_range_clause(key: str, year: int) -> dict:
    start, end = _year_bounds_iso(year)
    return {
        "andAll": [
            {"greaterThanOrEquals": {"key": key, "value": start}},
            {"lessThan": {"key": key, "value": end}},
        ]
    }


def _calendar_year_range_clause_numeric(key: str, year: int) -> dict:
    start = year * 10000 + 101
    end = (year + 1) * 10000 + 101
    return {
        "andAll": [
            {"greaterThanOrEquals": {"key": key, "value": start}},
            {"lessThan": {"key": key, "value": end}},
        ]
    }


def _year_columns_indexed_on_kb() -> bool:
    return (KB_YEAR_COLUMNS_INDEXED or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def strip_year_metadata_clauses(
    node: Any,
    year_keys: FrozenSet[str],
    depth: int = 0,
) -> Any:
    """Drop filters on Year* keys when those columns are not on the KB index yet."""
    if depth > 12 or not year_keys or not isinstance(node, dict) or len(node) != 1:
        return node
    op = next(iter(node))
    if op in _BOOL_OPS:
        inner = node[op]
        if not isinstance(inner, list):
            return node
        cleaned = [strip_year_metadata_clauses(x, year_keys, depth + 1) for x in inner]
        cleaned = [x for x in cleaned if x is not None]
        if not cleaned:
            return None
        if len(cleaned) == 1:
            return cleaned[0]
        return {op: cleaned}
    if op in _LEAF_OPS or op == "startsWith":
        spec = node.get(op)
        if not isinstance(spec, dict):
            return node
        key = spec.get("key")
        if key in year_keys:
            print(
                f"Metadata filter: dropped {op} on year key {key!r} "
                "(KB_YEAR_COLUMNS_INDEXED=0 or Year* not on index)"
            )
            return None
    return node


def strip_date_metadata_clauses(
    node: Any,
    date_keys: FrozenSet[str],
    depth: int = 0,
) -> Any:
    """S3 Vectors: string date metadata cannot use range — drop date clauses, keep text filters."""
    if depth > 12 or not isinstance(node, dict) or len(node) != 1:
        return node
    op = next(iter(node))
    if op in _BOOL_OPS:
        inner = node[op]
        if not isinstance(inner, list):
            return node
        cleaned = [strip_date_metadata_clauses(x, date_keys, depth + 1) for x in inner]
        cleaned = [x for x in cleaned if x is not None]
        if not cleaned:
            return None
        if len(cleaned) == 1:
            return cleaned[0]
        return {op: cleaned}
    if op in _LEAF_OPS or op == "startsWith":
        spec = node.get(op)
        if not isinstance(spec, dict):
            return node
        key = spec.get("key")
        if key in date_keys:
            print(
                f"Metadata filter: dropped {op} on date key {key!r} "
                "(date strategy does not use day-level metadata filters)"
            )
            return None
    return node


def rewrite_filters_to_year_in(
    node: Any,
    date_keys: FrozenSet[str],
    date_to_year: dict[str, str],
    allowed_keys: FrozenSet[str],
    depth: int = 0,
) -> Any:
    """Map calendar-year intent on date keys to in-filter on Year* string columns; drop ranges."""
    if depth > 12 or not date_to_year:
        return node
    if not isinstance(node, dict) or len(node) != 1:
        return node
    op = next(iter(node))
    if op in _BOOL_OPS:
        inner = node[op]
        if not isinstance(inner, list):
            return node
        rewritten = [
            rewrite_filters_to_year_in(
                x, date_keys, date_to_year, allowed_keys, depth + 1
            )
            for x in inner
        ]
        rewritten = [x for x in rewritten if x is not None]
        if not rewritten:
            return None
        if len(rewritten) == 1:
            return rewritten[0]
        return {op: rewritten}
    if op in _RANGE_OPS:
        spec = node[op]
        if isinstance(spec, dict) and spec.get("key") in date_keys:
            print(
                f"Metadata filter: dropped day-level range on {spec.get('key')!r} "
                "(KB_DATE_FILTER_STRATEGY=year_in)"
            )
            return None
        return node
    if op in ("equals", "in", "notIn"):
        spec = node[op]
        if not isinstance(spec, dict):
            return node
        key = spec.get("key")
        val = spec.get("value")
        if key not in date_keys:
            return node
        year_key = date_to_year.get(key)
        if not year_key or year_key not in allowed_keys:
            return None
        if op == "equals":
            y = _year_string_from_date_value(val)
            if y:
                return {"in": {"key": year_key, "value": [y]}}
            return None
        if op in ("in", "notIn") and isinstance(val, list):
            years: list[str] = []
            for item in val:
                y = _year_string_from_date_value(item)
                if y:
                    years.append(y)
            if not years:
                return None
            target = "notIn" if op == "notIn" else "in"
            return {target: {"key": year_key, "value": years}}
    return node


def _extract_calendar_year(val: Any) -> Optional[int]:
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        year = int(val)
        if 1900 <= year <= 2100:
            return year
        return None
    if isinstance(val, str) and _CALENDAR_YEAR_RE.match(val.strip()):
        return int(val.strip())
    return None


_DEFAULT_EXACT_EQUAL_KEYS = frozenset({"DocumentId"})


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_filter_options_path(path: str) -> Path:
    raw = (path or "").strip()
    if not raw:
        return Path()
    candidate = Path(raw)
    if candidate.is_file():
        return candidate
    return _repo_root() / raw


def load_filter_group_options(path: str) -> frozenset[str]:
    """Load allow-listed group labels (one per line; # comments and blank lines ignored)."""
    resolved = resolve_filter_options_path(path)
    if not resolved.is_file():
        print(f"Metadata filter group options file not found: {resolved}")
        return frozenset()
    values: list[str] = []
    for line in resolved.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if text:
            values.append(text)
    return frozenset(values)


def load_filter_enum_options(path: str) -> frozenset[str]:
    """Deprecated alias for load_filter_group_options."""
    return load_filter_group_options(path)


def _format_group_options_for_prompt(options: FrozenSet[str]) -> str:
    if not options:
        return "(none configured)"
    return "\n".join(f"- {o}" for o in sorted(options))


def constrain_filter_group_values(
    node: Any,
    group_filter_key: str,
    allowed_groups: FrozenSet[str],
    depth: int = 0,
) -> Any:
    """Keep only allow-listed strings on in/notIn for the group metadata key."""
    if depth > 12 or not group_filter_key or not allowed_groups:
        return node
    if not isinstance(node, dict) or len(node) != 1:
        return node
    op = next(iter(node))
    if op in _BOOL_OPS:
        inner = node[op]
        if not isinstance(inner, list):
            return node
        cleaned = []
        for x in inner:
            c = constrain_filter_group_values(
                x, group_filter_key, allowed_groups, depth + 1
            )
            if c is not None:
                cleaned.append(c)
        if not cleaned:
            return None
        if len(cleaned) == 1:
            return cleaned[0]
        return {op: cleaned}
    if op in ("in", "notIn"):
        spec = node[op]
        if not isinstance(spec, dict):
            return node
        key = spec.get("key")
        val = spec.get("value")
        if key != group_filter_key or not isinstance(val, list):
            return node
        kept = [v for v in val if isinstance(v, str) and v in allowed_groups]
        dropped = [v for v in val if v not in kept]
        if dropped:
            print(
                f"Metadata filter: dropped non-allow-listed group values on "
                f"{group_filter_key!r}: {dropped!r}"
            )
        if not kept:
            return None
        return {op: {"key": key, "value": kept}}
    if op == "equals" and isinstance(node.get("equals"), dict):
        spec = node["equals"]
        if spec.get("key") == group_filter_key:
            v = spec.get("value")
            if v not in allowed_groups:
                print(
                    f"Metadata filter: dropped equals on group key "
                    f"{group_filter_key!r} (not in allow-list): {v!r}"
                )
                return None
    return node


def constrain_filter_enum_values(
    node: Any,
    enum_key: str,
    allowed: FrozenSet[str],
    depth: int = 0,
) -> Any:
    """Deprecated alias for constrain_filter_group_values."""
    return constrain_filter_group_values(node, enum_key, allowed, depth)


def resolve_in_metadata_keys(
    allowed_keys: FrozenSet[str],
    date_keys: FrozenSet[str],
    *,
    in_keys_override: FrozenSet[str] | None = None,
    exact_equal_keys: FrozenSet[str] | None = None,
) -> FrozenSet[str]:
    exact = exact_equal_keys or _DEFAULT_EXACT_EQUAL_KEYS
    if in_keys_override:
        return frozenset(k for k in in_keys_override if k in allowed_keys)
    return frozenset(allowed_keys - date_keys - exact)


def rewrite_text_equals_to_in(
    node: Any,
    in_keys: FrozenSet[str],
    depth: int = 0,
) -> Any:
    """Use in/notIn for categorical text — the LLM rarely knows exact stored labels."""
    if depth > 12 or not isinstance(node, dict) or len(node) != 1:
        return node
    op = next(iter(node))
    if op in _BOOL_OPS:
        inner = node[op]
        if not isinstance(inner, list):
            return node
        rewritten = [rewrite_text_equals_to_in(x, in_keys, depth + 1) for x in inner]
        return {op: rewritten}
    if op == "equals":
        spec = node[op]
        if not isinstance(spec, dict):
            return node
        key = spec.get("key")
        val = spec.get("value")
        if key in in_keys and val is not None:
            values = val if isinstance(val, list) else [val]
            print(
                f"Metadata filter: promoted equals to in for text key {key!r}: {values!r}"
            )
            return {"in": {"key": key, "value": values}}
    if op == "notEquals":
        spec = node[op]
        if not isinstance(spec, dict):
            return node
        key = spec.get("key")
        val = spec.get("value")
        if key in in_keys and val is not None:
            values = val if isinstance(val, list) else [val]
            return {"notIn": {"key": key, "value": values}}
    return node


def rewrite_retrieval_filter_dates(
    node: Any,
    date_keys: FrozenSet[str],
    depth: int = 0,
) -> Any:
    """Expand year-only / loose date filters into ISO range clauses (Lambda-compatible)."""
    if depth > 12 or not isinstance(node, dict) or len(node) != 1:
        return node
    op = next(iter(node))
    if op in _BOOL_OPS:
        inner = node[op]
        if not isinstance(inner, list):
            return node
        rewritten = [
            rewrite_retrieval_filter_dates(x, date_keys, depth + 1) for x in inner
        ]
        rewritten = [x for x in rewritten if x is not None]
        if not rewritten:
            return None
        if len(rewritten) == 1:
            return rewritten[0]
        return {op: rewritten}
    if op == "startsWith":
        spec = node.get(op)
        if not isinstance(spec, dict):
            return node
        key, val = spec.get("key"), spec.get("value")
        if key in date_keys:
            year = _extract_calendar_year(val)
            if year is not None:
                print(
                    f"Metadata filter: replaced unsupported startsWith on {key!r} "
                    f"with ISO year range for {year}"
                )
                return _calendar_year_range_clause(str(key), year)
        print(f"Metadata filter: dropped unsupported startsWith clause: {node}")
        return None
    if op in _LEAF_OPS:
        spec = node.get(op)
        if not isinstance(spec, dict):
            return node
        key = spec.get("key")
        val = spec.get("value")
        if key not in date_keys:
            return node
        year = _extract_calendar_year(val)
        if op in ("equals", "notEquals") and year is not None:
            if op == "equals":
                return _calendar_year_range_clause(str(key), year)
            return node
        if op in _RANGE_OPS:
            iso_val = _coerce_to_iso_datetime(str(key), val, date_keys)
            if iso_val != val:
                return {op: {"key": key, "value": iso_val}}
            if year is not None and op in ("greaterThanOrEquals", "greaterThan"):
                start, _ = _year_bounds_iso(year)
                return {"greaterThanOrEquals": {"key": key, "value": start}}
            if year is not None and op in ("lessThan", "lessThanOrEquals"):
                _, end = _year_bounds_iso(year)
                return {"lessThan": {"key": key, "value": end}}
        return node
    return node


def rewrite_retrieval_filter_dates_numeric(
    node: Any,
    date_keys: FrozenSet[str],
    depth: int = 0,
) -> Any:
    """Year / loose date filters as integer YYYYMMDD bounds (S3 Vectors NUMBER metadata)."""
    if depth > 12 or not isinstance(node, dict) or len(node) != 1:
        return node
    op = next(iter(node))
    if op in _BOOL_OPS:
        inner = node[op]
        if not isinstance(inner, list):
            return node
        rewritten = [
            rewrite_retrieval_filter_dates_numeric(x, date_keys, depth + 1)
            for x in inner
        ]
        rewritten = [x for x in rewritten if x is not None]
        if not rewritten:
            return None
        if len(rewritten) == 1:
            return rewritten[0]
        return {op: rewritten}
    if op in _LEAF_OPS:
        spec = node.get(op)
        if not isinstance(spec, dict):
            return node
        key = spec.get("key")
        val = spec.get("value")
        if key not in date_keys:
            return node
        year = _extract_calendar_year(val)
        if op in ("equals", "notEquals") and year is not None and op == "equals":
            return _calendar_year_range_clause_numeric(str(key), year)
        if op in _RANGE_OPS and year is not None:
            if op in ("greaterThanOrEquals", "greaterThan"):
                return {
                    "greaterThanOrEquals": {
                        "key": key,
                        "value": year * 10000 + 101,
                    }
                }
            if op in ("lessThan", "lessThanOrEquals"):
                return {"lessThan": {"key": key, "value": (year + 1) * 10000 + 101}}
        if op in _RANGE_OPS and isinstance(val, int):
            return node
    return node


def _valid_scalar(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool)) and v != ""


def validate_retrieval_filter(
    node: Any,
    allowed_keys: FrozenSet[str],
    date_keys: FrozenSet[str],
    depth: int = 0,
    *,
    strategy: DateFilterStrategy | None = None,
) -> bool:
    strategy = strategy or normalized_date_filter_strategy()
    if depth > 10 or not isinstance(node, dict) or len(node) != 1:
        return False
    op = next(iter(node))
    if op == "startsWith":
        return False
    if op in _BOOL_OPS:
        inner = node[op]
        if not isinstance(inner, list) or not inner or len(inner) > 12:
            return False
        return all(
            validate_retrieval_filter(
                x, allowed_keys, date_keys, depth + 1, strategy=strategy
            )
            for x in inner
        )
    if op in ("in", "notIn"):
        spec = node[op]
        if not isinstance(spec, dict):
            return False
        k, val = spec.get("key"), spec.get("value")
        if k not in allowed_keys:
            return False
        if not isinstance(val, list) or not val or len(val) > 25:
            return False
        return all(_valid_scalar(x) for x in val)
    if op in ("equals", "notEquals"):
        spec = node[op]
        if not isinstance(spec, dict):
            return False
        k, val = spec.get("key"), spec.get("value")
        if k not in allowed_keys or val is None:
            return False
        return _valid_scalar(val)
    if op in _RANGE_OPS:
        spec = node[op]
        if not isinstance(spec, dict):
            return False
        k, val = spec.get("key"), spec.get("value")
        if k not in allowed_keys or val is None:
            return False
        if not _valid_scalar(val):
            return False
        return _is_valid_range_filter_value(k, val, date_keys, strategy=strategy)
    return False


def _date_strategy_prompt_note(date_keys: FrozenSet[str]) -> str:
    strategy = normalized_date_filter_strategy()
    date_list = ", ".join(sorted(date_keys)) if date_keys else "(none)"
    if strategy == "semantic_only":
        return (
            f"\nDate columns ({date_list}): **do not** add date filters — this knowledge base "
            "uses S3 Vectors with string date metadata; range filters are not supported. "
            'Apply text filters only, or output {"filter": null} and rely on semantic '
            "search for the date aspect.\n"
        )
    if strategy == "numeric_yyyymmdd":
        return (
            f"\nDate columns ({date_list}) must be filtered with **integer YYYYMMDD** "
            "range operators (e.g. greaterThanOrEquals 20230101, lessThan 20240101). "
            "Do NOT use ISO strings unless KB_DATE_FILTER_STRATEGY=iso_range.\n"
        )
    if strategy == "year_in":
        year_cols = (
            ", ".join(sorted(KB_YEAR_METADATA_KEY_SET))
            or "(configure KB_YEAR_METADATA_KEYS)"
        )
        return (
            f"\nDo **not** use range operators on {date_list}. For calendar **year** only, "
            f"use **in** on year metadata keys ({year_cols}) with four-digit string values "
            '(e.g. {{"in":{{"key":"<YearKey>","value":["2023"]}}}}). '
            "Do not filter specific months or days.\n"
        )
    return (
        f"\nDate columns ({date_list}): use ISO-8601 with Z for range comparisons "
        "(greaterThanOrEquals / lessThan). Calendar year → andAll with "
        "YYYY-01-01T00:00:00Z and (YYYY+1)-01-01T00:00:00Z. No startsWith.\n"
    )


def _normalize_filter_leaf(op: str, spec: Any, allowed_keys: FrozenSet[str]) -> Any:
    if not isinstance(spec, dict) or not spec:
        return spec
    if spec.get("key") in allowed_keys and "value" in spec:
        return spec
    bad = set(spec.keys()) - {"key", "value"}
    if len(bad) == 1:
        k = next(iter(bad))
        if k in allowed_keys:
            v = spec[k]
            if op in ("in", "notIn"):
                if isinstance(v, list):
                    return {"key": k, "value": v}
            else:
                return {"key": k, "value": v}
    return spec


def normalize_retrieval_filter(
    node: Any,
    allowed_keys: FrozenSet[str],
    depth: int = 0,
) -> Any:
    if depth > 12 or not isinstance(node, dict) or len(node) != 1:
        return node
    op = next(iter(node))
    if op in _BOOL_OPS:
        inner = node[op]
        if not isinstance(inner, list):
            return node
        return {
            op: [normalize_retrieval_filter(x, allowed_keys, depth + 1) for x in inner]
        }
    if op in _LEAF_OPS:
        return {op: _normalize_filter_leaf(op, node[op], allowed_keys)}
    return node


def sanitize_retrieval_filter(
    node: Any,
    allowed_keys: FrozenSet[str],
    date_keys: FrozenSet[str],
    depth: int = 0,
    *,
    strategy: DateFilterStrategy | None = None,
) -> Optional[dict]:
    strategy = strategy or normalized_date_filter_strategy()
    if depth > 12 or not isinstance(node, dict) or len(node) != 1:
        return None
    op = next(iter(node))
    if op == "startsWith":
        print(f"Metadata filter dropped unsupported startsWith clause: {node}")
        return None
    if op in _BOOL_OPS:
        inner = node[op]
        if not isinstance(inner, list):
            return None
        cleaned = []
        for x in inner:
            c = sanitize_retrieval_filter(
                x, allowed_keys, date_keys, depth + 1, strategy=strategy
            )
            if c is not None:
                cleaned.append(c)
        if not cleaned:
            return None
        if len(cleaned) == 1:
            return cleaned[0]
        return {op: cleaned}
    if op in _RANGE_OPS:
        spec = node[op]
        if not isinstance(spec, dict):
            return None
        k, val = spec.get("key"), spec.get("value")
        if k not in allowed_keys or val is None:
            return None
        if not _is_valid_range_filter_value(k, val, date_keys, strategy=strategy):
            print(
                f"Metadata filter dropped unsupported range clause (key={k!r}): {node}"
            )
            return None
        return node
    if op in _LEAF_OPS:
        spec = node[op]
        if not isinstance(spec, dict):
            return None
        k, val = spec.get("key"), spec.get("value")
        if k not in allowed_keys:
            return None
        if op in ("in", "notIn"):
            if not isinstance(val, list) or not val:
                return None
        elif val is None:
            return None
        return node
    return None


def _inject_prompt_placeholders(
    text: str,
    *,
    group_filter_key: str,
    group_options: FrozenSet[str],
) -> str:
    options_block = _format_group_options_for_prompt(group_options)
    group_key = (group_filter_key or "").strip()
    out = text.replace("{{GROUP_OPTIONS}}", options_block).replace(
        "{{GROUP_FILTER_KEY}}", group_key
    )
    # Legacy placeholders (older custom prompts):
    return (
        out.replace("{{DIVISION_OPTIONS}}", options_block)
        .replace("{{ORG_FILTER_KEY}}", group_key)
        .replace("{{ENUM_OPTIONS}}", options_block)
    )


def _default_filter_prompt_template(
    allowed_keys: FrozenSet[str],
    date_keys: FrozenSet[str],
    *,
    group_filter_key: str = "",
    group_options: FrozenSet[str] | None = None,
) -> str:
    keys_line = ", ".join(sorted(allowed_keys))
    date_range_note = _date_strategy_prompt_note(date_keys) if date_keys else ""
    group_key = (group_filter_key or "").strip()
    group_options = group_options or frozenset()
    group_section = ""
    if group_key and group_key in allowed_keys and group_options:
        group_section = (
            f"\nWhen the user implies a broad category, filter on metadata key "
            f"**{group_key}** using **in** with one or more values taken exactly from "
            f"this group list:\n"
            f"{_format_group_options_for_prompt(group_options)}\n"
            "Map informal user wording to the closest list entries. "
            "Never invent group labels not in the list.\n"
        )
    return (
        "You help pre-filter a knowledge base. Output ONLY valid JSON (no markdown "
        'fences) with exactly this shape: {"filter": <object|null>}\n'
        "\n"
        'CRITICAL — structure for each operator. Every leaf condition MUST use "key" '
        'and "value" keys. Never put metadata field names as inner object keys.\n'
        '  Wrong: {"equals": {"Category": "Policy"}}\n'
        '  Right: {"equals": {"key": "Category", "value": "Policy"}}\n'
        '  Wrong: {"in": {"Status": ["Active", "Archived"]}}\n'
        '  Right: {"in": {"key": "Status", "value": ["Active", "Archived"]}}\n'
        '  Wrong: {"greaterThan": {"PublishedAt": "2026-01-01"}}\n'
        '  Right: {"greaterThan": {"key": "PublishedAt", "value": "2026-01-01T00:00:00Z"}} '
        "— range values for date keys must be full ISO-8601 with Z or offset.\n"
        f"{date_range_note}"
        "\n"
        "The filter object is an Amazon Bedrock RetrievalFilter — exactly one top-level "
        "key: equals, notEquals, in, notIn, andAll, orAll, greaterThan, "
        "greaterThanOrEquals, lessThan, or lessThanOrEquals. "
        "Do NOT use startsWith or stringContains.\n"
        "\n"
        f'Allowed metadata keys for "key" (exact spelling): {keys_line}.\n'
        f"{group_section}"
        "For other text/categorical fields, use **in** with exact stored labels when "
        "the user is specific. Use **equals** only for exact identifiers when the user "
        "gives a specific ID.\n"
        "Do not use embedded text column names as filter keys unless they appear in "
        "the allowed list above.\n"
        "\n"
        "Example combining conditions:\n"
        f'{{"filter":{{"andAll":[{{"in":{{"key":"{group_key or "Category"}",'
        f'"value":["Example group A","Example group B"]}}}},'
        '{"equals":{"key":"DocumentId","value":"DOC-123"}}]}}}}\n'
        "\n"
        "If the user question does not clearly imply filtering by these fields, output "
        '{"filter": null}.\n'
        "\n"
        "User question:\n"
    )


def build_filter_prompt(
    allowed_keys: FrozenSet[str],
    date_keys: FrozenSet[str],
    prompt_path: str = "",
    *,
    group_filter_key: str = "",
    group_options: FrozenSet[str] | None = None,
) -> str:
    group_options = group_options or frozenset()
    path = (prompt_path or "").strip()
    if path:
        resolved = resolve_filter_options_path(path)
        if not resolved.is_file():
            resolved = Path(path)
        text = resolved.read_text(encoding="utf-8")
        if not text.endswith("\n"):
            text += "\n"
        return _inject_prompt_placeholders(
            text,
            group_filter_key=group_filter_key,
            group_options=group_options,
        )
    return _default_filter_prompt_template(
        allowed_keys,
        date_keys,
        group_filter_key=group_filter_key,
        group_options=group_options,
    )


def _strip_markdown_json_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def prepare_retrieval_filter(
    filt: Any,
    allowed_keys: FrozenSet[str],
    date_keys: FrozenSet[str],
    *,
    in_keys: FrozenSet[str] | None = None,
    exact_equal_keys: FrozenSet[str] | None = None,
    date_strategy: str | None = None,
    group_filter_key: str = "",
    group_options: FrozenSet[str] | None = None,
) -> Optional[dict]:
    if filt is None:
        return None
    strategy = normalized_date_filter_strategy(date_strategy)
    group_options = group_options or frozenset()
    group_key = (group_filter_key or "").strip()
    in_metadata_keys = resolve_in_metadata_keys(
        allowed_keys,
        date_keys,
        in_keys_override=in_keys,
        exact_equal_keys=exact_equal_keys,
    )
    filt = normalize_retrieval_filter(filt, allowed_keys)
    filt = rewrite_text_equals_to_in(filt, in_metadata_keys)
    if group_key and group_options:
        filt = constrain_filter_group_values(filt, group_key, group_options)
        if filt is None:
            return None
    if strategy == "semantic_only":
        filt = strip_date_metadata_clauses(filt, date_keys)
    elif strategy == "year_in":
        year_map = resolve_date_to_year_map(date_keys, KB_YEAR_METADATA_KEY_SET)
        filt = rewrite_filters_to_year_in(filt, date_keys, year_map, allowed_keys)
        if filt is None:
            return None
        filt = strip_date_metadata_clauses(filt, date_keys)
    elif strategy == "numeric_yyyymmdd":
        filt = rewrite_retrieval_filter_dates_numeric(filt, date_keys)
    else:
        filt = rewrite_retrieval_filter_dates(filt, date_keys)
    if KB_YEAR_METADATA_KEY_SET and not _year_columns_indexed_on_kb():
        filt = strip_year_metadata_clauses(filt, KB_YEAR_METADATA_KEY_SET)
        if filt is None:
            print(
                "Metadata filter: all clauses removed after dropping unindexed Year* keys"
            )
    filt = sanitize_retrieval_filter(filt, allowed_keys, date_keys, strategy=strategy)
    if filt is None:
        return None
    if not validate_retrieval_filter(filt, allowed_keys, date_keys, strategy=strategy):
        print(f"Metadata filter rejected: {filt}")
        return None
    return filt


def parse_and_prepare_filter(
    raw_json_text: str,
    allowed_keys: FrozenSet[str],
    date_keys: FrozenSet[str],
    *,
    date_strategy: Optional[str] = None,
    group_filter_key: str = "",
    group_options: FrozenSet[str] | None = None,
) -> Optional[dict]:
    text = _strip_markdown_json_fence(raw_json_text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        print(f"Metadata filter JSON decode failed: {text[:300]}")
        return None
    filt = data.get("filter")
    if filt is None:
        return None
    return prepare_retrieval_filter(
        filt,
        allowed_keys,
        date_keys,
        date_strategy=date_strategy,
        group_filter_key=group_filter_key,
        group_options=group_options,
    )


def generate_metadata_filter(
    user_message: str,
    *,
    bedrock_runtime: Any,
    model_id: str,
    allowed_keys: FrozenSet[str],
    date_keys: FrozenSet[str],
    prompt_path: str = "",
    group_filter_key: str = "",
    group_options: FrozenSet[str] | None = None,
) -> Optional[dict]:
    if not allowed_keys:
        return None
    if bedrock_runtime is None:
        print("Metadata filter: bedrock-runtime client is not available")
        return None
    group_options = group_options or frozenset()
    prompt = (
        build_filter_prompt(
            allowed_keys,
            date_keys,
            prompt_path,
            group_filter_key=group_filter_key,
            group_options=group_options,
        )
        + user_message
    )
    try:
        resp = bedrock_runtime.converse(
            modelId=model_id,
            messages=[
                {"role": "user", "content": [{"text": prompt}]},
            ],
            inferenceConfig={"maxTokens": 800, "temperature": 0},
        )
        text = ""
        for block in resp.get("output", {}).get("message", {}).get("content", []):
            text += block.get("text", "")
    except Exception as e:
        print(f"Metadata filter converse error: {e}")
        return None
    return parse_and_prepare_filter(
        text,
        allowed_keys,
        date_keys,
        group_filter_key=group_filter_key,
        group_options=group_options,
    )


def kb_metadata_filter_enabled(flag: str) -> bool:
    return (flag or "").strip() in {"1", "true", "True", "yes", "on"}
