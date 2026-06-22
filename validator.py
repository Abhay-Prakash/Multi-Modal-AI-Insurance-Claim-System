"""
Validator module: parse and repair Gemini JSON output into EvidenceFacts.

Public interface
----------------
``validate_evidence_facts(raw_dict, valid_image_ids)``
    Parse a raw ``dict`` from the Gemini response and return a validated
    ``EvidenceFacts``.  Applies enum normalisation and formatting repair.
    Never invents observations.  Never infers evidence.

``parse_response_text(text)``
    Parse the raw JSON string from the Gemini response into a dict.
    Handles markdown code fences and whitespace-only responses.

Design notes — conservative repair only
----------------------------------------
The validator follows a strict hierarchy:

1. **Try direct Pydantic parse first.**  If it succeeds, return immediately.
2. **Enum normalisation:** map non-canonical strings to valid enum values
   (case-insensitive, dash/underscore tolerant).  Applied field by field.
3. **Type coercion:** convert string booleans, empty-string → None,
   out-of-range floats to their clamped values.
4. **Unknown image_id repair:** map "Image N" references to the Nth
   ``valid_image_id`` when possible.
5. **Structural fallback:** if an observation or finding cannot be repaired
   to a valid state, it is **dropped**, not replaced with fabricated data.
6. **Hard invariant:** no field from the decision layer (``claim_status``,
   ``evidence_standard_met``, ``risk_flags``, ``alignment``) may appear in
   the returned ``EvidenceFacts``.

If the entire response is unparseable, an empty ``EvidenceFacts`` is returned
(no observations, no findings).  ``rules.py`` will then determine that
evidence is insufficient.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from pydantic import ValidationError

from schemas import (
    EvidenceFacts,
    Finding,
    ImageObservation,
    IssueType,
    Severity,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enum normalisation tables
# ---------------------------------------------------------------------------

# Maps raw string variants → canonical IssueType value.
# Tries longest match first when multiple entries share a prefix.
_ISSUE_TYPE_MAP: dict[str, str] = {
    # Canonical
    "dent": "dent",
    "scratch": "scratch",
    "crack": "crack",
    "glass_shatter": "glass_shatter",
    "broken_part": "broken_part",
    "missing_part": "missing_part",
    "torn_packaging": "torn_packaging",
    "crushed_packaging": "crushed_packaging",
    "water_damage": "water_damage",
    "stain": "stain",
    "none": "none",
    "unknown": "unknown",
    # Variants
    "glass shatter": "glass_shatter",
    "glass-shatter": "glass_shatter",
    "glasshatter": "glass_shatter",
    "shatter": "glass_shatter",
    "shattered": "glass_shatter",
    "broken part": "broken_part",
    "broken-part": "broken_part",
    "broken": "broken_part",
    "missing part": "missing_part",
    "missing-part": "missing_part",
    "missing": "missing_part",
    "torn packaging": "torn_packaging",
    "torn-packaging": "torn_packaging",
    "torn": "torn_packaging",
    "ripped": "torn_packaging",
    "crushed packaging": "crushed_packaging",
    "crushed-packaging": "crushed_packaging",
    "crushed": "crushed_packaging",
    "water damage": "water_damage",
    "water-damage": "water_damage",
    "water": "water_damage",
    "wet": "water_damage",
    "scratched": "scratch",
    "dented": "dent",
    "cracked": "crack",
    "fracture": "crack",
    "fractured": "crack",
    "no damage": "none",
    "no_damage": "none",
    "undamaged": "none",
    "n/a": "unknown",
    "null": "unknown",
    "": "unknown",
}

_SEVERITY_MAP: dict[str, str] = {
    "none": "none",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "unknown": "unknown",
    # Variants
    "minimal": "low",
    "minor": "low",
    "slight": "low",
    "moderate": "medium",
    "significant": "medium",
    "severe": "high",
    "major": "high",
    "critical": "high",
    "extensive": "high",
    "n/a": "unknown",
    "null": "unknown",
    "": "unknown",
}

# Maps raw quality issue strings → canonical form expected by rules.py.
_QUALITY_ISSUE_MAP: dict[str, str] = {
    "blurry": "blurry",
    "blur": "blurry",
    "blurred": "blurry",
    "out of focus": "blurry",
    "low_light": "low_light",
    "low light": "low_light",
    "dark": "low_light",
    "dim": "low_light",
    "underexposed": "low_light",
    "glare": "glare",
    "overexposed": "glare",
    "reflection": "glare",
    "bright spot": "glare",
    "wrong_angle": "wrong_angle",
    "wrong angle": "wrong_angle",
    "bad angle": "wrong_angle",
    "angle": "wrong_angle",
    "cropped_or_obstructed": "cropped_or_obstructed",
    "cropped": "cropped_or_obstructed",
    "obstructed": "cropped_or_obstructed",
    "obscured": "cropped_or_obstructed",
    "partial": "cropped_or_obstructed",
    "partially visible": "cropped_or_obstructed",
    "wrong_object": "wrong_object",
    "wrong object": "wrong_object",
    "incorrect object": "wrong_object",
    "different object": "wrong_object",
}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def parse_response_text(text: str) -> dict[str, Any]:
    """Parse a raw Gemini response string into a dict.

    Handles:
    - Clean JSON strings.
    - Responses wrapped in markdown code fences (```json...```).
    - Leading/trailing whitespace.

    Args:
        text: The ``response.text`` string from the Gemini SDK.

    Returns:
        Parsed dict.

    Raises:
        ValueError: If the text cannot be parsed as JSON after stripping
            code fences.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("Gemini returned an empty response body.")

    # Strip markdown code fences if present.
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped)
    if fence_match:
        stripped = fence_match.group(1).strip()

    try:
        result = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not parse Gemini response as JSON: {exc}. "
            f"First 200 chars: {stripped[:200]!r}"
        ) from exc

    if not isinstance(result, dict):
        raise ValueError(
            f"Expected a JSON object (dict), got {type(result).__name__}."
        )
    return result


def validate_evidence_facts(
    raw_dict: dict[str, Any],
    valid_image_ids: list[str],
) -> EvidenceFacts:
    """Parse and repair a raw Gemini output dict into a validated ``EvidenceFacts``.

    Applies conservative repair in this order:
    1. Try direct Pydantic validation — return immediately if successful.
    2. Repair ``image_observations``: normalise enums, coerce types,
       map unknown image_ids, drop unfixable entries.
    3. Repair ``findings``: normalise enums, drop unfixable entries.
    4. Re-validate with Pydantic.
    5. Return empty ``EvidenceFacts`` if still invalid.

    Invariants:
    - Never invents observations.
    - Never infers or synthesises evidence.
    - Drops rather than fabricates unfixable entries.
    - No decision-layer fields survive into the output.

    Args:
        raw_dict: Parsed JSON dict from the Gemini response.
        valid_image_ids: The ``image_id`` values that were submitted with
            this claim (from ``VLMRequest.image_paths``).  Used for
            image_id normalisation.

    Returns:
        Validated ``EvidenceFacts`` (may have empty lists if nothing survived).
    """
    # Strip any decision-layer fields that must never appear in EvidenceFacts.
    _strip_decision_fields(raw_dict)

    # Normalise image_ids and enums before the first Pydantic attempt so that
    # the fast path also benefits from repair (Pydantic accepts any string for
    # image_id, so 'Image 1' would pass validation uncorrected otherwise).
    _pre_normalise(raw_dict, valid_image_ids)

    # Try direct Pydantic parse first — cheapest path.
    try:
        return EvidenceFacts.model_validate(raw_dict)
    except (ValidationError, Exception):
        pass  # Fall through to full repair

    logger.debug("Direct Pydantic validation failed; attempting field repair.")

    repaired: dict[str, Any] = {}

    # Repair image_observations.
    raw_obs = raw_dict.get("image_observations", [])
    if isinstance(raw_obs, list):
        repaired["image_observations"] = _repair_observations(
            raw_obs, valid_image_ids
        )
    else:
        repaired["image_observations"] = []

    # Repair findings.
    raw_findings = raw_dict.get("findings", [])
    if isinstance(raw_findings, list):
        repaired["findings"] = _repair_findings(raw_findings, valid_image_ids)
    else:
        repaired["findings"] = []

    # Second Pydantic attempt with repaired data.
    try:
        facts = EvidenceFacts.model_validate(repaired)
        logger.debug(
            "Repair succeeded: %d observations, %d findings.",
            len(facts.image_observations),
            len(facts.findings),
        )
        return facts
    except (ValidationError, Exception) as exc:
        logger.warning(
            "Repair failed; returning empty EvidenceFacts. Error: %s", exc
        )
        return EvidenceFacts(image_observations=[], findings=[])


# ---------------------------------------------------------------------------
# Decision-field stripping
# ---------------------------------------------------------------------------

_DECISION_FIELDS: frozenset[str] = frozenset({
    "claim_status",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "alignment",
    "evidence_sufficient",
    "severity",          # top-level severity belongs to DecisionRecord, not EvidenceFacts
    "valid_image",
    "supporting_image_ids",  # at EvidenceFacts level; findings have their own
    "claim_status_justification",
    "object_part",       # at EvidenceFacts level
    "issue_type",        # at EvidenceFacts level (valid inside findings/observations)
})


def _strip_decision_fields(d: dict[str, Any]) -> None:
    """Remove top-level decision-layer fields from *d* in place."""
    for field in _DECISION_FIELDS:
        if field in d:
            logger.warning(
                "Gemini response contained decision-layer field %r; stripped.", field
            )
            del d[field]


def _pre_normalise(raw_dict: dict[str, Any], valid_image_ids: list[str]) -> None:
    """Apply lightweight in-place normalisation before the first Pydantic attempt.

    Targets the most common Gemini quirks:
    - image_id: "Image 1" / "image1" / "1" → mapped to the Nth valid_image_id.
    - issue_type_observed: non-canonical case or spacing → canonical string.
    - severity_estimate: non-canonical → canonical string.
    - quality_issues: string instead of list → converted to list.

    Mutates *raw_dict* in place.  If the field is absent or already valid,
    it is left untouched.
    """
    raw_obs = raw_dict.get("image_observations")
    if isinstance(raw_obs, list):
        for i, obs in enumerate(raw_obs):
            if not isinstance(obs, dict):
                continue
            # image_id
            raw_id = obs.get("image_id")
            repaired_id = _repair_image_id(raw_id, valid_image_ids, i)
            if repaired_id is not None:
                obs["image_id"] = repaired_id
            # enums
            if "issue_type_observed" in obs:
                obs["issue_type_observed"] = _normalize_issue_type(obs["issue_type_observed"])
            if "severity_estimate" in obs:
                obs["severity_estimate"] = _normalize_severity(obs["severity_estimate"])
            if "quality_issues" in obs:
                obs["quality_issues"] = _normalize_quality_issues(obs["quality_issues"])

    raw_findings = raw_dict.get("findings")
    if isinstance(raw_findings, list):
        for j, finding in enumerate(raw_findings):
            if not isinstance(finding, dict):
                continue
            if "issue_type" in finding:
                finding["issue_type"] = _normalize_issue_type(finding["issue_type"])
            if "severity_estimate" in finding:
                finding["severity_estimate"] = _normalize_severity(finding["severity_estimate"])
            # supporting_image_ids
            raw_ids = finding.get("supporting_image_ids", [])
            if isinstance(raw_ids, str):
                raw_ids = [raw_ids]
            if isinstance(raw_ids, list):
                resolved = []
                for k, rid in enumerate(raw_ids):
                    vid = _repair_image_id(rid, valid_image_ids, k)
                    if vid and vid not in resolved:
                        resolved.append(vid)
                finding["supporting_image_ids"] = resolved


# ---------------------------------------------------------------------------
# Image observation repair
# ---------------------------------------------------------------------------


def _repair_observations(
    raw_list: list[Any],
    valid_image_ids: list[str],
) -> list[dict[str, Any]]:
    """Repair a list of raw image observation dicts.

    Returns only successfully repaired entries.  Drops the rest.
    """
    repaired: list[dict[str, Any]] = []
    for i, raw in enumerate(raw_list):
        if not isinstance(raw, dict):
            logger.debug("Observation %d is not a dict; dropped.", i)
            continue
        result = _repair_single_observation(raw, valid_image_ids, i)
        if result is not None:
            repaired.append(result)
    return repaired


def _repair_single_observation(
    raw: dict[str, Any],
    valid_image_ids: list[str],
    index: int,
) -> Optional[dict[str, Any]]:
    """Attempt to repair a single image observation dict.

    Returns the repaired dict, or None if it cannot be salvaged.
    """
    obs: dict[str, Any] = {}

    # image_id — required field; drop the observation if missing after repair.
    raw_image_id = raw.get("image_id")
    repaired_id = _repair_image_id(raw_image_id, valid_image_ids, index)
    if repaired_id is None:
        logger.debug(
            "Observation %d: image_id %r could not be resolved; dropped.",
            index,
            raw_image_id,
        )
        return None
    obs["image_id"] = repaired_id

    # Boolean fields — coerce string "true"/"false" and int 0/1.
    obs["object_visible"] = _coerce_bool(raw.get("object_visible", False))
    obs["part_visible"] = _coerce_bool(raw.get("part_visible", False))
    obs["damage_observed"] = _coerce_bool(raw.get("damage_observed", False))
    obs["text_or_instructions_present"] = _coerce_bool(
        raw.get("text_or_instructions_present", False)
    )

    # Optional string fields — None when empty or absent.
    obs["object_type_observed"] = _coerce_optional_str(raw.get("object_type_observed"))
    obs["part_observed"] = _coerce_optional_str(raw.get("part_observed"))
    obs["damage_description"] = _coerce_optional_str(raw.get("damage_description"))

    # Enum fields.
    obs["issue_type_observed"] = _normalize_issue_type(
        raw.get("issue_type_observed")
    )
    obs["severity_estimate"] = _normalize_severity(raw.get("severity_estimate"))

    # Quality issues — may be a string, list of strings, or missing.
    obs["quality_issues"] = _normalize_quality_issues(raw.get("quality_issues", []))

    # Confidence — clamp to [0.0, 1.0].
    obs["confidence"] = _clamp_confidence(raw.get("confidence", 0.0))

    return obs


def _repair_image_id(
    raw: Any,
    valid_image_ids: list[str],
    observation_index: int,
) -> Optional[str]:
    """Attempt to resolve a raw image_id to a valid image_id.

    Resolution steps:
    1. If raw is in valid_image_ids → accept.
    2. If raw matches "Image N" or "image N" → map to Nth valid_image_id.
    3. If raw is an integer → treat as 1-based index.
    4. Otherwise → return None (cannot resolve).
    """
    if raw is None:
        # Try to infer from observation index.
        if observation_index < len(valid_image_ids):
            logger.debug(
                "Observation %d: null image_id; inferred %r from position.",
                observation_index,
                valid_image_ids[observation_index],
            )
            return valid_image_ids[observation_index]
        return None

    raw_str = str(raw).strip()

    # Direct match.
    if raw_str in valid_image_ids:
        return raw_str

    # "Image N" pattern (1-based).
    m = re.match(r"(?:image\s*)?(\d+)$", raw_str, re.IGNORECASE)
    if m:
        n = int(m.group(1)) - 1  # Convert to 0-based.
        if 0 <= n < len(valid_image_ids):
            logger.debug(
                "Observation %d: image_id %r mapped to %r via position.",
                observation_index,
                raw_str,
                valid_image_ids[n],
            )
            return valid_image_ids[n]

    # Case-insensitive match.
    raw_lower = raw_str.lower()
    for vid in valid_image_ids:
        if vid.lower() == raw_lower:
            return vid

    # Partial prefix match (e.g. "img_1" matches "img_1.jpg" stripped stem).
    for vid in valid_image_ids:
        if raw_lower.startswith(vid.lower()) or vid.lower().startswith(raw_lower):
            logger.debug(
                "Observation %d: image_id %r matched %r via prefix.",
                observation_index,
                raw_str,
                vid,
            )
            return vid

    return None


# ---------------------------------------------------------------------------
# Finding repair
# ---------------------------------------------------------------------------


def _repair_findings(
    raw_list: list[Any],
    valid_image_ids: list[str],
) -> list[dict[str, Any]]:
    """Repair a list of raw finding dicts."""
    repaired: list[dict[str, Any]] = []
    for i, raw in enumerate(raw_list):
        if not isinstance(raw, dict):
            logger.debug("Finding %d is not a dict; dropped.", i)
            continue
        result = _repair_single_finding(raw, valid_image_ids, i)
        if result is not None:
            repaired.append(result)
    return repaired


def _repair_single_finding(
    raw: dict[str, Any],
    valid_image_ids: list[str],
    index: int,
) -> Optional[dict[str, Any]]:
    """Repair a single finding dict.  Returns None if unsalvageable."""
    description = _coerce_optional_str(raw.get("description"))
    if not description:
        logger.debug("Finding %d: missing description; dropped.", index)
        return None

    finding: dict[str, Any] = {"description": description}

    finding["issue_type"] = _normalize_issue_type(raw.get("issue_type"))
    finding["object_part"] = _coerce_optional_str(raw.get("object_part"))
    finding["severity_estimate"] = _normalize_severity(raw.get("severity_estimate"))

    # supporting_image_ids — keep only those that resolve to valid ids.
    raw_ids = raw.get("supporting_image_ids", [])
    if isinstance(raw_ids, str):
        raw_ids = [raw_ids]
    resolved: list[str] = []
    for j, rid in enumerate(raw_ids):
        vid = _repair_image_id(rid, valid_image_ids, j)
        if vid and vid not in resolved:
            resolved.append(vid)
    finding["supporting_image_ids"] = resolved

    return finding


# ---------------------------------------------------------------------------
# Field-level coercion helpers
# ---------------------------------------------------------------------------


def _normalize_issue_type(raw: Any) -> Optional[str]:
    """Normalize a raw issue_type value to a canonical IssueType string or None."""
    if raw is None:
        return None
    normalised = str(raw).strip().lower().replace("-", "_")
    if normalised in (v for v in IssueType.__members__.values()):
        return normalised
    # Try the flat map.
    mapped = _ISSUE_TYPE_MAP.get(normalised) or _ISSUE_TYPE_MAP.get(
        str(raw).strip().lower()
    )
    if mapped:
        return mapped
    logger.debug("issue_type %r could not be normalised; set to 'unknown'.", raw)
    return "unknown"


def _normalize_severity(raw: Any) -> Optional[str]:
    """Normalize a raw severity value to a canonical Severity string or None."""
    if raw is None:
        return None
    normalised = str(raw).strip().lower()
    if normalised in (v for v in Severity.__members__.values()):
        return normalised
    mapped = _SEVERITY_MAP.get(normalised)
    if mapped:
        return mapped
    logger.debug("severity %r could not be normalised; set to 'unknown'.", raw)
    return "unknown"


def _normalize_quality_issues(raw: Any) -> list[str]:
    """Normalize the quality_issues field to a list of canonical strings."""
    if isinstance(raw, str):
        # May be a comma-separated string.
        raw = [s.strip() for s in raw.split(",") if s.strip()]
    if not isinstance(raw, list):
        return []
    result: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        key = item.strip().lower()
        canonical = _QUALITY_ISSUE_MAP.get(key) or _QUALITY_ISSUE_MAP.get(
            re.sub(r"[_\-]", " ", key)
        )
        if canonical and canonical not in result:
            result.append(canonical)
        elif key and key not in result:
            # Keep unrecognised but non-empty strings as-is (for logging).
            result.append(key)
    return result


def _coerce_bool(raw: Any) -> bool:
    """Coerce a value to bool.  Handles string 'true'/'false', int 0/1."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return bool(raw)
    if isinstance(raw, str):
        return raw.strip().lower() in ("true", "1", "yes")
    return False


def _coerce_optional_str(raw: Any) -> Optional[str]:
    """Return None for missing, null, or empty values; otherwise str."""
    if raw is None:
        return None
    s = str(raw).strip()
    return s if s and s.lower() not in ("null", "none", "n/a") else None


def _clamp_confidence(raw: Any) -> float:
    """Clamp a confidence value to [0.0, 1.0]."""
    try:
        v = float(raw)
        return max(0.0, min(1.0, v))
    except (TypeError, ValueError):
        return 0.0
