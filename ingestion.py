"""
Ingestion module: loads all dataset CSV files and resolves image references.

Public interface
----------------
``load_claims(csv_path, dataset_dir)``
    Load ``claims.csv`` or ``sample_claims.csv`` into a list of
    ``ClaimInput`` objects with resolved ``ImageRef`` entries.

``load_user_history(csv_path)``
    Load ``user_history.csv`` into a ``dict[user_id, UserHistory]``
    for O(1) lookup during retrieval.

``load_evidence_requirements(csv_path)``
    Load ``evidence_requirements.csv`` into a list of
    ``EvidenceRequirement`` objects.

``resolve_image_ref(relative_path, dataset_dir)``
    Resolve a single relative image path string to an ``ImageRef``.
    Raises ``FileNotFoundError`` if the file does not exist.
    Raises ``ValueError`` if the path format is unrecognised.

Design notes
------------
- All CSV reads use ``utf-8-sig`` encoding to handle optional BOM.
- Image path resolution is tolerant of mixed slash direction (``/`` and ``\\``).
- Missing or unresolvable image paths are collected as ``IngestionWarning``
  objects rather than raised as exceptions, so a single bad path does not
  abort the entire batch.
- Row index (0-based) is recorded on each ``ClaimInput`` for traceability.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from schemas import (
    ClaimInput,
    ClaimObject,
    EvidenceRequirement,
    ImageRef,
    UserHistory,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Warning container
# ---------------------------------------------------------------------------


@dataclass
class IngestionWarning:
    """A non-fatal issue encountered during ingestion.

    Collected rather than raised so that a single bad image path does not
    abort processing of an entire CSV file.
    """

    row_index: int
    field_name: str
    raw_value: str
    message: str


# ---------------------------------------------------------------------------
# Image path resolution
# ---------------------------------------------------------------------------


def resolve_image_ref(relative_path: str, dataset_dir: Path) -> ImageRef:
    """Resolve a relative image path to an ``ImageRef``.

    Args:
        relative_path: Path as it appears in the CSV, e.g.
            ``'images/test/case_001/img_1.jpg'``.  May use either
            forward or back slashes.
        dataset_dir: Absolute path to the dataset directory.  The
            ``relative_path`` is resolved relative to this directory.

    Returns:
        ``ImageRef`` with resolved absolute path, derived ``image_id``,
        ``case_id``, and ``split``.

    Raises:
        ValueError: If the path format does not match the expected
            ``images/{split}/{case_id}/{filename}`` structure.
        FileNotFoundError: If the resolved absolute path does not exist.

    Examples:
        >>> ref = resolve_image_ref(
        ...     "images/test/case_001/img_1.jpg", Path("/repo/dataset")
        ... )
        >>> ref.image_id
        'img_1'
        >>> ref.case_id
        'case_001'
        >>> ref.split
        'test'
    """
    # Normalise to forward slashes for consistent parsing.
    normalised = relative_path.replace("\\", "/").strip()

    parts = normalised.split("/")
    # Expected: images / {split} / {case_id} / {filename}
    if len(parts) < 4 or parts[0] != "images":
        raise ValueError(
            f"Unrecognised image path format: {relative_path!r}. "
            f"Expected 'images/{{split}}/{{case_id}}/{{filename}}'."
        )

    split = parts[1]       # 'sample' or 'test'
    case_id = parts[2]     # 'case_001', etc.
    filename = parts[-1]   # 'img_1.jpg'
    image_id = Path(filename).stem  # 'img_1'

    absolute = (dataset_dir / Path(normalised)).resolve()

    if not absolute.exists():
        raise FileNotFoundError(
            f"Image file not found: {absolute} "
            f"(resolved from relative path {relative_path!r})"
        )

    return ImageRef(
        image_id=image_id,
        path=str(absolute),
        relative_path=relative_path,
        case_id=case_id,
        split=split,
    )


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------


def load_claims(
    csv_path: Path,
    dataset_dir: Path,
) -> tuple[list[ClaimInput], list[IngestionWarning]]:
    """Load a claims CSV file into a list of ``ClaimInput`` objects.

    Handles both ``claims.csv`` (test set, no output columns) and
    ``sample_claims.csv`` (development set, has output columns).  Output
    columns present in ``sample_claims.csv`` are silently ignored here;
    they are consumed by the evaluation module separately.

    Args:
        csv_path: Absolute path to the CSV file to load.
        dataset_dir: Absolute path to the dataset root directory.  Used
            to resolve ``image_paths`` relative to the dataset.

    Returns:
        A tuple of:
        - ``list[ClaimInput]``: One ``ClaimInput`` per data row.
        - ``list[IngestionWarning]``: Non-fatal issues encountered (e.g.
          missing image files).  Empty when everything resolved cleanly.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        KeyError: If a required input column is absent from the CSV.

    Examples:
        >>> claims, warnings = load_claims(CLAIMS_CSV, DATASET_DIR)
        >>> len(claims)
        45
        >>> claims[0].claim_object
        <ClaimObject.CAR: 'car'>
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Claims CSV not found: {csv_path}")

    claims: list[ClaimInput] = []
    warnings: list[IngestionWarning] = []

    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)

        for row_index, row in enumerate(reader):
            image_paths_raw: str = row["image_paths"]
            raw_paths: list[str] = [
                p.strip()
                for p in image_paths_raw.split(";")
                if p.strip()
            ]

            images: list[ImageRef] = []
            for rp in raw_paths:
                try:
                    images.append(resolve_image_ref(rp, dataset_dir))
                except (FileNotFoundError, ValueError) as exc:
                    warnings.append(
                        IngestionWarning(
                            row_index=row_index,
                            field_name="image_paths",
                            raw_value=rp,
                            message=str(exc),
                        )
                    )
                    logger.warning(
                        "Row %d: could not resolve image path %r: %s",
                        row_index,
                        rp,
                        exc,
                    )

            claim_object_raw: str = row["claim_object"].strip().lower()
            try:
                claim_object = ClaimObject(claim_object_raw)
            except ValueError:
                warnings.append(
                    IngestionWarning(
                        row_index=row_index,
                        field_name="claim_object",
                        raw_value=claim_object_raw,
                        message=f"Unknown claim_object value {claim_object_raw!r}; "
                                f"defaulting to 'car'.",
                    )
                )
                logger.warning(
                    "Row %d: unknown claim_object %r, defaulting to 'car'.",
                    row_index,
                    claim_object_raw,
                )
                claim_object = ClaimObject.CAR

            claim = ClaimInput(
                user_id=row["user_id"].strip(),
                image_paths_raw=image_paths_raw,
                images=images,
                user_claim=row["user_claim"],
                claim_object=claim_object,
                row_index=row_index,
            )
            claims.append(claim)

    logger.info(
        "Loaded %d claims from %s (%d image resolution warnings).",
        len(claims),
        csv_path.name,
        len(warnings),
    )
    return claims, warnings


def load_user_history(csv_path: Path) -> dict[str, UserHistory]:
    """Load ``user_history.csv`` into a ``dict`` keyed by ``user_id``.

    The returned dictionary enables O(1) lookup during retrieval.

    Args:
        csv_path: Absolute path to ``user_history.csv``.

    Returns:
        ``dict[str, UserHistory]`` keyed by ``user_id``.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.

    Examples:
        >>> history = load_user_history(USER_HISTORY_CSV)
        >>> history["user_001"].rejected_claim
        0
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"User history CSV not found: {csv_path}")

    history: dict[str, UserHistory] = {}

    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)

        for row in reader:
            flags_raw: str = row.get("history_flags", "none").strip()
            # Semicolon-separated or the literal string "none".
            flags: list[str] = (
                []
                if flags_raw.lower() == "none"
                else [f.strip() for f in flags_raw.split(";") if f.strip()]
            )

            user_id = row["user_id"].strip()
            record = UserHistory(
                user_id=user_id,
                past_claim_count=_parse_int(row.get("past_claim_count", "0"), 0),
                accept_claim=_parse_int(row.get("accept_claim", "0"), 0),
                manual_review_claim=_parse_int(row.get("manual_review_claim", "0"), 0),
                rejected_claim=_parse_int(row.get("rejected_claim", "0"), 0),
                last_90_days_claim_count=_parse_int(
                    row.get("last_90_days_claim_count", "0"), 0
                ),
                history_flags=flags,
                history_summary=row.get("history_summary", "").strip(),
            )
            history[user_id] = record

    logger.info(
        "Loaded %d user history records from %s.", len(history), csv_path.name
    )
    return history


def load_evidence_requirements(csv_path: Path) -> list[EvidenceRequirement]:
    """Load ``evidence_requirements.csv`` into a list of ``EvidenceRequirement``.

    Args:
        csv_path: Absolute path to ``evidence_requirements.csv``.

    Returns:
        ``list[EvidenceRequirement]`` in CSV row order.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.

    Examples:
        >>> reqs = load_evidence_requirements(EVIDENCE_REQUIREMENTS_CSV)
        >>> reqs[0].requirement_id
        'REQ_GENERAL_OBJECT_PART'
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Evidence requirements CSV not found: {csv_path}")

    requirements: list[EvidenceRequirement] = []

    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)

        for row in reader:
            req = EvidenceRequirement(
                requirement_id=row["requirement_id"].strip(),
                claim_object=row["claim_object"].strip().lower(),
                applies_to=row["applies_to"].strip(),
                minimum_image_evidence=row["minimum_image_evidence"].strip(),
            )
            requirements.append(req)

    logger.info(
        "Loaded %d evidence requirements from %s.",
        len(requirements),
        csv_path.name,
    )
    return requirements


def load_sample_labels(csv_path: Path) -> dict[int, dict[str, str]]:
    """Load expected output columns from ``sample_claims.csv``.

    Returns a dict keyed by 0-based row index mapping to the output
    column values.  Used by the evaluation module to compare predictions
    against ground truth.

    Args:
        csv_path: Absolute path to ``sample_claims.csv``.

    Returns:
        ``dict[int, dict[str, str]]`` where keys are row indices (0-based)
        and values are dicts of output column name → raw string value.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Sample claims CSV not found: {csv_path}")

    OUTPUT_COLUMNS = {
        "evidence_standard_met",
        "evidence_standard_met_reason",
        "risk_flags",
        "issue_type",
        "object_part",
        "claim_status",
        "claim_status_justification",
        "supporting_image_ids",
        "valid_image",
        "severity",
    }

    labels: dict[int, dict[str, str]] = {}

    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = set(reader.fieldnames or [])
        available_output_cols = OUTPUT_COLUMNS & fieldnames

        for row_index, row in enumerate(reader):
            labels[row_index] = {
                col: row[col].strip()
                for col in available_output_cols
            }

    logger.info(
        "Loaded %d sample labels from %s.", len(labels), csv_path.name
    )
    return labels


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_int(value: str, default: int = 0) -> int:
    """Parse an integer from a CSV cell, returning *default* on failure."""
    try:
        return int(value.strip())
    except (ValueError, AttributeError):
        return default
