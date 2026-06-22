"""
Output formatting module: Converts DecisionRecords to the exact output CSV format.

Public interface
----------------
``format_output_row(claim, record)``
    Convert a ClaimInput and its DecisionRecord into a FinalOutputRow,
    enforcing lowercase booleans, enum values, and semicolon list formatting.

``write_output_csv(rows, output_path)``
    Write a list of FinalOutputRow objects to a CSV file matching OUTPUT_CSV_COLUMNS.
"""

import csv
import logging
from pathlib import Path

from schemas import ClaimInput, DecisionRecord, FinalOutputRow, OUTPUT_CSV_COLUMNS

logger = logging.getLogger(__name__)


def _format_bool(value: bool) -> str:
    """Format boolean as 'true' or 'false'."""
    return "true" if value else "false"


def _format_list(items: list, default: str = "none") -> str:
    """Format a list of strings or enums as semicolon-separated string."""
    if not items:
        return default
    
    # Handle Enums by getting .value, otherwise stringify
    string_items = []
    for item in items:
        if hasattr(item, "value"):
            string_items.append(str(item.value))
        else:
            string_items.append(str(item))
            
    return ";".join(string_items)


def format_output_row(claim: ClaimInput, record: DecisionRecord) -> FinalOutputRow:
    """Format the results of a claim into the final output schema.

    Enforces all output constraints:
    - Lowercase booleans ('true', 'false').
    - Semicolon-separated lists (fallback to 'none').
    - Exact enum value unboxing.
    """
    return FinalOutputRow(
        user_id=claim.user_id,
        image_paths=claim.image_paths_raw,
        user_claim=claim.user_claim,
        claim_object=claim.claim_object.value,
        evidence_standard_met=_format_bool(record.evidence_standard_met),
        evidence_standard_met_reason=record.evidence_standard_met_reason,
        risk_flags=_format_list(record.risk_flags, default="none"),
        issue_type=record.issue_type.value,
        object_part=record.object_part.value if hasattr(record.object_part, "value") else str(record.object_part),
        claim_status=record.claim_status.value,
        claim_status_justification=record.claim_status_justification,
        supporting_image_ids=_format_list(record.supporting_image_ids, default="none"),
        valid_image=_format_bool(record.valid_image),
        severity=record.severity.value,
    )


def write_output_csv(rows: list[FinalOutputRow], output_path: Path) -> None:
    """Write formatted rows to the output CSV file."""
    if not rows:
        logger.warning("No rows to write to CSV.")
        return

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_CSV_COLUMNS)
        writer.writeheader()
        
        for row in rows:
            writer.writerow(row.to_dict())
            
    logger.info("Wrote %d rows to %s", len(rows), output_path)
