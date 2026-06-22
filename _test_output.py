import sys
from pathlib import Path
import tempfile
import csv

sys.path.insert(0, ".")

from schemas import (
    ClaimInput,
    DecisionRecord,
    ClaimObject,
    ClaimStatus,
    Severity,
    IssueType,
    ObjectPart,
    RiskFlag,
    OUTPUT_CSV_COLUMNS
)
from output import format_output_row, write_output_csv


PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))
        FAIL += 1


print("\n[GROUP 1] output.py - Unit Tests")

claim = ClaimInput(
    user_id="user_test",
    image_paths_raw="images/test/case_001/img_1.jpg;images/test/case_001/img_2.jpg",
    images=[],
    user_claim="Customer: My car is broken.",
    claim_object=ClaimObject.CAR,
    row_index=0
)

record = DecisionRecord(
    evidence_standard_met=True,
    evidence_standard_met_reason="Good evidence.",
    risk_flags=[RiskFlag.BLURRY_IMAGE, RiskFlag.MANUAL_REVIEW_REQUIRED],
    issue_type=IssueType.DENT,
    object_part=ObjectPart.FRONT_BUMPER,
    claim_status=ClaimStatus.SUPPORTED,
    claim_status_justification="Verified damage.",
    supporting_image_ids=["img_1", "img_2"],
    valid_image=True,
    severity=Severity.MEDIUM,
)

# Test 1: format_output_row
formatted = format_output_row(claim, record)
check("Lowercase booleans", formatted.evidence_standard_met == "true" and formatted.valid_image == "true")
check("Enum unboxing", formatted.issue_type == "dent" and formatted.claim_status == "supported" and formatted.claim_object == "car")
check("Semicolon joining", formatted.risk_flags == "blurry_image;manual_review_required" and formatted.supporting_image_ids == "img_1;img_2")

# Test 2: Fallback formatting (empty lists)
record_fallback = DecisionRecord(
    evidence_standard_met=False,
    evidence_standard_met_reason="Missing images.",
    risk_flags=[],
    issue_type=IssueType.UNKNOWN,
    object_part=ObjectPart.UNKNOWN,
    claim_status=ClaimStatus.NOT_ENOUGH_INFORMATION,
    claim_status_justification="Need more.",
    supporting_image_ids=[],
    valid_image=False,
    severity=Severity.UNKNOWN,
)
formatted_fallback = format_output_row(claim, record_fallback)
check("Lowercase booleans (false)", formatted_fallback.evidence_standard_met == "false" and formatted_fallback.valid_image == "false")
check("Empty list fallback to 'none'", formatted_fallback.risk_flags == "none" and formatted_fallback.supporting_image_ids == "none")

# Test 3: CSV Writer
with tempfile.TemporaryDirectory() as tmp:
    out_path = Path(tmp) / "output.csv"
    write_output_csv([formatted, formatted_fallback], out_path)
    
    check("CSV file created", out_path.exists())
    
    with out_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    check("CSV column ordering", tuple(reader.fieldnames) == OUTPUT_CSV_COLUMNS)
    check("Row count correct", len(rows) == 2)
    check("First row values preserved", rows[0]["user_id"] == "user_test" and rows[0]["risk_flags"] == "blurry_image;manual_review_required")
    check("Second row fallback values preserved", rows[1]["supporting_image_ids"] == "none")

print(f"\nResults: {PASS} passed, {FAIL} failed")
if FAIL > 0:
    sys.exit(1)
