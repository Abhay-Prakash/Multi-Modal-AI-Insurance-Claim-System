import sys
from pathlib import Path
import tempfile

sys.path.insert(0, ".")
from ingestion import load_claims
from output import format_output_row, write_output_csv
from schemas import DecisionRecord, ClaimStatus, Severity, IssueType, ObjectPart, RiskFlag

def main():
    base_dir = Path(__file__).resolve().parent.parent
    claims_path = base_dir / "dataset" / "claims.csv"
    
    print(f"Loading {claims_path.name}...")
    claims, _ = load_claims(claims_path, base_dir / "dataset")
    print(f"Loaded {len(claims)} claims.")
    
    rows = []
    for claim in claims:
        # Create a dummy DecisionRecord for smoke testing
        dummy_record = DecisionRecord(
            evidence_standard_met=True,
            evidence_standard_met_reason="Smoke test reason",
            risk_flags=[RiskFlag.DAMAGE_NOT_VISIBLE],
            issue_type=IssueType.DENT,
            object_part=ObjectPart.FRONT_BUMPER,
            claim_status=ClaimStatus.SUPPORTED,
            claim_status_justification="Smoke test justification",
            supporting_image_ids=["img_1", "img_2"],
            valid_image=True,
            severity=Severity.MEDIUM,
        )
        row = format_output_row(claim, dummy_record)
        rows.append(row)
        
    print(f"Formatted {len(rows)} output rows successfully.")
    
    # Write to temp CSV to ensure no exceptions
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "output.csv"
        write_output_csv(rows, out_path)
        print(f"Successfully wrote CSV to {out_path} with {len(rows)} records.")
        
        # Verify first row
        with out_path.open("r", encoding="utf-8-sig") as f:
            lines = f.readlines()
            print("\nHeader:", lines[0].strip())
            print("Row 1: ", lines[1].strip())
            print("Row 2: ", lines[2].strip())

if __name__ == "__main__":
    main()
