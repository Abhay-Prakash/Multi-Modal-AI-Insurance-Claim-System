import sys
import time
import tempfile
import csv
from pathlib import Path

sys.path.insert(0, ".")

import config
config.MAX_RETRIES = 0

from ingestion import load_claims, load_user_history, load_evidence_requirements
from pipeline import process_claim
from output import format_output_row, write_output_csv
from schemas import OUTPUT_CSV_COLUMNS, ClaimObject, ClaimStatus, Severity, RiskFlag, IssueType, ObjectPart

def main():
    base_dir = Path(__file__).resolve().parent.parent
    dataset_dir = base_dir / "dataset"
    claims_path = dataset_dir / "claims.csv"
    history_path = dataset_dir / "user_history.csv"
    reqs_path = dataset_dir / "evidence_requirements.csv"

    print("Loading dataset...")
    claims, _ = load_claims(claims_path, dataset_dir)
    history_db = load_user_history(history_path)
    reqs_db = load_evidence_requirements(reqs_path)
    
    print(f"Loaded {len(claims)} claims. Running pipeline...")
    
    t0 = time.time()
    rows = []
    fallback_count = 0
    
    for i, claim in enumerate(claims):
        record, trace = process_claim(
            claim=claim,
            history_db=history_db,
            requirements_db=reqs_db,
            strategy="B",
            verbose=False
        )
        if trace.errors:
            fallback_count += 1
            
        out_row = format_output_row(claim, record)
        
        # Validation of enums against schema
        assert out_row.claim_object in [e.value for e in ClaimObject], f"Invalid claim_object: {out_row.claim_object}"
        assert out_row.issue_type in [e.value for e in IssueType], f"Invalid issue_type: {out_row.issue_type}"
        assert out_row.object_part in [e.value for e in ObjectPart], f"Invalid object_part: {out_row.object_part}"
        assert out_row.claim_status in [e.value for e in ClaimStatus], f"Invalid claim_status: {out_row.claim_status}"
        assert out_row.severity in [e.value for e in Severity], f"Invalid severity: {out_row.severity}"
        
        if out_row.risk_flags != "none":
            for flag in out_row.risk_flags.split(";"):
                assert flag in [e.value for e in RiskFlag], f"Invalid risk_flag: {flag}"
                
        rows.append(out_row)
        
    duration = time.time() - t0
    
    # Write to a temp CSV to ensure no exceptions
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "output.csv"
        write_output_csv(rows, out_path)
        
        # Validate columns
        with out_path.open("r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert tuple(headers) == OUTPUT_CSV_COLUMNS, "Column ordering mismatch!"
            
    print("\n========================================")
    print("E2E Output Smoke Test Complete")
    print("========================================")
    print(f"Rows processed : {len(rows)}")
    print(f"Fallback count : {fallback_count}")
    print(f"Runtime        : {duration:.2f}s")
    print("Column ordering: VALIDATED")
    print("Enum values    : VALIDATED")

if __name__ == "__main__":
    main()
