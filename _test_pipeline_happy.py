import sys
import time
import tempfile
import csv
from pathlib import Path

sys.path.insert(0, ".")

from ingestion import load_claims, load_user_history, load_evidence_requirements
from pipeline import process_claim
from output import format_output_row, write_output_csv
from schemas import OUTPUT_CSV_COLUMNS, ClaimObject, ClaimStatus, Severity, RiskFlag, IssueType, ObjectPart

# Configurable delay to avoid Gemini free-tier 429 limits
DELAY_BETWEEN_CALLS = 20.0

def main():
    print("Waiting 60 seconds to clear Gemini free-tier rate limit bucket...")
    time.sleep(60)
    
    base_dir = Path(__file__).resolve().parent.parent
    dataset_dir = base_dir / "dataset"
    claims_path = dataset_dir / "claims.csv"
    history_path = dataset_dir / "user_history.csv"
    reqs_path = dataset_dir / "evidence_requirements.csv"

    print("Loading dataset...")
    claims, _ = load_claims(claims_path, dataset_dir)
    history_db = load_user_history(history_path)
    reqs_db = load_evidence_requirements(reqs_path)
    
    # Take only the first 3 claims for the happy path test
    test_claims = claims[:3]
    print(f"Loaded {len(claims)} claims. Running happy-path pipeline on the first {len(test_claims)}...")
    
    t0 = time.time()
    rows = []
    fallback_count = 0
    success_count = 0
    total_duration_ms = 0.0
    
    for i, claim in enumerate(test_claims):
        if i > 0:
            print(f"Sleeping for {DELAY_BETWEEN_CALLS}s to avoid rate limits...")
            time.sleep(DELAY_BETWEEN_CALLS)
            
        print(f"[{i+1}/{len(test_claims)}] Processing Row {claim.row_index} (User: {claim.user_id})")
        record, trace = process_claim(
            claim=claim,
            history_db=history_db,
            requirements_db=reqs_db,
            strategy="B",
            verbose=False
        )
        
        # Verify trace has timings and handles warnings/errors structurally
        assert hasattr(trace, "duration_ms")
        assert trace.duration_ms > 0
        total_duration_ms += trace.duration_ms
        
        if trace.errors:
            print(f"  -> Fallback (Errors: {trace.errors})")
            fallback_count += 1
        else:
            print(f"  -> Success! (Status: {record.claim_status.value})")
            success_count += 1
            
        out_row = format_output_row(claim, record)
        
        # Validation of enums against schema (no fabricated values)
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
    
    # Write to a temp CSV to ensure serialization succeeds
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "output.csv"
        write_output_csv(rows, out_path)
        
        # Validate exact output columns
        with out_path.open("r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert tuple(headers) == OUTPUT_CSV_COLUMNS, "Column ordering mismatch!"
            
    print("\n========================================")
    print("Happy Path E2E Test Complete")
    print("========================================")
    print(f"Rows processed : {len(rows)}")
    print(f"Success count  : {success_count}")
    print(f"Fallback count : {fallback_count}")
    print(f"Runtime        : {duration:.2f}s")
    if len(rows) > 0:
        print(f"Average latency: {(total_duration_ms / len(rows)):.2f}ms per claim")
    print("Column ordering: VALIDATED")
    print("Enum values    : VALIDATED")
    print("Traces active  : VALIDATED")
    
    # Assert at least one success to validate the happy path works
    assert success_count > 0, "Expected at least one non-fallback record! (API Quota Exhausted?)"

if __name__ == "__main__":
    main()
