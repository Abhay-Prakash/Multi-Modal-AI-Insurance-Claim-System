import sys
import logging
from pathlib import Path

from ingestion import load_claims, load_user_history, load_evidence_requirements
from pipeline import process_claim

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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
    
    print(f"Loaded {len(claims)} claims.")
    
    success = 0
    fallback = 0
    
    print("\nRunning pipeline (Strategy B) over all claims...")
    for i, claim in enumerate(claims):
        print(f"\n[{i+1}/{len(claims)}] Processing Row {claim.row_index} (User: {claim.user_id})")
        record, trace = process_claim(
            claim=claim,
            history_db=history_db,
            requirements_db=reqs_db,
            strategy="B",
            verbose=False
        )
        
        if trace.errors:
            print(f"  ❌ Fallback Triggered. Errors: {trace.errors}")
            fallback += 1
        else:
            print(f"  ✅ Success. Status: {record.claim_status.value}, Severity: {record.severity.value}")
            success += 1
            
        print(f"  ⏱️ Time: {trace.duration_ms:.0f}ms")
    
    print("\n" + "="*40)
    print("Pipeline Smoke Test Complete")
    print(f"Total claims: {len(claims)}")
    print(f"Successful:   {success}")
    print(f"Fallbacks:    {fallback}")
    print("="*40)

if __name__ == "__main__":
    main()
