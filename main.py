"""
CLI entry point for HackerRank Orchestrate (June 2026).

Usage:
    python code/main.py [--strategy {A,B}] [--verbose]

This script:
1. Loads all datasets.
2. Iterates over claims.csv sequentially.
3. Executes the full pipeline per claim.
4. Collects and reports execution statistics.
5. Emits the final output.csv.
"""

import argparse
import logging
import time
import sys
from pathlib import Path

from ingestion import load_claims, load_user_history, load_evidence_requirements
from pipeline import process_claim
from output import format_output_row, write_output_csv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="HackerRank Orchestrate - Claim Verification")
    parser.add_argument(
        "--strategy", 
        type=str, 
        choices=["A", "B"], 
        default="B",
        help="Strategy to use for grounding (A=Checklist, B=No Checklist)."
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable detailed logging."
    )
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    base_dir = Path(__file__).resolve().parent.parent
    dataset_dir = base_dir / "dataset"
    claims_path = dataset_dir / "claims.csv"
    history_path = dataset_dir / "user_history.csv"
    reqs_path = dataset_dir / "evidence_requirements.csv"
    output_path = base_dir / "output.csv"

    logger.info("Loading datasets...")
    try:
        claims, _ = load_claims(claims_path, dataset_dir)
        history_db = load_user_history(history_path)
        reqs_db = load_evidence_requirements(reqs_path)
    except FileNotFoundError as e:
        logger.error(f"Dataset loading failed: {e}")
        sys.exit(1)

    logger.info(f"Loaded {len(claims)} claims. Beginning sequential execution (Strategy {args.strategy}).")

    start_time = time.time()
    
    # Statistics
    claims_processed = 0
    success_count = 0
    fallback_count = 0
    gemini_call_count = 0
    gemini_failure_count = 0
    total_latency_ms = 0.0

    out_rows = []

    for claim in claims:
        claims_processed += 1
        gemini_call_count += 1
        
        record, trace = process_claim(
            claim=claim,
            history_db=history_db,
            requirements_db=reqs_db,
            strategy=args.strategy,
            verbose=args.verbose
        )
        
        total_latency_ms += trace.duration_ms
        
        if trace.errors:
            fallback_count += 1
            gemini_failure_count += 1
        else:
            success_count += 1
            
        out_row = format_output_row(claim, record)
        out_rows.append(out_row)

    runtime_seconds = time.time() - start_time
    average_latency_ms = total_latency_ms / max(claims_processed, 1)

    logger.info("Writing final output...")
    write_output_csv(out_rows, output_path)

    print("\n========================================")
    print("Execution Statistics")
    print("========================================")
    print(f"claims_processed      : {claims_processed}")
    print(f"success_count         : {success_count}")
    print(f"fallback_count        : {fallback_count}")
    print(f"gemini_call_count     : {gemini_call_count}")
    print(f"gemini_failure_count  : {gemini_failure_count}")
    print(f"runtime_seconds       : {runtime_seconds:.2f}s")
    print(f"average_latency_ms    : {average_latency_ms:.2f}ms")
    print("========================================\n")

if __name__ == "__main__":
    main()
