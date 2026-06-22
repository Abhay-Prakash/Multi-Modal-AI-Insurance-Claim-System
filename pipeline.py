import time
import logging
import traceback
from dataclasses import dataclass, field
from typing import Optional

from schemas import (
    ClaimInput,
    DecisionRecord,
    ClaimStatus,
    Severity,
    RiskFlag,
    IssueType,
    ObjectPart,
    EvidenceFacts,
    UserHistory,
    EvidenceRequirement,
)

from sanitization import sanitize
from parsing import parse_claim
from retrieval import retrieve_context
from grounding import build_vlm_request
from agent import extract_evidence
from validator import validate_evidence_facts
from rules import evaluate

logger = logging.getLogger(__name__)

@dataclass
class PipelineTrace:
    """Telemetry and trace information for a single claim's pipeline run."""
    row_index: int
    strategy: str
    start_time: float
    duration_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _make_fallback_record(reason: str) -> DecisionRecord:
    """Produce a safe default record when the pipeline fails unexpectedly."""
    return DecisionRecord(
        evidence_standard_met=False,
        evidence_standard_met_reason=f"Pipeline failure: {reason}",
        risk_flags=[RiskFlag.MANUAL_REVIEW_REQUIRED],
        issue_type=IssueType.UNKNOWN,
        object_part=ObjectPart.UNKNOWN,
        claim_status=ClaimStatus.NOT_ENOUGH_INFORMATION,
        claim_status_justification=f"Exception during processing: {reason}",
        supporting_image_ids=[],
        valid_image=False,
        severity=Severity.UNKNOWN,
    )


def process_claim(
    claim: ClaimInput,
    history_db: dict[str, UserHistory],
    requirements_db: list[EvidenceRequirement],
    strategy: str = "A",
    verbose: bool = False
) -> tuple[DecisionRecord, PipelineTrace]:
    """Orchestrate the complete claim lifecycle for a single claim.

    Args:
        claim: The input claim to process.
        history_db: Lookup table for user history.
        requirements_db: List of all evidence requirements.
        strategy: "A" (Checklist grounding) or "B" (No checklist grounding).
        verbose: If True, logs detailed progress.

    Returns:
        A tuple of (DecisionRecord, PipelineTrace).
        Guaranteed to never raise an exception; returns a fallback record on failure.
    """
    start_time = time.time()
    trace = PipelineTrace(
        row_index=claim.row_index,
        strategy=strategy,
        start_time=start_time,
    )

    try:
        # 1. Sanitization
        if verbose:
            logger.info(f"[Row {claim.row_index}] Sanitizing claim...")
        sanitization = sanitize(claim)

        # 2. Parsing
        if verbose:
            logger.info(f"[Row {claim.row_index}] Parsing claim...")
        parsed = parse_claim(claim, sanitization)

        # 3. Retrieval
        if verbose:
            logger.info(f"[Row {claim.row_index}] Retrieving context...")
        context = retrieve_context(parsed, history_db, requirements_db)

        # 4. Grounding
        if verbose:
            logger.info(f"[Row {claim.row_index}] Grounding (strategy {strategy})...")
        vlm_req = build_vlm_request(parsed, context, strategy)

        # 5. Agent (VLM) + Validation (handled inside agent.py)
        if verbose:
            logger.info(f"[Row {claim.row_index}] Extracting evidence with Gemini...")
        facts = extract_evidence(vlm_req)

        # 6. Rule Evaluation
        if verbose:
            logger.info(f"[Row {claim.row_index}] Evaluating rules...")
        record = evaluate(parsed, facts, context)

    except Exception as e:
        err_msg = f"{type(e).__name__}: {str(e)}"
        trace.errors.append(err_msg)
        if verbose:
            logger.error(f"[Row {claim.row_index}] Pipeline failed with {err_msg}")
            logger.error(traceback.format_exc())
        record = _make_fallback_record(err_msg)

    trace.duration_ms = (time.time() - start_time) * 1000
    return record, trace

