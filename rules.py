"""
Rule engine: deterministic decision layer for evidence-based claim adjudication.

Public interface
----------------
``evaluate(parsed, facts, context) -> DecisionRecord``
    The sole entry point.  Accepts the full pipeline context and returns a
    complete ``DecisionRecord``.  No LLM calls, no embeddings, no retrieval,
    no randomness.

Internal decision pipeline (all deterministic):
  1. ``_assess_valid_image``         → bool
  2. ``_assess_atoms``               → list[ClaimAtomAssessment]
  3. ``_assess_evidence_standard``   → (bool, str)
  4. ``_project_issue_type``         → IssueType
  5. ``_project_object_part``        → ObjectPart
  6. ``_determine_claim_status``     → (ClaimStatus, str, list[str])
  7. ``_project_severity``           → Severity
  8. ``_collect_risk_flags``         → list[RiskFlag]

Invariants (also enforced by DecisionRecord validators):
  - ``claim_status = supported`` requires ``evidence_standard_met = True``.
  - ``claim_status = contradicted`` requires alignment = 'mismatch' (never 'unclear').
  - Confidence values from EvidenceFacts never gate any branch.
  - History flags appear only in ``risk_flags``, never in ``claim_status``.
  - Severity is projected from ``ISSUE_SEVERITY_MAP`` (deterministic),
    with VLM ``severity_estimate`` used as a tiebreaker only for 'unknown'.

Alignment semantics (ClaimAtomAssessment.alignment):
  'match'    — observations confirm the claimed damage type and location.
  'mismatch' — unambiguous contradiction: clear images (no quality issues)
               show either (a) no damage anywhere, or (b) confirmed damage of
               a different type.  NEVER inferred from ambiguous evidence.
  'unclear'  — insufficient or conflicting evidence.  Covers two sub-cases:
               • All images are degraded — cannot confirm or contradict.
               • CONFLICTING (Policy B): a clear image shows no damage, but
                 another image observes damage of undetermined type.  Routes
                 to not_enough_information + MANUAL_REVIEW_REQUIRED rather
                 than committing to contradiction on ambiguous evidence.
"""

from __future__ import annotations

import logging
from typing import Optional

from schemas import (
    ClaimAtom,
    ClaimAtomAssessment,
    ClaimObject,
    ClaimStatus,
    DecisionRecord,
    EvidenceFacts,
    Finding,
    ImageObservation,
    ISSUE_SEVERITY_MAP,
    IssueType,
    ObjectPart,
    ParsedClaim,
    RetrievedContext,
    RiskFlag,
    SEVERITY_ORDER,
    Severity,
    UserHistory,
    most_severe,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quality-issue keyword sets (from ImageObservation.quality_issues values)
# ---------------------------------------------------------------------------

_Q_BLURRY: frozenset[str] = frozenset({"blurry", "blur", "blurred"})
_Q_LOW_LIGHT: frozenset[str] = frozenset({"low_light", "dark", "dim", "underexposed"})
_Q_GLARE: frozenset[str] = frozenset({"glare", "overexposed", "reflection"})
_Q_WRONG_ANGLE: frozenset[str] = frozenset({"wrong_angle", "bad_angle"})
_Q_CROPPED: frozenset[str] = frozenset({"cropped_or_obstructed", "cropped", "obstructed"})
_Q_WRONG_OBJECT: frozenset[str] = frozenset({"wrong_object"})

# An image is "blocking-degraded" when it has at least 2 of the 3 core
# quality issues simultaneously (blurry, low_light/glare, wrong_angle).
_CORE_QUALITY_ISSUES: tuple[frozenset[str], ...] = (
    _Q_BLURRY,
    _Q_LOW_LIGHT | _Q_GLARE,
    _Q_WRONG_ANGLE,
    _Q_CROPPED,
)


# ---------------------------------------------------------------------------
# History-risk thresholds (deterministic)
# ---------------------------------------------------------------------------

_HISTORY_HIGH_REJECT_RATE_THRESHOLD: float = 0.30   # 30% of past claims rejected
_HISTORY_MIN_CLAIMS_FOR_RATE: int = 3               # Need ≥3 past claims to compute a rate
_HISTORY_RECENT_CLAIM_THRESHOLD: int = 3            # ≥3 claims in last 90 days → frequent
_HISTORY_FLAG_KEYWORDS: frozenset[str] = frozenset({
    "user_history_risk", "manual_review_required", "fraud_suspected",
    "high_risk",
})


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate(
    parsed: ParsedClaim,
    facts: EvidenceFacts,
    context: RetrievedContext,
) -> DecisionRecord:
    """Produce a ``DecisionRecord`` deterministically from observations and context.

    This is the sole public function.  It orchestrates eight sub-steps in order,
    each of which is deterministic and testable in isolation.

    Args:
        parsed: ``ParsedClaim`` from parsing.py — supplies claim atoms and
            sanitization flags.
        facts: ``EvidenceFacts`` from agent.py / validator.py — observations only.
        context: ``RetrievedContext`` from retrieval.py — history and requirements.

    Returns:
        A fully validated ``DecisionRecord``.

    Raises:
        ValueError: If the constructed record violates a schema invariant
            (e.g. supported without evidence_standard_met).  This indicates
            a logic error in the rule engine, not a data error.
    """
    valid_image = _assess_valid_image(facts)
    atom_assessments = _assess_atoms(parsed, facts)
    evidence_standard_met, esm_reason = _assess_evidence_standard(
        facts, atom_assessments, context, valid_image
    )
    issue_type = _project_issue_type(parsed, atom_assessments, facts)
    object_part = _project_object_part(parsed, atom_assessments)
    claim_status, justification, supporting_ids = _determine_claim_status(
        atom_assessments, evidence_standard_met, facts, parsed
    )
    severity = _project_severity(claim_status, issue_type, facts)
    risk_flags = _collect_risk_flags(
        facts, parsed, context.user_history, atom_assessments, valid_image
    )

    record = DecisionRecord(
        evidence_standard_met=evidence_standard_met,
        evidence_standard_met_reason=esm_reason,
        risk_flags=risk_flags,
        issue_type=issue_type,
        object_part=object_part,
        claim_status=claim_status,
        claim_status_justification=justification,
        supporting_image_ids=supporting_ids,
        valid_image=valid_image,
        severity=severity,
        atom_assessments=atom_assessments,
    )

    logger.info(
        "Row %d: status=%s esm=%s vi=%s sev=%s flags=%s",
        parsed.claim_input.row_index,
        claim_status.value,
        evidence_standard_met,
        valid_image,
        severity.value,
        [f.value for f in risk_flags],
    )
    return record


# ---------------------------------------------------------------------------
# Step 1 — valid_image
# ---------------------------------------------------------------------------


def _assess_valid_image(facts: EvidenceFacts) -> bool:
    """Return True when at least one submitted image is usable for review.

    An image is considered USABLE when ALL of the following hold:
    - ``object_visible = True``
    - ``wrong_object`` is NOT in its quality_issues
    - It is NOT blocking-degraded (fewer than 2 simultaneous core quality issues)

    Returns False only when ALL images are unusable or no images were submitted.

    Confidence values are NOT consulted.
    """
    if not facts.image_observations:
        return False

    for obs in facts.image_observations:
        if _is_usable_image(obs):
            return True
    return False


def _is_usable_image(obs: ImageObservation) -> bool:
    """Return True when a single image is usable for review."""
    if not obs.object_visible:
        return False
    issues = set(obs.quality_issues)
    if issues & _Q_WRONG_OBJECT:
        return False
    # Count how many core issue families are present simultaneously.
    degraded_count = sum(1 for family in _CORE_QUALITY_ISSUES if issues & family)
    return degraded_count < 2


def _is_clear_image(obs: ImageObservation) -> bool:
    """Return True when an image has no quality issues and shows both object and part."""
    return (
        obs.object_visible
        and obs.part_visible
        and not obs.quality_issues
    )


# ---------------------------------------------------------------------------
# Step 2 — atom assessments
# ---------------------------------------------------------------------------


def _assess_atoms(
    parsed: ParsedClaim,
    facts: EvidenceFacts,
) -> list[ClaimAtomAssessment]:
    """Produce a ``ClaimAtomAssessment`` for every ClaimAtom.

    Assessment order:
    1. Collect usable observations (object visible, no wrong_object).
    2. Find findings that match the atom's issue_type_hint.
    3. Compute alignment ('match' | 'mismatch' | 'unclear').
    4. Determine evidence_sufficient.
    """
    assessments: list[ClaimAtomAssessment] = []
    usable_obs = [obs for obs in facts.image_observations if _is_usable_image(obs)]
    clear_obs = [obs for obs in usable_obs if _is_clear_image(obs)]

    for idx, atom in enumerate(parsed.atoms):
        assessment = _assess_single_atom(idx, atom, usable_obs, clear_obs, facts.findings)
        assessments.append(assessment)

    return assessments


def _assess_single_atom(
    idx: int,
    atom: ClaimAtom,
    usable_obs: list[ImageObservation],
    clear_obs: list[ImageObservation],
    findings: list[Finding],
) -> ClaimAtomAssessment:
    """Assess one atom against usable observations and findings."""

    # ---- Supporting image IDs ----
    # An observation "supports" this atom when damage of the same or compatible
    # issue type is observed.
    supporting_ids: list[str] = []
    contradicting_ids: list[str] = []
    damage_obs_ids: list[str] = []  # Any image with damage, regardless of type

    # Match findings to this atom
    matching_findings = _find_matching_findings(atom, findings)
    finding_match = len(matching_findings) > 0

    for obs in usable_obs:
        if obs.damage_observed:
            damage_obs_ids.append(obs.image_id)
            if _observation_matches_atom(obs, atom):
                # Confirmed matching type — supports the claim.
                supporting_ids.append(obs.image_id)
            elif obs.issue_type_observed is not None:
                # Confirmed DIFFERENT type — genuine contradiction.
                # Only when the VLM identified a specific issue type that
                # does not match the claimed type.
                contradicting_ids.append(obs.image_id)
            # else: issue_type_observed=None — damage observed but type
            # undetermined.  This is AMBIGUOUS, not a contradiction.
            # _compute_alignment handles it as the CONFLICTING sub-case.

    # Also add finding-supported images
    for f in matching_findings:
        for iid in f.supporting_image_ids:
            if iid not in supporting_ids:
                supporting_ids.append(iid)

    # ---- Alignment determination ----
    alignment, reasoning = _compute_alignment(
        atom=atom,
        usable_obs=usable_obs,
        clear_obs=clear_obs,
        supporting_ids=supporting_ids,
        contradicting_ids=contradicting_ids,
        finding_match=finding_match,
        damage_obs_ids=damage_obs_ids,
    )

    # ---- Evidence sufficiency ----
    # Sufficient = alignment is 'match' and there is at least one supporting image
    evidence_sufficient = alignment == "match" and len(supporting_ids) > 0

    return ClaimAtomAssessment(
        atom_index=idx,
        atom=atom,
        alignment=alignment,
        evidence_sufficient=evidence_sufficient,
        supporting_image_ids=supporting_ids,
        reasoning=reasoning,
    )


def _find_matching_findings(atom: ClaimAtom, findings: list[Finding]) -> list[Finding]:
    """Return findings whose issue_type matches the atom's issue_type_hint."""
    if atom.issue_type_hint is None:
        return findings  # No hint → any finding is potentially relevant
    return [f for f in findings if f.issue_type == atom.issue_type_hint]


def _observation_matches_atom(obs: ImageObservation, atom: ClaimAtom) -> bool:
    """Return True when an observation's damage type is compatible with the atom.

    Compatibility rules:
    - If atom has no issue_type_hint → any damage observation is a match.
    - If atom has a hint and obs has an issue_type_observed → compare directly.
    - If obs has no issue_type_observed but damage is observed → treat as unclear
      (handled in alignment, not here).
    """
    if atom.issue_type_hint is None:
        return True  # Permissive: any damage is compatible
    if obs.issue_type_observed is None:
        return False  # Unknown type observed — cannot confirm match
    return obs.issue_type_observed == atom.issue_type_hint


def _compute_alignment(
    atom: ClaimAtom,
    usable_obs: list[ImageObservation],
    clear_obs: list[ImageObservation],
    supporting_ids: list[str],
    contradicting_ids: list[str],
    finding_match: bool,
    damage_obs_ids: list[str],
) -> tuple[str, str]:
    """Compute alignment and reasoning for one atom.

    Decision table:
    ┌──────────────────────────────────────────────────────────┬───────────┐
    │ Condition                                                │ Alignment │
    ├──────────────────────────────────────────────────────────┼───────────┤
    │ supporting_ids ≥ 1 (damage type confirmed, matches claim)│ match     │
    │ clear images + NO damage observed anywhere               │ mismatch  │
    │ confirmed wrong type in clear images, no match           │ mismatch  │
    │ clear images show no damage + OTHER images show          │ unclear   │
    │   undetermined damage (Policy B — conflicting signals)   │           │
    │ no usable images                                         │ unclear   │
    │ all usable images have quality issues                    │ unclear   │
    │ damage observed but type undetermined                    │ unclear   │
    └──────────────────────────────────────────────────────────┴───────────┘

    Conservative principle: ambiguity always resolves to 'unclear',
    never to 'mismatch'.  The CONFLICTING sub-case (clear_no_damage AND
    ambiguous_damage) resolves to 'unclear' + MANUAL_REVIEW_REQUIRED rather
    than risking a false contradiction.
    """
    # MATCH: at least one observation (or finding) directly supports the atom.
    if supporting_ids:
        reason = f"Damage matching claimed type observed in {supporting_ids}."
        return "match", reason

    # No supporting evidence — check for contradiction or conflict.
    if not usable_obs:
        return "unclear", "No usable images available for assessment."

    # Classify usable observations with damage by resolution category.
    clear_no_damage = [
        obs for obs in clear_obs
        if not obs.damage_observed and obs.part_visible
    ]
    # Ambiguous damage: damage_observed=True but issue type not identified.
    # These are in damage_obs_ids but NOT in supporting_ids or contradicting_ids.
    ambiguous_damage_ids: list[str] = [
        iid for iid in damage_obs_ids
        if iid not in supporting_ids and iid not in contradicting_ids
    ]

    # MISMATCH case 1: Clear images show the part but NO damage is observed
    # anywhere — neither confirmed type nor ambiguous.  Pure contradiction.
    if clear_no_damage and not damage_obs_ids:
        ids = [obs.image_id for obs in clear_no_damage]
        reason = (
            f"Clear images {ids} show the claimed part with no damage visible. "
            "No other image observes any damage. Claimed damage not substantiated."
        )
        return "mismatch", reason

    # CONFLICTING (Policy B): clear images show no damage, but at least one
    # other image observes damage of undetermined type.  The signals directly
    # conflict.  Committing to 'mismatch' risks a false contradiction if the
    # undetermined damage turns out to be the claimed type.  Conservative
    # resolution: 'unclear' → not_enough_information + MANUAL_REVIEW_REQUIRED.
    if clear_no_damage and ambiguous_damage_ids:
        ids_clear = [obs.image_id for obs in clear_no_damage]
        reason = (
            f"Conflicting signals: clear images {ids_clear} show no damage, "
            f"but images {ambiguous_damage_ids} observe damage of undetermined "
            "type. Manual review required to resolve the conflict."
        )
        return "unclear", reason

    # MISMATCH case 2: At least one image shows confirmed damage of a DIFFERENT
    # type (contradicting_ids), no image supports the claim, and there is at
    # least one clear image (quality is good enough to be definitive).
    if contradicting_ids and not supporting_ids and clear_obs:
        reason = (
            f"Images {contradicting_ids} show confirmed damage of a type "
            "different from the claim; no image supports the claimed type."
        )
        return "mismatch", reason

    # UNCLEAR: quality issues or ambiguous observations prevent determination.
    quality_degraded = [obs for obs in usable_obs if obs.quality_issues]
    if len(quality_degraded) == len(usable_obs):
        reason = "All usable images have quality issues; cannot confirm or contradict claim."
        return "unclear", reason

    if damage_obs_ids and not supporting_ids:
        reason = (
            f"Damage observed in {damage_obs_ids} but type cannot be confirmed "
            "to match the claim."
        )
        return "unclear", reason

    return "unclear", "Insufficient evidence to confirm or contradict the claim."


# ---------------------------------------------------------------------------
# Step 3 — evidence standard
# ---------------------------------------------------------------------------


def _assess_evidence_standard(
    facts: EvidenceFacts,
    assessments: list[ClaimAtomAssessment],
    context: RetrievedContext,
    valid_image: bool,
) -> tuple[bool, str]:
    """Determine whether the submitted images meet the minimum evidence standard.

    Criteria (ALL must hold for True):
    1. At least one usable image (valid_image = True).
    2. At least one atom assessment has alignment ≠ 'unclear'
       (i.e. the evidence is evaluable — either match or mismatch).
    3. The minimum image count in applicable requirements is satisfied.

    Failing criterion 2 while valid_image=True produces False with a
    'wrong_object' or 'wrong_angle' type reason (evidence not evaluable).

    History NEVER influences this determination.
    """
    if not valid_image:
        return False, "No usable images: all submitted images have disqualifying quality issues."

    if not facts.image_observations:
        return False, "No images submitted."

    # Minimum image count check.
    min_images_needed = _minimum_images_required(context)
    usable_count = sum(1 for obs in facts.image_observations if _is_usable_image(obs))
    if usable_count < min_images_needed:
        return (
            False,
            f"Minimum {min_images_needed} usable image(s) required; "
            f"only {usable_count} provided.",
        )

    # At least one atom must be evaluable (not 'unclear').
    evaluable = [a for a in assessments if a.alignment != "unclear"]
    if not evaluable:
        # All unclear — images are present but don't provide usable evidence.
        reason = _build_unclear_reason(facts)
        return False, reason

    return True, _build_met_reason(assessments, usable_count)


def _minimum_images_required(context: RetrievedContext) -> int:
    """Return the minimum number of usable images required for this claim.

    Currently fixed at 1 (the general requirement).  Multi-image requirements
    are advisory and handled via checklist items, not hard gates.
    """
    # REQ_GENERAL_OBJECT_PART applies to all claims and requires 1 usable image.
    return 1


def _build_unclear_reason(facts: EvidenceFacts) -> str:
    """Build the evidence_standard_met_reason when all alignments are unclear."""
    all_issues: list[str] = []
    for obs in facts.image_observations:
        if not obs.object_visible:
            all_issues.append(f"{obs.image_id}: object not visible")
        elif obs.quality_issues:
            all_issues.append(f"{obs.image_id}: quality issues {obs.quality_issues}")
        elif not obs.part_visible:
            all_issues.append(f"{obs.image_id}: claimed part not visible")
    if all_issues:
        return "Evidence not evaluable: " + "; ".join(all_issues) + "."
    return "Evidence not evaluable: images do not provide sufficient context."


def _build_met_reason(assessments: list[ClaimAtomAssessment], usable_count: int) -> str:
    """Build the evidence_standard_met_reason when standard is met."""
    matches = [a for a in assessments if a.alignment == "match"]
    mismatches = [a for a in assessments if a.alignment == "mismatch"]
    if matches:
        return (
            f"{usable_count} usable image(s); "
            f"{len(matches)} atom(s) have confirming observations."
        )
    if mismatches:
        return (
            f"{usable_count} usable image(s); "
            f"clear contradiction detected in {len(mismatches)} atom(s)."
        )
    return f"{usable_count} usable image(s); evidence is evaluable."


# ---------------------------------------------------------------------------
# Step 4 — issue_type projection
# ---------------------------------------------------------------------------


def _project_issue_type(
    parsed: ParsedClaim,
    assessments: list[ClaimAtomAssessment],
    facts: EvidenceFacts,
) -> IssueType:
    """Project a scalar issue_type from atom assessments.

    Priority:
    1. Issue type from a 'match' assessment (highest-severity type wins).
    2. Issue type from the primary atom's hint.
    3. Issue type from any VLM finding (if present).
    4. Issue type from the primary atom's hint regardless of alignment.
    5. IssueType.UNKNOWN.

    Confidence values are NOT consulted.
    """
    # 1. From matching assessments — prefer the highest-severity confirmed type.
    matching = [a for a in assessments if a.alignment == "match"]
    if matching:
        # Collect issue types observed in supporting images of matching atoms.
        candidate_types: list[IssueType] = []
        supporting_obs_ids = {iid for a in matching for iid in a.supporting_image_ids}
        for obs in facts.image_observations:
            if obs.image_id in supporting_obs_ids and obs.issue_type_observed:
                candidate_types.append(obs.issue_type_observed)
        for f in facts.findings:
            if f.issue_type and set(f.supporting_image_ids) & supporting_obs_ids:
                candidate_types.append(f.issue_type)
        if candidate_types:
            return _highest_severity_issue_type(candidate_types)

    # 2. Primary atom hint.
    primary_hint = parsed.primary_atom.issue_type_hint
    if primary_hint:
        return primary_hint

    # 3. From any VLM finding.
    for f in facts.findings:
        if f.issue_type and f.issue_type not in (IssueType.UNKNOWN, IssueType.NONE):
            return f.issue_type

    # 4. Any atom hint.
    for atom in parsed.atoms:
        if atom.issue_type_hint:
            return atom.issue_type_hint

    return IssueType.UNKNOWN


def _highest_severity_issue_type(types: list[IssueType]) -> IssueType:
    """Return the issue type with the highest deterministic severity."""
    return max(
        types,
        key=lambda t: SEVERITY_ORDER.index(ISSUE_SEVERITY_MAP.get(t, Severity.UNKNOWN)),
    )


# ---------------------------------------------------------------------------
# Step 5 — object_part projection
# ---------------------------------------------------------------------------


def _project_object_part(
    parsed: ParsedClaim,
    assessments: list[ClaimAtomAssessment],
) -> ObjectPart:
    """Project a scalar object_part from the primary atom assessment.

    Falls back to the primary atom's part hint, then UNKNOWN.
    """
    primary_hint = parsed.primary_atom.object_part_hint
    if primary_hint:
        return primary_hint
    # Check matching assessments for any part hint.
    for a in assessments:
        if a.alignment == "match" and a.atom.object_part_hint:
            return a.atom.object_part_hint
    return ObjectPart.UNKNOWN


# ---------------------------------------------------------------------------
# Step 6 — claim_status
# ---------------------------------------------------------------------------


def _determine_claim_status(
    assessments: list[ClaimAtomAssessment],
    evidence_standard_met: bool,
    facts: EvidenceFacts,
    parsed: ParsedClaim,
) -> tuple[ClaimStatus, str, list[str]]:
    """Determine claim_status, justification, and supporting_image_ids.

    Decision table:
    ┌─────────────────────────────────────────────────────────┬───────────────────────┐
    │ Condition                                               │ claim_status          │
    ├─────────────────────────────────────────────────────────┼───────────────────────┤
    │ evidence_standard_met = False                           │ not_enough_information│
    │ evidence_standard_met = True                            │                       │
    │   AND any atom has alignment = 'match'                  │ supported             │
    │   AND no atom has alignment = 'mismatch'                │                       │
    │ evidence_standard_met = True                            │                       │
    │   AND any atom has alignment = 'mismatch'               │ contradicted          │
    │   AND no atom has alignment = 'match'                   │                       │
    │ evidence_standard_met = True                            │ contradicted          │
    │   AND mismatch atoms > match atoms                      │ (majority mismatch)   │
    │ evidence_standard_met = True                            │ supported             │
    │   AND match atoms >= mismatch atoms                     │ (majority match)      │
    │   AND match atoms > 0                                   │                       │
    └─────────────────────────────────────────────────────────┴───────────────────────┘

    Ambiguity resolution:
    - 'unclear'-only assessments → not_enough_information (already handled by esm=False)
    - Mixed match/mismatch → majority wins; ties go to 'supported' (conservative)
    - 'unclear' alignments are ignored in the majority count

    Invariants:
    - 'contradicted' requires at least one 'mismatch' alignment (never from 'unclear').
    - 'supported' requires evidence_standard_met = True (enforced by schema).
    """
    if not evidence_standard_met:
        reason = _build_not_enough_reason(facts, assessments)
        return ClaimStatus.NOT_ENOUGH_INFORMATION, reason, []

    match_assessments = [a for a in assessments if a.alignment == "match"]
    mismatch_assessments = [a for a in assessments if a.alignment == "mismatch"]

    match_count = len(match_assessments)
    mismatch_count = len(mismatch_assessments)

    supporting_ids = _collect_supporting_ids(match_assessments, mismatch_assessments)

    # Pure match (or majority match).
    if match_count > 0 and match_count >= mismatch_count:
        reason = _build_supported_reason(match_assessments, facts)
        return ClaimStatus.SUPPORTED, reason, supporting_ids

    # Pure mismatch or majority mismatch.
    if mismatch_count > 0:
        reason = _build_contradicted_reason(mismatch_assessments, facts)
        return ClaimStatus.CONTRADICTED, reason, supporting_ids

    # Fallback: evaluable but no clear alignment — should be covered by esm=False,
    # but as a safety net:
    reason = "Evidence is evaluable but does not clearly confirm or contradict the claim."
    return ClaimStatus.NOT_ENOUGH_INFORMATION, reason, []


def _collect_supporting_ids(
    match_assessments: list[ClaimAtomAssessment],
    mismatch_assessments: list[ClaimAtomAssessment],
) -> list[str]:
    """Collect supporting image IDs for the decision.

    For SUPPORTED: images that support the match.
    For CONTRADICTED: images that support the contradiction (the mismatch images).
    """
    ids: list[str] = []
    for a in match_assessments:
        for iid in a.supporting_image_ids:
            if iid not in ids:
                ids.append(iid)
    for a in mismatch_assessments:
        for iid in a.supporting_image_ids:
            if iid not in ids:
                ids.append(iid)
    return ids


def _build_supported_reason(
    match_assessments: list[ClaimAtomAssessment],
    facts: EvidenceFacts,
) -> str:
    parts = []
    for a in match_assessments:
        if a.supporting_image_ids:
            parts.append(
                f"Atom '{a.atom.described_issue} on {a.atom.described_part}' "
                f"confirmed by {a.supporting_image_ids}."
            )
    if not parts:
        return "Claimed damage is consistent with observed evidence."
    return " ".join(parts)


def _build_contradicted_reason(
    mismatch_assessments: list[ClaimAtomAssessment],
    facts: EvidenceFacts,
) -> str:
    parts = []
    for a in mismatch_assessments:
        parts.append(a.reasoning)
    return " ".join(parts) if parts else "Observed evidence contradicts the claimed damage."


def _build_not_enough_reason(
    facts: EvidenceFacts,
    assessments: list[ClaimAtomAssessment],
) -> str:
    if not facts.image_observations:
        return "No images submitted."
    issues = []
    for obs in facts.image_observations:
        if not obs.object_visible:
            issues.append(f"{obs.image_id}: claimed object not visible")
        elif not obs.part_visible:
            issues.append(f"{obs.image_id}: claimed part not visible")
        elif obs.quality_issues:
            issues.append(f"{obs.image_id}: {obs.quality_issues}")
    if issues:
        return "Insufficient evidence: " + "; ".join(issues) + "."
    return "Insufficient evidence to evaluate the claim."


# ---------------------------------------------------------------------------
# Step 7 — severity projection
# ---------------------------------------------------------------------------


def _project_severity(
    claim_status: ClaimStatus,
    issue_type: IssueType,
    facts: EvidenceFacts,
) -> Severity:
    """Determine severity deterministically.

    Rules (in priority order):
    1. If claim_status = not_enough_information → Severity.UNKNOWN
       (cannot determine severity without evaluable evidence).
    2. If claim_status = contradicted AND issue_type = NONE or UNKNOWN:
       - If damage of any type is visible → use that severity.
       - Otherwise → Severity.NONE (damage claimed but not observed).
    3. Start from ISSUE_SEVERITY_MAP[issue_type] (deterministic baseline).
    4. If baseline is UNKNOWN, consult VLM severity_estimate from findings
       as advisory tiebreaker (still deterministic: take the most severe).
    5. For contradicted claims: use the actually-observed severity (from findings),
       not the claimed issue type's default.

    Confidence values are NOT consulted.
    """
    if claim_status == ClaimStatus.NOT_ENOUGH_INFORMATION:
        return Severity.UNKNOWN

    if claim_status == ClaimStatus.CONTRADICTED:
        return _severity_for_contradicted(issue_type, facts)

    # SUPPORTED path: use issue_type from ISSUE_SEVERITY_MAP.
    baseline = ISSUE_SEVERITY_MAP.get(issue_type, Severity.UNKNOWN)

    if baseline != Severity.UNKNOWN:
        # Pure deterministic: no VLM advisory needed.
        return baseline

    # Advisory tiebreaker: VLM severity_estimate from findings.
    advisory = _best_advisory_severity(facts)
    return advisory if advisory != Severity.UNKNOWN else Severity.UNKNOWN


def _severity_for_contradicted(issue_type: IssueType, facts: EvidenceFacts) -> Severity:
    """Severity for contradicted claims.

    For contradicted claims:
    - If observed damage is 'none' (clean images show nothing) → Severity.NONE.
    - If observed damage is of a different type → use that type's severity.
    - If no damage observed at all → Severity.NONE.
    """
    # Collect actually-observed issue types from usable images.
    observed_types: list[IssueType] = []
    for obs in facts.image_observations:
        if _is_usable_image(obs) and obs.issue_type_observed:
            t = obs.issue_type_observed
            if t not in (IssueType.NONE, IssueType.UNKNOWN):
                observed_types.append(t)
    for f in facts.findings:
        if f.issue_type and f.issue_type not in (IssueType.NONE, IssueType.UNKNOWN):
            observed_types.append(f.issue_type)

    if not observed_types:
        # No damage observed — the contradiction is "nothing there".
        return Severity.NONE

    # Use the highest severity of actually-observed damage.
    return most_severe([ISSUE_SEVERITY_MAP.get(t, Severity.UNKNOWN) for t in observed_types])


def _best_advisory_severity(facts: EvidenceFacts) -> Severity:
    """Collect the best (most severe) advisory severity_estimate from findings."""
    estimates: list[Severity] = []
    for f in facts.findings:
        if f.severity_estimate and f.severity_estimate != Severity.UNKNOWN:
            estimates.append(f.severity_estimate)
    for obs in facts.image_observations:
        if obs.severity_estimate and obs.severity_estimate != Severity.UNKNOWN:
            estimates.append(obs.severity_estimate)
    return most_severe(estimates) if estimates else Severity.UNKNOWN


# ---------------------------------------------------------------------------
# Step 8 — risk flags
# ---------------------------------------------------------------------------


def _collect_risk_flags(
    facts: EvidenceFacts,
    parsed: ParsedClaim,
    history: Optional[UserHistory],
    assessments: list[ClaimAtomAssessment],
    valid_image: bool,
) -> list[RiskFlag]:
    """Collect all applicable risk flags.

    Sources:
    - Image quality signals (from EvidenceFacts).
    - Claim sanitization signals (from ParsedClaim.sanitization).
    - Observation signals (wrong object, text instructions, damage not visible).
    - Alignment signals (claim mismatch).
    - History signals (from UserHistory — advisory, never gate claim_status).

    Deduplication: each flag appears at most once.
    """
    flags: set[RiskFlag] = set()

    # ---- Image quality flags ----
    for obs in facts.image_observations:
        issues = set(obs.quality_issues)
        if issues & _Q_BLURRY:
            flags.add(RiskFlag.BLURRY_IMAGE)
        if issues & (_Q_LOW_LIGHT | _Q_GLARE):
            flags.add(RiskFlag.LOW_LIGHT_OR_GLARE)
        if issues & _Q_WRONG_ANGLE:
            flags.add(RiskFlag.WRONG_ANGLE)
        if issues & _Q_CROPPED:
            flags.add(RiskFlag.CROPPED_OR_OBSTRUCTED)
        if issues & _Q_WRONG_OBJECT:
            flags.add(RiskFlag.WRONG_OBJECT)

    # ---- Observation content flags ----
    for obs in facts.image_observations:
        if obs.text_or_instructions_present:
            flags.add(RiskFlag.TEXT_INSTRUCTION_PRESENT)
        # Wrong object: visible object doesn't match claim object type.
        if obs.object_visible and obs.object_type_observed:
            # We can't do semantic matching here, so we rely on the VLM flag.
            if "wrong_object" in obs.quality_issues:
                flags.add(RiskFlag.WRONG_OBJECT)

    # ---- Damage not visible flag ----
    # Fire when: object is visible in clear images, and those clear images show NO damage.
    # This correctly catches both:
    # 1. Pure contradiction (all images clear, no damage)
    # 2. Conflicting (Policy B: clear image shows no damage, blurry image shows damage)
    has_clear_visible = any(
        obs.object_visible and not obs.quality_issues
        for obs in facts.image_observations
    )
    any_damage_in_clear_images = any(
        obs.damage_observed and not obs.quality_issues
        for obs in facts.image_observations
    )
    if has_clear_visible and not any_damage_in_clear_images:
        flags.add(RiskFlag.DAMAGE_NOT_VISIBLE)

    # ---- Claim mismatch flag (from atom alignments) ----
    has_mismatch = any(a.alignment == "mismatch" for a in assessments)
    if has_mismatch:
        flags.add(RiskFlag.CLAIM_MISMATCH)

    # ---- Part mismatch flag ----
    for obs in facts.image_observations:
        if (obs.part_visible and obs.part_observed and
                parsed.primary_atom.object_part_hint):
            # Advisory: if the VLM describes a different part than claimed.
            # We can only do this heuristically (string check).
            pass  # Structural comparison deferred to future enhancement.

    # ---- Sanitization / injection flags ----
    if parsed.sanitization.injection_detected:
        flags.add(RiskFlag.POSSIBLE_MANIPULATION)
    if "text_instruction_present" in parsed.sanitization.flags:
        flags.add(RiskFlag.TEXT_INSTRUCTION_PRESENT)

    # ---- History flags (advisory — NEVER influence claim_status) ----
    if history is not None:
        history_flags = _compute_history_flags(history)
        flags.update(history_flags)

    # ---- Manual review composite flag ----
    # Fired when any condition warrants human escalation.
    # DAMAGE_NOT_VISIBLE is included because it fires in both:
    #   - Pure contradictions (clear image, zero damage anywhere → contradicted)
    #   - Conflicting evidence (clear image + ambiguous damage → unclear/NEI)
    # In both cases, human review is appropriate per the sample data pattern.
    _MANUAL_REVIEW_TRIGGERS: frozenset[RiskFlag] = frozenset({
        RiskFlag.POSSIBLE_MANIPULATION,
        RiskFlag.TEXT_INSTRUCTION_PRESENT,
        RiskFlag.WRONG_OBJECT,
        RiskFlag.CLAIM_MISMATCH,
        RiskFlag.USER_HISTORY_RISK,
        RiskFlag.NON_ORIGINAL_IMAGE,
        RiskFlag.DAMAGE_NOT_VISIBLE,  # Policy B: conflicting evidence → manual review
    })
    if flags & _MANUAL_REVIEW_TRIGGERS:
        flags.add(RiskFlag.MANUAL_REVIEW_REQUIRED)

    # Return sorted list for deterministic output ordering.
    return _sort_flags(flags)


def _compute_history_flags(history: UserHistory) -> set[RiskFlag]:
    """Derive history-based risk flags.

    Thresholds:
    - rejected_claim / past_claim_count ≥ 30% (with ≥3 past claims) → USER_HISTORY_RISK
    - last_90_days_claim_count ≥ 3 → USER_HISTORY_RISK (frequent claimant)
    - history_flags contains known risk keywords → USER_HISTORY_RISK
    - manual_review_claim > 0 AND rejected_claim > 0 → USER_HISTORY_RISK

    History NEVER determines claim_status.  These flags are advisory only.
    """
    flags: set[RiskFlag] = set()

    # Raw history flags from CSV.
    if any(kw in f for f in history.history_flags for kw in _HISTORY_FLAG_KEYWORDS):
        flags.add(RiskFlag.USER_HISTORY_RISK)

    # Rejection rate.
    if (
        history.past_claim_count >= _HISTORY_MIN_CLAIMS_FOR_RATE
        and history.past_claim_count > 0
        and history.rejected_claim / history.past_claim_count >= _HISTORY_HIGH_REJECT_RATE_THRESHOLD
    ):
        flags.add(RiskFlag.USER_HISTORY_RISK)

    # Frequent claimant.
    if history.last_90_days_claim_count >= _HISTORY_RECENT_CLAIM_THRESHOLD:
        flags.add(RiskFlag.USER_HISTORY_RISK)

    # Manual + rejected history together.
    if history.manual_review_claim > 0 and history.rejected_claim > 0:
        flags.add(RiskFlag.USER_HISTORY_RISK)

    return flags


# ---------------------------------------------------------------------------
# Output ordering helper
# ---------------------------------------------------------------------------

# Canonical flag output order (determines semicolon-separated string in CSV).
_FLAG_ORDER: tuple[RiskFlag, ...] = (
    RiskFlag.BLURRY_IMAGE,
    RiskFlag.CROPPED_OR_OBSTRUCTED,
    RiskFlag.LOW_LIGHT_OR_GLARE,
    RiskFlag.WRONG_ANGLE,
    RiskFlag.WRONG_OBJECT,
    RiskFlag.WRONG_OBJECT_PART,
    RiskFlag.DAMAGE_NOT_VISIBLE,
    RiskFlag.CLAIM_MISMATCH,
    RiskFlag.POSSIBLE_MANIPULATION,
    RiskFlag.NON_ORIGINAL_IMAGE,
    RiskFlag.TEXT_INSTRUCTION_PRESENT,
    RiskFlag.USER_HISTORY_RISK,
    RiskFlag.MANUAL_REVIEW_REQUIRED,
    RiskFlag.NONE,
)

_FLAG_RANK: dict[RiskFlag, int] = {f: i for i, f in enumerate(_FLAG_ORDER)}


def _sort_flags(flags: set[RiskFlag]) -> list[RiskFlag]:
    """Return flags in canonical output order, deduped."""
    return sorted(flags, key=lambda f: _FLAG_RANK.get(f, 999))
