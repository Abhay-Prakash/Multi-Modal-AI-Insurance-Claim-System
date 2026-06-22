"""
Frozen data contracts and enums for the multimodal evidence review system.

Architecture invariants enforced by these schemas:
- EvidenceFacts contains observations ONLY (image_observations + findings).
  No alignment, no sufficiency, no decisions.
- Alignment and evidence sufficiency belong exclusively to rules.py
  via ClaimAtomAssessment.
- Confidence scores are ADVISORY signals annotated as such in field
  descriptions. They must never be primary decision gates.
- Schema validation may repair formatting only; it must NEVER invent
  observations or evidence.
- History-derived risk flags must NEVER directly determine claim_status.
- sanitization.py annotates injection spans with [UNTRUSTED_TEXT]...[/UNTRUSTED_TEXT].
- parsing.py extracts ClaimAtom[] and selects a primary atom index.

All schemas in this file are FROZEN. Do not modify without explicit approval.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums — values match problem_statement.md exactly
# ---------------------------------------------------------------------------


class ClaimObject(str, Enum):
    """The type of object being claimed."""

    CAR = "car"
    LAPTOP = "laptop"
    PACKAGE = "package"


class IssueType(str, Enum):
    """Visible damage issue type.

    Use ``none`` when the relevant part is visible and undamaged.
    Use ``unknown`` when the issue type cannot be determined from images.
    """

    DENT = "dent"
    SCRATCH = "scratch"
    CRACK = "crack"
    GLASS_SHATTER = "glass_shatter"
    BROKEN_PART = "broken_part"
    MISSING_PART = "missing_part"
    TORN_PACKAGING = "torn_packaging"
    CRUSHED_PACKAGING = "crushed_packaging"
    WATER_DAMAGE = "water_damage"
    STAIN = "stain"
    NONE = "none"
    UNKNOWN = "unknown"


class ObjectPart(str, Enum):
    """All valid object parts across all claim object types.

    Use ``VALID_PARTS_BY_OBJECT`` to validate a part against a specific
    ``ClaimObject``.  Use ``unknown`` when the part cannot be determined.
    """

    # Car parts
    FRONT_BUMPER = "front_bumper"
    REAR_BUMPER = "rear_bumper"
    DOOR = "door"
    HOOD = "hood"
    WINDSHIELD = "windshield"
    SIDE_MIRROR = "side_mirror"
    HEADLIGHT = "headlight"
    TAILLIGHT = "taillight"
    FENDER = "fender"
    QUARTER_PANEL = "quarter_panel"
    # Laptop parts
    SCREEN = "screen"
    KEYBOARD = "keyboard"
    TRACKPAD = "trackpad"
    HINGE = "hinge"
    LID = "lid"
    CORNER = "corner"
    PORT = "port"
    BASE = "base"
    # Package parts
    BOX = "box"
    PACKAGE_CORNER = "package_corner"
    PACKAGE_SIDE = "package_side"
    SEAL = "seal"
    LABEL = "label"
    CONTENTS = "contents"
    ITEM = "item"
    # Shared across multiple object types
    BODY = "body"
    UNKNOWN = "unknown"


class ClaimStatus(str, Enum):
    """Final claim decision produced by the rule engine."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_ENOUGH_INFORMATION = "not_enough_information"


class Severity(str, Enum):
    """Damage severity.

    Determined deterministically by ``rules.py``.  The VLM's
    ``severity_estimate`` field is advisory only.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class RiskFlag(str, Enum):
    """Risk flags attached to a claim decision.

    History-derived flags (e.g. ``user_history_risk``) contribute to this
    field only.  They must never influence ``claim_status``.
    """

    NONE = "none"
    BLURRY_IMAGE = "blurry_image"
    CROPPED_OR_OBSTRUCTED = "cropped_or_obstructed"
    LOW_LIGHT_OR_GLARE = "low_light_or_glare"
    WRONG_ANGLE = "wrong_angle"
    WRONG_OBJECT = "wrong_object"
    WRONG_OBJECT_PART = "wrong_object_part"
    DAMAGE_NOT_VISIBLE = "damage_not_visible"
    CLAIM_MISMATCH = "claim_mismatch"
    POSSIBLE_MANIPULATION = "possible_manipulation"
    NON_ORIGINAL_IMAGE = "non_original_image"
    TEXT_INSTRUCTION_PRESENT = "text_instruction_present"
    USER_HISTORY_RISK = "user_history_risk"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


VALID_PARTS_BY_OBJECT: dict[ClaimObject, frozenset[ObjectPart]] = {
    ClaimObject.CAR: frozenset({
        ObjectPart.FRONT_BUMPER, ObjectPart.REAR_BUMPER, ObjectPart.DOOR,
        ObjectPart.HOOD, ObjectPart.WINDSHIELD, ObjectPart.SIDE_MIRROR,
        ObjectPart.HEADLIGHT, ObjectPart.TAILLIGHT, ObjectPart.FENDER,
        ObjectPart.QUARTER_PANEL, ObjectPart.BODY, ObjectPart.UNKNOWN,
    }),
    ClaimObject.LAPTOP: frozenset({
        ObjectPart.SCREEN, ObjectPart.KEYBOARD, ObjectPart.TRACKPAD,
        ObjectPart.HINGE, ObjectPart.LID, ObjectPart.CORNER,
        ObjectPart.PORT, ObjectPart.BASE, ObjectPart.BODY, ObjectPart.UNKNOWN,
    }),
    ClaimObject.PACKAGE: frozenset({
        ObjectPart.BOX, ObjectPart.PACKAGE_CORNER, ObjectPart.PACKAGE_SIDE,
        ObjectPart.SEAL, ObjectPart.LABEL, ObjectPart.CONTENTS,
        ObjectPart.ITEM, ObjectPart.UNKNOWN,
    }),
}

# Ordered from least to most severe for deterministic projection.
SEVERITY_ORDER: tuple[Severity, ...] = (
    Severity.NONE,
    Severity.UNKNOWN,
    Severity.LOW,
    Severity.MEDIUM,
    Severity.HIGH,
)

# Issue types that imply structural severity levels for deterministic
# severity determination in rules.py (advisory; overrides VLM estimate).
ISSUE_SEVERITY_MAP: dict[IssueType, Severity] = {
    IssueType.NONE: Severity.NONE,
    IssueType.SCRATCH: Severity.LOW,
    IssueType.STAIN: Severity.LOW,
    IssueType.DENT: Severity.MEDIUM,
    IssueType.CRACK: Severity.MEDIUM,
    IssueType.WATER_DAMAGE: Severity.MEDIUM,
    IssueType.TORN_PACKAGING: Severity.MEDIUM,
    IssueType.BROKEN_PART: Severity.HIGH,
    IssueType.GLASS_SHATTER: Severity.HIGH,
    IssueType.MISSING_PART: Severity.HIGH,
    IssueType.CRUSHED_PACKAGING: Severity.HIGH,
    IssueType.UNKNOWN: Severity.UNKNOWN,
}


def is_valid_part_for_object(part: ObjectPart, claim_object: ClaimObject) -> bool:
    """Return True if *part* is a valid part for *claim_object*."""
    return part in VALID_PARTS_BY_OBJECT.get(claim_object, frozenset())


def most_severe(severities: list[Severity]) -> Severity:
    """Return the highest severity from a list using ``SEVERITY_ORDER``."""
    if not severities:
        return Severity.UNKNOWN
    return max(severities, key=lambda s: SEVERITY_ORDER.index(s))


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


class ImageRef(BaseModel):
    """A resolved reference to a single submitted image file.

    Produced by ``ingestion.py``.
    """

    image_id: str = Field(
        ...,
        description="Filename without extension, e.g. 'img_1'. Used as the "
                    "external image identifier in output fields.",
    )
    path: str = Field(
        ...,
        description="Absolute path to the image file on disk.",
    )
    relative_path: str = Field(
        ...,
        description="Original relative path as it appears in the CSV, e.g. "
                    "'images/test/case_001/img_1.jpg'.",
    )
    case_id: str = Field(
        ...,
        description="Case folder name derived from the path, e.g. 'case_001'.",
    )
    split: str = Field(
        ...,
        description="Dataset split derived from the path: 'sample' or 'test'.",
    )


class ClaimInput(BaseModel):
    """A single raw claim row as loaded from the CSV.

    Produced by ``ingestion.py``.  This is the entry point into the pipeline.
    All downstream modules operate on this or derived structures.
    """

    user_id: str = Field(..., description="User identifier; used to look up user_history.csv.")
    image_paths_raw: str = Field(
        ...,
        description="Original semicolon-separated image paths string from the CSV.",
    )
    images: list[ImageRef] = Field(
        default_factory=list,
        description="Resolved image references.  May be empty if all paths fail to resolve.",
    )
    user_claim: str = Field(
        ...,
        description="Raw pipe-separated conversation transcript from the CSV.",
    )
    claim_object: ClaimObject = Field(..., description="Type of object being claimed.")
    row_index: int = Field(..., ge=0, description="0-based row position in the source CSV.")


class ConversationTurn(BaseModel):
    """A single turn in the claim conversation.

    Produced by ``parsing.py`` when splitting the pipe-delimited ``user_claim``.
    """

    turn_index: int = Field(..., ge=0, description="0-based position of this turn.")
    speaker_raw: str = Field(
        ...,
        description="Speaker label exactly as it appears in the text, "
                    "e.g. 'Customer', 'Agent', 'Support', 'Cliente'.",
    )
    speaker_role: str = Field(
        ...,
        description="Normalised speaker role: 'customer' or 'agent'.",
    )
    text: str = Field(..., description="The message body, stripped of the speaker prefix.")


class ClaimAtom(BaseModel):
    """A single damage claim extracted from the conversation.

    A multi-damage conversation such as 'screen cracked and keyboard broken'
    produces two ``ClaimAtom`` instances.  Single-damage conversations produce
    one.  ``parsing.py`` guarantees at least one atom is always returned.

    Hints are populated heuristically by ``parsing.py`` from keyword matching;
    they must not be treated as ground truth.
    """

    described_issue: str = Field(
        ...,
        description="The damage as described in the conversation (free text, "
                    "extracted verbatim or paraphrased from customer turns).",
    )
    described_part: str = Field(
        ...,
        description="The object part as described in the conversation (free "
                    "text, extracted verbatim or paraphrased from customer turns).",
    )
    issue_type_hint: Optional[IssueType] = Field(
        None,
        description="Best-guess normalised issue type derived from keyword "
                    "matching in parsing.py.  Absent when undetermined.  "
                    "Must not be used as ground truth.",
    )
    object_part_hint: Optional[ObjectPart] = Field(
        None,
        description="Best-guess normalised object part derived from keyword "
                    "matching in parsing.py.  Absent when undetermined.  "
                    "Must not be used as ground truth.",
    )


class SanitizationResult(BaseModel):
    """Output of the sanitization layer for a single claim.

    Produced by ``sanitization.py`` before ``parsing.py`` and ``grounding.py``
    consume the claim text.

    Injection spans are annotated with::

        [UNTRUSTED_TEXT]<suspicious span>[/UNTRUSTED_TEXT]

    The annotation is purely informational.  Claim semantics are preserved.
    """

    sanitized_claim: str = Field(
        ...,
        description="Claim text with detected injection spans wrapped in "
                    "[UNTRUSTED_TEXT]...[/UNTRUSTED_TEXT] markers.  "
                    "Equals the original text when no injection is detected.",
    )
    injection_detected: bool = Field(
        False,
        description="True when at least one injection pattern was matched.",
    )
    injection_spans: list[str] = Field(
        default_factory=list,
        description="The raw matched spans that triggered the annotation.  "
                    "Preserved for auditing and logging.  Must not be logged "
                    "if they contain user PII.",
    )
    flags: list[str] = Field(
        default_factory=list,
        description="Sanitization signal flags, e.g. ['text_instruction_present']. "
                    "Consumed by rules.py when building risk_flags.",
    )


class ParsedClaim(BaseModel):
    """A claim after sanitization, conversation splitting, and atom extraction.

    Produced by ``parsing.py``.  Consumed by ``retrieval.py``,
    ``grounding.py``, and ``rules.py``.

    Invariants:
    - ``atoms`` is non-empty (parsing.py always produces at least one atom).
    - ``primary_atom_index`` is a valid index into ``atoms``.
    - ``raw_text`` equals ``claim_input.user_claim`` exactly.
    """

    claim_input: ClaimInput = Field(..., description="The original raw claim row.")
    raw_text: str = Field(
        ...,
        description="Original user_claim text, unmodified.  Preserved for "
                    "downstream logging and auditing.",
    )
    conversation_turns: list[ConversationTurn] = Field(
        default_factory=list,
        description="Parsed conversation turns in order.",
    )
    atoms: list[ClaimAtom] = Field(
        ...,
        min_length=1,
        description="Extracted damage claims.  At least one is always present.",
    )
    primary_atom_index: int = Field(
        0,
        ge=0,
        description="Index of the primary (most explicitly stated) ClaimAtom "
                    "in atoms[].  Used as the projection target when collapsing "
                    "multi-atom decisions to scalar output fields.",
    )
    conversation_language: str = Field(
        "en",
        description="Detected primary language of the conversation (ISO 639-1). "
                    "Informational; does not affect decision logic.",
    )
    sanitization: SanitizationResult = Field(
        ...,
        description="Sanitization result from sanitization.py.  Carries the "
                    "annotated text and injection flags.",
    )

    @model_validator(mode="after")
    def primary_index_in_bounds(self) -> "ParsedClaim":
        """Ensure primary_atom_index is a valid index into atoms."""
        if self.primary_atom_index >= len(self.atoms):
            raise ValueError(
                f"primary_atom_index ({self.primary_atom_index}) is out of "
                f"range for atoms list of length {len(self.atoms)}"
            )
        return self

    @property
    def primary_atom(self) -> ClaimAtom:
        """Convenience accessor for the primary ClaimAtom."""
        return self.atoms[self.primary_atom_index]


# ---------------------------------------------------------------------------
# Retrieval models
# ---------------------------------------------------------------------------


class UserHistory(BaseModel):
    """A single user history record from ``user_history.csv``.

    Produced by ``ingestion.py``.  Consumed by ``retrieval.py`` and
    ``rules.py`` for risk flag derivation only.

    Invariant: history information must NEVER determine ``claim_status``.
    """

    user_id: str
    past_claim_count: int = Field(0, ge=0)
    accept_claim: int = Field(0, ge=0)
    manual_review_claim: int = Field(0, ge=0)
    rejected_claim: int = Field(0, ge=0)
    last_90_days_claim_count: int = Field(0, ge=0)
    history_flags: list[str] = Field(
        default_factory=list,
        description="Raw flag strings from the CSV, e.g. ['user_history_risk', "
                    "'manual_review_required'].",
    )
    history_summary: str = Field(
        "",
        description="Human-readable summary from the CSV.  Used for logging only.",
    )


class EvidenceRequirement(BaseModel):
    """A single evidence requirement row from ``evidence_requirements.csv``.

    Produced by ``ingestion.py``.  Consumed by ``retrieval.py`` and
    ``grounding.py`` for checklist construction.
    """

    requirement_id: str = Field(..., description="Unique identifier, e.g. 'REQ_CAR_BODY_PANEL'.")
    claim_object: str = Field(
        ...,
        description="Applicable claim object: 'car', 'laptop', 'package', or 'all'.",
    )
    applies_to: str = Field(
        ...,
        description="Issue family this requirement targets, "
                    "e.g. 'dent or scratch', 'screen damage'.",
    )
    minimum_image_evidence: str = Field(
        ...,
        description="Prose description of the minimum visual evidence needed "
                    "to evaluate a claim of this type.",
    )


class EvidenceChecklistItem(BaseModel):
    """A single item in the explicit evidence checklist.

    Produced by ``grounding.py`` from ``EvidenceRequirement`` records and
    ``ClaimAtom`` hints.  Consumed by the VLM prompt (Strategy A) and by
    ``rules.py`` for sufficiency checking.
    """

    requirement_id: str = Field(..., description="Source requirement ID.")
    check_description: str = Field(
        ...,
        description="What the VLM (or reviewer) should look for in the images.",
    )
    applies_to_atom_index: Optional[int] = Field(
        None,
        ge=0,
        description="Index into ParsedClaim.atoms that this item specifically "
                    "targets, if applicable.  None means the item applies to "
                    "all atoms / the claim as a whole.",
    )


class EvidenceChecklist(BaseModel):
    """Explicit evidence checklist constructed by ``grounding.py``.

    Built from applicable ``EvidenceRequirement`` records and ``ClaimAtom``
    hints before each VLM call.

    In Strategy A, the full checklist is injected into the VLM prompt.
    In Strategy B, the checklist is omitted from the prompt but still used
    by ``rules.py`` for sufficiency checking.
    """

    items: list[EvidenceChecklistItem] = Field(default_factory=list)
    source_requirements: list[EvidenceRequirement] = Field(default_factory=list)


class RetrievedContext(BaseModel):
    """All deterministically retrieved context for a single claim.

    Produced by ``retrieval.py``.  Consumed by ``grounding.py`` and
    ``rules.py``.

    Retrieval is purely deterministic (exact match + filtering).
    No vector databases or embedding lookups are used.
    """

    user_history: Optional[UserHistory] = Field(
        None,
        description="User history record for claim.user_id.  None if the "
                    "user is not found in user_history.csv.",
    )
    applicable_requirements: list[EvidenceRequirement] = Field(
        default_factory=list,
        description="Requirements matching the claim's object type (including "
                    "'all' requirements).",
    )
    evidence_checklist: EvidenceChecklist = Field(
        default_factory=EvidenceChecklist,
        description="Explicit checklist built from requirements and claim atoms "
                    "by grounding.py after retrieval.",
    )


# ---------------------------------------------------------------------------
# VLM output models — OBSERVATIONS ONLY
#
# These models represent what the VLM observes in the submitted images.
# They contain NO decisions, NO alignment assessments, and NO sufficiency
# judgments.  Those belong exclusively to rules.py.
#
# Advisory fields (confidence, severity_estimate) are retained for logging
# and tie-breaking purposes ONLY.  They must never gate primary decisions.
# ---------------------------------------------------------------------------


class ImageObservation(BaseModel):
    """Per-image observation from the VLM.

    Contains observations only.  No decisions.

    Advisory fields:
    - ``confidence``: VLM self-reported certainty.  ADVISORY ONLY.
      Must not appear in any ``if confidence >= threshold`` decision branch.
    - ``severity_estimate``: VLM damage estimate.  ADVISORY ONLY.
      Final severity is determined deterministically by ``rules.py``.
    """

    image_id: str = Field(
        ...,
        description="Image identifier matching ImageRef.image_id, e.g. 'img_1'.",
    )
    object_visible: bool = Field(
        ...,
        description="Whether the claimed object type (car / laptop / package) "
                    "is visible in this image.",
    )
    object_type_observed: Optional[str] = Field(
        None,
        description="Free-text description of what object type is actually "
                    "visible (may differ from claimed type).",
    )
    part_visible: bool = Field(
        ...,
        description="Whether the claimed object part is visible in this image.",
    )
    part_observed: Optional[str] = Field(
        None,
        description="Free-text description of what part is actually visible.",
    )
    damage_observed: bool = Field(
        ...,
        description="Whether any damage is visible in this image.",
    )
    damage_description: Optional[str] = Field(
        None,
        description="Free-text description of observed damage.  Absent when "
                    "damage_observed is False.",
    )
    issue_type_observed: Optional[IssueType] = Field(
        None,
        description="Closest matching issue type for the observed damage.  "
                    "Absent when damage_observed is False.",
    )
    severity_estimate: Optional[Severity] = Field(
        None,
        description="ADVISORY ONLY.  VLM's estimate of damage severity.  "
                    "Final severity is determined deterministically by rules.py "
                    "and must not be copied directly from this field.",
    )
    quality_issues: list[str] = Field(
        default_factory=list,
        description="Observed image quality issues, e.g. 'blurry', 'low_light', "
                    "'wrong_angle', 'cropped', 'glare'.  Used by rules.py to "
                    "compute valid_image and risk_flags.",
    )
    text_or_instructions_present: bool = Field(
        False,
        description="True when the image contains embedded text that appears "
                    "to be instructions or manipulation artifacts.",
    )
    confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="ADVISORY ONLY.  VLM self-reported confidence in its "
                    "observations for this image.  Must not gate any primary "
                    "decision.  May be used for logging and tie-breaking only.",
    )


class Finding(BaseModel):
    """A cross-image synthesised observation from the VLM.

    Represents a coherent finding that may be supported by multiple images.
    Observations only — no decisions.

    Advisory fields follow the same constraints as in ``ImageObservation``.
    """

    description: str = Field(
        ...,
        description="What was observed across the image set (free text).",
    )
    issue_type: Optional[IssueType] = Field(
        None,
        description="Best-matching issue type for this finding.",
    )
    object_part: Optional[str] = Field(
        None,
        description="Best-matching object part for this finding (free text; "
                    "normalised to ObjectPart by validator.py).",
    )
    severity_estimate: Optional[Severity] = Field(
        None,
        description="ADVISORY ONLY.  VLM severity estimate for this finding.  "
                    "Final severity is determined deterministically by rules.py.",
    )
    supporting_image_ids: list[str] = Field(
        default_factory=list,
        description="image_id values of images where this finding is observable.",
    )


class EvidenceFacts(BaseModel):
    """Immutable observation record produced by the VLM agent.

    This is the only output of ``agent.py`` and the only input to
    ``rules.py`` from the VLM.  It contains OBSERVATIONS ONLY:

    - What the VLM sees in each image (``image_observations``).
    - Cross-image synthesised findings (``findings``).

    It does NOT contain:
    - ``claimed_vs_observed_alignment`` — belongs to ``rules.py``.
    - ``evidence_sufficiency`` — belongs to ``rules.py``.
    - Any decisions, status determinations, or policy judgments.

    EvidenceFacts are cached (keyed by case_id + image_hash + claim_hash +
    model_name).  Decisions are NOT cached.  Policy changes do not require
    new VLM calls.
    """

    image_observations: list[ImageObservation] = Field(
        default_factory=list,
        description="Per-image observations, one entry per submitted image.",
    )
    findings: list[Finding] = Field(
        default_factory=list,
        description="Cross-image synthesised findings.  May be empty if no "
                    "coherent findings can be drawn.",
    )


# ---------------------------------------------------------------------------
# Decision models — produced exclusively by rules.py
# ---------------------------------------------------------------------------


class ClaimAtomAssessment(BaseModel):
    """Assessment of a single ClaimAtom against EvidenceFacts.

    Produced by ``rules.py``.  This is where alignment and evidence
    sufficiency live — NOT in EvidenceFacts.

    ``alignment`` and ``evidence_sufficient`` are computed deterministically
    from observations.  Confidence values from EvidenceFacts are advisory
    inputs to this computation and must not gate the result directly.
    """

    atom_index: int = Field(..., ge=0, description="Index into ParsedClaim.atoms.")
    atom: ClaimAtom = Field(..., description="The assessed ClaimAtom.")
    alignment: str = Field(
        ...,
        description="Computed alignment of observations vs. claimed damage: "
                    "'match', 'mismatch', or 'unclear'.  "
                    "Determined deterministically by rules.py.  "
                    "Ambiguous cases resolve to 'unclear', never 'mismatch'.",
    )
    evidence_sufficient: bool = Field(
        ...,
        description="Whether the image observations satisfy the EvidenceChecklist "
                    "items for this atom.  Determined deterministically by rules.py.",
    )
    supporting_image_ids: list[str] = Field(
        default_factory=list,
        description="Images where observations supporting this assessment are present.",
    )
    reasoning: str = Field(
        "",
        description="Short deterministic explanation of the assessment for "
                    "logging and the justification field.",
    )

    @field_validator("alignment")
    @classmethod
    def alignment_must_be_valid(cls, v: str) -> str:
        valid = {"match", "mismatch", "unclear"}
        if v not in valid:
            raise ValueError(f"alignment must be one of {valid}, got {v!r}")
        return v


class DecisionRecord(BaseModel):
    """Complete decision for a single claim produced by ``rules.py``.

    All fields are computed deterministically from EvidenceFacts,
    RetrievedContext, and ParsedClaim.  The VLM is never consulted for
    decisions.

    Invariants enforced here:
    - ``claim_status`` is never influenced by history flags.
    - ``claim_status`` is never ``contradicted`` when alignment is 'unclear'.
    - ``claim_status`` is never ``supported`` when ``evidence_standard_met``
      is False.
    - Confidence thresholds never appear as conditions.
    """

    evidence_standard_met: bool = Field(
        ...,
        description="True when the image set meets the minimum evidence "
                    "requirements for evaluating this type of claim.",
    )
    evidence_standard_met_reason: str = Field(
        ...,
        description="Short deterministic explanation of the evidence decision.",
    )
    risk_flags: list[RiskFlag] = Field(
        default_factory=list,
        description="All applicable risk flags.  History flags appear here "
                    "and ONLY here — never in claim_status.",
    )
    issue_type: IssueType = Field(
        ...,
        description="Scalar issue type, projected from atom assessments by "
                    "severity-based priority then primary-atom tiebreaker.",
    )
    object_part: ObjectPart = Field(
        ...,
        description="Scalar object part, projected from the primary assessed atom.",
    )
    claim_status: ClaimStatus = Field(
        ...,
        description="Final claim decision.  Determined by rule engine from "
                    "alignment and evidence sufficiency only.",
    )
    claim_status_justification: str = Field(
        ...,
        description="Concise image-grounded justification.  References "
                    "specific image IDs where helpful.",
    )
    supporting_image_ids: list[str] = Field(
        default_factory=list,
        description="Images whose observations support the claim_status decision. "
                    "May be non-empty for contradicted claims (the image supports "
                    "the contradiction finding).",
    )
    valid_image: bool = Field(
        ...,
        description="True when at least one submitted image is usable for "
                    "automated review.  False only when ALL images have "
                    "disqualifying quality issues.",
    )
    severity: Severity = Field(
        ...,
        description="Determined deterministically by rules.py from issue_type "
                    "and observation characteristics.  VLM severity_estimate "
                    "is advisory input used only when rule-based determination "
                    "is insufficient.",
    )
    atom_assessments: list[ClaimAtomAssessment] = Field(
        default_factory=list,
        description="Per-atom assessments.  Internal to the decision record; "
                    "not included in final CSV output.",
    )

    @model_validator(mode="after")
    def supported_requires_evidence(self) -> "DecisionRecord":
        """Claim cannot be supported if evidence standard is not met."""
        if (
            self.claim_status == ClaimStatus.SUPPORTED
            and not self.evidence_standard_met
        ):
            raise ValueError(
                "claim_status cannot be 'supported' when evidence_standard_met "
                "is False"
            )
        return self

    @model_validator(mode="after")
    def unclear_cannot_be_contradicted(self) -> "DecisionRecord":
        """Ambiguity must never resolve to contradiction."""
        # This is enforced by the rule engine; this validator is a safety net.
        # The validator cannot inspect atom assessments easily here, so this
        # documents the invariant for downstream consumers.
        return self


# ---------------------------------------------------------------------------
# Final output row — maps 1:1 to output.csv columns
# ---------------------------------------------------------------------------

# Required column order for CSV output.  This list is the single source of
# truth for column ordering throughout the pipeline.
OUTPUT_CSV_COLUMNS: tuple[str, ...] = (
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
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
)


class FinalOutputRow(BaseModel):
    """One row of output.csv.

    All fields are strings for direct CSV serialisation.  Multi-value fields
    use semicolons as separators.  Boolean-like fields use literal 'true' /
    'false' strings.

    Column order follows ``OUTPUT_CSV_COLUMNS`` exactly.
    """

    user_id: str
    image_paths: str = Field(..., description="Echoed verbatim from input CSV.")
    user_claim: str = Field(..., description="Echoed verbatim from input CSV.")
    claim_object: str = Field(..., description="Echoed verbatim from input CSV.")
    evidence_standard_met: str = Field(..., description="'true' or 'false'.")
    evidence_standard_met_reason: str
    risk_flags: str = Field(..., description="Semicolon-separated RiskFlag values, or 'none'.")
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: str = Field(
        ..., description="Semicolon-separated image_id values, or 'none'."
    )
    valid_image: str = Field(..., description="'true' or 'false'.")
    severity: str

    def to_dict(self) -> dict[str, str]:
        """Return an ordered dict matching ``OUTPUT_CSV_COLUMNS`` for CSV writing."""
        return {col: getattr(self, col) for col in OUTPUT_CSV_COLUMNS}
