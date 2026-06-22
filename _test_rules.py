"""
Unit tests for rules.py — deterministic rule engine.

Tests cover all 8 sub-steps individually plus worked end-to-end examples.
No Gemini calls. All EvidenceFacts are constructed from synthetic data.

Run with:
    python code/_test_rules.py
"""
import sys
sys.path.insert(0, '.')

from schemas import (
    ClaimAtom, ClaimInput, ClaimObject, ClaimStatus, EvidenceFacts, EvidenceChecklist,
    Finding, ImageObservation, ImageRef, IssueType, ObjectPart, ParsedClaim,
    RetrievedContext, RiskFlag, Severity, SanitizationResult, UserHistory,
)
from rules import (
    evaluate,
    _assess_valid_image,
    _assess_atoms,
    _assess_evidence_standard,
    _project_issue_type,
    _project_object_part,
    _determine_claim_status,
    _project_severity,
    _collect_risk_flags,
    _compute_history_flags,
    _is_usable_image,
    _is_clear_image,
)

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_image_ref(image_id="img_1"):
    return ImageRef(
        image_id=image_id,
        path=f"/fake/{image_id}.jpg",
        relative_path=f"images/test/case_001/{image_id}.jpg",
        case_id="case_001",
        split="test",
    )

def make_claim_input(user_id="user_001", obj=ClaimObject.CAR, image_ids=None, row=0):
    image_ids = image_ids or ["img_1"]
    return ClaimInput(
        user_id=user_id,
        image_paths_raw=";".join(f"images/test/case_001/{i}.jpg" for i in image_ids),
        images=[make_image_ref(i) for i in image_ids],
        user_claim="Customer: My car has a dent on the front bumper.",
        claim_object=obj,
        row_index=row,
    )

def make_sanitization(injection=False):
    return SanitizationResult(
        sanitized_claim="Customer: My car has a dent on the front bumper.",
        injection_detected=injection,
        injection_spans=[],
        flags=[],
    )

def make_atom(issue=IssueType.DENT, part=ObjectPart.FRONT_BUMPER,
              desc_issue="dent", desc_part="front bumper"):
    return ClaimAtom(
        described_issue=desc_issue,
        described_part=desc_part,
        issue_type_hint=issue,
        object_part_hint=part,
    )

def make_parsed(atoms=None, claim_input=None, sanitization=None, primary_index=0):
    atoms = atoms or [make_atom()]
    claim_input = claim_input or make_claim_input()
    sanitization = sanitization or make_sanitization()
    return ParsedClaim(
        claim_input=claim_input,
        raw_text=claim_input.user_claim,
        atoms=atoms,
        primary_atom_index=primary_index,
        sanitization=sanitization,
    )

def make_obs(image_id="img_1", object_visible=True, part_visible=True,
             damage_observed=True, issue_type=IssueType.DENT,
             quality_issues=None, text_instructions=False, confidence=0.9,
             severity_est=None, object_type="car", part_obs="front bumper"):
    return ImageObservation(
        image_id=image_id,
        object_visible=object_visible,
        object_type_observed=object_type,
        part_visible=part_visible,
        part_observed=part_obs,
        damage_observed=damage_observed,
        damage_description="dent visible" if damage_observed else None,
        issue_type_observed=issue_type if damage_observed else None,
        severity_estimate=severity_est,
        quality_issues=quality_issues or [],
        text_or_instructions_present=text_instructions,
        confidence=confidence,
    )

def make_finding(issue=IssueType.DENT, part="front bumper", sev=None,
                 supporting=None, desc="Dent visible on front bumper."):
    return Finding(
        description=desc,
        issue_type=issue,
        object_part=part,
        severity_estimate=sev,
        supporting_image_ids=supporting or ["img_1"],
    )

def make_history(past=2, accept=2, manual=0, rejected=0, last90=1, flags=None):
    return UserHistory(
        user_id="user_001",
        past_claim_count=past,
        accept_claim=accept,
        manual_review_claim=manual,
        rejected_claim=rejected,
        last_90_days_claim_count=last90,
        history_flags=flags or [],
        history_summary="",
    )

def make_context(history=None):
    return RetrievedContext(
        user_history=history,
        applicable_requirements=[],
        evidence_checklist=EvidenceChecklist(),
    )


# ---------------------------------------------------------------------------
# TEST GROUP 1 — valid_image
# ---------------------------------------------------------------------------
print("\n[GROUP 1] valid_image")

# 1.1 No images → False
check("no images → False",
      _assess_valid_image(EvidenceFacts()) is False)

# 1.2 Single clear image → True
facts = EvidenceFacts(image_observations=[make_obs()])
check("single clear image → True", _assess_valid_image(facts) is True)

# 1.3 Object not visible → False
facts = EvidenceFacts(image_observations=[make_obs(object_visible=False)])
check("object_visible=False → False", _assess_valid_image(facts) is False)

# 1.4 wrong_object quality issue → False
facts = EvidenceFacts(image_observations=[make_obs(quality_issues=["wrong_object"])])
check("wrong_object → False", _assess_valid_image(facts) is False)

# 1.5 Two simultaneous core quality issues → False (blocking-degraded)
facts = EvidenceFacts(image_observations=[make_obs(quality_issues=["blurry", "wrong_angle"])])
check("blurry+wrong_angle → False (blocking-degraded)", _assess_valid_image(facts) is False)

# 1.6 Single non-blocking quality issue → True
facts = EvidenceFacts(image_observations=[make_obs(quality_issues=["blurry"])])
check("single blurry → True (not blocking)", _assess_valid_image(facts) is True)

# 1.7 One bad image + one good image → True
facts = EvidenceFacts(image_observations=[
    make_obs("img_1", object_visible=False),
    make_obs("img_2"),
])
check("one bad + one good → True", _assess_valid_image(facts) is True)


# ---------------------------------------------------------------------------
# TEST GROUP 2 — _is_usable_image / _is_clear_image
# ---------------------------------------------------------------------------
print("\n[GROUP 2] _is_usable_image / _is_clear_image")

obs_clear = make_obs()
check("clear image is usable", _is_usable_image(obs_clear))
check("clear image is clear", _is_clear_image(obs_clear))

obs_blurry = make_obs(quality_issues=["blurry"])
check("blurry image is usable (1 issue)", _is_usable_image(obs_blurry))
check("blurry image is NOT clear", not _is_clear_image(obs_blurry))

obs_two_issues = make_obs(quality_issues=["blurry", "wrong_angle"])
check("blurry+wrong_angle is NOT usable", not _is_usable_image(obs_two_issues))


# ---------------------------------------------------------------------------
# TEST GROUP 3 — atom alignment
# ---------------------------------------------------------------------------
print("\n[GROUP 3] Atom alignment")

def make_assessments_from_parsed_facts(parsed, facts):
    return _assess_atoms(parsed, facts)

# 3.1 Match: damage observed, type matches hint
parsed = make_parsed([make_atom(IssueType.DENT)])
facts = EvidenceFacts(
    image_observations=[make_obs(issue_type=IssueType.DENT, damage_observed=True)],
    findings=[make_finding(IssueType.DENT)],
)
assessments = make_assessments_from_parsed_facts(parsed, facts)
check("dent claimed + dent observed → match", assessments[0].alignment == "match")
check("match → evidence_sufficient=True", assessments[0].evidence_sufficient is True)

# 3.2 Mismatch: clear image, part visible, NO damage
parsed = make_parsed([make_atom(IssueType.DENT)])
facts = EvidenceFacts(
    image_observations=[make_obs(damage_observed=False, issue_type=None, quality_issues=[])],
    findings=[],
)
assessments = make_assessments_from_parsed_facts(parsed, facts)
check("clear image, no damage → mismatch", assessments[0].alignment == "mismatch")
check("mismatch → evidence_sufficient=False", assessments[0].evidence_sufficient is False)

# 3.3 Mismatch: wrong damage type visible (scratch claimed, dent observed)
parsed = make_parsed([make_atom(IssueType.SCRATCH)])
facts = EvidenceFacts(
    image_observations=[make_obs(issue_type=IssueType.DENT, quality_issues=[])],
    findings=[make_finding(IssueType.DENT)],
)
assessments = make_assessments_from_parsed_facts(parsed, facts)
check("scratch claimed + dent observed in clear image → mismatch",
      assessments[0].alignment == "mismatch")

# 3.4 Unclear: blurry + wrong_angle (degraded), damage of unknown type
parsed = make_parsed([make_atom(IssueType.DENT)])
obs = ImageObservation(
    image_id="img_1", object_visible=True, part_visible=True,
    damage_observed=True, issue_type_observed=None,
    quality_issues=["blurry"], confidence=0.5,
)
facts = EvidenceFacts(image_observations=[obs], findings=[])
assessments = make_assessments_from_parsed_facts(parsed, facts)
check("damage observed but type unclear → unclear", assessments[0].alignment == "unclear")

# 3.5 POLICY B: clear-no-damage + blurry-damage-UNKNOWN-TYPE → unclear (conflicting)
# The blurry image has damage_observed=True but issue_type_observed=None.
# Under Policy B: ambiguous damage is not a contradiction — route to unclear.
parsed = make_parsed([make_atom(IssueType.DENT)])
facts = EvidenceFacts(
    image_observations=[
        make_obs("img_1", damage_observed=False, quality_issues=[]),          # clear: no damage
        make_obs("img_2", damage_observed=True, issue_type=None, quality_issues=["blurry"]),
    ],
    findings=[],
)
assessments = make_assessments_from_parsed_facts(parsed, facts)
check("Policy B: clear-no-damage + blurry-unknown-damage → unclear (not mismatch)",
      assessments[0].alignment == "unclear",
      f"got: {assessments[0].alignment}")

# 3.6 Policy B: clear-no-damage + NO other damage at all → still mismatch (pure case)
# When no other image observes any damage, the clear no-damage is authoritative.
parsed = make_parsed([make_atom(IssueType.DENT)])
facts = EvidenceFacts(
    image_observations=[
        make_obs("img_1", damage_observed=False, quality_issues=[]),  # clear: no damage
        make_obs("img_2", damage_observed=False, quality_issues=["blurry"]),  # blurry: also no damage
    ],
    findings=[],
)
assessments = make_assessments_from_parsed_facts(parsed, facts)
check("Policy A retained: clear-no-damage + blurry-also-no-damage → mismatch",
      assessments[0].alignment == "mismatch",
      f"got: {assessments[0].alignment}")

# 3.7 Policy B: clear-no-damage + confirmed-WRONG-type → still mismatch
# When the other image has a confirmed DIFFERENT issue type, it is a genuine contradiction.
parsed = make_parsed([make_atom(IssueType.DENT)])
facts = EvidenceFacts(
    image_observations=[
        make_obs("img_1", damage_observed=False, quality_issues=[]),      # clear: no damage
        make_obs("img_2", damage_observed=True, issue_type=IssueType.SCRATCH, quality_issues=[]),  # confirmed scratch
    ],
    findings=[make_finding(IssueType.SCRATCH)],
)
assessments = make_assessments_from_parsed_facts(parsed, facts)
check("Confirmed wrong type still mismatch (scratch confirmed, dent claimed)",
      assessments[0].alignment == "mismatch",
      f"got: {assessments[0].alignment}")


# ---------------------------------------------------------------------------
# TEST GROUP 4 — evidence_standard_met
# ---------------------------------------------------------------------------
print("\n[GROUP 4] evidence_standard_met")

# 4.1 No images → False
parsed = make_parsed()
esm, reason = _assess_evidence_standard(EvidenceFacts(), [], make_context(), False)
check("no images → esm=False", esm is False)

# 4.2 All unclear alignments → False
parsed = make_parsed([make_atom(IssueType.DENT)])
facts = EvidenceFacts(image_observations=[make_obs(quality_issues=["blurry", "wrong_angle"])])
assessments = _assess_atoms(parsed, facts)
vi = _assess_valid_image(facts)
esm, reason = _assess_evidence_standard(facts, assessments, make_context(), vi)
check("all unclear → esm=False", esm is False)

# 4.3 Match alignment → True
parsed = make_parsed([make_atom(IssueType.DENT)])
facts = EvidenceFacts(
    image_observations=[make_obs(issue_type=IssueType.DENT)],
    findings=[make_finding(IssueType.DENT)],
)
assessments = _assess_atoms(parsed, facts)
vi = _assess_valid_image(facts)
esm, reason = _assess_evidence_standard(facts, assessments, make_context(), vi)
check("match alignment → esm=True", esm is True)

# 4.4 Mismatch alignment → True (standard met, but claim is contradicted)
parsed = make_parsed([make_atom(IssueType.SCRATCH)])
facts = EvidenceFacts(
    image_observations=[make_obs(issue_type=IssueType.DENT, quality_issues=[])],
    findings=[make_finding(IssueType.DENT)],
)
assessments = _assess_atoms(parsed, facts)
vi = _assess_valid_image(facts)
esm, reason = _assess_evidence_standard(facts, assessments, make_context(), vi)
check("mismatch alignment → esm=True (evaluable contradiction)", esm is True)


# ---------------------------------------------------------------------------
# TEST GROUP 5 — claim_status
# ---------------------------------------------------------------------------
print("\n[GROUP 5] claim_status")

# 5.1 esm=False → not_enough_information
parsed = make_parsed()
assessments = []
status, just, ids = _determine_claim_status(assessments, False, EvidenceFacts(), parsed)
check("esm=False → not_enough_information", status == ClaimStatus.NOT_ENOUGH_INFORMATION)

# 5.2 match → supported
parsed = make_parsed([make_atom(IssueType.DENT)])
facts = EvidenceFacts(
    image_observations=[make_obs(issue_type=IssueType.DENT)],
    findings=[make_finding(IssueType.DENT)],
)
assessments = _assess_atoms(parsed, facts)
status, just, ids = _determine_claim_status(assessments, True, facts, parsed)
check("match → supported", status == ClaimStatus.SUPPORTED)
check("supporting_ids populated for supported", len(ids) > 0)

# 5.3 mismatch → contradicted
parsed = make_parsed([make_atom(IssueType.SCRATCH)])
facts = EvidenceFacts(
    image_observations=[make_obs(issue_type=IssueType.DENT, quality_issues=[])],
    findings=[make_finding(IssueType.DENT)],
)
assessments = _assess_atoms(parsed, facts)
status, just, ids = _determine_claim_status(assessments, True, facts, parsed)
check("mismatch → contradicted", status == ClaimStatus.CONTRADICTED)

# 5.4 Schema invariant: supported requires esm=True
from schemas import DecisionRecord
try:
    bad = DecisionRecord(
        evidence_standard_met=False,
        evidence_standard_met_reason="no evidence",
        risk_flags=[],
        issue_type=IssueType.DENT,
        object_part=ObjectPart.FRONT_BUMPER,
        claim_status=ClaimStatus.SUPPORTED,   # INVALID
        claim_status_justification="test",
        supporting_image_ids=[],
        valid_image=True,
        severity=Severity.MEDIUM,
    )
    check("schema rejects supported+esm=False", False, "should have raised ValueError")
except Exception:
    check("schema rejects supported+esm=False", True)


# ---------------------------------------------------------------------------
# TEST GROUP 6 — severity
# ---------------------------------------------------------------------------
print("\n[GROUP 6] severity projection")

# 6.1 not_enough_information → UNKNOWN
sev = _project_severity(ClaimStatus.NOT_ENOUGH_INFORMATION, IssueType.DENT, EvidenceFacts())
check("not_enough_information → severity UNKNOWN", sev == Severity.UNKNOWN)

# 6.2 supported + dent → MEDIUM (from ISSUE_SEVERITY_MAP)
facts = EvidenceFacts(image_observations=[make_obs(issue_type=IssueType.DENT)])
sev = _project_severity(ClaimStatus.SUPPORTED, IssueType.DENT, facts)
check("supported + dent → MEDIUM", sev == Severity.MEDIUM)

# 6.3 supported + scratch → LOW
sev = _project_severity(ClaimStatus.SUPPORTED, IssueType.SCRATCH, EvidenceFacts())
check("supported + scratch → LOW", sev == Severity.LOW)

# 6.4 supported + glass_shatter → HIGH
sev = _project_severity(ClaimStatus.SUPPORTED, IssueType.GLASS_SHATTER, EvidenceFacts())
check("supported + glass_shatter → HIGH", sev == Severity.HIGH)

# 6.5 contradicted + no damage observed → NONE
facts = EvidenceFacts(
    image_observations=[make_obs(damage_observed=False, issue_type=None)],
    findings=[],
)
sev = _project_severity(ClaimStatus.CONTRADICTED, IssueType.DENT, facts)
check("contradicted + no damage visible → NONE", sev == Severity.NONE)

# 6.6 contradicted + wrong damage visible (scratch seen) → LOW (from observed type)
facts = EvidenceFacts(
    image_observations=[make_obs(issue_type=IssueType.SCRATCH)],
    findings=[make_finding(IssueType.SCRATCH, sev=Severity.LOW)],
)
sev = _project_severity(ClaimStatus.CONTRADICTED, IssueType.DENT, facts)
check("contradicted + scratch visible → LOW (actual severity)", sev == Severity.LOW)


# ---------------------------------------------------------------------------
# TEST GROUP 7 — risk flags
# ---------------------------------------------------------------------------
print("\n[GROUP 7] risk flags")

# 7.1 No issues, no history, clean claim → NONE flag
parsed = make_parsed()
facts = EvidenceFacts(image_observations=[make_obs()])
flags = _collect_risk_flags(facts, parsed, None, [], True)
check("clean claim, no history → no flags (or NONE)", 
      not flags or flags == [RiskFlag.NONE])

# 7.2 Blurry image → BLURRY_IMAGE
parsed = make_parsed()
facts = EvidenceFacts(image_observations=[make_obs(quality_issues=["blurry"])])
flags = _collect_risk_flags(facts, parsed, None, [], True)
check("blurry image → BLURRY_IMAGE", RiskFlag.BLURRY_IMAGE in flags)

# 7.3 Text instructions in image → TEXT_INSTRUCTION_PRESENT
parsed = make_parsed()
facts = EvidenceFacts(image_observations=[make_obs(text_instructions=True)])
flags = _collect_risk_flags(facts, parsed, None, [], True)
check("text instructions in image → TEXT_INSTRUCTION_PRESENT",
      RiskFlag.TEXT_INSTRUCTION_PRESENT in flags)

# 7.4 Text instructions → also MANUAL_REVIEW_REQUIRED
check("text instructions → also MANUAL_REVIEW_REQUIRED",
      RiskFlag.MANUAL_REVIEW_REQUIRED in flags)

# 7.5 Injection detected → POSSIBLE_MANIPULATION
parsed_injected = make_parsed(sanitization=make_sanitization(injection=True))
facts = EvidenceFacts(image_observations=[make_obs()])
flags = _collect_risk_flags(facts, parsed_injected, None, [], True)
check("injection detected → POSSIBLE_MANIPULATION", RiskFlag.POSSIBLE_MANIPULATION in flags)
check("injection detected → MANUAL_REVIEW_REQUIRED", RiskFlag.MANUAL_REVIEW_REQUIRED in flags)

# 7.6 History flags: high rejection rate
hist = make_history(past=10, accept=6, rejected=4, last90=1)  # 40% rejection
h_flags = _compute_history_flags(hist)
check("40% rejection rate → USER_HISTORY_RISK", RiskFlag.USER_HISTORY_RISK in h_flags)

# 7.7 History flags: frequent claimant (≥3 in last 90 days)
hist = make_history(past=5, accept=5, rejected=0, last90=3)
h_flags = _compute_history_flags(hist)
check("3 claims in 90 days → USER_HISTORY_RISK", RiskFlag.USER_HISTORY_RISK in h_flags)

# 7.8 History flags: clean history → no USER_HISTORY_RISK
hist = make_history(past=2, accept=2, rejected=0, last90=1)
h_flags = _compute_history_flags(hist)
check("clean history → no USER_HISTORY_RISK", RiskFlag.USER_HISTORY_RISK not in h_flags)

# 7.9 USER_HISTORY_RISK → also MANUAL_REVIEW_REQUIRED
hist = make_history(past=10, accept=6, rejected=4, last90=1)
parsed = make_parsed()
facts = EvidenceFacts(image_observations=[make_obs()])
flags = _collect_risk_flags(facts, parsed, hist, [], True)
check("USER_HISTORY_RISK → MANUAL_REVIEW_REQUIRED",
      RiskFlag.MANUAL_REVIEW_REQUIRED in flags)

# 7.10 History flags NEVER influence claim_status (tested via full evaluate)
# We verify that even a high-risk user can get SUPPORTED if evidence is good
parsed = make_parsed([make_atom(IssueType.DENT)])
facts = EvidenceFacts(
    image_observations=[make_obs(issue_type=IssueType.DENT)],
    findings=[make_finding(IssueType.DENT)],
)
hist = make_history(past=10, accept=6, rejected=4, last90=1)
ctx = make_context(history=hist)
record = evaluate(parsed, facts, ctx)
check("high-risk history user + good evidence → still SUPPORTED",
      record.claim_status == ClaimStatus.SUPPORTED,
      f"got {record.claim_status}")
check("high-risk history → USER_HISTORY_RISK in flags",
      RiskFlag.USER_HISTORY_RISK in record.risk_flags)

# 7.11 Damage not visible in clear images → DAMAGE_NOT_VISIBLE
parsed = make_parsed([make_atom(IssueType.DENT)])
facts = EvidenceFacts(
    image_observations=[make_obs(damage_observed=False, quality_issues=[])],
    findings=[],
)
flags = _collect_risk_flags(facts, parsed, None, [], True)
check("clear image + no damage → DAMAGE_NOT_VISIBLE", RiskFlag.DAMAGE_NOT_VISIBLE in flags)


# ---------------------------------------------------------------------------
# TEST GROUP 8 — Worked end-to-end examples
# ---------------------------------------------------------------------------
print("\n[GROUP 8] End-to-end worked examples")


def e2e(label, atoms, obs_list, findings_list, history=None, check_status=None,
        check_sev=None, check_flags_include=None, check_flags_exclude=None,
        check_esm=None, check_vi=None):
    parsed = make_parsed(atoms)
    facts = EvidenceFacts(image_observations=obs_list, findings=findings_list)
    ctx = make_context(history=history)
    record = evaluate(parsed, facts, ctx)
    ok = True
    details = []
    if check_status is not None and record.claim_status != check_status:
        ok = False
        details.append(f"status={record.claim_status.value} want {check_status.value}")
    if check_sev is not None and record.severity != check_sev:
        ok = False
        details.append(f"sev={record.severity.value} want {check_sev.value}")
    if check_esm is not None and record.evidence_standard_met != check_esm:
        ok = False
        details.append(f"esm={record.evidence_standard_met} want {check_esm}")
    if check_vi is not None and record.valid_image != check_vi:
        ok = False
        details.append(f"vi={record.valid_image} want {check_vi}")
    if check_flags_include:
        for f in check_flags_include:
            if f not in record.risk_flags:
                ok = False
                details.append(f"missing flag {f.value}")
    if check_flags_exclude:
        for f in check_flags_exclude:
            if f in record.risk_flags:
                ok = False
                details.append(f"unexpected flag {f.value}")
    check(label, ok, "; ".join(details))
    return record


# Example 1: Classic supported car dent
e2e(
    "Example 1: car dent — clear image, damage matches → supported/medium",
    atoms=[make_atom(IssueType.DENT, ObjectPart.FRONT_BUMPER)],
    obs_list=[make_obs("img_1", issue_type=IssueType.DENT)],
    findings_list=[make_finding(IssueType.DENT, sev=Severity.MEDIUM)],
    check_status=ClaimStatus.SUPPORTED,
    check_sev=Severity.MEDIUM,
    check_esm=True,
    check_vi=True,
)

# Example 2: Contradicted — clear image shows no damage
e2e(
    "Example 2: dent claimed, clear images show nothing → contradicted/none",
    atoms=[make_atom(IssueType.DENT, ObjectPart.FRONT_BUMPER)],
    obs_list=[make_obs("img_1", damage_observed=False, quality_issues=[])],
    findings_list=[],
    check_status=ClaimStatus.CONTRADICTED,
    check_sev=Severity.NONE,
    check_esm=True,
    check_vi=True,
)

# Example 3: Not enough info — all images blurry+wrong_angle
e2e(
    "Example 3: all images blurry+wrong_angle → not_enough_information/unknown",
    atoms=[make_atom(IssueType.DENT)],
    obs_list=[make_obs("img_1", quality_issues=["blurry", "wrong_angle"])],
    findings_list=[],
    check_status=ClaimStatus.NOT_ENOUGH_INFORMATION,
    check_sev=Severity.UNKNOWN,
    check_vi=False,
)

# Example 4: wrong_object → not_enough_information + WRONG_OBJECT flag
e2e(
    "Example 4: wrong_object in image → not_enough_information + WRONG_OBJECT",
    atoms=[make_atom(IssueType.DENT)],
    obs_list=[make_obs("img_1", quality_issues=["wrong_object"])],
    findings_list=[],
    check_status=ClaimStatus.NOT_ENOUGH_INFORMATION,
    check_flags_include=[RiskFlag.WRONG_OBJECT],
    check_vi=False,
)

# Example 5: glass_shatter → HIGH severity when supported
e2e(
    "Example 5: glass_shatter supported → severity HIGH",
    atoms=[make_atom(IssueType.GLASS_SHATTER, ObjectPart.WINDSHIELD)],
    obs_list=[make_obs("img_1", issue_type=IssueType.GLASS_SHATTER)],
    findings_list=[make_finding(IssueType.GLASS_SHATTER, sev=Severity.HIGH)],
    check_status=ClaimStatus.SUPPORTED,
    check_sev=Severity.HIGH,
)

# Example 6: Multi-atom — one matches, one doesn't → supported (majority)
e2e(
    "Example 6: 2 atoms, 1 match 1 unclear → supported",
    atoms=[
        make_atom(IssueType.DENT, ObjectPart.FRONT_BUMPER),
        make_atom(IssueType.SCRATCH, ObjectPart.DOOR),
    ],
    obs_list=[
        make_obs("img_1", issue_type=IssueType.DENT),
        make_obs("img_2", damage_observed=False, quality_issues=["blurry"]),
    ],
    findings_list=[make_finding(IssueType.DENT)],
    check_status=ClaimStatus.SUPPORTED,
)

# Example 7: History risk + good evidence → still SUPPORTED
e2e(
    "Example 7: high-risk history + good evidence → supported + USER_HISTORY_RISK",
    atoms=[make_atom(IssueType.DENT)],
    obs_list=[make_obs("img_1", issue_type=IssueType.DENT)],
    findings_list=[make_finding(IssueType.DENT)],
    history=make_history(past=10, accept=6, rejected=4, last90=1),
    check_status=ClaimStatus.SUPPORTED,
    check_flags_include=[RiskFlag.USER_HISTORY_RISK, RiskFlag.MANUAL_REVIEW_REQUIRED],
)

# Example 8: vi=False but evidence_standard_met=True? No — vi=False → esm=False
e2e(
    "Example 8: vi=False → esm=False → not_enough_information",
    atoms=[make_atom(IssueType.DENT)],
    obs_list=[make_obs("img_1", object_visible=False)],
    findings_list=[],
    check_vi=False,
    check_esm=False,
    check_status=ClaimStatus.NOT_ENOUGH_INFORMATION,
)


# Example 3b: POLICY B — conflicting evidence → NEI + MANUAL_REVIEW_REQUIRED
e2e(
    "Example 3b: Policy B: clear-no-damage + blurry-unknown-damage -> NEI + manual review",
    atoms=[make_atom(IssueType.DENT)],
    obs_list=[
        make_obs("img_1", damage_observed=False, quality_issues=[]),
        make_obs("img_2", damage_observed=True, issue_type=None, quality_issues=["blurry"]),
    ],
    findings_list=[],
    check_status=ClaimStatus.NOT_ENOUGH_INFORMATION,
    check_sev=Severity.UNKNOWN,
    check_esm=False,
    check_vi=True,
    check_flags_include=[RiskFlag.MANUAL_REVIEW_REQUIRED],
    check_flags_exclude=[RiskFlag.CLAIM_MISMATCH],
)

# Example 3c: POLICY A retained — no image observes any damage → contradicted
e2e(
    "Example 3c: Policy A retained: clear-no-damage only -> contradicted/none",
    atoms=[make_atom(IssueType.DENT)],
    obs_list=[
        make_obs("img_1", damage_observed=False, quality_issues=[]),
        make_obs("img_2", damage_observed=False, quality_issues=["blurry"]),
    ],
    findings_list=[],
    check_status=ClaimStatus.CONTRADICTED,
    check_sev=Severity.NONE,
    check_esm=True,
    check_vi=True,
)

# Example 3d: Confirmed wrong type in clear image → still contradicted (Policy A)
e2e(
    "Example 3d: Confirmed wrong type (scratch seen, dent claimed) -> contradicted",
    atoms=[make_atom(IssueType.DENT)],
    obs_list=[
        make_obs("img_1", damage_observed=False, quality_issues=[]),
        make_obs("img_2", damage_observed=True, issue_type=IssueType.SCRATCH, quality_issues=[]),
    ],
    findings_list=[make_finding(IssueType.SCRATCH)],
    check_status=ClaimStatus.CONTRADICTED,
    check_sev=Severity.LOW,
    check_esm=True,
    check_flags_include=[RiskFlag.CLAIM_MISMATCH],
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL TESTS PASSED")
else:
    print(f"ATTENTION: {FAIL} test(s) failed")
    sys.exit(1)
