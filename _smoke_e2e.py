"""
End-to-end smoke test: ClaimInput -> VLMRequest -> Gemini -> EvidenceFacts
Covers: agent.py, validator.py, and their integration with grounding.py
"""
import sys, json, logging
from pathlib import Path
sys.path.insert(0, '.')

# Load env
try:
    from dotenv import load_dotenv
    load_dotenv('.env', override=False)
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

from ingestion import load_claims, load_user_history, load_evidence_requirements
from sanitization import sanitize
from parsing import parse_claim
from retrieval import retrieve_context
from grounding import build_evidence_checklist, build_vlm_request
from agent import extract_evidence
from validator import parse_response_text, validate_evidence_facts, _normalize_issue_type, _normalize_severity, _normalize_quality_issues
from schemas import ClaimObject, EvidenceFacts

dataset = Path(__file__).parent.parent / 'dataset'

claims, warns = load_claims(dataset / 'claims.csv', dataset)
history = load_user_history(dataset / 'user_history.csv')
reqs = load_evidence_requirements(dataset / 'evidence_requirements.csv')

print(f"Dataset: {len(claims)} claims, {len(history)} users, {len(reqs)} reqs")

# ---- UNIT TEST: validator enum normalisation ----
print("\n[UNIT] Validator enum normalisation...")
assert _normalize_issue_type("DENT") == "dent"
assert _normalize_issue_type("glass shatter") == "glass_shatter"
assert _normalize_issue_type("broken") == "broken_part"
assert _normalize_issue_type("torn") == "torn_packaging"
assert _normalize_issue_type(None) is None
assert _normalize_severity("MEDIUM") == "medium"
assert _normalize_severity("severe") == "high"
assert _normalize_severity(None) is None
assert _normalize_quality_issues("blurry, low_light") == ["blurry", "low_light"]
assert _normalize_quality_issues(["blur", "wrong angle"]) == ["blurry", "wrong_angle"]
print("  PASS")

# ---- UNIT TEST: validator repair of decision-layer fields ----
print("\n[UNIT] Validator strips decision-layer fields...")
from validator import validate_evidence_facts
raw_with_decision = {
    "claim_status": "supported",       # must be stripped
    "evidence_standard_met": True,     # must be stripped
    "risk_flags": ["blurry_image"],    # must be stripped
    "image_observations": [],
    "findings": []
}
facts = validate_evidence_facts(raw_with_decision, ["img_1"])
assert not hasattr(facts, "claim_status") or not getattr(facts, "claim_status", None)
print("  PASS — no decision fields in EvidenceFacts")

# ---- UNIT TEST: validator image_id repair ----
print("\n[UNIT] Validator image_id repair...")
raw_imageid = {
    "image_observations": [
        {"image_id": "Image 1", "object_visible": True, "part_visible": True,
         "damage_observed": True, "damage_description": "dent",
         "issue_type_observed": "dent", "quality_issues": [], "confidence": 0.9,
         "text_or_instructions_present": False},
    ],
    "findings": []
}
facts2 = validate_evidence_facts(raw_imageid, ["img_1", "img_2"])
assert facts2.image_observations[0].image_id == "img_1", f"Got: {facts2.image_observations[0].image_id}"
print("  PASS — 'Image 1' mapped to 'img_1'")

# ---- INTEGRATION TEST: full pipeline on first claim with images ----
print("\n[INTEGRATION] Full pipeline: ClaimInput -> EvidenceFacts...")
# Pick a claim that actually has resolved images
claim_with_imgs = next(
    (c for c in claims if len(c.images) >= 1), None
)
if claim_with_imgs is None:
    print("  SKIP — no claims with resolved images")
    sys.exit(0)

print(f"  Claim: user={claim_with_imgs.user_id}, object={claim_with_imgs.claim_object.value}, images={len(claim_with_imgs.images)}")

# Stage 1: Sanitize
san = sanitize(claim_with_imgs)
print(f"  Sanitization: injection={san.injection_detected}")

# Stage 2: Parse
parsed = parse_claim(claim_with_imgs, san)
print(f"  Parsing: {len(parsed.atoms)} atom(s), lang={parsed.conversation_language}")
for i, a in enumerate(parsed.atoms):
    print(f"    [{i}] {a.issue_type_hint} on {a.object_part_hint}")

# Stage 3: Retrieve
ctx = retrieve_context(parsed, history, reqs)
print(f"  Retrieval: {len(ctx.applicable_requirements)} requirements")

# Stage 4: Ground (Strategy A)
checklist = build_evidence_checklist(parsed, ctx)
print(f"  Checklist: {len(checklist.items)} items")
request_a = build_vlm_request(parsed, ctx, use_checklist=True)
print(f"  VLMRequest: model={request_a.model}, images={len(request_a.image_paths)}, prompt={len(request_a.user_prompt)} chars")

# Stage 5: Call Gemini
print("  Calling Gemini... (may take 10-30s)")
facts = extract_evidence(request_a)

print(f"\n  EvidenceFacts:")
print(f"    image_observations: {len(facts.image_observations)}")
for obs in facts.image_observations:
    print(f"      [{obs.image_id}] visible={obs.object_visible}, damage={obs.damage_observed}, issue={obs.issue_type_observed}, conf={obs.confidence:.2f}")
    if obs.quality_issues:
        print(f"        quality_issues: {obs.quality_issues}")
print(f"    findings: {len(facts.findings)}")
for f in facts.findings:
    print(f"      [{f.issue_type}] {f.description[:80]} supports={f.supporting_image_ids}")

# Validate EvidenceFacts contract
assert isinstance(facts, EvidenceFacts)
assert not hasattr(facts, "claim_status")
assert len(facts.image_observations) > 0, "Expected at least one observation"
print("\n  Contract assertions PASS")

# Strategy B comparison
print("\n[INTEGRATION] Strategy B (no checklist)...")
ctx_b = retrieve_context(parsed, history, reqs)
request_b = build_vlm_request(parsed, ctx_b, use_checklist=False)
facts_b = extract_evidence(request_b)
print(f"  Strategy B: {len(facts_b.image_observations)} obs, {len(facts_b.findings)} findings")
print(f"  Strategy A: {len(facts.image_observations)} obs, {len(facts.findings)} findings")

print("\n=== ALL TESTS PASSED ===")
