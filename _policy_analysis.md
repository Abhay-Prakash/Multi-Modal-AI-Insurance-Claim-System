# Policy Analysis: clear_no_damage > blurry_damage (Policy A) vs conflicting_evidence → NEI (Policy B)

## Ground Truth Taxonomy (sample_claims.csv — 20 rows)

### `contradicted` (5 rows)

| vi | sev | damage_not_visible | claim_mismatch | wrong_object | Inferred trigger |
|---|---|---|---|---|---|
| true | low | ❌ | ✅ | ❌ | Different damage type clearly visible |
| **false** | **high** | ❌ | ✅ | ❌ | Wrong object / non-original image visible |
| true | **none** | ✅ | ❌ | ❌ | Clear image, part visible, **zero damage observed** |
| true | low | ❌ | ✅ | ✅ | Wrong object visible |
| true | **none** | ✅ | ❌ | ❌ | Clear image, part visible, **zero damage observed** |

**Key finding:** All `damage_not_visible` contradictions have `sev=none` — meaning **no damage was observed anywhere** (not even in blurry images). These are the unambiguous "nothing to see here" cases.

The `claim_mismatch` contradictions have `sev=low|high` — **damage of the wrong type was clearly visible**.

### `not_enough_information` (3 rows)

| vi | damage_not_visible | wrong_angle | wrong_object | manual_review |
|---|---|---|---|---|
| true | ❌ | ❌ | ✅ | ✅ |
| true | ✅ | ✅ | ❌ | ❌ |
| false | ✅ | ❌ | ❌ | ✅ |

**Key finding:** NEI+`damage_not_visible`+`wrong_angle` (row 2) = images are angled wrong, and no damage visible. This is the **conflicting-quality** scenario — images were probably taken at bad angles that prevented seeing damage.

---

## The Core Question

The specific scenario under debate:

```
img_1: object_visible=True, part_visible=True, damage_observed=False, quality_issues=[]
img_2: object_visible=True, damage_observed=True, issue_type_observed=None, quality_issues=["blurry"]
```

**Policy A** (current): `img_1` is clear-and-authoritative → `mismatch` → `contradicted`

**Policy B** (proposed): signals conflict → `unclear` → `not_enough_information` + `MANUAL_REVIEW_REQUIRED`

---

## Policy Comparison

### 1. False Contradiction Risk

**Policy A — HIGH risk in two sub-cases:**

Sub-case A1: img_1 shows the **wrong part** (but VLM marks part_visible=True for whatever part it sees), img_2 shows the correct claimed part with blurry damage. Policy A would wrongly contradict the claim because img_1's "clear no damage" is authoritative even though it's showing a different part.

Sub-case A2: img_1 is a wide-angle shot where the part is visible but damage is too small to see clearly, img_2 is a close-up but blurry. Policy A over-trusts img_1's negative.

**Policy B — LOW risk:**
Routes the conflict to human review instead of committing to "contradicted".

### 2. False Support Risk

**Policy A — not applicable here** (Policy A produces contradiction, not support)

**Policy B — negligible:**
Policy B produces `not_enough_information`, not `supported`. A fraudulent claimant cannot gain by Policy B — they still need unambiguous supporting evidence to get `supported`.

### 3. Sample Data Alignment

All 5 `contradicted` cases in sample_claims.csv match **one of two distinct patterns**:

| Pattern | Description | Policy A/B agree? |
|---|---|---|
| **P1**: `damage_not_visible`, `sev=none` | No damage observed in **any** image | ✅ Both agree → `mismatch` |
| **P2**: `claim_mismatch`, `sev>none` | Damage of **confirmed wrong type** is visible | ✅ Both agree → `mismatch` |

The disputed case (clear-no-damage + blurry-unknown-damage) appears in **neither pattern** in the sample. It has positive damage signal (img_2) without type confirmation — which structurally matches the `not_enough_information+wrong_angle+damage_not_visible` NEI case more than any contradiction case.

### 4. Explainability

**Policy A:** "We have a clear photograph showing the part with no damage."
→ True but misleading when another photo suggests otherwise.

**Policy B:** "The submitted images give conflicting signals. Human review required."
→ Fully accurate and defensible.

### 5. Insurance Workflow Realism

Insurance adjustors do not contradict a claim on the basis of one clear image when another image in the same submission shows possible damage. Standard practice:

1. If clear images show **no damage and no other image suggests damage** → reject (Policy A: correct)
2. If clear images show **no damage but another image suggests possible damage** → request additional photos / human review (Policy B: correct)
3. If clear images show **a different type of damage** → contradict (both agree)

---

## Root Cause of the Bug in Current Implementation

The bug is in `_assess_single_atom`:

```python
for obs in usable_obs:
    if obs.damage_observed:
        damage_obs_ids.append(obs.image_id)
        if _observation_matches_atom(obs, atom):
            supporting_ids.append(obs.image_id)
        else:
            contradicting_ids.append(obs.image_id)  # ← BUG: includes issue_type=None
```

When `damage_observed=True` but `issue_type_observed=None` (type undetermined), the observation is placed in `contradicting_ids`. This is wrong: an unknown damage type is not a contradiction of the claimed type — it's an absence of type confirmation.

The correct categorization:
- `issue_type_observed == claimed_type` → `supporting_ids`
- `issue_type_observed is not None and ≠ claimed_type` → `contradicting_ids` (confirmed wrong type)
- `issue_type_observed is None, damage_observed=True` → `ambiguous_ids` (possibly confirming, type unclear)

---

## Verdict: Implement Policy B (minimal patch)

**Reason:** Policy B strictly dominates in:
- False contradiction risk (significantly lower)
- Insurance workflow realism (standard practice)
- Explainability (always accurate)
- Sample data alignment (no sample contradicted case has this pattern)

Policy B does not harm benchmark alignment because the conflicting evidence scenario does not appear in the `contradicted` category of sample data.

---

## Minimal Patch

Three targeted changes to `rules.py`:

### Change 1: `_assess_single_atom` — separate ambiguous damage from wrong-type damage

```python
# Before:
else:
    contradicting_ids.append(obs.image_id)  # ← wrong: issue_type=None ≠ contradiction

# After:
elif obs.issue_type_observed is not None:
    # Confirmed different damage type = genuine contradiction
    contradicting_ids.append(obs.image_id)
# else: damage_observed=True, issue_type=None → ambiguous; handled in _compute_alignment
```

### Change 2: `_compute_alignment` — detect the conflicting case

```python
# After "MISMATCH case 1" (clear_no_damage and not damage_obs_ids):
# NEW: Conflicting — clear images say no damage, but other images observe unknown damage
ambiguous_damage_ids = [
    iid for iid in damage_obs_ids
    if iid not in supporting_ids and iid not in contradicting_ids
]
if clear_no_damage and ambiguous_damage_ids:
    ids_clear = [obs.image_id for obs in clear_no_damage]
    reason = (
        f"Conflicting signals: clear images {ids_clear} show no damage, "
        f"but images {ambiguous_damage_ids} show possible damage of undetermined type. "
        "Routed to manual review."
    )
    return "unclear", reason
```

### Change 3: `_collect_risk_flags` — surface conflicting evidence via existing flag

No new flag needed. The conflicting case produces `unclear` alignment → `not_enough_information`. 
The `DAMAGE_NOT_VISIBLE` flag already fires correctly (clear visible images + at least one has no damage).
`MANUAL_REVIEW_REQUIRED` fires via `DAMAGE_NOT_VISIBLE` is not a trigger... need to add it.

Actually add: when `alignment=unclear` AND clear_no_damage AND damage_obs_ids → add `MANUAL_REVIEW_REQUIRED` directly via the not_enough_information path.

Actually the simplest approach: add `DAMAGE_NOT_VISIBLE` to the `_MANUAL_REVIEW_TRIGGERS` set.

---

## Schema Constraint

`ClaimAtomAssessment.alignment` is validated to `{"match", "mismatch", "unclear"}` (frozen schema). Policy B is implemented using `"unclear"` — no schema change required.
