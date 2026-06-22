import sys, csv, pathlib
sys.path.insert(0, '.')

d = pathlib.Path(__file__).parent.parent / 'dataset'
rows = list(csv.DictReader(open(d / 'sample_claims.csv', encoding='utf-8')))

groups = {}
for r in rows:
    k = r['claim_status']
    groups.setdefault(k, []).append(r)

for status, grp in sorted(groups.items()):
    print(f"=== {status} ({len(grp)} rows) ===")
    for r in grp:
        vi  = r['valid_image']
        esm = r['evidence_standard_met']
        sev = r['severity']
        fl  = r['risk_flags']
        print(f"  vi={vi:5} esm={esm:5} sev={sev:7} flags={fl}")
    print()

print("--- contradicted: flag breakdown ---")
for r in groups.get('contradicted', []):
    flags = r['risk_flags'].split(';')
    has_dnv = 'damage_not_visible' in flags
    has_cm  = 'claim_mismatch' in flags
    has_wo  = 'wrong_object' in flags
    vi  = r['valid_image']
    esm = r['evidence_standard_met']
    sev = r['severity']
    print(
        f"  vi={vi} esm={esm} sev={sev:7} "
        f"damage_not_visible={has_dnv} claim_mismatch={has_cm} wrong_object={has_wo}"
    )

print()
print("--- not_enough_information: flag breakdown ---")
for r in groups.get('not_enough_information', []):
    flags = r['risk_flags'].split(';')
    has_dnv = 'damage_not_visible' in flags
    has_mr  = 'manual_review_required' in flags
    has_wo  = 'wrong_object' in flags
    has_wa  = 'wrong_angle' in flags
    vi  = r['valid_image']
    esm = r['evidence_standard_met']
    print(
        f"  vi={vi} esm={esm} "
        f"damage_not_visible={has_dnv} wrong_object={has_wo} wrong_angle={has_wa} manual_review={has_mr}"
    )
