import csv, pathlib
with open(pathlib.Path(__file__).parent.parent / 'dataset/sample_claims.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
print('Columns:', list(rows[0].keys()))
print()
for col in ['claim_status', 'severity', 'evidence_standard_met', 'valid_image', 'risk_flags']:
    vals = sorted(set(r[col] for r in rows))
    print(f'{col}: {vals}')
print()
for r in rows:
    print(
        f"status={r['claim_status']:22} sev={r['severity']:8} "
        f"esm={r['evidence_standard_met']:5} vi={r['valid_image']:5} "
        f"flags={r['risk_flags']}"
    )
