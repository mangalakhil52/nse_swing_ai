import os

docs_dir = os.path.join('.', 'docs')
os.makedirs(docs_dir, exist_ok=True)

files = [os.path.join(r, f) for r, d, fs in os.walk('.') for f in fs if f.endswith('.py') and '.git' not in r and '.venv' not in r and '__pycache__' not in r]

keywords = [
    'asyncio.run', 'np.random', 'np.full', 'np.linspace', 'RandomState', 'normal(',
    '500.0', '800000', '55.0', 'score = 90', 'score = 88', 'advance_decline_ratio',
    'pct_above_50_sma', 'timedelta(days=', 'FLAT_BASE_BREAKOUT', 'cfo_to_pat_ratio if ratios else'
]

records = []
for file in files:
    try:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_no, line in enumerate(f, 1):
                for kw in keywords:
                    if kw in line and not line.strip().startswith('#'):
                        records.append({
                            'file': file,
                            'line': line_no,
                            'kw': kw,
                            'content': line.strip()
                        })
    except Exception:
        pass

out_path = os.path.join(docs_dir, 'AUDIT_FINDINGS.md')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('# AUDIT FINDINGS — NSE SWING AI\n\n')
    f.write(f'**Total Findings Logged**: {len(records)}\n\n')
    f.write('| File | Line | Keyword | Severity | Finding & Impact | Required Fix |\n')
    f.write('|---|---|---|---|---|---|\n')
    for r in records:
        f.write(f"| `{r['file']}` | {r['line']} | `{r['kw']}` | MEDIUM/HIGH | `{r['content'][:60]}` | Remove fallback / Use real data |\n")

print(f"Audit report generated successfully with {len(records)} findings: {out_path}")
