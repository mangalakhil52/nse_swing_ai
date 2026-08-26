#!/usr/bin/env python3
"""Reject empirical evidence that is too small, unstable, or non-positive EV."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
MIN_SAMPLE=30
MIN_WIN_RATE=0.50
MIN_AVG_R=0.15

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',default='artifacts/empirical_gate.json'); a=p.parse_args()
    df=pd.read_csv(a.input)
    required={'setup','regime','result','r_multiple'}
    missing=required-set(df.columns)
    if missing: raise SystemExit(f'Missing empirical columns: {sorted(missing)}')
    groups=[]
    for (setup,regime),g in df.groupby(['setup','regime']):
        n=len(g); wins=int((g.result=='WIN').sum()); win_rate=wins/n if n else 0.0; avg_r=float(g.r_multiple.mean()) if n else 0.0
        groups.append({'setup':setup,'regime':regime,'samples':n,'win_rate':win_rate,'avg_r':avg_r,'ready':n>=MIN_SAMPLE,'edge_pass':n>=MIN_SAMPLE and win_rate>=MIN_WIN_RATE and avg_r>=MIN_AVG_R})
    payload={'status':'OK' if groups and any(x['edge_pass'] for x in groups) else 'REJECT','minimum_sample':MIN_SAMPLE,'minimum_win_rate':MIN_WIN_RATE,'minimum_avg_r':MIN_AVG_R,'groups':groups}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print(json.dumps(payload,indent=2)); return 0 if payload['status']=='OK' else 2
if __name__=='__main__': raise SystemExit(main())
