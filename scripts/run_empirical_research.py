#!/usr/bin/env python3
"""Research candidate setups without promoting them to live signals."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from scripts.build_empirical_outcomes import process_symbol


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',required=True); p.add_argument('--output',default='artifacts/empirical_research.json'); a=p.parse_args()
    rows=[]
    for path in sorted(Path(a.data_dir).glob('*.csv')):
        if path.name.startswith('_'): continue
        df=pd.read_csv(path)
        if not {'timestamp','open','high','low','close','volume'}.issubset(df.columns): continue
        df['timestamp']=pd.to_datetime(df['timestamp'],errors='coerce'); df=df.dropna(subset=['timestamp','open','high','low','close','volume'])
        if len(df)>=80: rows.extend(process_symbol(path.stem.upper(),df))
    if not rows: raise SystemExit('No empirical observations generated')
    d=pd.DataFrame(rows)
    groups=[]
    for (setup,regime),g in d.groupby(['setup','regime']):
        n=len(g); wins=int((g.result=='WIN').sum()); losses=int((g.result=='LOSS').sum()); avg_r=float(g.r_multiple.mean());
        groups.append({'setup':setup,'regime':regime,'samples':n,'wins':wins,'losses':losses,'win_rate':wins/n,'avg_r':avg_r,'median_r':float(g.r_multiple.median()),'profit_factor':float(g.loc[g.r_multiple>0,'r_multiple'].sum()/abs(g.loc[g.r_multiple<0,'r_multiple'].sum())) if losses else None})
    payload={'status':'RESEARCH_ONLY','observations':len(d),'symbols':int(d.symbol.nunique()),'groups':groups,'promotion_rule':'No setup may enter live signal generation without independent OOS evidence.'}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print(json.dumps(payload,indent=2))

if __name__=='__main__': main()
