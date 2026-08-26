#!/usr/bin/env python3
"""Strict chronological OOS test for frozen 4/4 confluence.
No threshold fitting is performed in the test period.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quant.indicators import TechnicalIndicators
from src.agents.trade_construction_agent import TradeConstructionEngine

H=10

def one(df,i):
    c=float(df.loc[i,'close']); e20=float(df.loc[i,'ema_20']); e50=float(df.loc[i,'ema_50']); rsi=float(df.loc[i,'rsi_14']); adx=float(df.loc[i,'adx_14']); rv=float(df.loc[i,'rvol_20']); atr=float(df.loc[i,'atr_pct'])
    if not all(np.isfinite(v) for v in [c,e20,e50,rsi,adx,rv,atr]): return None
    score=int(c>e20>e50)+int(55<=rsi<=75 and adx>=20)+int(rv>=1.2)+int(c>=float(df.high.iloc[i-20:i].max())*0.99)
    if score!=4:return None
    levels,rejection=TradeConstructionEngine.construct_trade_levels('OOS',df.iloc[:i+1])
    if levels is None:return None
    future=df.iloc[i+1:i+1+H]
    trig=next((j for j,b in future.iterrows() if float(b.high)>=float(levels.entry_trigger_price)),None)
    if trig is None:return {'status':'UNTRIGGERED'}
    path=df.loc[trig+1:trig+H]; entry=float(levels.entry_trigger_price);stop=float(levels.stop_loss_price);target=float(levels.target_1);risk=entry-stop
    for _,b in path.iterrows():
        if float(b.low)<=stop:return {'status':'LOSS','r':-1.0}
        if float(b.high)>=target:return {'status':'WIN','r':float(levels.risk_reward_t1)}
    last=float(path.iloc[-1].close);return {'status':'TIMEOUT','r':(last-entry)/risk if risk>0 else 0.0}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--start',required=True);p.add_argument('--end',required=True);p.add_argument('--output',default='artifacts/confluence_oos.json');a=p.parse_args();rows=[]
    for f in sorted(Path(a.data_dir).glob('*.csv')):
        if f.name.startswith('_'):continue
        d=pd.read_csv(f);req={'timestamp','open','high','low','close','volume'}
        if not req.issubset(d.columns):continue
        d['timestamp']=pd.to_datetime(d.timestamp,errors='coerce');d=d.dropna(subset=list(req)).sort_values('timestamp').reset_index(drop=True)
        x=TechnicalIndicators.compute_all_indicators(d);mask=(x.timestamp>=pd.Timestamp(a.start))&(x.timestamp<=pd.Timestamp(a.end));x=x.loc[mask].reset_index(drop=True)
        for i in range(60,len(x)-H-1):
            o=one(x,i)
            if o:o.update({'symbol':f.stem.upper(),'date':str(x.loc[i,'timestamp'])});rows.append(o)
    if not rows:raise SystemExit('No 4/4 confluence observations')
    d=pd.DataFrame(rows);tr=d[d.status!='UNTRIGGERED'];res=tr[tr.status.isin(['WIN','LOSS'])];wins=int((res.status=='WIN').sum());loss=int((res.status=='LOSS').sum());payload={'status':'OOS_RESEARCH_ONLY','definition':'4/4 frozen confluence','observations':len(d),'triggered':len(tr),'untriggered':int((d.status=='UNTRIGGERED').sum()),'resolved':len(res),'wins':wins,'losses':loss,'win_rate':wins/len(res) if len(res) else None,'avg_r':float(tr.r.mean()) if len(tr) else None,'profit_factor':float(tr.loc[tr.r>0,'r'].sum()/abs(tr.loc[tr.r<0,'r'].sum())) if (tr.r<0).any() else None,'symbols':int(d.symbol.nunique()),'promotion_rule':'Must pass independent multi-window walk-forward before live use.'}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
