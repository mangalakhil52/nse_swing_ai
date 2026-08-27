#!/usr/bin/env python3
"""Analyze frozen 4/4 confluence with simple market-regime and cost stress.
Research only; no live promotion. Uses NIFTY proxy if available, otherwise per-symbol
trend regime is used. Costs are expressed as R deductions per completed trade.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quant.indicators import TechnicalIndicators
from src.agents.trade_construction_agent import TradeConstructionEngine
H=10
COSTS_R=[0.00,0.05,0.10,0.15,0.20]

def calc(df,i):
    x=df
    vals=[x.loc[i,'close'],x.loc[i,'ema_20'],x.loc[i,'ema_50'],x.loc[i,'rsi_14'],x.loc[i,'adx_14'],x.loc[i,'rvol_20']]
    if not all(np.isfinite(v) for v in vals):return None
    c,e20,e50,rsi,adx,rv=map(float,vals)
    if not(c>e20>e50 and 55<=rsi<=75 and adx>=20 and rv>=1.2 and c>=float(x.high.iloc[i-20:i].max())*.99):return None
    levels,_=TradeConstructionEngine.construct_trade_levels('RST',x.iloc[:i+1]);
    if levels is None:return None
    future=x.iloc[i+1:i+1+H]; trig=next((j for j,b in future.iterrows() if float(b.high)>=float(levels.entry_trigger_price)),None)
    if trig is None:return None
    path=x.loc[trig+1:trig+H];entry=float(levels.entry_trigger_price);stop=float(levels.stop_loss_price);target=float(levels.target_1);risk=entry-stop
    if risk<=0 or path.empty:return None
    for _,b in path.iterrows():
        if float(b.low)<=stop:return {'r':-1.,'status':'LOSS','regime':'BULL' if c>e50 else 'BEAR'}
        if float(b.high)>=target:return {'r':float(levels.risk_reward_t1),'status':'WIN','regime':'BULL' if c>e50 else 'BEAR'}
    return {'r':(float(path.iloc[-1].close)-entry)/risk,'status':'TIMEOUT','regime':'BULL' if c>e50 else 'BEAR'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--start',required=True);p.add_argument('--end',required=True);p.add_argument('--output',default='artifacts/regime_cost_stress.json');a=p.parse_args();rows=[]
    for f in sorted(Path(a.data_dir).glob('*.csv')):
        if f.name.startswith('_'):continue
        d=pd.read_csv(f);req={'timestamp','open','high','low','close','volume'}
        if not req.issubset(d.columns):continue
        d['timestamp']=pd.to_datetime(d.timestamp,errors='coerce');d=d.dropna(subset=list(req)).sort_values('timestamp').reset_index(drop=True);x=TechnicalIndicators.compute_all_indicators(d);m=(x.timestamp>=pd.Timestamp(a.start))&(x.timestamp<=pd.Timestamp(a.end));x=x.loc[m].reset_index(drop=True)
        for i in range(60,len(x)-H-1):
            o=calc(x,i)
            if o:o.update({'symbol':f.stem.upper(),'date':str(x.loc[i,'timestamp'])});rows.append(o)
    if not rows:raise SystemExit('No trades')
    d=pd.DataFrame(rows);out={'status':'RESEARCH_ONLY','trades':len(d),'regimes':{},'cost_stress':{}}
    for regime,g in d.groupby('regime'):
        out['regimes'][regime]={'trades':len(g),'win_rate':float((g.status=='WIN').mean()),'avg_r':float(g.r.mean()),'profit_factor':float(g.loc[g.r>0,'r'].sum()/abs(g.loc[g.r<0,'r'].sum())) if (g.r<0).any() else None}
    for cost in COSTS_R:
        r=d.r-cost;out['cost_stress'][str(cost)]={'avg_r':float(r.mean()),'total_r':float(r.sum()),'positive_trades':int((r>0).sum()),'negative_trades':int((r<0).sum()),'profit_factor':float(r[r>0].sum()/abs(r[r<0].sum())) if (r<0).any() else None}
    o=Path(a.output);o.parent.mkdir(parents=True,exist_ok=True);o.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
