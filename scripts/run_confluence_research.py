#!/usr/bin/env python3
"""Research the production-style confluence stack without live promotion."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.quant.indicators import TechnicalIndicators
from src.agents.trade_construction_agent import TradeConstructionEngine

HORIZON=10

def evaluate(df,i,levels):
    future=df.iloc[i+1:i+1+HORIZON]
    if len(future)<HORIZON:return None
    trig=next((j for j,b in future.iterrows() if float(b.high)>=float(levels.entry_trigger_price)),None)
    if trig is None:return {'status':'UNTRIGGERED'}
    path=df.loc[trig+1:trig+HORIZON]
    entry=float(levels.entry_trigger_price); stop=float(levels.stop_loss_price); target=float(levels.target_1); risk=entry-stop
    for _,b in path.iterrows():
        sl=float(b.low)<=stop; tp=float(b.high)>=target
        if sl:return {'status':'LOSS','r':-1.0}
        if tp:return {'status':'WIN','r':float(levels.risk_reward_t1)}
    return {'status':'TIMEOUT','r':(float(path.iloc[-1].close)-entry)/risk if risk>0 else 0.0}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',default='artifacts/confluence_research.json');a=p.parse_args();rows=[]
    for f in sorted(Path(a.data_dir).glob('*.csv')):
        if f.name.startswith('_'):continue
        d=pd.read_csv(f)
        req={'timestamp','open','high','low','close','volume'}
        if not req.issubset(d.columns):continue
        d['timestamp']=pd.to_datetime(d.timestamp,errors='coerce');d=d.dropna(subset=list(req)).sort_values('timestamp').reset_index(drop=True)
        if len(d)<100:continue
        x=TechnicalIndicators.compute_all_indicators(d)
        for i in range(60,len(x)-HORIZON-1):
            c=float(x.loc[i,'close']); ema20=float(x.loc[i,'ema_20']); ema50=float(x.loc[i,'ema_50']); rsi=float(x.loc[i,'rsi_14']); adx=float(x.loc[i,'adx_14']); rvol=float(x.loc[i,'rvol_20']); atrpct=float(x.loc[i,'atr_pct'])
            if not all(np.isfinite(v) for v in [c,ema20,ema50,rsi,adx,rvol,atrpct]):continue
            trend=int(c>ema20>ema50); momentum=int(55<=rsi<=75 and adx>=20); volume=int(rvol>=1.2); breakout=int(c>=float(x.high.iloc[i-20:i].max())*0.99); quality=trend+momentum+volume+breakout
            if quality<3:continue
            levels,rejection=TradeConstructionEngine.construct_trade_levels(f.stem.upper(),x.iloc[:i+1])
            if levels is None:continue
            o=evaluate(x,i,levels)
            if o:o.update({'symbol':f.stem.upper(),'setup_date':str(x.loc[i,'timestamp']),'quality':quality});rows.append(o)
    if not rows:raise SystemExit('No confluence observations generated')
    d=pd.DataFrame(rows);groups=[]
    for q,g in d.groupby('quality'):
        tr=g[g.status!='UNTRIGGERED'];res=tr[tr.status.isin(['WIN','LOSS'])];wins=(res.status=='WIN').sum();loss=(res.status=='LOSS').sum();groups.append({'quality':int(q),'observations':len(g),'triggered':len(tr),'untriggered':int((g.status=='UNTRIGGERED').sum()),'resolved':len(res),'wins':int(wins),'losses':int(loss),'win_rate':float(wins/len(res)) if len(res) else None,'avg_r':float(tr.r.mean()) if len(tr) else None,'profit_factor':float(tr.loc[tr.r>0,'r'].sum()/abs(tr.loc[tr.r<0,'r'].sum())) if (tr.r<0).any() else None,'symbols':int(g.symbol.nunique())})
    payload={'status':'RESEARCH_ONLY','confluence_definition':'trend + momentum + volume + breakout, minimum quality 3/4','horizon_bars':HORIZON,'groups':groups,'promotion_rule':'Independent walk-forward OOS required before live use.'};out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
