#!/usr/bin/env python3
"""Chronological multi-window evaluation of frozen 4/4 confluence."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.quant.indicators import TechnicalIndicators
from src.agents.trade_construction_agent import TradeConstructionEngine
H=10

def run_symbol(df):
    x=TechnicalIndicators.compute_all_indicators(df); rows=[]
    for i in range(60,len(x)-H-1):
        vals=[x.loc[i,'close'],x.loc[i,'ema_20'],x.loc[i,'ema_50'],x.loc[i,'rsi_14'],x.loc[i,'adx_14'],x.loc[i,'rvol_20']]
        if not all(np.isfinite(v) for v in vals):continue
        c,e20,e50,rsi,adx,rv=map(float,vals)
        if not(c>e20>e50 and 55<=rsi<=75 and adx>=20 and rv>=1.2 and c>=float(x.high.iloc[i-20:i].max())*.99):continue
        levels,_=TradeConstructionEngine.construct_trade_levels('WF',x.iloc[:i+1]);
        if levels is None:continue
        future=x.iloc[i+1:i+1+H]; trig=next((j for j,b in future.iterrows() if float(b.high)>=float(levels.entry_trigger_price)),None)
        if trig is None:continue
        path=x.loc[trig+1:trig+H]; entry=float(levels.entry_trigger_price);stop=float(levels.stop_loss_price);target=float(levels.target_1);risk=entry-stop
        status='TIMEOUT';r=(float(path.iloc[-1].close)-entry)/risk if not path.empty and risk>0 else 0
        for _,b in path.iterrows():
            if float(b.low)<=stop:status='LOSS';r=-1.;break
            if float(b.high)>=target:status='WIN';r=float(levels.risk_reward_t1);break
        rows.append({'date':x.loc[i,'timestamp'],'status':status,'r':r,'symbol':df.attrs.get('symbol','')})
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--start',required=True);p.add_argument('--end',required=True);p.add_argument('--test-days',type=int,default=126);p.add_argument('--step-days',type=int,default=126);p.add_argument('--output',default='artifacts/confluence_walkforward.json');a=p.parse_args();allrows=[]
    for f in sorted(Path(a.data_dir).glob('*.csv')):
        if f.name.startswith('_'):continue
        d=pd.read_csv(f);req={'timestamp','open','high','low','close','volume'}
        if not req.issubset(d.columns):continue
        d['timestamp']=pd.to_datetime(d.timestamp,errors='coerce');d=d.dropna(subset=list(req)).sort_values('timestamp').reset_index(drop=True);d.attrs['symbol']=f.stem.upper()
        allrows.extend(run_symbol(d))
    if not allrows:raise SystemExit('No confluence observations')
    d=pd.DataFrame(allrows); dates=pd.Series(sorted(pd.to_datetime(d.date).dt.normalize().unique())); start=pd.Timestamp(a.start).normalize();end=pd.Timestamp(a.end).normalize();dates=dates[(dates>=start)&(dates<=end)].tolist();windows=[]
    for n in range(a.test_days,len(dates)+1,a.step_days):
        ws=pd.Timestamp(dates[n-a.test_days]);we=pd.Timestamp(dates[n-1]);g=d[(pd.to_datetime(d.date)>=ws)&(pd.to_datetime(d.date)<=we)];res=g[g.status.isin(['WIN','LOSS'])];wins=int((res.status=='WIN').sum());loss=int((res.status=='LOSS').sum());windows.append({'start':str(ws.date()),'end':str(we.date()),'trades':len(g),'resolved':len(res),'wins':wins,'losses':loss,'win_rate':wins/len(res) if len(res) else None,'avg_r':float(g.r.mean()) if len(g) else None,'profit_factor':float(g.loc[g.r>0,'r'].sum()/abs(g.loc[g.r<0,'r'].sum())) if (g.r<0).any() else None})
    report={'status':'RESEARCH_ONLY','definition':'Frozen 4/4 confluence','windows':windows,'windows_positive_avg_r':sum((w['avg_r'] or 0)>0 for w in windows),'promotion_rule':'Require majority positive windows and cost-stress pass before live use.'};out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
