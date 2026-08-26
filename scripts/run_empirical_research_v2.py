#!/usr/bin/env python3
"""Research candidate setups using production indicator/trade geometry.
Research only: this script never promotes a setup to live signals.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.quant.indicators import TechnicalIndicators
from src.quant.patterns import PatternRecognizer
from src.agents.trade_construction_agent import TradeConstructionEngine

HORIZON=10

def outcome(df, i, levels):
    future=df.iloc[i+1:i+1+HORIZON]
    if len(future)<HORIZON: return None
    stop=float(levels.stop_loss_price); t1=float(levels.target_1); t2=float(levels.target_2)
    triggered=False; entry=None; entry_i=None
    for j,bar in future.iterrows():
        if float(bar.high)>=float(levels.entry_trigger_price):
            triggered=True; entry=float(levels.entry_trigger_price); entry_i=j; break
    if not triggered: return {'status':'UNTRIGGERED'}
    path=df.loc[entry_i+1:entry_i+HORIZON]
    if path.empty: return {'status':'TIMEOUT','entry':entry}
    for _,bar in path.iterrows():
        hit_sl=float(bar.low)<=stop; hit_t1=float(bar.high)>=t1
        if hit_sl and hit_t1: return {'status':'LOSS','r':-1.0,'entry':entry,'exit':stop,'outcome_date':str(bar.timestamp)}
        if hit_sl: return {'status':'LOSS','r':-1.0,'entry':entry,'exit':stop,'outcome_date':str(bar.timestamp)}
        if hit_t1: return {'status':'WIN_T1','r':float(levels.risk_reward_t1),'entry':entry,'exit':t1,'outcome_date':str(bar.timestamp)}
    last=float(path.iloc[-1].close); risk=entry-stop
    return {'status':'TIMEOUT','r':(last-entry)/risk if risk>0 else 0.0,'entry':entry,'exit':last,'outcome_date':str(path.iloc[-1].timestamp)}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',required=True); p.add_argument('--output',default='artifacts/empirical_research_v2.json'); a=p.parse_args()
    observations=[]
    for path in sorted(Path(a.data_dir).glob('*.csv')):
        if path.name.startswith('_'): continue
        df=pd.read_csv(path)
        req={'timestamp','open','high','low','close','volume'}
        if not req.issubset(df.columns): continue
        df['timestamp']=pd.to_datetime(df.timestamp,errors='coerce'); df=df.dropna(subset=list(req)).sort_values('timestamp').reset_index(drop=True)
        if len(df)<80: continue
        x=TechnicalIndicators.compute_all_indicators(df)
        for i in range(60,len(x)-HORIZON-2):
            window=x.iloc[:i+1].copy()
            matches=PatternRecognizer.evaluate_all_patterns(window)
            if not matches: continue
            levels,rejection=TradeConstructionEngine.construct_trade_levels(path.stem.upper(),window)
            if levels is None: continue
            # Only study patterns actually matched at the decision bar.
            for m in matches:
                o=outcome(x,i,levels)
                if o:
                    o.update({'symbol':path.stem.upper(),'setup':m.pattern_type.value,'quality':m.quality_score,'setup_date':str(x.loc[i,'timestamp'])})
                    observations.append(o)
    if not observations: raise SystemExit('No canonical empirical observations generated')
    d=pd.DataFrame(observations)
    groups=[]
    for setup,g in d.groupby('setup'):
        triggered=g[g.status!='UNTRIGGERED']; wins=triggered[triggered.status=='WIN_T1']; losses=triggered[triggered.status=='LOSS'];
        resolved=triggered[triggered.status.isin(['WIN_T1','LOSS'])]
        groups.append({'setup':setup,'observations':len(g),'triggered':len(triggered),'untriggered':int((g.status=='UNTRIGGERED').sum()),'wins_t1':len(wins),'losses':len(losses),'timeouts':int((triggered.status=='TIMEOUT').sum()),'resolved_win_rate':float(len(wins)/len(resolved)) if len(resolved) else None,'avg_r':float(triggered.r.mean()) if len(triggered) else None,'median_r':float(triggered.r.median()) if len(triggered) else None,'profit_factor':float(wins.r.sum()/abs(losses.r.sum())) if len(losses) and losses.r.sum()!=0 else None,'symbols':int(g.symbol.nunique())})
    payload={'status':'RESEARCH_ONLY','method':'CANONICAL_TRADE_CONSTRUCTION','horizon_bars':HORIZON,'groups':groups,'promotion_rule':'Independent walk-forward OOS required before live use.'}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print(json.dumps(payload,indent=2))

if __name__=='__main__': main()
