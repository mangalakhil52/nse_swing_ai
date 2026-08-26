#!/usr/bin/env python3
"""Build an empirical setup/outcome table from point-in-time OHLCV data.

This deliberately uses only deterministic, production-style technical setup
rules and future bars strictly after the observation date. It never uses live
or current-universe information and never assigns a probability when the
sample is insufficient.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

MIN_SAMPLE = 30
HORIZON = 10


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().sort_values("timestamp").reset_index(drop=True)
    c = x["close"].astype(float)
    h = x["high"].astype(float)
    l = x["low"].astype(float)
    v = x["volume"].astype(float)
    x["ema20"] = c.ewm(span=20, adjust=False).mean()
    x["ema50"] = c.ewm(span=50, adjust=False).mean()
    x["sma50"] = c.rolling(50).mean()
    x["avg_vol20"] = v.rolling(20).mean()
    x["atr14"] = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1).rolling(14).mean()
    x["hh20"] = h.shift(1).rolling(20).max()
    x["ret20"] = c.pct_change(20)
    return x


def setup_at(x: pd.DataFrame, i: int) -> bool:
    if i < 60 or any(pd.isna(x.loc[i, k]) for k in ["ema20","ema50","sma50","avg_vol20","atr14","hh20"]):
        return False
    close = float(x.loc[i, "close"])
    return bool(close > x.loc[i,"ema20"] > x.loc[i,"ema50"] and close > x.loc[i,"sma50"] and close >= x.loc[i,"hh20"] * 0.995 and float(x.loc[i,"volume"]) >= 1.25 * float(x.loc[i,"avg_vol20"]))


def outcome_at(x: pd.DataFrame, i: int) -> dict | None:
    entry = float(x.loc[i, "close"])
    atr = float(x.loc[i, "atr14"])
    if entry <= 0 or not np.isfinite(atr) or atr <= 0:
        return None
    stop = entry - 1.5 * atr
    target = entry + 2.0 * (entry - stop)
    future = x.iloc[i+1:i+1+HORIZON]
    if len(future) < HORIZON:
        return None
    result = "NEITHER"
    exit_price = float(future.iloc[-1]["close"])
    exit_date = future.iloc[-1]["timestamp"]
    for _, bar in future.iterrows():
        hit_stop = float(bar["low"]) <= stop
        hit_target = float(bar["high"]) >= target
        # Conservative same-bar convention: if both are touched, assume SL first.
        if hit_stop:
            result = "LOSS"
            exit_price = stop
            exit_date = bar["timestamp"]
            break
        if hit_target:
            result = "WIN"
            exit_price = target
            exit_date = bar["timestamp"]
            break
    risk = entry - stop
    r_multiple = (exit_price - entry) / risk
    return {"setup_date": x.loc[i,"timestamp"], "entry":entry,"stop":stop,"target":target,"result":result,"r_multiple":r_multiple,"outcome_date":exit_date}


def process_symbol(symbol: str, df: pd.DataFrame) -> list[dict]:
    x = indicators(df)
    rows=[]
    for i in range(len(x)-HORIZON-1):
        if setup_at(x, i):
            out = outcome_at(x, i)
            if out:
                out.update({"symbol":symbol,"setup":"TREND_BREAKOUT_VOLUME","regime":"UNKNOWN"})
                rows.append(out)
    return rows


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",required=True)
    p.add_argument("--output",default="artifacts/empirical_outcomes.csv")
    args=p.parse_args()
    data_dir=Path(args.data_dir)
    rows=[]
    for path in sorted(data_dir.glob("*.csv")):
        if path.name.startswith("_"): continue
        df=pd.read_csv(path)
        required={"timestamp","open","high","low","close","volume"}
        if not required.issubset(df.columns): continue
        df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
        df=df.dropna(subset=["timestamp","open","high","low","close","volume"])
        if len(df)>=80: rows.extend(process_symbol(path.stem.upper(),df))
    if not rows: raise SystemExit("No empirical setup outcomes were generated")
    out=pd.DataFrame(rows).sort_values(["setup_date","symbol"]).reset_index(drop=True)
    out["setup_date"]=out["setup_date"].astype(str)
    out["outcome_date"]=out["outcome_date"].astype(str)
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True); out.to_csv(output,index=False)
    grouped=out.groupby(["setup","regime"]).agg(samples=("result","size"),wins=("result",lambda s:(s=="WIN").sum()),avg_r=("r_multiple","mean"),median_r=("r_multiple","median")).reset_index()
    grouped["win_rate"]=grouped["wins"]/grouped["samples"]
    grouped["empirical_ready"]=grouped["samples"]>=MIN_SAMPLE
    manifest={"rows":len(out),"symbols":out.symbol.nunique(),"minimum_sample":MIN_SAMPLE,"horizon_bars":HORIZON,"groups":grouped.to_dict(orient="records"),"probability_status":"EMPIRICAL_ONLY_WHEN_SAMPLE_GE_30"}
    output.with_name(output.stem+"_summary.json").write_text(json.dumps(manifest,indent=2,default=str),encoding="utf-8")
    print(json.dumps(manifest,indent=2,default=str))
    return 0

if __name__=="__main__": raise SystemExit(main())
