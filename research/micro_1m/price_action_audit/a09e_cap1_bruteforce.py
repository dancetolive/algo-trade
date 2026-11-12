"""Audit: brute-force confirmation of the one-position-per-coin figures, from the raw frame by timestamp.

For each coin the events (my event table = the researcher's events) are walked in time order with a plain loop:
if no position is open, enter at the open of the next minute and leave at the open 240 minutes later; otherwise skip.
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_lib as al
from algotrade.data import store
MIN = pd.Timedelta(minutes=1)
out = {}
for segment in ("design", "validation"):
    ev = pd.read_parquet(os.path.join(al.OUT, f"events_{segment}_um.parquet"))
    ev = ev[(ev.def_id == "sweep_N240_x10_k1") & np.isfinite(ev["g240"]) & np.isfinite(ev["b240"])]
    tot, cnt, all_r = 0.0, 0, []
    for sym, sub in ev.groupby("symbol"):
        end = al.T_DESIGN_END if segment == "design" else None
        start = None if segment == "design" else pd.Timestamp("2023-12-25", tz="UTC")
        o = store.load_klines("um", "1m", sym, start=start, end=end, columns=["open"])["open"]
        free_at = None
        for ts, d in zip(sub.sort_values("ts")["ts"], sub.sort_values("ts")["dir"]):
            entry_t = ts + MIN
            if free_at is not None and entry_t < free_at:
                continue
            exit_t = entry_t + 240 * MIN
            r = d * (o.loc[exit_t] / o.loc[entry_t] - 1.0) * 1e4
            all_r.append(r)
            free_at = exit_t
    all_r = np.array(all_r)
    out[segment] = dict(trades=int(len(all_r)), gross=float(all_r.mean()), median=float(np.median(all_r)))
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(al.OUT, "cap1_bruteforce.json"), "w"), indent=1)
