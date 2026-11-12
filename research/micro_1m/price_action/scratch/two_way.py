import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import eventstudy as es, analyze as an
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
daily = an.load_daily("design", "um")
cost = an.costs("um", daily.symbol.unique())["taker"]
def two_way(def_id, h):
    part = daily[daily.def_id == def_id]
    rows = []
    for (d, y), chunk in part.groupby(["dir", "year"]):
        r = es.pooled_stats(chunk, (h,), cost_bp=cost).loc[h].to_dict(); r.update(dir="long" if d > 0 else "short", year=y); rows.append(r)
    t = pd.DataFrame(rows)[["dir", "year", "events", "days", "gross", "base", "excess", "t_excess", "net"]]
    print(f"\n=== {def_id} @ {h}: direction x year ===")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
def trim(def_id, h, drop=(0, 5, 15, 30)):
    part = daily[daily.def_id == def_id]
    by = part.groupby("day")[[f"n{h}", f"sg{h}", f"sb{h}"]].sum()
    by = by[by[f"n{h}"] > 0]
    e = by[f"sg{h}"] - by[f"sb{h}"]
    order = e.abs().sort_values(ascending=False).index
    out = []
    for k in drop:
        keep = by.drop(order[:k])
        m, se, t = es.cluster_mean_t((keep[f"sg{h}"] - keep[f"sb{h}"]).to_numpy(), keep[f"n{h}"].to_numpy(), keep.index.to_numpy(), es.hac_lag(h))
        out.append(f"drop {k} largest |day sum| days: excess {m:.2f} t {t:.2f}")
    # day-equal-weight view: median and mean of daily mean excess
    dm = e / by[f"n{h}"]
    out.append(f"per-day mean excess: mean {dm.mean():.2f}, median {dm.median():.2f}, share of days > 0: {(dm > 0).mean():.3f}, t(day means) {dm.mean() / dm.std(ddof=1) * np.sqrt(len(dm)):.2f}")
    print(f"--- {def_id} @ {h}: " + " | ".join(out))
for d, h in [("sweep_N240_x10_k1", 60), ("sweep_N240_x10_k1", 240), ("hold_N240_x10_k3", 60), ("sweep_N30_x2_k3", 5)]:
    two_way(d, h); trim(d, h)
