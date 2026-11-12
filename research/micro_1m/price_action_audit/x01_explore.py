import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_lib as al
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
pd.set_option("display.width", 220)
for seg in ("design", "validation"):
    ev = pd.read_parquet(os.path.join(al.OUT, f"events_{seg}_um.parquet"))
    ev = ev[ev.def_id == "sweep_N240_x10_k1"].copy()
    ev["day"] = ((ev.ts - EPOCH) // pd.Timedelta(days=1)).astype(int)
    for h in (60, 240):
        e = ev[np.isfinite(ev[f"g{h}"])]
        g = e[f"g{h}"]
        print(seg, h, "n", len(e), "mean", g.mean(), "median", g.median(), "std", g.std(), "p1,p5,p25,p75,p95,p99", np.percentile(g, [1, 5, 25, 75, 95, 99]).round(1))
        byday = e.groupby("day")[f"g{h}"].agg(["sum", "count", "mean"])
        n = byday["count"].to_numpy()
        print("   days", len(byday), "effective clusters (sum n)^2/sum n^2:", n.sum() ** 2 / (n ** 2).sum(), " max events/day", n.max(),
              " top10 days share of events", np.sort(n)[-10:].sum() / n.sum())
        print("   equal-day-weight mean of daily means:", byday["mean"].mean(), " t:", byday["mean"].mean() / (byday["mean"].std(ddof=1) / np.sqrt(len(byday))))
        contrib = byday["sum"].sort_values(ascending=False)
        tot = contrib.sum()
        print("   total sum", tot, " top 5 day contributions:", [(str((EPOCH + pd.Timedelta(days=int(d))).date()), round(v), int(byday.loc[d, "count"])) for d, v in contrib.head(5).items()])
        print("   bottom 5:", [(str((EPOCH + pd.Timedelta(days=int(d))).date()), round(v), int(byday.loc[d, "count"])) for d, v in contrib.tail(5).items()])
        for k in (1, 5, 10, 20):
            keep = contrib.index[k:]
            sub = byday.loc[keep]
            print(f"   drop top {k} contributing days: mean {sub['sum'].sum() / sub['count'].sum():.2f}", end=";")
        print()
        # long/short
        for d, lab in ((1, "long"), (-1, "short")):
            s = e[e.dir == d]
            print("  ", lab, len(s), "mean", s[f"g{h}"].mean(), "median", s[f"g{h}"].median(), "base", s[f"b{h}"].mean())
