"""Audit, Part B: the small-sweep wait pooled over design and validation (trend trades; every calendar day)."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_lib as al
import a10_part_b as pb
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
pieces = []
for segment in ("design", "validation"):
    trades = pb.trend_trades(segment)
    first_event = None if segment == "design" else al.T_DESIGN_END
    for sym in pb.SYMBOLS:
        g, _ = al.load_grid("spot", sym, segment)
        o, h, l, c, present = al.arrays(g)
        signals = {"sweep_N30_x2_k3": al.sweep_signal(h, l, c, 30, 2.0, 3), "bos_s5_all": pb.bos_signal(h, l, c, 5)}
        mine = trades[trades["symbol"] == sym]
        midnights = g.index[(g.index.hour == 0) & (g.index.minute == 0)]
        sets = {"trend": [(pd.DatetimeIndex(mine.loc[mine["side"] > 0, "fill_time"]), 1), (pd.DatetimeIndex(mine.loc[mine["side"] < 0, "fill_time"]), -1)],
                "all_days": [(midnights, 1), (midnights, -1)]}
        for label, items in sets.items():
            for times, side in items:
                times = times[(times >= g.index[0]) & (times <= g.index[-1])]
                if first_event is not None:
                    times = times[times >= first_event]
                part = pb.evaluate(g, signals, times, side)
                part.insert(0, "set", label)
                part.insert(0, "segment", segment)
                pieces.append(part)
tr = pd.concat(pieces, ignore_index=True)
tr = tr[tr["window"] == 240]
tr["day"] = ((tr["time"] - EPOCH) // pd.Timedelta(days=1)).astype(np.int64)
out = {}
for (label, name), part in tr.groupby(["set", "structure"]):
    for seg, sub in (("design", part[part.segment == "design"]), ("validation", part[part.segment == "validation"]), ("both", part)):
        s = al.cluster_stats(sub["improvement"].to_numpy(), sub["day"].to_numpy())
        out[f"{label}|{name}|{seg}"] = dict(trades=s["n"], mean_bp=round(s["mean"], 2), t=round(s["t"], 2), median=round(float(sub["improvement"].median()), 2),
                                            worse=round(float((sub["improvement"] < 0).mean()), 3))
print(pd.DataFrame(out).T.to_string())
json.dump(out, open(os.path.join(al.OUT, "part_b_pooled.json"), "w"), indent=1)
