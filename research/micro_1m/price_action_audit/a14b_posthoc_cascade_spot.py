"""POST HOC, exploratory: the a14 rule on SPOT (six coins; costs 22-24 bp round trip). Same caveats as a14."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_lib as al
import a09_overlap as ov
H = 240
out = {}
for segment in ("design", "validation"):
    ev = ov.load(segment, "spot")
    e = ev[np.isfinite(ev[f"g{H}"]) & np.isfinite(ev[f"b{H}"]) & (ev["dir"] > 0)].reset_index(drop=True)
    n_prior = np.zeros(len(e), dtype=np.int64)
    for _, idx in e.groupby("symbol").indices.items():
        m = e["minute"].to_numpy()[idx]
        n_prior[idx] = np.arange(len(m)) - np.searchsorted(m, m - 240, side="left")
    e["n_prior"] = n_prior
    for k in (0, 2, 3, 4):
        s = e[ov.take_with_cap(e, H, 1, min_prior=k)]
        g = al.cluster_stats(s[f"g{H}"].to_numpy(), s["day"].to_numpy(), adjacent=True)
        cost = float(s["cost"].mean())
        out[f"{segment}|k={k}"] = dict(trades=g["n"], days=g["clusters"], gross=g["mean"], t_gross=g["t"], median=float(s[f"g{H}"].median()), cost=cost,
                                       net_taker=g["mean"] - cost, t_net_taker=(g["mean"] - cost) / g["se"])
json.dump(out, open(os.path.join(al.OUT, "posthoc_cascade_spot.json"), "w"), indent=1)
print(pd.DataFrame(out).T.to_string(float_format=lambda v: f"{v:.2f}"))
