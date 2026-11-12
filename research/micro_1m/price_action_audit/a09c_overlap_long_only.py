"""Audit: the long side only (the researcher's hypothesis for the hold-out), with position caps. Perpetuals."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_lib as al
import a09_overlap as ov

res = {}
for segment in ("design", "validation"):
    ev = ov.load(segment)
    h = 240
    e = ev[np.isfinite(ev[f"g{h}"]) & np.isfinite(ev[f"b{h}"]) & (ev["dir"] > 0)].reset_index(drop=True)
    # earlier LONG events in the same coin in the previous 240 minutes
    n_prior = np.zeros(len(e), dtype=np.int64)
    for _, idx in e.groupby("symbol").indices.items():
        m = e["minute"].to_numpy()[idx]
        n_prior[idx] = np.arange(len(m)) - np.searchsorted(m, m - 240, side="left")
    e["n_prior"] = n_prior
    rec = {"all long events": ov.summarise(e, h)}
    for name, a, b in (("0 earlier long events", 0, 0), ("1", 1, 1), ("2-3", 2, 3), ("4+", 4, 10 ** 9)):
        rec[f"earlier: {name}"] = ov.summarise(e[(e["n_prior"] >= a) & (e["n_prior"] <= b)], h)
    for cap in (1, 2, 3, 5, 10):
        rec[f"cap {cap} per coin"] = ov.summarise(e[ov.take_with_cap(e, h, cap)], h)
    res[segment] = rec
json.dump(res, open(os.path.join(al.OUT, "overlap_long_only.json"), "w"), indent=1)
pd.set_option("display.width", 240)
for key, rec in res.items():
    print("\n", key, "LONG ONLY, sweep_N240_x10_k1 @ 240")
    print(pd.DataFrame(rec).T[["events", "gross", "t_gross", "excess", "t_excess", "median", "net_taker", "t_net_taker", "maker_gross_per_fill", "maker_net_4bp", "t_maker_net_4bp"]]
          .to_string(float_format=lambda v: f"{v:.2f}"))
