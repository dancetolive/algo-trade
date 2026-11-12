"""Audit: the overlap decomposition of a09 repeated on SPOT (six coins; four of them never loaded by the researcher)."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_lib as al
import a09_overlap as ov

res = {}
for segment in ("design", "validation"):
    ev = ov.load(segment, "spot")
    for label, pick in (("spot six coins", ev), ("spot SOL+BNB+XRP+DOGE", ev[~ev["symbol"].isin(["BTCUSDT", "ETHUSDT"])])):
        h = 240
        e = pick[np.isfinite(pick[f"g{h}"]) & np.isfinite(pick[f"b{h}"])].reset_index(drop=True)
        rec = {"all": ov.summarise(e, h)}
        for name, a, b in (("0", 0, 0), ("1", 1, 1), ("2-3", 2, 3), ("4-7", 4, 7), ("8+", 8, 10 ** 9)):
            s = e[(e["n_prior"] >= a) & (e["n_prior"] <= b)]
            rec[f"earlier_events_{name}"] = ov.summarise(s, h)
        for cap in (1, 3, 10):
            rec[f"cap_{cap}"] = ov.summarise(e[ov.take_with_cap(e, h, cap)], h)
        res[f"{segment}|{label}"] = rec
json.dump(res, open(os.path.join(al.OUT, "overlap_spot.json"), "w"), indent=1)
pd.set_option("display.width", 220)
for key, rec in res.items():
    print("\n", key)
    print(pd.DataFrame(rec).T[["events", "gross", "t_gross", "median", "net_taker", "t_net_taker", "long_gross", "short_gross"]].to_string(float_format=lambda v: f"{v:.2f}"))
