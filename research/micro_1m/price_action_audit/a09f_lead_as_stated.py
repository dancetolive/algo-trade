"""Audit: the researcher's follow-up lead exactly as stated ("buying after a sharp push to a new 4-hour low on the
altcoin perpetuals, with resting orders, holding hours"), under the portfolio rule the report itself names
(one position per coin). Long events of sweep_N240_x10_k1 on SOL, BNB, XRP, DOGE perpetuals, 240-minute hold."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_lib as al
import a09_overlap as ov
H = 240
out = {}
for segment in ("design", "validation"):
    ev = ov.load(segment)
    e = ev[np.isfinite(ev[f"g{H}"]) & np.isfinite(ev[f"b{H}"]) & (ev["dir"] > 0) & ~ev["symbol"].isin(["BTCUSDT", "ETHUSDT"])].reset_index(drop=True)
    n_prior = np.zeros(len(e), dtype=np.int64)
    for _, idx in e.groupby("symbol").indices.items():
        m = e["minute"].to_numpy()[idx]
        n_prior[idx] = np.arange(len(m)) - np.searchsorted(m, m - 240, side="left")
    e["n_prior"] = n_prior
    rec = {"every long event (stacked)": ov.summarise(e, H), "first long event in 240 min": ov.summarise(e[e["n_prior"] == 0], H)}
    for cap in (1, 2, 3, 5):
        rec[f"at most {cap} open per coin"] = ov.summarise(e[ov.take_with_cap(e, H, cap)], H)
    out[segment] = rec
json.dump(out, open(os.path.join(al.OUT, "lead_as_stated.json"), "w"), indent=1)
pd.set_option("display.width", 240)
for seg, rec in out.items():
    print("\n", seg, "- altcoin perpetuals, LONG after a sweep of the 4-hour low, 240-minute hold")
    print(pd.DataFrame(rec).T[["events", "gross", "t_gross", "excess", "t_excess", "net_taker", "t_net_taker", "maker_gross_per_fill", "maker_net_4bp", "t_maker_net_4bp",
                               "maker_in_taker_out", "t_maker_in_taker_out"]].to_string(float_format=lambda v: f"{v:.2f}"))
