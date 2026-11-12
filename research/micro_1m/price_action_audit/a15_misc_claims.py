"""Audit: remaining figures quoted in the researcher's 'selected' text, recomputed from my event tables."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_lib as al
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
out = {}
for seg in ("design", "validation"):
    ev = pd.read_parquet(os.path.join(al.OUT, f"events_{seg}_um.parquet"))
    ev["day"] = ((ev["ts"] - EPOCH) // pd.Timedelta(days=1)).astype(np.int64)
    e = ev[ev.def_id == "sweep_N240_x10_k1"]
    for h in (240, 60):
        p = e[e.symbol.isin(["BTCUSDT", "ETHUSDT"]) & np.isfinite(e[f"g{h}"]) & np.isfinite(e[f"b{h}"])]
        g = al.cluster_stats(p[f"g{h}"].to_numpy(), p["day"].to_numpy(), adjacent=h >= 240)
        x = al.cluster_stats((p[f"g{h}"] - p[f"b{h}"]).to_numpy(), p["day"].to_numpy(), adjacent=h >= 240)
        out[f"{seg} BTC+ETH perp @{h}"] = dict(events=g["n"], gross=round(g["mean"], 2), t_excess=round(x["t"], 2), net_taker=round(g["mean"] - 12.0, 2))
        p = e[~e.symbol.isin(["BTCUSDT", "ETHUSDT"]) & np.isfinite(e[f"g{h}"]) & np.isfinite(e[f"b{h}"])]
        g = al.cluster_stats(p[f"g{h}"].to_numpy(), p["day"].to_numpy(), adjacent=h >= 240)
        x = al.cluster_stats((p[f"g{h}"] - p[f"b{h}"]).to_numpy(), p["day"].to_numpy(), adjacent=h >= 240)
        out[f"{seg} four altcoin perps @{h}"] = dict(events=g["n"], gross=round(g["mean"], 2), t_excess=round(x["t"], 2), net_taker=round(g["mean"] - 14.0, 2),
                                                     t_net_taker=round((g["mean"] - 14.0) / g["se"], 2))
    # late design sub-period quoted in the report (2022-07 .. 2023-12)
    if seg == "design":
        late = e[e.ts >= pd.Timestamp("2022-07-01", tz="UTC")]
        for h in (240, 60):
            p = late[np.isfinite(late[f"g{h}"]) & np.isfinite(late[f"b{h}"])]
            g = al.cluster_stats(p[f"g{h}"].to_numpy(), p["day"].to_numpy(), adjacent=h >= 240)
            out[f"design 2022-07..2023-12 @{h} (baseline of the full design run)"] = dict(events=g["n"], gross=round(g["mean"], 2), t_gross=round(g["t"], 2))
    # small sweep: where does the effect sit relative to the resting order
    s = ev[ev.def_id == "sweep_N30_x2_k3"]
    ok = np.isfinite(s["g5"])
    out[f"{seg} small sweep: taker gross if resting order later filled / never filled"] = [round(float(s.loc[ok & (s.mk_off > 0), "g5"].mean()), 2),
                                                                                             round(float(s.loc[ok & (s.mk_off == 0), "g5"].mean()), 2),
                                                                                             round(float((s.loc[ok, "mk_off"] == 0).mean()), 4)]
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(al.OUT, "misc_claims.json"), "w"), indent=1)
