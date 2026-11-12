"""Exploration 2: caps on stacked positions, and the cascade depth needed (post hoc, auditor-only, both segments seen)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_lib as al
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
pd.set_option("display.width", 240)

def add_counts(ev, window=240):
    ev = ev.sort_values(["symbol", "ts"], kind="stable").reset_index(drop=True)
    ev["minute"] = ((ev.ts - EPOCH) // pd.Timedelta(minutes=1)).astype(np.int64)
    n_any = np.zeros(len(ev), dtype=int)
    for sym, idx in ev.groupby("symbol").indices.items():
        m = ev["minute"].to_numpy()[idx]
        n_any[idx] = np.arange(len(m)) - np.searchsorted(m, m - window, side="left")
    ev["n_prior"] = n_any
    return ev

def capped(ev, h, cap, min_prior=0):
    """Take an event when fewer than `cap` positions are open in that coin and it has >= min_prior earlier events in 240 min."""
    keep = np.zeros(len(ev), dtype=bool)
    m_all, n_all = ev["minute"].to_numpy(), ev["n_prior"].to_numpy()
    for sym, idx in ev.groupby("symbol").indices.items():
        open_until = []
        for i in idx:
            m = m_all[i]
            open_until = [u for u in open_until if u > m]
            if n_all[i] >= min_prior and len(open_until) < cap:
                keep[i] = True
                open_until.append(m + h + 1)
    return keep

for seg in ("design", "validation"):
    ev = pd.read_parquet(os.path.join(al.OUT, f"events_{seg}_um.parquet"))
    ev = add_counts(ev[ev.def_id == "sweep_N240_x10_k1"])
    ev["day"] = ev["minute"] // 1440
    ev["cost"] = ev["symbol"].map(al.taker_round_trip_bp)
    h = 240
    e = ev[np.isfinite(ev[f"g{h}"])].reset_index(drop=True)
    print(f"\n===== {seg}: all events {len(e)}, gross {e[f'g{h}'].mean():.2f}")
    print("exact number of earlier events in the coin within 240 min:")
    rows = []
    for k in range(0, 11):
        s = e[e.n_prior == k]
        if len(s) < 30: continue
        st = al.cluster_stats(s[f"g{h}"].to_numpy(), s["day"].to_numpy(), adjacent=True)
        rows.append(dict(n_prior=k, events=len(s), days=st["clusters"], gross=st["mean"], t=st["t"], median=s[f"g{h}"].median(), net_taker=st["mean"] - s["cost"].mean(),
                         long_gross=s.loc[s.dir > 0, f"g{h}"].mean(), short_gross=s.loc[s.dir < 0, f"g{h}"].mean()))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("cap on simultaneously open positions per coin (take every event while below the cap):")
    rows = []
    for cap in (1, 2, 3, 5, 10, 1000):
        keep = capped(e, h, cap)
        s = e[keep]
        st = al.cluster_stats(s[f"g{h}"].to_numpy(), s["day"].to_numpy(), adjacent=True)
        mk = s[(s.mk_off > 0) & s.mk_ctx & np.isfinite(s[f"mk_g{h}"])]
        ms = al.cluster_stats(mk[f"mk_g{h}"].to_numpy(), mk["day"].to_numpy(), adjacent=True)
        rows.append(dict(cap=cap, events=len(s), share=len(s) / len(e), gross=st["mean"], t_gross=st["t"], net_taker=st["mean"] - s["cost"].mean(),
                         t_net_taker=(st["mean"] - s["cost"].mean()) / st["se"], maker_gross_fill=ms["mean"], maker_net4=ms["mean"] - 4, t_maker_net4=(ms["mean"] - 4) / ms["se"]))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("one position per coin, entered only at an event with at least k earlier events in 240 min (POST HOC, exploratory):")
    rows = []
    for k in (0, 1, 2, 3, 4, 6, 8):
        keep = capped(e, h, 1, min_prior=k)
        s = e[keep]
        st = al.cluster_stats(s[f"g{h}"].to_numpy(), s["day"].to_numpy(), adjacent=True)
        wk = al.cluster_stats(s[f"g{h}"].to_numpy(), (s["day"] // 7).to_numpy())
        rows.append(dict(min_prior=k, trades=len(s), days=st["clusters"], gross=st["mean"], t_gross=st["t"], t_week=wk["t"], median=s[f"g{h}"].median(),
                         net_taker=st["mean"] - s["cost"].mean(), t_net_taker=(st["mean"] - s["cost"].mean()) / st["se"],
                         long_share=(s.dir > 0).mean(), long_gross=s.loc[s.dir > 0, f"g{h}"].mean(), short_gross=s.loc[s.dir < 0, f"g{h}"].mean()))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
