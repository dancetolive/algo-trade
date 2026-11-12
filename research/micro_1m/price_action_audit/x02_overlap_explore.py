"""Exploration: how does the per-event mean of the selected pair depend on where the event sits in a burst?"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_lib as al
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
pd.set_option("display.width", 220)

def prior_counts(ev, window_min=240):
    """Number of earlier events in the same coin within the previous `window_min` minutes (any direction / same direction)."""
    n_any = np.zeros(len(ev), dtype=int); n_same = np.zeros(len(ev), dtype=int); gap = np.full(len(ev), np.inf)
    for sym, idx in ev.groupby("symbol").indices.items():
        sub = ev.iloc[idx]
        m = (sub["ts"].astype("int64") // (60 * 10**9 if str(sub["ts"].dtype).find("ns") >= 0 else 1)).to_numpy()
        m = ((sub["ts"] - EPOCH) // pd.Timedelta(minutes=1)).to_numpy()
        d = sub["dir"].to_numpy()
        order = np.argsort(m, kind="stable")
        m, d, ii = m[order], d[order], idx[order]
        lo = np.searchsorted(m, m - window_min, side="left")
        cnt = np.arange(len(m)) - lo
        n_any[ii] = cnt
        for sgn in (1, -1):
            sel = d == sgn
            ms = m[sel]
            los = np.searchsorted(ms, ms - window_min, side="left")
            n_same[ii[sel]] = np.arange(len(ms)) - los
        g = np.diff(m, prepend=-10**12)
        gap[ii] = g
    return n_any, n_same, gap

for seg in ("design", "validation"):
    ev = pd.read_parquet(os.path.join(al.OUT, f"events_{seg}_um.parquet"))
    ev = ev[ev.def_id == "sweep_N240_x10_k1"].reset_index(drop=True)
    ev["day"] = ((ev.ts - EPOCH) // pd.Timedelta(days=1)).astype(int)
    n_any, n_same, gap = prior_counts(ev)
    ev["n_any"], ev["n_same"], ev["gap"] = n_any, n_same, gap
    for h in (240, 60):
        e = ev[np.isfinite(ev[f"g{h}"])]
        print(f"\n{seg} h={h}: all events {len(e)}, mean {e[f'g{h}'].mean():.2f}")
        bins = [(0, 0), (1, 1), (2, 3), (4, 7), (8, 15), (16, 10**6)]
        rows = []
        for a, b in bins:
            s = e[(e.n_any >= a) & (e.n_any <= b)]
            st = al.cluster_stats(s[f"g{h}"].to_numpy(), s["day"].to_numpy(), adjacent=h >= 240)
            lo, sh = s[s.dir > 0], s[s.dir < 0]
            rows.append(dict(prior_events_240m=f"{a}-{b}" if b < 10**6 else f"{a}+", events=len(s), share=len(s) / len(e), gross=st["mean"], t=st["t"],
                             long_n=len(lo), long_gross=lo[f"g{h}"].mean(), short_n=len(sh), short_gross=sh[f"g{h}"].mean(),
                             maker_fill_gross=s.loc[(s.mk_off > 0) & s.mk_ctx, f"mk_g{h}"].mean()))
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        fresh = e[e.n_any == 0]
        rest = e[e.n_any > 0]
        for name, s in (("fresh (no event in the coin in the previous 240 min)", fresh), ("not fresh", rest)):
            st = al.cluster_stats(s[f"g{h}"].to_numpy(), s["day"].to_numpy(), adjacent=h >= 240)
            ex = al.cluster_stats((s[f"g{h}"] - s[f"b{h}"]).to_numpy(), s["day"].to_numpy(), adjacent=h >= 240)
            print(f"   {name}: n {len(s)} gross {st['mean']:.2f} (t {st['t']:.2f}) excess {ex['mean']:.2f} (t {ex['t']:.2f})")
