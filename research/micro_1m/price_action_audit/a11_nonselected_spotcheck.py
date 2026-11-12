"""Audit (mirror image of check 3): the researcher's NEGATIVE findings rest on detectors I had not re-implemented.
Here eight non-selected definitions, one or two per family, are rebuilt independently and run through my own event
study on the six perpetuals (design). They must reproduce the corresponding rows of out/table_design.csv at all six
horizons. If a detector of theirs had a bug that hid a real effect, the independent version would disagree.

    python -I a11_nonselected_spotcheck.py

hold_N240_x10_k3, hold_N30_x2_k3 : breach and k closes beyond the level (continuation)
sweepmat_N60_x2_k1               : sweep of a level at least 10 minutes old
pdsweep_x2_k3                    : sweep of the previous UTC day's high / low
bos_s5_all                       : break of structure, 5/5 swings (bar-by-bar state machine)
fvg_m0p5, fvgfill_m0p5_W60       : fair value gap, and first return into it (monotonic stack)
disp_m3                          : displacement candle
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_lib as al  # noqa: E402
from a10_part_b import bos_signal, fvg_fill_signal  # noqa: E402  (my own detectors)

THEIRS = os.path.join(os.path.dirname(al.HERE), "price_action", "out")
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def breach(h, l, c, lev_hi, lev_lo, x_bp):
    c_prev = np.concatenate([[np.nan], c[:-1]])
    with np.errstate(invalid="ignore"):
        return (h >= lev_hi * (1.0 + x_bp * 1e-4)) & (c_prev <= lev_hi), (l <= lev_lo * (1.0 - x_bp * 1e-4)) & (c_prev >= lev_lo)


def sweep_levels(h, l, c, lev_hi, lev_lo, x_bp, k, ok_hi=None, ok_lo=None):
    """Sweep and rejection of arbitrary levels (forward scatter from the breach bar)."""
    n = len(c)
    b_hi, b_lo = breach(h, l, c, lev_hi, lev_lo, x_bp)
    if ok_hi is not None:
        b_hi &= ok_hi
        b_lo &= ok_lo
    short_evt, long_evt = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    for is_high, br, level, target in ((True, b_hi, lev_hi, short_evt), (False, b_lo, lev_lo, long_evt)):
        s = np.flatnonzero(br)
        lev = level[s]
        waiting = np.ones(len(s), dtype=bool)
        for j in range(k):
            t = s + j
            inside = t < n
            cc = np.where(inside, c[np.minimum(t, n - 1)], np.nan)
            with np.errstate(invalid="ignore"):
                back = (cc < lev) if is_high else (cc > lev)
                beyond = (cc >= lev) if is_high else (cc <= lev)
            hit = waiting & inside & back
            target[t[hit]] = True
            waiting = waiting & beyond & inside
    return long_evt.astype(np.int8) - short_evt.astype(np.int8)


def hold_levels(h, l, c, lev_hi, lev_lo, x_bp, k):
    """Breach at s and bars s .. s+k-1 all close beyond the level: +1 (above the high) / -1 at bar s+k-1."""
    n = len(c)
    b_hi, b_lo = breach(h, l, c, lev_hi, lev_lo, x_bp)
    up, dn = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    for is_high, br, level, target in ((True, b_hi, lev_hi, up), (False, b_lo, lev_lo, dn)):
        s = np.flatnonzero(br)
        s = s[s + k - 1 < n]
        lev = level[s]
        good = np.ones(len(s), dtype=bool)
        for j in range(k):
            with np.errstate(invalid="ignore"):
                good &= (c[s + j] > lev) if is_high else (c[s + j] < lev)
        target[s[good] + k - 1] = True
    return up.astype(np.int8) - dn.astype(np.int8)


def previous_day_levels(index: pd.DatetimeIndex, h, l, present):
    day = index.floor("D")
    frame = pd.DataFrame({"h": h, "l": l, "p": present}, index=index)
    by = frame.groupby(day).agg(h=("h", "max"), l=("l", "min"), n=("p", "sum"))
    by.loc[by["n"] != 1440, ["h", "l"]] = np.nan
    prev = by[["h", "l"]].copy()
    prev.index = prev.index + pd.Timedelta(days=1)          # yesterday's levels, valid today
    lev = prev.reindex(day)
    return lev["h"].to_numpy(), lev["l"].to_numpy()


def run(symbol: str) -> pd.DataFrame:
    g, _ = al.load_grid("um", symbol, "design")
    o, h, l, c, present = al.arrays(g)
    ctx = al.context_mask(present)
    key = al.baseline_key(g.index, al.realised_vol(c), None)
    fwd = {hh: al.forward_return(o, present, hh) for hh in al.HS}
    base = {hh: al.cell_mean(fwd[hh], key) for hh in al.HS}
    sig = {}
    for n in (240, 30, 60):
        hi, lo = al.prior_extreme(h, n, "max"), al.prior_extreme(l, n, "min")
        if n == 240:
            sig["hold_N240_x10_k3"] = hold_levels(h, l, c, hi, lo, 10.0, 3)
        if n == 30:
            sig["hold_N30_x2_k3"] = hold_levels(h, l, c, hi, lo, 2.0, 3)
        if n == 60:
            with np.errstate(invalid="ignore"):
                old_hi = al.prior_extreme(h, 10, "max") < hi
                old_lo = al.prior_extreme(l, 10, "min") > lo
            sig["sweepmat_N60_x2_k1"] = sweep_levels(h, l, c, hi, lo, 2.0, 1, old_hi, old_lo)
    pd_hi, pd_lo = previous_day_levels(g.index, h, l, present)
    sig["pdsweep_x2_k3"] = sweep_levels(h, l, c, pd_hi, pd_lo, 2.0, 3)
    sig["bos_s5_all"] = bos_signal(h, l, c, 5)
    # fair value gap (continuation)
    ref = pd.Series(h - l).rolling(60, min_periods=60).mean().shift(3).to_numpy()
    h2, l2 = np.concatenate([[np.nan, np.nan], h[:-2]]), np.concatenate([[np.nan, np.nan], l[:-2]])
    with np.errstate(invalid="ignore"):
        up = (l > h2) & ((l - h2) >= 0.5 * ref) & (ref > 0)
        dn = (h < l2) & ((l2 - h) >= 0.5 * ref) & (ref > 0)
    sig["fvg_m0p5"] = up.astype(np.int8) - dn.astype(np.int8)
    sig["fvgfill_m0p5_W60"] = fvg_fill_signal(h, l, c, 0.5, 60)
    # displacement
    body = c - o
    ref_body = pd.Series(np.abs(body)).rolling(60, min_periods=60).mean().shift(1).to_numpy()
    with np.errstate(invalid="ignore"):
        d_up = (body >= 3.0 * ref_body) & (ref_body > 0) & ((c - l) >= 0.75 * (h - l))
        d_dn = (-body >= 3.0 * ref_body) & (ref_body > 0) & ((h - c) >= 0.75 * (h - l))
    sig["disp_m3"] = d_up.astype(np.int8) - d_dn.astype(np.int8)
    frames = []
    day_all = ((g.index - EPOCH) // pd.Timedelta(days=1)).to_numpy()
    for def_id, s in sig.items():
        t = np.flatnonzero(s)
        t = t[ctx[t]]
        d = s[t].astype(float)
        cols = {"def_id": def_id, "symbol": symbol, "day": day_all[t]}
        for hh in al.HS:
            cols[f"g{hh}"] = d * fwd[hh][t]
            cols[f"b{hh}"] = d * base[hh][t]
        frames.append(pd.DataFrame(cols))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    with ProcessPoolExecutor(max_workers=3) as pool:
        ev = pd.concat(list(pool.map(run, al.PERPS)), ignore_index=True)
    ref = pd.read_csv(os.path.join(THEIRS, "table_design.csv")).set_index(["def_id", "h"])
    rows = []
    for def_id, part in ev.groupby("def_id", sort=False):
        for hh in al.HS:
            p = part[np.isfinite(part[f"g{hh}"]) & np.isfinite(part[f"b{hh}"])]
            adj = hh >= 240
            g = al.cluster_stats(p[f"g{hh}"].to_numpy(), p["day"].to_numpy(), adjacent=adj)
            e = al.cluster_stats((p[f"g{hh}"] - p[f"b{hh}"]).to_numpy(), p["day"].to_numpy(), adjacent=adj)
            r = ref.loc[(def_id, hh)]
            rows.append(dict(def_id=def_id, h=hh, events=g["n"], r_events=int(r["events"]), gross=g["mean"], r_gross=r["gross"], excess=e["mean"],
                             r_excess=r["excess"], t_excess=e["t"], r_t_excess=r["t_excess"]))
    t = pd.DataFrame(rows)
    t["d_events"] = t["events"] - t["r_events"]
    t["d_gross"] = (t["gross"] - t["r_gross"]).abs()
    t["d_t"] = (t["t_excess"] - t["r_t_excess"]).abs()
    t.to_csv(os.path.join(al.OUT, "nonselected_spotcheck.csv"), index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nrows:", len(t), " event-count mismatches:", int((t["d_events"] != 0).sum()), " max |gross diff|:", t["d_gross"].max(), " max |t diff|:", t["d_t"].max())


if __name__ == "__main__":
    main()
