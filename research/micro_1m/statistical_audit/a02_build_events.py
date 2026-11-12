"""Audit step 2/3: rebuild the events of the three selected definitions with the auditor's own code.

    python -I a02_build_events.py design|validation [um|spot] [SYMBOL ...]

Writes out/events_<segment>_<market>.parquet with one row per event:
    symbol, time (open time of the event bar t), name, side (+1 buy / -1 sell = the TRADE side of the selection),
    f<h>      open[t+1] -> open[t+1+h]            for h in 1, 5, 15, 60
    b<h>      matched baseline of f<h> (same coin x year x hour x volatility state; funding: without the hour)
    u<h>      unconditional mean of f<h> over all bars of the coin in the segment
    lag2_<h>  open[t+2] -> open[t+2+h]            (fill one minute later)
    lag3_<h>  open[t+3] -> open[t+3+h]
    vwap_<h>  vwap[t+1] -> vwap[t+1+h]
    worst_<h> high[t+1] (buy) or low[t+1] (sell) -> open[t+1+h]
    cc_<h>    close[t] -> open[t+1+h] (the forbidden fill at the event bar's own close; diagnostic only)
    nohalt_<h> f<h> without the halt rule (only missing bars invalidate)
    c0, o1, h1, l1, c1   close of the event bar; open/high/low/close of the entry bar
    r5, z, rate  the triggering quantities
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import auditlib as al  # noqa: E402

OUT = HERE / "out"
HS = (1, 5, 15, 60)


def events_for(market: str, symbol: str, segment: str) -> pd.DataFrame:
    b = al.Bars(market, symbol, segment)
    falls, rises = al.flag_move5_abs2pct(b)
    dips, pops = al.flag_move1_z3(b)
    r5, z = al.ret_k(b, 5), al.z1(b)
    # the mirrors (rises) are NOT evaluated: the selection carries only the two falls and the funding pair
    defs = [("move5_abs2pct_dn", falls, np.ones(b.n), True), ("move1_z3_dn", dips, np.ones(b.n), True)]
    rate = np.full(b.n, np.nan)
    if market == "um":
        funding = al.funding_series(symbol, segment)
        mask, side = al.flag_fund_post5_big(b, funding)
        defs.append(("fund_post5_big", mask, side, False))
        stamps = funding.index.as_unit("ns").asi8
        k = np.searchsorted(stamps, b.ns, side="right") - 1
        rate = np.where(k >= 0, funding.to_numpy(dtype=float)[np.maximum(k, 0)], np.nan)
    f = {h: b.fwd(h) for h in HS}
    base_hour = {h: al.matched_baseline(b, f[h], True) for h in HS}
    base_nohour = {h: al.matched_baseline(b, f[h], False) for h in HS}
    uncond = {h: float(np.nanmean(f[h][b.inside])) for h in HS}
    lag2 = {h: b.fwd(h, lag=2) for h in HS}
    lag3 = {h: b.fwd(h, lag=3) for h in HS}
    vwap = {h: b.fwd(h, entry="vwap", exit_="vwap") for h in HS}
    hi = {h: b.fwd(h, entry="high") for h in HS}
    lo = {h: b.fwd(h, entry="low") for h in HS}
    nohalt = {h: b.fwd(h, halt_rule=False) for h in HS}
    nxt = lambda a: np.concatenate([a[1:], [np.nan]])  # value of bar t+1 (an OUTCOME column, never a signal)
    frames = []
    for name, mask, side, by_hour in defs:
        rows = np.flatnonzero(mask & b.inside)
        if len(rows) == 0:
            continue
        s = side[rows]
        data = {"symbol": symbol, "market": market, "name": name, "time": b.index[rows], "side": s,
                "c0": b.close[rows], "o1": nxt(b.open)[rows], "h1": nxt(b.high)[rows], "l1": nxt(b.low)[rows], "c1": nxt(b.close)[rows],
                "qv0": b.qv[rows], "r5": r5[rows], "z": z[rows], "rate": rate[rows]}
        for h in HS:
            data[f"f{h}"] = f[h][rows]
            data[f"b{h}"] = (base_hour if by_hour else base_nohour)[h][rows]
            data[f"u{h}"] = uncond[h]
            data[f"lag2_{h}"] = lag2[h][rows]
            data[f"lag3_{h}"] = lag3[h][rows]
            data[f"vwap_{h}"] = vwap[h][rows]
            data[f"worst_{h}"] = np.where(s > 0, hi[h][rows], lo[h][rows])
            data[f"nohalt_{h}"] = nohalt[h][rows]
            with np.errstate(invalid="ignore", divide="ignore"):
                # diagnostic: what the number would be if the trade were (wrongly) entered at the event bar's own close
                cc = np.full(b.n, np.nan)
                cc[: b.n - h - 1] = b.open[h + 1:] / b.close[: b.n - h - 1] - 1.0  # close[t] -> open[t+1+h]
            data[f"cc_{h}"] = cc[rows]
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)


def main(argv):
    segment = argv[0]
    market = argv[1] if len(argv) > 1 else "um"
    symbols = argv[2:] or al.PERPS
    OUT.mkdir(exist_ok=True)
    tag = "" if symbols == al.PERPS else "_" + "_".join(s[:-4] for s in symbols)
    frames = []
    for symbol in symbols:
        t0 = time.time()
        e = events_for(market, symbol, segment)
        frames.append(e)
        print(segment, market, symbol, e.groupby("name").size().to_dict(), f"{time.time() - t0:.0f}s", flush=True)
    pd.concat(frames, ignore_index=True).to_parquet(OUT / f"events_{segment}_{market}{tag}.parquet")


if __name__ == "__main__":
    main(sys.argv[1:])
