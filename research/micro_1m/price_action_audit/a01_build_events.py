"""Audit step 1: build per-event tables for the three selected (definition, horizon) pairs with my own code.

    python -I a01_build_events.py design um
    python -I a01_build_events.py validation um
    python -I a01_build_events.py design spot      (all six coins on spot; BTC/ETH = researcher's cross-check, the rest unused by them)
    python -I a01_build_events.py validation spot

Output: out/events_<segment>_<market>.parquet, one row per event:
  def_id, symbol, ts (open time of the event bar), dir (+1 long / -1 short),
  g<h>  = dir * forward return (open t+1 -> open t+1+h), bp
  b<h>  = dir * matched baseline for that bar
  d<h>  = the same forward return with the fill delayed by one more minute (open t+2 -> open t+2+h)
  mk_*  = resting-order simulation (limit at the event bar's close, 5 minutes, strict trade-through)
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script directory
import audit_lib as al  # noqa: E402

DEFS = {"sweep_N240_x10_k1": (240, 10.0, 1), "sweep_N30_x2_k3": (30, 2.0, 3)}
REST = 5


def build(args) -> pd.DataFrame:
    segment, market, symbol = args
    t0 = time.time()
    g, first_event = al.load_grid(market, symbol, segment)
    o, h, l, c, present = al.arrays(g)
    n = len(o)
    index = g.index
    ctx = al.context_mask(present)
    if first_event is not None:
        ctx &= np.asarray(index >= first_event)
    rv = al.realised_vol(c)
    key = al.baseline_key(index, rv, first_event)
    fwd = {hh: al.forward_return(o, present, hh, delay=1) for hh in al.HS}
    fwd2 = {hh: al.forward_return(o, present, hh, delay=2) for hh in al.HS}
    base = {hh: al.cell_mean(fwd[hh], key) for hh in al.HS}
    cs = np.concatenate([[0], np.cumsum(present)])
    frames = []
    for def_id, (nn, x_bp, k) in DEFS.items():
        sig = al.sweep_signal(h, l, c, nn, x_bp, k)
        t = np.flatnonzero(sig)
        n_raw = len(t)
        t = t[ctx[t]]
        d = sig[t].astype(float)
        cols = {"def_id": def_id, "symbol": symbol, "market": market, "ts": index[t], "dir": d.astype(np.int8),
                "rv": rv[t], "close_t": c[t]}
        nxt = np.minimum(t + 1, n - 1)
        cols["open_next"] = np.where(t + 1 < n, o[nxt], np.nan)
        for hh in al.HS:
            cols[f"g{hh}"] = d * fwd[hh][t]
            cols[f"b{hh}"] = d * base[hh][t]
            cols[f"d{hh}"] = d * fwd2[hh][t]
        # resting order at the close of the event bar, strict trade-through within REST bars
        limit = c[t]
        fill = np.full(len(t), -1, dtype=np.int64)
        for j in range(1, REST + 1):
            u = t + j
            inside = u < n
            uu = np.minimum(u, n - 1)
            with np.errstate(invalid="ignore"):
                through = np.where(d > 0, l[uu] < limit, h[uu] > limit) & inside
            new = (fill < 0) & through
            fill[new] = u[new]
        got = fill >= 0
        f = np.where(got, fill, 0)
        cols["mk_off"] = np.where(got, fill - t, 0).astype(np.int8)
        cols["mk_ctx"] = got & ctx[f]
        cols["mk_ts"] = index[f]
        for hh in (0,) + al.HS:
            j = f + 1 + hh
            okj = got & (j < n)
            jj = np.where(okj, j, 0)
            complete = okj & ((cs[np.minimum(jj + 1, n)] - cs[f]) == (jj - f + 1))
            cols[f"mk_g{hh}"] = np.where(complete, d * (o[jj] / limit - 1.0) * 1e4, np.nan)
        frame = pd.DataFrame(cols)
        frame.attrs = {}
        frames.append(frame)
        print(f"  {segment} {market} {symbol} {def_id}: raw events {n_raw}, with context {len(t)}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    print(f"{segment} {market} {symbol}: bars {int(present.sum())} grid {n} missing {int((~present).sum())} "
          f"usable {int(ctx.sum())} first {index[0]} last {index[-1]}  {time.time() - t0:.1f}s", flush=True)
    return out


def main() -> None:
    segment, market = sys.argv[1], sys.argv[2]
    symbols = sys.argv[3].split(",") if len(sys.argv) > 3 else al.PERPS
    tag = sys.argv[4] if len(sys.argv) > 4 else ""
    os.makedirs(al.OUT, exist_ok=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        parts = list(pool.map(build, [(segment, market, s) for s in symbols]))
    table = pd.concat(parts, ignore_index=True)
    path = os.path.join(al.OUT, f"events_{segment}_{market}{tag}.parquet")
    table.to_parquet(path, index=False)
    print("wrote", path, len(table))


if __name__ == "__main__":
    main()
