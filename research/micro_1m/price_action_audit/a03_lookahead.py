"""Audit check 1: look-ahead by truncation, on REAL data, of the researcher's detectors (all 70 definitions)
and of my own re-implementation of the selected ones.

    python -I -B a03_lookahead.py          (-B: do not write byte-code into the researcher's directory)

For each symbol the detectors are run on the full design series and on the series cut after bar T-1.
Every event stamped at a bar < T must be identical (bar, direction, resting-order price). Cut points are
placed at random bars, at a UTC midnight, inside the 2022 data gap of SOL / the spot outages, and 1 bar
after an event of the selected definition (the most sensitive place for a detector that peeks one bar ahead).
A deliberately cheating detector (centred swing window) is run through the same harness to show it is caught.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
THEIR = os.path.join(os.path.dirname(HERE), "price_action")
sys.path.insert(0, HERE)
sys.path.insert(1, THEIR)

import audit_lib as al  # noqa: E402
import eventstudy as es  # noqa: E402  (researcher's module: the object under test)
import structures as st  # noqa: E402  (researcher's module: the object under test)

SYMBOLS = [("um", "BTCUSDT"), ("um", "SOLUSDT"), ("spot", "ETHUSDT")]


def snapshot(bars) -> dict:
    out = {}
    for ev in st.detect_all(bars):
        px = ev.entry_px if ev.entry_px is not None else np.zeros(len(ev.idx))
        out[ev.def_id] = (np.asarray(ev.idx, dtype=np.int64), np.asarray(ev.direction, dtype=np.int64), np.asarray(px, dtype=float))
    return out


def restrict(events, stop: int):
    idx, d, px = events
    keep = idx < stop
    rows = np.stack([idx[keep].astype(float), d[keep].astype(float), px[keep]], axis=1)
    return rows[np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))]


def cheat_swing_high(h: np.ndarray, size: int) -> np.ndarray:
    """A detector WITH look-ahead (centred window): flags the swing-high bar itself."""
    w = pd.Series(h).rolling(2 * size + 1, center=True, min_periods=2 * size + 1).max().to_numpy()
    return np.flatnonzero(h >= w)


def run(args) -> dict:
    market, symbol = args
    t0 = time.time()
    bars = es.load_bars(market, symbol, end="2024-01-01")
    n = bars.n
    full = snapshot(bars)
    rng = np.random.default_rng(sum(map(ord, market + symbol)))
    cuts = set(int(x) for x in rng.integers(n // 10, n - 10, size=3))
    midnight = np.flatnonzero((bars.minute % 1440 == 0))
    cuts.add(int(midnight[len(midnight) // 2]))
    missing = np.flatnonzero(~bars.valid)
    if len(missing):
        cuts.add(int(missing[len(missing) // 2]))       # inside a data gap
        cuts.add(int(missing[0]) + 1)                    # one bar into the first gap
    sel = full["sweep_N240_x10_k1"][0]
    cuts.add(int(sel[len(sel) // 3]) + 1)               # the event bar is the last bar of the cut series
    cuts.add(int(sel[2 * len(sel) // 3]) + 2)
    results, failures = [], []
    ctx_full, rv_full = es.context_ok(bars), es.realised_vol(bars)
    fwd_full = es.forward_returns(bars)
    for stop in sorted(cuts):
        cut = bars.cut(stop)
        part = snapshot(cut)
        n_events = 0
        for def_id, events in full.items():
            a, b = restrict(events, stop), restrict(part[def_id], stop)
            n_events += len(a)
            if a.shape != b.shape or not np.array_equal(a, b):
                failures.append({"symbol": symbol, "market": market, "cut": stop, "def_id": def_id, "full": int(len(a)), "cut_n": int(len(b))})
        # helper series used for filtering / the baseline key
        same_ctx = bool(np.array_equal(ctx_full[:stop], es.context_ok(cut)))
        same_rv = bool(np.array_equal(rv_full[:stop], es.realised_vol(cut), equal_nan=True))
        # forward returns are forward-looking BY DESIGN; they must agree wherever the cut series can know them
        fwd_cut = es.forward_returns(cut)
        fwd_ok = True
        for h in es.HORIZONS:
            known = max(stop - 1 - h, 0)
            fwd_ok &= bool(np.array_equal(fwd_full[h][:known], fwd_cut[h][:known], equal_nan=True))
            fwd_ok &= bool(np.isnan(fwd_cut[h][known:]).all())
        # the cheat must be caught by the same comparison
        ch_full, ch_cut = cheat_swing_high(bars.h, 5), cheat_swing_high(cut.h, 5)
        cheat_caught = not np.array_equal(ch_full[ch_full < stop], ch_cut)
        results.append({"market": market, "symbol": symbol, "cut": stop, "cut_time": str(bars.index[stop - 1]), "definitions": len(full),
                        "events_before_cut": int(n_events), "context_same": same_ctx, "rv_same": same_rv, "fwd_consistent": fwd_ok,
                        "cheat_caught": bool(cheat_caught)})
    # my own detectors, same harness
    g, _ = al.load_grid(market, symbol, "design")
    o, h, l, c, present = al.arrays(g)
    mine_fail = 0
    for (nn, x, k) in ((240, 10.0, 1), (30, 2.0, 3)):
        sig_full = al.sweep_signal(h, l, c, nn, x, k)
        for stop in sorted(cuts):
            sig_cut = al.sweep_signal(h[:stop], l[:stop], c[:stop], nn, x, k)
            mine_fail += int(not np.array_equal(sig_full[:stop], sig_cut))
        # and my detector must give the researcher's events (before the context filter)
        theirs = full[f"sweep_N{nn}_x{int(x)}_k{k}"]
        dense = np.zeros(len(c), dtype=np.int8)
        dense[theirs[0]] = theirs[1]
        mine_fail += int(not np.array_equal(dense, sig_full))
    return {"market": market, "symbol": symbol, "bars": int(n), "cuts": results, "failures": failures, "audit_impl_failures": mine_fail,
            "seconds": round(time.time() - t0, 1)}


def main() -> None:
    with ProcessPoolExecutor(max_workers=3) as pool:
        out = list(pool.map(run, SYMBOLS))
    with open(os.path.join(al.OUT, "lookahead_truncation.json"), "w") as handle:
        json.dump(out, handle, indent=1)
    for r in out:
        print(f"{r['market']} {r['symbol']}: bars {r['bars']}, {len(r['cuts'])} cut points, failures {len(r['failures'])}, "
              f"audit implementation failures {r['audit_impl_failures']}, {r['seconds']}s")
        for cpt in r["cuts"]:
            print("   ", cpt)
        for f in r["failures"][:20]:
            print("    FAIL", f)
    total_checks = sum(len(r["cuts"]) * r["cuts"][0]["definitions"] for r in out)
    print("definition x cut comparisons:", total_checks, " failures:", sum(len(r["failures"]) for r in out))


if __name__ == "__main__":
    main()
