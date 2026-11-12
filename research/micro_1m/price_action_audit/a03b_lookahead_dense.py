"""Audit check 1, dense variant: cut the real series at EVERY bar of short stretches.

    python -I -B a03b_lookahead_dense.py

A sparse set of cut points can miss a detector that peeks only one or two bars ahead, because only events
within that distance of the cut are affected. Here, for windows of real data of about four days, the series
is cut after every one of 300 consecutive bars, so every event in the stretch is at some point the last bar,
the second-to-last bar, ... of a truncated series. All 70 definitions must give identical events.
Stretches are centred on events of the rarest definitions so that every definition is exercised.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
THEIR = os.path.join(os.path.dirname(HERE), "price_action")
sys.path.insert(0, HERE)
sys.path.insert(1, THEIR)

import audit_lib as al  # noqa: E402
import eventstudy as es  # noqa: E402  (researcher's module: the object under test)
import structures as st  # noqa: E402

SYMBOLS = [("um", "BTCUSDT"), ("um", "SOLUSDT"), ("um", "XRPUSDT"), ("spot", "ETHUSDT")]
HALF = 150
RARE = ["pdsweep_x10_k1", "sweep_N240_x10_k1", "squeeze_M120_q0p1_v2", "disp_m6"]


def snapshot(bars) -> dict:
    out = {}
    for ev in st.detect_all(bars):
        px = ev.entry_px if ev.entry_px is not None else np.zeros(len(ev.idx))
        rows = np.stack([np.asarray(ev.idx, dtype=float), np.asarray(ev.direction, dtype=float), np.asarray(px, dtype=float)], axis=1)
        out[ev.def_id] = rows[np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))]
    return out


def sub(bars, a: int, b: int):
    return es.Bars(bars.symbol, bars.market, bars.minute[a:b], bars.o[a:b], bars.h[a:b], bars.l[a:b], bars.c[a:b], bars.v[a:b], bars.valid[a:b])


def run(args) -> dict:
    market, symbol = args
    t0 = time.time()
    bars = es.load_bars(market, symbol, end="2024-01-01")
    rng = np.random.default_rng(7 + sum(map(ord, market + symbol)))
    centres = []
    for def_id in RARE:
        ev = st.detect(bars, [def_id])[def_id]
        idx = ev.idx[(ev.idx > 6000) & (ev.idx < bars.n - 2000)]
        centres.append(int(rng.choice(idx)))
    fails, tested, per_def = [], 0, {}
    for p in centres:
        day_start = p - int(bars.minute[p] % 1440)          # UTC midnight of the event's day
        a = day_start - 3 * 1440                             # three full days of history before it
        b = p + HALF + 1
        window = sub(bars, a, b)
        ref = snapshot(window)
        for stop in range(p - HALF - a, b - a + 1):
            part = snapshot(window.cut(stop))
            for def_id, rows in ref.items():
                want = rows[rows[:, 0] < stop]
                got = part[def_id]
                tested += 1
                if want.shape != got.shape or not np.array_equal(want, got):
                    fails.append({"market": market, "symbol": symbol, "centre": p, "cut": int(stop + a), "def_id": def_id})
        lo = p - HALF - a
        for def_id, rows in ref.items():
            per_def[def_id] = per_def.get(def_id, 0) + int((rows[:, 0] >= lo).sum())
    return {"market": market, "symbol": symbol, "stretches": [str(bars.index[p]) for p in centres], "comparisons": tested,
            "failures": fails[:50], "n_failures": len(fails), "events_in_stretches": per_def, "seconds": round(time.time() - t0, 1)}


def main() -> None:
    with ProcessPoolExecutor(max_workers=4) as pool:
        out = list(pool.map(run, SYMBOLS))
    with open(os.path.join(al.OUT, "lookahead_dense.json"), "w") as handle:
        json.dump(out, handle, indent=1)
    total = {}
    for r in out:
        print(f"{r['market']} {r['symbol']}: stretches centred at {r['stretches']}; comparisons {r['comparisons']}, failures {r['n_failures']}, {r['seconds']}s")
        for f in r["failures"][:10]:
            print("   FAIL", f)
        for k, v in r["events_in_stretches"].items():
            total[k] = total.get(k, 0) + v
    print("events inside the densely cut stretches, per definition (all symbols): min", min(total.values()), "max", max(total.values()))
    print("definitions with fewer than 5 events in the stretches:", {k: v for k, v in total.items() if v < 5})
    print("total comparisons:", sum(r["comparisons"] for r in out), " failures:", sum(r["n_failures"] for r in out))


if __name__ == "__main__":
    main()
