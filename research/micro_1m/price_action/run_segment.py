"""Detect every definition on one segment and store per-day sufficient statistics.

    python -I -B run_segment.py design                 # all 70 definitions, bars before 2024-01-01
    python -I -B run_segment.py validation             # ONLY the (definition, horizon) pairs in selection.json
    python -I -B run_segment.py dryrun                 # the validation code path on 2022-07 .. 2023-12 (design data)

The validation run refuses to start unless selection.json exists, and it
computes nothing but the selected definitions at their selected horizons. The hold-out (from 2025-07-01)
is never loaded: the loader stops there and this script never sets the flag
that would lift the lock.

Output: out/daily_<segment>_<market>_<symbol>.parquet with one row per
(definition, day, direction) and, per horizon h, n / sg / sb / sw / sgg / see
(see eventstudy.aggregate_daily).
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import eventstudy as es  # noqa: E402
import structures as st  # noqa: E402

OUT = os.path.join(HERE, "out")
PERPS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]
SPOT = ["BTCUSDT", "ETHUSDT"]
DESIGN_END = "2024-01-01"
VALIDATION_START = "2024-01-01"
VALIDATION_WARMUP = "2023-12-25"  # history for the look-back windows only; events before 2024-01-01 are discarded
HORIZONS0 = (0,) + es.HORIZONS


def segment_window(segment: str):
    """(load from, load until, first admissible event time) for a segment.

    "dryrun" runs the VALIDATION code path (warm-up, selected pairs only) on the
    last 18 months of the DESIGN segment. It exists so that the validation
    scripts can be debugged without reading a single validation bar.
    """
    if segment == "design":
        return None, DESIGN_END, None
    if segment == "validation":
        return VALIDATION_WARMUP, None, VALIDATION_START  # the loader itself stops at the hold-out
    if segment == "dryrun":
        return "2022-06-24", DESIGN_END, "2022-07-01"
    raise ValueError(segment)


def load_segment_bars(segment: str, market: str, symbol: str):
    """Bars of a segment and the first minute at which an event may be stamped (None = no restriction)."""
    start, end, first = segment_window(segment)
    bars = es.load_bars(market, symbol, start=start, end=end)
    first_minute = None if first is None else int(pd.Timestamp(first, tz="UTC").timestamp() // 60)
    return bars, first_minute


def control_events(bars: es.Bars, ok: np.ndarray, seed: int):
    """Machinery checks on real data, not hypotheses: random bars with random direction, and random bars long."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero((rng.random(bars.n) < 0.005) & ok)
    yield st.EventSet("ctl_random", "control", {}, "random bars, random direction", idx,
                      rng.choice(np.array([-1, 1], dtype=np.int8), size=len(idx)))
    idx2 = np.flatnonzero((rng.random(bars.n) < 0.005) & ok)
    yield st.EventSet("ctl_long", "control", {}, "random bars, always long", idx2, np.ones(len(idx2), dtype=np.int8))


def run_symbol(args) -> dict:
    segment, market, symbol, wanted = args
    t0 = time.time()
    bars, first_minute = load_segment_bars(segment, market, symbol)
    fwd = es.forward_returns(bars)
    cell, n_cells = es.baseline_cells(bars)
    if first_minute is not None:
        cell = np.where(bars.minute >= first_minute, cell, -1)  # baseline from validation bars only
    base = es.matched_baseline(fwd, cell, n_cells)
    base0 = dict(base)
    base0[0] = np.zeros(bars.n)
    ok = es.context_ok(bars)
    if first_minute is not None:
        ok &= bars.minute >= first_minute
    frames, meta = [], {}
    sets = st.detect(bars, list(wanted)).values() if wanted is not None else st.detect_all(bars)
    seed = sum(ord(ch) for ch in market + symbol)  # deterministic per market and symbol
    for ev in list(sets) + (list(control_events(bars, ok, seed)) if wanted is None else []):
        if wanted is not None:
            # validation: nothing but the pre-registered horizons of this definition is computed
            daily = es.event_daily(bars, ev.idx, ev.direction, fwd, base0, ok, ev.entry_px, horizons=tuple(wanted[ev.def_id]))
        elif ev.entry_px is not None:
            daily = es.event_daily(bars, ev.idx, ev.direction, fwd, base0, ok, ev.entry_px, horizons=HORIZONS0)
        else:
            daily = es.event_daily(bars, ev.idx, ev.direction, fwd, base, ok)
        daily.insert(0, "def_id", ev.def_id)
        frames.append(daily)
        meta[ev.def_id] = {"family": ev.family, "params": ev.params, "hypothesis": ev.hypothesis,
                           "fill": "resting limit order at the zone edge" if ev.entry_px is not None else "next bar open"}
    table = pd.concat(frames, ignore_index=True)
    table.insert(1, "symbol", symbol)
    table.insert(1, "market", market)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"daily_{segment}_{market}_{symbol}.parquet")
    table.to_parquet(path, index=False)
    index = bars.index
    return {"segment": segment, "market": market, "symbol": symbol, "bars": int(bars.valid.sum()), "grid": int(bars.n),
            "missing": int((~bars.valid).sum()), "first": str(index[0]), "last": str(index[-1]),
            "usable_bars": int(ok.sum()), "rows": int(len(table)), "seconds": round(time.time() - t0, 1), "meta": meta}


def main() -> None:
    segment = sys.argv[1]
    assert segment in ("design", "validation", "dryrun")
    wanted = None
    if segment != "design":
        with open(os.path.join(HERE, "selection.json")) as handle:
            selection = json.load(handle)
        wanted = {}
        for item in selection["selected"]:
            wanted.setdefault(item["def_id"], []).append(int(item["h"]))
        print(f"{segment} run for the pre-registered (definition, horizons) only:", wanted)
    jobs = [(segment, "um", s, wanted) for s in PERPS] + [(segment, "spot", s, wanted) for s in SPOT]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_symbol, jobs))
    meta = results[0]["meta"]
    for r in results:
        r.pop("meta")
        print(r)
    with open(os.path.join(OUT, f"run_{segment}.json"), "w") as handle:
        json.dump({"runs": results, "definitions": meta}, handle, indent=1)


if __name__ == "__main__":
    main()
