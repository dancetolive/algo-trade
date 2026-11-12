"""Audit check 2: fill timing, tested literally by TIMESTAMP on the raw frames.

    python -I -B a04_fill_timing.py

(a) The researcher's forward_returns()[h][t] must equal open(ts + 1 min + h) / open(ts + 1 min) - 1 looked up by
    timestamp in the raw frame, and must be NaN exactly when a minute in [ts, ts + 1 + h] is missing.
(b) The gross returns behind the reproduced headline (my event table, which matches the researcher's sums to
    1e-12) are recomputed by timestamp lookup; and the alternative "entry at the event bar's own close" is
    computed to show the headline is NOT that number.
(c) Literal check of the selected definition on a sample of events and of non-events, from raw slices.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
THEIR = os.path.join(os.path.dirname(HERE), "price_action")
sys.path.insert(0, HERE)
sys.path.insert(1, THEIR)

import audit_lib as al  # noqa: E402
import eventstudy as es  # noqa: E402  (researcher's module: object under test)
from algotrade.data import store  # noqa: E402

MIN = pd.Timedelta(minutes=1)


def part_a(market: str, symbol: str, rng) -> dict:
    raw = store.load_klines(market, "1m", symbol, end="2024-01-01", columns=["open", "high", "low", "close", "volume"])
    bars = es.bars_from_frame(raw, symbol, market)
    fwd = es.forward_returns(bars)
    index = bars.index
    present = pd.Series(True, index=raw.index)
    t = rng.integers(0, bars.n, size=20000)
    gaps = np.flatnonzero(~bars.valid)
    if len(gaps):  # every bar within 1500 minutes before each gap start and 5 after each gap end
        starts = gaps[np.r_[True, np.diff(gaps) > 1]]
        ends = gaps[np.r_[np.diff(gaps) > 1, True]]
        near = np.concatenate([np.arange(max(s - 1500, 0), s) for s in starts] + [np.arange(e + 1, min(e + 6, bars.n)) for e in ends])
        t = np.concatenate([t, near])
    ts = index[t]
    out = {"market": market, "symbol": symbol, "bars_checked": int(len(t)), "missing_minutes": int((~bars.valid).sum())}
    worst, nan_mismatch = 0.0, 0
    for h in es.HORIZONS:
        entry = raw["open"].reindex(ts + MIN).to_numpy()
        exit_ = raw["open"].reindex(ts + MIN * (1 + h)).to_numpy()
        literal = (exit_ / entry - 1.0) * 1e4
        # completeness of [ts, ts + 1 + h] by counting raw rows in the closed interval
        pos_lo = raw.index.searchsorted(ts, side="left")
        pos_hi = raw.index.searchsorted(ts + MIN * (1 + h), side="right")
        complete = (pos_hi - pos_lo) == (h + 2)
        literal = np.where(complete, literal, np.nan)
        theirs = fwd[h][t]
        nan_mismatch += int((np.isnan(literal) != np.isnan(theirs)).sum())
        both = np.isfinite(literal) & np.isfinite(theirs)
        worst = max(worst, float(np.abs(literal[both] - theirs[both]).max()))
        out[f"nan_h{h}"] = int(np.isnan(theirs).sum())
    out["max_abs_diff_bp"] = worst
    out["nan_pattern_mismatches"] = nan_mismatch
    return out


def part_b(symbol: str) -> dict:
    """Recompute the gross returns of the selected definition by timestamp lookup, design, perpetual."""
    raw = store.load_klines("um", "1m", symbol, end="2024-01-01", columns=["open", "close", "high", "low"])
    ev = pd.read_parquet(os.path.join(al.OUT, "events_design_um.parquet"))
    ev = ev[(ev["symbol"] == symbol) & (ev["def_id"] == "sweep_N240_x10_k1")].reset_index(drop=True)
    ts = pd.DatetimeIndex(ev["ts"])
    d = ev["dir"].to_numpy(dtype=float)
    res = {"symbol": symbol, "events": int(len(ev))}
    for h in (60, 240):
        entry_next_open = raw["open"].reindex(ts + MIN).to_numpy()
        exit_open = raw["open"].reindex(ts + MIN * (1 + h)).to_numpy()
        event_close = raw["close"].reindex(ts).to_numpy()
        lit = d * (exit_open / entry_next_open - 1.0) * 1e4
        have = np.isfinite(ev[f"g{h}"].to_numpy())
        res[f"h{h}_max_abs_diff_vs_table_bp"] = float(np.nanmax(np.abs(lit[have] - ev[f"g{h}"].to_numpy()[have])))
        res[f"h{h}_mean_next_open_entry"] = float(np.mean(lit[have]))
        res[f"h{h}_mean_if_entry_at_event_close"] = float(np.mean((d * (exit_open / event_close - 1.0) * 1e4)[have]))
        res[f"h{h}_mean_if_exit_at_close_of_last_bar"] = float(np.nanmean((d * (raw["close"].reindex(ts + MIN * h).to_numpy() / entry_next_open - 1.0) * 1e4)[have]))
    gap = d * (raw["open"].reindex(ts + MIN).to_numpy() / raw["close"].reindex(ts).to_numpy() - 1.0) * 1e4
    res["mean_signed_gap_close_to_next_open_bp"] = float(np.nanmean(gap))
    res["share_next_open_differs_from_close"] = float(np.nanmean(np.abs(gap) > 1e-9))
    return res


def part_c(symbol: str, rng) -> dict:
    """Literal check of sweep_N240_x10_k1 from raw slices: 400 events and 3000 random non-event bars."""
    raw = store.load_klines("um", "1m", symbol, end="2024-01-01", columns=["open", "high", "low", "close"])
    ev = pd.read_parquet(os.path.join(al.OUT, "events_design_um.parquet"))
    ev = ev[(ev["symbol"] == symbol) & (ev["def_id"] == "sweep_N240_x10_k1")]
    event_time = set(ev["ts"])

    def literal(ts) -> int:
        w = raw.loc[ts - 240 * MIN: ts - MIN]
        if len(w) != 240 or ts not in raw.index:
            return 0
        bar, prev_close = raw.loc[ts], w["close"].iloc[-1]
        ph, pl = w["high"].max(), w["low"].min()
        short = bar["high"] >= ph * (1 + 10e-4) and prev_close <= ph and bar["close"] < ph
        long_ = bar["low"] <= pl * (1 - 10e-4) and prev_close >= pl and bar["close"] > pl
        return int(long_) - int(short)

    sample = ev.sample(400, random_state=3)
    wrong_events = sum(int(literal(ts) != d) for ts, d in zip(sample["ts"], sample["dir"]))
    pool = raw.index[2000:]
    picks = pool[rng.integers(0, len(pool), size=3000)]
    false_neg = 0
    for ts in picks:
        lit = literal(ts)
        if (lit != 0) != (ts in event_time):
            # an event may legitimately be dropped by the 1,500-minute context rule; count only real disagreements
            ctx = len(raw.loc[ts - 1500 * MIN: ts]) == 1501
            false_neg += int(ctx)
    return {"symbol": symbol, "events_checked": 400, "event_disagreements": int(wrong_events), "random_bars_checked": 3000,
            "random_bar_disagreements": int(false_neg)}


def main() -> None:
    rng = np.random.default_rng(11)
    res = {"forward_returns_vs_timestamp_lookup": [part_a(m, s, rng) for m, s in (("um", "BTCUSDT"), ("um", "SOLUSDT"), ("spot", "BTCUSDT"))],
           "gross_by_timestamp_lookup": [part_b(s) for s in ("BTCUSDT", "XRPUSDT")],
           "literal_definition": [part_c(s, rng) for s in ("BTCUSDT", "SOLUSDT")]}
    with open(os.path.join(al.OUT, "fill_timing.json"), "w") as handle:
        json.dump(res, handle, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
