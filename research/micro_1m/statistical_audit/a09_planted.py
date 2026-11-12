"""Audit, mirror-image check: would the researcher's code have seen a real effect? Plant one in real bars.

    python -I a09_planted.py

The study calls every frequent structure "too small" and Part B "no improvement". Both are negative results, so
a bug that hides effects would be fatal. Real design bars are altered IN MEMORY (the loader is patched inside
this process; no file is touched) and pushed through the researcher's own functions:

Part A (run_events.load_segment -> definitions.generate -> eventstudy.aggregate / summarise), BTCUSDT and DOGEUSDT
perpetuals, design:
    plant T  +20 bp of drift spread over the 15 minutes after every 00:00 / 08:00 / 16:00 stamp. The events of
             fund_post5_all / fund_post15_all / fund_post60_all depend on the clock only, so the same events must
             show +6.67 / +20 / +20 bp more than on the untouched bars.
    plant P  +20 bp over the 15 minutes after every 3-sigma down minute (flags of the untouched bars). The flags
             are recomputed by the researcher's code on the altered bars, so the event set moves a little; the
             expected gain is the planted drift summed over the windows of the events actually flagged.
    A definition that got no plant (sign60 at 60 minutes) is shown as a control.
Part B (part_b.rule_changes / timing_table / summarise), BTC and ETH spot, design rule days:
    the open right after every 3-sigma dip is made 20 bp cheaper; the 4-hour dip rule must gain
    20 bp x (share of days on which a dip occurred) and the 24-hour rule about 20 bp.
Writes out/planted.json.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
THEIR = HERE.parent / "statistical"
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(THEIR))
sys.path.insert(2, str(HERE.parents[1] / "baseline"))

import auditlib as al  # noqa: E402
import definitions as defs  # noqa: E402  researcher's
import eventstudy as es  # noqa: E402  researcher's
import run_events as runner  # noqa: E402  researcher's
import part_b as pb  # noqa: E402  researcher's (never calls main(): nothing is written)

sys.dont_write_bytecode = True  # part_b switches it off again after its own import
from algotrade.data import store  # noqa: E402

OUT = HERE / "out"
PLANT = 20e-4
ORIGINAL_LOAD = store.load_klines
ONLY = {"fund_post5_all", "fund_post15_all", "fund_post60_all", "move1_z3_dn", "sign60"}


def with_drift(frame: pd.DataFrame, starts: np.ndarray, total: float = PLANT, bars: int = 15) -> tuple[pd.DataFrame, np.ndarray]:
    """Add ``total`` log return, spread evenly over the ``bars`` rows that follow each row in ``starts``."""
    n = len(frame)
    g = np.zeros(n)
    for k in range(1, bars + 1):
        idx = starts + k
        np.add.at(g, idx[idx < n], total / bars)
    before = np.concatenate([[0.0], np.cumsum(g)[:-1]])  # drift earned strictly before the bar: applies to its open
    after = np.cumsum(g)  # ... up to the end of the bar: applies to its close
    out = frame.copy()
    out["open"] = frame["open"].to_numpy() * np.exp(before)
    out["close"] = frame["close"].to_numpy() * np.exp(after)
    out["high"] = np.maximum.reduce([frame["high"].to_numpy() * np.exp(after), out["open"].to_numpy(), out["close"].to_numpy()])
    out["low"] = np.minimum.reduce([frame["low"].to_numpy() * np.exp(before), out["open"].to_numpy(), out["close"].to_numpy()])
    return out, g


def their_tables(market: str, symbol: str) -> dict:
    """The researcher's pipeline for one symbol, in memory: {definition: summary table by horizon} plus the masks."""
    grid, feats, fwd, inside, day0, n_days = runner.load_segment(market, symbol, "design")
    base = runner.baselines(grid, feats, fwd, inside)
    day_index = np.clip(grid.day - day0, 0, n_days - 1)
    wide = store.load_funding([symbol], end=store.DESIGN_END)
    funding = wide[symbol] if symbol in wide.columns else None
    out = {}
    for d in defs.generate(grid, feats, funding, only=ONLY):
        block = es.aggregate(d.mask & inside, d.direction, fwd, base[d.baseline], day_index, n_days)
        out[d.name] = (es.summarise(block), np.flatnonzero(d.mask & inside), grid.exists)
    return out


def part_a(symbol: str) -> dict:
    frame = ORIGINAL_LOAD("um", "1m", symbol, end=store.DESIGN_END)
    assert frame.index[-1] < store.DESIGN_END
    store.load_klines = ORIGINAL_LOAD
    plain = their_tables("um", symbol)
    minute_of_day = (frame.index.hour * 60 + frame.index.minute).to_numpy()
    res = {}

    # ---- plant T: after every funding stamp (clock only) -------------------------------------------------
    stamp_rows = np.flatnonzero(np.isin(minute_of_day, [1439, 479, 959]))  # the bar before the stamp = the event bar t; drift in t+1 .. t+15
    planted_frame, _ = with_drift(frame, stamp_rows)
    store.load_klines = lambda market, interval, sym, start=None, end=None, columns=None: planted_frame if (market, interval, sym) == ("um", "1m", symbol) else ORIGINAL_LOAD(market, interval, sym, start, end, columns)
    planted = their_tables("um", symbol)
    store.load_klines = ORIGINAL_LOAD
    for name, h, expect in (("fund_post5_all", 5, PLANT / 3), ("fund_post15_all", 15, PLANT), ("fund_post60_all", 60, PLANT), ("sign60", 60, None)):
        a, b = plain[name][0].loc[h], planted[name][0].loc[h]
        res[f"T/{name}@{h}"] = {"events": int(b.events), "gross_before_bp": float(a.gross_bp), "gross_after_bp": float(b.gross_bp), "gain_bp": float(b.gross_bp - a.gross_bp),
                                "expected_gain_bp": None if expect is None else (np.exp(expect) - 1) * 1e4, "t_day_before": float(a.t_day), "t_day_after": float(b.t_day)}

    # ---- plant P: after every 3-sigma down minute ------------------------------------------------------------
    pos_of_row = ((frame.index.as_unit("ns").asi8 - frame.index.as_unit("ns").asi8[0]) // al.MIN_NS)
    grid_to_row = np.full(int(pos_of_row[-1]) + 1, -1)
    grid_to_row[pos_of_row] = np.arange(len(frame))
    ev_rows = grid_to_row[plain["move1_z3_dn"][1]]
    ev_rows = ev_rows[ev_rows >= 0]
    planted_frame, g = with_drift(frame, ev_rows)
    store.load_klines = lambda market, interval, sym, start=None, end=None, columns=None: planted_frame if (market, interval, sym) == ("um", "1m", symbol) else ORIGINAL_LOAD(market, interval, sym, start, end, columns)
    planted = their_tables("um", symbol)
    store.load_klines = ORIGINAL_LOAD
    # expected: planted drift summed over the 15 bars after each event flagged on the ALTERED bars
    new_rows = grid_to_row[planted["move1_z3_dn"][1]]
    new_rows = new_rows[(new_rows >= 0) & (new_rows + 16 < len(frame))]
    cg = np.concatenate([[0.0], np.cumsum(g)])
    planted_in_window = cg[new_rows + 16] - cg[new_rows + 1]  # bars t+1 .. t+15
    a, b = plain["move1_z3_dn"][0].loc[15], planted["move1_z3_dn"][0].loc[15]
    # the study's hypothesised side for this definition is "continuation" (short after a fall): a planted rebound shows as a NEGATIVE gross there
    res["P/move1_z3_dn@15"] = {"events_before": int(a.events), "events_after": int(b.events), "gross_before_bp_hyp_side": float(a.gross_bp), "gross_after_bp_hyp_side": float(b.gross_bp),
                               "gain_on_long_side_bp": float(-(b.gross_bp - a.gross_bp)), "planted_drift_in_event_windows_bp": float(planted_in_window.mean()) * 1e4,
                               "t_day_before": float(a.t_day), "t_day_after": float(b.t_day)}
    a, b = plain["sign60"][0].loc[60], planted["sign60"][0].loc[60]
    res["P/control sign60@60"] = {"gross_before_bp": float(a.gross_bp), "gross_after_bp": float(b.gross_bp), "t_day_before": float(a.t_day), "t_day_after": float(b.t_day)}
    return res


def part_b() -> dict:
    names = ["move1_z3", "move5_abs2pct"]
    trades = pb.rule_changes("design")
    original_segment = runner.load_segment
    base = pb.summarise(pb.timing_table("spot", "design", names, trades, all_days=False))

    def planted_segment(market, symbol, segment):
        grid, feats, fwd, inside, day0, n_days = original_segment(market, symbol, segment)
        d = next(iter(defs.generate(grid, feats, None, only={"move1_z3_dn"})))
        idx = np.flatnonzero(d.mask)
        idx = idx[idx + 1 < len(grid)]
        grid.raw_open[idx + 1] *= (1.0 - PLANT)  # the fill right after a dip becomes 20 bp cheaper; flags (closes) are untouched
        return grid, feats, fwd, inside, day0, n_days

    runner.load_segment = planted_segment
    try:
        planted = pb.summarise(pb.timing_table("spot", "design", names, trades, all_days=False))
    finally:
        runner.load_segment = original_segment
    key = ["side", "structure", "window_min"]
    m = base.merge(planted, on=key, suffixes=("", "_planted"))
    m["gain_bp"] = m.mean_bp_planted - m.mean_bp
    m["expected_gain_bp"] = np.where((m.structure == "move1_z3") & (m.side == "entry (buy)"), PLANT * 1e4 * m.triggered, 0.0)
    cols = key + ["trades", "triggered", "mean_bp", "mean_bp_planted", "gain_bp", "expected_gain_bp", "t_day", "t_day_planted", "vs_delay_bp", "vs_delay_bp_planted", "vs_delay_t", "vs_delay_t_planted"]
    print(m[cols].round(2).to_string(index=False))
    return {"rows": json.loads(m[cols].to_json(orient="records"))}


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    res = {}
    for symbol in ("BTCUSDT", "DOGEUSDT"):
        r = part_a(symbol)
        res[f"part_a/{symbol}"] = r
        print(f"\n=== Part A plant through the researcher's pipeline: {symbol} perpetual, design ===")
        for k, v in r.items():
            print(" ", k, {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    print("\n=== Part B plant through the researcher's part_b code: design rule days, spot ===")
    res["part_b"] = part_b()
    (OUT / "planted.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
