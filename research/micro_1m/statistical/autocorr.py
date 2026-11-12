"""Family 4, descriptive part: short-horizon return autocorrelation and the bid-ask bounce.

    python -I autocorr.py design          -> out/autocorr_design.json

For w in {1, 5, 15, 60} minutes, per symbol and calendar year (perpetuals):
    ac_cc     correlation of adjacent non-overlapping w-minute close-to-close returns
              (the textbook number; at w = 1 it contains the bid-ask bounce because both returns
              share the price close[t])
    ac_trade  correlation of the trailing w-minute return (known at close[t]) with the tradable
              next return open[t+1] -> open[t+1+w]

For w = 1, the reversal trade "sell after an up minute, buy after a down minute" measured three ways:
    from_close   close[t]  -> close[t+1]     not achievable: you cannot trade at the print that made the signal
    from_open    open[t+1] -> open[t+2]      the convention of this study (first print of the next minute)
    skip_one     close[t+1] -> close[t+2]    one full minute later
    step_bp      mean |open[t+1] / close[t] - 1|: the size of the close -> open step, about one bounce
    roll_spread  2 * sqrt(-cov(r_t, r_t+1)): the effective spread implied by the Roll (1984) model
The part of from_close that is missing in from_open is the bounce that cannot be captured.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eventstudy as es  # noqa: E402
import run_events as runner  # noqa: E402

WINDOWS = (1, 5, 15, 60)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 100:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def symbol_stats(market: str, symbol: str, segment: str) -> dict:
    grid, feats, fwd, inside, day0, n_days = runner.load_segment(market, symbol, segment)
    n = len(grid)
    close = grid.close
    year = grid.index.year.to_numpy()
    day_index = np.clip(grid.day - day0, 0, n_days - 1)
    fwd_w = es.forward_returns(grid, WINDOWS)
    zeros = {1: np.zeros(n)}
    out = {"symbol": symbol, "market": market, "years": {}}
    r1 = feats["r1"]
    with np.errstate(invalid="ignore", divide="ignore"):
        next_cc = np.concatenate([close[1:] / close[:-1] - 1.0, [np.nan]])  # close[t] -> close[t+1]
        skip_cc = np.concatenate([close[2:] / close[1:-1] - 1.0, [np.nan, np.nan]])  # close[t+1] -> close[t+2]
        step = np.concatenate([np.abs(grid.open[1:] / close[:-1] - 1.0), [np.nan]])  # |open[t+1] / close[t] - 1|
    side = -np.sign(np.nan_to_num(r1))  # reversal: sell after an up minute
    has_side = grid.usable & np.isfinite(r1) & (r1 != 0)
    for label, sel in [(str(y), year == y) for y in sorted(set(year[inside]))] + [("all", np.ones(n, dtype=bool))]:
        sel = sel & inside
        row = {}
        for w in WINDOWS:
            ret = feats[f"ret{w}"]
            # adjacent non-overlapping windows on the clock: t at the last minute of each w-minute block
            block_end = sel & ((grid.minute_of_day + 1) % w == 0)
            idx = np.flatnonzero(block_end)
            idx = idx[idx + w < n]
            row[f"ac_cc_{w}"] = _corr(ret[idx], ret[idx + w])
            row[f"ac_trade_{w}"] = _corr(ret[idx], fwd_w[w][idx])
        mask = has_side & sel
        for name, series in (("from_close", next_cc), ("from_open", fwd[1]), ("skip_one", skip_cc)):
            agg = es.aggregate(mask, side, {1: series}, zeros, day_index, n_days, (1,))
            stat = es.cluster_mean(agg[0, 0], agg[1, 0])
            row[f"rev1_{name}_bp"] = stat["mean"] * 1e4
            row[f"rev1_{name}_t"] = stat["t"]
        ok = sel & np.isfinite(r1) & np.isfinite(next_cc)
        cov = float(np.cov(r1[ok], next_cc[ok])[0, 1]) if ok.sum() > 100 else float("nan")
        row["roll_spread_bp"] = float(2.0 * np.sqrt(-cov) * 1e4) if cov < 0 else 0.0
        row["step_bp"] = float(np.nanmean(step[sel]) * 1e4)
        row["abs_r1_bp"] = float(np.nanmean(np.abs(r1[sel])) * 1e4)
        row["bars"] = int(sel.sum())
        out["years"][label] = row
    return out


def main(argv: list[str]) -> None:
    segment = argv[0]
    market = argv[1] if len(argv) > 1 else "um"
    symbols = argv[2:] or runner.PERPS
    if segment != "design":
        raise SystemExit("the autocorrelation tables are a design-only description")
    results = [symbol_stats(market, symbol, segment) for symbol in symbols]
    path = runner.OUT / f"autocorr_{segment}_{market}.json"
    path.write_text(json.dumps(results, indent=1))
    rows = []
    for item in results:
        for label, row in item["years"].items():
            rows.append({"symbol": item["symbol"], "year": label, **row})
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 200)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
