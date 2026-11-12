"""Event-level table for a handful of definitions (robustness checks that need more than day sums).

    python -I event_level.py design um BTCUSDT ETHUSDT ...   -> out/events_design/um_<SYMBOL>.parquet
    python -I event_level.py validation ...                   (only definitions listed in SELECTION.json)

Columns per event: symbol, time, definition, side (+1 buy / -1 sell: the side that was hypothesised),
the triggering return and z-score, and forward returns under several execution assumptions:
    f<h>        open[t+1] -> open[t+1+h]                 (the convention)
    b<h>        matched baseline of f<h>
    lag2_<h>    open[t+2] -> open[t+2+h]                 (one minute late)
    lag3_<h>    open[t+3] -> open[t+3+h]                 (two minutes late)
    vwap_<h>    vwap[t+1] -> vwap[t+1+h]                 (filled at the average price of the minute)
    hi_<h>      high[t+1] -> open[t+1+h]                 (worst buy price of the first minute)
    lo_<h>      low[t+1]  -> open[t+1+h]                 (worst sell price of the first minute)
    path<m>     open[t+1] -> open[t+1+m] for a finer set of minutes (average path after the event)
    c0, o1      close of the event bar and open of the next bar (prices)
    minlow<m>   lowest low of bars t+1 .. t+m, maxhigh<m> the highest high (for resting-order fill tests)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import definitions as defs  # noqa: E402
import eventstudy as es  # noqa: E402
import run_events as runner  # noqa: E402
from algotrade.data import store  # noqa: E402

DESIGN_NAMES = [
    "move1_z3_dn", "move1_z5_dn", "move1_z8_dn", "move5_z3_dn", "move5_z8_dn", "move5_abs2pct_dn", "move15_z3_dn",
    "move1_z3_up", "move1_z8_up", "move5_abs2pct_up",
    "trdspike5_exh_dn", "volspike5_exh_dn", "trdspike10_cont_dn",
    "fund_post5_big", "fund_post5_signed",
]
PATH_MINUTES = (1, 2, 3, 5, 10, 15, 30, 60, 120, 240, 480, 1440)
ROBUST_H = (5, 15, 60)


def event_table(market: str, symbol: str, segment: str, names: list[str]) -> pd.DataFrame:
    grid, feats, fwd, inside, day0, n_days = runner.load_segment(market, symbol, segment)
    base = runner.baselines(grid, feats, fwd, inside)
    funding = None
    if market == "um":
        load_start, load_end, _, _ = runner.SEGMENTS[segment]
        wide = store.load_funding([symbol], start=None if load_start is None else load_start - pd.Timedelta(days=2), end=load_end)
        funding = wide[symbol] if symbol in wide.columns else None
    path = es.forward_returns(grid, PATH_MINUTES)
    lag2 = es.forward_returns(grid, ROBUST_H, entry_lag=2)
    lag3 = es.forward_returns(grid, ROBUST_H, entry_lag=3)
    vwap = es.forward_returns(grid, ROBUST_H, entry_price="vwap", exit_price="vwap")
    high = es.forward_returns(grid, ROBUST_H, entry_price="high")
    low = es.forward_returns(grid, ROBUST_H, entry_price="low")
    rate = defs.previous_funding(grid, funding) if funding is not None else np.full(len(grid), np.nan)
    next_open = np.concatenate([grid.raw_open[1:], [np.nan]])
    extremes = {}
    for m in (1, 5, 15):
        extremes[f"minlow{m}"] = pd.Series(grid.raw_low).rolling(m, min_periods=m).min().shift(-m).to_numpy()
        extremes[f"maxhigh{m}"] = pd.Series(grid.raw_high).rolling(m, min_periods=m).max().shift(-m).to_numpy()
    frames = []
    for d in defs.generate(grid, feats, funding, only=set(names)):
        rows = np.flatnonzero(d.mask & inside)
        if len(rows) == 0:
            continue
        side = d.direction[rows] if isinstance(d.direction, np.ndarray) else np.full(len(rows), float(d.direction))
        data = {
            "symbol": symbol,
            "market": market,
            "definition": d.name,
            "time": grid.index[rows],
            "side": side,
            "ret1": feats["ret1"][rows],
            "ret5": feats["ret5"][rows],
            "z1": feats["z1"][rows],
            "z5": feats["z5"][rows],
            "vratio": feats["vratio"][rows],
            "funding_prev": rate[rows],
            "c0": grid.close[rows],
            "o1": next_open[rows],
        }
        for key, values in extremes.items():
            data[key] = values[rows]
        for h in es.HORIZONS:
            data[f"f{h}"] = fwd[h][rows]
            data[f"b{h}"] = base[d.baseline][h][rows]
        for m in PATH_MINUTES:
            data[f"path{m}"] = path[m][rows]
        for h in ROBUST_H:
            data[f"lag2_{h}"] = lag2[h][rows]
            data[f"lag3_{h}"] = lag3[h][rows]
            data[f"vwap_{h}"] = vwap[h][rows]
            data[f"hi_{h}"] = high[h][rows]
            data[f"lo_{h}"] = low[h][rows]
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main(argv: list[str]) -> None:
    segment, market, *symbols = argv
    names = DESIGN_NAMES
    horizon_of = None
    if segment == "validation":
        selection = HERE / "SELECTION.json"
        if not selection.exists():
            raise SystemExit("refusing to touch the validation segment before SELECTION.json is written")
        record = json.loads(selection.read_text())
        names = record["definition_names_to_evaluate"]
        horizon_of = {name: int(h) for name, h in record["horizons_to_evaluate"].items()}
    folder = runner.OUT / f"events_{segment}"
    folder.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        table = event_table(market, symbol, segment, names)
        if horizon_of is not None and len(table):
            # validation: keep the selected horizon only; every other forward return is blanked (not looked at)
            keep_h = table["definition"].map(horizon_of)
            for column in list(table.columns):
                head, _, tail = column.rpartition("_")
                if column.startswith("path"):
                    table = table.drop(columns=column)
                elif column[0] in "fb" and column[1:].isdigit():
                    table.loc[keep_h != int(column[1:]), column] = np.nan
                elif head in ("lag2", "lag3", "vwap", "hi", "lo") and tail.isdigit():
                    table.loc[keep_h != int(tail), column] = np.nan
        table.to_parquet(folder / f"{market}_{symbol}.parquet")
        print(segment, market, symbol, len(table), "events", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
