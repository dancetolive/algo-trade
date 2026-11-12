"""Run the event definitions on one segment and store per-day sums.

    python -I run_events.py design um BTCUSDT ETHUSDT ...
    python -I run_events.py validation um BTCUSDT ETHUSDT ...      (needs SELECTION.json: only the selected
                                                                    definitions, at their selected horizon, are computed)

Output: out/<segment>/<market>_<SYMBOL>.npz with
    names   definition names
    agg     array (definition, field, horizon, day) of per-day sums; fields = eventstudy.AGG_FIELDS
    meta    JSON: definitions (family, params, hypothesis, horizons that count as looks, baseline), day0, n_days

The design segment is loaded with end = 2024-01-01, so no validation bar is ever in memory in a
design run and a forward return that would end in 2024 is dropped. The validation segment is loaded
from 2023-12-25 only to warm up the trailing windows; events and the baseline start on 2024-01-01.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import definitions as defs  # noqa: E402
import eventstudy as es  # noqa: E402
from algotrade.data import store  # noqa: E402

OUT = HERE / "out"
SEGMENTS = {
    # name: (load start, load end, first event day, first day after the segment)
    "design": (None, store.DESIGN_END, pd.Timestamp("2020-01-01", tz="UTC"), store.DESIGN_END),
    "validation": (pd.Timestamp("2023-12-25", tz="UTC"), store.HOLDOUT_START, store.DESIGN_END, store.HOLDOUT_START),
}
PERPS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]


def load_segment(market: str, symbol: str, segment: str):
    """Grid, features, forward returns and in-segment flag for one symbol."""
    load_start, load_end, first_day, end_day = SEGMENTS[segment]
    frame = store.load_klines(market, "1m", symbol, start=load_start, end=load_end)
    if segment == "design":
        assert frame.index[-1] < store.DESIGN_END, "a design run must not see validation bars"
    assert frame.index[-1] < store.HOLDOUT_START, "hold-out bars must never be loaded"
    grid = es.to_grid(frame, market, symbol)
    del frame
    feats = es.trailing_features(grid)
    fwd = es.forward_returns(grid)
    inside = np.asarray((grid.index >= first_day) & (grid.index < end_day))
    day0 = int(first_day.value // (86400 * 10**9))
    n_days = int((end_day - first_day).days)
    return grid, feats, fwd, inside, day0, n_days


def baselines(grid, feats, fwd, inside):
    """Matched baselines, computed from in-segment bars only."""
    fwd_in = {h: np.where(inside, values, np.nan) for h, values in fwd.items()}
    out = {}
    for key, by_hour in (("hour", True), ("nohour", False)):
        cell, n_cells = es.baseline_cells(grid, feats["vol_state"], by_hour=by_hour)
        out[key] = es.baseline_returns(fwd_in, cell, n_cells)
    return out


def run_symbol(market: str, symbol: str, segment: str, only: set | None = None, tag: str = "", horizon_of: dict | None = None) -> Path:
    """``horizon_of`` (validation): definition name -> the one horizon that was selected; every other horizon is left empty."""
    started = time.time()
    grid, feats, fwd, inside, day0, n_days = load_segment(market, symbol, segment)
    base = baselines(grid, feats, fwd, inside)
    day_index = np.clip(grid.day - day0, 0, n_days - 1)
    funding = None
    if market == "um":
        load_start, load_end, _, _ = SEGMENTS[segment]
        wide = store.load_funding([symbol], start=None if load_start is None else load_start - pd.Timedelta(days=2), end=load_end)
        if symbol in wide.columns:
            funding = wide[symbol]
    names, metas, blocks = [], [], []
    for d in defs.generate(grid, feats, funding, only=only):
        mask = d.mask & inside
        block = es.aggregate(mask, d.direction, fwd, base[d.baseline], day_index, n_days)
        horizons = list(d.horizons)
        if horizon_of is not None:
            keep = es.HORIZONS.index(horizon_of[d.name])
            block[:, [j for j in range(len(es.HORIZONS)) if j != keep], :] = 0.0  # not selected: not looked at
            horizons = [horizon_of[d.name]]
        blocks.append(block)
        names.append(d.name)
        metas.append({"name": d.name, "family": d.family, "params": d.params, "hypothesis": d.hypothesis, "horizons": horizons, "baseline": d.baseline})
    folder = OUT / segment
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{market}_{symbol}{tag}.npz"
    meta = {
        "market": market,
        "symbol": symbol,
        "segment": segment,
        "day0": day0,
        "n_days": n_days,
        "first_bar": str(grid.index[0]),
        "last_bar": str(grid.index[-1]),
        "bars_in_segment": int((grid.exists & inside).sum()),
        "outage_bars_in_segment": int((grid.outage & inside).sum()),
        "horizons": list(es.HORIZONS),
        "fields": list(es.AGG_FIELDS),
        "definitions": metas,
    }
    np.savez_compressed(path, names=np.array(names), agg=np.stack(blocks), meta=json.dumps(meta))
    print(f"{segment} {market} {symbol}: {len(names)} definitions, {meta['bars_in_segment']} bars, "
          f"{grid.index[0]} .. {grid.index[-1]}, {time.time() - started:.0f}s", flush=True)
    return path


def main(argv: list[str]) -> None:
    segment, market, *rest = argv
    only = None
    tag = ""
    symbols = []
    it = iter(rest)
    for item in it:
        if item == "--only":
            only = set(next(it).split(","))
        elif item == "--tag":
            tag = next(it)
        else:
            symbols.append(item)
    horizon_of = None
    if segment == "validation":
        selection = HERE / "SELECTION.json"
        if not selection.exists():
            raise SystemExit("refusing to touch the validation segment before SELECTION.json is written")
        record = json.loads(selection.read_text())
        only = set(record["definition_names_to_evaluate"])  # nothing else is evaluated, whatever was asked for
        horizon_of = {name: int(h) for name, h in record["horizons_to_evaluate"].items()}
    for symbol in symbols:
        run_symbol(market, symbol, segment, only, tag, horizon_of)


if __name__ == "__main__":
    main(sys.argv[1:])
