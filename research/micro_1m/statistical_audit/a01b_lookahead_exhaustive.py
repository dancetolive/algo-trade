"""Audit step 1 (second form): bar-by-bar truncation around a crash, for the selected definitions and their mirrors.

    python -I a01b_lookahead_exhaustive.py

For BTCUSDT and DOGEUSDT perpetuals (design) and the four days 2021-05-17 .. 2021-05-20 (the largest cascade of
the sample, 19 May 2021) plus 2022-11-08 .. 2022-11-09 for SOLUSDT: the researcher's code is run on the slice cut
after EVERY bar of the last two days (2,880 truncations per symbol). The flag and trade direction that the code
gives for the last bar of each truncated slice must equal what the same code gives for that bar when it sees the
whole slice. This is the literal statement "the flag at t uses bars <= t only", checked at every t.
Writes out/lookahead_exhaustive.json.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / "statistical"))

import definitions as defs  # noqa: E402  researcher's
import eventstudy as es  # noqa: E402  researcher's
from algotrade.data import store  # noqa: E402

NAMES = {"move5_abs2pct_dn", "move5_abs2pct_up", "move1_z3_dn", "move1_z3_up", "fund_post5_big", "move1_z8_dn", "volspike5_exh_dn", "ofi5_z2_both"}
CASES = [("BTCUSDT", "2021-05-17", "2021-05-21"), ("DOGEUSDT", "2021-05-17", "2021-05-21"), ("SOLUSDT", "2022-11-06", "2022-11-10")]


def flags(frame: pd.DataFrame, symbol: str, funding) -> dict:
    grid = es.to_grid(frame, "um", symbol)
    feats = es.trailing_features(grid)
    out = {}
    for d in defs.generate(grid, feats, funding, only=NAMES):
        direction = d.direction if isinstance(d.direction, np.ndarray) else np.full(len(grid), float(d.direction))
        out[d.name] = (d.mask, np.where(d.mask, direction, 0.0))
    return out


def main():
    res = {}
    for symbol, start, end in CASES:
        t0 = time.time()
        frame = store.load_klines("um", "1m", symbol, start=start, end=end)
        assert frame.index[-1] < store.DESIGN_END
        wide = store.load_funding([symbol], end=store.DESIGN_END)
        funding = wide[symbol]
        full = flags(frame, symbol, funding)
        n = len(frame)
        first_cut = n - 2880
        bad = {name: 0 for name in full}
        flagged = {name: int(full[name][0][first_cut:].sum()) for name in full}
        for k in range(first_cut, n):
            part = flags(frame.iloc[: k + 1], symbol, funding)
            for name in full:
                if part[name][0][k] != full[name][0][k] or part[name][1][k] != full[name][1][k]:
                    bad[name] += 1
        res[symbol] = {"slice": [start, end], "bars": n, "truncations": n - first_cut, "flags_in_checked_bars": flagged, "bars_where_truncated_flag_differs": bad}
        print(f"{symbol} {start}..{end}: {n} bars, {n - first_cut} truncations, flags in the checked bars {flagged}; differences {bad}; {time.time() - t0:.0f}s", flush=True)
    (HERE / "out" / "lookahead_exhaustive.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
