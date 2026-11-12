"""Independent recomputation of the headline number with plain pandas (no shared code with eventstudy.py).

    python -I independent_check.py design|validation

Definition re-implemented from its description: on each perpetual, flag bar t when
close[t] / close[t-5] - 1 <= -2% and no bar among the previous four had |5-minute return| >= 2%
(first bar of a crossing in either direction, as in definitions.py); buy at open[t+1], sell at
open[t+16]. Gaps: the bars are reindexed to a complete minute grid, so any missing bar makes the
returns that touch it NaN; zero-volume halts are NOT treated specially here, which is the one
deliberate difference from the main module (it shows the result does not hinge on that rule).
"""

import sys

import numpy as np
import pandas as pd

from algotrade.data import store

SEGMENT = sys.argv[1] if len(sys.argv) > 1 else "design"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]
if SEGMENT == "design":
    start, end, first = None, store.DESIGN_END, pd.Timestamp("2020-01-01", tz="UTC")
else:
    start, end, first = pd.Timestamp("2023-12-25", tz="UTC"), store.HOLDOUT_START, store.DESIGN_END

rows = []
for symbol in SYMBOLS:
    bars = store.load_klines("um", "1m", symbol, start=start, end=end, columns=["open", "close"])
    bars = bars.reindex(pd.date_range(bars.index[0], bars.index[-1], freq="1min"))
    ret5 = bars["close"] / bars["close"].shift(5) - 1.0
    crossing = (ret5.abs() >= 0.02).to_numpy()
    kept = np.zeros(len(bars), dtype=bool)
    last = -10
    for t in np.flatnonzero(crossing):  # first bar of a crossing: none kept in the previous four bars
        if t - last >= 5:
            kept[t] = True
            last = t
    flag = kept & (ret5.to_numpy() < 0) & (bars.index >= first)
    forward = (bars["open"].shift(-16) / bars["open"].shift(-1) - 1.0).to_numpy()
    part = pd.DataFrame({"symbol": symbol, "time": bars.index[flag], "f15": forward[flag]}).dropna()
    rows.append(part)
events = pd.concat(rows, ignore_index=True)
day_sum = events.groupby(events.time.dt.floor("D"))["f15"].agg(["sum", "count"])
mean = events.f15.mean()
score = day_sum["sum"] - mean * day_sum["count"]
groups = len(day_sum)
se = np.sqrt((score**2).sum() * groups / (groups - 1)) / len(events)
print(f"{SEGMENT}: events {len(events)}, days {groups}, mean {mean * 1e4:.2f} bp, day-clustered t {mean / se:.2f}, hit {(events.f15 > 0).mean():.3f}")
print(events.groupby("symbol")["f15"].agg(["count", "mean"]).assign(mean=lambda d: d["mean"] * 1e4).round(2).to_string())
