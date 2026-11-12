"""Baseline: the simplest long/flat trend rule on BTC and ETH spot, daily bars.

Not a strategy proposal: a reference point that every later idea has to beat,
and an end-to-end exercise of the engine on real data.

Rule: hold the coin while its close is above its N-day simple moving average,
otherwise hold USDT. Decided at the daily close, filled at the next open.
"""

import pandas as pd

from algotrade import research, stats
from algotrade.data import store
from algotrade.validate import check_no_lookahead

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
TRACK = "baseline_trend"
VENUE = research.VENUES["spot"]


def sma_weights(panel, lookback: int, symbols=SYMBOLS) -> pd.DataFrame:
    close = panel["close"][symbols]
    average = close.rolling(lookback).mean()
    above = (close > average) & average.notna()
    return above.astype(float) / len(symbols)


def blend_weights(panel, lookbacks=(50, 100, 200), symbols=SYMBOLS) -> pd.DataFrame:
    return sum(sma_weights(panel, n, symbols) for n in lookbacks) / len(lookbacks)


def main() -> None:
    panel = store.load_panel("spot", "1d", SYMBOLS)
    open_, close = panel["open"], panel["close"]
    rows = []

    def add(name, params, weights):
        record, _ = research.run_trial(TRACK, name, params, weights, open_, close, VENUE)
        for segment in ("design", "validation"):
            m = record["metrics"][segment]
            rows.append({"name": name, "segment": segment, "cagr": m["cagr"], "vol": m["vol"], "sharpe": m["sharpe"],
                         "max_dd": m["max_drawdown"], "turnover/yr": m["turnover_per_year"], "cost/yr": m["cost_per_year"],
                         "in_market": m["time_in_market"]})
        return record

    for lookback in (50, 100, 150, 200):
        problems = check_no_lookahead(lambda p, n=lookback: sma_weights(p, n), panel)
        assert not problems, problems
        add(f"sma{lookback}", {"rule": "close > SMA", "lookback": lookback, "symbols": SYMBOLS}, sma_weights(panel, lookback))
    assert not check_no_lookahead(blend_weights, panel)
    blend = add("blend_50_100_200", {"rule": "mean of SMA rules", "lookbacks": [50, 100, 200], "symbols": SYMBOLS}, blend_weights(panel))

    # Benchmarks (not trials of the rule): buy and hold, half each.
    half = pd.DataFrame(float("nan"), index=close.index, columns=SYMBOLS)
    half.loc[close["ETHUSDT"].first_valid_index()] = 0.5
    hold, _ = research.run_trial(TRACK, "buy_and_hold_50_50", {"benchmark": True}, half, open_, close, VENUE, log=False)
    for segment in ("design", "validation"):
        m = hold["metrics"][segment]
        rows.append({"name": "buy&hold 50/50", "segment": segment, "cagr": m["cagr"], "vol": m["vol"], "sharpe": m["sharpe"],
                     "max_dd": m["max_drawdown"], "turnover/yr": m["turnover_per_year"], "cost/yr": m["cost_per_year"],
                     "in_market": m["time_in_market"]})

    table = pd.DataFrame(rows).set_index(["segment", "name"]).sort_index()
    pd.set_option("display.width", 200)
    print(table.round(3).to_string())
    print()
    print("overfitting report for the blend (design segment):")
    for key, value in research.overfitting_report(TRACK, blend["trial_id"]).items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
