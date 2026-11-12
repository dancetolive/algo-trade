"""Appendix (DESIGN ONLY, hourly bars): is the rebound after violent falls a property of six surviving majors?

    python -I survivorship_check.py        -> out/survivorship_design.json

The 1-minute study covers six coins that are large today. Buying falls in coins that are known, after the
fact, to have survived is a biased test. One-minute bars exist only for those six, but hourly bars exist for
every USDT perpetual Binance has listed, including the ones that were later delisted (LUNA, FTT, SRM, ...).
This script repeats the idea at hourly resolution on that whole universe, on design data only:

    event     the hourly close-to-close return of a liquid perpetual is <= -5% (or <= -10%)
    trade     buy at the next hourly open, sell one hour (or four hours) later at the open
    liquid    trailing 7-day mean hourly quote volume >= 1 million USDT at the event (known at the time)
    dead      the perpetual's last hourly bar is before 2023-12-31 (delisted inside the design period)

Four looks (2 thresholds x 2 horizons), split by group. Statistics are day-clustered across all coins and
measured against each coin-year's own mean forward return. Not carried to validation.
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
from algotrade import universe  # noqa: E402
from algotrade.data import store  # noqa: E402
from run_events import PERPS  # noqa: E402

THRESHOLDS = (-0.05, -0.10)
HORIZONS_H = (1, 4)
MIN_HOURLY_VOLUME = 1_000_000.0
DAY0 = pd.Timestamp("2020-01-01", tz="UTC")
N_DAYS = int((store.DESIGN_END - DAY0).days)
LAST_HOUR = store.DESIGN_END - pd.Timedelta(hours=1)


def symbol_events(symbol: str) -> pd.DataFrame | None:
    bars = store.load_klines("um", "1h", symbol, end=store.DESIGN_END, columns=["open", "close", "quote_volume"])
    if len(bars) < 24 * 30:
        return None
    assert bars.index[-1] < store.DESIGN_END
    dead = bars.index[-1] < LAST_HOUR - pd.Timedelta(days=1)
    bars = bars.reindex(pd.date_range(bars.index[0], bars.index[-1], freq="1h"))
    ret = bars["close"] / bars["close"].shift(1) - 1.0
    liquid = bars["quote_volume"].rolling(168, min_periods=120).mean().shift(1) >= MIN_HOURLY_VOLUME  # known before the bar
    out = {"time": bars.index, "ret": ret.to_numpy(), "liquid": liquid.to_numpy()}
    year = bars.index.year
    for h in HORIZONS_H:
        fwd = bars["open"].shift(-1 - h) / bars["open"].shift(-1) - 1.0
        if h > 1:  # every bar of the holding window must exist
            present = bars["open"].notna().astype(float)
            complete = present.rolling(h + 1).sum().shift(-1 - h) == h + 1
            fwd = fwd.where(complete)
        out[f"f{h}"] = fwd.to_numpy()
        out[f"b{h}"] = fwd.groupby(year).transform("mean").to_numpy()
    frame = pd.DataFrame(out)
    frame = frame[(frame.ret <= max(THRESHOLDS)) & frame.liquid]
    if frame.empty:
        return None
    frame["symbol"] = symbol
    frame["dead"] = dead
    return frame


def stat(part: pd.DataFrame, h: int) -> dict:
    values = part[f"f{h}"].to_numpy()
    base = part[f"b{h}"].to_numpy()
    ok = np.isfinite(values) & np.isfinite(base)
    if ok.sum() < 20:
        return {"events": int(ok.sum())}
    day = ((part["time"] - DAY0) / pd.Timedelta(days=1)).astype(int).to_numpy()[ok]
    n_day = np.bincount(day, minlength=N_DAYS).astype(float)
    lags = 1 if h > 1 else 0
    gross = es.cluster_mean(n_day, np.bincount(day, weights=values[ok], minlength=N_DAYS), lags)
    excess = es.cluster_mean(n_day, np.bincount(day, weights=(values - base)[ok], minlength=N_DAYS), lags)
    return {"events": gross["n"], "days": gross["days"], "coins": int(part.loc[ok, "symbol"].nunique()),
            "gross_bp": gross["mean"] * 1e4, "excess_bp": excess["mean"] * 1e4, "t_day": excess["t"],
            "median_bp": float(np.median(values[ok])) * 1e4, "hit": float((values[ok] > 0).mean()),
            "net_taker_bp": gross["mean"] * 1e4 - 14.0}


def main() -> None:
    names = universe.coin_symbols(store.symbols("um", "1h"))
    frames = [f for f in (symbol_events(name) for name in names) if f is not None]
    events = pd.concat(frames, ignore_index=True)
    events = events[events.time >= DAY0]
    majors = events.symbol.isin(PERPS)
    groups = {
        "all liquid perpetuals": np.ones(len(events), dtype=bool),
        "the six coins of the 1-minute study": majors.to_numpy(),
        "all others": (~majors).to_numpy(),
        "delisted before 2024 (dead coins)": events.dead.to_numpy(),
        "others still listed at the end of 2023": (~majors & ~events.dead).to_numpy(),
    }
    out = {"symbols_scanned": len(names), "symbols_with_events": int(events.symbol.nunique()),
           "dead_symbols_with_events": sorted(events.loc[events.dead, "symbol"].unique().tolist()), "rows": []}
    for threshold in THRESHOLDS:
        hit = (events.ret <= threshold).to_numpy()
        for h in HORIZONS_H:
            for label, mask in groups.items():
                out["rows"].append({"fall": f"<= {threshold:.0%} in one hour", "hold_hours": h, "group": label, **stat(events[hit & mask], h)})
            for year in (2020, 2021, 2022, 2023):
                out["rows"].append({"fall": f"<= {threshold:.0%} in one hour", "hold_hours": h, "group": f"all liquid perpetuals, {year}",
                                    **stat(events[hit & (events.time.dt.year == year).to_numpy()], h)})
    (HERE / "out" / "survivorship_design.json").write_text(json.dumps(out, indent=1, default=float))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print("symbols scanned:", out["symbols_scanned"], "with events:", out["symbols_with_events"], "dead with events:", len(out["dead_symbols_with_events"]), out["dead_symbols_with_events"][:40])
    print(pd.DataFrame(out["rows"]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
