"""Audit step 8: Part B (entry timing of the daily trend rule) re-implemented from its description.

    python -I a08_partb.py

Own code throughout (auditlib flags; the daily rule is recomputed here from spot daily closes):
    weight of a coin = (number of the 50/100/200-day SMA filters with close > SMA) / 3 / 2, decided at a daily close,
    filled at the next 00:00 UTC open. For every (coin, day) where the weight goes up (down) the reference fill is
    the 1-minute open at 00:00. Alternative: wait for the first bullish (bearish) structure flagged at the close of
    a bar from 00:00 on, fill at the next minute's open; if none within the window, fill at the open of the bar at
    the end of the window. improvement = side * (P_open - P_wait) / P_open in bp, positive = better price.

Fairness checks made explicitly:
    * the same (coin, day) rows feed the structure rule, the plain delay and the reference (a row is dropped for
      all three or for none);
    * the fallback fill equals the plain-delay fill whenever no structure occurs;
    * the entry bar is never earlier than 00:01 and the flag that triggers it lies strictly before the fill;
    * the 1-minute open at 00:00 equals the daily open.
The researcher's summary tables are read at the end for a side-by-side comparison.
Writes out/partb.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import auditlib as al  # noqa: E402
from algotrade.data import store  # noqa: E402

OUT = HERE / "out"
THEIRS = HERE.parent / "statistical" / "out" / "part_b"
COINS = ["BTCUSDT", "ETHUSDT"]
WINDOWS = (240, 1440)
DAY = pd.Timedelta(days=1)


def rule_changes(segment: str) -> pd.DataFrame:
    _, load_end, first, last = al.SEGMENTS[segment]
    rows = []
    for coin in COINS:
        daily = store.load_klines("spot", "1d", coin, end=load_end)
        close = daily["close"]
        on = sum(((close > close.rolling(n).mean()) & close.rolling(n).mean().notna()).astype(int) for n in (50, 100, 200))
        step = on.diff()
        for decided, d in step.dropna().items():
            if d == 0:
                continue
            fill = decided + DAY
            if first + 2 * DAY <= fill < last - DAY:
                rows.append({"symbol": coin, "fill": fill, "kind": "increase" if d > 0 else "decrease", "filters_before": int(on.loc[decided] - d), "filters_after": int(on.loc[decided]),
                             "daily_open": float(daily["open"].get(fill, np.nan))})
    return pd.DataFrame(rows)


def first_at_or_after(mask: np.ndarray) -> np.ndarray:
    n = len(mask)
    pos = np.where(mask, np.arange(n), n)
    return np.minimum.accumulate(pos[::-1])[::-1]


def timing(segment: str, trades: pd.DataFrame | None, market: str = "spot", plant_bp: float = 0.0) -> pd.DataFrame:
    """One row per (coin, day, kind, structure, window). ``trades`` = None: every day of the segment, both kinds."""
    _, _, first, last = al.SEGMENTS[segment]
    rows = []
    for coin in COINS:
        b = al.Bars(market, coin, segment)
        falls, rises = al.flag_move5_abs2pct(b)
        dips, pops = al.flag_move1_z3(b)
        fmask, fside = al.flag_fund_post5_big(b, al.funding_series(coin, segment))
        flags = {"move1_z3": (dips, pops), "move5_abs2pct": (falls, rises), "fund_post5_big": (fmask & (fside > 0), fmask & (fside < 0))}
        ok = b.present & ~b.halt & (b.qv > 0)
        price = np.where(ok, b.open, np.nan)
        if plant_bp:
            # planted effect (test of the measuring code only): the open right after every 3-sigma dip is made cheaper
            price = price.copy()
            price[np.flatnonzero(dips)[np.flatnonzero(dips) + 1 < b.n] + 1] *= (1.0 - plant_bp / 1e4)
        csum = np.concatenate([[0.0], np.nancumsum(price)])
        ccnt = np.concatenate([[0], np.cumsum(np.isfinite(price))])
        nxt = {name: (first_at_or_after(a), first_at_or_after(c)) for name, (a, c) in flags.items()}
        if trades is None:
            days = pd.date_range(first + 2 * DAY, last - 2 * DAY, freq="1D")
            todo = [(d, k) for d in days for k in ("increase", "decrease")]
        else:
            mine = trades[trades.symbol == coin]
            todo = list(zip(mine.fill, mine.kind))
        for fill, kind in todo:
            p0 = int((fill.value - b.ns[0]) // al.MIN_NS)
            if p0 < 0 or p0 >= b.n or not np.isfinite(price[p0]):
                continue
            sign = 1.0 if kind == "increase" else -1.0
            for name in flags:
                first_flag = nxt[name][0 if kind == "increase" else 1]
                for w in WINDOWS:
                    end = p0 + w
                    if end >= b.n or not np.isfinite(price[end]):
                        continue
                    e = int(first_flag[p0])
                    hit = e <= end - 1
                    at = e + 1 if hit else end
                    if not np.isfinite(price[at]):
                        continue
                    assert at >= p0 + 1 and (not hit or e < at)  # the fill is after 00:00 and after its trigger
                    twap = (csum[end + 1] - csum[p0 + 1]) / max(ccnt[end + 1] - ccnt[p0 + 1], 1)
                    rows.append({"symbol": coin, "fill": fill, "kind": kind, "structure": name, "window": w, "triggered": hit, "waited": at - p0,
                                 "p_open": price[p0], "p_wait": price[at], "p_delay": price[end],
                                 "imp_wait": sign * (price[p0] - price[at]) / price[p0] * 1e4,
                                 "imp_delay": sign * (price[p0] - price[end]) / price[p0] * 1e4,
                                 "imp_twap": sign * (price[p0] - twap) / price[p0] * 1e4})
    return pd.DataFrame(rows)


def day_stat(values: pd.Series, days: pd.Series) -> dict:
    codes = (days.dt.as_unit("ns").astype("int64") // al.DAY_NS).to_numpy()  # the unit matters: the column is not always in ns
    st = al.cluster_stat(values.to_numpy(dtype=float), codes)
    return {"n": st["n"], "days": st["clusters"], "mean": st["mean"], "t": st["t"]}


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (kind, name, w), part in table.groupby(["kind", "structure", "window"], sort=False):
        wait, delay = day_stat(part.imp_wait, part.fill), day_stat(part.imp_delay, part.fill)
        vs = day_stat(part.imp_wait - part.imp_delay, part.fill)
        vt = day_stat(part.imp_wait - part.imp_twap, part.fill)
        fallback_same = bool(np.allclose(part.loc[~part.triggered, "p_wait"], part.loc[~part.triggered, "p_delay"]))
        rows.append({"side": "entry (buy)" if kind == "increase" else "exit (sell)", "structure": name, "window_min": w, "trades": wait["n"], "days": wait["days"],
                     "triggered": float(part.triggered.mean()), "mean_bp": wait["mean"], "median_bp": float(part.imp_wait.median()), "t_day": wait["t"],
                     "worse": float((part.imp_wait < 0).mean()), "sd_bp": float(part.imp_wait.std()), "delay_mean_bp": delay["mean"], "delay_t": delay["t"],
                     "vs_delay_bp": vs["mean"], "vs_delay_t": vs["t"], "vs_twap_bp": vt["mean"], "vs_twap_t": vt["t"],
                     "p05_bp": float(part.imp_wait.quantile(0.05)), "fallback_equals_delay": fallback_same})
    return pd.DataFrame(rows)


KEY = ["side", "structure", "window_min"]
COMPARE = ["trades", "days", "triggered", "mean_bp", "median_bp", "t_day", "worse", "delay_mean_bp", "delay_t", "vs_delay_bp", "vs_delay_t"]


def main():
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    res = {}
    tables = {}
    for segment in ("design", "validation"):
        trades = rule_changes(segment)
        theirs_rc = pd.read_csv(THEIRS / f"rule_changes_{segment}.csv", parse_dates=["fill"])
        same_rc = set(zip(trades.symbol, trades.fill, trades.kind)) == set(zip(theirs_rc.symbol, theirs_rc.fill, theirs_rc.kind))
        print(f"\n=== {segment}: rule changes own {len(trades)} {trades.groupby(['symbol', 'kind']).size().to_dict()}; identical to the researcher's list: {same_rc}")
        for label, tr in (("rule_days", trades), ("all_days", None)):
            table = timing(segment, tr)
            tables[(segment, label)] = table
            mine = summarise(table)
            theirs = pd.read_csv(THEIRS / f"summary_{segment}_spot_{label}.csv")
            both = mine.merge(theirs, on=KEY, suffixes=("", "_theirs"))
            worst = max(float((both[c] - both[c + "_theirs"]).abs().max()) for c in COMPARE)
            print(f"\n--- {segment} / spot / {label}: {len(mine)} rows; largest absolute difference to the researcher's table over {len(COMPARE)} columns: {worst:.2e}; "
                  f"fallback fill equals plain delay in every non-triggered row: {bool(mine.fallback_equals_delay.all())}")
            print(mine[KEY + ["trades", "days", "triggered", "mean_bp", "median_bp", "t_day", "worse", "sd_bp", "p05_bp", "delay_mean_bp", "delay_t", "vs_delay_bp", "vs_delay_t", "vs_twap_bp", "vs_twap_t"]].round(2).to_string(index=False))
            res[f"{segment}/{label}"] = {"rows": json.loads(mine.to_json(orient="records")), "max_abs_diff_to_researcher": worst}
            if tr is not None:
                # the 1-minute open at 00:00 against the daily open
                chk = table.drop_duplicates(["symbol", "fill"]).merge(trades.drop_duplicates(["symbol", "fill"]), on=["symbol", "fill"])
                res[f"{segment}/{label}"]["max_rel_diff_minute_open_vs_daily_open"] = float((chk.p_open / chk.daily_open - 1).abs().max())
                print("  max |1-minute open at 00:00 / daily open - 1|:", res[f"{segment}/{label}"]["max_rel_diff_minute_open_vs_daily_open"])
    for label in ("rule_days", "all_days"):
        both = pd.concat([tables[("design", label)], tables[("validation", label)]], ignore_index=True)
        mine = summarise(both)
        print(f"\n--- design + validation / spot / {label}")
        print(mine[KEY + ["trades", "days", "triggered", "mean_bp", "median_bp", "t_day", "worse", "sd_bp", "delay_mean_bp", "delay_t", "vs_delay_bp", "vs_delay_t"]].round(2).to_string(index=False))
        res[f"both/{label}"] = {"rows": json.loads(mine.to_json(orient="records"))}

    # ---- power: plant a known improvement and see that the measuring code returns it ----------------------------------
    print("\n=== planted effect (design, rule days): the open right after every 3-sigma dip is made 20 bp cheaper ===")
    trades = rule_changes("design")
    base = summarise(timing("design", trades))
    planted = summarise(timing("design", trades, plant_bp=20.0))
    m = base.merge(planted, on=KEY, suffixes=("", "_planted"))
    m["gain_bp"] = m.mean_bp_planted - m.mean_bp
    m["expected_bp"] = np.where((m.structure == "move1_z3") & (m.side == "entry (buy)"), 20.0 * m.triggered, np.nan)
    print(m[KEY + ["trades", "triggered", "mean_bp", "mean_bp_planted", "gain_bp", "expected_bp", "t_day", "t_day_planted"]].round(2).to_string(index=False))
    res["planted_20bp_design_rule_days"] = json.loads(m[KEY + ["trades", "triggered", "mean_bp", "mean_bp_planted", "gain_bp", "expected_bp", "t_day", "t_day_planted"]].to_json(orient="records"))
    (OUT / "partb.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
