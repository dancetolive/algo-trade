"""Audit check 8: Part B (entry timing of the daily trend rule) re-implemented from scratch.

    python -I a10_part_b.py

Nothing is imported from the researcher's code or from research/baseline: the trend rule (blend of the 50/100/200-day
SMA filters on BTC and ETH spot daily closes, change decided at the close of day D, executed at 00:00 UTC of D+1),
the four structures (two sweeps, break of structure with 5/5 swings, return into a fair value gap) and the
comparison are written here independently:
  * break of structure  : explicit bar-by-bar state machine;
  * fair-value-gap fill : first return found with a monotonic stack (next lower-or-equal low / higher-or-equal high);
  * trades are located by TIMESTAMP on the 1-minute frame.
Output: out/part_b_audit.csv (same layout as the researcher's summary) and a comparison with out/part_b_*.csv.
Fairness checks: identical trade set in both arms, fallback fill, trigger never earlier than the 00:00 bar, fill
strictly after the trigger bar, and a brute-force recomputation of a random sample of trades from raw slices.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_lib as al  # noqa: E402
from algotrade.data import store  # noqa: E402

THEIRS = os.path.join(os.path.dirname(al.HERE), "price_action", "out")
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
WINDOWS = (240, 1440)
MIN = pd.Timedelta(minutes=1)


# ----------------------------------------------------------------------------- trend rule (own)
def trend_trades(segment: str) -> pd.DataFrame:
    end = al.T_DESIGN_END if segment == "design" else None
    rows = []
    for sym in SYMBOLS:
        close = store.load_klines("spot", "1d", sym, end=end, columns=["close"])["close"]
        weight = sum(((close > close.rolling(n).mean()) & close.rolling(n).mean().notna()).astype(float) for n in (50, 100, 200)) / 3.0 / len(SYMBOLS)
        delta = weight.diff().dropna()
        for day, dv in delta[delta.abs() > 1e-12].items():
            rows.append(dict(symbol=sym, fill_time=day + pd.Timedelta(days=1), side=1 if dv > 0 else -1, size=abs(float(dv))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- structures (own)
def bos_signal(h: np.ndarray, l: np.ndarray, c: np.ndarray, size: int) -> np.ndarray:
    """First close beyond the latest confirmed swing (swing = `size` bars left and right), by an explicit loop."""
    n = len(c)
    pad_h, pad_l = np.concatenate([h, [np.nan]]), np.concatenate([l, [np.nan]])
    left_hi, left_lo = al.prior_extreme(pad_h, size, "max"), al.prior_extreme(pad_l, size, "min")   # extreme of [s-size, s-1] at s
    is_hi = np.zeros(n, dtype=bool)
    is_lo = np.zeros(n, dtype=bool)
    s = np.arange(size, n - size)
    with np.errstate(invalid="ignore"):
        is_hi[s] = (h[s] > left_hi[s]) & (h[s] >= left_hi[s + size + 1])    # left_hi[s+size+1] = max h[s+1 .. s+size]
        is_lo[s] = (l[s] < left_lo[s]) & (l[s] <= left_lo[s + size + 1])
    hi_l, lo_l, c_l, ih, il = h.tolist(), l.tolist(), c.tolist(), is_hi.tolist(), is_lo.tolist()
    out = np.zeros(n, dtype=np.int8)
    lvl_hi = lvl_lo = None
    used_hi = used_lo = True
    for t in range(size, n):
        sw = t - size                               # the bar whose swing status becomes known at t
        if ih[sw]:
            lvl_hi, used_hi = hi_l[sw], False
        if il[sw]:
            lvl_lo, used_lo = lo_l[sw], False
        ct = c_l[t]
        up = (not used_hi) and ct > lvl_hi
        dn = (not used_lo) and ct < lvl_lo
        if up:
            used_hi = True
        if dn:
            used_lo = True
        if up != dn:
            out[t] = 1 if up else -1
    return out


def fvg_fill_signal(h: np.ndarray, l: np.ndarray, c: np.ndarray, m: float, wait: int, ref_n: int = 60) -> np.ndarray:
    """First return into a fair value gap within `wait` bars, kept if that bar does not close beyond the far edge."""
    n = len(c)
    ref = pd.Series(h - l).rolling(ref_n, min_periods=ref_n).mean().shift(3).to_numpy()
    h2 = np.concatenate([[np.nan, np.nan], h[:-2]])
    l2 = np.concatenate([[np.nan, np.nan], l[:-2]])
    with np.errstate(invalid="ignore"):
        up = (l > h2) & ((l - h2) >= m * ref) & (ref > 0)
        dn = (h < l2) & ((l2 - h) >= m * ref) & (ref > 0)
    low = np.where(np.isfinite(l), l, np.inf).tolist()
    high = np.where(np.isfinite(h), h, -np.inf).tolist()
    upl, dnl = up.tolist(), dn.tolist()
    score = np.zeros(n, dtype=np.int64)
    stack_up: list[int] = []      # bullish zones waiting for a bar with low <= their near edge (= their own low)
    stack_dn: list[int] = []
    close, far_up, far_dn = c, h2, l2
    for t in range(n):
        lt, ht = low[t], high[t]
        while stack_up and lt <= low[stack_up[-1]]:
            s = stack_up.pop()
            if t - s <= wait and close[t] > far_up[s]:
                score[t] += 1
        while stack_dn and ht >= high[stack_dn[-1]]:
            s = stack_dn.pop()
            if t - s <= wait and close[t] < far_dn[s]:
                score[t] -= 1
        if upl[t]:
            stack_up.append(t)
        if dnl[t]:
            stack_dn.append(t)
    return np.sign(score).astype(np.int8)


# ----------------------------------------------------------------------------- evaluation (own)
def evaluate(g: pd.DataFrame, signals: dict[str, np.ndarray], fill_times: pd.DatetimeIndex, side: int) -> pd.DataFrame:
    o = g["open"].to_numpy()
    present = np.isfinite(o) & np.isfinite(g["close"].to_numpy())
    n = len(o)
    cs = np.concatenate([[0], np.cumsum(present)])
    pos = g.index.get_indexer(fill_times)
    rows = []
    for window in WINDOWS:
        good = (pos >= al.CONTEXT) & (pos + window < n) & (pos >= 0)
        p = pos[good]
        good2 = (cs[p + window + 1] - cs[p - al.CONTEXT]) == (window + al.CONTEXT + 1)
        p = p[good2]
        times = fill_times[good][good2]
        p0 = o[p]
        end_improve = side * (p0 - o[p + window]) / p0 * 1e4
        for name, sig in signals.items():
            ev_pos = np.flatnonzero(sig == side)
            k = np.searchsorted(ev_pos, p, side="left")                 # first event at or after the 00:00 bar
            first = np.where(k < len(ev_pos), ev_pos[np.minimum(k, len(ev_pos) - 1)], n + 10 ** 9)
            trig = first <= p + window - 1
            fill = np.where(trig, first + 1, p + window)
            assert (fill > p).all() and (fill <= p + window).all()
            assert (fill[trig] > first[trig]).all()
            imp = side * (p0 - o[fill]) / p0 * 1e4
            delay = fill - p
            # time-matched placebo: each trade filled after the waiting times of all the OTHER trades of this group
            plc = np.empty(len(p))
            for i in range(len(p)):
                others = np.delete(delay, i)
                plc[i] = np.mean(side * (p0[i] - o[p[i] + others]) / p0[i] * 1e4) if len(others) else 0.0
            rows.append(pd.DataFrame({"structure": name, "window": window, "time": times, "side": side, "triggered": trig, "delay": delay,
                                      "improvement": imp, "delay_end": end_improve, "placebo": plc, "info": imp - plc,
                                      "trigger_pos": np.where(trig, first, -1), "t0_pos": p}))
    return pd.concat(rows, ignore_index=True)


def summarise(tr: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (label, name, window), part in tr.groupby(["set", "structure", "window"], sort=False):
        day = ((part["time"] - EPOCH) // pd.Timedelta(days=1)).to_numpy()
        adj = window >= 1440
        s = al.cluster_stats(part["improvement"].to_numpy(), day, adjacent=adj)
        e = al.cluster_stats(part["delay_end"].to_numpy(), day, adjacent=adj)
        i = al.cluster_stats(part["info"].to_numpy(), day, adjacent=adj)
        imp = part["improvement"]
        out.append(dict(set=label, structure=name, window=window, trades=len(part), days=s["clusters"], triggered=part["triggered"].mean(),
                        mean_wait_min=part["delay"].mean(), median_wait_min=part["delay"].median(), mean_bp=s["mean"], median_bp=imp.median(),
                        se_bp=s["se"], t=s["t"], worse=(imp < 0).mean(), better=(imp > 0).mean(), mean_if_better=imp[imp > 0].mean(),
                        mean_if_worse=imp[imp < 0].mean(), delay_end_bp=part["delay_end"].mean(), t_delay_end=e["t"],
                        placebo_bp=part["placebo"].mean(), info_bp=i["mean"], t_info=i["t"]))
    return pd.DataFrame(out)


def brute_force(g: pd.DataFrame, raw: pd.DataFrame, tr: pd.DataFrame, rng, k: int = 12) -> int:
    """Recompute sampled trend trades for the small sweep (N=30, 2 bp, k=3) bar by bar from raw slices."""
    part = tr[(tr["structure"] == "sweep_N30_x2_k3") & (tr["window"] == 240) & tr["set"].isin(["trend_entry", "trend_exit"])]
    bad = 0
    for _, row in part.sample(min(k, len(part)), random_state=int(rng.integers(1 << 30))).iterrows():
        t0, side = row["time"], int(row["side"])
        p0 = raw.loc[t0, "open"]
        fill_px = None
        for j in range(240):
            t = t0 + j * MIN                                       # candidate event bar (close of this bar)
            hit = False
            for back in range(3):                                  # breach bar s = t - back
                s = t - back * MIN
                w = raw.loc[s - 30 * MIN: s - MIN]
                lev = w["low"].min() if side > 0 else w["high"].max()
                prev_close = w["close"].iloc[-1]
                b = raw.loc[s]
                breach = (b["low"] <= lev * (1 - 2e-4) and prev_close >= lev) if side > 0 else (b["high"] >= lev * (1 + 2e-4) and prev_close <= lev)
                if not breach:
                    continue
                closes = raw.loc[s: t, "close"].to_numpy()
                if side > 0:
                    ok = closes[-1] > lev and (closes[:-1] <= lev).all()
                else:
                    ok = closes[-1] < lev and (closes[:-1] >= lev).all()
                hit |= bool(ok)
            if hit:
                # a bar that is at once a bullish and a bearish event is dropped by the definition
                opp = False
                for back in range(3):
                    s = t - back * MIN
                    w = raw.loc[s - 30 * MIN: s - MIN]
                    lev = w["high"].max() if side > 0 else w["low"].min()
                    prev_close = w["close"].iloc[-1]
                    b = raw.loc[s]
                    breach = (b["high"] >= lev * (1 + 2e-4) and prev_close <= lev) if side > 0 else (b["low"] <= lev * (1 - 2e-4) and prev_close >= lev)
                    if breach:
                        closes = raw.loc[s: t, "close"].to_numpy()
                        opp |= bool((closes[-1] < lev and (closes[:-1] >= lev).all()) if side > 0 else (closes[-1] > lev and (closes[:-1] <= lev).all()))
                if not opp:
                    fill_px = raw.loc[t + MIN, "open"]
                    break
        if fill_px is None:
            fill_px = raw.loc[t0 + 240 * MIN, "open"]
        lit = side * (p0 - fill_px) / p0 * 1e4
        bad += int(abs(lit - row["improvement"]) > 1e-9)
    return bad


def main() -> None:
    rng = np.random.default_rng(5)
    report = {}
    all_tables = []
    for segment in ("design", "validation"):
        trades = trend_trades(segment)
        first_event = None if segment == "design" else al.T_DESIGN_END
        counts = trades[trades["fill_time"] >= pd.Timestamp("2020-01-01", tz="UTC")]
        if first_event is not None:
            counts = counts[counts["fill_time"] >= first_event]
        report[f"{segment}_rule_trades_by_year_and_side"] = {f"{y}:{'buy' if sd > 0 else 'sell'}": int(v)
                                                             for (y, sd), v in counts.groupby([counts["fill_time"].dt.year, "side"]).size().items()}
        report[f"{segment}_rule_trades"] = int(len(counts))
        report[f"{segment}_mean_trade_size"] = float(counts["size"].mean())
        pieces = []
        bad_total, checked = 0, 0
        for sym in SYMBOLS:
            g, _ = al.load_grid("spot", sym, segment)
            o, h, l, c, present = al.arrays(g)
            signals = {"sweep_N240_x10_k1": al.sweep_signal(h, l, c, 240, 10.0, 1), "sweep_N30_x2_k3": al.sweep_signal(h, l, c, 30, 2.0, 3),
                       "bos_s5_all": bos_signal(h, l, c, 5), "fvgfill_m0p5_W60": fvg_fill_signal(h, l, c, 0.5, 60)}
            mine = trades[trades["symbol"] == sym]
            midnights = g.index[(g.index.hour == 0) & (g.index.minute == 0)]
            sets = {"trend_entry": (pd.DatetimeIndex(mine.loc[mine["side"] > 0, "fill_time"]), 1),
                    "trend_exit": (pd.DatetimeIndex(mine.loc[mine["side"] < 0, "fill_time"]), -1),
                    "all_days_buy": (midnights, 1), "all_days_sell": (midnights, -1)}
            for label, (times, side) in sets.items():
                times = times[(times >= g.index[0]) & (times <= g.index[-1])]
                if first_event is not None:
                    times = times[times >= first_event]
                if len(times) == 0:
                    continue
                part = evaluate(g, signals, times, side)
                part.insert(0, "set", label)
                part.insert(0, "symbol", sym)
                pieces.append(part)
            raw = g.dropna(subset=["open"])
            sym_tr = pd.concat([p for p in pieces if p["symbol"].iloc[0] == sym], ignore_index=True)
            bad_total += brute_force(g, raw, sym_tr, rng)
            checked += 12
        tr = pd.concat(pieces, ignore_index=True)
        both = tr[tr["set"].isin(["trend_entry", "trend_exit"])].assign(set="trend_both")
        every = tr[tr["set"].isin(["all_days_buy", "all_days_sell"])].assign(set="all_days_both")
        table = summarise(pd.concat([tr, both, every], ignore_index=True))
        table.insert(0, "segment", segment)
        all_tables.append(table)
        report[f"{segment}_brute_force_trades_checked"] = checked
        report[f"{segment}_brute_force_mismatches"] = bad_total
        # fairness facts
        te = tr[tr["set"].isin(["trend_entry", "trend_exit"])]
        report[f"{segment}_min_delay_minutes"] = int(te["delay"].min())
        report[f"{segment}_max_delay_minutes"] = int(te["delay"].max())
        report[f"{segment}_trigger_before_t0"] = int(((te["trigger_pos"] >= 0) & (te["trigger_pos"] < te["t0_pos"])).sum())
        report[f"{segment}_fallback_share_small_sweep_240"] = float(1 - te[(te["structure"] == "sweep_N30_x2_k3") & (te["window"] == 240)]["triggered"].mean())
        report[f"{segment}_fallback_share_big_sweep_240"] = float(1 - te[(te["structure"] == "sweep_N240_x10_k1") & (te["window"] == 240)]["triggered"].mean())
    table = pd.concat(all_tables, ignore_index=True)
    table.to_csv(os.path.join(al.OUT, "part_b_audit.csv"), index=False)

    # comparison with the researcher's summary (spot)
    ref = pd.concat([pd.read_csv(os.path.join(THEIRS, f"part_b_{s}.csv")) for s in ("design", "validation")], ignore_index=True)
    ref = ref[ref["market"] == "spot"]
    keys = ["segment", "set", "structure", "window"]
    merged = table.merge(ref, on=keys, suffixes=("", "_r"))
    diffs = {}
    for col in ("trades", "days", "triggered", "mean_wait_min", "mean_bp", "median_bp", "t", "worse", "delay_end_bp", "t_delay_end", "placebo_bp", "info_bp", "t_info"):
        diffs[col] = float((merged[col] - merged[col + "_r"]).abs().max())
    report["rows_compared"] = int(len(merged))
    report["rows_in_researcher_table"] = int(len(ref))
    report["max_abs_difference_by_column"] = diffs
    with open(os.path.join(al.OUT, "part_b_audit.json"), "w") as handle:
        json.dump(report, handle, indent=1)
    print(json.dumps(report, indent=1))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    show = merged[merged["set"].isin(["trend_entry", "trend_exit", "trend_both", "all_days_both"])]
    cols = ["segment", "set", "structure", "window", "trades", "trades_r", "triggered", "mean_wait_min", "mean_bp", "mean_bp_r", "median_bp", "t", "t_r", "worse", "worse_r",
            "delay_end_bp", "info_bp", "info_bp_r", "t_info"]
    print(show[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
