"""Side analyses quoted in the report. Everything here is DESIGN-ONLY except the last block.

    python -I design_extras.py        -> out/design_extras.json

1. resting orders   What a maker entry really gets: a bid resting at (or below) the first print of the next
                    minute is filled only if a later low trades through it. Fill rate, result of the filled
                    signals, result of the signals that ran away unfilled, and net per signal with a 2 bp maker
                    entry and a taker exit. This is why the flat "maker bound" in the tables is optimistic.
2. cascade anatomy  For the validated structure (fall of >= 2% in 5 minutes): first event of a cascade versus
                    later legs, one position at a time per coin, size of the fall, number of coins falling
                    together. NOT validated: these splits were looked at after the validation run and only on
                    design data.
3. cost stress      Per-day sums of the selected pair (one unit of notional per event, no cap on concurrent
                    positions) under extra slippage. Design and validation: same events as the selected pair,
                    so no new definition is examined on validation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analysis as an  # noqa: E402
import eventstudy as es  # noqa: E402
from run_events import PERPS  # noqa: E402


def load_events(segment: str = "design") -> pd.DataFrame:
    frames = [pd.read_parquet(an.OUT / f"events_{segment}" / f"um_{symbol}.parquet") for symbol in PERPS]
    events = pd.concat(frames, ignore_index=True)
    events["taker_side_bp"] = events.symbol.map(lambda s: 5.0 + es.LOW_SLIPPAGE.get(s, 2.0))
    return events


def clustered(values: np.ndarray, times: pd.Series, lags: int = 0) -> dict:
    day = times.dt.floor("D")
    codes = ((day - day.min()) / pd.Timedelta(days=1)).astype(int).to_numpy()
    ok = np.isfinite(values)
    n_days = int(codes.max()) + 1
    stat = es.cluster_mean(np.bincount(codes[ok], minlength=n_days).astype(float), np.bincount(codes[ok], weights=values[ok], minlength=n_days), lags)
    v = values[ok]
    return {"n": stat["n"], "days": stat["days"], "mean_bp": stat["mean"] * 1e4, "t_day": stat["t"], "median_bp": float(np.median(v)) * 1e4, "hit": float((v > 0).mean())}


def resting_orders(events: pd.DataFrame) -> list[dict]:
    rows = []
    for name, h, long_only in (("move5_abs2pct_dn", 15, True), ("move1_z8_dn", 15, True), ("move1_z3_dn", 15, True), ("fund_post5_big", 5, False)):
        e = events[events.definition == name].dropna(subset=[f"f{h}", "o1", "minlow1", "minlow5", "maxhigh1", "maxhigh5"])
        side = np.ones(len(e)) if long_only else e["side"].to_numpy()
        gross = e[f"f{h}"].to_numpy() * side * 1e4
        exit_price = e["o1"].to_numpy() * (1.0 + e[f"f{h}"].to_numpy())
        taker_side = e["taker_side_bp"].to_numpy()
        base = {"definition": name, "h": h, "signals": int(len(e)), "taker_gross_bp": float(gross.mean()), "taker_net_bp": float((gross - 2 * taker_side).mean())}
        for alive in (1, 5):
            for depth in (0.0, 10.0):
                limit = e["o1"].to_numpy() * (1.0 - side * depth / 1e4)
                through = np.where(side > 0, e[f"minlow{alive}"].to_numpy() < limit, e[f"maxhigh{alive}"].to_numpy() > limit)
                filled_gross = side * (exit_price / limit - 1.0) * 1e4
                net = np.where(through, filled_gross - (2.0 + taker_side), 0.0)  # maker entry, taker exit; unfilled signals earn nothing
                rows.append({**base, "order_alive_min": alive, "bp_below_first_print": depth, "fill_rate": float(through.mean()),
                             "gross_bp_if_filled": float(filled_gross[through].mean()) if through.any() else float("nan"),
                             "gross_bp_of_unfilled_signals": float(gross[~through].mean()) if (~through).any() else float("nan"),
                             "net_bp_per_signal": float(net.mean())})
    return rows


def cascade_anatomy(events: pd.DataFrame) -> dict:
    e = events[events.definition == "move5_abs2pct_dn"].sort_values(["symbol", "time"]).reset_index(drop=True)
    gap = e.groupby("symbol")["time"].diff().dt.total_seconds() / 60.0
    first = (gap.isna() | (gap > 60)).to_numpy()
    out = {"all": clustered(e.f15.to_numpy(), e.time)}
    out["first event in 60 minutes (same coin)"] = clustered(e.f15.to_numpy()[first], e.time[first])
    out["later legs of the same cascade"] = clustered(e.f15.to_numpy()[~first], e.time[~first])
    taken = np.zeros(len(e), dtype=bool)
    for _, group in e.groupby("symbol"):
        busy_until = pd.Timestamp("1970-01-01", tz="UTC")
        for i, stamp in zip(group.index, group.time):
            if stamp >= busy_until:
                taken[i] = True
                busy_until = stamp + pd.Timedelta(minutes=16)
    out["one position at a time per coin"] = {**clustered(e.f15.to_numpy()[taken], e.time[taken]), "share_of_events_taken": float(taken.mean())}
    buckets = pd.cut(e.ret5 * 1e4, [-1e9, -500, -300, -250, -200], labels=["below -500 bp", "-500 to -300 bp", "-300 to -250 bp", "-250 to -200 bp"])
    out["by size of the 5-minute fall"] = {str(label): clustered(e.f15.to_numpy()[(buckets == label).to_numpy()], e.time[(buckets == label).to_numpy()])
                                           for label in buckets.cat.categories}
    block = e.time.dt.floor("15min")
    together = block.map(e.groupby(block)["symbol"].nunique())
    out["by number of coins falling in the same quarter hour"] = {
        "1 coin": clustered(e.f15.to_numpy()[(together == 1).to_numpy()], e.time[(together == 1).to_numpy()]),
        "2-3 coins": clustered(e.f15.to_numpy()[together.isin([2, 3]).to_numpy()], e.time[together.isin([2, 3]).to_numpy()]),
        "4-6 coins": clustered(e.f15.to_numpy()[(together >= 4).to_numpy()], e.time[(together >= 4).to_numpy()]),
    }
    net = e.f15 * 1e4 - 2 * e.taker_side_bp
    out["single trades net of taker costs"] = {"sd_bp": float(net.std()), "worst_bp": float(net.min()), "best_bp": float(net.max()),
                                               "share_below_minus_500bp": float((net < -500).mean()), "share_above_plus_500bp": float((net > 500).mean())}
    return out


def cost_stress(segment: str) -> list[dict]:
    runs = an.load(segment, "um", PERPS)
    j = an.H_INDEX[15]
    name = "move5_abs2pct_dn"
    n = sum(run.block(name)[0, j] for run in runs)
    gross = sum(-run.block(name)[1, j] for run in runs) * 1e4  # trade side is the opposite of the hypothesised one
    cost = sum(run.block(name)[0, j] * es.round_trip_cost_bp(run.symbol, "perp_taker") for run in runs)
    rows = []
    for label, extra in (("gross", None), ("taker costs of the study", 0.0), ("+10 bp per round trip", 10.0), ("+20 bp per round trip", 20.0), ("+30 bp per round trip", 30.0)):
        x = gross if extra is None else gross - cost - extra * n
        cum = np.cumsum(x)
        rows.append({
            "segment": segment,
            "costs": label,
            "events": int(n.sum()),
            "event_days": int((n > 0).sum()),
            "calendar_days": int(len(n)),
            "bp_per_event": float(x.sum() / n.sum()),
            "annualised_sharpe_of_daily_sums": float(x.mean() / x.std(ddof=1) * np.sqrt(365.0)),
            "share_of_event_days_positive": float(((x > 0) & (n > 0)).sum() / (n > 0).sum()),
            "worst_day_bp_units": float(x.min()),
            "max_drawdown_bp_units": float((cum - np.maximum.accumulate(cum)).min()),
            "max_events_in_one_day": int(n.max()),
        })
    return rows


def main() -> None:
    events = load_events("design")
    out = {
        "resting_orders_design": resting_orders(events),
        "cascade_anatomy_design": cascade_anatomy(events),
        "cost_stress": cost_stress("design") + cost_stress("validation"),
    }
    (an.OUT / "design_extras.json").write_text(json.dumps(out, indent=1, default=float))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(pd.DataFrame(out["resting_orders_design"]).round(2).to_string(index=False))
    for key, value in out["cascade_anatomy_design"].items():
        print(key, json.dumps(value, default=float)[:400])
    print(pd.DataFrame(out["cost_stress"]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
