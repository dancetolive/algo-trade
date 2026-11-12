"""The selected (definition, horizon) pairs, computed identically on design and on validation.

    python -I selected_report.py design
    python -I selected_report.py validation      (after run_events.py validation ... and event_level.py validation ...)

Writes out/selected_<segment>.json. All figures are on the TRADE side recorded in SELECTION.json.
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

SPOT = ["BTCUSDT", "ETHUSDT"]


def pair_stats(runs, name: str, h: int, side: int) -> dict | None:
    if not any(run.block(name) is not None for run in runs):
        return None
    row = an.pooled(runs, name, h)
    if not row or row["events"] == 0:
        return None
    out = {
        "events": row["events"],
        "days": row["days"],
        "gross_bp": side * row["gross_bp"],
        "baseline_bp": side * row["base_bp"],
        "excess_bp": side * row["excess_bp"],
        "t_day": side * row["t_day"],
        "t_naive": side * row["t_naive"],
        "hit_rate": row["hit"] if side > 0 else 1.0 - row["hit"],
        "cost_perp_taker_bp": row["cost_perp_taker_bp"],
        "net_perp_taker_bp": side * row["gross_bp"] - row["cost_perp_taker_bp"],
        "net_perp_maker_bound_bp": side * row["gross_bp"] - row["cost_perp_maker_bp"],
        "net_spot_taker_bp": side * row["gross_bp"] - row["cost_spot_taker_bp"],
    }
    # t of the net-of-taker-cost figure: same standard error as the gross mean (costs are a constant per coin)
    j = an.H_INDEX[h]
    total = sum(run.block(name) for run in runs if run.block(name) is not None)
    n_day, s_day, b_day = total[0, j], side * total[1, j], side * total[2, j]
    cost_day = sum(run.block(name)[0, j] * es.round_trip_cost_bp(run.symbol, "perp_taker") / 1e4 for run in runs if run.block(name) is not None)
    net = es.cluster_mean(n_day, s_day - cost_day, es.HAC_LAGS[h])
    out["t_net_perp_taker"] = net["t"]
    boot = es.day_block_bootstrap(n_day, s_day - b_day, block=5, n_boot=4000, seed=1)
    out["bootstrap_excess_ci95_bp"] = [boot.get("low", float("nan")) * 1e4, boot.get("high", float("nan")) * 1e4]
    out["bootstrap_p_two_sided"] = boot["p_two_sided"]
    out["by_year"] = {str(int(k)): {"events": int(v["events"]), "days": int(v["days"]), "gross_bp": side * v["gross_bp"], "excess_bp": side * v["excess_bp"],
                                    "t_day": side * v["t_day"]} for k, v in an.by_year(runs, name, h).iterrows() if v["events"] > 0}
    out["by_symbol"] = {str(k): {"events": int(v["events"]), "days": int(v["days"]), "gross_bp": side * v["gross_bp"], "excess_bp": side * v["excess_bp"],
                                 "t_day": side * v["t_day"], "net_perp_taker_bp": side * v["gross_bp"] - es.round_trip_cost_bp(str(k), "perp_taker")}
                        for k, v in an.by_symbol(runs, name, h).iterrows() if v["events"] > 0}
    return out


def execution_robustness(segment: str, market: str, symbols, name: str, h: int, side: int) -> dict | None:
    """Mean on the trade side under other fill assumptions, from the event-level table."""
    frames = []
    for symbol in symbols:
        path = an.OUT / f"events_{segment}" / f"{market}_{symbol}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return None
    events = pd.concat(frames, ignore_index=True)
    events = events[events.definition == name]
    if events.empty or f"lag2_{h}" not in events.columns:
        return None
    day = events["time"].dt.floor("D")
    codes = ((day - day.min()) / pd.Timedelta(days=1)).astype(int).to_numpy()
    n_days = int(codes.max()) + 1
    trade_side = side * events["side"].to_numpy()  # events carry the hypothesised side; `side` flips it where the data did

    def stat(values: np.ndarray) -> dict:
        ok = np.isfinite(values)
        n_day = np.bincount(codes[ok], minlength=n_days).astype(float)
        s_day = np.bincount(codes[ok], weights=values[ok], minlength=n_days)
        res = es.cluster_mean(n_day, s_day, es.HAC_LAGS[h])
        v = values[ok]
        lo, hi = np.quantile(v, [0.01, 0.99])
        return {"n": res["n"], "mean_bp": res["mean"] * 1e4, "t_day": res["t"], "median_bp": float(np.median(v)) * 1e4,
                "trimmed_1pct_mean_bp": float(v[(v >= lo) & (v <= hi)].mean()) * 1e4, "hit_rate": float((v > 0).mean())}

    out = {
        "next_open (convention)": stat(trade_side * events[f"f{h}"].to_numpy()),
        "one minute late": stat(trade_side * events[f"lag2_{h}"].to_numpy()),
        "two minutes late": stat(trade_side * events[f"lag3_{h}"].to_numpy()),
        "minute vwap to minute vwap": stat(trade_side * events[f"vwap_{h}"].to_numpy()),
    }
    # worst price of the first minute: the high for a buy, the low for a sell
    worst = np.where(trade_side > 0, events[f"hi_{h}"].to_numpy(), events[f"lo_{h}"].to_numpy())
    out["worst print of the first minute"] = stat(trade_side * worst)
    # concentration of the convention result by day
    values = trade_side * events[f"f{h}"].to_numpy()
    ok = np.isfinite(values)
    by_day = pd.Series(values[ok]).groupby(codes[ok]).sum().sort_values(ascending=False)
    total = by_day.sum()
    out["concentration"] = {
        "event_days": int(len(by_day)),
        "share_of_days_positive": float((by_day > 0).mean()),
        "top5_days_share_of_total": float(by_day.head(5).sum() / total) if total else float("nan"),
        "top20_days_share_of_total": float(by_day.head(20).sum() / total) if total else float("nan"),
    }
    return out


def main(argv: list[str]) -> None:
    segment = argv[0]
    selection = json.loads((HERE / "SELECTION.json").read_text())
    perps = an.load(segment, "um", PERPS)
    spot = an.load(segment, "spot", SPOT)
    perps_btc_eth = [run for run in perps if run.symbol in SPOT]
    report = {"segment": segment, "pairs": []}
    for item in selection["selected"]:
        name, h, side = item["name"], item["horizon"], item["trade_side"]
        entry = {
            "name": name,
            "horizon": h,
            "trade": item["trade"],
            "perp_pooled": pair_stats(perps, name, h, side),
            "perp_btc_eth": pair_stats(perps_btc_eth, name, h, side),
            "spot_btc_eth": pair_stats(spot, name, h, side),
            "execution_perp": execution_robustness(segment, "um", PERPS, name, h, side),
            "execution_spot_btc_eth": execution_robustness(segment, "spot", SPOT, name, h, side),
        }
        report["pairs"].append(entry)
        p = entry["perp_pooled"]
        print(f"\n=== {segment}: {name} at {h} min, trade side ===")
        print(f"perps pooled: events {p['events']}, days {p['days']}, gross {p['gross_bp']:.2f} bp, baseline {p['baseline_bp']:.2f}, excess {p['excess_bp']:.2f}, "
              f"t_day {p['t_day']:.2f} (naive {p['t_naive']:.1f}), hit {p['hit_rate']:.3f}, net taker {p['net_perp_taker_bp']:.2f} (t {p['t_net_perp_taker']:.2f}), "
              f"maker bound {p['net_perp_maker_bound_bp']:.2f}, spot taker {p['net_spot_taker_bp']:.2f}, bootstrap 95% CI of excess {p['bootstrap_excess_ci95_bp'][0]:.1f}..{p['bootstrap_excess_ci95_bp'][1]:.1f}")
        print("  by year:", {k: (v["events"], round(v["gross_bp"], 1), round(v["t_day"], 1)) for k, v in p["by_year"].items()})
        print("  by coin:", {k[:-4]: (v["events"], round(v["gross_bp"], 1), round(v["t_day"], 1), round(v["net_perp_taker_bp"], 1)) for k, v in p["by_symbol"].items()})
        for label in ("perp_btc_eth", "spot_btc_eth"):
            q = entry[label]
            if q:
                print(f"  {label}: events {q['events']}, gross {q['gross_bp']:.2f}, excess {q['excess_bp']:.2f}, t_day {q['t_day']:.2f}, net perp taker {q['net_perp_taker_bp']:.2f}, net spot taker {q['net_spot_taker_bp']:.2f}")
        if entry["execution_perp"]:
            for label, row in entry["execution_perp"].items():
                print("  exec:", label, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})
    (an.OUT / f"selected_{segment}.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1:])
