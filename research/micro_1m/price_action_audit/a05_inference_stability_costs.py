"""Audit checks 4, 6 and 7 on my own event tables (six perpetuals pooled).

    python -I a05_inference_stability_costs.py

4. Inference: day clusters (plain, and with the adjacent-day term the researcher adds), 7-day and calendar-month
   clusters, moving-block bootstrap over days with blocks of 1, 5, 10 and 20 days; excess, gross, gross minus the
   taker round trip, gross minus 4 bp. Concentration of the result in few days. Multiple-look thresholds.
6. Stability: by year, by coin, by side, and with the fill delayed by one more minute.
7. Costs: net at the stated costs, break-even cost per round trip and per side, resting-order simulation.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_lib as al  # noqa: E402

PAIRS = [("sweep_N240_x10_k1", 240), ("sweep_N240_x10_k1", 60), ("sweep_N30_x2_k3", 5)]
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def prep(segment: str, market: str = "um", tag: str = "") -> pd.DataFrame:
    ev = pd.read_parquet(os.path.join(al.OUT, f"events_{segment}_{market}{tag}.parquet"))
    ev["day"] = ((ev["ts"] - EPOCH) // pd.Timedelta(days=1)).astype(np.int64)
    ev["week"] = ev["day"] // 7
    ev["month"] = ev["ts"].dt.year * 12 + ev["ts"].dt.month
    ev["year"] = ev["ts"].dt.year
    ev["cost"] = ev["symbol"].map(lambda s: al.taker_round_trip_bp(s, market))
    ev["slip"] = ev["symbol"].map(al.slip_bp)
    ev["segment"] = segment
    return ev


def inference_block(part: pd.DataFrame, h: int, n_boot: int = 4000) -> dict:
    p = part[np.isfinite(part[f"g{h}"]) & np.isfinite(part[f"b{h}"])]
    g, b, cost = p[f"g{h}"].to_numpy(), p[f"b{h}"].to_numpy(), p["cost"].to_numpy()
    day, week, month = p["day"].to_numpy(), p["week"].to_numpy(), p["month"].to_numpy()
    series = {"excess": g - b, "gross": g, "net_taker": g - cost, "net_maker4": g - al.MAKER_ROUND_TRIP_BP}
    out = {"events": int(len(p)), "days": int(len(np.unique(day))), "weeks": int(len(np.unique(week))), "months": int(len(np.unique(month)))}
    for name, v in series.items():
        d0 = al.cluster_stats(v, day)
        d1 = al.cluster_stats(v, day, adjacent=True)
        w = al.cluster_stats(v, week)
        m = al.cluster_stats(v, month)
        rec = {"mean": d0["mean"], "t_naive": d0["t_naive"], "t_day": d0["t"], "t_day_adjacent": d1["t"], "t_week": w["t"], "t_month": m["t"],
               "se_day_adjacent": d1["se"], "se_week": w["se"], "se_month": m["se"]}
        for block in (1, 5, 10, 20):
            bs = al.day_block_bootstrap(v, day, block, n_boot=n_boot)
            rec[f"boot{block}_se"] = bs["se"]
            rec[f"boot{block}_t"] = bs["t"]
            rec[f"boot{block}_ci"] = [bs["lo"], bs["hi"]]
            rec[f"boot{block}_p_le_0"] = bs["p_le_0"]
        rec["t_most_conservative"] = float(np.nanmin(np.abs([rec["t_day"], rec["t_day_adjacent"], rec["t_week"], rec["t_month"],
                                                             rec["boot5_t"], rec["boot10_t"], rec["boot20_t"]])) * np.sign(rec["mean"]))
        out[name] = rec
    # concentration
    by = p.groupby("day")[f"g{h}"].agg(["sum", "count", "mean"])
    n = by["count"].to_numpy(dtype=float)
    out["effective_days"] = float(n.sum() ** 2 / (n ** 2).sum())
    out["max_events_in_a_day"] = int(n.max())
    out["top10_days_share_of_events"] = float(np.sort(n)[-10:].sum() / n.sum())
    out["median_event_bp"] = float(np.median(g))
    out["mean_of_daily_means"] = float(by["mean"].mean())
    order = by["sum"].sort_values()
    for k in (5, 10, 20):
        keep = order.index[k:-k]
        out[f"mean_dropping_{k}_best_and_{k}_worst_days"] = float(by.loc[keep, "sum"].sum() / by.loc[keep, "count"].sum())
        keep = order.index[:-k]
        out[f"mean_dropping_{k}_best_days"] = float(by.loc[keep, "sum"].sum() / by.loc[keep, "count"].sum())
    lo, hi = np.percentile(g, [1, 99])
    out["mean_winsorised_1_99"] = float(np.clip(g, lo, hi).mean())
    return out


def split_rows(ev: pd.DataFrame, def_id: str, h: int) -> list[dict]:
    rows = []
    sub = ev[ev["def_id"] == def_id]
    for label, col in (("all", None), ("year", "year"), ("symbol", "symbol"), ("side", "dir")):
        groups = [("all", sub)] if col is None else sub.groupby(col)
        for value, chunk in groups:
            p = chunk[np.isfinite(chunk[f"g{h}"]) & np.isfinite(chunk[f"b{h}"])]
            if len(p) == 0:
                continue
            adj = h >= 240
            g = al.cluster_stats(p[f"g{h}"].to_numpy(), p["day"].to_numpy(), adjacent=adj)
            e = al.cluster_stats((p[f"g{h}"] - p[f"b{h}"]).to_numpy(), p["day"].to_numpy(), adjacent=adj)
            w = al.cluster_stats((p[f"g{h}"] - p[f"b{h}"]).to_numpy(), p["week"].to_numpy())
            q = chunk[np.isfinite(chunk[f"d{h}"]) & np.isfinite(chunk[f"b{h}"])]
            gd = al.cluster_stats(q[f"d{h}"].to_numpy(), q["day"].to_numpy(), adjacent=adj)
            ed = al.cluster_stats((q[f"d{h}"] - q[f"b{h}"]).to_numpy(), q["day"].to_numpy(), adjacent=adj)
            name = value if label != "side" else ("long" if value > 0 else "short")
            cost = float(p["cost"].mean())
            rows.append(dict(def_id=def_id, h=h, by=label, value=str(name), events=g["n"], days=g["clusters"], gross=g["mean"], excess=e["mean"],
                             t_excess_day=e["t"], t_excess_week=w["t"], t_gross=g["t"], cost_taker=cost, net_taker=g["mean"] - cost,
                             t_net_taker=(g["mean"] - cost) / g["se"], net_maker4=g["mean"] - 4.0, t_net_maker4=(g["mean"] - 4.0) / g["se"],
                             breakeven_round_trip=g["mean"], breakeven_per_side=g["mean"] / 2.0,
                             delayed_events=gd["n"], gross_delayed_1min=gd["mean"], excess_delayed_1min=ed["mean"], t_excess_delayed=ed["t"]))
    return rows


def maker_block(part: pd.DataFrame, h: int) -> dict:
    """Resting order at the event bar's close for 5 minutes, strict trade-through (my own simulation)."""
    adj = h >= 240
    all_ok = part[np.isfinite(part[f"g{h}"])]
    filled = part[(part["mk_off"] > 0) & part["mk_ctx"] & np.isfinite(part[f"mk_g{h}"])].copy()
    filled["fday"] = ((filled["mk_ts"] - EPOCH) // pd.Timedelta(days=1)).astype(np.int64)
    g = filled[f"mk_g{h}"].to_numpy()
    st = al.cluster_stats(g, filled["fday"].to_numpy(), adjacent=adj)
    wk = al.cluster_stats(g, (filled["fday"] // 7).to_numpy())
    mo = al.cluster_stats(g, filled["month"].to_numpy())
    b5 = al.day_block_bootstrap(g, filled["fday"].to_numpy(), 5)
    b10 = al.day_block_bootstrap(g, filled["fday"].to_numpy(), 10)
    slip = float(filled["slip"].mean())
    mark = part[(part["mk_off"] > 0) & part["mk_ctx"] & np.isfinite(part["mk_g0"])]["mk_g0"].mean()
    later = part[(part["mk_off"] > 0) & np.isfinite(part[f"g{h}"])][f"g{h}"].mean()
    never = part[(part["mk_off"] == 0) & np.isfinite(part[f"g{h}"])][f"g{h}"].mean()
    in_out = st["mean"] - (2.0 + 5.0 + slip)
    se_cons = max(st["se"], wk["se"], mo["se"], b5["se"], b10["se"])
    return dict(events=int(len(all_ok)), fills=int(st["n"]), fill_rate=float(st["n"] / len(all_ok)), mark_at_next_open=float(mark),
                taker_gross_if_later_filled=float(later), taker_gross_if_never_filled=float(never),
                maker_gross_per_fill=st["mean"], se_day=st["se"], t_gross_day=st["t"],
                maker_net_4bp=st["mean"] - 4.0, t_maker_net_4bp_day=(st["mean"] - 4.0) / st["se"],
                t_maker_net_4bp_week=(st["mean"] - 4.0) / wk["se"], t_maker_net_4bp_month=(st["mean"] - 4.0) / mo["se"],
                t_maker_net_4bp_boot5=(st["mean"] - 4.0) / b5["se"], t_maker_net_4bp_boot10=(st["mean"] - 4.0) / b10["se"],
                t_maker_net_4bp_most_conservative=(st["mean"] - 4.0) / se_cons,
                maker_in_taker_out=in_out, t_maker_in_taker_out_day=in_out / st["se"], t_maker_in_taker_out_most_conservative=in_out / se_cons,
                maker_per_event=st["mean"] * st["n"] / len(all_ok))


def main() -> None:
    des, val = prep("design"), prep("validation")
    both = pd.concat([des, val], ignore_index=True)
    res = {"inference": {}, "maker": {}}
    rows = []
    for def_id, h in PAIRS:
        for name, ev in (("design", des), ("validation", val), ("both", both)):
            part = ev[ev["def_id"] == def_id]
            res["inference"][f"{def_id}@{h}|{name}"] = inference_block(part, h)
            res["maker"][f"{def_id}@{h}|{name}"] = maker_block(part, h)
            for r in split_rows(ev, def_id, h):
                r["segment"] = name
                rows.append(r)
    # multiple looks
    looks = {"design_grid": 420, "design_grid_plus_spot_and_paxg_tables": 1260, "validation_pairs": 3}
    res["bonferroni_two_sided_5pct_z"] = {k: float(sps.norm.isf(0.025 / v)) for k, v in looks.items()}
    with open(os.path.join(al.OUT, "inference.json"), "w") as handle:
        json.dump(res, handle, indent=1)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(al.OUT, "stability_costs.csv"), index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print("BONFERRONI z thresholds:", res["bonferroni_two_sided_5pct_z"])
    for key, block in res["inference"].items():
        print(f"\n=== {key}: events {block['events']}, days {block['days']}, weeks {block['weeks']}, months {block['months']}, effective days "
              f"{block['effective_days']:.0f}, max events/day {block['max_events_in_a_day']}, top-10 days share {block['top10_days_share_of_events']:.3f}")
        t = pd.DataFrame({k: {kk: vv for kk, vv in block[k].items() if not kk.endswith("_ci")} for k in ("excess", "gross", "net_taker", "net_maker4")}).T
        print(t[["mean", "t_naive", "t_day", "t_day_adjacent", "t_week", "t_month", "boot1_t", "boot5_t", "boot10_t", "boot20_t", "t_most_conservative"]]
              .to_string(float_format=lambda v: f"{v:.2f}"))
        print("   excess CI boot5:", np.round(block["excess"]["boot5_ci"], 2), " boot10:", np.round(block["excess"]["boot10_ci"], 2),
              " gross CI boot10:", np.round(block["gross"]["boot10_ci"], 2))
        print("   median event %.2f, mean of daily means %.2f, winsorised(1-99) %.2f, drop 10 best + 10 worst days %.2f, drop 20+20 %.2f, drop 10 best only %.2f" % (
            block["median_event_bp"], block["mean_of_daily_means"], block["mean_winsorised_1_99"], block["mean_dropping_10_best_and_10_worst_days"],
            block["mean_dropping_20_best_and_20_worst_days"], block["mean_dropping_10_best_days"]))
    print("\nMAKER SIMULATION (own)")
    print(pd.DataFrame(res["maker"]).T.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nSTABILITY / COSTS")
    cols = ["segment", "def_id", "h", "by", "value", "events", "gross", "excess", "t_excess_day", "t_excess_week", "cost_taker", "net_taker", "t_net_taker",
            "net_maker4", "t_net_maker4", "breakeven_per_side", "gross_delayed_1min", "excess_delayed_1min", "t_excess_delayed"]
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
