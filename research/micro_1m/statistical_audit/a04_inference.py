"""Audit step 4: inference for the selected pairs, recomputed from the auditor's own event tables.

    python -I a04_inference.py

* cluster-robust t with clusters = calendar day, ISO week, calendar month (events of one crash span days);
* day-block bootstrap (blocks of 1, 5 and 20 consecutive calendar days, empty days included);
* sign-flip (randomisation) test on day sums and a binomial test on the share of positive event days;
* robust location: median, 1% and 5% trimmed means, mean of per-day means (each event day one vote);
* the multiple-testing bar for the number of looks that were taken.
Writes out/inference.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import auditlib as al  # noqa: E402

OUT = HERE / "out"
PAIRS = [("move5_abs2pct_dn", 15), ("move1_z3_dn", 15), ("fund_post5_big", 5)]
SEG_DAYS = {"design": 1461, "validation": 547}


def bonferroni_t(looks: int, alpha: float = 0.05) -> float:
    return float(sps.norm.isf(alpha / looks / 2.0))


def analyse(e: pd.DataFrame, h: int, segment: str) -> dict:
    x = (e[f"f{h}"] * e["side"]).to_numpy()
    base = (e[f"b{h}"] * e["side"]).to_numpy()
    ok = np.isfinite(x)
    e, x, base = e[ok], x[ok], base[ok]
    t = e["time"]
    day = (t.dt.floor("D").astype("int64") // al.DAY_NS).to_numpy()
    week = ((day + 3) // 7)  # 1970-01-01 was a Thursday: +3 puts the week boundary on Monday
    month = (t.dt.year * 12 + t.dt.month).to_numpy()
    cost = e["symbol"].map(al.taker_round_trip_bp).to_numpy() / 1e4
    out = {"events": int(len(x)), "gross_bp": float(x.mean()) * 1e4, "excess_bp": float((x - base).mean()) * 1e4, "net_bp": float((x - cost).mean()) * 1e4}
    for label, cl in (("day", day), ("week", week), ("month", month)):
        g, ex, net = al.cluster_stat(x, cl), al.cluster_stat(x - base, cl), al.cluster_stat(x - cost, cl)
        out[f"cluster_{label}"] = {"clusters": g["clusters"], "t_gross": g["t"], "t_excess": ex["t"], "t_net_taker": net["t"]}
    out["t_naive_excess"] = al.naive_t(x - base)
    d0 = int((al.SEGMENTS[segment][2].value) // al.DAY_NS)
    for block in (1, 5, 20):
        for label, v in (("excess", x - base), ("net_taker", x - cost)):
            # days on the segment's full calendar, so that days without events are resampled as well
            b = al.day_bootstrap(v, day, n_boot=4000, block=block, seed=11 + block, n_days_total=SEG_DAYS[segment], day0=d0)
            out[f"bootstrap_block{block}_{label}"] = {"se_bp": b["se"] * 1e4, "ci95_bp": [b["lo"] * 1e4, b["hi"] * 1e4], "share_of_draws_le_0": b["p_le_0"],
                                                    "t_equiv": float(v.mean() / b["se"])}
    # randomisation: flip the sign of whole days
    s_day = pd.Series(x - base).groupby(day).sum().to_numpy()
    s_net = pd.Series(x - cost).groupby(day).sum().to_numpy()
    rng = np.random.default_rng(5)
    flips = rng.choice([-1.0, 1.0], size=(20000, len(s_day)))
    out["signflip_p_excess"] = float(((flips @ s_day) >= s_day.sum()).mean())
    out["signflip_p_net"] = float(((flips @ s_net) >= s_net.sum()).mean())
    pos = int((s_day > 0).sum())
    out["event_days"] = int(len(s_day))
    out["event_days_positive_share"] = pos / len(s_day)
    out["binomial_p_days_positive"] = float(sps.binomtest(pos, len(s_day), 0.5, alternative="greater").pvalue)
    posn = int((s_net > 0).sum())
    out["event_days_positive_share_net"] = posn / len(s_net)
    # robust location
    lo1, hi1 = np.quantile(x, [0.01, 0.99])
    lo5, hi5 = np.quantile(x, [0.05, 0.95])
    per_day_mean = pd.Series(x).groupby(day).mean()
    per_day_n = pd.Series(x).groupby(day).size()
    out["median_bp"] = float(np.median(x)) * 1e4
    out["trim1_bp"] = float(x[(x >= lo1) & (x <= hi1)].mean()) * 1e4
    out["trim5_bp"] = float(x[(x >= lo5) & (x <= hi5)].mean()) * 1e4
    out["winsor1_bp"] = float(np.clip(x, lo1, hi1).mean()) * 1e4
    out["mean_of_day_means_bp"] = float(per_day_mean.mean()) * 1e4
    out["t_of_day_means"] = float(per_day_mean.mean() / (per_day_mean.std(ddof=1) / np.sqrt(len(per_day_mean))))
    out["median_of_day_means_bp"] = float(per_day_mean.median()) * 1e4
    # concentration
    by_day = pd.Series(x).groupby(day).sum().sort_values(ascending=False)
    tot = by_day.sum()
    out["top5_days_share"] = float(by_day.head(5).sum() / tot)
    out["top20_days_share"] = float(by_day.head(20).sum() / tot)
    busiest = per_day_n.sort_values(ascending=False)
    out["events_on_10_busiest_days_share"] = float(busiest.head(10).sum() / len(x))
    out["max_events_one_day"] = int(busiest.iloc[0])
    # leave out the k best days
    order = by_day.index.to_numpy()
    for k in (5, 10, 20):
        keep = ~np.isin(day, order[:k])
        g = al.cluster_stat(x[keep], day[keep])
        out[f"without_best_{k}_days"] = {"gross_bp": g["mean"] * 1e4, "t": g["t"], "net_bp": float((x - cost)[keep].mean()) * 1e4}
    return out


def main():
    looks = {"counted by the study (630 grid rows + 11 autocorrelation statistics)": 641,
             "plus the uncounted repeats (BTC/ETH spot grid 612, PAXG grid 612, 180 execution variants, 72 resting-order variants, 4 hourly, 10 cascade subsets)": 641 + 612 + 612 + 180 + 72 + 4 + 10}
    res = {"bonferroni_abs_t": {k: {"looks": v, "abs_t_for_5pct_familywise": bonferroni_t(v)} for k, v in looks.items()},
           "validation_tests": {"pairs": 3, "abs_t_for_5pct_familywise": bonferroni_t(3)}}
    for k, v in res["bonferroni_abs_t"].items():
        print(f"looks: {v['looks']:5d} -> |t| > {v['abs_t_for_5pct_familywise']:.2f}   ({k})")
    print(f"validation: 3 pre-registered pairs -> |t| > {res['validation_tests']['abs_t_for_5pct_familywise']:.2f}")
    for segment in ("design", "validation"):
        events = pd.read_parquet(OUT / f"events_{segment}_um.parquet")
        events["time"] = events["time"].dt.as_unit("ns")
        for name, h in PAIRS:
            r = analyse(events[events.name == name], h, segment)
            res[f"{segment}/{name}"] = r
            print(f"\n=== {segment}: {name} @ {h} ===")
            print(f"  events {r['events']}, gross {r['gross_bp']:.2f}, excess {r['excess_bp']:.2f}, net taker {r['net_bp']:.2f}; naive t (excess) {r['t_naive_excess']:.1f}")
            for label in ("day", "week", "month"):
                c = r[f"cluster_{label}"]
                print(f"  clustered by {label:5s}: {c['clusters']:4d} clusters, t gross {c['t_gross']:.2f}, t excess {c['t_excess']:.2f}, t net taker {c['t_net_taker']:.2f}")
            for block in (1, 5, 20):
                a, b = r[f"bootstrap_block{block}_excess"], r[f"bootstrap_block{block}_net_taker"]
                print(f"  bootstrap, blocks of {block:2d} days: excess se {a['se_bp']:.2f} (t-equiv {a['t_equiv']:.2f}), CI {a['ci95_bp'][0]:.1f}..{a['ci95_bp'][1]:.1f}; "
                      f"net taker CI {b['ci95_bp'][0]:.1f}..{b['ci95_bp'][1]:.1f} (t-equiv {b['t_equiv']:.2f}), share of draws <= 0: {b['share_of_draws_le_0']:.4f}")
            print(f"  sign-flip p (excess) {r['signflip_p_excess']:.5f}, (net) {r['signflip_p_net']:.5f}; event days {r['event_days']}, positive {r['event_days_positive_share']:.2f} "
                  f"(binomial p {r['binomial_p_days_positive']:.1e}), positive net of costs {r['event_days_positive_share_net']:.2f}")
            print(f"  median {r['median_bp']:.1f}, trimmed 1% {r['trim1_bp']:.1f}, trimmed 5% {r['trim5_bp']:.1f}, winsorised 1% {r['winsor1_bp']:.1f}, mean of day means {r['mean_of_day_means_bp']:.1f} "
                  f"(t {r['t_of_day_means']:.2f}), median of day means {r['median_of_day_means_bp']:.1f}")
            print(f"  top 5 days {r['top5_days_share']:.2f} of total, top 20 days {r['top20_days_share']:.2f}; 10 busiest days hold {r['events_on_10_busiest_days_share']:.2f} of events (max {r['max_events_one_day']} in a day)")
            for k in (5, 10, 20):
                w = r[f"without_best_{k}_days"]
                print(f"  without the {k:2d} best days: gross {w['gross_bp']:.1f} (t {w['t']:.2f}), net taker {w['net_bp']:.1f}")
    (OUT / "inference.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
