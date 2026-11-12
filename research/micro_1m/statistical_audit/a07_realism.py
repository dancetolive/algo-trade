"""Audit steps 6-7: stability, costs and what an account with limited capital would have kept.

    python -I a07_realism.py

Headline pair only (buy after a fall of >= 2% in 5 minutes, hold 15 minutes), auditor's own event tables,
design and validation. Everything below the "selected pair" line is a post-hoc diagnostic of the SAME events
(no new definition is selected); the validation figures of these splits are diagnostics, not confirmations.

1. costs: net per event under the stated taker costs, extra round-trip cost of 5..40 bp, break-even cost,
   and the day-clustered t of each net figure;
2. bar-based proxies of the spread in the entry minute (close -> next open step, range of the entry bar);
3. tail of single trades;
4. anatomy: first fall in 60 minutes vs later legs; events per day; size of the fall;
5. capital: one position at a time per coin; a cap of K concurrent positions over all coins
   (first come first served, ties in random order), return on capital with one unit = capital / K;
6. events lost to the gap rule.
Writes out/realism.json.
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

OUT = HERE / "out"
NAME, H = "move5_abs2pct_dn", 15
YEARS = {"design": 4.0, "validation": 1.5}


def cl(x, day):
    return al.cluster_stat(np.asarray(x, dtype=float), np.asarray(day))


def main():
    res = {}
    pd.set_option("display.width", 250)
    for segment in ("design", "validation"):
        ev = pd.read_parquet(OUT / f"events_{segment}_um.parquet")
        ev = ev[ev.name == NAME].copy()
        ev["time"] = ev["time"].dt.as_unit("ns")
        lost = ev[~np.isfinite(ev[f"f{H}"])]
        e = ev[np.isfinite(ev[f"f{H}"])].sort_values(["time", "symbol"]).reset_index(drop=True)
        e["day"] = e["time"].dt.floor("D").astype("int64") // al.DAY_NS
        e["x"] = e[f"f{H}"]
        e["cost"] = e["symbol"].map(al.taker_round_trip_bp) / 1e4
        out = {}
        print(f"\n################ {segment}: {NAME} @ {H} min, {len(e)} events on {e.day.nunique()} days ################")

        # 1. costs --------------------------------------------------------------------------------
        gross = cl(e.x, e.day)
        print(f"gross {gross['mean'] * 1e4:.2f} bp (t {gross['t']:.2f}); stated taker round trip {e.cost.mean() * 1e4:.2f} bp on average")
        rows = []
        for extra in (0, 5, 10, 15, 20, 25, 30, 35, 40):
            net = cl(e.x - e.cost - extra / 1e4, e.day)
            rows.append({"extra_round_trip_bp": extra, "total_round_trip_bp": e.cost.mean() * 1e4 + extra, "net_bp": net["mean"] * 1e4, "t_day": net["t"]})
        costs = pd.DataFrame(rows)
        print(costs.round(2).to_string(index=False))
        se = gross["se"]
        out["costs"] = {"gross_bp": gross["mean"] * 1e4, "se_bp": se * 1e4, "stated_cost_bp": float(e.cost.mean()) * 1e4,
                        "net_bp": (gross["mean"] - e.cost.mean()) * 1e4, "break_even_total_round_trip_bp": gross["mean"] * 1e4,
                        "break_even_extra_round_trip_bp": (gross["mean"] - e.cost.mean()) * 1e4,
                        "extra_cost_at_which_t_net_falls_to_2_bp": (gross["mean"] - e.cost.mean() - 2 * se) * 1e4,
                        "break_even_cost_per_side_bp": gross["mean"] * 1e4 / 2, "table": rows}
        print(f"break-even: total round trip {gross['mean'] * 1e4:.1f} bp = {gross['mean'] * 1e4 / 2:.1f} bp per side (stated: {e.cost.mean() * 1e4 / 2:.1f} per side); "
              f"extra round-trip cost {out['costs']['break_even_extra_round_trip_bp']:.1f} bp; net stays above 2 standard errors up to +{out['costs']['extra_cost_at_which_t_net_falls_to_2_bp']:.1f} bp")
        for label, col in (("one minute late", f"lag2_{H}"), ("two minutes late", f"lag3_{H}"), ("minute vwap", f"vwap_{H}"), ("worst print", f"worst_{H}")):
            v = ev[col].to_numpy()
            okv = np.isfinite(v)
            st = cl(v[okv], (ev["time"].dt.floor("D").astype("int64") // al.DAY_NS).to_numpy()[okv])
            cost_v = ev["symbol"].map(al.taker_round_trip_bp).to_numpy()[okv] / 1e4
            out["costs"][f"fill {label}"] = {"gross_bp": st["mean"] * 1e4, "t": st["t"], "net_bp": float((v[okv] - cost_v).mean()) * 1e4}
            print(f"  fill {label:16s}: gross {st['mean'] * 1e4:7.2f} (t {st['t']:5.2f}), net taker {float((v[okv] - cost_v).mean()) * 1e4:7.2f}")

        # 2. spread proxies -------------------------------------------------------------------------
        step = (e.o1 / e.c0 - 1.0).abs() * 1e4
        up_to_high = (e.h1 / e.o1 - 1.0) * 1e4
        down_to_low = (1.0 - e.l1 / e.o1) * 1e4
        rng1 = (e.h1 / e.l1 - 1.0) * 1e4
        proxies = pd.DataFrame({"|open[t+1]/close[t]-1| mean": step.groupby(e.symbol).mean(), "same, median": step.groupby(e.symbol).median(),
                                "entry-bar range mean": rng1.groupby(e.symbol).mean(), "entry-bar range median": rng1.groupby(e.symbol).median(),
                                "open -> high of entry bar, mean": up_to_high.groupby(e.symbol).mean(), "open -> low, mean": down_to_low.groupby(e.symbol).mean()}).reindex(al.PERPS)
        proxies.loc["ALL"] = [step.mean(), step.median(), rng1.mean(), rng1.median(), up_to_high.mean(), down_to_low.mean()]
        print("\nbar-based proxies in the entry minute (bp):")
        print(proxies.round(1).to_string())
        out["spread_proxies_bp"] = proxies.round(2).to_dict(orient="index")

        # 3. tails ------------------------------------------------------------------------------------
        net = (e.x - e.cost) * 1e4
        out["tails"] = {"sd_bp": float(net.std()), "worst_bp": float(net.min()), "best_bp": float(net.max()), "q01_bp": float(net.quantile(0.01)), "q05_bp": float(net.quantile(0.05)),
                        "q95_bp": float(net.quantile(0.95)), "share_below_minus_100bp": float((net < -100).mean()), "share_below_minus_500bp": float((net < -500).mean()),
                        "hit_rate_net": float((net > 0).mean())}
        print("\nsingle trades net of taker costs:", {k: round(v, 3) for k, v in out["tails"].items()})
        worst = e.loc[net.nsmallest(5).index, ["symbol", "time", "r5", "x"]]
        print("worst five trades:\n", worst.assign(r5=lambda d: (d.r5 * 100).round(2), x=lambda d: (d.x * 100).round(2)).to_string(index=False))

        # 4. anatomy ----------------------------------------------------------------------------------
        e = e.sort_values(["symbol", "time"]).reset_index(drop=True)
        gap = e.groupby("symbol")["time"].diff().dt.total_seconds() / 60.0
        first60 = (gap.isna() | (gap > 60)).to_numpy()
        prev20 = (gap <= 20).fillna(False).to_numpy()
        per_day = e.groupby("day")["x"].transform("size")
        anatomy = {}

        def add(label, sel):
            st = cl(e.x[sel], e.day[sel])
            netm = float((e.x - e.cost)[sel].mean()) * 1e4
            nett = cl((e.x - e.cost)[sel], e.day[sel])["t"]
            anatomy[label] = {"n": st["n"], "days": st["clusters"], "gross_bp": st["mean"] * 1e4, "t": st["t"], "net_taker_bp": netm, "t_net": nett,
                              "median_bp": float(e.x[sel].median()) * 1e4, "share_of_events": float(np.mean(sel)), "share_of_total_gross": float(e.x[sel].sum() / e.x.sum())}
            print(f"  {label:48s} n {st['n']:5d} ({np.mean(sel):4.0%}) days {st['clusters']:4d} gross {st['mean'] * 1e4:7.2f} (t {st['t']:5.2f}) net {netm:7.2f} (t {nett:5.2f}) "
                  f"median {float(e.x[sel].median()) * 1e4:6.1f}; share of total gross {e.x[sel].sum() / e.x.sum():5.0%}")

        print("\nanatomy (post-hoc splits of the same events):")
        add("all events", np.ones(len(e), dtype=bool))
        add("first fall in 60 min (same coin)", first60)
        add("later leg (a fall in the previous 60 min)", ~first60)
        add("no fall in the previous 20 min", ~prev20)
        add("a fall in the previous 20 min", prev20)
        for lo, hi, label in ((1, 1, "day with 1 event"), (2, 5, "day with 2-5 events"), (6, 20, "day with 6-20 events"), (21, 10**6, "day with more than 20 events")):
            add(label, ((per_day >= lo) & (per_day <= hi)).to_numpy())
        for lo, hi, label in ((-1, -0.05, "fall of 5% or more"), (-0.05, -0.03, "fall of 3-5%"), (-0.03, -0.025, "fall of 2.5-3%"), (-0.025, -0.0199, "fall of 2-2.5%")):
            add(label, ((e.r5 > lo) & (e.r5 <= hi)).to_numpy())
        out["anatomy"] = anatomy

        # 5. capital ----------------------------------------------------------------------------------
        print("\ncapital-constrained versions (post-hoc):")
        cap = {}
        taken = np.zeros(len(e), dtype=bool)
        for _, g in e.groupby("symbol"):
            busy = pd.Timestamp("1970-01-01", tz="UTC")
            for i, stamp in zip(g.index, g.time):
                if stamp >= busy:
                    taken[i] = True
                    busy = stamp + pd.Timedelta(minutes=H + 1)  # entered at t+1, out at t+1+H: free again for an event flagged at t+H+1
        st = cl(e.x[taken], e.day[taken])
        netv = (e.x - e.cost)[taken]
        nst = cl(netv, e.day[taken])
        cap["one position at a time per coin"] = {"trades": int(taken.sum()), "share_of_events": float(taken.mean()), "gross_bp": st["mean"] * 1e4, "t": st["t"],
                                                  "net_taker_bp": nst["mean"] * 1e4, "t_net": nst["t"], "total_net_bp_units": float(netv.sum()) * 1e4,
                                                  "share_of_all_events_net_total": float(netv.sum() / (e.x - e.cost).sum())}
        print(f"  one position at a time per coin: {taken.sum()} trades ({taken.mean():.0%} of events), gross {st['mean'] * 1e4:.2f} (t {st['t']:.2f}), "
              f"net taker {nst['mean'] * 1e4:.2f} (t {nst['t']:.2f}); keeps {netv.sum() / (e.x - e.cost).sum():.0%} of the summed net P&L of all events")
        # cap of K concurrent positions over all coins
        ee = e.sort_values("time").reset_index(drop=True)
        tmin = (ee.time.astype("int64") // al.MIN_NS).to_numpy()
        xnet = (ee.x - ee.cost).to_numpy()
        xg = ee.x.to_numpy()
        dayv = ee.day.to_numpy()
        rng = np.random.default_rng(3)
        for K in (1, 2, 3, 6, 10**6):
            sums, counts, means, gmeans, tnets = [], [], [], [], []
            for rep in range(20 if K < 10**6 else 1):
                order = np.lexsort((rng.random(len(ee)), tmin))  # by time, ties in random order
                release: list[int] = []  # minutes at which open positions end
                take = np.zeros(len(ee), dtype=bool)
                for i in order:
                    now = tmin[i]
                    release = [r for r in release if r > now]
                    if len(release) < K:
                        take[i] = True
                        release.append(now + H + 1)
                sums.append(xnet[take].sum())
                counts.append(take.sum())
                means.append(xnet[take].mean())
                gmeans.append(xg[take].mean())
                tnets.append(cl(xnet[take], dayv[take])["t"])
            label = f"cap of {K} concurrent positions" if K < 10**6 else "no cap (every event)"
            units = K if K < 10**6 else int(max_concurrent(tmin, H))
            total = float(np.mean(sums))
            cap[label] = {"trades": float(np.mean(counts)), "gross_bp": float(np.mean(gmeans)) * 1e4, "net_taker_bp": float(np.mean(means)) * 1e4, "t_net": float(np.mean(tnets)),
                          "total_net_pct_of_one_unit": total * 100, "units_of_capital": units, "return_on_capital_pct_per_year_unlevered": total * 100 / units / YEARS[segment]}
            print(f"  {label:32s}: {np.mean(counts):7.0f} trades, gross {np.mean(gmeans) * 1e4:6.2f}, net taker {np.mean(means) * 1e4:6.2f} bp (t {np.mean(tnets):5.2f}); "
                  f"sum of net = {total * 100:7.1f}% of one unit; capital = {units} units -> {total * 100 / units / YEARS[segment]:5.1f}% per year on capital, unlevered")
        out["capital"] = cap

        # 6. gap rule -----------------------------------------------------------------------------------
        out["lost_to_gap_rule"] = {"events_flagged": int(len(ev)), "without_valid_return": int(len(lost)),
                                   "rows": lost[["symbol", "time", "r5", f"nohalt_{H}"]].assign(time=lambda d: d.time.astype(str)).to_dict(orient="records")}
        print(f"\nevents flagged {len(ev)}, without a valid forward return (gap rule) {len(lost)}:", out["lost_to_gap_rule"]["rows"])
        res[segment] = out
    (OUT / "realism.json").write_text(json.dumps(res, indent=1, default=float))


def max_concurrent(tmin: np.ndarray, h: int) -> int:
    """Largest number of positions open at the same time if every event is taken."""
    starts = np.sort(tmin)
    ends = starts + h + 1
    times = np.concatenate([starts, ends])
    delta = np.concatenate([np.ones(len(starts)), -np.ones(len(ends))])
    order = np.lexsort((delta, times))  # ends before starts at the same minute
    return int(np.cumsum(delta[order]).max())


if __name__ == "__main__":
    main()
