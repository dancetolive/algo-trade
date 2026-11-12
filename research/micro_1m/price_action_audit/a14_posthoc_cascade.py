"""POST HOC, exploratory (auditor; both segments were already seen when this was written): where the per-event mean
comes from, expressed as a rule ONE account could follow.

    python -I a14_posthoc_cascade.py

Long side only. One position per coin. Enter at a sweep of the 4-hour LOW only if at least k earlier low-sweeps
occurred in that coin during the previous 240 minutes; hold 240 minutes; ignore events while in the position.
k = 0 is the plain reading of the selected pair for a single-position, long-only account.
This is NOT a finding: it was not pre-registered, neither segment is clean for it, and it is a volatility /
capitulation effect rather than a property of the 1-minute pattern. It is recorded so that a follow-up can be
pre-registered and judged once on the hold-out.
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_lib as al
import a09_overlap as ov

H = 240
res = {}
for segment in ("design", "validation"):
    ev = ov.load(segment)
    e = ev[np.isfinite(ev[f"g{H}"]) & np.isfinite(ev[f"b{H}"]) & (ev["dir"] > 0)].reset_index(drop=True)
    n_prior = np.zeros(len(e), dtype=np.int64)
    for _, idx in e.groupby("symbol").indices.items():
        m = e["minute"].to_numpy()[idx]
        n_prior[idx] = np.arange(len(m)) - np.searchsorted(m, m - 240, side="left")
    e["n_prior"] = n_prior
    e["year"] = e["ts"].dt.year
    seg = {}
    for k in (0, 1, 2, 3, 4, 6):
        s = e[ov.take_with_cap(e, H, 1, min_prior=k)]
        g = al.cluster_stats(s[f"g{H}"].to_numpy(), s["day"].to_numpy(), adjacent=True)
        x = al.cluster_stats((s[f"g{H}"] - s[f"b{H}"]).to_numpy(), s["day"].to_numpy(), adjacent=True)
        w = al.cluster_stats(s[f"g{H}"].to_numpy(), (s["day"] // 7).to_numpy())
        mth = al.cluster_stats(s[f"g{H}"].to_numpy(), s["month"].to_numpy() if "month" in s else (s["ts"].dt.year * 12 + s["ts"].dt.month).to_numpy())
        cost = float(s["cost"].mean())
        se = max(g["se"], w["se"], mth["se"])
        by = s.groupby("day")[f"g{H}"].sum().sort_values()
        tot = float(s[f"g{H}"].sum())
        seg[str(k)] = dict(trades=g["n"], days=g["clusters"], gross=g["mean"], t_gross_day=g["t"], t_gross_most_conservative=g["mean"] / se,
                           excess_over_baseline=x["mean"], t_excess=x["t"], median=float(s[f"g{H}"].median()), hit=float((s[f"g{H}"] > 0).mean()),
                           p01=float(np.percentile(s[f"g{H}"], 1)), p05=float(np.percentile(s[f"g{H}"], 5)), worst=float(s[f"g{H}"].min()),
                           cost_taker=cost, net_taker=g["mean"] - cost, t_net_taker_day=(g["mean"] - cost) / g["se"], t_net_taker_most_conservative=(g["mean"] - cost) / se,
                           net_taker_excess=x["mean"] - cost, t_net_taker_excess=(x["mean"] - cost) / x["se"],
                           share_of_total_from_10_best_days=float(by.tail(10).sum() / tot) if tot else np.nan,
                           net_taker_without_10_best_days=float((tot - by.tail(10).sum()) / (len(s) - s[s["day"].isin(by.tail(10).index)].shape[0]) - cost),
                           by_year={int(y): [int(len(c)), float(c[f"g{H}"].mean())] for y, c in s.groupby("year")},
                           by_coin={sym: [int(len(c)), float(c[f"g{H}"].mean())] for sym, c in s.groupby("symbol")})
    res[segment] = seg
json.dump(res, open(os.path.join(al.OUT, "posthoc_cascade.json"), "w"), indent=1)
pd.set_option("display.width", 250)
for segment, seg in res.items():
    print("\n", segment, "LONG only, one position per coin, entry needs >= k earlier low-sweeps in 240 min, hold 240 min")
    t = pd.DataFrame(seg).T
    print(t[["trades", "days", "gross", "t_gross_day", "t_gross_most_conservative", "excess_over_baseline", "t_excess", "median", "hit", "p01", "worst", "net_taker",
             "t_net_taker_day", "t_net_taker_most_conservative", "net_taker_excess", "t_net_taker_excess", "share_of_total_from_10_best_days", "net_taker_without_10_best_days"]]
          .astype(float).to_string(float_format=lambda v: f"{v:.2f}"))
    for k in ("3", "4"):
        print(f"   k={k} by year [trades, gross]:", {y: [n, round(g, 1)] for y, (n, g) in seg[k]["by_year"].items()})
        print(f"   k={k} by coin [trades, gross]:", {y: [n, round(g, 1)] for y, (n, g) in seg[k]["by_coin"].items()})
