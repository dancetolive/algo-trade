"""Audit check 7 (continued): how much of the maker figure survives less generous assumptions.

    python -I a08_costs_realism.py

Pair: sweep_N240_x10_k1 held 240 minutes (and 60), six perpetuals, design and validation.

A. Entry-queue sensitivity: the researcher counts a fill when a later bar trades strictly through the limit (by any
   amount). Here the required trade-through is 0 (theirs), 1, 2 and 5 bp, and the order rests 1, 5 or 15 minutes.
B. Simulated passive exit instead of an assumed one: at the scheduled exit a limit order is placed at that bar's
   open, rests 5 minutes, strict trade-through; otherwise a market order at the open 5 minutes later (taker fee
   and slippage). Fees: 2 bp maker, 5 bp + slippage taker.
C. Funding actually paid or received while the position is open (Binance funding history).
D. One position per coin at a time (an account cannot take overlapping events in the same coin), and at most one
   position in total: the event-weighted mean versus what a single small account would have traded.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_lib as al  # noqa: E402
from algotrade.data import store  # noqa: E402

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
DEF = (240, 10.0, 1)
HS = (60, 240)


def first_through(h, l, limit, d, start, rest, margin_bp, n):
    """First bar in start .. start+rest-1 that trades through `limit` by more than margin (strictly); -1 if none."""
    fill = np.full(len(start), -1, dtype=np.int64)
    for j in range(rest):
        u = start + j
        inside = u < n
        uu = np.minimum(u, n - 1)
        with np.errstate(invalid="ignore"):
            if margin_bp == 0:
                thr = np.where(d > 0, l[uu] < limit, h[uu] > limit)
            else:
                thr = np.where(d > 0, l[uu] <= limit * (1 - margin_bp * 1e-4), h[uu] >= limit * (1 + margin_bp * 1e-4))
        new = (fill < 0) & thr & inside
        fill[new] = u[new]
    return fill


def one(args) -> pd.DataFrame:
    segment, symbol = args
    g, first_event = al.load_grid("um", symbol, segment)
    o, h, l, c, present = al.arrays(g)
    n = len(o)
    ctx = al.context_mask(present)
    if first_event is not None:
        ctx &= np.asarray(g.index >= first_event)
    cs = np.concatenate([[0], np.cumsum(present)])
    sig = al.sweep_signal(h, l, c, *DEF)
    t = np.flatnonzero(sig)
    t = t[ctx[t]]
    d = sig[t].astype(float)
    slip = al.slip_bp(symbol)
    cols = {"segment": segment, "symbol": symbol, "ts": g.index[t], "dir": d.astype(np.int8), "t": t}
    limit = c[t]

    def complete(a, b):  # bars a..b all present (b may exceed n)
        okb = b < n
        bb = np.where(okb, b, 0)
        return okb & ((cs[bb + 1] - cs[a]) == (bb - a + 1))

    for hh in HS:
        cols[f"taker{hh}"] = d * al.forward_return(o, present, hh)[t]
    # A. entry sensitivity (+ assumed passive exit at the open, as the researcher)
    for rest in (1, 5, 15):
        for margin in (0, 1, 2, 5):
            fill = first_through(h, l, limit, d, t + 1, rest, margin, n)
            got = fill >= 0
            f = np.where(got, fill, 0)
            got &= ctx[f]
            for hh in HS:
                x = f + 1 + hh
                okx = got & complete(f, x)
                xx = np.where(okx, x, 0)
                cols[f"mk_r{rest}_m{margin}_h{hh}"] = np.where(okx, d * (o[xx] / limit - 1.0) * 1e4, np.nan)
    # B. simulated passive exit for the researcher's entry rule (rest 5, margin 0)
    fill = first_through(h, l, limit, d, t + 1, 5, 0, n)
    got = fill >= 0
    f = np.where(got, fill, 0)
    got &= ctx[f]
    cols["fill_bar"] = np.where(got, f, -1)
    for hh in HS:
        x = f + 1 + hh                                   # scheduled exit bar: limit at its open, on the opposite side
        okx = got & complete(f, x + 5)
        xx = np.where(okx, x, 0)
        px = o[xx]
        ex = first_through(h, l, px, -d, xx, 5, 0, n)    # the exit order is a SELL for a long: trades through upward
        passive = okx & (ex >= 0)
        fallback_px = o[np.minimum(xx + 5, n - 1)]
        exit_px = np.where(passive, px, fallback_px)
        gross = d * (exit_px / limit - 1.0) * 1e4
        fee = 2.0 + np.where(passive, 2.0, 5.0 + slip)
        cols[f"sim_exit_gross{hh}"] = np.where(okx, gross, np.nan)
        cols[f"sim_exit_net{hh}"] = np.where(okx, gross - fee, np.nan)
        cols[f"sim_exit_passive{hh}"] = np.where(okx, passive, False)
        cols[f"exit_ts{hh}"] = g.index[np.where(okx, np.minimum(xx, n - 1), 0)]
    cols["entry_ts"] = g.index[np.minimum(t + 1, n - 1)]
    return pd.DataFrame(cols)


def funding_cost(ev: pd.DataFrame, hh: int) -> np.ndarray:
    """bp paid (+) or received (-) by the position: funding stamps in (entry, exit], long pays a positive rate."""
    out = np.zeros(len(ev))
    for sym, chunk in ev.groupby("symbol"):
        rates = store.load_funding([sym])
        if sym not in rates:
            continue
        r = rates[sym].dropna()
        times = r.index.as_unit("ns").asi8
        cum = np.concatenate([[0.0], np.cumsum(r.to_numpy())])
        a = np.searchsorted(times, pd.DatetimeIndex(chunk["entry_ts"]).as_unit("ns").asi8, side="right")
        b = np.searchsorted(times, (pd.DatetimeIndex(chunk["entry_ts"]) + pd.Timedelta(minutes=hh)).as_unit("ns").asi8, side="right")
        out[ev.index.get_indexer(chunk.index)] = chunk["dir"].to_numpy() * (cum[b] - cum[a]) * 1e4
    return out


def clustered(v, day, adj=True):
    s = al.cluster_stats(np.asarray(v, dtype=float), np.asarray(day), adjacent=adj)
    return s["mean"], s["se"], s["t"], s["n"]


def non_overlap(ev: pd.DataFrame, hh: int, per: str) -> np.ndarray:
    """Mask of events an account could take with one position per coin (per='symbol') or one in total (per='all')."""
    keep = np.zeros(len(ev), dtype=bool)
    order = np.argsort(ev["ts"].to_numpy(), kind="stable")
    busy_until: dict = {}
    ts = ev["ts"].to_numpy()
    sym = ev["symbol"].to_numpy()
    hold = np.timedelta64(hh + 1, "m")
    for i in order:
        k = sym[i] if per == "symbol" else "all"
        if k not in busy_until or ts[i] >= busy_until[k]:
            keep[i] = True
            busy_until[k] = ts[i] + hold
    return keep


def main() -> None:
    jobs = [(seg, s) for seg in ("design", "validation") for s in al.PERPS]
    with ProcessPoolExecutor(max_workers=3) as pool:
        parts = list(pool.map(one, jobs))
    ev = pd.concat(parts, ignore_index=True)
    ev["day"] = ((ev["ts"] - EPOCH) // pd.Timedelta(days=1)).astype(np.int64)
    ev["cost"] = ev["symbol"].map(lambda s: al.taker_round_trip_bp(s))
    ev["slip"] = ev["symbol"].map(al.slip_bp)
    res = {}
    for segment in ("design", "validation", "both"):
        e = ev if segment == "both" else ev[ev["segment"] == segment]
        e = e.reset_index(drop=True)
        block = {}
        for hh in HS:
            adj = hh >= 240
            rec = {}
            base_ok = np.isfinite(e[f"taker{hh}"])
            m, se, t, nn = clustered(e.loc[base_ok, f"taker{hh}"], e.loc[base_ok, "day"], adj)
            rec["taker_all"] = {"events": nn, "gross": m, "net": m - float(e.loc[base_ok, "cost"].mean()), "t_net": (m - float(e.loc[base_ok, "cost"].mean())) / se}
            # A
            sens = {}
            for rest in (1, 5, 15):
                for margin in (0, 1, 2, 5):
                    col = f"mk_r{rest}_m{margin}_h{hh}"
                    ok = np.isfinite(e[col])
                    m, se, t, nn = clustered(e.loc[ok, col], e.loc[ok, "day"], adj)
                    sens[f"rest{rest}_through{margin}bp"] = {"fill_rate": nn / int(base_ok.sum()), "gross_per_fill": m, "net_4bp": m - 4.0, "t_net_4bp": (m - 4.0) / se,
                                                             "net_4bp_per_event": (m - 4.0) * nn / int(base_ok.sum()),
                                                             "maker_in_taker_out": m - (7.0 + float(e.loc[ok, "slip"].mean())),
                                                             "t_maker_in_taker_out": (m - (7.0 + float(e.loc[ok, "slip"].mean()))) / se}
            rec["entry_sensitivity"] = sens
            # B
            ok = np.isfinite(e[f"sim_exit_net{hh}"])
            m, se, t, nn = clustered(e.loc[ok, f"sim_exit_net{hh}"], e.loc[ok, "day"], adj)
            gm = float(e.loc[ok, f"sim_exit_gross{hh}"].mean())
            rec["simulated_passive_exit"] = {"fills": nn, "exit_passive_share": float(e.loc[ok, f"sim_exit_passive{hh}"].mean()), "gross_per_fill": gm,
                                             "net_per_fill": m, "se": se, "t_net": t}
            # C funding (taker timing, all events)
            fc = funding_cost(e, hh)
            rec["funding_bp_paid_per_event"] = {"all": float(fc[base_ok].mean()), "long": float(fc[base_ok & (e["dir"] > 0)].mean()),
                                                "short": float(fc[base_ok & (e["dir"] < 0)].mean()),
                                                "share_events_crossing_a_funding_time": float((fc[base_ok] != 0).mean())}
            okn = np.isfinite(e[f"sim_exit_net{hh}"])
            v = e.loc[okn, f"sim_exit_net{hh}"].to_numpy() - fc[okn]
            m, se, t, nn = clustered(v, e.loc[okn, "day"], adj)
            rec["simulated_passive_exit_after_funding"] = {"net_per_fill": m, "t_net": t}
            v = e.loc[base_ok, f"taker{hh}"].to_numpy() - e.loc[base_ok, "cost"].to_numpy() - fc[base_ok]
            m, se, t, nn = clustered(v, e.loc[base_ok, "day"], adj)
            rec["taker_after_funding"] = {"net": m, "t_net": t}
            # D non-overlapping
            for per in ("symbol", "all"):
                sub = e[base_ok].reset_index(drop=True)
                keep = non_overlap(sub, hh, per)
                s2 = sub[keep]
                m, se, t, nn = clustered(s2[f"taker{hh}"], s2["day"], adj)
                cost = float(s2["cost"].mean())
                mk = s2[np.isfinite(s2[f"mk_r5_m0_h{hh}"])]
                mm, mse, mt, mn = clustered(mk[f"mk_r5_m0_h{hh}"], mk["day"], adj)
                lo = s2[s2["dir"] > 0]
                rec[f"one_position_per_{per}"] = {"events": nn, "share_of_events": nn / int(base_ok.sum()), "gross": m, "t_gross": t, "net_taker": m - cost,
                                                  "t_net_taker": (m - cost) / se, "maker_gross_per_fill": mm, "maker_net_4bp": mm - 4.0, "t_maker_net_4bp": (mm - 4.0) / mse,
                                                  "long_share": float((s2["dir"] > 0).mean()), "long_gross": float(lo[f"taker{hh}"].mean())}
            block[str(hh)] = rec
        res[segment] = block
    with open(os.path.join(al.OUT, "costs_realism.json"), "w") as handle:
        json.dump(res, handle, indent=1)
    for segment, block in res.items():
        for hh, rec in block.items():
            print(f"\n=== {segment} sweep_N240_x10_k1 @ {hh}")
            print("  taker, all events:", {k: round(v, 3) for k, v in rec["taker_all"].items()})
            print("  A. entry sensitivity (assumed passive exit, 4 bp):")
            print(pd.DataFrame(rec["entry_sensitivity"]).T.to_string(float_format=lambda v: f"{v:.3f}"))
            print("  B. simulated passive exit:", {k: round(v, 3) for k, v in rec["simulated_passive_exit"].items()})
            print("  C. funding:", {k: round(v, 3) for k, v in rec["funding_bp_paid_per_event"].items()},
                  "| B after funding:", {k: round(v, 3) for k, v in rec["simulated_passive_exit_after_funding"].items()},
                  "| taker after funding:", {k: round(v, 3) for k, v in rec["taker_after_funding"].items()})
            for per in ("symbol", "all"):
                print(f"  D. one position per {per}:", {k: round(v, 3) for k, v in rec[f"one_position_per_{per}"].items()})


if __name__ == "__main__":
    main()
