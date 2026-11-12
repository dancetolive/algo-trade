"""Audit step 3 (extension): spot check of the design grid behind "641 looks, 187 with |t| > 3, nearly all too small".

    python -I a11_grid_spotcheck.py

A sample of rows from all five families is recomputed with the auditor's own code (definitions written from the
docstring of the study, pooled over the six perpetuals, design segment) and set against the study's table
out/tables/design_perp_pooled.csv: events, days, gross bp on the hypothesised side, matched baseline, and the
day-clustered t of the excess (with the study's Bartlett lags across days for h >= 60, re-implemented here).
Also: the resting-order figure for the 3-sigma dip and the bid-ask-bounce figures.
Writes out/grid_spotcheck.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import auditlib as al  # noqa: E402

OUT = HERE / "out"
THEIR_TABLE = HERE.parent / "statistical" / "out" / "tables" / "design_perp_pooled.csv"
N_DAYS = 1461
DAY0 = int(pd.Timestamp("2020-01-01", tz="UTC").value // al.DAY_NS)
LAGS = {1: 0, 5: 0, 15: 0, 60: 1, 240: 1, 1440: 3}
HS = (1, 5, 15, 60, 1440)


def hac_t(n_day: np.ndarray, s_day: np.ndarray, lags: int) -> tuple[float, float, int]:
    """Mean and t of a ratio mean from per-day sums on consecutive calendar days (Bartlett weights across days)."""
    total = n_day.sum()
    g = int((n_day > 0).sum())
    mean = s_day.sum() / total
    e = s_day - mean * n_day
    var = float(e @ e)
    extra = sum(2.0 * (1.0 - k / (lags + 1.0)) * float(e[k:] @ e[:-k]) for k in range(1, lags + 1))
    if var + extra > 0:
        var += extra
    var *= g / (g - 1.0)
    return float(mean), float(mean / (np.sqrt(var) / total)), g


def rolling_median_prev(x: np.ndarray, w: int, min_count: int) -> np.ndarray:
    """Median of the finite values among x[t-w .. t-1]."""
    return al.lagged(pd.Series(x).rolling(w, min_periods=min_count).median().to_numpy(), 1)


def definitions(b: al.Bars, funding: pd.Series) -> dict:
    """name -> (mask, side on the study's hypothesised side, baseline with hour?)."""
    out = {}
    ok = b.known_ok
    with np.errstate(invalid="ignore", divide="ignore"):
        for w in (1, 5, 60):
            r = al.ret_k(b, w)
            out[f"sign{w}"] = (ok & np.isfinite(r) & (r != 0), np.sign(np.nan_to_num(r)), True)
        z = al.z1(b)
        for k in (3, 5, 8):
            out[f"move1_z{k}_dn"] = (ok & (z < -k), -np.ones(b.n), True)  # hypothesis: continuation = short after a fall
            out[f"move1_z{k}_up"] = (ok & (z > k), np.ones(b.n), True)
        falls, rises = al.flag_move5_abs2pct(b)
        out["move5_abs2pct_dn"] = (falls, -np.ones(b.n), True)
        out["move5_abs2pct_up"] = (rises, np.ones(b.n), True)
        # spikes: bar t against the median of the 60 bars before it
        body = b.c - b.o
        span = b.h - b.l
        up, dn = body > 0, body < 0
        wick = np.where(up, (b.h - b.c) / span, np.where(dn, (b.c - b.l) / span, np.nan))
        wick[~(span > 0)] = np.nan
        med_q = rolling_median_prev(b.q, 60, 54)
        med_t = rolling_median_prev(b.tr, 60, 54)
        vr = np.where(med_q > 0, b.q / med_q, np.nan)
        trr = np.where(med_t > 0, b.tr / med_t, np.nan)
        out["volspike5_all_dn"] = (ok & (vr > 5) & dn, -np.ones(b.n), True)
        out["volspike5_all_up"] = (ok & (vr > 5) & up, np.ones(b.n), True)
        out["volspike5_exh_dn"] = (ok & (vr > 5) & dn & (wick >= 0.5), np.ones(b.n), True)  # exhaustion: fade the bar = long after a down bar
        out["volspike5_cont_dn"] = (ok & (vr > 5) & dn & (wick <= 0.2), -np.ones(b.n), True)
        out["trdspike10_cont_up"] = (ok & (trr > 10) & up & (wick <= 0.2), np.ones(b.n), True)
        # order flow: taker-buy share of quote volume over w minutes, z against its own trailing day (ending w bars earlier)
        for w, k in ((5, 2), (1, 3)):
            sq, cq = al.roll_sum_count(b.q, w)
            sb, cb = al.roll_sum_count(b.tb, w)
            share = np.where((cq == w) & (cb == w) & (sq > 0), sb / np.where(sq > 0, sq, 1.0), np.nan)
            s1, c1 = al.roll_sum_count(share, 1440)
            mean = np.where(c1 >= 1296, s1 / np.maximum(c1, 1), np.nan)
            sd = al.roll_std(share, 1440, 1296)
            mean, sd = al.lagged(mean, w), al.lagged(sd, w)
            zz = np.where(sd > 0, (share - mean) / sd, np.nan)
            out[f"ofi{w}_z{k}_both"] = (al.thin(ok & (np.abs(zz) > k), w), np.sign(np.nan_to_num(zz)), True)
        out["minute00"] = (ok & (b.minute_of_day % 60 == 59), np.ones(b.n), False)
        out["hour14"] = (ok & (b.minute_of_day == 14 * 60 - 1), np.ones(b.n), False)
        stamps = funding.index.as_unit("ns").asi8
        k_ = np.searchsorted(stamps, b.ns, side="right") - 1
        rate = np.where(k_ >= 0, funding.to_numpy(dtype=float)[np.maximum(k_, 0)], np.nan)
        big = np.abs(rate) >= 3e-4
        out["fund_pre5_big"] = (ok & np.isin(b.minute_of_day, [1434, 474, 954]) & big, -np.sign(np.nan_to_num(rate)), False)
    return out


ROWS = [("sign1", 1), ("sign5", 5), ("sign60", 60), ("sign60", 1440), ("ofi5_z2_both", 5), ("ofi1_z3_both", 5), ("volspike5_all_up", 5), ("volspike5_all_dn", 5),
        ("volspike5_exh_dn", 5), ("volspike5_cont_dn", 5), ("trdspike10_cont_up", 5), ("move1_z3_up", 15), ("move1_z5_dn", 15), ("move1_z8_dn", 15), ("move1_z8_up", 15),
        ("move5_abs2pct_dn", 5), ("move5_abs2pct_dn", 60), ("move5_abs2pct_dn", 1440), ("move5_abs2pct_up", 15), ("hour14", 60), ("minute00", 1), ("fund_pre5_big", 5)]


def main():
    acc = {row: np.zeros((3, N_DAYS)) for row in ROWS}  # n, sum x, sum baseline per day
    bounce = {}
    t0 = time.time()
    for symbol in al.PERPS:
        b = al.Bars("um", symbol, "design")
        d = definitions(b, al.funding_series(symbol, "design"))
        day_idx = np.clip(b.day - DAY0, 0, N_DAYS - 1)
        fwd = {h: b.fwd(h) for h in HS}
        base = {(h, hour): al.matched_baseline(b, fwd[h], hour) for h in HS for hour in (True, False) if any(r[1] == h and d[r[0]][2] == hour for r in ROWS)}
        for name, h in ROWS:
            mask, side, hour = d[name]
            rows = np.flatnonzero(mask & b.inside)
            x = fwd[h][rows] * side[rows]
            ok = np.isfinite(x)
            rows, x = rows[ok], x[ok]
            bs = base[(h, hour)][rows] * side[rows]
            a = acc[(name, h)]
            a[0] += np.bincount(day_idx[rows], minlength=N_DAYS)
            a[1] += np.bincount(day_idx[rows], weights=x, minlength=N_DAYS)
            a[2] += np.bincount(day_idx[rows], weights=bs, minlength=N_DAYS)
        # bid-ask bounce: the 1-minute reversal trade measured from the closing print and from the next open
        r1 = al.ret_k(b, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            nxt_cc = np.concatenate([b.c[1:] / b.c[:-1] - 1.0, [np.nan]])
        side = -np.sign(np.nan_to_num(r1))
        m = b.known_ok & np.isfinite(r1) & (r1 != 0) & b.inside
        for label, series in (("from_close", nxt_cc), ("from_open", fwd[1])):
            v = series[m] * side[m]
            okv = np.isfinite(v)
            st = al.cluster_stat(v[okv], b.day[m][okv])
            bounce[f"{symbol}/{label}"] = {"bp": st["mean"] * 1e4, "t": st["t"]}
        print(symbol, f"{time.time() - t0:.0f}s", flush=True)

    theirs = pd.read_csv(THEIR_TABLE).set_index(["definition", "h"])
    rows = []
    for (name, h), a in acc.items():
        gross, t_gross, days = hac_t(a[0], a[1], LAGS[h])
        _, t_ex, _ = hac_t(a[0], a[1] - a[2], LAGS[h])
        th = theirs.loc[(name, h)]
        rows.append({"definition": name, "h": h, "events": int(a[0].sum()), "events_theirs": int(th.events), "days": days, "days_theirs": int(th.days),
                     "gross_bp": gross * 1e4, "gross_theirs": float(th.gross_bp), "base_bp": float(a[2].sum() / a[0].sum()) * 1e4, "base_theirs": float(th.base_bp),
                     "t_day": t_ex, "t_theirs": float(th.t_day)})
    table = pd.DataFrame(rows)
    table["d_events"] = table.events - table.events_theirs
    table["d_gross"] = table.gross_bp - table.gross_theirs
    table["d_t"] = table.t_day - table.t_theirs
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(table.round(3).to_string(index=False))
    print(f"\nlargest |difference|: events {int(table.d_events.abs().max())}, gross {table.d_gross.abs().max():.4f} bp, t {table.d_t.abs().max():.3f}")
    print("\nbid-ask bounce (design, 1-minute reversal trade, bp per trade):")
    for k, v in bounce.items():
        print(f"  {k:22s} {v['bp']:+.3f} (t {v['t']:.1f})")

    # resting-order figure quoted for the 3-sigma dip: bid at the first print of the next minute, alive one minute,
    # filled only if the low of that minute trades THROUGH it; maker entry 2 bp, taker exit; unfilled signals earn nothing
    e = pd.read_parquet(OUT / "events_design_um.parquet")
    e = e[(e.name == "move1_z3_dn") & np.isfinite(e.f15) & np.isfinite(e.l1) & np.isfinite(e.o1)]
    taker_side = e.symbol.map(lambda s: al.taker_round_trip_bp(s) / 2.0).to_numpy()
    gross = e.f15.to_numpy() * 1e4
    through = (e.l1 < e.o1).to_numpy()
    net = np.where(through, gross - (2.0 + taker_side), 0.0)
    resting = {"signals": int(len(e)), "fill_rate": float(through.mean()), "gross_if_filled_bp": float(gross[through].mean()),
               "gross_of_unfilled_bp": float(gross[~through].mean()), "net_per_signal_bp": float(net.mean())}
    print("\nresting bid at the first print after a 3-sigma dip (design):", {k: round(v, 3) if isinstance(v, float) else v for k, v in resting.items()})
    (OUT / "grid_spotcheck.json").write_text(json.dumps({"rows": json.loads(table.to_json(orient="records")), "bounce": bounce, "resting_move1_z3_dn": resting}, indent=1, default=float))


if __name__ == "__main__":
    main()
