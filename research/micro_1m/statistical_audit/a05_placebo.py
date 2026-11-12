"""Audit step 5a: placebo tests. Event times are moved; the forward return is re-read at the moved bar.

    python -I a05_placebo.py design|validation

For each selected pair (six perpetuals, auditor's own flags and forward returns):
    real          the events themselves
    hours         every event moved by a random offset of +-(2..6) hours                    ("a few hours")
    days          every event moved by +-(1..7) whole days: identical hour-of-day distribution
    same_day      a random minute of the same coin and UTC day
    near_before   20..60 minutes BEFORE the event (inside the same episode, before the trigger)
    near_after    20..60 minutes AFTER the event
"raw" keeps every placebo bar; "clean" drops placebo bars that sit at or within 20 minutes AFTER a real event of the
same definition and coin (they carry the effect itself; the exclusion is causal). 200 draws each; the table shows the
mean over draws of the pooled mean, the 2.5% / 97.5% quantiles over draws, and the day-clustered t of one draw.
The trade side of the event is carried along (always long for the two falls; the sign of the funding rate
for the funding pair).

Design only, as a volatility-matched control for the headline pair (no validation statistic of a mirror is
computed, in line with the study's plan): a LONG position after a RISE of >= 2% in 5 minutes, and after a
"big range, no direction" bar (5-minute high-low range >= 3% of price, |5-minute return| < 1%).
Writes out/placebo_<segment>.json.
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
DRAWS = 200
PAIRS = [("move5_abs2pct_dn", 15), ("move1_z3_dn", 15), ("fund_post5_big", 5)]


def just_after_event(mask: np.ndarray, radius: int) -> np.ndarray:
    """True at a flagged bar and for the ``radius`` bars after it (causal: uses bars <= i only).

    An exclusion that also dropped the bars BEFORE an event would condition the placebo on "no fall in the next
    20 minutes", i.e. on the future, and bias it upwards (tried first: +17 bp instead of +5 bp on design)."""
    c = np.concatenate([[0], np.cumsum(mask)])
    i = np.arange(len(mask))
    lo = np.maximum(i - radius, 0)
    return (c[i + 1] - c[lo]) > 0


def main(argv):
    segment = argv[0]
    rng = np.random.default_rng(20261112)
    # per pair: lists over symbols of (values per draw x kind)
    acc = {name: {} for name, _ in PAIRS}
    one_draw = {name: {} for name, _ in PAIRS}
    controls = {"long after a 2% RISE in 5 min": [], "long after a big-range, no-direction 5 min": []}
    t00 = time.time()
    for symbol in al.PERPS:
        b = al.Bars("um", symbol, segment)
        falls, rises = al.flag_move5_abs2pct(b)
        dips, _ = al.flag_move1_z3(b)
        fmask, fside = al.flag_fund_post5_big(b, al.funding_series(symbol, segment))
        flags = {"move5_abs2pct_dn": (falls, np.ones(b.n)), "move1_z3_dn": (dips, np.ones(b.n)), "fund_post5_big": (fmask, fside)}
        day_first = np.searchsorted(b.day, b.day, side="left")
        day_last = np.searchsorted(b.day, b.day, side="right") - 1
        for name, h in PAIRS:
            mask, side = flags[name]
            f = b.fwd(h)
            f_in = np.where(b.inside, f, np.nan)
            ev = np.flatnonzero(mask & b.inside)
            s = side[ev]
            close_to_event = just_after_event(mask, 20)
            n_ev = len(ev)
            kinds = {
                "hours": lambda: ev[None, :] + rng.choice([-1, 1], size=(DRAWS, n_ev)) * rng.integers(120, 361, size=(DRAWS, n_ev)),
                "days": lambda: ev[None, :] + rng.choice([-1, 1], size=(DRAWS, n_ev)) * rng.integers(1, 8, size=(DRAWS, n_ev)) * 1440,
                "same_day": lambda: day_first[ev][None, :] + (rng.random((DRAWS, n_ev)) * (day_last[ev] - day_first[ev] + 1)[None, :]).astype(np.int64),
                "near_before": lambda: ev[None, :] - rng.integers(20, 61, size=(DRAWS, n_ev)),
                "near_after": lambda: ev[None, :] + rng.integers(20, 61, size=(DRAWS, n_ev)),
            }
            real = f_in[ev] * s
            store = acc[name].setdefault("real", {"sum": 0.0, "n": 0})
            store["sum"] += float(np.nansum(real))
            store["n"] += int(np.isfinite(real).sum())
            one_draw[name].setdefault("real", []).append(pd.DataFrame({"x": real, "day": b.day[ev]}))
            for kind, draw in kinds.items():
                idx = draw()
                valid = (idx >= 0) & (idx < b.n)
                idx = np.clip(idx, 0, b.n - 1)
                x = np.where(valid, f_in[idx] * s[None, :], np.nan)
                clean = np.where(close_to_event[idx], np.nan, x)
                for label, arr in ((kind + "/raw", x), (kind + "/clean", clean)):
                    st = acc[name].setdefault(label, {"sum": np.zeros(DRAWS), "n": np.zeros(DRAWS)})
                    st["sum"] += np.nansum(arr, axis=1)
                    st["n"] += np.isfinite(arr).sum(axis=1)
                    one_draw[name].setdefault(label, []).append(pd.DataFrame({"x": arr[0], "day": b.day[idx[0]]}))
        if segment == "design":
            f15 = np.where(b.inside, b.fwd(15), np.nan)
            ev = np.flatnonzero(rises & b.inside)
            controls["long after a 2% RISE in 5 min"].append(pd.DataFrame({"x": f15[ev], "day": b.day[ev], "year": b.year[ev]}))
            # big range without direction: max high / min low of the last 5 bars, small net move; thinned like the events
            hi5 = pd.Series(b.h).rolling(5, min_periods=5).max().to_numpy()
            lo5 = pd.Series(b.l).rolling(5, min_periods=5).min().to_numpy()
            r5 = al.ret_k(b, 5)
            with np.errstate(invalid="ignore"):
                wild = b.known_ok & ((hi5 / lo5 - 1.0) >= 0.03) & (np.abs(r5) < 0.01)
            ev = np.flatnonzero(al.thin(wild, 5) & b.inside)
            controls["long after a big-range, no-direction 5 min"].append(pd.DataFrame({"x": f15[ev], "day": b.day[ev], "year": b.year[ev]}))
        print(symbol, f"{time.time() - t00:.0f}s", flush=True)

    out = {}
    for name, h in PAIRS:
        real = acc[name]["real"]
        real_mean = real["sum"] / real["n"] * 1e4
        r1 = pd.concat(one_draw[name]["real"])
        rt = al.cluster_stat(r1.x.to_numpy(), r1.day.to_numpy())
        print(f"\n=== {segment}: {name} @ {h}: real {real_mean:.2f} bp over {real['n']} events (t_day {rt['t']:.2f}) ===")
        out[name] = {"real_bp": real_mean, "real_n": real["n"], "real_t_day": rt["t"], "placebo": {}}
        for label, st in acc[name].items():
            if label == "real":
                continue
            means = st["sum"] / np.maximum(st["n"], 1) * 1e4
            d1 = pd.concat(one_draw[name][label])
            t1 = al.cluster_stat(d1.x.to_numpy(), d1.day.to_numpy())
            row = {"mean_bp": float(means.mean()), "q025_bp": float(np.quantile(means, 0.025)), "q975_bp": float(np.quantile(means, 0.975)), "max_bp": float(means.max()),
                   "avg_n": float(st["n"].mean()), "one_draw_mean_bp": t1["mean"] * 1e4, "one_draw_t_day": t1["t"], "draws_ge_real": int((means >= real_mean).sum())}
            out[name]["placebo"][label] = row
            print(f"  {label:18s}: mean over draws {row['mean_bp']:7.2f} bp [{row['q025_bp']:7.2f}, {row['q975_bp']:7.2f}], max {row['max_bp']:7.2f}, n {row['avg_n']:8.0f}; "
                  f"one draw {row['one_draw_mean_bp']:7.2f} (t_day {row['one_draw_t_day']:5.2f}); draws >= real: {row['draws_ge_real']}/{DRAWS}")
    if segment == "design":
        out["volatility_controls_design"] = {}
        for label, frames in controls.items():
            d = pd.concat(frames)
            d = d[np.isfinite(d.x)]
            st = al.cluster_stat(d.x.to_numpy(), d.day.to_numpy())
            by_year = {int(y): (int(len(p)), float(p.x.mean()) * 1e4, al.cluster_stat(p.x.to_numpy(), p.day.to_numpy())["t"]) for y, p in d.groupby("year")}
            out["volatility_controls_design"][label] = {"n": st["n"], "mean_bp": st["mean"] * 1e4, "t_day": st["t"], "by_year": by_year}
            print(f"\ncontrol (design): {label}: n {st['n']}, mean {st['mean'] * 1e4:.2f} bp, t_day {st['t']:.2f}; by year " +
                  "; ".join(f"{y}: {v[1]:.1f} (t {v[2]:.1f}, n {v[0]})" for y, v in by_year.items()))
    (OUT / f"placebo_{segment}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1:])
