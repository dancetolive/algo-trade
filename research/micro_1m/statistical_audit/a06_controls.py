"""Audit step 5 (extra, DESIGN ONLY): is the drift after a 2% fall specific to falls, or common to every violent minute?

    python -I a06_controls.py

A long position for 15 minutes from the next open after
    fall      |5-min return| >= 2%, negative   (the selected structure)
    rise      |5-min return| >= 2%, positive   (its mirror)
    wild      5-minute high-low range >= 3% of price with |5-min return| < 1% (violent, no direction)
each split by whether the same coin had a 2% FALL event in the previous 20 minutes (a causal split), and for
falls by whether it is the first fall in 60 minutes.
No mirror statistic is computed on validation (the study's plan forbids it and the audit does not need it).
Writes out/controls_design.json.
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


def events_in_last(mask: np.ndarray, radius: int, include_self: bool) -> np.ndarray:
    """Number of flagged bars among i-radius .. i (or .. i-1)."""
    c = np.concatenate([[0], np.cumsum(mask)])
    i = np.arange(len(mask))
    lo = np.maximum(i - radius, 0)
    hi = i + 1 if include_self else i
    return c[hi] - c[lo]


def main():
    rows = {}
    frames = []
    for symbol in al.PERPS:
        b = al.Bars("um", symbol, "design")
        falls, rises = al.flag_move5_abs2pct(b)
        hi5 = pd.Series(b.h).rolling(5, min_periods=5).max().to_numpy()
        lo5 = pd.Series(b.l).rolling(5, min_periods=5).min().to_numpy()
        r5 = al.ret_k(b, 5)
        with np.errstate(invalid="ignore"):
            wild = al.thin(b.known_ok & ((hi5 / lo5 - 1.0) >= 0.03) & (np.abs(r5) < 0.01), 5)
        f15 = np.where(b.inside, b.fwd(15), np.nan)
        fall_recent = events_in_last(falls, 20, include_self=False) > 0  # a fall in the previous 20 bars (not this bar)
        fall_60 = events_in_last(falls, 60, include_self=False) > 0
        for kind, mask in (("fall", falls), ("rise", rises), ("wild", wild)):
            ev = np.flatnonzero(mask & b.inside)
            frames.append(pd.DataFrame({"kind": kind, "symbol": symbol, "x": f15[ev], "day": b.day[ev], "year": b.year[ev],
                                        "after_fall_20": fall_recent[ev], "after_fall_60": fall_60[ev], "r5": r5[ev]}))
    d = pd.concat(frames, ignore_index=True)
    d = d[np.isfinite(d.x)]

    def show(label, part):
        st = al.cluster_stat(part.x.to_numpy(), part.day.to_numpy())
        by_year = {int(y): round(float(p.x.mean()) * 1e4, 1) for y, p in part.groupby("year")}
        rows[label] = {"n": st["n"], "days": st["clusters"], "mean_bp": st["mean"] * 1e4, "t_day": st["t"], "median_bp": float(part.x.median()) * 1e4, "by_year_bp": by_year}
        print(f"{label:62s} n {st['n']:5d} days {st['clusters']:4d} mean {st['mean'] * 1e4:7.2f} bp (t {st['t']:5.2f}) median {float(part.x.median()) * 1e4:6.1f}  by year {by_year}")

    for kind in ("fall", "rise", "wild"):
        part = d[d.kind == kind]
        show(f"{kind}: all", part)
        show(f"{kind}: a 2% fall in the previous 20 min (same coin)", part[part.after_fall_20])
        show(f"{kind}: NO 2% fall in the previous 20 min", part[~part.after_fall_20])
    falls = d[d.kind == "fall"]
    show("fall: first fall in 60 min (same coin)", falls[~falls.after_fall_60])
    show("fall: later leg (a fall in the previous 60 min)", falls[falls.after_fall_60])
    for lo, hi, label in ((-1, -0.05, "fall of 5% or more"), (-0.05, -0.03, "fall of 3-5%"), (-0.03, -0.025, "fall of 2.5-3%"), (-0.025, -0.0199, "fall of 2-2.5%")):
        show(f"fall: {label}", falls[(falls.r5 > lo) & (falls.r5 <= hi)])
        show(f"fall: {label}, first in 60 min", falls[(falls.r5 > lo) & (falls.r5 <= hi) & ~falls.after_fall_60])
    (OUT / "controls_design.json").write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
