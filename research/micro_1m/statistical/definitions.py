"""The event definitions that were tested (the complete list: nothing is hidden).

Each definition yields a boolean mask over the bars of a :class:`eventstudy.Grid`
and the side of the pre-stated hypothesis (+1 buy / -1 sell at the next open).
All inputs come from :func:`eventstudy.trailing_features`, which is causal.

Families
--------
1  spike   quote-volume / trade-count of bar t above k x the median of the 60 bars before it,
           k in {5, 10}; shape of the bar: all | cont (closes within 20% of the range from the
           extreme it travelled to) | exh (gave back at least half of the range from that extreme);
           split by bar direction (both | up | dn).
           Hypothesis: all, cont -> continuation (trade with the bar); exh -> reversal (fade the bar).
2  ofi     taker-buy share of quote volume over the last w in {1, 5, 15} minutes, standardised by
           its own trailing 1-day mean and standard deviation, |z| > k for k in {2, 3};
           split both | buy | sell. Hypothesis: continuation (trade with the pressure).
3  move    w-minute return, w in {1, 5, 15}, beyond k trailing 1-day standard deviations
           (scaled by sqrt(w)), k in {3, 5, 8}; k = 8 are the liquidation-cascade candidates;
           plus |5-minute return| >= 2% in absolute terms; split both | up | dn.
           Hypothesis: continuation (trade with the move). A negative result is a reversal.
4  sign    every bar: the sign of the trailing w-minute return, w in {1, 5, 15, 60}.
           Hypothesis: continuation. This is the autocorrelation map in tradable form.
5  time    hour of day (24), minute of the hour (60), and windows of 5 / 15 / 60 minutes before
           and after the 00:00 / 08:00 / 16:00 UTC funding stamps on perpetuals: unconditional,
           signed by the last funding rate already paid, and signed with |rate| >= 3 bp.
           Hypothesis (signed): price falls into the stamp when longs pay and recovers after.

For windows w > 1 an event is the first bar of a crossing (no other event of the same definition
in the previous w - 1 bars), so one move is not counted w times.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from eventstudy import HORIZONS, Grid, first_in_window

FUNDING_MINUTES = (0, 480, 960)  # 00:00, 08:00, 16:00 UTC as minute of day
BIG_FUNDING = 3e-4


@dataclass
class Definition:
    name: str
    family: str
    params: dict
    hypothesis: str
    horizons: tuple = HORIZONS  # the horizons that count as looks for this definition
    baseline: str = "hour"  # "hour": matched by year x hour x vol state; "nohour": year x vol state
    mask: np.ndarray | None = field(default=None, repr=False)
    direction: object = field(default=None, repr=False)


def _true(values: np.ndarray) -> np.ndarray:
    """Comparison results with NaN treated as False."""
    return np.nan_to_num(values, nan=0.0).astype(bool) if values.dtype != bool else values


def previous_funding(grid: Grid, funding) -> np.ndarray:
    """Last funding rate stamped at or before each bar's open time (NaN before the first one).

    ``funding`` is a Series indexed by UTC stamp. The rate stamped exactly at a
    bar's close is deliberately NOT used: it is only final at that instant.
    """
    out = np.full(len(grid), np.nan)
    if funding is None or len(funding) == 0:
        return out
    funding = funding.dropna().sort_index()
    stamps = funding.index.as_unit("ns").asi8
    bars = grid.index.as_unit("ns").asi8
    position = np.searchsorted(stamps, bars, side="right") - 1
    known = position >= 0
    out[known] = funding.to_numpy(dtype=float)[position[known]]
    return out


def generate(grid: Grid, feats: dict, funding=None, only: set | None = None) -> Iterator[Definition]:
    """Yield every definition with its mask and direction for this symbol."""
    usable = grid.usable

    def want(name: str) -> bool:
        return only is None or name in only

    bar_dir = feats["bar_dir"]
    has_dir = np.isfinite(bar_dir) & (bar_dir != 0)
    bar_side = np.where(has_dir, bar_dir, 0.0)

    # ---- family 1: volume and trade-count spikes -------------------------------------
    wick = feats["wick"]
    with np.errstate(invalid="ignore"):
        shapes = {"all": has_dir, "cont": has_dir & (wick <= 0.2), "exh": has_dir & (wick >= 0.5)}
        for kind, key in (("vol", "vratio"), ("trd", "tratio")):
            ratio = feats[key]
            for k in (5, 10):
                spike = ratio > k
                for shape, cond in shapes.items():
                    hyp = -1.0 if shape == "exh" else 1.0
                    for side, sel in (("both", has_dir), ("up", bar_dir > 0), ("dn", bar_dir < 0)):
                        name = f"{kind}spike{k}_{shape}_{side}"
                        if not want(name):
                            continue
                        yield Definition(
                            name,
                            "1 spike",
                            {"measure": "quote volume" if kind == "vol" else "trade count", "k": k, "shape": shape, "bars": side},
                            "reversal (fade the bar)" if shape == "exh" else "continuation (with the bar)",
                            mask=usable & spike & cond & sel,
                            direction=hyp * bar_side,
                        )

    # ---- family 2: order-flow imbalance ----------------------------------------------
    for w in (1, 5, 15):
        z = feats[f"ofiz{w}"]
        with np.errstate(invalid="ignore"):
            for k in (2, 3):
                names = [f"ofi{w}_z{k}_{side}" for side in ("both", "buy", "sell")]
                if not any(want(name) for name in names):
                    continue
                extreme = first_in_window(usable & (np.abs(z) > k), w)
                side_of = np.sign(np.nan_to_num(z))
                for side, sel in (("both", extreme), ("buy", extreme & (z > 0)), ("sell", extreme & (z < 0))):
                    name = f"ofi{w}_z{k}_{side}"
                    if want(name):
                        yield Definition(name, "2 ofi", {"window": w, "k": k, "side": side}, "continuation (with the pressure)", mask=sel, direction=side_of)

    # ---- family 3: large standardised moves -------------------------------------------
    for w in (1, 5, 15):
        z = feats[f"z{w}"]
        with np.errstate(invalid="ignore"):
            for k in (3, 5, 8):
                names = [f"move{w}_z{k}_{side}" for side in ("both", "up", "dn")]
                if not any(want(name) for name in names):
                    continue
                extreme = first_in_window(usable & (np.abs(z) > k), w)
                side_of = np.sign(np.nan_to_num(z))
                for side, sel in (("both", extreme), ("up", extreme & (z > 0)), ("dn", extreme & (z < 0))):
                    name = f"move{w}_z{k}_{side}"
                    if want(name):
                        yield Definition(name, "3 move", {"window": w, "k": k, "side": side}, "continuation (with the move)", mask=sel, direction=side_of)
    ret5 = feats["ret5"]
    names = [f"move5_abs2pct_{side}" for side in ("both", "up", "dn")]
    if any(want(name) for name in names):
        with np.errstate(invalid="ignore"):
            extreme = first_in_window(usable & (np.abs(ret5) >= 0.02), 5)
            side_of = np.sign(np.nan_to_num(ret5))
            for side, sel in (("both", extreme), ("up", extreme & (ret5 > 0)), ("dn", extreme & (ret5 < 0))):
                name = f"move5_abs2pct_{side}"
                if want(name):
                    yield Definition(name, "3 move", {"window": 5, "abs": 0.02, "side": side}, "continuation (with the move)", mask=sel, direction=side_of)

    # ---- family 4: sign of the trailing return, every bar ---------------------------------
    for w in (1, 5, 15, 60):
        name = f"sign{w}"
        if not want(name):
            continue
        ret = feats[f"ret{w}"]
        with np.errstate(invalid="ignore"):
            mask = usable & np.isfinite(ret) & (ret != 0)
        yield Definition(name, "4 sign", {"window": w}, "continuation (momentum)", mask=mask, direction=np.sign(np.nan_to_num(ret)))

    # ---- family 5: time structure ---------------------------------------------------------
    minute_of_day = grid.minute_of_day
    for hour in range(24):
        name = f"hour{hour:02d}"
        if want(name):
            # flagged at the close of the last minute of the previous hour: the 60-minute return is that hour
            yield Definition(name, "5 time", {"hour": hour}, "long during this hour", horizons=(60,), baseline="nohour",
                             mask=usable & (minute_of_day == (hour * 60 - 1) % 1440), direction=1.0)
    minute = minute_of_day % 60
    for m in range(60):
        name = f"minute{m:02d}"
        if want(name):
            yield Definition(name, "5 time", {"minute": m}, "long during this minute", horizons=(1,), baseline="nohour",
                             mask=usable & (minute == (m - 1) % 60), direction=1.0)
    if funding is not None:
        rate = previous_funding(grid, funding)
        with np.errstate(invalid="ignore"):
            signed = np.isfinite(rate) & (rate != 0)
            big = signed & (np.abs(rate) >= BIG_FUNDING)
        rate_side = np.sign(np.nan_to_num(rate))
        for k in (5, 15, 60):
            pre = usable & np.isin(minute_of_day, [(t - k - 1) % 1440 for t in FUNDING_MINUTES])
            post = usable & np.isin(minute_of_day, [(t - 1) % 1440 for t in FUNDING_MINUTES])
            for when, mask, sign in (("pre", pre, -1.0), ("post", post, 1.0)):
                variants = (
                    ("all", mask, 1.0, "long over this window"),
                    ("signed", mask & signed, sign * rate_side, "against the payers before the stamp, with them after"),
                    ("big", mask & big, sign * rate_side, "same, |last funding| >= 3 bp"),
                )
                for variant, sel, direction, hypothesis in variants:
                    name = f"fund_{when}{k}_{variant}"
                    if want(name):
                        yield Definition(name, "5 time", {"when": when, "minutes": k, "variant": variant}, hypothesis,
                                         horizons=(k,), baseline="nohour", mask=sel, direction=direction)
