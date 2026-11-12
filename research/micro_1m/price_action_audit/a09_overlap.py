"""Audit checks 6/7 (decisive part): is the per-event mean of the selected pair what ONE account can earn?

    python -I a09_overlap.py

The event study averages over events. Sweeps of the 4-hour extreme come in bursts: a new event often arrives while
the position opened at an earlier event in the same coin is still inside its 240-minute hold. The per-event mean
then describes an account that opens a NEW, equal-sized position at every event, however many are already open.

1. Dose-response: events by the number of earlier events in the same coin during the previous 240 minutes
   (known at the event; no look-ahead). Counted twice, by two different methods.
2. Cap on simultaneously open positions per coin: 1, 2, 3, 5, 10, none. Gross, taker net, resting-order net.
3. A single position that is extended by each new signal (exit 240 minutes after the LAST signal; an opposite signal
   reverses it): one round trip per episode.
4. How many positions the uncapped study has open at once.
5. POST HOC (auditor, both segments already seen): one position per coin, entered only once k earlier events exist.

Six perpetuals, sweep_N240_x10_k1, horizons 240 and 60; design, validation.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_lib as al  # noqa: E402

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
DEF = "sweep_N240_x10_k1"
WINDOW = 240


def load(segment: str, market: str = "um") -> pd.DataFrame:
    ev = pd.read_parquet(os.path.join(al.OUT, f"events_{segment}_{market}.parquet"))
    ev = ev[ev["def_id"] == DEF].sort_values(["symbol", "ts"], kind="stable").reset_index(drop=True)
    ev["minute"] = ((ev["ts"] - EPOCH) // pd.Timedelta(minutes=1)).astype(np.int64)
    ev["day"] = ev["minute"] // 1440
    ev["cost"] = ev["symbol"].map(lambda s: al.taker_round_trip_bp(s, market))
    ev["slip"] = ev["symbol"].map(al.slip_bp)
    # earlier events in the same coin within the previous WINDOW minutes: method 1, binary search on sorted minutes
    n1 = np.zeros(len(ev), dtype=np.int64)
    for _, idx in ev.groupby("symbol").indices.items():
        m = ev["minute"].to_numpy()[idx]
        n1[idx] = np.arange(len(m)) - np.searchsorted(m, m - WINDOW, side="left")
    # method 2: per-minute indicator series and a rolling sum (independent of method 1)
    n2 = np.zeros(len(ev), dtype=np.int64)
    for _, idx in ev.groupby("symbol").indices.items():
        m = ev["minute"].to_numpy()[idx]
        flag = np.zeros(int(m.max() - m.min()) + 1)
        flag[m - m.min()] = 1.0
        roll = pd.Series(flag).rolling(WINDOW + 1, min_periods=1).sum().to_numpy()  # events in [t-240, t]
        n2[idx] = (roll[m - m.min()] - 1).astype(np.int64)
    assert np.array_equal(n1, n2), "the two counts of earlier events disagree"
    ev["n_prior"] = n1
    return ev


def cs(v, day, adj) -> dict:
    return al.cluster_stats(np.asarray(v, dtype=float), np.asarray(day), adjacent=adj)


def take_with_cap(ev: pd.DataFrame, h: int, cap: int, min_prior: int = 0) -> np.ndarray:
    """Events an account takes when it may hold at most `cap` positions per coin (decided from the past only)."""
    keep = np.zeros(len(ev), dtype=bool)
    minute, n_prior = ev["minute"].to_numpy(), ev["n_prior"].to_numpy()
    for _, idx in ev.groupby("symbol").indices.items():
        closing: list[int] = []
        for i in idx:
            now = minute[i]
            closing = [u for u in closing if u > now]          # positions whose exit (open of t+1+h) is still ahead
            if n_prior[i] >= min_prior and len(closing) < cap:
                keep[i] = True
                closing.append(now + h)                         # entry open of t+1, exit open of t+1+h: free again for an event at t+h
    return keep


def open_positions(ev: pd.DataFrame, h: int) -> np.ndarray:
    """Positions of the uncapped study already open in the same coin when each event arrives."""
    out = np.zeros(len(ev), dtype=np.int64)
    for _, idx in ev.groupby("symbol").indices.items():
        m = ev["minute"].to_numpy()[idx]
        out[idx] = np.arange(len(m)) - np.searchsorted(m, m - h, side="right")
    return out


def summarise(sub: pd.DataFrame, h: int) -> dict:
    adj = h >= 240
    if len(sub) == 0:
        return {"events": 0}
    g = cs(sub[f"g{h}"], sub["day"], adj)
    e = cs(sub[f"g{h}"] - sub[f"b{h}"], sub["day"], adj)
    wk = cs(sub[f"g{h}"], sub["day"] // 7, False)
    cost = float(sub["cost"].mean())
    mk = sub[(sub["mk_off"] > 0) & sub["mk_ctx"] & np.isfinite(sub[f"mk_g{h}"])]
    m = cs(mk[f"mk_g{h}"], mk["day"], adj) if len(mk) > 1 else dict(mean=np.nan, se=np.nan, n=0)
    slip = float(mk["slip"].mean()) if len(mk) else np.nan
    lo, sh = sub[sub["dir"] > 0], sub[sub["dir"] < 0]
    return dict(events=g["n"], days=g["clusters"], gross=g["mean"], t_gross=g["t"], t_gross_week=wk["t"], excess=e["mean"], t_excess=e["t"],
                median=float(sub[f"g{h}"].median()), net_taker=g["mean"] - cost, t_net_taker=(g["mean"] - cost) / g["se"],
                maker_fills=m["n"], maker_gross_per_fill=m["mean"], maker_net_4bp=m["mean"] - 4.0, t_maker_net_4bp=(m["mean"] - 4.0) / m["se"],
                maker_in_taker_out=m["mean"] - (7.0 + slip), t_maker_in_taker_out=(m["mean"] - (7.0 + slip)) / m["se"],
                long_events=int(len(lo)), long_gross=float(lo[f"g{h}"].mean()) if len(lo) else np.nan,
                short_events=int(len(sh)), short_gross=float(sh[f"g{h}"].mean()) if len(sh) else np.nan)


def extended_single_position(segment: str, h: int) -> dict:
    """One position per coin that follows the latest signal: exit h minutes after the last signal, reverse on an opposite one."""
    ev = load(segment)
    rows = []
    for symbol, sub in ev.groupby("symbol"):
        g, first_event = al.load_grid("um", symbol, segment)
        o, hh, l, c, present = al.arrays(g)
        base = int((g.index[0] - EPOCH) // pd.Timedelta(minutes=1))
        csum = np.concatenate([[0], np.cumsum(present)])
        pos = sub["minute"].to_numpy() - base
        d = sub["dir"].to_numpy()
        i = 0
        while i < len(pos):
            start, direction, last = pos[i], d[i], pos[i]
            j = i + 1
            while j < len(pos) and pos[j] <= last + h and d[j] == direction:   # same-side signal while still in the position
                last = pos[j]
                j += 1
            reversed_ = j < len(pos) and pos[j] <= last + h                      # opposite signal arrives first: reverse there
            exit_bar = pos[j] + 1 if reversed_ else last + 1 + h
            entry_bar = start + 1
            if exit_bar < len(o) and (csum[exit_bar + 1] - csum[start]) == (exit_bar - start + 1):
                rows.append(dict(symbol=symbol, day=int(sub["minute"].to_numpy()[i] // 1440), dir=int(direction), signals=int(j - i),
                                 held=int(exit_bar - entry_bar), ret=float(direction * (o[exit_bar] / o[entry_bar] - 1.0) * 1e4),
                                 cost=al.taker_round_trip_bp(symbol)))
            i = j
    ep = pd.DataFrame(rows)
    g = cs(ep["ret"], ep["day"], True)
    cost = float(ep["cost"].mean())
    return dict(episodes=g["n"], mean_signals_per_episode=float(ep["signals"].mean()), mean_minutes_held=float(ep["held"].mean()),
                gross_per_episode=g["mean"], t_gross=g["t"], net_taker=g["mean"] - cost, t_net_taker=(g["mean"] - cost) / g["se"],
                net_maker_4bp_bound=g["mean"] - 4.0, t_net_maker_4bp_bound=(g["mean"] - 4.0) / g["se"],
                gross_per_100_minutes_held=float(ep["ret"].sum() / ep["held"].sum() * 100),
                long_gross=float(ep.loc[ep["dir"] > 0, "ret"].mean()), short_gross=float(ep.loc[ep["dir"] < 0, "ret"].mean()))


def main() -> None:
    res = {}
    for segment in ("design", "validation"):
        ev = load(segment)
        seg = {}
        for h in (240, 60):
            e = ev[np.isfinite(ev[f"g{h}"]) & np.isfinite(ev[f"b{h}"])].reset_index(drop=True)
            block = {"all": summarise(e, h)}
            total = float(e[f"g{h}"].sum())
            dose = {}
            for label, a, b in (("0", 0, 0), ("1", 1, 1), ("2-3", 2, 3), ("4-7", 4, 7), ("8+", 8, 10 ** 9)):
                s = e[(e["n_prior"] >= a) & (e["n_prior"] <= b)]
                r = summarise(s, h)
                r["share_of_events"] = len(s) / len(e)
                r["share_of_total_gross"] = float(s[f"g{h}"].sum()) / total
                dose[label] = r
            block["by_earlier_events_in_240min"] = dose
            caps = {}
            for cap in (1, 2, 3, 5, 10, 10 ** 6):
                s = e[take_with_cap(e, h, cap)]
                r = summarise(s, h)
                r["share_of_events"] = len(s) / len(e)
                caps["none" if cap == 10 ** 6 else str(cap)] = r
            block["cap_on_open_positions_per_coin"] = caps
            n_open = open_positions(e, h)
            by_min = e.groupby("minute").size()
            block["concurrency_uncapped"] = {
                "share_of_events_arriving_with_a_position_open_in_the_coin": float((n_open > 0).mean()),
                "open_positions_in_coin_at_event_median": float(np.median(n_open)), "p90": float(np.percentile(n_open, 90)),
                "p99": float(np.percentile(n_open, 99)), "max": int(n_open.max())}
            if h == 240:
                post = {}
                for k in (1, 2, 3, 4, 6, 8):
                    s = e[take_with_cap(e, h, 1, min_prior=k)]
                    post[str(k)] = summarise(s, h)
                block["POST_HOC_one_position_entered_after_k_earlier_events"] = post
                block["single_position_extended_by_each_signal"] = extended_single_position(segment, h)
            seg[str(h)] = block
        res[segment] = seg
    with open(os.path.join(al.OUT, "overlap.json"), "w") as handle:
        json.dump(res, handle, indent=1)

    pd.set_option("display.width", 250)
    cols = ["events", "share_of_events", "gross", "t_gross", "t_gross_week", "median", "net_taker", "t_net_taker", "maker_gross_per_fill", "maker_net_4bp",
            "t_maker_net_4bp", "maker_in_taker_out", "long_gross", "short_gross"]
    for segment, seg in res.items():
        for h, block in seg.items():
            print(f"\n===== {segment} {DEF} @ {h}: all events gross {block['all']['gross']:.2f} (t {block['all']['t_gross']:.2f})")
            print("by number of earlier events in the same coin within 240 minutes:")
            t = pd.DataFrame(block["by_earlier_events_in_240min"]).T
            print(t[cols + ["share_of_total_gross"]].to_string(float_format=lambda v: f"{v:.2f}"))
            print("cap on open positions per coin:")
            print(pd.DataFrame(block["cap_on_open_positions_per_coin"]).T[cols].to_string(float_format=lambda v: f"{v:.2f}"))
            print("concurrency:", {k: round(v, 3) for k, v in block["concurrency_uncapped"].items()})
            if "single_position_extended_by_each_signal" in block:
                print("single position extended by each signal:", {k: round(v, 2) for k, v in block["single_position_extended_by_each_signal"].items()})
                print("POST HOC, one position per coin entered only after k earlier events:")
                print(pd.DataFrame(block["POST_HOC_one_position_entered_after_k_earlier_events"]).T[
                    ["events", "days", "gross", "t_gross", "t_gross_week", "median", "net_taker", "t_net_taker", "long_gross", "short_gross"]]
                    .to_string(float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
