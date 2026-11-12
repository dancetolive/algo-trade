"""Audit check 5a: placebo. Every event time is moved and the study is re-run with the same direction.

    python -I a06_placebo.py

Three variants, six perpetuals pooled, design and validation separately:
  hours    : each event moved by a random 2-8 hours, earlier or later (the brief's placebo; clock times change);
  days     : each event moved by a random 1-5 whole days, earlier or later (keeps the minute of the day exactly,
             the coin and the direction; breaks the link with the price move);
  profile  : fixed offsets from -8 h to +8 h and +-1 day, to see where the effect sits relative to the event.
For the random variants the whole placebo is drawn many times; reported: mean and spread of the placebo means,
and the share of draws whose mean is at least the real one.
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

DEFS = {"sweep_N240_x10_k1": ((240, 10.0, 1), (240, 60), 200), "sweep_N30_x2_k3": ((30, 2.0, 3), (5,), 40)}
PROFILE = (-1440, -480, -360, -240, -120, -60, -30, -10, -5, -1, 0, 1, 5, 10, 30, 60, 120, 240, 360, 480, 1440)


def one(args) -> dict:
    segment, symbol = args
    g, first_event = al.load_grid("um", symbol, segment)
    o, h, l, c, present = al.arrays(g)
    n = len(o)
    ctx = al.context_mask(present)
    if first_event is not None:
        ctx &= np.asarray(g.index >= first_event)
    key = al.baseline_key(g.index, al.realised_vol(c), first_event)
    rng = np.random.default_rng(1000 + sum(map(ord, segment + symbol)))
    out = {}
    for def_id, (params, horizons, draws) in DEFS.items():
        sig = al.sweep_signal(h, l, c, *params)
        t = np.flatnonzero(sig)
        t = t[ctx[t]]
        d = sig[t].astype(float)
        for hh in horizons:
            f = al.forward_return(o, present, hh)
            base = al.cell_mean(f, key)
            rec = {}

            def sums(pos: np.ndarray, dirs: np.ndarray):
                ok = (pos >= 0) & (pos < n)
                p = np.where(ok, pos, 0)
                good = ok & ctx[p] & np.isfinite(f[p]) & np.isfinite(base[p])
                gg = dirs * f[p]
                ee = dirs * (f[p] - base[p])
                return np.where(good, gg, 0.0), np.where(good, ee, 0.0), good

            gg, ee, good = sums(t, d)
            rec["real"] = [float(gg.sum()), float(ee.sum()), int(good.sum())]
            # profile: fixed offsets
            rec["profile"] = {}
            for off in PROFILE:
                gg, ee, good = sums(t + off, d)
                rec["profile"][off] = [float(gg.sum()), float(ee.sum()), int(good.sum())]
            # random placebos
            for name in ("hours", "days", "hours_before", "hours_after"):
                acc = np.zeros((draws, 3))
                for r in range(draws):
                    if name == "hours":
                        off = rng.integers(120, 481, size=len(t)) * rng.choice(np.array([-1, 1]), size=len(t))
                    elif name == "hours_before":
                        off = -rng.integers(120, 481, size=len(t))
                    elif name == "hours_after":
                        off = rng.integers(120, 481, size=len(t))
                    else:
                        off = rng.integers(1, 6, size=len(t)) * 1440 * rng.choice(np.array([-1, 1]), size=len(t))
                    gg, ee, good = sums(t + off, d)
                    acc[r] = (gg.sum(), ee.sum(), good.sum())
                rec[name] = acc.tolist()
            # one stored draw of the day-shift placebo with its calendar days, for a clustered t
            off = rng.integers(1, 6, size=len(t)) * 1440 * rng.choice(np.array([-1, 1]), size=len(t))
            pos = t + off
            ok = (pos >= 0) & (pos < n)
            p = np.where(ok, pos, 0)
            good = ok & ctx[p] & np.isfinite(f[p]) & np.isfinite(base[p])
            day = (g.index[p[good]].asi8 // (86400 * 10 ** 9)) if str(g.index.dtype).find("ns") >= 0 else None
            if day is None:
                day = ((g.index[p[good]] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(days=1)).to_numpy()
            rec["one_draw"] = {"day": np.asarray(day).tolist(), "g": (d[good] * f[p[good]]).tolist(), "e": (d[good] * (f[p[good]] - base[p[good]])).tolist()}
            out[f"{def_id}@{hh}"] = rec
    return {"segment": segment, "symbol": symbol, "res": out}


def main() -> None:
    jobs = [(seg, s) for seg in ("design", "validation") for s in al.PERPS]
    with ProcessPoolExecutor(max_workers=3) as pool:
        parts = list(pool.map(one, jobs))
    summary = {}
    for segment in ("design", "validation"):
        mine = [p["res"] for p in parts if p["segment"] == segment]
        for key in mine[0]:
            real = np.sum([m[key]["real"] for m in mine], axis=0)
            rec = {"real_gross": real[0] / real[2], "real_excess": real[1] / real[2], "events": int(real[2])}
            for name in ("hours", "days", "hours_before", "hours_after"):
                acc = np.sum([np.asarray(m[key][name]) for m in mine], axis=0)
                gm, em = acc[:, 0] / acc[:, 2], acc[:, 1] / acc[:, 2]
                rec[name] = {"draws": int(len(gm)), "gross_mean": float(gm.mean()), "gross_sd_across_draws": float(gm.std(ddof=1)),
                             "gross_min": float(gm.min()), "gross_max": float(gm.max()),
                             "excess_mean": float(em.mean()), "excess_sd_across_draws": float(em.std(ddof=1)), "excess_max": float(em.max()),
                             "share_draws_gross_ge_real": float(np.mean(gm >= rec["real_gross"])),
                             "share_draws_excess_ge_real": float(np.mean(em >= rec["real_excess"])), "events_mean": float(acc[:, 2].mean())}
            day = np.concatenate([np.asarray(m[key]["one_draw"]["day"]) for m in mine])
            gv = np.concatenate([np.asarray(m[key]["one_draw"]["g"]) for m in mine])
            evv = np.concatenate([np.asarray(m[key]["one_draw"]["e"]) for m in mine])
            adj = int(key.split("@")[1]) >= 240
            sg, se = al.cluster_stats(gv, day, adjacent=adj), al.cluster_stats(evv, day, adjacent=adj)
            rec["days_one_draw"] = {"events": sg["n"], "gross": sg["mean"], "t_gross": sg["t"], "excess": se["mean"], "t_excess": se["t"]}
            prof = {}
            for off in PROFILE:
                tot = np.sum([m[key]["profile"][off] for m in mine], axis=0)
                prof[off] = {"gross": float(tot[0] / tot[2]), "excess": float(tot[1] / tot[2]), "events": int(tot[2])}
            rec["profile"] = prof
            summary[f"{segment}|{key}"] = rec
    with open(os.path.join(al.OUT, "placebo.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
    for key, rec in summary.items():
        print(f"\n=== {key}: real gross {rec['real_gross']:+.2f}, real excess {rec['real_excess']:+.2f}, events {rec['events']}")
        for name in ("hours", "hours_before", "hours_after", "days"):
            r = rec[name]
            print(f"   placebo {name:13s}: gross {r['gross_mean']:+.2f} (sd across {r['draws']} draws {r['gross_sd_across_draws']:.2f}, range {r['gross_min']:+.2f} .. {r['gross_max']:+.2f}), "
                  f"excess {r['excess_mean']:+.2f} (sd {r['excess_sd_across_draws']:.2f}, max {r['excess_max']:+.2f}); draws >= real: gross {r['share_draws_gross_ge_real']:.3f}, excess {r['share_draws_excess_ge_real']:.3f}")
        o = rec["days_one_draw"]
        print(f"   one day-shift draw, day-clustered: gross {o['gross']:+.2f} (t {o['t_gross']:+.2f}), excess {o['excess']:+.2f} (t {o['t_excess']:+.2f}), events {o['events']}")
        print("   profile (offset min: gross / excess): " + "; ".join(f"{off:+d}: {v['gross']:+.1f}/{v['excess']:+.1f}" for off, v in rec["profile"].items()))


if __name__ == "__main__":
    main()
