"""Audit (mirror image): would the researcher's pipeline have SEEN a real effect in the families it calls dead?

    python -I -B a12_planted_effect.py

On real bars (BTC and SOL perpetuals, design) a synthetic effect is planted for one definition at a time: every
open from bar t+2 onward is multiplied by (1 + d * 10 bp) for each event (bar t, direction d). Highs, lows and
closes are untouched, so the events stay exactly where they were. The return from the open of t+1 to the open of
t+2, the first minute after the fill, is thereby raised by 10 bp in the direction of the hypothesis.

The researcher's chain (forward_returns -> matched_baseline -> event_daily -> pooled_stats) is then run on the
planted bars and must report
  * a gross return at 1 minute that is 10 bp higher than on the original bars;
  * at every horizon exactly the mean that my own forward-return code gives for the same events on the same planted
    opens (longer horizons also collect the plants of later events inside the window, so the uplift exceeds 10 bp
    where same-direction events cluster);
  * NO uplift when the plant is put one bar earlier (between the open of t and the open of t+1, i.e. before the
    fill): a pipeline that credited the event bar's own move would show it here.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
THEIR = os.path.join(os.path.dirname(HERE), "price_action")
sys.path.insert(0, HERE)
sys.path.insert(1, THEIR)

import audit_lib as al  # noqa: E402
import eventstudy as es  # noqa: E402  (researcher's: object under test)
import structures as st  # noqa: E402  (researcher's: object under test)

DELTA_BP = 10.0
DEFS = ["bos_s5_all", "bos_s5_with", "bos_s5_against", "fvg_m0p5", "fvgfill_m0p5_W60", "squeeze_M30_q0p1_v2", "ob_W60", "disp_m3",
        "pdsweep_x2_k3", "pdhold_x2_k1", "hold_N240_x10_k3", "sweepmat_N240_x2_k3", "sweep_N240_x10_k1", "sweep_N30_x2_k3"]


def planted_bars(bars, idx, direction, offset: int):
    inc = np.zeros(bars.n)
    pos = idx + offset
    ok = pos < bars.n
    np.add.at(inc, pos[ok], np.log1p(direction[ok] * DELTA_BP * 1e-4))
    o2 = bars.o * np.exp(np.cumsum(inc))
    return es.Bars(bars.symbol, bars.market, bars.minute, o2, bars.h, bars.l, bars.c, bars.v, bars.valid)


def their_stats(bars, ev, ok, cell, n_cells) -> pd.DataFrame:
    fwd = es.forward_returns(bars)
    base = es.matched_baseline(fwd, cell, n_cells)
    daily = es.event_daily(bars, ev.idx, ev.direction, fwd, base, ok)
    return es.pooled_stats(daily)


def my_means(bars, ev, ok) -> dict:
    present = bars.valid
    keep = ok[ev.idx]
    t, d = ev.idx[keep], ev.direction[keep].astype(float)
    out = {}
    for h in es.HORIZONS:
        f = al.forward_return(bars.o, present, h)
        v = d * f[t]
        out[h] = float(np.nanmean(v))
    return out


def run(symbol: str) -> list[dict]:
    bars = es.load_bars("um", symbol, end="2024-01-01")
    ok = es.context_ok(bars)
    cell, n_cells = es.baseline_cells(bars)
    events = st.detect(bars, DEFS)
    rows = []
    for def_id in DEFS:
        ev = events[def_id]
        d = ev.direction.astype(float)
        base_stats = their_stats(bars, ev, ok, cell, n_cells)
        planted = planted_bars(bars, ev.idx, d, 2)
        got = their_stats(planted, ev, ok, cell, n_cells)
        mine = my_means(planted, ev, ok)
        early = their_stats(planted_bars(bars, ev.idx, d, 1), ev, ok, cell, n_cells)
        row = {"symbol": symbol, "def_id": def_id, "events": int(base_stats.loc[1, "events"])}
        for h in es.HORIZONS:
            row[f"orig_{h}"] = float(base_stats.loc[h, "gross"])
            row[f"uplift_{h}"] = float(got.loc[h, "gross"] - base_stats.loc[h, "gross"])
            row[f"their_minus_mine_{h}"] = float(got.loc[h, "gross"] - mine[h])
            row[f"early_uplift_{h}"] = float(early.loc[h, "gross"] - base_stats.loc[h, "gross"])
        row["t_excess_1_planted"] = float(got.loc[1, "t_excess"])
        row["excess_1_planted"] = float(got.loc[1, "excess"])
        row["excess_1_orig"] = float(base_stats.loc[1, "excess"])
        rows.append(row)
    return rows


def main() -> None:
    rows = run("BTCUSDT") + run("SOLUSDT")
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(al.OUT, "planted_effect.csv"), index=False)
    pd.set_option("display.width", 250)
    cols = ["symbol", "def_id", "events", "orig_1", "uplift_1", "early_uplift_1", "uplift_5", "uplift_60", "uplift_240", "excess_1_orig", "excess_1_planted", "t_excess_1_planted"]
    print(t[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    worst = max(t[f"their_minus_mine_{h}"].abs().max() for h in es.HORIZONS)
    print("\nmax |their pooled gross - my direct mean| over all definitions and horizons (planted bars):", worst)
    print("1-minute uplift: min %.3f max %.3f (planted %.1f bp)" % (t["uplift_1"].min(), t["uplift_1"].max(), DELTA_BP))
    print("uplift when planted one bar too early (before the fill), 1-minute horizon: min %.3f max %.3f" % (t["early_uplift_1"].min(), t["early_uplift_1"].max()))
    summary = {"planted_bp": DELTA_BP, "definitions": len(DEFS), "symbols": ["BTCUSDT", "SOLUSDT"], "uplift_1min_min": float(t["uplift_1"].min()),
               "uplift_1min_max": float(t["uplift_1"].max()), "max_abs_their_minus_mine_all_horizons": float(worst),
               "early_plant_uplift_1min_min": float(t["early_uplift_1"].min()), "early_plant_uplift_1min_max": float(t["early_uplift_1"].max()),
               "min_t_excess_1min_planted": float(t["t_excess_1_planted"].min())}
    with open(os.path.join(al.OUT, "planted_effect.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
