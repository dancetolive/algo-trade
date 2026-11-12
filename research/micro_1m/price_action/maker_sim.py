"""What a resting (maker) order really gets on the selected structures.

The headline "maker bound" in the tables is gross minus 4 bp, i.e. it pretends
that a passive order is filled at the taker's price. This script replaces the
pretence by a simulation that needs only 1-minute bars:

* At the close of the event bar t a limit order is placed AT THAT CLOSE, on the
  side of the hypothesis (buy for +1, sell for -1).
* It is filled in the first of the next ``REST`` bars that trades STRICTLY
  through the price (low < limit for a buy, high > limit for a sell). Touching
  is not enough: queue position is unknown, trading through is certain.
* If it is not filled within ``REST`` bars it is cancelled: no trade.
* A filled position is closed at the open of the bar ``h`` minutes after the
  bar following the fill (the same clock as the taker test).

Reported per definition / horizon: fill rate, gross per filled trade, the
taker gross of the SAME events split into later-filled and never-filled (the
never-filled ones are the trades a maker misses), and net figures with a maker
fee on both sides (4 bp, still optimistic about the exit) and with maker in /
taker out.

    python -I -B maker_sim.py design sweep_N240_x10_k1:60 sweep_N240_x10_k1:240 sweep_N30_x2_k3:5
    python -I -B maker_sim.py validation        # pairs are read from selection.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import eventstudy as es  # noqa: E402
import structures as st  # noqa: E402
from run_segment import OUT, PERPS, load_segment_bars  # noqa: E402

REST = 5  # minutes the order is left in the book


def simulate(segment: str, pairs: list[tuple[str, int]]) -> pd.DataFrame:
    def_ids = sorted({d for d, _ in pairs})
    horizons = tuple(sorted({h for _, h in pairs}))
    horizons0 = (0,) + horizons
    parts: dict[tuple[str, str], list[pd.DataFrame]] = {}
    for symbol in PERPS:
        bars, first_minute = load_segment_bars(segment, "um", symbol)
        fwd = es.forward_returns(bars, horizons)
        zero = {h: np.zeros(bars.n) for h in horizons0}
        ok = es.context_ok(bars)
        if first_minute is not None:
            ok &= bars.minute >= first_minute
        for def_id, ev in st.detect(bars, def_ids).items():
            keep = ok[ev.idx]
            idx, d = ev.idx[keep], ev.direction[keep]
            fill = st.zone_first_touch(bars.h, bars.l, idx, bars.c[idx], d, REST, strict=True)
            got = fill >= 0
            sets = {
                "all_taker": es.event_daily(bars, idx, d, fwd, zero, ok, horizons=horizons),
                "filled_taker": es.event_daily(bars, idx[got], d[got], fwd, zero, ok, horizons=horizons),
                "missed_taker": es.event_daily(bars, idx[~got], d[~got], fwd, zero, ok, horizons=horizons),
                "filled_maker": es.event_daily(bars, fill[got], d[got], fwd, zero, ok, bars.c[idx][got], horizons=horizons0),
            }
            for name, daily in sets.items():
                daily["symbol"] = symbol
                parts.setdefault((def_id, name), []).append(daily)
    rows = []
    for def_id, h in pairs:
        stats = {}
        for name in ("all_taker", "filled_taker", "missed_taker", "filled_maker"):
            daily = pd.concat(parts[(def_id, name)], ignore_index=True)
            hs = (0, h) if name == "filled_maker" else (h,)
            stats[name] = es.pooled_stats(daily, hs)
            per_symbol = daily.groupby("symbol")[f"n{h}"].sum()
            stats[name + "_slip"] = float(sum(per_symbol[s] * es.slippage_bp(s) for s in per_symbol.index) / per_symbol.sum())
        a, f, m, k = (stats[n].loc[h] for n in ("all_taker", "filled_taker", "missed_taker", "filled_maker"))
        slip = stats["filled_maker_slip"]
        rows.append({
            "segment": segment, "def_id": def_id, "h": h, "events": int(a["events"]),
            "taker_gross_all": a["gross"], "taker_t_all": a["t_gross"],
            "fill_rate": k["events"] / a["events"],
            "taker_gross_if_later_filled": f["gross"], "taker_gross_if_never_filled": m["gross"],
            "maker_mark_at_next_open": stats["filled_maker"].loc[0, "gross"],
            "maker_gross_per_fill": k["gross"], "maker_t_per_fill": k["t_gross"],
            "maker_net_per_fill_maker_exit_4bp": k["gross"] - 4.0,
            "maker_net_per_fill_taker_exit": k["gross"] - (2.0 + 5.0 + slip),
            "maker_gross_per_event": k["gross"] * k["events"] / a["events"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    segment = sys.argv[1]
    if segment in ("validation", "dryrun"):
        with open(os.path.join(HERE, "selection.json")) as handle:
            pairs = [(item["def_id"], int(item["h"])) for item in json.load(handle)["selected"]]
    else:
        pairs = [(a.split(":")[0], int(a.split(":")[1])) for a in sys.argv[2:]]
    table = simulate(segment, pairs)
    table.to_csv(os.path.join(OUT, f"maker_sim_{segment}.csv"), index=False)
    pd.set_option("display.width", 250)
    print(table.T.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
