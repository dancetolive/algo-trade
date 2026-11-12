"""Day-block bootstrap for the pre-registered pairs (a second opinion on the clustered t-statistics).

    python -I -B bootstrap_selected.py design
    python -I -B bootstrap_selected.py validation

Resamples whole calendar days (blocks of 1 day for horizons below 240 minutes,
5 consecutive days from 240 minutes on, because those returns overlap across
midnight), all six perpetuals together, 4000 resamples, seed 0. Reports the
bootstrap standard error, the 95% interval of the mean excess return over the
matched baseline and of the mean gross return, and the share of resamples in
which the mean gross return exceeds the taker round trip.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyze  # noqa: E402
import eventstudy as es  # noqa: E402

N_BOOT = 4000


def main() -> None:
    segment = sys.argv[1]
    with open(os.path.join(HERE, "selection.json")) as handle:
        pairs = [(item["def_id"], int(item["h"])) for item in json.load(handle)["selected"]]
    daily = analyze.load_daily(segment, "um")
    cost = analyze.costs("um", daily["symbol"].unique())["taker"]
    rows = []
    for def_id, h in pairs:
        part = daily[daily["def_id"] == def_id]
        by_day = part.groupby("day")[[f"n{h}", f"sg{h}", f"sb{h}"]].sum()
        by_day = by_day[by_day[f"n{h}"] > 0]
        ids, n = by_day.index.to_numpy(), by_day[f"n{h}"].to_numpy(dtype=float)
        gross, excess = by_day[f"sg{h}"].to_numpy(), (by_day[f"sg{h}"] - by_day[f"sb{h}"]).to_numpy()
        per_symbol = part.groupby("symbol", observed=True)[f"n{h}"].sum()
        cost_bp = float(sum(per_symbol[s] * cost[s] for s in per_symbol.index) / per_symbol.sum())
        block = 5 if h >= 240 else 1
        b_e = es.block_bootstrap_p(excess, n, ids, block=block, n_boot=N_BOOT, seed=0)
        b_g = es.block_bootstrap_p(gross, n, ids, block=block, n_boot=N_BOOT, seed=0)
        # share of resamples in which the mean gross return beats the taker round trip
        first, span = int(ids.min()), int(ids.max() - ids.min() + 1)
        s, c = np.zeros(span), np.zeros(span)
        np.add.at(s, ids - first, gross)
        np.add.at(c, ids - first, n)
        rng = np.random.default_rng(0)
        starts = rng.integers(0, span - block + 1, size=(N_BOOT, int(np.ceil(span / block))))
        pick = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(N_BOOT, -1)[:, :span]
        boot = s[pick].sum(axis=1) / np.maximum(c[pick].sum(axis=1), 1.0)
        rows.append({"segment": segment, "def_id": def_id, "h": h, "block_days": block,
                     "excess": b_e["mean"], "boot_se_excess": b_e["se"], "excess_ci_lo": b_e["ci95"][0], "excess_ci_hi": b_e["ci95"][1],
                     "p_excess_two_sided": b_e["p_two_sided"],
                     "gross": b_g["mean"], "gross_ci_lo": b_g["ci95"][0], "gross_ci_hi": b_g["ci95"][1],
                     "cost_taker": cost_bp, "share_resamples_gross_above_cost": float(np.mean(boot > cost_bp))})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(analyze.OUT, f"bootstrap_{segment}.csv"), index=False)
    pd.set_option("display.width", 250)
    print(table.T.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
