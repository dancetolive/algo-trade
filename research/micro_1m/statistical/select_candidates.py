"""Write down the selection BEFORE the validation segment is touched.

    python -I select_candidates.py

Reads only out/design/*.npz. Refuses to overwrite an existing SELECTION.json: the selection is made once.
The record is SELECTION.json (the same text is printed to the terminal).
The choice itself (the three names and horizons below) was made by reading the design tables; the
reasons are recorded in the file this script writes.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analysis as an  # noqa: E402
from run_events import PERPS  # noqa: E402

# trade_side: +1 means the trade is taken on the side of the definition's pre-stated hypothesis,
# -1 means the opposite side (the design data contradicted the hypothesis and the trade follows the data).
SELECTED = [
    {
        "name": "move5_abs2pct_dn",
        "horizon": 15,
        "trade_side": -1,
        "trade": "BUY at the next minute's open after the 5-minute return (close to close) falls to -2% or lower "
                 "(first bar of the crossing; no other such bar in the previous 4); hold 15 minutes",
        "why": "Largest day-clustered |t| of any look whose gross size exceeds a taker round trip (design |t| about 10). "
               "Same sign in each of the four design years and in each of the six coins; survives a one- and two-minute "
               "late entry and a fill at the next minute's volume-weighted price. The pre-stated hypothesis was "
               "continuation; the data say reversal, so the trade is the opposite side.",
        "part_b": {"bullish": "move5_abs2pct_dn", "bearish_mirror": "move5_abs2pct_up"},
    },
    {
        "name": "move1_z3_dn",
        "horizon": 15,
        "trade_side": -1,
        "trade": "BUY at the next minute's open after a 1-minute return below -3 trailing (1-day) standard deviations; hold 15 minutes",
        "why": "The most frequent bullish structure with a pooled day-clustered |t| above 7 (about ten events per coin and day), "
               "so it is the natural candidate for entry timing in Part B. Known weaknesses on design: gross size (about 8 bp) is "
               "below a taker round trip, it is weak in 2022 and on BTC. Expected NOT to pay costs; carried to validation to see "
               "whether even the gross effect persists.",
        "part_b": {"bullish": "move1_z3_dn", "bearish_mirror": "move1_z3_up"},
    },
    {
        "name": "fund_post5_big",
        "horizon": 5,
        "trade_side": 1,
        "trade": "Perpetuals only. At the 00:00 / 08:00 / 16:00 UTC funding stamp, if the last funding rate already paid was "
                 ">= +3 bp BUY at the stamp's first print (<= -3 bp: SELL); hold 5 minutes",
        "why": "A different family (time structure) with a mechanical explanation (positions avoid the payment and return after it). "
               "Design |t| about 5, same sign in all four years and all six coins. Gross size (about 8 bp) is below a taker round trip.",
        "part_b": {"bullish": "fund_post5_big (last funding >= +3 bp)", "bearish_mirror": "fund_post5_big (last funding <= -3 bp)"},
    },
]


def design_numbers(runs, name: str, h: int, side: int) -> dict:
    row = an.pooled(runs, name, h)
    out = {
        "events": row["events"],
        "days": row["days"],
        "gross_bp": side * row["gross_bp"],
        "baseline_bp": side * row["base_bp"],
        "excess_bp": side * row["excess_bp"],
        "t_day": side * row["t_day"],
        "hit_rate": row["hit"] if side > 0 else 1.0 - row["hit"],
        "net_perp_taker_bp": side * row["gross_bp"] - row["cost_perp_taker_bp"],
        "net_perp_maker_bound_bp": side * row["gross_bp"] - row["cost_perp_maker_bp"],
        "net_spot_taker_bp": side * row["gross_bp"] - row["cost_spot_taker_bp"],
    }
    out["by_year"] = {str(k): {"events": int(v["events"]), "gross_bp": side * v["gross_bp"], "t_day": side * v["t_day"]}
                      for k, v in an.by_year(runs, name, h).iterrows()}
    out["by_symbol"] = {str(k): {"events": int(v["events"]), "gross_bp": side * v["gross_bp"], "t_day": side * v["t_day"]}
                        for k, v in an.by_symbol(runs, name, h).iterrows()}
    return out


def main() -> None:
    path = HERE / "SELECTION.json"
    if path.exists():
        raise SystemExit("SELECTION.json already exists; the selection is made once")
    validation_outputs = list((HERE / "out").glob("validation/*")) + list((HERE / "out").glob("events_validation/*"))
    assert not validation_outputs, "validation outputs exist before the selection was written"
    runs = an.load("design", "um", PERPS)
    looks = len(an.table(runs))
    selected = []
    for item in SELECTED:
        entry = dict(item)
        entry["design"] = design_numbers(runs, item["name"], item["horizon"], item["trade_side"])
        selected.append(entry)
    record = {
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "basis": "DESIGN segment only: 1-minute bars before 2024-01-01, six USDT perpetuals pooled, day-clustered statistics",
        "looks_on_design_grid": looks,
        "selected": selected,
        "definition_names_to_evaluate": [item["name"] for item in SELECTED],
        "horizons_to_evaluate": {item["name"]: item["horizon"] for item in SELECTED},
        "part_b_mirror_names": ["move5_abs2pct_up", "move1_z3_up"],
        "validation_plan": [
            "Evaluate only the three (definition, horizon) pairs above on 2024-01-01 .. 2025-06-30, once.",
            "Same statistics as on design: events, days, gross bp on the trade side, matched baseline, day-clustered t of the excess, "
            "net after a perpetual taker round trip, perpetual maker bound, spot taker.",
            "Pooled over the six perpetuals; by coin and by calendar year as a breakdown; BTC/ETH spot as a cross-check of the two price-based structures.",
            "Execution robustness of the same pairs only: entry one and two minutes late, fills at the minute's volume-weighted price, buy at the next bar's high.",
            "Part B uses the event flags of the selected structures and of their mirrors (part_b_mirror_names) to time entries and exits; "
            "no forward-return statistic of the mirrors is computed on validation.",
            "A structure counts as confirmed if the validation estimate has the design sign with day-clustered |t| > 2 (one-sided test of three pre-registered pairs) "
            "and as paying costs only if gross minus the taker round trip is positive with the same margin.",
        ],
    }
    path.write_text(json.dumps(record, indent=1))
    lines = ["# Selection written before the validation segment was touched", "",
             f"Written (UTC): {record['written_at_utc']}", "", f"Looks on the design grid: {looks}", ""]
    for entry in selected:
        d = entry["design"]
        lines += [
            f"## {entry['name']} at {entry['horizon']} minutes",
            "",
            f"Trade: {entry['trade']}",
            "",
            f"Why: {entry['why']}",
            "",
            f"Design (six perpetuals pooled): {d['events']} events on {d['days']} days; gross {d['gross_bp']:.2f} bp on the trade side; "
            f"baseline {d['baseline_bp']:.2f} bp; excess {d['excess_bp']:.2f} bp; day-clustered t {d['t_day']:.2f}; "
            f"net after a taker round trip {d['net_perp_taker_bp']:.2f} bp; maker bound {d['net_perp_maker_bound_bp']:.2f} bp; "
            f"spot taker {d['net_spot_taker_bp']:.2f} bp.",
            "",
            "By year: " + "; ".join(f"{y}: {v['gross_bp']:.1f} bp (t {v['t_day']:.1f}, n {v['events']})" for y, v in d["by_year"].items()),
            "",
            "By coin: " + "; ".join(f"{s[:-4]}: {v['gross_bp']:.1f} bp (t {v['t_day']:.1f}, n {v['events']})" for s, v in d["by_symbol"].items()),
            "",
        ]
    lines += ["## Validation plan", ""] + [f"- {x}" for x in record["validation_plan"]] + [""]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
