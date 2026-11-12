"""Assemble results.json from the stored outputs (and print the tables as markdown on request).

    python -I make_results.py            -> results.json
    python -I make_results.py --tables   -> also prints every table as markdown to the terminal

Nothing is recomputed from bars here: the script only reads out/ and SELECTION.json, so every number in
results.json can be traced to a file written by run_events.py, event_level.py, autocorr.py,
selected_report.py, part_b.py, design_extras.py or survivorship_check.py.
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

import analysis as an  # noqa: E402
import eventstudy as es  # noqa: E402
import part_b as pb  # noqa: E402
from run_events import PERPS  # noqa: E402

TABLES = an.OUT / "tables"


def md_table(frame: pd.DataFrame, formats: dict | None = None, default: str = "{:.2f}") -> str:
    formats = formats or {}

    def cell(column, value):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return ""
        if column in formats:
            return formats[column].format(value)
        if isinstance(value, (bool, np.bool_)):
            return str(bool(value))
        if isinstance(value, (int, np.integer)):
            return f"{int(value)}"
        if isinstance(value, (float, np.floating)):
            return default.format(value)
        return str(value)

    lines = ["| " + " | ".join(str(c) for c in frame.columns) + " |", "|" + "|".join("---" for _ in frame.columns) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(cell(c, row[c]) for c in frame.columns) + " |")
    return "\n".join(lines)


def compact(pooled: pd.DataFrame, pairs: list[tuple[str, int]]) -> pd.DataFrame:
    """Report columns for chosen (definition, horizon) rows of the pooled design table."""
    index = pooled.set_index(["definition", "h"])
    rows = []
    for name, h in pairs:
        r = index.loc[(name, h)]
        rows.append({
            "definition": name,
            "h": h,
            "events": int(r.events),
            "days": int(r.days),
            "gross": r.gross_bp,
            "base": r.base_bp,
            "excess": r.excess_bp,
            "t_day": r.t_day,
            "net taker": r.net_perp_taker_bp,
            "net taker, better side": abs(r.gross_bp) - r.cost_perp_taker_bp,
            "maker bound, better side": abs(r.gross_bp) - r.cost_perp_maker_bp,
            "spot taker, better side": abs(r.gross_bp) - r.cost_spot_taker_bp,
        })
    return pd.DataFrame(rows)


INT = {"events": "{:d}", "days": "{:d}", "h": "{:d}"}


def family_overview(pooled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pooled = pooled.assign(net_best=pooled.gross_bp.abs() - pooled.cost_perp_taker_bp)
    for family, part in list(pooled.groupby("family")) + [("all", pooled)]:
        strong = part[part.t_day.abs() > 3]
        rows.append({
            "family": family,
            "looks": len(part),
            "|t| > 3": len(strong),
            "|t| > 4": int((part.t_day.abs() > 4).sum()),
            "median |gross| bp where |t| > 3": float(strong.gross_bp.abs().median()) if len(strong) else float("nan"),
            "largest |gross| bp where |t| > 3": float(strong.gross_bp.abs().max()) if len(strong) else float("nan"),
            "|t| > 3 and |gross| > taker round trip": int((strong.net_best > 0).sum()),
        })
    return pd.DataFrame(rows)


def sign_matrix(pooled: pd.DataFrame) -> pd.DataFrame:
    part = pooled[pooled.family == "4 sign"]
    rows = []
    for name in ("sign1", "sign5", "sign15", "sign60"):
        row = {"trailing window (min)": int(name[4:])}
        for h in es.HORIZONS:
            r = part[(part.definition == name) & (part.h == h)].iloc[0]
            row[f"h={h}"] = f"{r.excess_bp:.2f} ({r.t_day:.1f})"
        rows.append(row)
    return pd.DataFrame(rows)


def autocorr_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = json.loads((an.OUT / "autocorr_design_um.json").read_text())
    by_year, bounce = [], []
    years = ["2020", "2021", "2022", "2023", "all"]
    for w in (1, 5, 15, 60):
        row = {"window (min)": w}
        for year in years:
            cc = np.mean([item["years"][year][f"ac_cc_{w}"] for item in data if year in item["years"]])
            tr = np.mean([item["years"][year][f"ac_trade_{w}"] for item in data if year in item["years"]])
            row[year] = f"{cc:+.3f} / {tr:+.3f}"
        by_year.append(row)
    for item in data:
        r = item["years"]["all"]
        bounce.append({
            "coin": item["symbol"][:-4],
            "mean |1-min return| bp": r["abs_r1_bp"],
            "close->open step bp": r["step_bp"],
            "from the close print": f"{r['rev1_from_close_bp']:+.3f} ({r['rev1_from_close_t']:.1f})",
            "from the next open": f"{r['rev1_from_open_bp']:+.3f} ({r['rev1_from_open_t']:.1f})",
            "one minute later": f"{r['rev1_skip_one_bp']:+.3f} ({r['rev1_skip_one_t']:.1f})",
        })
    return pd.DataFrame(by_year), pd.DataFrame(bounce)


def breakdown_table(pairs: list[tuple[str, int, int]]) -> pd.DataFrame:
    """Gross bp on the trade side (t) by year and by coin, from design_breakdowns.csv."""
    data = pd.read_csv(TABLES / "design_breakdowns.csv", dtype={"key": str})
    rows = []
    for name, h, side in pairs:
        part = data[(data.definition == name) & (data.h == h)]
        row = {"definition": name, "h": h, "side": "as hypothesised" if side > 0 else "opposite"}
        for key in ["2020", "2021", "2022", "2023"] + PERPS:
            r = part[part.key == key]
            if len(r):
                label = key[:-4] if key.endswith("USDT") else key
                row[label] = f"{side * float(r.gross_bp.iloc[0]):.1f} ({side * float(r.t_day.iloc[0]):.1f})"
        rows.append(row)
    return pd.DataFrame(rows)


def selected_table(design: dict, validation: dict) -> pd.DataFrame:
    rows = []
    for d, v in zip(design["pairs"], validation["pairs"]):
        for label, src in (("design", d), ("validation", v)):
            p = src["perp_pooled"]
            rows.append({
                "pair": f"{d['name']} @ {d['horizon']}m",
                "segment": label,
                "events": p["events"],
                "days": p["days"],
                "gross": p["gross_bp"],
                "base": p["baseline_bp"],
                "excess": p["excess_bp"],
                "t_day": p["t_day"],
                "hit": p["hit_rate"],
                "net taker": p["net_perp_taker_bp"],
                "t of net": p["t_net_perp_taker"],
                "maker bound": p["net_perp_maker_bound_bp"],
                "spot taker": p["net_spot_taker_bp"],
            })
    return pd.DataFrame(rows)


def execution_table(design: dict, validation: dict) -> pd.DataFrame:
    rows = []
    for d, v in zip(design["pairs"], validation["pairs"]):
        for key in ("next_open (convention)", "one minute late", "two minutes late", "minute vwap to minute vwap", "worst print of the first minute"):
            a, b = d["execution_perp"][key], v["execution_perp"][key]
            rows.append({
                "pair": f"{d['name']} @ {d['horizon']}m",
                "fill assumption": key,
                "design mean bp": a["mean_bp"],
                "design t": a["t_day"],
                "design median bp": a["median_bp"],
                "validation mean bp": b["mean_bp"],
                "validation t": b["t_day"],
                "validation median bp": b["median_bp"],
            })
    return pd.DataFrame(rows)


def split_table(design: dict, validation: dict) -> pd.DataFrame:
    rows = []
    for d, v in zip(design["pairs"], validation["pairs"]):
        row_d = {"pair": f"{d['name']} @ {d['horizon']}m", "segment": "design"}
        row_v = {"pair": f"{d['name']} @ {d['horizon']}m", "segment": "validation"}
        for row, src in ((row_d, d), (row_v, v)):
            p = src["perp_pooled"]
            for year in ("2020", "2021", "2022", "2023", "2024", "2025"):
                if year in p["by_year"]:
                    row[year] = f"{p['by_year'][year]['gross_bp']:.1f} ({p['by_year'][year]['t_day']:.1f})"
                else:
                    row[year] = ""
            for symbol in PERPS:
                if symbol in p["by_symbol"]:
                    row[symbol[:-4]] = f"{p['by_symbol'][symbol]['gross_bp']:.1f} ({p['by_symbol'][symbol]['t_day']:.1f})"
        rows += [row_d, row_v]
    return pd.DataFrame(rows).fillna("")


def cross_market_table(design: dict, validation: dict) -> pd.DataFrame:
    rows = []
    for d, v in zip(design["pairs"], validation["pairs"]):
        for label, src in (("design", d), ("validation", v)):
            for market in ("perp_btc_eth", "spot_btc_eth"):
                q = src[market]
                if q:
                    rows.append({"pair": f"{d['name']} @ {d['horizon']}m", "segment": label, "market": market.replace("_btc_eth", " BTC+ETH"),
                                 "events": q["events"], "gross": q["gross_bp"], "excess": q["excess_bp"], "t_day": q["t_day"],
                                 "net perp taker": q["net_perp_taker_bp"], "net spot taker": q["net_spot_taker_bp"]})
    return pd.DataFrame(rows)


PART_B_COLUMNS = ["side", "structure", "window_min", "trades", "days", "triggered", "median_wait_min", "mean_bp", "median_bp", "t_day", "worse",
                  "delay_mean_bp", "delay_t", "vs_delay_bp", "vs_delay_t", "vs_twap_bp", "vs_twap_t"]
PART_B_FORMATS = {"trades": "{:d}", "days": "{:d}", "window_min": "{:d}", "triggered": "{:.0%}", "worse": "{:.0%}", "median_wait_min": "{:.0f}",
                  "mean_bp": "{:.1f}", "median_bp": "{:.1f}", "delay_mean_bp": "{:.1f}", "vs_delay_bp": "{:.1f}", "vs_twap_bp": "{:.1f}"}


def part_b_tables() -> dict:
    folder = an.OUT / "part_b"
    out = {}
    for market in ("spot", "um"):
        for label in ("rule_days", "all_days"):
            frames = {}
            for segment in ("design", "validation"):
                path = folder / f"timing_{segment}_{market}_{label}.parquet"
                if path.exists():
                    frames[segment] = pd.read_parquet(path)
            if not frames:
                continue
            for segment, table in frames.items():
                out[f"{segment}/{market}/{label}"] = pb.summarise(table)
            if len(frames) == 2:
                out[f"both/{market}/{label}"] = pb.summarise(pd.concat(list(frames.values()), ignore_index=True))
    return out


def order(frame: pd.DataFrame) -> pd.DataFrame:
    key = frame.side.map({"entry (buy)": 0, "exit (sell)": 1}) * 100 + frame.structure.map({"move1_z3": 0, "move5_abs2pct": 1, "fund_post5_big": 2}) * 10 + (frame.window_min > 240)
    return frame.iloc[np.argsort(key.to_numpy(), kind="stable")]


TESTS = {
    "file": "test_eventstudy.py (16 tests, synthetic data only, run before any real bar was used)",
    "no_look_ahead": "for all 190 definitions and every feature, values up to bar t are identical when later bars are replaced by a different history "
                     "(with its own gaps, halts and funding prints) or cut off; a deliberately centred window fails the same check",
    "planted_effect": "a 20 bp drift over the 15 minutes after a planted volume spike is recovered at 15, 60, 240 and 1440 minutes and pro rata at 1 and 5 minutes, "
                      "within 4 standard errors; nothing is found without the plant; a move inside the event bar or between its close and the next open is not credited",
    "gaps": "every forward return whose window touches a missing bar or a halt (>= 5 zero-volume bars) is dropped",
    "null_rejection_rate_at_5pct": {"h=1": {"day_clustered": 0.047, "naive": 0.053}, "h=60": {"day_clustered": 0.063, "naive": 0.707},
                                    "h=240": {"day_clustered": 0.043, "naive": 0.727}},
    "se_calibration_every_bar_a_buy_event (sd of estimate / reported se)": {"h=1440, day clustering only": 1.18, "h=1440, 3 Bartlett lags": 1.01,
                                                                             "h=240, 1 lag": 0.99, "h=60, 1 lag": 1.02},
    "independent_recomputation_of_headline_pair": {"design": "7966 events, 40.43 bp, t 10.07", "validation": "1004 events, 47.12 bp, t 5.02"},
}

CAVEATS = [
    "Execution in crash minutes is untested: fills are assumed at the first print of the next minute with 1-2 bp slippage. The validated rebound is intact at the "
    "minute's volume-weighted price and one or two minutes late, negative at the worst print of the first minute, and roughly break-even with 30 bp of extra round-trip cost.",
    "Rare and lumpy: on validation 1,004 events on 157 of 547 days over six coins; the 5 best days carry 30% and the 20 best 72% of the total. Events bunch across coins.",
    "Tail risk: single 15-minute losses of 15% to 30% are in the sample (19 May 2021, 12 March 2020). About 4% of design trades lose more than 5%.",
    "Universe: six coins that are large today. An hourly check on all liquid USDT perpetuals (design only) shows about one third of the rebound and none for perpetuals that were later delisted.",
    "Design-only observation, not validated: the first 2% fall of a cascade rebounds by only about 12 bp; the average is carried by later legs and by the largest falls.",
    "Maker figures are upper bounds: a resting order fills when the price trades through it and the signals that run away unfilled are the best ones (shown with a fill simulation on design).",
    "Halts are excluded after the fact: an event followed by an exchange outage is dropped, which a live trader cannot do.",
    "Funding is not charged (irrelevant for the 5- and 15-minute pairs). Standard errors are slightly liberal at the one-day horizon; nothing is claimed there.",
    "Part B has little power on rule-change days alone (214 entries, sd about 100 bp); the all-days version (about 4,000 observations) resolves about 5 bp and still shows nothing.",
    "Protocol: the execution check of the funding pair was run after the selection was written and before validation was loaded; it already showed the first-print problem on design. "
    "An initial scan of bar timestamps and zero-volume counts covered the validation period; no price or return of that period was examined before the validation run.",
    "Only 1-minute bars were available: no order book, no trades inside the minute, no other exchanges.",
]


def plain_language(pooled: pd.DataFrame, design: dict, validation: dict, part_b: dict, extras: dict, survivorship: dict, cross_corr: float, n_cross: int) -> dict:
    """The findings in plain words, section by section, with every number taken from the stored tables."""
    idx = pooled.set_index(["definition", "h"])

    def g(name: str, h: int) -> float:  # size on the side the data favour
        return abs(float(idx.loc[(name, h), "gross_bp"]))

    def t(name: str, h: int) -> float:
        return abs(float(idx.loc[(name, h), "t_day"]))

    def pb(key: str, structure: str, window: int, column: str, side: str = "entry (buy)") -> float:
        frame = part_b[key]
        return float(frame[(frame.side == side) & (frame.structure == structure) & (frame.window_min == window)][column].iloc[0])

    d = [pair["perp_pooled"] for pair in design["pairs"]]
    v = [pair["perp_pooled"] for pair in validation["pairs"]]
    xd = [pair["execution_perp"] for pair in design["pairs"]]
    xv = [pair["execution_perp"] for pair in validation["pairs"]]
    an_ = extras["cascade_anatomy_design"]
    stress = {(row["segment"], row["costs"]): row for row in extras["cost_stress"]}
    surv = {(row["fall"], row["hold_hours"], row["group"]): row for row in survivorship["rows"]}
    five = "<= -5% in one hour"
    hours = pooled[pooled.definition.str.startswith("hour")]
    quarter = [g(f"minute{m:02d}", 1) for m in (0, 15, 30, 45)]
    rest = {(row["definition"], row["order_alive_min"], row["bp_below_first_print"]): row for row in extras["resting_orders_design"]}
    return {
        "method": (
            "Six USDT perpetuals (BTC, ETH, SOL, BNB, XRP, DOGE) on 1-minute bars; BTC/ETH spot and PAXG spot as cross-checks. Design = bars before 2024-01-01, "
            "validation = 2024-01-01 to 2025-06-30, hold-out never loaded. An event is flagged at the close of bar t from bars up to t and filled at the open of bar t+1; "
            "forward returns run from that open over 1, 5, 15, 60, 240 and 1440 minutes. Missing bars, and halts printed as runs of five or more zero-volume bars, are gaps: "
            "any forward return whose window touches one is dropped. Statistics are sums per calendar day, clustered by day across coins, and measured against the mean forward "
            "return of bars in the same coin, year, hour of day and volatility state. Round-trip costs: perpetual taker 12 bp (BTC, ETH) or 14 bp, maker bound 4 bp, spot taker 22 or 24 bp."
        ),
        "family_1_volume_and_trade_spikes": (
            f"Bars with quote volume or trade count above 5x or 10x the median of the previous 60 bars. After a down bar on a spike the price recovers a little "
            f"({g('volspike5_all_dn', 5):.1f} bp over 5 minutes at 5x volume, {g('trdspike10_all_dn', 5):.1f} bp at 10x trade count); after an up bar it continues by a similar amount "
            f"({g('volspike5_all_up', 5):.1f} and {g('trdspike10_all_up', 5):.1f} bp). Wick shape adds nothing: a down bar that closes at its low and one with a long lower wick are followed "
            f"by the same recovery ({g('volspike5_cont_dn', 5):.1f} and {g('volspike5_exh_dn', 5):.1f} bp at 5 minutes). The largest effects are bars closing at their extreme on a 10x trade-count spike: "
            f"{g('trdspike10_cont_up', 5):.0f} bp at 5 minutes and {g('trdspike10_cont_up', 60):.0f} bp at an hour after up bars, below a taker round trip, and the up side lives in 2021 and in XRP and DOGE "
            "(2022: 0.9 bp, BTC: -0.7 bp at 5 minutes). Two of 216 looks exceed a taker round trip with |t| > 3 (at 4 hours and one day, |t| 3.1 and 3.6), which is not evidence with this many looks."
        ),
        "family_2_order_flow_imbalance": (
            f"Taker-buy share of quote volume over the last 1, 5 and 15 minutes, as a z-score against its own trailing day. Extreme buying or selling is followed by a small move the other way: "
            f"{g('ofi1_z2_both', 5):.1f} to {g('ofi15_z2_both', 15):.1f} bp over 5 to 15 minutes at |z| > 2 (|t| up to {t('ofi5_z2_both', 5):.1f}), {g('ofi15_z2_both', 60):.1f} bp at an hour. "
            f"The 1-minute extreme at |z| > 3 ({int(idx.loc[('ofi1_z3_both', 5), 'events'])} events) shows nothing. Pressure does not continue, and the reversal is an order of magnitude below costs."
        ),
        "family_3_large_moves": (
            f"Returns over 1, 5 and 15 minutes beyond 3, 5 and 8 trailing standard deviations, and 5-minute moves of 2% or more. After falls the price rebounds: {g('move1_z3_dn', 15):.1f} bp over 15 minutes "
            f"after a 3-sigma minute, {g('move1_z5_dn', 15):.1f} bp after 5 sigma, {g('move1_z8_dn', 15):.1f} bp after 8 sigma, {g('move5_abs2pct_dn', 15):.1f} bp after a 2% fall in 5 minutes "
            f"({g('move5_abs2pct_dn', 60):.0f} bp at one hour), with the same sign in every design year and every coin. After rises it tends to continue ({g('move1_z3_up', 15):.1f}, {g('move1_z5_up', 15):.1f}, "
            f"{g('move1_z8_up', 15):.1f} bp), but only in 2021 and in XRP and DOGE: zero in 2022 and on BTC. These are the only looks whose size exceeds a taker round trip. At one day nothing is significant."
        ),
        "family_4_autocorrelation_and_bid_ask_bounce": (
            f"Trading the sign of the trailing return on every bar: +{g('sign1', 1):.2f} bp of continuation at one minute (t {t('sign1', 1):.1f}), then {g('sign1', 5):.2f} to {g('sign5', 5):.2f} bp of reversal at 5 minutes "
            f"(|t| {t('sign1', 5):.1f} to {t('sign5', 5):.1f}); nothing from a 60-minute return and nothing at one day. BTC and ETH have no 1-minute autocorrelation to speak of. Where a 1-minute reversal exists "
            "when measured from the closing print (XRP: +0.26 bp per trade; DOGE in 2020 and 2023), most of it disappears when the trade starts at the next open (XRP: +0.04 bp): that part is bid-ask bounce. "
            "The step from a close to the next open, about one tick, is 0.1 bp on BTC and about 1 bp on XRP, DOGE and SOL. Five-minute returns are negatively autocorrelated in 2021-2023 (-0.03 to -0.06)."
        ),
        "family_5_time_structure": (
            f"No hour of the day reaches |t| = 3 (largest {hours.t_day.abs().max():.1f} of 24). The first minute of each quarter hour (:00, :15, :30, :45) returns +{min(quarter):.2f} to +{max(quarter):.2f} bp "
            f"(t about {t('minute00', 1):.1f} each, on every coin and on spot), partly given back in the next two minutes; it was 1.3-1.8 bp in 2021 and 0.1-0.2 bp in 2023. Around funding stamps there is nothing "
            f"unconditional; when the last paid funding was at least 3 bp the price moves {g('fund_post5_big', 5):.1f} bp with the payers in the 5 minutes after the stamp (t {t('fund_post5_big', 5):.1f}), "
            "which the execution check shows to be complete within the first prints after the stamp."
        ),
        "cross_checks_spot_and_gold": (
            f"The same grid on BTC and ETH spot agrees with the perpetuals of the same coins (correlation of excess returns over {n_cross} looks: {cross_corr:.2f}). On PAXG spot about a quarter of the minutes "
            "have no trade and every short horizon shows reversal of 1 to 5 bp, which is what prints alternating between the two sides of a wide spread look like, against a 24 bp round trip; "
            "a 2% fall in 5 minutes occurred 6 times in the design period. There is nothing to trade on 1-minute gold-token bars."
        ),
        "execution_checks": (
            f"Rebound after a 2% fall (design, 15 minutes): {xd[0]['next_open (convention)']['mean_bp']:.1f} bp at the next open, {xd[0]['one minute late']['mean_bp']:.1f} bp one minute late, "
            f"{xd[0]['two minutes late']['mean_bp']:.1f} bp two minutes late, {xd[0]['minute vwap to minute vwap']['mean_bp']:.1f} bp at the minute's volume-weighted price, "
            f"{xd[0]['worst print of the first minute']['mean_bp']:.1f} bp at the worst print of the first minute. Post-funding move (design, 5 minutes): {xd[2]['next_open (convention)']['mean_bp']:.1f} bp at the first print, "
            f"{xd[2]['one minute late']['mean_bp']:.1f} bp one minute late, {xd[2]['minute vwap to minute vwap']['mean_bp']:.1f} bp at the minute's volume-weighted price: a first-print artefact. "
            f"Resting orders (design): a bid at the first print of the next minute, alive for one minute, fills {rest[('move1_z3_dn', 1, 0.0)]['fill_rate'] * 100:.0f}% of the 3-sigma dips; the filled ones earn "
            f"{rest[('move1_z3_dn', 1, 0.0)]['gross_bp_if_filled']:.1f} bp, the ones that ran away unfilled would have earned {rest[('move1_z3_dn', 1, 0.0)]['gross_bp_of_unfilled_signals']:.1f} bp, and the net per signal "
            f"with a maker entry and a taker exit is {rest[('move1_z3_dn', 1, 0.0)]['net_bp_per_signal']:.1f} bp, not the +{d[1]['net_perp_maker_bound_bp']:.1f} bp of the flat maker bound."
        ),
        "selection": (
            "Three (definition, horizon) pairs were chosen on design evidence and written to SELECTION.json before any validation bar was loaded: buy after a fall of at least 2% in 5 minutes and hold 15 minutes "
            "(largest day-clustered |t| among the looks that exceed a taker round trip; the pre-stated hypothesis was continuation and the data said rebound); buy after a 1-minute fall beyond 3 sigma and hold 15 minutes "
            "(the most frequent bullish structure with |t| > 7, selected with its weaknesses stated: below costs, weak in 2022 and on BTC); trade with the payers at the first print after a funding stamp "
            "when the last paid funding was at least 3 bp and hold 5 minutes (a different family with a mechanical explanation). Confirmation rule, written with the selection: same sign with day-clustered |t| > 2; "
            "pays costs only if gross minus the taker round trip is positive with the same margin. The execution check of the funding pair was run after the selection and before validation; the selection was left as written."
        ),
        "validation": (
            f"Pair 1: {v[0]['events']} events on {v[0]['days']} days, {v[0]['gross_bp']:.1f} bp gross (design {d[0]['gross_bp']:.1f}), day-clustered t {v[0]['t_day']:.1f}, hit rate {v[0]['hit_rate'] * 100:.0f}%, "
            f"net of taker costs {v[0]['net_perp_taker_bp']:.1f} bp (t {v[0]['t_net_perp_taker']:.1f}); positive in 2024 and 2025 and in each coin; {xv[0]['one minute late']['mean_bp']:.1f} bp one minute late, "
            f"{xv[0]['minute vwap to minute vwap']['mean_bp']:.1f} bp at the minute's volume-weighted price, {xv[0]['worst print of the first minute']['mean_bp']:.1f} bp at the worst print; "
            f"{xv[0]['concentration']['top5_days_share_of_total'] * 100:.0f}% of the total from the 5 best days and {xv[0]['concentration']['top20_days_share_of_total'] * 100:.0f}% from the 20 best. Confirmed. "
            f"Pair 2: {v[1]['gross_bp']:.1f} bp gross (design {d[1]['gross_bp']:.1f}), t {v[1]['t_day']:.1f}, net {v[1]['net_perp_taker_bp']:.1f} bp, maker bound {v[1]['net_perp_maker_bound_bp']:.1f} bp: "
            f"the gross effect persists at half size and does not pay. Pair 3: {v[2]['gross_bp']:.1f} bp (design {d[2]['gross_bp']:.1f}), t {v[2]['t_day']:.1f}: not confirmed."
        ),
        "part_b_entry_timing": (
            f"Days on which the 50/100/200-day rule raises a coin's weight: {pb('design/spot/rule_days', 'move1_z3', 240, 'trades'):.0f} on design, {pb('validation/spot/rule_days', 'move1_z3', 240, 'trades'):.0f} on validation "
            f"(BTC and ETH spot). Waiting up to 4 hours for the first 1-minute fall beyond 3 sigma instead of buying at the 00:00 UTC open: design mean {pb('design/spot/rule_days', 'move1_z3', 240, 'mean_bp'):.1f} bp, "
            f"median {pb('design/spot/rule_days', 'move1_z3', 240, 'median_bp'):.1f}, t {pb('design/spot/rule_days', 'move1_z3', 240, 't_day'):.2f}, worse price in {pb('design/spot/rule_days', 'move1_z3', 240, 'worse') * 100:.0f}% of cases; "
            f"validation mean {pb('validation/spot/rule_days', 'move1_z3', 240, 'mean_bp'):.1f} bp, median {pb('validation/spot/rule_days', 'move1_z3', 240, 'median_bp'):.1f}, t {pb('validation/spot/rule_days', 'move1_z3', 240, 't_day'):.2f}, "
            f"worse in {pb('validation/spot/rule_days', 'move1_z3', 240, 'worse') * 100:.0f}%. Simply buying at 04:00 gave {pb('design/spot/rule_days', 'move1_z3', 240, 'delay_mean_bp'):.1f} and "
            f"{pb('validation/spot/rule_days', 'move1_z3', 240, 'delay_mean_bp'):.1f} bp, so the structure adds {pb('design/spot/rule_days', 'move1_z3', 240, 'vs_delay_bp'):.1f} bp (t {pb('design/spot/rule_days', 'move1_z3', 240, 'vs_delay_t'):.2f}) "
            f"and {pb('validation/spot/rule_days', 'move1_z3', 240, 'vs_delay_bp'):.1f} bp (t {pb('validation/spot/rule_days', 'move1_z3', 240, 'vs_delay_t'):.2f}). Up to 24 hours: {pb('design/spot/rule_days', 'move1_z3', 1440, 'mean_bp'):.1f} bp "
            f"(t {pb('design/spot/rule_days', 'move1_z3', 1440, 't_day'):.2f}) and {pb('validation/spot/rule_days', 'move1_z3', 1440, 'mean_bp'):.1f} bp (t {pb('validation/spot/rule_days', 'move1_z3', 1440, 't_day'):.2f}). "
            f"On every day of both segments ({pb('both/spot/all_days', 'move1_z3', 240, 'trades'):.0f} observations) the 4-hour wait gives {pb('both/spot/all_days', 'move1_z3', 240, 'mean_bp'):.1f} bp "
            f"(t {pb('both/spot/all_days', 'move1_z3', 240, 't_day'):.2f}). The 2% fall almost never occurs inside the window ({pb('both/spot/rule_days', 'move5_abs2pct', 240, 'triggered') * 100:.0f}% within 4 hours, "
            f"{pb('both/spot/rule_days', 'move5_abs2pct', 1440, 'triggered') * 100:.0f}% within 24), so that trader ends up buying at the end of the window ({pb('both/spot/rule_days', 'move5_abs2pct', 1440, 'mean_bp'):.1f} bp, "
            f"t {pb('both/spot/rule_days', 'move5_abs2pct', 1440, 't_day'):.2f} for 24 hours). Exits, selling at the first 1-minute rise beyond 3 sigma within 4 hours: {pb('both/spot/rule_days', 'move1_z3', 240, 'mean_bp', 'exit (sell)'):.1f} bp "
            f"(t {pb('both/spot/rule_days', 'move1_z3', 240, 't_day', 'exit (sell)'):.2f}, {pb('both/spot/rule_days', 'move1_z3', 240, 'trades', 'exit (sell)'):.0f} exits). No measurable improvement. For a price close to a random walk "
            "no waiting rule changes the expected fill; the most a structure can add is its own forward return, a few basis points here, against about 100 bp of noise from a 4-hour wait."
        ),
        "validated_structure_closer_look_design_only": (
            f"Not validated; looked at after the validation run, on design data. First 2% fall of a cascade (none on that coin in the previous hour): {an_['first event in 60 minutes (same coin)']['mean_bp']:.1f} bp "
            f"(median {an_['first event in 60 minutes (same coin)']['median_bp']:.1f}); later legs: {an_['later legs of the same cascade']['mean_bp']:.1f} bp; one position at a time per coin: "
            f"{an_['one position at a time per coin']['mean_bp']:.1f} bp. By size of the fall: {an_['by size of the 5-minute fall']['-250 to -200 bp']['mean_bp']:.0f} bp for falls of 2.0-2.5%, "
            f"{an_['by size of the 5-minute fall']['-500 to -300 bp']['mean_bp']:.0f} bp for 3-5%, {an_['by size of the 5-minute fall']['below -500 bp']['mean_bp']:.0f} bp beyond 5%. Single trades net of costs: "
            f"standard deviation {an_['single trades net of taker costs']['sd_bp']:.0f} bp, worst {an_['single trades net of taker costs']['worst_bp']:.0f} bp. Extra round-trip cost that the pair can bear: per event "
            f"{stress[('design', '+20 bp per round trip')]['bp_per_event']:.1f} bp remain on design and {stress[('validation', '+20 bp per round trip')]['bp_per_event']:.1f} bp on validation with +20 bp, "
            f"{stress[('design', '+30 bp per round trip')]['bp_per_event']:.1f} and {stress[('validation', '+30 bp per round trip')]['bp_per_event']:.1f} bp with +30 bp. Hourly check on all liquid USDT perpetuals including "
            f"delisted ones (buy at the next hourly open after an hourly fall of 5%, hold one hour): {surv[(five, 1, 'all liquid perpetuals')]['gross_bp']:.0f} bp (t {surv[(five, 1, 'all liquid perpetuals')]['t_day']:.1f}) "
            f"on {surv[(five, 1, 'all liquid perpetuals')]['coins']} perpetuals, {surv[(five, 1, 'the six coins of the 1-minute study')]['gross_bp']:.0f} bp (t {surv[(five, 1, 'the six coins of the 1-minute study')]['t_day']:.1f}) "
            f"on the six coins of this study, {surv[(five, 1, 'delisted before 2024 (dead coins)')]['gross_bp']:.0f} bp on perpetuals that were later delisted: the rebound belongs to large liquid coins."
        ),
        "what_it_means_for_the_idea": (
            "As a source of trades, ordinary 1-minute structures do not work for an account that pays taker fees: the structure is there, and it is worth tenths of a basis point to a few basis points per trade. "
            "As an entry timer for a daily trade they do not work either: entering at the daily open is as good as anything tested. The one thing that pays on paper is liquidity provision in a crash "
            "(buying a fall of 2% within 5 minutes): rare, fat-tailed and decided by execution in the worst minute of the day. It deserves its own study with order-book or trade data, a position cap "
            "and the locked hold-out before any money is put on it."
        ),
    }


def conclusions(design: dict, validation: dict, part_b: dict, looks: int, n_t3: int, n_pays: int, median_gross: float, pays_note: str, day_note: str) -> list[str]:
    d = [pair["perp_pooled"] for pair in design["pairs"]]
    v = [pair["perp_pooled"] for pair in validation["pairs"]]

    def pb(key: str, structure: str, window: int, column: str, side: str = "entry (buy)") -> float:
        frame = part_b[key]
        return float(frame[(frame.side == side) & (frame.structure == structure) & (frame.window_min == window)][column].iloc[0])

    return [
        f"{looks} definition x horizon looks on design (six USDT perpetuals, 2020-2023) plus 11 descriptive autocorrelation statistics. {n_t3} looks have a day-clustered |t| above 3 "
        f"(about {looks * 0.0027:.1f} expected by chance); their median size is {median_gross:.1f} bp against a taker round trip of 12-14 bp. {n_pays} exceed a taker round trip: {pays_note}. "
        f"{day_note}",
        "Volume and trade-count spikes, wick shape (exhaustion versus continuation), order-flow imbalance, 3-sigma moves, return autocorrelation, hour of day, minute of the hour and "
        "funding windows are statistically visible and economically 5 to 100 times too small. Wick shape adds nothing. Extreme taker pressure reverses slightly rather than continues.",
        "At one minute BTC and ETH have no autocorrelation to speak of (reversal trade within 0.01 bp of zero). Where a 1-minute reversal exists when measured from the closing print "
        "(XRP: +0.26 bp per trade over the design period; DOGE in 2020 and 2023), most of it disappears when the trade starts at the next open (XRP: +0.04 bp, so 83% is gone): "
        "it is bid-ask bounce and cannot be captured. Five-minute returns are negatively autocorrelated in 2021-2023 (-0.03 to -0.06), worth about 0.3 bp per trade.",
        f"Selected pair 1 (buy a fall of >= 2% in 5 minutes, hold 15 minutes): design {d[0]['events']} events, {d[0]['gross_bp']:.1f} bp gross, t {d[0]['t_day']:.1f}, "
        f"net of taker costs {d[0]['net_perp_taker_bp']:.1f} bp; validation {v[0]['events']} events, {v[0]['gross_bp']:.1f} bp, t {v[0]['t_day']:.1f}, net {v[0]['net_perp_taker_bp']:.1f} bp "
        f"(t {v[0]['t_net_perp_taker']:.1f}). Confirmed.",
        f"Selected pair 2 (buy a 1-minute fall beyond 3 sigma, hold 15 minutes): design {d[1]['gross_bp']:.1f} bp gross (t {d[1]['t_day']:.1f}), net {d[1]['net_perp_taker_bp']:.1f} bp; "
        f"validation {v[1]['gross_bp']:.1f} bp (t {v[1]['t_day']:.1f}), net {v[1]['net_perp_taker_bp']:.1f} bp. The gross effect persists at half size and does not pay.",
        f"Selected pair 3 (after a funding stamp with |last funding| >= 3 bp, hold 5 minutes): design {d[2]['gross_bp']:.1f} bp (t {d[2]['t_day']:.1f}); validation {v[2]['gross_bp']:.1f} bp "
        f"(t {v[2]['t_day']:.1f}). Not confirmed, and zero under any fill other than the very first print after the stamp.",
        f"Part B (entry timing of the daily trend rule, BTC and ETH spot): waiting up to 4 hours for the first 3-sigma dip instead of buying at the 00:00 UTC open gave "
        f"{pb('design/spot/rule_days', 'move1_z3', 240, 'mean_bp'):.1f} bp on design (t {pb('design/spot/rule_days', 'move1_z3', 240, 't_day'):.2f}, "
        f"{pb('design/spot/rule_days', 'move1_z3', 240, 'trades'):.0f} entries) and {pb('validation/spot/rule_days', 'move1_z3', 240, 'mean_bp'):.1f} bp on validation "
        f"(t {pb('validation/spot/rule_days', 'move1_z3', 240, 't_day'):.2f}, {pb('validation/spot/rule_days', 'move1_z3', 240, 'trades'):.0f} entries); "
        f"the structure adds {pb('both/spot/rule_days', 'move1_z3', 240, 'vs_delay_bp'):.1f} bp over simply waiting until 04:00 (t {pb('both/spot/rule_days', 'move1_z3', 240, 'vs_delay_t'):.2f}). "
        "No measurable improvement; exits likewise.",
        "Verdict: edge_after_taker_costs in a narrow sense (one rare crash-time structure, on paper, under the cost convention of the study). "
        "Frequent 1-minute structures do not pay costs and do not improve the entry of a daily trade.",
    ]


def main() -> None:
    pooled = pd.read_csv(TABLES / "design_perp_pooled.csv")
    selection = json.loads((HERE / "SELECTION.json").read_text())
    design = json.loads((an.OUT / "selected_design.json").read_text())
    validation = json.loads((an.OUT / "selected_validation.json").read_text())
    cross = pd.read_csv(TABLES / "design_spot_vs_perp_btc_eth.csv")
    paxg = an.table(an.load("design", "spot", ["PAXGUSDT"]))
    part_b = part_b_tables()

    T: dict[str, str] = {}
    T["overview"] = md_table(family_overview(pooled), {"looks": "{:d}", "|t| > 3": "{:d}", "|t| > 4": "{:d}", "|t| > 3 and |gross| > taker round trip": "{:d}"})

    spikes = [("volspike5_all_up", 5), ("volspike5_all_dn", 5), ("volspike10_all_up", 5), ("volspike10_all_dn", 5),
              ("trdspike10_all_up", 5), ("trdspike10_all_dn", 5), ("trdspike10_all_up", 60), ("trdspike10_all_dn", 60),
              ("volspike5_cont_up", 5), ("volspike5_exh_up", 5), ("volspike5_cont_dn", 5), ("volspike5_exh_dn", 5),
              ("trdspike10_cont_up", 5), ("trdspike10_exh_up", 5), ("trdspike10_cont_dn", 5), ("trdspike10_exh_dn", 5),
              ("volspike5_all_both", 1), ("volspike5_all_both", 5), ("volspike5_all_both", 60), ("volspike5_all_both", 1440)]
    T["spikes"] = md_table(compact(pooled, spikes), INT)
    ofi = [("ofi1_z2_both", 1), ("ofi1_z2_both", 5), ("ofi1_z2_buy", 5), ("ofi1_z2_sell", 5), ("ofi5_z2_both", 5), ("ofi5_z2_buy", 5), ("ofi5_z2_sell", 5),
           ("ofi5_z2_both", 60), ("ofi15_z2_both", 15), ("ofi15_z2_both", 60), ("ofi15_z2_both", 1440), ("ofi1_z3_both", 5), ("ofi5_z3_both", 5), ("ofi15_z3_both", 15)]
    T["ofi"] = md_table(compact(pooled, ofi), INT)
    moves = [("move1_z3_up", 5), ("move1_z3_dn", 5), ("move1_z3_up", 15), ("move1_z3_dn", 15), ("move1_z3_up", 60), ("move1_z3_dn", 60),
             ("move1_z5_up", 15), ("move1_z5_dn", 15), ("move1_z8_up", 15), ("move1_z8_dn", 15), ("move1_z8_up", 60), ("move1_z8_dn", 60),
             ("move5_z3_up", 15), ("move5_z3_dn", 15), ("move5_z8_up", 15), ("move5_z8_dn", 15), ("move15_z3_up", 60), ("move15_z3_dn", 60),
             ("move5_abs2pct_up", 5), ("move5_abs2pct_dn", 5), ("move5_abs2pct_up", 15), ("move5_abs2pct_dn", 15), ("move5_abs2pct_up", 60), ("move5_abs2pct_dn", 60),
             ("move5_abs2pct_up", 240), ("move5_abs2pct_dn", 240), ("move5_abs2pct_up", 1440), ("move5_abs2pct_dn", 1440)]
    T["moves"] = md_table(compact(pooled, moves), INT)
    T["sign"] = md_table(sign_matrix(pooled), {"trailing window (min)": "{:d}"})
    ac_year, bounce = autocorr_tables()
    T["autocorr"] = md_table(ac_year, {"window (min)": "{:d}"})
    T["bounce"] = md_table(bounce, {"mean |1-min return| bp": "{:.2f}", "close->open step bp": "{:.2f}"})

    hours = pooled[pooled.definition.str.startswith("hour")].sort_values("t_day")
    hour_rows = pd.concat([hours.head(3), hours.tail(3)])
    T["hours"] = md_table(compact(pooled, [(r.definition, int(r.h)) for r in hour_rows.itertuples()])[["definition", "h", "events", "days", "gross", "base", "excess", "t_day"]], INT)
    T["hours_max_t"] = f"{hours.t_day.abs().max():.2f}"
    minute_names = [f"minute{m:02d}" for m in (59, 0, 1, 2, 14, 15, 16, 17, 29, 30, 31, 44, 45, 46, 47)]
    T["minutes"] = md_table(compact(pooled, [(n, 1) for n in minute_names])[["definition", "h", "events", "days", "gross", "base", "excess", "t_day"]], INT, "{:.3f}")
    funding = [(f"fund_{when}{k}_{variant}", k) for when in ("pre", "post") for k in (5, 15, 60) for variant in ("all", "signed", "big")]
    T["funding"] = md_table(compact(pooled, funding)[["definition", "h", "events", "days", "gross", "base", "excess", "t_day", "net taker", "maker bound, better side"]], INT)

    T["breakdown"] = md_table(breakdown_table([
        ("move5_abs2pct_dn", 15, -1), ("move5_abs2pct_dn", 60, -1), ("move1_z8_dn", 15, -1), ("move1_z5_dn", 15, -1), ("move1_z3_dn", 15, -1),
        ("move1_z8_up", 15, 1), ("move1_z5_up", 15, 1), ("move5_abs2pct_up", 15, 1),
        ("trdspike10_cont_up", 5, 1), ("trdspike10_cont_dn", 5, -1), ("volspike5_exh_dn", 5, 1), ("volspike5_cont_dn", 5, -1),
        ("ofi5_z2_both", 5, -1), ("sign5", 5, -1), ("minute00", 1, 1), ("minute15", 1, 1), ("fund_post5_big", 5, 1),
    ]), {"h": "{:d}"})

    T["selected"] = md_table(selected_table(design, validation), {"events": "{:d}", "days": "{:d}", "hit": "{:.3f}"})
    T["selected_splits"] = md_table(split_table(design, validation))
    T["execution"] = md_table(execution_table(design, validation))
    T["cross_market"] = md_table(cross_market_table(design, validation), {"events": "{:d}"})

    for key, frame in part_b.items():
        T["part_b/" + key] = md_table(order(frame)[PART_B_COLUMNS], PART_B_FORMATS)

    extras = json.loads((an.OUT / "design_extras.json").read_text())
    survivorship = json.loads((an.OUT / "survivorship_design.json").read_text())
    resting = pd.DataFrame(extras["resting_orders_design"])
    resting = resting[["definition", "h", "signals", "taker_gross_bp", "taker_net_bp", "order_alive_min", "bp_below_first_print", "fill_rate",
                       "gross_bp_if_filled", "gross_bp_of_unfilled_signals", "net_bp_per_signal"]]
    T["resting"] = md_table(resting, {"h": "{:d}", "signals": "{:d}", "order_alive_min": "{:d}", "bp_below_first_print": "{:.0f}", "fill_rate": "{:.0%}"}, "{:.1f}")
    anatomy = extras["cascade_anatomy_design"]
    rows = []
    for label in ("all", "first event in 60 minutes (same coin)", "later legs of the same cascade", "one position at a time per coin"):
        rows.append({"subset (design only)": label, **{k: anatomy[label][k] for k in ("n", "days", "mean_bp", "t_day", "median_bp", "hit")}})
    for group in ("by size of the 5-minute fall", "by number of coins falling in the same quarter hour"):
        for label, row in anatomy[group].items():
            rows.append({"subset (design only)": f"{group.split(' of ')[0].replace('by ', '')}: {label}", **{k: row[k] for k in ("n", "days", "mean_bp", "t_day", "median_bp", "hit")}})
    T["cascade"] = md_table(pd.DataFrame(rows), {"n": "{:d}", "days": "{:d}", "hit": "{:.2f}"}, "{:.1f}")
    stress = pd.DataFrame(extras["cost_stress"])[["segment", "costs", "events", "event_days", "bp_per_event", "annualised_sharpe_of_daily_sums",
                                                  "share_of_event_days_positive", "worst_day_bp_units", "max_drawdown_bp_units"]]
    T["stress"] = md_table(stress, {"events": "{:d}", "event_days": "{:d}", "share_of_event_days_positive": "{:.0%}", "worst_day_bp_units": "{:.0f}",
                                    "max_drawdown_bp_units": "{:.0f}", "annualised_sharpe_of_daily_sums": "{:.2f}"}, "{:.1f}")
    surv = pd.DataFrame(survivorship["rows"])
    surv = surv[~surv.group.str.contains(", 20")][["fall", "hold_hours", "group", "events", "days", "coins", "gross_bp", "excess_bp", "t_day", "median_bp", "hit"]]
    T["survivorship"] = md_table(surv, {"hold_hours": "{:d}", "events": "{:d}", "days": "{:d}", "coins": "{:d}", "hit": "{:.2f}"}, "{:.1f}")
    surv_year = pd.DataFrame(survivorship["rows"])
    surv_year = surv_year[surv_year.group.str.contains(", 20") & (surv_year.fall.str.contains("-5%"))][["fall", "hold_hours", "group", "events", "days", "coins", "gross_bp", "excess_bp", "t_day"]]
    T["survivorship_years"] = md_table(surv_year, {"hold_hours": "{:d}", "events": "{:d}", "days": "{:d}", "coins": "{:d}"}, "{:.1f}")

    strong = pooled[pooled.t_day.abs() > 3]
    pays = strong[strong.gross_bp.abs() > strong.cost_perp_taker_bp]

    # ---- results.json -------------------------------------------------------------------------
    def records(frame: pd.DataFrame) -> list:
        return json.loads(frame.to_json(orient="records", double_precision=4))

    results = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": "Do structures detected on 1-minute bars predict the next 1 minute to 1 day by enough to pay costs (A), "
                    "and does waiting for one improve the entry price of the daily trend trade (B)?",
        "conventions": {
            "event": "flagged at the close of 1-minute bar t from bars <= t",
            "fill": "open of bar t+1; forward return over h minutes = open[t+1+h] / open[t+1] - 1",
            "horizons_min": list(es.HORIZONS),
            "gaps": f"missing bars and zero-volume runs of >= {es.OUTAGE_RUN} bars are outages; a forward return whose window touches one is dropped; entry and exit bars must have traded",
            "inference": "sums per calendar day, cluster-robust standard error by day pooled over coins; Bartlett lags across days by horizon " + json.dumps(es.HAC_LAGS),
            "baseline": "mean forward return of all bars in the same coin x calendar year x hour of day x volatility state (3 states); family 5 without the hour",
            "costs_round_trip_bp": {"perp_taker": {"BTC/ETH": 12.0, "others": 14.0}, "perp_maker_bound": 4.0, "spot_taker": {"BTC/ETH": 22.0, "others": 24.0}},
            "segments": {"design": "< 2024-01-01", "validation": "2024-01-01 .. 2025-06-30", "holdout": "never loaded"},
        },
        "looks": {
            "design_grid_definition_x_horizon": len(pooled),
            "descriptive_autocorrelation_statistics": 11,
            "total_counted": len(pooled) + 11,
            "by_family": pooled.groupby("family").size().to_dict(),
            "not_counted_as_new_looks": {
                "same grid repeated on BTC/ETH spot (cross-check)": len(cross),
                "same grid repeated on PAXG spot (cross-check)": len(paxg),
                "execution variants of 15 candidate definitions on design (4 alternative fills x 3 horizons)": 180,
                "resting-order variants of 4 candidate definitions on design (6 variants x 3 horizons looked at; 16 rows kept)": 72,
                "hourly survivorship appendix on all perpetuals, design only (2 thresholds x 2 horizons)": 4,
                "cascade anatomy of the validated structure, design only (subsets, not new definitions carried anywhere)": 10,
                "by-year and by-coin breakdowns": "not counted",
            },
            "abs_t_above_3": len(strong),
            "abs_t_above_4": int((pooled.t_day.abs() > 4).sum()),
            "expected_by_chance_above_3": round(len(pooled) * 0.0027, 1),
            "abs_t_above_3_and_gross_above_taker_round_trip": len(pays),
        },
        "design_overview_by_family": records(family_overview(pooled)),
        "design_all_looks": records(pooled.drop(columns=["hypothesis"])),
        "design_hypotheses": pooled.drop_duplicates("definition").set_index("definition")["hypothesis"].to_dict(),
        "design_breakdowns": records(pd.read_csv(TABLES / "design_breakdowns.csv", dtype={"key": str})),
        "design_autocorrelation": json.loads((an.OUT / "autocorr_design_um.json").read_text()),
        "design_spot_vs_perp_btc_eth": {"looks": len(cross), "correlation_of_excess_bp": float(np.corrcoef(cross.excess_bp_spot, cross.excess_bp_perp)[0, 1])},
        "design_paxg_spot": {"looks": len(paxg), "abs_t_above_3": int((paxg.t_day.abs() > 3).sum()),
                             "note": "illiquid: a quarter of the minutes have no trade; every short-horizon reversal is far below the 24 bp spot round trip",
                             "rows": records(paxg[paxg.definition.isin(["move5_abs2pct_dn", "move1_z3_dn", "move1_z8_dn", "sign1", "sign5", "sign60"]) & paxg.h.isin([5, 15, 60])].drop(columns=["hypothesis"]))},
        "selection": selection,
        "selected_design": design,
        "selected_validation": validation,
        "part_b": {key: records(frame) for key, frame in part_b.items()},
        "design_side_analyses": extras,
        "design_survivorship_check_hourly": survivorship,
        "verdict": "edge_after_taker_costs",
        "verdict_scope": "One rare structure only (buying a fall of >= 2% within 5 minutes and holding 15 minutes). Every frequent structure fails to pay costs, "
                         "and waiting for a structure does not improve the entry price of the daily trend trade.",
    }
    results["tests_on_synthetic_data"] = TESTS
    side = pays.definition.str.rsplit("_", n=1).str[-1]
    moves = pays.family == "3 move"
    pays_note = (f"{int(moves.sum())} in the large-move family ({int((moves & (side == 'dn')).sum())} after falls, {int((moves & (side == 'both')).sum())} pooled over both directions "
                 f"and carried by the falls, {int((moves & (side == 'up')).sum())} after rises) and {int((~moves).sum())} elsewhere")
    one_day = pooled[pooled.h == 1440]
    day_note = (f"At the one-day horizon {int((one_day.t_day.abs() > 3).sum())} of {len(one_day)} looks reaches |t| > 3, which is what chance gives: "
                "nothing on a 1-minute bar predicts the next day.")
    results["conclusions"] = conclusions(design, validation, part_b, len(pooled), len(strong), len(pays), float(strong.gross_bp.abs().median()), pays_note, day_note)
    results["caveats"] = CAVEATS
    results["plain_language"] = plain_language(pooled, design, validation, part_b, extras, survivorship,
                                               float(np.corrcoef(cross.excess_bp_spot, cross.excess_bp_perp)[0, 1]), len(cross))
    (HERE / "results.json").write_text(json.dumps(results, indent=1, default=float))
    print("wrote results.json", (HERE / "results.json").stat().st_size, "bytes")
    for line in results["conclusions"]:
        print(" -", line)
    if "--tables" in sys.argv:
        for key in sorted(T):
            print(f"\n### {key}\n")
            print(T[key])


if __name__ == "__main__":
    main()
