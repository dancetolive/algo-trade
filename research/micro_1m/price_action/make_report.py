"""Assemble REPORT.md, results.json and out/report_tables.md from the files in out/.

    python -I -B make_report.py

Nothing is computed from market data here. Every number in the report is read
from the tables written by run_segment.py / analyze.py / maker_sim.py /
bootstrap_selected.py / part_b.py / paxg_check.py, so the prose cannot drift
away from the data.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
H = (1, 5, 15, 60, 240, 1440)
VERDICT = "edge_only_with_maker_fills"

FAMILY_LABEL = {
    "sweep": "1a. Sweep of N-min extreme, close back inside (reversal)",
    "sweepmat": "1b. Same, swept level at least 10 min old (reversal)",
    "hold": "1c. Break of N-min extreme that holds (continuation)",
    "bos": "2. Break of structure / change of character (continuation)",
    "fvg": "3a. Fair value gap (continuation)",
    "fvgfill": "3b. Return into the gap, next-open entry (bounce)",
    "fvglimit": "3c. Return into the gap, resting limit at the edge (bounce)",
    "squeeze": "4. Compression then breakout (continuation)",
    "pdsweep": "5a. Sweep of previous day's high / low (reversal)",
    "pdhold": "5b. Acceptance beyond previous day's high / low (continuation)",
    "ob": "5c. Order block retest, next-open entry (bounce)",
    "oblimit": "5d. Order block retest, resting limit (bounce)",
    "disp": "5e. Displacement candle (continuation)",
}
STRUCT_LABEL = {
    "sweep_N30_x2_k3": "small sweep (N=30, 2 bp, k=3)",
    "sweep_N240_x10_k1": "big sweep (N=240, 10 bp, k=1)",
    "bos_s5_all": "break of structure (5/5 swings)",
    "fvgfill_m0p5_W60": "return into fair value gap",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def md(table: pd.DataFrame, digits: dict | None = None, default: int = 2) -> str:
    digits = digits or {}
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in table.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else f"{v:,.{digits.get(c, default)}f}")
            elif isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)):
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else round(float(obj), 4)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def records(table: pd.DataFrame) -> list[dict]:
    return clean(table.to_dict(orient="records"))


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(OUT, name))


# --------------------------------------------------------------------------- #
# load and derive
# --------------------------------------------------------------------------- #

def build() -> dict:
    c: dict = {}
    design = read("table_design.csv")
    c["controls"] = design[design["family"] == "control"]
    looks = design[design["family"] != "control"].copy()
    c["looks"] = looks
    c["spot"], c["major"] = read("table_design_spot.csv"), read("table_design_um_btc_eth.csv")
    bd_design, bd_valid = read("breakdown_design.csv"), read("breakdown_validation.csv")
    for bd in (bd_design, bd_valid):
        bd["events"], bd["days"] = bd["events"].astype(int), bd["days"].astype(int)
    c["bd_design"], c["bd_valid"] = bd_design, bd_valid
    valid, valid_spot, valid_major = read("table_validation.csv"), read("table_validation_spot.csv"), read("table_validation_um_btc_eth.csv")
    c["valid"] = valid
    maker = pd.concat([read(f"maker_sim_{s}.csv") for s in ("design", "validation")], ignore_index=True)
    boot = pd.concat([read(f"bootstrap_{s}.csv") for s in ("design", "validation")], ignore_index=True)
    c["boot"] = boot
    part_b = pd.concat([read(f"part_b_{s}.csv") for s in ("design", "validation")], ignore_index=True)
    c["part_b"] = part_b
    paxg = read("table_design_paxg.csv")
    c["paxg"] = paxg[~paxg["def_id"].str.startswith("ctl")]
    c["dry"] = read("table_dryrun.csv")
    with open(os.path.join(HERE, "selection.json")) as handle:
        c["selection"] = json.load(handle)
    with open(os.path.join(OUT, "run_design.json")) as handle:
        c["run_design"] = json.load(handle)
    with open(os.path.join(OUT, "run_validation.json")) as handle:
        c["run_valid"] = json.load(handle)
    pairs = [(item["def_id"], int(item["h"])) for item in c["selection"]["selected"]]
    c["pairs"] = pairs

    # headline counts
    c["n_looks"] = len(looks)
    c["sig3"], c["sig39"] = int((looks["t_excess"].abs() > 3).sum()), int((looks["t_excess"].abs() > 3.9).sum())
    c["against"] = int((looks["t_excess"] < -3).sum())
    c["pos_net"] = looks[looks["net_taker"] > 0]
    c["pos_net_sig"] = c["pos_net"][c["pos_net"]["t_excess"].abs() > 3]
    c["maker_pos"] = int((looks["net_maker_bound"] > 0).sum())
    c["maker_pos_sig"] = int(((looks["net_maker_bound"] > 0) & (looks["t_excess"].abs() > 3)).sum())
    inflation = (looks["t_naive"].abs() / looks["t_excess"].abs()).replace([np.inf, -np.inf], np.nan)
    c["inflation"] = inflation.groupby(looks["h"]).median()
    c["max_short_gross"] = float(looks[looks["h"] <= 15]["gross"].abs().max())

    # family summary
    rows = []
    for fam, label in FAMILY_LABEL.items():
        part = looks[looks["family"] == fam]
        best = part.loc[part["t_excess"].abs().idxmax()]
        top = part.loc[part["net_taker"].idxmax()]
        rows.append({"family (textbook reading)": label, "defs": int(part["def_id"].nunique()), "looks": len(part),
                     "t>3 for": int((part["t_excess"] > 3).sum()), "t<-3 against": int((part["t_excess"] < -3).sum()),
                     "strongest look": f"{best.def_id} @{int(best.h)}m", "gross bp": best.gross, "t": best.t_excess,
                     "best net taker bp": top.net_taker, "where": f"{top.def_id} @{int(top.h)}m (t {top.t_excess:+.1f})"})
    c["family"] = pd.DataFrame(rows)

    # horizon profiles
    reps = ["sweep_N30_x2_k3", "sweep_N60_x10_k1", "sweep_N240_x10_k1", "sweepmat_N60_x2_k1", "sweepmat_N240_x2_k3",
            "hold_N30_x2_k3", "hold_N240_x10_k3", "pdsweep_x2_k3", "pdsweep_x10_k1", "pdhold_x2_k1",
            "bos_s3_all", "bos_s5_all", "bos_s5_with", "bos_s5_against", "bos_s10_all",
            "fvg_m0p5", "fvg_m2", "fvgfill_m0p5_W60", "fvgfill_m2_W240", "fvglimit_m0p5_W60",
            "squeeze_M30_q0p1_v0", "squeeze_M30_q0p1_v2", "squeeze_M120_q0p1_v2",
            "ob_W60", "oblimit_W60", "disp_m3", "disp_m6"]
    rows = []
    for d in reps:
        part = looks[looks["def_id"] == d].set_index("h")
        row = {"definition": d, "events": int(part.loc[5, "events"]), "days": int(part.loc[5, "days"])}
        for h in H:
            row[f"{h}m"] = f"{part.loc[h, 'gross']:+.2f} ({part.loc[h, 't_excess']:+.1f})"
        best = part["net_taker"].idxmax()
        row["best net taker"] = f"{part.loc[best, 'net_taker']:+.1f} @{best}m"
        rows.append(row)
    c["profile"] = pd.DataFrame(rows)

    keys = [("sweep_N30_x2_k3", 5), ("sweep_N240_x10_k1", 60), ("sweep_N240_x10_k1", 240), ("sweep_N60_x10_k1", 240),
            ("sweepmat_N60_x2_k1", 1), ("hold_N30_x2_k3", 5), ("hold_N240_x10_k3", 60), ("pdsweep_x2_k3", 60), ("pdhold_x2_k1", 1),
            ("bos_s5_all", 1), ("bos_s5_with", 15), ("bos_s5_against", 60), ("fvg_m0p5", 5), ("fvgfill_m0p5_W60", 5),
            ("fvglimit_m0p5_W60", 1), ("squeeze_M30_q0p1_v2", 1), ("squeeze_M30_q0p1_v2", 60), ("ob_W60", 15), ("oblimit_W60", 1),
            ("disp_m3", 60), ("disp_m6", 240)]
    c["compact"] = looks.set_index(["def_id", "h"]).loc[keys].reset_index()[
        ["def_id", "h", "events", "days", "gross", "base", "excess", "t_excess", "t_naive", "hit", "net_taker", "net_maker_bound"]]

    # rejection versus acceptance as a contrarian trade
    rows = []
    for name in ("sweep_N240_x10_k1", "sweep_N240_x10_k3", "hold_N240_x10_k1", "hold_N240_x10_k3", "sweep_N30_x2_k3", "hold_N30_x2_k3"):
        for h in (15, 60, 240):
            part = bd_design[(bd_design["def_id"] == name) & (bd_design["h"] == h) & (bd_design["by"] == "direction")].set_index("value")
            sign = 1 if name.startswith("sweep") else -1
            low, high = (part.loc["long"], part.loc["short"]) if sign == 1 else (part.loc["short"], part.loc["long"])
            rows.append({"event": name, "what happened after the breach": "closed back inside" if sign == 1 else "closed beyond (held)",
                         "h": h, "new-low events": int(low["events"]), "buyer excess bp": sign * low["excess"], "t": sign * low["t_excess"],
                         "new-high events": int(high["events"]), "seller excess bp": sign * high["excess"], "t ": sign * high["t_excess"]})
    c["reject_accept"] = pd.DataFrame(rows)

    # selected pairs
    rows = []
    for d, h in pairs:
        for seg, table in (("design", looks), ("validation", valid)):
            r = table[(table["def_id"] == d) & (table["h"] == h)].iloc[0]
            m = maker[(maker["segment"] == seg) & (maker["def_id"] == d) & (maker["h"] == h)].iloc[0]
            b = boot[(boot["segment"] == seg) & (boot["def_id"] == d) & (boot["h"] == h)].iloc[0]
            se_gross = r["gross"] / r["t_gross"]
            se_maker = m["maker_gross_per_fill"] / m["maker_t_per_fill"]
            rows.append({
                "definition": d, "h": h, "segment": seg, "events": int(r["events"]), "days": int(r["days"]),
                "gross": r["gross"], "base": r["base"], "excess": r["excess"], "t_excess": r["t_excess"], "t_naive": r["t_naive"],
                "boot_ci_lo": b["excess_ci_lo"], "boot_ci_hi": b["excess_ci_hi"], "share_boot_gross_above_cost": b["share_resamples_gross_above_cost"],
                "cost_taker": r["cost_taker"], "net_taker": r["net_taker"], "t_net_taker": r["net_taker"] / se_gross,
                "maker_bound": r["net_maker_bound"], "fill_rate": m["fill_rate"], "maker_gross_per_fill": m["maker_gross_per_fill"],
                "maker_net_4bp": m["maker_net_per_fill_maker_exit_4bp"], "t_maker_net_4bp": m["maker_net_per_fill_maker_exit_4bp"] / se_maker,
                "maker_in_taker_out": m["maker_net_per_fill_taker_exit"], "t_maker_in_taker_out": m["maker_net_per_fill_taker_exit"] / se_maker,
                "missed_taker_gross": m["taker_gross_if_never_filled"], "filled_taker_gross": m["taker_gross_if_later_filled"],
                "mark_next_open": m["maker_mark_at_next_open"]})
    selected = pd.DataFrame(rows)
    c["selected"] = selected
    rows = []
    for d, h in pairs:
        part = selected[(selected["definition"] == d) & (selected["h"] == h)]
        for col, tcol in (("net_taker", "t_net_taker"), ("maker_net_4bp", "t_maker_net_4bp"), ("maker_in_taker_out", "t_maker_in_taker_out")):
            se = (part[col] / part[tcol]).abs().to_numpy()
            w = 1.0 / se ** 2
            mean = float((part[col].to_numpy() * w).sum() / w.sum())
            rows.append({"definition": d, "h": h, "figure": col, "design": part[col].iloc[0], "validation": part[col].iloc[1],
                         "both": mean, "t both": mean / float(np.sqrt(1.0 / w.sum()))})
    c["pooled"] = pd.DataFrame(rows)

    rows = []
    for d, h in pairs:
        for seg, bd in (("design", bd_design), ("validation", bd_valid)):
            for r in bd[(bd["def_id"] == d) & (bd["h"] == h) & (bd["by"] == "year")].itertuples():
                rows.append({"definition": d, "h": h, "year": str(int(float(r.value))), "segment": seg, "events": int(r.events), "days": int(r.days),
                             "gross": r.gross, "excess": r.excess, "t_excess": r.t_excess, "net_taker": r.net})
    c["by_year"] = pd.DataFrame(rows)
    rows = []
    for d, h in pairs:
        for side in ("long", "short"):
            for seg, bd in (("design", bd_design), ("validation", bd_valid)):
                r = bd[(bd["def_id"] == d) & (bd["h"] == h) & (bd["by"] == "direction") & (bd["value"] == side)].iloc[0]
                se = abs(r["excess"] / r["t_excess"])
                rows.append({"definition": d, "h": h, "side": side, "segment": seg, "events": int(r["events"]), "gross": r["gross"],
                             "base": r["base"], "excess": r["excess"], "t_excess": r["t_excess"], "net_taker": r["net"],
                             "excess minus cost": r["excess"] - (r["gross"] - r["net"]), "t (excess - cost)": (r["excess"] - (r["gross"] - r["net"])) / se})
    c["by_side"] = pd.DataFrame(rows)
    rows = []
    for d, h in pairs:
        for seg, bd in (("design", bd_design), ("validation", bd_valid)):
            for r in bd[(bd["def_id"] == d) & (bd["h"] == h) & (bd["by"] == "symbol")].itertuples():
                rows.append({"definition": d, "h": h, "coin": r.value.replace("USDT", ""), "segment": seg, "events": int(r.events),
                             "gross": r.gross, "excess": r.excess, "t_excess": r.t_excess, "net_taker": r.net})
    c["by_coin"] = pd.DataFrame(rows)
    rows = []
    for d, h in pairs:
        for seg, pm, sp in (("design", c["major"], c["spot"]), ("validation", valid_major, valid_spot)):
            a = pm[(pm["def_id"] == d) & (pm["h"] == h)].iloc[0]
            b = sp[(sp["def_id"] == d) & (sp["h"] == h)].iloc[0]
            rows.append({"definition": d, "h": h, "segment": seg, "perp events": int(a["events"]), "perp gross": a["gross"], "perp t": a["t_excess"],
                         "perp net taker": a["net_taker"], "spot events": int(b["events"]), "spot gross": b["gross"], "spot t": b["t_excess"],
                         "spot net taker": b["net_taker"]})
    c["cross"] = pd.DataFrame(rows)
    index = list(zip(looks.def_id, looks.h))
    c["t_corr"] = float(np.corrcoef(c["major"].set_index(["def_id", "h"]).loc[index, "t_excess"],
                                    c["spot"].set_index(["def_id", "h"]).loc[index, "t_excess"])[0, 1])
    spot_real = c["spot"][~c["spot"]["def_id"].str.startswith("ctl")]
    c["spot_pos"] = int((spot_real["net_taker"] > 0).sum())
    c["counts"] = {seg: read(f"part_b_trade_counts_{seg}.csv") for seg in ("design", "validation")}
    return c


def pb_table(c: dict, market: str, label: str) -> pd.DataFrame:
    part = c["part_b"]
    part = part[(part["market"] == market) & (part["set"] == label)].copy()
    part = part[~((part["window"] == 1440) & part["structure"].isin(["sweep_N30_x2_k3", "bos_s5_all", "fvgfill_m0p5_W60"]))]
    part["structure"] = part["structure"].map(STRUCT_LABEL)
    part = part.sort_values(["structure", "window", "segment"])
    part = part.rename(columns={"window": "wait up to (min)", "triggered": "occurred", "mean_wait_min": "mean wait (min)",
                                "mean_bp": "mean bp", "median_bp": "median bp", "worse": "share worse",
                                "mean_if_better": "mean if better", "mean_if_worse": "mean if worse",
                                "delay_end_bp": "just waiting, bp", "info_bp": "structure adds, bp", "t_info": "t (adds)"})
    return part[["structure", "wait up to (min)", "segment", "trades", "days", "occurred", "mean wait (min)", "mean bp", "median bp", "t",
                 "share worse", "mean if better", "mean if worse", "just waiting, bp", "structure adds, bp", "t (adds)"]]


PB_DIGITS = {"occurred": 2, "share worse": 2, "mean wait (min)": 0, "mean bp": 1, "median bp": 1, "mean if better": 1, "mean if worse": 1,
             "just waiting, bp": 1, "structure adds, bp": 1}


def pb(c: dict, market: str, label: str, structure: str, window: int, segment: str) -> pd.Series:
    t = c["part_b"]
    return t[(t["market"] == market) & (t["set"] == label) & (t["structure"] == structure) & (t["window"] == window) & (t["segment"] == segment)].iloc[0]


def sel(c: dict, d: str, h: int, segment: str) -> pd.Series:
    t = c["selected"]
    return t[(t["definition"] == d) & (t["h"] == h) & (t["segment"] == segment)].iloc[0]


# --------------------------------------------------------------------------- #
# text blocks shared by REPORT.md and results.json
# --------------------------------------------------------------------------- #

def narrative(c: dict) -> dict:
    a_d, a_v = sel(c, "sweep_N240_x10_k1", 240, "design"), sel(c, "sweep_N240_x10_k1", 240, "validation")
    b_d, b_v = sel(c, "sweep_N240_x10_k1", 60, "design"), sel(c, "sweep_N240_x10_k1", 60, "validation")
    s_d, s_v = sel(c, "sweep_N30_x2_k3", 5, "design"), sel(c, "sweep_N30_x2_k3", 5, "validation")
    looks = c["looks"].set_index(["def_id", "h"])
    fl, ol = looks.loc[("fvglimit_m0p5_W60", 1)], looks.loc[("oblimit_W60", 1)]
    e_d = pb(c, "spot", "trend_both", "sweep_N30_x2_k3", 240, "design")
    e_v = pb(c, "spot", "trend_both", "sweep_N30_x2_k3", 240, "validation")
    q_d = pb(c, "spot", "trend_both", "bos_s5_all", 240, "design")
    q_v = pb(c, "spot", "trend_both", "bos_s5_all", 240, "validation")
    summary = (
        f"70 exact definitions of ICT-style 1-minute structures x 6 horizons = {c['n_looks']} looks on six Binance perpetuals, 2020-2023. "
        f"{c['sig3']} looks have day-clustered |t| > 3, but {c['against']} of them point against the textbook reading, and no effect up to "
        f"15 minutes exceeds {c['max_short_gross']:.1f} bp gross versus 12-14 bp taker costs. Break of structure, fair value gaps, "
        f"order blocks, compression breakouts, displacement and previous-day sweeps give nothing tradeable; limit orders at gap or block "
        f"edges lose {abs(ol['gross']):.1f}-{abs(fl['gross']):.1f} bp to adverse selection (t {ol['t_excess']:.0f}, {fl['t_excess']:.0f}). "
        f"The only cost-sized effect is the rebound after a sweep of the 4-hour extreme: {a_d['gross']:+.1f} bp at 240 min on design "
        f"(t {a_d['t_excess']:.1f}, net of taker costs {a_d['net_taker']:+.1f}) and {a_v['gross']:+.1f} bp on validation (t {a_v['t_excess']:.1f}, "
        f"net {a_v['net_taker']:+.1f}, t of net {a_v['t_net_taker']:.1f}). It is not specific to the pattern, sits on the long side and in "
        f"altcoins, and faded to about zero in 2023. Resting orders net {a_d['maker_net_4bp']:+.1f} / {a_v['maker_net_4bp']:+.1f} bp per fill at "
        f"maker fees (t {a_d['t_maker_net_4bp']:.1f} / {a_v['t_maker_net_4bp']:.1f}). Part B: waiting for a structure did not improve "
        f"trend-trade fills: small sweep {e_d['mean_bp']:+.1f} bp (t {e_d['t']:.1f}) design, {e_v['mean_bp']:+.1f} (t {e_v['t']:.1f}) "
        f"validation; waiting for a break of structure made {q_d['worse']:.0%}-{q_v['worse']:.0%} of fills worse ({q_v['mean_bp']:+.1f} bp, "
        f"t {q_v['t']:.1f}, validation)."
    )

    def text(r):
        return (f"{int(r['events']):,} events on {int(r['days']):,} days; mean gross {r['gross']:+.2f} bp (matched baseline {r['base']:+.2f}, excess "
                f"{r['excess']:+.2f}); day-clustered t {r['t_excess']:.2f} (naive {r['t_naive']:.1f}; bootstrap 95% CI of excess "
                f"{r['boot_ci_lo']:+.2f} .. {r['boot_ci_hi']:+.2f}); taker round trip {r['cost_taker']:.2f} bp -> net {r['net_taker']:+.2f} bp "
                f"(t of net {r['t_net_taker']:.2f}); maker bound (gross - 4 bp) {r['maker_bound']:+.2f}; resting-order simulation: fill rate "
                f"{r['fill_rate']:.1%}, {r['maker_net_4bp']:+.2f} bp per fill at maker fees both sides (t {r['t_maker_net_4bp']:.2f}), "
                f"{r['maker_in_taker_out']:+.2f} bp maker in / taker out (t {r['t_maker_in_taker_out']:.2f})")

    definitions = {
        "sweep_N240_x10_k1": "Liquidity sweep and rejection of the 4-hour extreme. Bar t trades at least 10 bp beyond the highest high "
                             "(lowest low) of the previous 240 one-minute bars and closes back inside that level in the same bar. Short after "
                             "a sweep of the high, long after a sweep of the low; fill at the open of bar t+1.",
        "sweep_N30_x2_k3": "Liquidity sweep and rejection of the 30-minute extreme. A bar trades at least 2 bp beyond the highest high "
                           "(lowest low) of the previous 30 one-minute bars; the event is the first close back inside that level if it comes "
                           "within 3 bars (the breach bar or the next two). Short after a sweep of the high, long after a sweep of the low; "
                           "fill at the next bar's open.",
    }
    selected = []
    for d, h, name in (("sweep_N240_x10_k1", 240, "Big sweep reversal, 4-hour hold"), ("sweep_N240_x10_k1", 60, "Big sweep reversal, 1-hour hold"),
                       ("sweep_N30_x2_k3", 5, "Small sweep reversal, 5-minute hold")):
        selected.append({"name": name, "def_id": d, "definition": definitions[d], "market": "Binance USDT-margined perpetuals, six coins pooled "
                         "(spot BTC/ETH as cross-check)", "symbols": "BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, DOGEUSDT", "horizon_minutes": h,
                         "design": text(sel(c, d, h, "design")), "validation": text(sel(c, d, h, "validation"))})

    def pbline(market, label, structure, window):
        d, v = pb(c, market, label, structure, window, "design"), pb(c, market, label, structure, window, "validation")
        return (f"design {int(d['trades'])} trades: mean {d['mean_bp']:+.1f} bp, median {d['median_bp']:+.1f}, t {d['t']:.2f}, worse in "
                f"{d['worse']:.0%}; validation {int(v['trades'])} trades: mean {v['mean_bp']:+.1f} bp, median {v['median_bp']:+.1f}, t {v['t']:.2f}, "
                f"worse in {v['worse']:.0%}")

    entry_timing = (
        "Daily trend rule = blend of 50/100/200-day SMA filters on BTC and ETH spot; every weight change is filled at 00:00 UTC. Improvement = "
        "price at the 00:00 open versus price after waiting for the first structure in the trade's direction (spot 1-minute bars; positive = "
        "better). ENTRIES, wait up to 4 h -- small sweep (N30, 2 bp, k3): " + pbline("spot", "trend_entry", "sweep_N30_x2_k3", 240)
        + ". Big sweep (N240, 10 bp, k1), up to 4 h (occurs in "
        f"{pb(c, 'spot', 'trend_entry', 'sweep_N240_x10_k1', 240, 'design')['triggered']:.0%} / "
        f"{pb(c, 'spot', 'trend_entry', 'sweep_N240_x10_k1', 240, 'validation')['triggered']:.0%} of cases, otherwise bought 4 h later): "
        + pbline("spot", "trend_entry", "sweep_N240_x10_k1", 240) + "; up to 24 h: " + pbline("spot", "trend_entry", "sweep_N240_x10_k1", 1440)
        + ". Companion triggers (not selected in Part A): bullish break of structure: " + pbline("spot", "trend_entry", "bos_s5_all", 240)
        + "; return into a bullish fair value gap: " + pbline("spot", "trend_entry", "fvgfill_m0p5_W60", 240)
        + ". EXITS (mirror), small sweep: " + pbline("spot", "trend_exit", "sweep_N30_x2_k3", 240)
        + "; big sweep up to 24 h: " + pbline("spot", "trend_exit", "sweep_N240_x10_k1", 1440)
        + "; break of structure: " + pbline("spot", "trend_exit", "bos_s5_all", 240)
        + ". ENTRIES AND EXITS TOGETHER, small sweep: " + pbline("spot", "trend_both", "sweep_N30_x2_k3", 240)
        + "; break of structure: " + pbline("spot", "trend_both", "bos_s5_all", 240)
        + ". Higher-powered supplement, every calendar day, both sides, small sweep: " + pbline("spot", "all_days_both", "sweep_N30_x2_k3", 240)
        + ". Typical shape: the wait gives a better price about three times in four (by 20-40 bp) and a worse one the fourth time (by 60-80 bp), "
        "so the mean is statistically zero. Nothing is significant in both segments; the only consistent feature is that waiting for a break "
        "of structure (confirmation) gives a worse price in 63-78% of trades (its mean cost is significant on validation only). At 13-16x "
        "annual one-way turnover of the trend rule, each bp of fill improvement is worth 0.13-0.16% of equity per year."
    )
    caveats = [
        "Verdict 'edge_only_with_maker_fills' rests on ONE pre-registered pair (sweep of the 4-hour extreme, 240-minute hold). Its taker net "
        f"({a_d['net_taker']:+.1f} design, {a_v['net_taker']:+.1f} validation) is not distinguishable from zero (t {a_d['t_net_taker']:.1f} and "
        f"{a_v['t_net_taker']:.1f}); the maker figure assumes a passive exit at the open price 240 minutes later (optimistic); with a taker exit it is "
        f"{a_d['maker_in_taker_out']:+.1f} / {a_v['maker_in_taker_out']:+.1f} bp (t {a_d['t_maker_in_taker_out']:.1f} / {a_v['t_maker_in_taker_out']:.1f}).",
        "The effect is not a property of the 'liquidity sweep' pattern: on design, a thrust to a new 4-hour low was followed by the same rebound "
        "whether the bar closed back inside (sweep) or price stayed below for three closes (acceptance). It is a generic dip-rebound, on the long "
        "side only; the short side (sweeps of highs) earns nothing.",
        "It is unstable in time and across coins: about zero in 2023 "
        f"({c['by_year'][(c['by_year']['definition'] == 'sweep_N240_x10_k1') & (c['by_year']['h'] == 240) & (c['by_year']['year'] == '2023')]['gross'].iloc[0]:+.1f} bp "
        "at 240 min, t 0.2) and weak in the last 18 months of design (2022-07..2023-12: "
        f"{c['dry'].set_index('h').loc[240, 'gross']:+.1f} bp, t {c['dry'].set_index('h').loc[240, 't_excess']:.1f}), back in 2024-25; carried by "
        "XRP, SOL, DOGE, BNB. For BTC and ETH alone the validation figure is "
        f"{c['cross'][(c['cross']['definition'] == 'sweep_N240_x10_k1') & (c['cross']['h'] == 240) & (c['cross']['segment'] == 'validation')]['perp gross'].iloc[0]:+.1f} bp "
        "(t 0.6) on perpetuals and about the same on spot.",
        "This is an event study, not a portfolio backtest: events come in bursts across correlated coins (many per sell-off), so a small account "
        "could take only a fraction of them; capital limits, overlapping positions, funding payments and liquidation risk while buying falling "
        "prices are not modelled.",
        "Fills are assumed at the next 1-minute open with 1-2 bp slippage. These events occur in the fastest minutes of the day, when spreads are "
        "widest, so taker results are more likely over- than under-stated. Bars cannot show queue position; the resting-order simulation counts a "
        "fill only when price trades strictly through the order.",
        f"Multiple testing: {c['n_looks']} looks on design (plus {len(c['paxg'])} on PAXG, reported separately, and year/coin/side splits of about 15 "
        "looks). Design figures of the selected pairs are biased upward by selection; the validation figures are not. Validation was evaluated once, "
        "for the three pre-registered pairs and the pre-registered diagnostics only. The hold-out (from 2025-07-01) was never loaded.",
        "The long-only split of the selected pair (excess over baseline minus taker cost about +12-13 bp, t 2.1 design / 1.3 validation) was noticed "
        "on design before validation but was not pre-registered as its own test; treat it as a hypothesis for the hold-out, not a finding.",
        "Part B has little power on the actual trend trades (284 design, 132 validation; standard errors of 4 to 25 bp depending on the structure). The all-days supplement "
        "narrows the unconditional effect of waiting for the small sweep to +4.2 bp (design) and +1.3 bp (validation). All Part B trades start at "
        "00:00 UTC, an unusual minute (daily close, funding), so the result may not carry to other clock times.",
        "The matched baseline uses volatility quintiles formed within each calendar year (a benchmark, not a signal, so it is not causal by "
        "construction). Disclosure: at the very start a data-availability printout of the full files (row counts, date range, number of gaps; no "
        "prices) was seen before the segments were separated in code.",
        "PAXG (tokenised gold) cannot be judged on 1-minute bars: its tick is about 5 bp of price and about a third of its bars have no trade, so "
        f"{int((c['paxg']['t_excess'].abs() > 3).sum())} of {len(c['paxg'])} looks are 'significant' through bid-ask bounce alone and none clears the "
        "24 bp spot round trip.",
    ]
    return {"summary": summary, "selected": selected, "entry_timing": entry_timing, "caveats": caveats}


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #

def write_tables(c: dict) -> None:
    out = [
        f"looks: {c['n_looks']}; |t|>3: {c['sig3']}; |t|>3.9: {c['sig39']}; t<-3 (against the textbook reading): {c['against']}",
        "net of taker costs > 0 with |t|>3: " + ", ".join(f"{r.def_id}@{r.h} ({r.net_taker:+.2f}, t={r.t_excess:.2f})" for r in c["pos_net_sig"].itertuples()),
        "net > 0 with |t|<=3: " + ", ".join(f"{r.def_id}@{r.h} ({r.net_taker:+.2f}, t={r.t_excess:.2f})"
                                               for r in c["pos_net"][c["pos_net"]["t_excess"].abs() <= 3].itertuples()),
        f"maker bound > 0: {c['maker_pos']} (|t|>3: {c['maker_pos_sig']})",
        "\n### controls\n" + md(c["controls"][["def_id", "h", "events", "days", "gross", "base", "excess", "t_excess"]]),
        "\n### family summary\n" + md(c["family"]),
        "\n### horizon profiles\n" + md(c["profile"]),
        "\n### compact\n" + md(c["compact"], {"hit": 3}),
        "\n### reject vs accept\n" + md(c["reject_accept"]),
        "\n### selected\n" + md(c["selected"]),
        "\n### pooled\n" + md(c["pooled"]),
        "\n### by year\n" + md(c["by_year"]),
        "\n### by side\n" + md(c["by_side"]),
        "\n### by coin\n" + md(c["by_coin"]),
        "\n### BTC/ETH perp vs spot\n" + md(c["cross"]),
        "\n### late design sub-period\n" + md(c["dry"][["def_id", "h", "events", "days", "gross", "base", "excess", "t_excess", "net_taker"]]),
    ]
    for market in ("spot", "um"):
        for label in ("trend_entry", "trend_exit", "trend_both", "all_days_both"):
            out.append(f"\n### Part B {market} {label}\n" + md(pb_table(c, market, label), PB_DIGITS))
    with open(os.path.join(OUT, "report_tables.md"), "w") as handle:
        handle.write("\n".join(out) + "\n")
    full = c["looks"][["def_id", "h", "events", "days", "gross", "base", "excess", "t_excess", "t_naive", "hit", "cost_taker", "net_taker", "net_maker_bound"]]
    with open(os.path.join(OUT, "table_design_full.md"), "w") as handle:
        handle.write("# All 420 design looks (six perpetuals pooled, bars before 2024-01-01)\n\n"
                     "gross / base / excess in bp, signed in the direction of the textbook reading; t_excess is clustered by UTC day; "
                     "net_taker = gross - taker round trip; net_maker_bound = gross - 4 bp (optimistic).\n\n" + md(full, {"hit": 3}) + "\n")


def write_results(c: dict, n: dict) -> None:
    looks = c["looks"]
    files = [os.path.join(HERE, f) for f in (
        "REPORT.md", "results.json", "selection.json", "eventstudy.py", "structures.py", "test_eventstudy.py", "run_segment.py", "analyze.py",
        "write_selection.py", "maker_sim.py", "bootstrap_selected.py", "part_b.py", "paxg_check.py", "make_report.py",
        "out/table_design.csv", "out/table_design_full.md", "out/breakdown_design.csv", "out/table_validation.csv", "out/breakdown_validation.csv",
        "out/maker_sim_design.csv", "out/maker_sim_validation.csv", "out/bootstrap_design.csv", "out/bootstrap_validation.csv",
        "out/part_b_design.csv", "out/part_b_validation.csv", "out/table_design_spot.csv", "out/table_design_paxg.csv", "out/report_tables.md",
        "out/test_log.txt")]
    table_rows = c["compact"].copy()
    results = {
        "verdict": VERDICT,
        "summary": n["summary"],
        "looks": c["n_looks"],
        "selected": n["selected"],
        "entry_timing": n["entry_timing"],
        "caveats": n["caveats"],
        "files": files,
        "question": "Do structures detected on 1-minute bars predict returns by enough to pay costs (A), and does waiting for one improve the fill "
                    "of a daily trend trade (B)?",
        "lens": "price-action structures as discretionary (smart money / ICT) traders define them",
        "segments": c["selection"]["protocol"]["segment_rules"],
        "symbols": {"perpetuals": [r["symbol"] for r in c["run_design"]["runs"] if r["market"] == "um"], "spot_cross_check": ["BTCUSDT", "ETHUSDT"],
                    "side_check": "PAXGUSDT spot (design only, never pooled)"},
        "costs_bp_round_trip": {"perp_taker": {"BTC/ETH": 12, "others": 14}, "perp_maker_bound": 4, "spot_taker": {"BTC/ETH": 22, "others": 24}},
        "data": {"design_runs": c["run_design"]["runs"], "validation_runs": c["run_valid"]["runs"]},
        "definitions": c["run_design"]["definitions"],
        "look_counts": {"design_definition_x_horizon": c["n_looks"], "definitions": int(looks["def_id"].nunique()), "horizons": list(H),
                        "abs_t_gt_3": c["sig3"], "abs_t_gt_3p9": c["sig39"], "t_lt_minus_3_against_textbook_reading": c["against"],
                        "net_taker_positive": int(len(c["pos_net"])), "net_taker_positive_and_abs_t_gt_3": int(len(c["pos_net_sig"])),
                        "maker_bound_positive": c["maker_pos"], "maker_bound_positive_and_abs_t_gt_3": c["maker_pos_sig"],
                        "median_naive_over_clustered_t_by_horizon": {int(h): round(float(c["inflation"][h]), 3) for h in H},
                        "other_looks": "420 on PAXG spot (separate); year / coin / side splits read for about 15 looks; Part B: 4 structures x 2 "
                                       "windows x entry / exit x 2 markets, plus pooled and all-days sets; validation: 3 pre-registered pairs only"},
        "controls_design": records(c["controls"][["def_id", "h", "events", "days", "gross", "base", "excess", "t_excess"]]),
        "design_all_looks": records(looks[["def_id", "family", "h", "events", "days", "gross", "base", "excess", "t_excess", "t_naive", "hit",
                                           "cost_taker", "net_taker", "net_maker_bound"]]),
        "design_family_summary": records(c["family"]),
        "design_compact_rows": records(table_rows),
        "design_reject_vs_accept": records(c["reject_accept"]),
        "selection": c["selection"],
        "selected_design_vs_validation": records(c["selected"]),
        "selected_design_and_validation_pooled": records(c["pooled"]),
        "selected_by_year": records(c["by_year"]),
        "selected_by_side": records(c["by_side"]),
        "selected_by_coin": records(c["by_coin"]),
        "selected_bootstrap": records(c["boot"]),
        "selected_btc_eth_perp_vs_spot": records(c["cross"]),
        "late_design_subperiod_2022H2_2023": records(c["dry"][["def_id", "h", "events", "days", "gross", "base", "excess", "t_excess", "net_taker"]]),
        "part_b": {"trade_counts": {seg: clean(t.to_dict(orient="records")) for seg, t in c["counts"].items()}, "tables": records(c["part_b"])},
        "paxg_design_side_check": {"looks": int(len(c["paxg"])), "abs_t_gt_3": int((c["paxg"]["t_excess"].abs() > 3).sum()),
                                   "net_of_24bp_positive": int((c["paxg"]["net_taker"] > 0).sum())},
        "synthetic_tests": "15 tests in test_eventstudy.py, all passing (out/test_log.txt): no look-ahead for all 70 definitions under altered "
                           "and truncated futures; each detector equal to a literal loop implementation; forward-return alignment and gap "
                           "handling; planted 2 bp/bar effect recovered; clustered t calibrated under bursts and overlap; baseline removes "
                           "hour-of-day drift.",
    }
    with open(os.path.join(HERE, "results.json"), "w") as handle:
        json.dump(results, handle, indent=1)
    stale = os.path.join(HERE, "results_tables.json")
    if os.path.exists(stale):
        os.remove(stale)


def main() -> None:
    c = build()
    n = narrative(c)
    n["verdict"] = VERDICT
    write_tables(c)
    write_results(c, n)
    import report_text  # the prose of REPORT.md, kept in its own file

    with open(os.path.join(HERE, "REPORT.md"), "w") as handle:
        handle.write(report_text.render(c, n, md, pb_table, PB_DIGITS, sel, pb))
    print(n["summary"])
    print(len(n["summary"].split()), "words")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, HERE)
    main()
