"""Collect the decisive audit numbers from out/*.json|csv into out/audit_summary.json (no new computation)."""
import json, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_lib as al
J = lambda name: json.load(open(os.path.join(al.OUT, name)))
r2 = lambda v: None if v is None else round(float(v), 2)
inf, ov, cr, pl, pb = J("inference.json"), J("overlap.json"), J("costs_realism.json"), J("placebo.json"), J("part_b_audit.json")
rep = pd.read_csv(os.path.join(al.OUT, "reproduce_headline.csv"))
out = {"reproduction_max_abs_diff_vs_researcher": float(rep["diff"].abs().max()),
       "bonferroni_z": inf["bonferroni_two_sided_5pct_z"], "pairs": {}}
for key in ("sweep_N240_x10_k1@240", "sweep_N240_x10_k1@60", "sweep_N30_x2_k3@5"):
    rec = {}
    for seg in ("design", "validation", "both"):
        b, m = inf["inference"][f"{key}|{seg}"], inf["maker"][f"{key}|{seg}"]
        rec[seg] = {"events": b["events"], "days": b["days"], "gross": r2(b["gross"]["mean"]), "excess": r2(b["excess"]["mean"]),
                    "t_excess_day": r2(b["excess"]["t_day_adjacent"]), "t_excess_week": r2(b["excess"]["t_week"]), "t_excess_month": r2(b["excess"]["t_month"]),
                    "t_excess_boot5": r2(b["excess"]["boot5_t"]), "t_excess_boot10": r2(b["excess"]["boot10_t"]), "t_excess_boot20": r2(b["excess"]["boot20_t"]),
                    "t_excess_most_conservative": r2(b["excess"]["t_most_conservative"]),
                    "net_taker": r2(b["net_taker"]["mean"]), "t_net_taker": r2(b["net_taker"]["t_day_adjacent"]),
                    "maker_fill_rate": r2(m["fill_rate"]), "maker_gross_per_fill": r2(m["maker_gross_per_fill"]), "maker_net_4bp": r2(m["maker_net_4bp"]),
                    "t_maker_net_4bp": r2(m["t_maker_net_4bp_day"]), "t_maker_net_4bp_most_conservative": r2(m["t_maker_net_4bp_most_conservative"]),
                    "maker_in_taker_out": r2(m["maker_in_taker_out"]), "t_maker_in_taker_out": r2(m["t_maker_in_taker_out_day"]),
                    "breakeven_round_trip_bp": r2(b["gross"]["mean"]), "breakeven_per_side_bp": r2(b["gross"]["mean"] / 2)}
    out["pairs"][key] = rec
out["overlap_sweep_N240_x10_k1"] = {}
for seg in ("design", "validation"):
    for h in ("240", "60"):
        blk = ov[seg][h]
        pick = lambda r: {k: r2(r[k]) if isinstance(r[k], float) else r[k] for k in ("events", "gross", "t_gross", "net_taker", "t_net_taker", "maker_net_4bp", "t_maker_net_4bp") if k in r}
        out["overlap_sweep_N240_x10_k1"][f"{seg}@{h}"] = {
            "by_earlier_events_in_240min": {k: dict(pick(v), share_of_events=r2(v["share_of_events"]), share_of_total_gross=r2(v["share_of_total_gross"]))
                                            for k, v in blk["by_earlier_events_in_240min"].items()},
            "cap_on_open_positions_per_coin": {k: pick(v) for k, v in blk["cap_on_open_positions_per_coin"].items()},
            "concurrency": blk["concurrency_uncapped"]}
out["concurrency_all_coins"] = J("concurrency.json")
out["cap1_bruteforce_by_timestamp"] = J("cap1_bruteforce.json")
out["costs_realism"] = {seg: {h: {"simulated_passive_exit": cr[seg][h]["simulated_passive_exit"], "funding": cr[seg][h]["funding_bp_paid_per_event"],
                                  "entry_through_2bp_rest5": cr[seg][h]["entry_sensitivity"]["rest5_through2bp"],
                                  "entry_through_5bp_rest5": cr[seg][h]["entry_sensitivity"]["rest5_through5bp"]} for h in ("240", "60")} for seg in ("design", "validation", "both")}
out["placebo"] = {k: {"real_gross": r2(v["real_gross"]), "days_shift_mean": r2(v["days"]["gross_mean"]), "days_shift_sd": r2(v["days"]["gross_sd_across_draws"]),
                      "days_shift_max": r2(v["days"]["gross_max"]), "hours_after_mean": r2(v["hours_after"]["gross_mean"]), "hours_before_mean": r2(v["hours_before"]["gross_mean"]),
                      "hours_mixed_mean": r2(v["hours"]["gross_mean"])} for k, v in pl.items()}
out["part_b"] = pb
out["lead_as_stated_long_alt_one_position"] = J("lead_as_stated.json")
out["posthoc_cascade_long_one_position"] = {seg: {k: {kk: (r2(vv) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk not in ("by_year", "by_coin")}
                                                 for k, v in J("posthoc_cascade.json")[seg].items()} for seg in ("design", "validation")}
out["planted_effect"] = J("planted_effect.json")
out["table_recount"] = {k: v for k, v in J("table_recount.json").items() if k in ("looks", "abs_t_gt_3", "abs_t_gt_3p9", "t_lt_minus_3_against_textbook", "net_taker_positive",
                                                                                 "net_taker_positive_and_abs_t_gt_3", "max_abs_gross_up_to_15min", "controls_max_abs_t")}
json.dump(out, open(os.path.join(al.OUT, "audit_summary.json"), "w"), indent=1)
print("written", os.path.join(al.OUT, "audit_summary.json"), len(json.dumps(out)))
print(json.dumps(out["pairs"]["sweep_N240_x10_k1@240"], indent=1))
