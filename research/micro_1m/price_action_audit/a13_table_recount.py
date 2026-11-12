"""Audit: recount the summary statistics the researcher quotes from the 420-row design table (their CSV, my arithmetic)."""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats as sps
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_lib as al
T = os.path.join(os.path.dirname(al.HERE), "price_action", "out")
t = pd.read_csv(os.path.join(T, "table_design.csv"))
real = t[t["family"] != "control"]
ctl = t[t["family"] == "control"]
out = {
    "definitions": int(real["def_id"].nunique()), "horizons": sorted(int(h) for h in real["h"].unique()), "looks": int(len(real)),
    "abs_t_gt_3": int((real["t_excess"].abs() > 3).sum()), "abs_t_gt_3p9": int((real["t_excess"].abs() > 3.9).sum()),
    "t_lt_minus_3_against_textbook": int((real["t_excess"] < -3).sum()), "t_gt_3_for_textbook": int((real["t_excess"] > 3).sum()),
    "net_taker_positive": int((real["net_taker"] > 0).sum()),
    "net_taker_positive_and_abs_t_gt_3": real[(real["net_taker"] > 0) & (real["t_excess"].abs() > 3)][["def_id", "h", "gross", "net_taker", "t_excess"]].round(2).values.tolist(),
    "net_taker_positive_others": real[(real["net_taker"] > 0) & (real["t_excess"].abs() <= 3)][["def_id", "h", "net_taker", "t_excess"]].round(2).values.tolist(),
    "max_abs_gross_up_to_15min": float(real[real["h"] <= 15]["gross"].abs().max()),
    "max_abs_gross_up_to_15min_row": real.loc[real[real["h"] <= 15]["gross"].abs().idxmax(), ["def_id", "h", "gross", "t_excess"]].tolist(),
    "controls_max_abs_t": float(ctl["t_excess"].abs().max()),
    "median_naive_over_clustered_t": {int(h): float((g["t_naive"].abs() / g["t_excess"].abs()).median()) for h, g in real.groupby("h")},
    "bonferroni_z_420_two_sided_5pct": float(sps.norm.isf(0.025 / 420)),
    "expected_false_positives_abs_t_gt_3_under_null_420": float(420 * 2 * sps.norm.sf(3)),
    "limit_definitions_1min": real[real["def_id"].str.contains("limit") & (real["h"] == 1)][["def_id", "gross", "t_excess"]].round(2).values.tolist(),
    "abs_t_gt_3_and_gross_minus_4_positive": int(((real["t_excess"].abs() > 3) & (real["net_maker_bound"] > 0)).sum()),
}
# rows of the reported results table
want = [("sweep_N240_x10_k1", 240), ("sweep_N240_x10_k1", 60), ("sweep_N30_x2_k3", 5), ("sweep_N60_x10_k1", 240), ("sweepmat_N60_x2_k1", 1), ("sweepmat_N240_x2_k3", 60),
        ("hold_N30_x2_k3", 5), ("hold_N240_x10_k3", 60), ("pdsweep_x2_k3", 60), ("pdhold_x2_k1", 1), ("bos_s5_all", 1), ("bos_s5_with", 15), ("bos_s5_against", 60),
        ("fvg_m0p5", 5), ("fvgfill_m0p5_W60", 5), ("fvglimit_m0p5_W60", 1), ("squeeze_M30_q0p1_v2", 1), ("squeeze_M30_q0p1_v2", 60), ("ob_W60", 15), ("oblimit_W60", 1),
        ("disp_m3", 240), ("disp_m6", 240)]
idx = real.set_index(["def_id", "h"])
print(idx.loc[want, ["events", "days", "gross", "base", "excess", "t_excess", "t_naive", "net_taker", "net_maker_bound"]].round(2).to_string())
print(json.dumps(out, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
json.dump(out, open(os.path.join(al.OUT, "table_recount.json"), "w"), indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v))
