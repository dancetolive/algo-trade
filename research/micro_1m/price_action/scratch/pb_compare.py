import pandas as pd
pd.set_option("display.width", 280); pd.set_option("display.max_rows", 500)
d = pd.read_csv("out/part_b_design.csv"); v = pd.read_csv("out/part_b_validation.csv")
t = pd.concat([d, v])
cols = ["segment", "market", "set", "structure", "window", "trades", "days", "triggered", "mean_wait_min", "mean_bp", "median_bp", "t", "worse", "mean_if_better", "mean_if_worse", "delay_end_bp", "t_delay_end", "info_bp", "t_info", "oracle_median_bp"]
for market in ("spot", "um"):
    for s in ("trend_entry", "trend_exit", "trend_both", "all_days_both"):
        x = t[(t.market == market) & (t.set == s)].sort_values(["structure", "window", "segment"])
        print(f"\n== {market} / {s} ==")
        print(x[cols[3:4] + cols[4:5] + cols[0:1] + cols[5:]].to_string(index=False, float_format=lambda z: f"{z:.2f}"))
