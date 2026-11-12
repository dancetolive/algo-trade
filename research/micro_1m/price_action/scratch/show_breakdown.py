import sys, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
b = pd.read_csv("out/breakdown_design.csv")
pairs = [a.split("@") for a in sys.argv[1:]]
cols = ["by", "value", "events", "days", "gross", "base", "excess", "t_excess", "hit", "net"]
for d, h in pairs:
    part = b[(b.def_id == d) & (b.h == int(h))]
    print(f"\n=== {d} @ {h} min ===")
    print(part[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
