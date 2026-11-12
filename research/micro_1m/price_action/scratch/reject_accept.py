import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
b = pd.read_csv("out/breakdown_design.csv")
rows = []
for N in (30, 60, 240):
    for x in (2, 10):
        for name in (f"sweep_N{N}_x{x}_k1", f"sweep_N{N}_x{x}_k3", f"hold_N{N}_x{x}_k1", f"hold_N{N}_x{x}_k3"):
            for h in (15, 60):
                part = b[(b.def_id == name) & (b.h == h) & (b.by == "direction")].set_index("value")
                # express everything as "what a BUYER earned after a thrust to a new LOW" and "what a SELLER earned after a thrust to a new HIGH"
                if name.startswith("sweep"):
                    low_buy, high_sell = part.loc["long"], part.loc["short"]
                    sign = 1
                else:  # hold: short = acceptance below the low (hypothesis short) -> the buyer earns minus that
                    low_buy, high_sell = part.loc["short"], part.loc["long"]
                    sign = -1
                rows.append({"def": name, "h": h, "after_new_low_n": int(low_buy.events), "buyer_excess": sign * low_buy.excess, "buyer_t": sign * low_buy.t_excess,
                             "after_new_high_n": int(high_sell.events), "seller_excess": sign * high_sell.excess, "seller_t": sign * high_sell.t_excess})
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
