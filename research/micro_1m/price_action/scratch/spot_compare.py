import pandas as pd, numpy as np
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
um = pd.read_csv("out/table_design_um_btc_eth.csv").set_index(["def_id", "h"])
sp = pd.read_csv("out/table_design_spot.csv").set_index(["def_id", "h"])
both = um[["events", "gross", "excess", "t_excess", "net_taker"]].join(sp[["events", "gross", "excess", "t_excess", "net_taker"]], lsuffix="_perp", rsuffix="_spot")
keys = [("sweep_N30_x2_k3", 5), ("sweep_N30_x2_k3", 15), ("sweep_N240_x10_k1", 60), ("sweep_N240_x10_k1", 240), ("sweep_N60_x10_k1", 240), ("hold_N30_x2_k3", 5), ("hold_N240_x10_k3", 60),
        ("sweepmat_N60_x2_k1", 1), ("pdsweep_x2_k3", 60), ("bos_s5_all", 1), ("bos_s5_with", 15), ("fvg_m0p5", 5), ("fvgfill_m0p5_W60", 5), ("fvglimit_m0p5_W60", 1), ("squeeze_M30_q0p1_v2", 1), ("ob_W60", 15), ("oblimit_W60", 1), ("disp_m3", 60)]
print(both.loc[keys].round(2).to_string())
real = both[~both.index.get_level_values(0).str.startswith("ctl")]
print("\ncorrelation of t_excess perp(BTC,ETH) vs spot(BTC,ETH) across all 420 looks:", round(np.corrcoef(real.t_excess_perp, real.t_excess_spot)[0,1], 3))
print("sign agreement among looks with |t_perp|>3:", round((np.sign(real.t_excess_perp) == np.sign(real.t_excess_spot))[real.t_excess_perp.abs() > 3].mean(), 3), "n=", int((real.t_excess_perp.abs() > 3).sum()))
print("spot looks with net_taker > 0:", int((real.net_taker_spot > 0).sum()), " perp(BTC,ETH) looks with net_taker>0:", int((real.net_taker_perp > 0).sum()))
print(real[real.net_taker_perp > 0].round(2).to_string())
