import pandas as pd, numpy as np
pd.set_option("display.width", 250)
t = pd.read_parquet("out/part_b_trades_design.parquet")
x = t[(t.market == "um") & (t.set == "trend_exit") & (t.window == 240) & (t.structure == "bos_s5_all")].copy()
x["time"] = pd.to_datetime(x.minute * 60, unit="s", utc=True)
print(x.sort_values("oracle", ascending=False)[["symbol", "time", "oracle", "delay_end", "improvement"]].head(8).to_string(index=False))
from algotrade.data import store
for sym, day in [(r.symbol, r.time) for r in x.sort_values("oracle", ascending=False).head(2).itertuples()]:
    df = store.load_klines("um", "1m", sym, start=day, end=day + pd.Timedelta(hours=4))
    i = df.high.idxmax()
    print(sym, day, "max high bar:", i, df.loc[i - pd.Timedelta(minutes=2): i + pd.Timedelta(minutes=2), ["open", "high", "low", "close", "volume"]].to_string())
    sp = store.load_klines("spot", "1m", sym, start=i - pd.Timedelta(minutes=2), end=i + pd.Timedelta(minutes=3))
    print("spot same minutes:", sp[["open", "high", "low", "close", "volume"]].to_string())
