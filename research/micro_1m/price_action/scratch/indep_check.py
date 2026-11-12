"""Independent re-computation (plain pandas on the raw frame, no project code) of one selected definition on one coin."""
import numpy as np, pandas as pd
from algotrade.data import store
df = store.load_klines("um", "1m", "BTCUSDT", end="2024-01-01")
assert (df.index[1:] - df.index[:-1] == pd.Timedelta("1min")).all()   # BTC perp has no gaps in design
N, x = 240, 10
ph = df.high.rolling(N).max().shift(1); pl = df.low.rolling(N).min().shift(1)
bear = (df.high >= ph * (1 + x / 1e4)) & (df.close < ph)
bull = (df.low <= pl * (1 - x / 1e4)) & (df.close > pl)
sig = bull.astype(int) - bear.astype(int)
out = {}
for h in (60, 240):
    fwd = (df.open.shift(-(1 + h)) / df.open.shift(-1) - 1) * 1e4
    g = (sig * fwd)[sig != 0].dropna()
    g = g[g.index >= df.index[1500]]
    out[h] = (len(g), g.mean())
print("independent:", out)
import pandas as pd
d = pd.read_parquet("out/daily_design_um_BTCUSDT.parquet")
d = d[d.def_id == "sweep_N240_x10_k1"]
print("pipeline   :", {h: (int(d[f"n{h}"].sum()), d[f"sg{h}"].sum() / d[f"n{h}"].sum()) for h in (60, 240)})
# N30 x2 k3, h=5, via an explicit loop over a 60-day slice
sl = df.loc["2021-05-01":"2021-06-29"]
h_, l_, c_, o_ = (sl[k].to_numpy() for k in ("high", "low", "close", "open"))
n = len(sl); N, x, k = 30, 2, 3
ev = []
for t in range(N + k, n - 6):
    bull = bear = False
    for s in range(t - k + 1, t + 1):
        hi = h_[s - N:s].max(); lo = l_[s - N:s].min()
        if h_[s] >= hi * (1 + x / 1e4) and all(c_[u] >= hi for u in range(s, t)) and c_[t] < hi: bear = True
        if l_[s] <= lo * (1 - x / 1e4) and all(c_[u] <= lo for u in range(s, t)) and c_[t] > lo: bull = True
    d_ = int(bull) - int(bear)
    if d_: ev.append((t, d_, d_ * (o_[t + 6] / o_[t + 1] - 1) * 1e4))
ev = pd.DataFrame(ev, columns=["t", "d", "g"])
print("loop slice 2021-05/06: events", len(ev), "mean gross@5", round(ev.g.mean(), 4))
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eventstudy as es, structures as st
bars = es.bars_from_frame(sl, "BTCUSDT", "um")
e = st.detect(bars, ["sweep_N30_x2_k3"])["sweep_N30_x2_k3"]
f = es.forward_returns(bars, (5,))[5]
keep = (e.idx >= N + k) & (e.idx < n - 6)
print("module slice         : events", int(keep.sum()), "mean gross@5", round(float(np.mean(e.direction[keep] * f[e.idx[keep]])), 4),
      "identical events:", np.array_equal(e.idx[keep], ev.t.to_numpy()) and np.array_equal(e.direction[keep], ev.d.to_numpy()))
