"""Independent checks: (1) one validated pair on one coin recomputed with plain pandas; (2) Part B mechanics on design trades by brute force."""
import numpy as np, pandas as pd
from algotrade.data import store

# (1) SOLUSDT perp, validation segment, sweep_N240_x10_k1 at 60 and 240 minutes (already looked at; this only re-derives the same numbers)
df = store.load_klines("um", "1m", "SOLUSDT", start="2023-12-25")
assert (df.index[1:] - df.index[:-1] == pd.Timedelta("1min")).all()
N, x = 240, 10
ph = df.high.rolling(N).max().shift(1); pl = df.low.rolling(N).min().shift(1)
bear = (df.high >= ph * (1 + x / 1e4)) & (df.close < ph)
bull = (df.low <= pl * (1 - x / 1e4)) & (df.close > pl)
sig = bull.astype(int) - bear.astype(int)
got = {}
for h in (60, 240):
    fwd = (df.open.shift(-(1 + h)) / df.open.shift(-1) - 1) * 1e4
    g = (sig * fwd)[(sig != 0) & (df.index >= pd.Timestamp("2024-01-01", tz="UTC"))].dropna()
    got[h] = (len(g), round(float(g.mean()), 6))
d = pd.read_parquet("out/daily_validation_um_SOLUSDT.parquet"); d = d[d.def_id == "sweep_N240_x10_k1"]
pipe = {h: (int(d[f"n{h}"].sum()), round(float(d[f"sg{h}"].sum() / d[f"n{h}"].sum()), 6)) for h in (60, 240)}
print("independent:", got); print("pipeline   :", pipe); assert got == pipe

# (2) Part B by brute force: BTC spot, design, first 12 trend entries and exits; small sweep (N30, x2, k3) and break of structure (5/5)
sp = store.load_klines("spot", "1m", "BTCUSDT", end="2024-01-01")
o, h_, l_, c_ = (sp[k].to_numpy() for k in ("open", "high", "low", "close"))
pos = {t: i for i, t in enumerate(sp.index.asi8 // 60_000_000)}
def small_sweep_dir(t, side):
    """Is bar t a small-sweep event in direction `side` (+1 bullish)?  Literal definition."""
    for s in range(t - 2, t + 1):
        hi, lo = h_[s - 30:s].max(), l_[s - 30:s].min()
        if side < 0 and h_[s] >= hi * (1 + 2e-4) and all(c_[u] >= hi for u in range(s, t)) and c_[t] < hi: return True
        if side > 0 and l_[s] <= lo * (1 - 2e-4) and all(c_[u] <= lo for u in range(s, t)) and c_[t] > lo: return True
    return False
tr = pd.read_parquet("out/part_b_trades_design.parquet")
tr = tr[(tr.market == "spot") & (tr.symbol == "BTCUSDT") & (tr.window == 240) & (tr.structure == "sweep_N30_x2_k3") & tr.set.isin(["trend_entry", "trend_exit"])]
checked = 0
for r in tr.sort_values("minute").iloc[::7].head(24).itertuples():
    i0 = pos[int(r.minute)]
    assert sp.index[i0].hour == 0 and sp.index[i0].minute == 0
    fill = i0 + 240
    for t in range(i0, i0 + 240):
        # contiguous minutes are guaranteed by the pipeline's gap rule; verify it here as well
        if small_sweep_dir(t, r.side):
            # the opposite sweep in the same bar cancels the event in the module (ambiguous bar)
            if small_sweep_dir(t, -r.side): continue
            fill = t + 1; break
    imp = r.side * (o[i0] - o[fill]) / o[i0] * 1e4
    assert (sp.index[fill] - sp.index[i0]) == pd.Timedelta(minutes=fill - i0)
    assert fill - i0 == r.delay and abs(imp - r.improvement) < 1e-9, (r, fill - i0, imp)
    checked += 1
print("Part B brute-force check passed for", checked, "trades")
# the trend trades themselves: recompute the blended weights by hand
dly = store.load_klines("spot", "1d", "BTCUSDT", end="2024-01-01")
w = sum(((dly.close > dly.close.rolling(n).mean()) & dly.close.rolling(n).mean().notna()).astype(float) for n in (50, 100, 200)) / 3 / 2
chg = w.diff()
mine = sorted(int((t + pd.Timedelta(days=1)).timestamp() // 60) for t in chg[chg > 1e-12].index if t >= pd.Timestamp("2019-12-31", tz="UTC"))
allent = pd.read_parquet("out/part_b_trades_design.parquet")
allent = allent[(allent.market == "spot") & (allent.symbol == "BTCUSDT") & (allent.window == 240) & (allent.structure == "bos_s5_all") & (allent.set == "trend_entry")]
theirs = sorted(allent.minute.tolist())
print("BTC entries by hand:", len(mine), "in pipeline (after gap / warm-up drops):", len(theirs), "pipeline subset of hand list:", set(theirs) <= set(mine),
      "dropped:", [str(pd.to_datetime(m * 60, unit="s", utc=True)) for m in sorted(set(mine) - set(theirs))])
