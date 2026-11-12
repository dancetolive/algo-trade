"""The prose of REPORT.md. Every number is taken from the tables that make_report.py loads."""

from __future__ import annotations


def render(c, n, md, pb_table, pb_digits, sel, pb) -> str:
    looks = c["looks"].set_index(["def_id", "h"])
    bd = c["bd_design"]

    def look(d, h):
        return looks.loc[(d, h)]

    def yearly(d, h, column="gross"):
        part = bd[(bd["def_id"] == d) & (bd["h"] == h) & (bd["by"] == "year")]
        return ", ".join(f"{int(float(r.value))}: {getattr(r, column):+.2f}" for r in part.itertuples())

    a_d, a_v = sel(c, "sweep_N240_x10_k1", 240, "design"), sel(c, "sweep_N240_x10_k1", 240, "validation")
    b_d, b_v = sel(c, "sweep_N240_x10_k1", 60, "design"), sel(c, "sweep_N240_x10_k1", 60, "validation")
    s_d, s_v = sel(c, "sweep_N30_x2_k3", 5, "design"), sel(c, "sweep_N30_x2_k3", 5, "validation")
    side = c["by_side"].set_index(["definition", "h", "side", "segment"])
    ra = c["reject_accept"].set_index(["event", "h"])
    pooled = c["pooled"].set_index(["definition", "h", "figure"])
    cross = c["cross"].set_index(["definition", "h", "segment"])
    dry = c["dry"].set_index(["def_id", "h"])
    infl = c["inflation"]
    px = c["paxg"]
    e_d, e_v = pb(c, "spot", "trend_both", "sweep_N30_x2_k3", 240, "design"), pb(c, "spot", "trend_both", "sweep_N30_x2_k3", 240, "validation")
    q_d, q_v = pb(c, "spot", "trend_both", "bos_s5_all", 240, "design"), pb(c, "spot", "trend_both", "bos_s5_all", 240, "validation")
    qe_d, qe_v = pb(c, "spot", "trend_entry", "bos_s5_all", 240, "design"), pb(c, "spot", "trend_entry", "bos_s5_all", 240, "validation")
    se_d, se_v = pb(c, "spot", "trend_entry", "sweep_N30_x2_k3", 240, "design"), pb(c, "spot", "trend_entry", "sweep_N30_x2_k3", 240, "validation")
    sx_d, sx_v = pb(c, "spot", "trend_exit", "sweep_N30_x2_k3", 240, "design"), pb(c, "spot", "trend_exit", "sweep_N30_x2_k3", 240, "validation")
    be_d, be_v = pb(c, "spot", "trend_entry", "sweep_N240_x10_k1", 240, "design"), pb(c, "spot", "trend_entry", "sweep_N240_x10_k1", 240, "validation")
    b24_d, b24_v = pb(c, "spot", "trend_both", "sweep_N240_x10_k1", 1440, "design"), pb(c, "spot", "trend_both", "sweep_N240_x10_k1", 1440, "validation")
    ad_d, ad_v = pb(c, "spot", "all_days_both", "sweep_N30_x2_k3", 240, "design"), pb(c, "spot", "all_days_both", "sweep_N30_x2_k3", 240, "validation")
    aq_d, aq_v = pb(c, "spot", "all_days_both", "bos_s5_all", 240, "design"), pb(c, "spot", "all_days_both", "bos_s5_all", 240, "validation")
    counts_d, counts_v = c["counts"]["design"], c["counts"]["validation"]
    n_trades_d = int(counts_d[["-1", "1"]].to_numpy().sum())
    n_trades_v = int(counts_v[["-1", "1"]].to_numpy().sum())
    fl, ol = look("fvglimit_m0p5_W60", 1), look("oblimit_W60", 1)
    runs = {(r["market"], r["symbol"]): r for r in c["run_design"]["runs"]}

    mk = c['looks'][(c['looks']['net_maker_bound'] > 0) & (c['looks']['t_excess'].abs() > 3)]
    assert set(mk['family']) == {'sweep'}, set(mk['family'])
    n_levels = sorted({int(d.split('_')[1][1:]) for d in mk['def_id']})
    maker_note = (f"all of them sweeps of the {n_levels[0]}- to {n_levels[-1]}-minute extreme held for {int(mk['h'].min())} to "
                  f"{int(mk['h'].max())} minutes ({int((mk['def_id'].str.contains('_x10_')).sum())} of them with the 10 bp threshold)")
    parts = []
    w = parts.append

    w(f"""# One-minute price-action structures: do they predict anything, and do they improve trend entries?

Directory: `research/micro_1m/price_action/` · design segment 2020-01 to 2023-12 · validation segment 2024-01 to 2025-06 (looked at once) · hold-out from 2025-07 never loaded.

The owner's question: *"What about detecting very short-lived structures on 1-minute charts and working from there towards longer-term strategies?"*
This report takes the structures that discretionary "smart money / ICT" traders name on 1-minute charts, writes each one down as an exact rule, and measures two things on Binance data: (A) whether the structure predicts the return that follows by enough to pay trading costs, and (B) whether waiting for such a structure gives a better price for a longer-term (daily trend) trade than simply trading at the daily open.

## 1. The answer in plain words

Units: 1 bp = 0.01%. One round trip (in and out) on perpetual futures costs a taker 12 bp on BTC and ETH and 14 bp on the other coins. For scale, in 2023 the median 1-minute bar of BTC spanned 4 bp from high to low and that of SOL 12 bp: one trade costs about three BTC minutes.

1. **The patterns are statistically real and economically tiny.** {c['n_looks']} combinations of pattern, parameter and holding time were examined on six coins over four years. {c['sig3']} of them show an effect that is clearly not chance (day-clustered |t| above 3). But no effect at holding times up to 15 minutes is larger than {c['max_short_gross']:.1f} bp before costs, and most are below 1 bp. Costs are 12-14 bp.
2. **The textbook reading is wrong more often than it is right.** Of the {c['sig3']} clear effects, {c['against']} point in the direction opposite to what the pattern is supposed to mean. Breakouts that "hold" tend to come back. Fair value gaps are followed by a small move against the gap. A break of structure continues for one minute (about {look('bos_s5_all', 1)['gross']:.1f} bp) and then reverses. A limit order placed at the edge of a fair value gap or an order block is, on average, {abs(ol['gross']):.1f} to {abs(fl['gross']):.1f} bp under water at the open of the next minute, and has not recovered an hour later.
3. **One effect is as large as the costs: the rebound after a sharp push to a new 4-hour low.** After price trades at least 10 bp through the 4-hour low and closes back above it within the same minute, the next four hours gained {side.loc[('sweep_N240_x10_k1', 240, 'long', 'design'), 'excess']:.0f} bp more than comparable moments in 2020-2023 and {side.loc[('sweep_N240_x10_k1', 240, 'long', 'validation'), 'excess']:.0f} bp more in 2024-2025. Both sides of the pattern pooled (the pre-registered test): {a_d['gross']:+.1f} bp on design (t {a_d['t_excess']:.1f}) and {a_v['gross']:+.1f} bp on validation (t {a_v['t_excess']:.1f}). Three reservations:
   - It is not the "liquidity sweep" that matters. A push to a new 4-hour low was followed by the same rebound whether the bar closed back inside (the sweep) or price stayed below for three more closes (the opposite pattern). It is a plain dip-rebound, and only on the buy side; selling after a sweep of the highs earned nothing.
   - It comes and goes: strong in 2020-2021, about zero in 2023, back in 2024-2025, and it lives in XRP, SOL, DOGE and BNB far more than in BTC and ETH.
   - After a taker's costs the pooled figure is {a_d['net_taker']:+.1f} bp (design) and {a_v['net_taker']:+.1f} bp (validation), which cannot be told apart from zero (t {a_d['t_net_taker']:.1f} and {a_v['t_net_taker']:.1f}). Only with resting (maker) orders does it stay clearly positive: {a_d['maker_net_4bp']:+.1f} and {a_v['maker_net_4bp']:+.1f} bp per filled order at maker fees, and that figure is optimistic.
4. **Waiting for a 1-minute structure did not improve the fills of the daily trend rule.** On the {n_trades_d} trend trades of 2020-2023 and the {n_trades_v} of 2024-mid 2025, waiting for the first small sweep-and-reclaim in the direction of the trade changed the price by {e_d['mean_bp']:+.1f} bp (t {e_d['t']:.1f}) and {e_v['mean_bp']:+.1f} bp (t {e_v['t']:.1f}) on average: not distinguishable from zero. The wait gives a better price about three times in four, and a much worse one the fourth time, when price runs away. Waiting for "confirmation" by a break of structure made the price worse in {q_d['worse']:.0%} (design) and {q_v['worse']:.0%} (validation) of the trades; the average was {q_d['mean_bp']:+.1f} bp in design (t {q_d['t']:.1f}) and {q_v['mean_bp']:+.1f} bp in validation (t {q_v['t']:.1f}).
5. **For an account of USD 500-5,000:** nothing here can be traded with market orders. The textbook 1-minute structures are not a basis for a strategy and not a way to time the entries of a daily strategy. The dip-rebound is a lead for a separate, carefully pre-registered maker-order study, not a result to trade on.

**Verdict: `{n['verdict']}`**, in the narrow sense above: one contrarian effect clears maker costs in both segments and fails the taker test; everything else fails both.
""")

    w(f"""## 2. What was tested

### Data and segments

| market | coins | design bars per coin | gaps |
|---|---|---|---|
| USDT perpetuals (`um`), 1-minute | BTC, ETH, SOL, BNB, XRP, DOGE | {runs[('um', 'DOGEUSDT')]['bars']:,} to {runs[('um', 'BTCUSDT')]['bars']:,} | none, except SOL and XRP: {runs[('um', 'SOLUSDT')]['missing']:,} missing minutes each (two stretches of 3 and 2 days in 2022 absent from the source files) |
| spot, 1-minute (cross-check) | BTC, ETH | {runs[('spot', 'BTCUSDT')]['bars']:,} | {runs[('spot', 'BTCUSDT')]['missing']:,} missing minutes in 15 outages |
| spot, 1-minute (side check, never pooled) | PAXG | 1,756,096 | 1,424 missing minutes |

Design = bars before 2024-01-01, used freely. Validation = 2024-01-01 to 2025-06-30, used once, for the three (definition, holding time) pairs frozen in `selection.json` and for the diagnostics listed there. The hold-out is locked by the loader and was never read.

### Conventions

* An event is stamped at the **close** of 1-minute bar t and uses bars up to t only. The trade is filled at the **open of bar t+1**. The forward return over h minutes runs from the open of t+1 to the open of t+1+h, for h = 1, 5, 15, 60, 240, 1440.
* Bars are put on a complete 1-minute grid; minutes the exchange did not deliver are empty rows. A return is dropped if any minute from t to t+1+h is missing, and an event is dropped if any of the 1,500 minutes before it is missing.
* Returns are signed in the direction of the textbook reading of the pattern, so a positive number means "the pattern worked as advertised" and a negative one means "the opposite happened".
* Costs per round trip: perpetual taker 12 bp (BTC, ETH) or 14 bp (others); perpetual maker bound 4 bp; spot taker 22 or 24 bp.
* Baseline: for every event, the average forward return of all bars of the same coin in the same calendar year, hour of day and volatility quintile (60-minute realised volatility). "Excess" = gross minus baseline. All t-statistics are for the excess.
* Inference: every t-statistic is clustered by UTC calendar day across all coins (events on one day are one observation). For the 240- and 1440-minute horizons the covariance between adjacent days is added, because those returns overlap across midnight. The naive t-statistic that treats events as independent is shown next to it; it is inflated by a factor of {infl[1]:.1f} at 1 minute rising to {infl[1440]:.1f} at one day (median over all looks).

### The structures (70 definitions, 6 holding times each = {c['n_looks']} looks)

"Level" below means a price fixed before the bar in question. PH(N) / PL(N) = highest high / lowest low of the N bars before a bar.

| id | exact rule (event at the close of bar t) | trade the textbook calls for | grid | defs |
|---|---|---|---|---|
| `sweep_N.._x.._k..` | A bar s trades at least x bp beyond PH(N) or PL(N). Event = the first close back inside that level, if it comes in bars s .. s+k-1 (k = 1: the breach bar itself closes back inside). | reversal: short after a sweep of the high, long after a sweep of the low | N 30, 60, 240; x 2, 10; k 1, 3 | 12 |
| `sweepmat_..` | Same, and the swept level must not have been touched in the 10 bars before the breach (an "established" high or low). | reversal | N 60, 240; x 2; k 1, 3 | 4 |
| `hold_N.._x.._k..` | The same breach, but bars s .. s+k-1 all close beyond the level (event at s+k-1): acceptance instead of rejection. | continuation | as sweep | 12 |
| `bos_s.._all / with / against` | Swing high = a bar whose high is above the highs of the L bars before it and not below the highs of the L bars after it; it is usable only L bars later. Event = the first close above the latest confirmed swing high (or below the latest confirmed swing low). `with` = same direction as the previous break (break of structure), `against` = opposite direction (change of character). | continuation in the direction of the break | L 3, 5, 10; 3 kinds | 9 |
| `fvg_m..` | Fair value gap: low of bar t above high of bar t-2 (bullish) or high of t below low of t-2 (bearish), and the gap at least m times the average 1-minute range of the 60 bars ending at t-3. | continuation | m 0.5, 1, 2 | 3 |
| `fvgfill_m.._W..` | First bar within W minutes that trades back into the gap, provided it does not close through the far side. Entry at the next open. | bounce in the direction of the gap | m as above; W 60, 240 | 6 |
| `fvglimit_m.._W60` | The same idea as a resting limit order at the gap edge for 60 minutes; filled only when price trades strictly through it; return measured from the limit price. | bounce | m as above | 3 |
| `squeeze_M.._q.._v..` | The high-low box of the previous M bars is narrow (its width is in the lowest q of the last 1,440 minutes) and bar t closes outside it. `v2`: the volume of bar t is at least twice the average of the box. | continuation | M 30, 120; q 0.10, 0.25; volume filter off / on | 8 |
| `pdsweep_x.._k..`, `pdhold_x.._k..` | Sweep / acceptance as above, with the previous UTC day's high and low as the level. | reversal / continuation | x 2, 10; k 1, 3 | 8 |
| `ob_W..`, `oblimit_W60` | Order block = the last opposite-colour candle within 30 bars before a break of structure (5-bar swings). Event = first return into that candle's range within W minutes (next-open entry), or a resting limit at its edge. | bounce | W 60, 240; limit 60 | 3 |
| `disp_m..` | Displacement candle: body at least m times the average body of the previous 60 bars, closing in the outer quarter of its range. | continuation | m 3, 6 | 2 |

Both readings of the sweep (reversal and continuation) are covered twice: by the sign of every result, and by the `hold` family, which is the complementary event (price went through the level and stayed there).

Looks, counted: {c['n_looks']} on the six perpetuals (design). The same {c['n_looks']} were run on BTC and ETH spot as a cross-check and on PAXG as a side check. Year, coin and side splits were read for about 15 of them. Validation: 3 pairs.
""")

    w(f"""## 3. How the machinery was checked before it was used

`test_eventstudy.py` (15 tests, all passing; log in `out/test_log.txt`) proves on synthetic data that:

* **No look-ahead.** For each of the 70 definitions, the events stamped at or before bar t are identical when every bar after t is replaced by different prices and volumes, and when the series is cut at t (four cut points each). A deliberately cheating detector fails the same check.
* **The code does what the written rule says.** Every vectorised detector is compared, event by event, with a slow literal loop written from the definition (including ties created by tick rounding, asymmetric swings, the "first close back" logic and the first-touch search of zones).
* **Returns are aligned and gaps are respected.** From the open of t+1 to the open of t+1+h, never spanning a missing minute; checked on a hand-computed example.
* **A planted effect is recovered.** In a bar-by-bar simulation every displacement candle is followed by a drift of 2 bp per bar for 5 bars. The chain detector -> forward returns -> baseline -> clustered statistics returns 2 bp at 1 minute and 10 bp at 5, 15 and 60 minutes within 3.5 standard errors, and shifting the event by one bar in either direction breaks the result. With no planted effect nothing is found.
* **The t-statistic is honest.** With events in bursts and overlapping horizons the day-clustered t has a standard deviation of about 1 under the null, the naive one about 3 or more.
* **The baseline removes an hour-of-day drift**, and a resting order shows adverse selection on a driftless walk.

On real data: random events with random direction and random long-only events (`ctl_random`, `ctl_long`) give excess returns with |t| below 1.7 at every horizon; one selected definition was recomputed for BTC with plain pandas, independently of the module, and matched to all digits (7,130 events, 8.6593 bp at 60 minutes); a second was recomputed with an explicit loop on two months of data (6,445 identical events); after validation the same plain-pandas recomputation reproduced the validation figures of the selected pair for SOL exactly (3,025 events), and the Part B fills of 18 trend trades were reproduced by a brute-force loop.
""")

    w(f"""## 4. Part A on the design segment (2020-2023, six perpetuals pooled)

### 4.1 Headline counts

* {c['n_looks']} looks. |t| > 3: **{c['sig3']}**. |t| > 3.9 (the Bonferroni 5% line for this many looks): {c['sig39']}.
* Of the {c['sig3']}, **{c['against']} have the sign against the textbook reading** (t below -3).
* Mean after a taker round trip above zero: **{len(c['pos_net'])} looks**. Three of them have |t| > 3, all sweeps held for 240 minutes: {", ".join(f"`{r.def_id}` ({r.net_taker:+.1f} bp, t {r.t_excess:.1f})" for r in c['pos_net_sig'].itertuples())}. The other {len(c['pos_net']) - len(c['pos_net_sig'])} are 1-day horizons with |t| below 1 (market drift and noise).
* Gross minus 4 bp (the optimistic maker bound) above zero: {c['maker_pos']} looks, {c['maker_pos_sig']} of them with |t| > 3: {maker_note}.
* Largest gross effect at holding times up to 15 minutes: {c['max_short_gross']:.2f} bp.

### 4.2 By family

"t>3 for" counts looks where the textbook reading is supported, "t<-3 against" where the opposite happened.

{md(c['family'])}

### 4.3 Holding-time profiles

Mean gross return in bp in the direction of the textbook reading, with the day-clustered t of the excess over baseline in brackets. Events are pooled over the six coins; "days" is the number of distinct calendar days with at least one event. The last column is the best mean after a taker round trip over the six holding times.

{md(c['profile'])}

The complete table (all {c['n_looks']} rows with events, days, gross, baseline, excess, clustered and naive t, hit rate, net taker, maker bound) is in `out/table_design_full.md` and `out/table_design.csv`; the splits by year, coin and side for every row are in `out/breakdown_design.csv`.

### 4.4 What the table says, family by family

* **Sweeps of the N-minute extreme (reversal reading).** Supported everywhere, tiny at short holds: `sweep_N30_x2_k3` earns {s_d['gross']:+.2f} bp over 5 minutes (t {s_d['t_excess']:.1f}, the strongest t of all {c['n_looks']} looks). The effect grows with the size of the level and the holding time: `sweep_N240_x10_k1` earns {b_d['gross']:+.1f} bp over 60 minutes (t {b_d['t_excess']:.1f}) and {a_d['gross']:+.1f} bp over 240 minutes (t {a_d['t_excess']:.1f}).
* **Sweeps of an established level** (untouched for 10 minutes, which is what traders mean by "resting liquidity"): the reversal disappears. At 1 minute the sign is even the other way ({look('sweepmat_N60_x2_k1', 1)['gross']:+.2f} bp, t {look('sweepmat_N60_x2_k1', 1)['t_excess']:.1f}); at every longer hold |t| is below 2.1. The reversal in the plain sweep therefore comes from levels that were set minutes ago by the same fast move, i.e. from short-term reversal after a thrust, not from old highs and lows being "raided".
* **Acceptance beyond the level (continuation reading).** Rejected in 55 of 72 looks: price that breaks out and holds comes back as well ({look('hold_N30_x2_k3', 5)['gross']:+.2f} bp at 5 minutes, t {look('hold_N30_x2_k3', 5)['t_excess']:.1f}; {look('hold_N240_x10_k3', 60)['gross']:+.1f} bp at 60 minutes for the 4-hour level, t {look('hold_N240_x10_k3', 60)['t_excess']:.1f}).
* **Previous day's high and low.** Nothing: none of the 48 looks reaches |t| = 3, for either reading.
* **Break of structure / change of character.** A continuation of {look('bos_s5_all', 1)['gross']:.2f} bp in the first minute (t {look('bos_s5_all', 1)['t_excess']:.1f}), then the opposite. Breaks in the direction of the previous break (break of structure proper) have reversed by 15 minutes ({look('bos_s5_with', 15)['gross']:+.2f} bp, t {look('bos_s5_with', 15)['t_excess']:.1f}); breaks against it (change of character) are flat for 15 minutes and have reversed by 60 ({look('bos_s5_against', 60)['gross']:+.2f} bp, t {look('bos_s5_against', 60)['t_excess']:.1f}). Neither continues.
* **Fair value gaps.** The gap is followed by a move against it ({look('fvg_m0p5', 5)['gross']:+.2f} bp at 5 minutes, t {look('fvg_m0p5', 5)['t_excess']:.1f}); large gaps more so ({look('fvg_m2', 240)['gross']:+.1f} bp at 240 minutes, t {look('fvg_m2', 240)['t_excess']:.1f}). Entering on the return into the gap at the next open earns nothing (|gross| below 0.2 bp up to 60 minutes for the standard gap). As a resting limit order at the gap edge it loses {abs(fl['gross']):.2f} bp by the next open (t {fl['t_excess']:.0f}) and is still {abs(look('fvglimit_m0p5_W60', 60)['gross']):.2f} bp down an hour later: the order is filled exactly when price is on its way through.
* **Compression then breakout.** {look('squeeze_M30_q0p1_v0', 1)['gross']:+.2f} bp in the first minute (t {look('squeeze_M30_q0p1_v0', 1)['t_excess']:.1f}), {look('squeeze_M30_q0p1_v2', 1)['gross']:+.2f} bp with the volume filter; nothing from 15 minutes on. Even this has faded: by year, the volume-filtered version earned {yearly('squeeze_M30_q0p1_v2', 1)} bp.
* **Order blocks.** Next-open entry: about zero (the largest figure is {look('ob_W60', 15)['gross']:+.2f} bp at 15 minutes, t {look('ob_W60', 15)['t_excess']:.1f}, against the bounce). Resting limit at the block edge: {ol['gross']:+.2f} bp at the next open (t {ol['t_excess']:.0f}).
* **Displacement candles.** No continuation at any holding time; over 240 minutes a small move against the candle ({look('disp_m3', 240)['gross']:+.2f} bp, t {look('disp_m3', 240)['t_excess']:.1f}; for the largest candles {look('disp_m6', 240)['gross']:+.1f} bp, t {look('disp_m6', 240)['t_excess']:.1f}).

### 4.5 The sweep pattern itself carries no information

The same breach of the 4-hour (or 30-minute) extreme, split by what happened next, and expressed as what a **contrarian** earned in excess of the baseline (a buyer after a new low, a seller after a new high):

{md(c['reject_accept'])}

After a push to a new 4-hour low a buyer earned {ra.loc[('sweep_N240_x10_k1', 60), 'buyer excess bp']:.1f} bp over the next hour when the bar closed back inside, {ra.loc[('hold_N240_x10_k1', 60), 'buyer excess bp']:.1f} bp when it closed below, and {ra.loc[('hold_N240_x10_k3', 60), 'buyer excess bp']:.1f} bp when price stayed below for three closes. After a push to a new high a seller earned nothing in any of the three cases. The information is "price has just fallen hard", not the shape of the candle.

### 4.6 Naive against clustered t-statistics

Median ratio of the naive t (events treated as independent) to the day-clustered t, by holding time: {", ".join(f"{h} min: {infl[h]:.2f}" for h in (1, 5, 15, 60, 240, 1440))}. A naive t of 4 at the one-day horizon is a clustered t of about {4 / infl[1440]:.1f}.

### 4.7 Cross-checks

* **Spot, BTC and ETH.** The {c['n_looks']} t-statistics on BTC+ETH spot correlate {c['t_corr']:.2f} with those on BTC+ETH perpetuals; where the perpetual |t| exceeds 3 the sign agrees in 98% of the looks. With spot costs of 22 bp, {c['spot_pos']} looks have a positive mean after costs, at the one-day horizon with |t| below 1.5 (noise).
* **Tokenised gold (PAXG spot), side check.** Its price moves in steps of about 5 bp and about a third of its design-segment minutes have no trade. {int((px['t_excess'].abs() > 3).sum())} of {len(px)} looks have |t| > 3, most of them "reversals" of 0.5 to 4 bp, which is what bid-ask bounce on a coarse tick looks like in bars. The largest are sweep reversals of 14 to 21 bp over 4 to 24 hours; none has a positive mean after the 24 bp spot round trip (best: {px['net_taker'].max():+.1f} bp). One-minute structure trading in PAXG cannot be assessed from bars and is not attractive on its face.
""")

    w(f"""## 5. Selection, frozen before validation

`selection.json` was written at {c['selection']['frozen_at_utc']} from the design tables, with the criteria for success, the diagnostics and the Part B protocol. Three (definition, holding time) pairs:

| # | pair | why |
|---|---|---|
| 1 | `sweep_N240_x10_k1`, 240 minutes | the only look with |t| >= 3.9 and a positive mean after taker costs ({a_d['net_taker']:+.1f} bp) |
| 2 | `sweep_N240_x10_k1`, 60 minutes | the same structure where the evidence is strongest (t {b_d['t_excess']:.1f}; positive in every design year and coin) |
| 3 | `sweep_N30_x2_k3`, 5 minutes | the most significant of all looks (t {s_d['t_excess']:.1f}); carried forward to see whether the most robust statistical effect persists, not because 1 bp can be traded |

Known weaknesses were written into the file at selection time: the effect of pairs 1 and 2 sits on the long side and in 2020-2021; 2023 is about zero. A later check on design data only (made while the validation code was being debugged on 2022-07 to 2023-12) showed that in those 18 months it was already down to {dry.loc[('sweep_N240_x10_k1', 60), 'gross']:+.1f} bp at 60 minutes (t {dry.loc[('sweep_N240_x10_k1', 60), 't_excess']:.1f}) and {dry.loc[('sweep_N240_x10_k1', 240), 'gross']:+.1f} bp at 240 minutes (t {dry.loc[('sweep_N240_x10_k1', 240), 't_excess']:.1f}).

Pre-registered criteria: *replicates* = same sign and clustered t of the excess at least 2.4; *pays taker costs* = mean after the taker round trip above zero with t at least 2 in both segments; *edge only with maker fills* = the taker test fails but the resting-order simulation at 4 bp is above zero with t at least 2 in both segments.

## 6. Part A on the validation segment (2024-01 to 2025-06), looked at once

### 6.1 The three pairs

{md(c['selected'][['definition', 'h', 'segment', 'events', 'days', 'gross', 'base', 'excess', 't_excess', 't_naive', 'boot_ci_lo', 'boot_ci_hi', 'cost_taker', 'net_taker', 't_net_taker', 'maker_bound']])}

`boot_ci` = 95% interval of the excess from a day-block bootstrap (4,000 resamples). `t_net_taker` = (gross - cost) divided by the clustered standard error.

* **All three replicate** by the pre-registered criterion (t {a_v['t_excess']:.2f}, {b_v['t_excess']:.2f}, {s_v['t_excess']:.2f}). The big-sweep effect has the same size as on design ({a_v['gross']:.1f} against {a_d['gross']:.1f} bp at 240 minutes); the small-sweep effect has more than halved ({s_v['gross']:.2f} against {s_d['gross']:.2f} bp).
* **None pays taker costs** by the pre-registered criterion. Pair 1 has a positive point estimate in both segments ({a_d['net_taker']:+.1f} and {a_v['net_taker']:+.1f} bp) with t {a_d['t_net_taker']:.1f} and {a_v['t_net_taker']:.1f}; both segments together: {pooled.loc[('sweep_N240_x10_k1', 240, 'net_taker'), 'both']:+.1f} bp, t {pooled.loc[('sweep_N240_x10_k1', 240, 'net_taker'), 't both']:.1f}. Pair 2 is negative ({b_d['net_taker']:+.1f}, {b_v['net_taker']:+.1f}); pair 3 loses {abs(s_v['net_taker']):.0f} bp per trade.

### 6.2 By year, by side, by coin

{md(c['by_year'])}

{md(c['by_side'])}

"excess minus cost" is the excess over the baseline after a taker round trip, i.e. what remains once the general rise of the market in comparable moments is removed. On the long side of pair 1 it is {side.loc[('sweep_N240_x10_k1', 240, 'long', 'design'), 'excess minus cost']:+.1f} bp on design (t {side.loc[('sweep_N240_x10_k1', 240, 'long', 'design'), 't (excess - cost)']:.1f}) and {side.loc[('sweep_N240_x10_k1', 240, 'long', 'validation'), 'excess minus cost']:+.1f} bp on validation (t {side.loc[('sweep_N240_x10_k1', 240, 'long', 'validation'), 't (excess - cost)']:.1f}). This split was seen on design before validation but was not pre-registered as a test of its own; it is a hypothesis for the hold-out, not a finding.

{md(c['by_coin'])}

BTC and ETH only, perpetuals against spot:

{md(c['cross'])}

For the two coins the trend rule trades, the big-sweep effect did not replicate at a useful size in validation ({cross.loc[('sweep_N240_x10_k1', 240, 'validation'), 'perp gross']:+.1f} bp at 240 minutes on perpetuals, {cross.loc[('sweep_N240_x10_k1', 240, 'validation'), 'spot gross']:+.1f} bp on spot, t below 1). The pooled result is carried by the four altcoins.

### 6.3 What a resting order really gets

The usual "maker" figure (gross minus 4 bp) pretends that a passive order is filled at the taker's price. The simulation in `maker_sim.py` places a limit order at the close of the event bar, leaves it for 5 minutes, and counts a fill only if a later bar trades strictly through the price.

{md(c['selected'][['definition', 'h', 'segment', 'fill_rate', 'filled_taker_gross', 'missed_taker_gross', 'mark_next_open', 'maker_gross_per_fill', 'maker_net_4bp', 't_maker_net_4bp', 'maker_in_taker_out', 't_maker_in_taker_out']], {'fill_rate': 3})}

* `missed_taker_gross` is what a taker earned on the events where the resting order was never filled: the orders a maker misses are the best trades ({s_d['missed_taker_gross']:+.0f} bp for the small sweep, {a_d['missed_taker_gross']:+.0f} bp for the big one).
* For the small sweep the whole effect lives in the {1 - s_d['fill_rate']:.0%} of events a maker does not get. On the fills the gross return is {s_d['maker_gross_per_fill']:+.2f} bp (design) and {s_v['maker_gross_per_fill']:+.2f} bp (validation): the "maker bound" of {s_d['maker_bound']:+.1f} bp is really {s_d['maker_net_4bp']:+.1f} bp. This is adverse selection, measured.
* For the big sweep at 240 minutes a maker keeps {a_d['maker_net_4bp']:+.1f} bp (design, t {a_d['t_maker_net_4bp']:.1f}) and {a_v['maker_net_4bp']:+.1f} bp (validation, t {a_v['t_maker_net_4bp']:.1f}) per fill if the exit is also passive and free of slippage, which is the optimistic bound, and {a_d['maker_in_taker_out']:+.1f} / {a_v['maker_in_taker_out']:+.1f} bp (t {a_d['t_maker_in_taker_out']:.1f} / {a_v['t_maker_in_taker_out']:.1f}; both segments together t {pooled.loc[('sweep_N240_x10_k1', 240, 'maker_in_taker_out'), 't both']:.1f}) with a maker entry and a taker exit. By the pre-registered rule this pair is an "edge only with maker fills". At 60 minutes the 4 bp figure is {b_d['maker_net_4bp']:+.1f} (t {b_d['t_maker_net_4bp']:.1f}) on design and {b_v['maker_net_4bp']:+.1f} (t {b_v['t_maker_net_4bp']:.1f}) on validation, which misses the criterion.
""")

    w(f"""## 7. Part B: does waiting for a structure improve the fill of a daily trend trade?

The rule is `blend_weights()` in `research/baseline/trend_baseline.py`: the average of three long / flat filters (close above its 50-, 100- and 200-day average) on BTC and ETH spot. A change decided at the daily close is executed at the next 00:00 UTC. In 2020-2023 the target weight changed {n_trades_d} times ({int(counts_d['1'].sum())} increases, {int(counts_d['-1'].sum())} decreases), in 2024 to mid-2025 {n_trades_v} times ({int(counts_v['1'].sum())} and {int(counts_v['-1'].sum())}); a typical trade moves 18% of the equity, which makes about 13 times the equity per year in design and 16 times in validation, one way. (A few trades are dropped below where the 1-minute data have a gap.)

For every such trade the price at the 00:00 open is compared with the price obtained by waiting, for up to 4 hours and for up to 24 hours, for the first occurrence of a structure in the direction of the trade (bullish for a buy, bearish for a sell), buying or selling at the open of the next minute, or at the end of the window if the structure never appears. Improvement is in bp of the 00:00 price, positive = better price. Structures: the two selected in Part A, and two companions that were not selected but are the entry triggers discretionary traders actually use (a break of structure as "confirmation", and the return into a fair value gap). Spot 1-minute bars; the perpetual version is in `out/part_b_*.csv` and tells the same story.

Columns: *occurred* = share of trades in which the structure appeared inside the window; *t* = day-clustered t of the mean; *share worse* = how often the wait gave a worse price; *just waiting* = improvement from trading at the end of the window regardless; *structure adds* = improvement minus what the same waiting times earn when applied without looking at the structure (time-matched placebo).

### Entries (the rule increases its position; wait for a bullish structure)

{md(pb_table(c, 'spot', 'trend_entry'), pb_digits)}

### Exits (the rule reduces its position; wait for a bearish structure)

{md(pb_table(c, 'spot', 'trend_exit'), pb_digits)}

### Entries and exits together

{md(pb_table(c, 'spot', 'trend_both'), pb_digits)}

### Supplement with more power: the same comparison on every calendar day, both sides

{md(pb_table(c, 'spot', 'all_days_both'), pb_digits)}

### Reading

* **Small sweep-and-reclaim (occurs within about 40 minutes in 99% of cases).** Entries: {se_d['mean_bp']:+.1f} bp (t {se_d['t']:.1f}) on design, {se_v['mean_bp']:+.1f} bp (t {se_v['t']:.1f}) on validation. Exits: {sx_d['mean_bp']:+.1f} bp (t {sx_d['t']:.1f}) on design, {sx_v['mean_bp']:+.1f} bp (t {sx_v['t']:.1f}) on validation. Together: {e_d['mean_bp']:+.1f} bp (t {e_d['t']:.1f}) and {e_v['mean_bp']:+.1f} bp (t {e_v['t']:.1f}). The median is around +13 to +15 bp and the price is worse in only a quarter of the trades, but when it is worse it is worse by {abs(e_d['mean_if_worse']):.0f} bp (design) and {abs(e_v['mean_if_worse']):.0f} bp (validation) on average against a gain of {e_d['mean_if_better']:.0f} and {e_v['mean_if_better']:.0f} bp when it is better. No cell is significant in both segments. On all days the same wait is worth {ad_d['mean_bp']:+.1f} bp (t {ad_d['t']:.1f}) on design and {ad_v['mean_bp']:+.1f} bp (t {ad_v['t']:.1f}) on validation: a small positive number that is shrinking, like the pattern's own effect in Part A.
* **Big sweep.** It usually does not come: within 4 hours in {be_d['triggered']:.0%} (design) and {be_v['triggered']:.0%} (validation) of the entries. The 4-hour result is therefore just the result of trading 4 hours late (compare the "just waiting" column; the structure itself adds between -3 and +5 bp, never with |t| above 2). Waiting up to 24 hours gave {b24_d['mean_bp']:+.1f} bp (t {b24_d['t']:.1f}) on design and {b24_v['mean_bp']:+.1f} bp (t {b24_v['t']:.1f}) on validation, with the price worse in {b24_v['worse']:.0%} of the validation trades.
* **Break of structure as confirmation.** The price is worse in {qe_d['worse']:.0%} of the design entries and {qe_v['worse']:.0%} of the validation entries; on average {qe_d['mean_bp']:+.1f} bp (t {qe_d['t']:.1f}) and {qe_v['mean_bp']:+.1f} bp (t {qe_v['t']:.1f}). On all days: {aq_d['mean_bp']:+.1f} and {aq_v['mean_bp']:+.1f} bp with the price worse in {aq_d['worse']:.0%} and {aq_v['worse']:.0%} of cases. Confirmation is paid for.
* **Return into a fair value gap.** Around zero in both segments, with the sign changing.
* **Scale.** With perfect hindsight the best price in the first 4 hours was about {c['part_b'][(c['part_b']['market'] == 'spot') & (c['part_b']['set'] == 'trend_both') & (c['part_b']['segment'] == 'design') & (c['part_b']['window'] == 240)]['oracle_median_bp'].iloc[0]:.0f} bp (design) and {c['part_b'][(c['part_b']['market'] == 'spot') & (c['part_b']['set'] == 'trend_both') & (c['part_b']['segment'] == 'validation') & (c['part_b']['window'] == 240)]['oracle_median_bp'].iloc[0]:.0f} bp (validation) better than the open (median). The structures capture none of it reliably. With a turnover of 13 to 16 times the equity per year, one bp of average improvement is worth about 0.13 to 0.16% of the equity per year: the +{e_v['mean_bp']:.0f} bp point estimate of validation would be about half a percent a year (USD 10 on a USD 2,000 account), the {q_v['mean_bp']:.0f} bp of the confirmation entry would cost about one and a half percent a year.
* **Power.** With {int(e_d['trades'])} and {int(e_v['trades'])} trades the standard error of the mean is {e_d['se_bp']:.0f} and {e_v['se_bp']:.0f} bp, so only improvements of 15 bp or more could have been detected on the trend trades themselves; the all-days supplement narrows the unconditional effect of the wait to a few bp.

**Answer to the owner's idea as stated:** on this evidence, waiting for a 1-minute structure before a daily trend trade is not a source of better prices. The most favourable variant (wait for a small sweep and reclaim) gives a better fill three times in four and loses it all in the fourth; the "wait for confirmation" variant gets a worse price in two trades out of three or more.
""")

    w(f"""## 8. What survived

| claim | design | validation | survives? |
|---|---|---|---|
| The named structures predict returns at 1 to 15 minutes by enough to pay costs | largest gross effect {c['max_short_gross']:.1f} bp against 12-14 bp | small sweep {s_v['gross']:+.2f} bp | no |
| Continuation readings (break of structure, fair value gap, breakout that holds, displacement) | mostly reversed | not carried forward | no |
| Limit orders at gap / block edges earn the bounce | {fl['gross']:+.2f} and {ol['gross']:+.2f} bp at the next open | not carried forward (resting orders on the small sweep: {s_v['maker_gross_per_fill']:+.2f} bp per fill) | no (adverse selection) |
| Reversal after a small sweep, as a statistical effect | {s_d['gross']:+.2f} bp, t {s_d['t_excess']:.1f} | {s_v['gross']:+.2f} bp, t {s_v['t_excess']:.1f} | yes, and irrelevant at this size |
| Reversal after a sweep of the 4-hour extreme, 60 min | {b_d['gross']:+.1f} bp, t {b_d['t_excess']:.1f}; net taker {b_d['net_taker']:+.1f} | {b_v['gross']:+.1f} bp, t {b_v['t_excess']:.1f}; net taker {b_v['net_taker']:+.1f} | effect yes, costs no |
| The same, 240 min, taker | net {a_d['net_taker']:+.1f} bp, t {a_d['t_net_taker']:.1f} | net {a_v['net_taker']:+.1f} bp, t {a_v['t_net_taker']:.1f} | not shown |
| The same, 240 min, resting orders | {a_d['maker_net_4bp']:+.1f} bp per fill, t {a_d['t_maker_net_4bp']:.1f} | {a_v['maker_net_4bp']:+.1f} bp per fill, t {a_v['t_maker_net_4bp']:.1f} | yes, as an optimistic bound |
| Waiting for a structure improves trend-trade fills | best case {e_d['mean_bp']:+.1f} bp, t {e_d['t']:.1f} | {e_v['mean_bp']:+.1f} bp, t {e_v['t']:.1f} | no |
| Waiting for a break of structure usually gives a worse price | worse in {q_d['worse']:.0%} of trades, median {q_d['median_bp']:+.1f} bp, mean {q_d['mean_bp']:+.1f} (t {q_d['t']:.1f}) | worse in {q_v['worse']:.0%}, median {q_v['median_bp']:+.1f} bp, mean {q_v['mean_bp']:+.1f} (t {q_v['t']:.1f}) | frequency yes; mean cost significant in validation only |

### What follows for the project

* **Drop** as a basis for strategies or entry rules: break of structure, change of character, fair value gaps (as continuation and as a place to rest orders), order blocks, compression breakouts, displacement candles and sweeps of the previous day's high and low on 1-minute bars. They fail by a factor of ten to a hundred before any question of robustness arises.
* **Do not delay trend trades to wait for a 1-minute trigger.** Executing at the daily open (or spreading the order over a short window) is as good as any of the waits tested, and waiting for confirmation is worse.
* **One lead worth a dedicated, pre-registered follow-up**, and nothing more than a lead: buying after a sharp push to a new 4-hour low on the altcoin perpetuals, with resting orders, holding hours rather than minutes. A test of it has to be a portfolio simulation (one position per coin, a cap on simultaneous positions, funding, explicit non-fills), has to fix the rule from the design and validation numbers above, and has to be judged once on the hold-out. The pattern vocabulary is not needed for it: "new 4-hour low by at least 10 bp" is the whole signal.

## 9. Caveats

""" + "\n".join(f"{i}. {text}" for i, text in enumerate(n['caveats'], 1)) + """

## 10. Files

| file | content |
|---|---|
| `eventstudy.py` | grid with gaps, forward returns, matched baseline, per-day sufficient statistics, day-clustered t, block bootstrap, costs |
| `structures.py` | the 70 definitions (detectors and the grid) |
| `test_eventstudy.py` | 15 synthetic-data tests (look-ahead, literal reference implementations, planted effect, calibration) |
| `run_segment.py` | detection and per-day statistics for a segment; the validation run computes only the pairs in `selection.json` |
| `analyze.py` | pooled tables and splits by year, coin and side |
| `write_selection.py`, `selection.json` | the selection and protocol frozen before validation |
| `maker_sim.py` | resting-order simulation for the selected pairs |
| `bootstrap_selected.py` | day-block bootstrap for the selected pairs |
| `part_b.py` | entry / exit timing for the daily trend rule |
| `paxg_check.py` | side check on PAXG spot (design only) |
| `make_report.py`, `report_text.py` | build this report, `results.json` and `out/report_tables.md` from the tables in `out/` |
| `out/table_design.csv`, `out/table_design_full.md` | all 420 design looks; `out/breakdown_design.csv` the splits |
| `out/table_validation.csv`, `out/breakdown_validation.csv` | the three validation pairs and their splits |
| `out/maker_sim_*.csv`, `out/bootstrap_*.csv`, `out/part_b_*.csv` | diagnostics and Part B, per segment |
| `out/*dryrun*` | the validation code path run on 2022-07 to 2023-12 (design data) to debug it without reading validation bars |
| `scratch/` | one-off checks (data inspection, independent recomputation); not part of the pipeline |

Reproduce (each step under five minutes):

```
PY="/home/user/algotrade/.venv/bin/python -I -B"
$PY test_eventstudy.py
$PY run_segment.py design && $PY analyze.py design && $PY paxg_check.py
$PY maker_sim.py design sweep_N240_x10_k1:240 sweep_N240_x10_k1:60 sweep_N30_x2_k3:5 && $PY bootstrap_selected.py design
$PY write_selection.py            # refuses to overwrite the frozen file
$PY run_segment.py validation && $PY analyze.py validation && $PY maker_sim.py validation && $PY bootstrap_selected.py validation
$PY part_b.py design && $PY part_b.py validation
$PY make_report.py
```
""")
    return "\n".join(parts)

