# One-minute price-action structures: do they predict anything, and do they improve trend entries?

Directory: `research/micro_1m/price_action/` · design segment 2020-01 to 2023-12 · validation segment 2024-01 to 2025-06 (looked at once) · hold-out from 2025-07 never loaded.

The owner's question: *"What about detecting very short-lived structures on 1-minute charts and working from there towards longer-term strategies?"*
This report takes the structures that discretionary "smart money / ICT" traders name on 1-minute charts, writes each one down as an exact rule, and measures two things on Binance data: (A) whether the structure predicts the return that follows by enough to pay trading costs, and (B) whether waiting for such a structure gives a better price for a longer-term (daily trend) trade than simply trading at the daily open.

## 1. The answer in plain words

Units: 1 bp = 0.01%. One round trip (in and out) on perpetual futures costs a taker 12 bp on BTC and ETH and 14 bp on the other coins. For scale, in 2023 the median 1-minute bar of BTC spanned 4 bp from high to low and that of SOL 12 bp: one trade costs about three BTC minutes.

1. **The patterns are statistically real and economically tiny.** 420 combinations of pattern, parameter and holding time were examined on six coins over four years. 181 of them show an effect that is clearly not chance (day-clustered |t| above 3). But no effect at holding times up to 15 minutes is larger than 5.1 bp before costs, and most are below 1 bp. Costs are 12-14 bp.
2. **The textbook reading is wrong more often than it is right.** Of the 181 clear effects, 110 point in the direction opposite to what the pattern is supposed to mean. Breakouts that "hold" tend to come back. Fair value gaps are followed by a small move against the gap. A break of structure continues for one minute (about 0.3 bp) and then reverses. A limit order placed at the edge of a fair value gap or an order block is, on average, 0.9 to 1.3 bp under water at the open of the next minute, and has not recovered an hour later.
3. **One effect is as large as the costs: the rebound after a sharp push to a new 4-hour low.** After price trades at least 10 bp through the 4-hour low and closes back above it within the same minute, the next four hours gained 26 bp more than comparable moments in 2020-2023 and 25 bp more in 2024-2025. Both sides of the pattern pooled (the pre-registered test): +19.4 bp on design (t 4.0) and +18.8 bp on validation (t 3.5). Three reservations:
   - It is not the "liquidity sweep" that matters. A push to a new 4-hour low was followed by the same rebound whether the bar closed back inside (the sweep) or price stayed below for three more closes (the opposite pattern). It is a plain dip-rebound, and only on the buy side; selling after a sweep of the highs earned nothing.
   - It comes and goes: strong in 2020-2021, about zero in 2023, back in 2024-2025, and it lives in XRP, SOL, DOGE and BNB far more than in BTC and ETH.
   - After a taker's costs the pooled figure is +6.0 bp (design) and +5.3 bp (validation), which cannot be told apart from zero (t 1.3 and 1.0). Only with resting (maker) orders does it stay clearly positive: +14.1 and +14.1 bp per filled order at maker fees, and that figure is optimistic.
4. **Waiting for a 1-minute structure did not improve the fills of the daily trend rule.** On the 292 trend trades of 2020-2023 and the 132 of 2024-mid 2025, waiting for the first small sweep-and-reclaim in the direction of the trade changed the price by +10.3 bp (t 1.8) and +3.0 bp (t 0.5) on average: not distinguishable from zero. The wait gives a better price about three times in four, and a much worse one the fourth time, when price runs away. Waiting for "confirmation" by a break of structure made the price worse in 66% (design) and 75% (validation) of the trades; the average was +0.8 bp in design (t 0.2) and -9.4 bp in validation (t -2.3).
5. **For an account of USD 500-5,000:** nothing here can be traded with market orders. The textbook 1-minute structures are not a basis for a strategy and not a way to time the entries of a daily strategy. The dip-rebound is a lead for a separate, carefully pre-registered maker-order study, not a result to trade on.

**Verdict: `edge_only_with_maker_fills`**, in the narrow sense above: one contrarian effect clears maker costs in both segments and fails the taker test; everything else fails both.

## 2. What was tested

### Data and segments

| market | coins | design bars per coin | gaps |
|---|---|---|---|
| USDT perpetuals (`um`), 1-minute | BTC, ETH, SOL, BNB, XRP, DOGE | 1,828,260 to 2,103,840 | none, except SOL and XRP: 7,200 missing minutes each (two stretches of 3 and 2 days in 2022 absent from the source files) |
| spot, 1-minute (cross-check) | BTC, ETH | 2,101,515 | 2,325 missing minutes in 15 outages |
| spot, 1-minute (side check, never pooled) | PAXG | 1,756,096 | 1,424 missing minutes |

Design = bars before 2024-01-01, used freely. Validation = 2024-01-01 to 2025-06-30, used once, for the three (definition, holding time) pairs frozen in `selection.json` and for the diagnostics listed there. The hold-out is locked by the loader and was never read.

### Conventions

* An event is stamped at the **close** of 1-minute bar t and uses bars up to t only. The trade is filled at the **open of bar t+1**. The forward return over h minutes runs from the open of t+1 to the open of t+1+h, for h = 1, 5, 15, 60, 240, 1440.
* Bars are put on a complete 1-minute grid; minutes the exchange did not deliver are empty rows. A return is dropped if any minute from t to t+1+h is missing, and an event is dropped if any of the 1,500 minutes before it is missing.
* Returns are signed in the direction of the textbook reading of the pattern, so a positive number means "the pattern worked as advertised" and a negative one means "the opposite happened".
* Costs per round trip: perpetual taker 12 bp (BTC, ETH) or 14 bp (others); perpetual maker bound 4 bp; spot taker 22 or 24 bp.
* Baseline: for every event, the average forward return of all bars of the same coin in the same calendar year, hour of day and volatility quintile (60-minute realised volatility). "Excess" = gross minus baseline. All t-statistics are for the excess.
* Inference: every t-statistic is clustered by UTC calendar day across all coins (events on one day are one observation). For the 240- and 1440-minute horizons the covariance between adjacent days is added, because those returns overlap across midnight. The naive t-statistic that treats events as independent is shown next to it; it is inflated by a factor of 1.2 at 1 minute rising to 2.3 at one day (median over all looks).

### The structures (70 definitions, 6 holding times each = 420 looks)

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

Looks, counted: 420 on the six perpetuals (design). The same 420 were run on BTC and ETH spot as a cross-check and on PAXG as a side check. Year, coin and side splits were read for about 15 of them. Validation: 3 pairs.

## 3. How the machinery was checked before it was used

`test_eventstudy.py` (15 tests, all passing; log in `out/test_log.txt`) proves on synthetic data that:

* **No look-ahead.** For each of the 70 definitions, the events stamped at or before bar t are identical when every bar after t is replaced by different prices and volumes, and when the series is cut at t (four cut points each). A deliberately cheating detector fails the same check.
* **The code does what the written rule says.** Every vectorised detector is compared, event by event, with a slow literal loop written from the definition (including ties created by tick rounding, asymmetric swings, the "first close back" logic and the first-touch search of zones).
* **Returns are aligned and gaps are respected.** From the open of t+1 to the open of t+1+h, never spanning a missing minute; checked on a hand-computed example.
* **A planted effect is recovered.** In a bar-by-bar simulation every displacement candle is followed by a drift of 2 bp per bar for 5 bars. The chain detector -> forward returns -> baseline -> clustered statistics returns 2 bp at 1 minute and 10 bp at 5, 15 and 60 minutes within 3.5 standard errors, and shifting the event by one bar in either direction breaks the result. With no planted effect nothing is found.
* **The t-statistic is honest.** With events in bursts and overlapping horizons the day-clustered t has a standard deviation of about 1 under the null, the naive one about 3 or more.
* **The baseline removes an hour-of-day drift**, and a resting order shows adverse selection on a driftless walk.

On real data: random events with random direction and random long-only events (`ctl_random`, `ctl_long`) give excess returns with |t| below 1.7 at every horizon; one selected definition was recomputed for BTC with plain pandas, independently of the module, and matched to all digits (7,130 events, 8.6593 bp at 60 minutes); a second was recomputed with an explicit loop on two months of data (6,445 identical events); after validation the same plain-pandas recomputation reproduced the validation figures of the selected pair for SOL exactly (3,025 events), and the Part B fills of 18 trend trades were reproduced by a brute-force loop.

## 4. Part A on the design segment (2020-2023, six perpetuals pooled)

### 4.1 Headline counts

* 420 looks. |t| > 3: **181**. |t| > 3.9 (the Bonferroni 5% line for this many looks): 147.
* Of the 181, **110 have the sign against the textbook reading** (t below -3).
* Mean after a taker round trip above zero: **9 looks**. Three of them have |t| > 3, all sweeps held for 240 minutes: `sweep_N60_x10_k1` (+1.0 bp, t 4.9), `sweep_N240_x10_k1` (+6.0 bp, t 4.0), `sweep_N240_x10_k3` (+0.3 bp, t 3.7). The other 6 are 1-day horizons with |t| below 1 (market drift and noise).
* Gross minus 4 bp (the optimistic maker bound) above zero: 45 looks, 15 of them with |t| > 3: all of them sweeps of the 30- to 240-minute extreme held for 15 to 240 minutes (13 of them with the 10 bp threshold).
* Largest gross effect at holding times up to 15 minutes: 5.07 bp.

### 4.2 By family

"t>3 for" counts looks where the textbook reading is supported, "t<-3 against" where the opposite happened.

| family (textbook reading) | defs | looks | t>3 for | t<-3 against | strongest look | gross bp | t | best net taker bp | where |
|---|---|---|---|---|---|---|---|---|---|
| 1a. Sweep of N-min extreme, close back inside (reversal) | 12 | 72 | 52 | 0 | sweep_N30_x2_k3 @5m | 0.91 | 12.24 | 6.02 | sweep_N240_x10_k1 @240m (t +4.0) |
| 1b. Same, swept level at least 10 min old (reversal) | 4 | 24 | 0 | 3 | sweepmat_N60_x2_k1 @1m | -0.66 | -6.38 | -11.33 | sweepmat_N60_x2_k3 @240m (t +2.0) |
| 1c. Break of N-min extreme that holds (continuation) | 12 | 72 | 0 | 55 | hold_N30_x2_k1 @5m | -1.11 | -11.87 | 14.19 | hold_N240_x10_k3 @1440m (t +0.6) |
| 2. Break of structure / change of character (continuation) | 9 | 54 | 9 | 16 | bos_s5_all @1m | 0.32 | 9.19 | -8.04 | bos_s5_with @1440m (t +1.7) |
| 3a. Fair value gap (continuation) | 3 | 18 | 0 | 11 | fvg_m0p5 @5m | -0.43 | -7.60 | -6.03 | fvg_m2 @1440m (t +0.5) |
| 3b. Return into the gap, next-open entry (bounce) | 6 | 36 | 2 | 4 | fvgfill_m2_W60 @240m | -3.29 | -3.56 | -8.30 | fvgfill_m1_W60 @1440m (t +1.4) |
| 3c. Return into the gap, resting limit at the edge (bounce) | 3 | 18 | 0 | 15 | fvglimit_m0p5_W60 @1m | -1.29 | -31.57 | -8.44 | fvglimit_m2_W60 @1440m (t +0.3) |
| 4. Compression then breakout (continuation) | 8 | 48 | 8 | 0 | squeeze_M30_q0p1_v0 @1m | 0.47 | 10.68 | -7.78 | squeeze_M30_q0p25_v2 @1440m (t +2.2) |
| 5a. Sweep of previous day's high / low (reversal) | 4 | 24 | 0 | 0 | pdsweep_x2_k1 @15m | -1.48 | -1.90 | -12.78 | pdsweep_x10_k3 @5m (t +1.0) |
| 5b. Acceptance beyond previous day's high / low (continuation) | 4 | 24 | 0 | 0 | pdhold_x2_k1 @1m | 0.33 | 2.76 | -3.92 | pdhold_x10_k1 @1440m (t +0.4) |
| 5c. Order block retest, next-open entry (bounce) | 2 | 12 | 0 | 1 | ob_W60 @15m | -0.49 | -4.37 | -12.35 | ob_W240 @1440m (t +0.7) |
| 5d. Order block retest, resting limit (bounce) | 1 | 6 | 0 | 4 | oblimit_W60 @1m | -0.86 | -17.33 | -13.36 | oblimit_W60 @1440m (t -0.1) |
| 5e. Displacement candle (continuation) | 2 | 12 | 0 | 1 | disp_m3 @240m | -1.26 | -3.03 | 2.99 | disp_m6 @1440m (t +0.9) |

### 4.3 Holding-time profiles

Mean gross return in bp in the direction of the textbook reading, with the day-clustered t of the excess over baseline in brackets. Events are pooled over the six coins; "days" is the number of distinct calendar days with at least one event. The last column is the best mean after a taker round trip over the six holding times.

| definition | events | days | 1m | 5m | 15m | 60m | 240m | 1440m | best net taker |
|---|---|---|---|---|---|---|---|---|---|
| sweep_N30_x2_k3 | 772,787 | 1,460 | +0.14 (+5.1) | +0.91 (+12.2) | +1.31 (+7.7) | +1.28 (+3.9) | +2.05 (+2.9) | -2.65 (-0.4) | -11.3 @240m |
| sweep_N60_x10_k1 | 84,474 | 1,458 | +0.39 (+2.8) | +2.55 (+7.6) | +4.01 (+5.1) | +7.13 (+4.8) | +14.37 (+4.9) | -0.55 (-0.3) | +1.0 @240m |
| sweep_N240_x10_k1 | 48,958 | 1,458 | +0.29 (+1.5) | +2.80 (+6.0) | +4.68 (+5.2) | +12.73 (+6.2) | +19.37 (+4.0) | -5.69 (-0.3) | +6.0 @240m |
| sweepmat_N60_x2_k1 | 60,886 | 1,460 | -0.66 (-6.4) | -0.29 (-1.2) | -0.42 (-0.6) | +0.03 (+0.5) | +0.78 (+0.6) | -0.51 (-0.6) | -12.5 @240m |
| sweepmat_N240_x2_k3 | 54,893 | 1,460 | -0.19 (-1.6) | +0.40 (+2.0) | +0.20 (+0.9) | +0.77 (+1.3) | +1.24 (+0.5) | -1.35 (-0.5) | -12.0 @240m |
| hold_N30_x2_k3 | 497,099 | 1,460 | -0.46 (-11.0) | -1.34 (-10.8) | -1.58 (-7.0) | -1.85 (-4.2) | -2.52 (-3.2) | +9.45 (+0.8) | -3.9 @1440m |
| hold_N240_x10_k3 | 90,076 | 1,460 | -0.98 (-6.4) | -3.28 (-7.0) | -5.07 (-6.4) | -9.99 (-4.8) | -15.16 (-4.0) | +27.58 (+0.6) | +14.2 @1440m |
| pdsweep_x2_k3 | 51,178 | 1,396 | +0.02 (+0.2) | +0.10 (+0.4) | -0.99 (-1.6) | -1.66 (-1.1) | -3.73 (-0.9) | -2.25 (-0.0) | -13.3 @5m |
| pdsweep_x10_k1 | 7,078 | 1,235 | -0.37 (-0.6) | -0.54 (-0.4) | -2.38 (-1.0) | -5.03 (-0.9) | -19.06 (-1.2) | -26.14 (-0.8) | -13.8 @1m |
| pdhold_x2_k1 | 54,328 | 1,399 | +0.33 (+2.8) | +0.55 (+1.9) | +1.13 (+2.1) | +0.91 (+0.5) | +1.99 (+0.4) | +0.64 (-0.3) | -11.4 @240m |
| bos_s3_all | 861,651 | 1,460 | +0.23 (+7.7) | -0.04 (-1.3) | -0.42 (-5.3) | -0.67 (-4.5) | -0.59 (-2.4) | +2.54 (+1.7) | -10.7 @1440m |
| bos_s5_all | 574,023 | 1,460 | +0.32 (+9.2) | +0.10 (+0.7) | -0.38 (-4.5) | -0.94 (-5.3) | -0.84 (-2.7) | +2.89 (+1.6) | -10.4 @1440m |
| bos_s5_with | 284,271 | 1,460 | +0.35 (+6.7) | -0.06 (-1.2) | -0.97 (-6.6) | -1.30 (-4.0) | -1.18 (-1.9) | +5.25 (+1.7) | -8.0 @1440m |
| bos_s5_against | 289,752 | 1,460 | +0.29 (+7.7) | +0.25 (+2.9) | +0.19 (+0.9) | -0.59 (-4.3) | -0.51 (-4.0) | +0.58 (-0.1) | -12.7 @1440m |
| bos_s10_all | 314,051 | 1,460 | +0.37 (+7.9) | +0.14 (+0.9) | -0.13 (-1.8) | -0.92 (-4.5) | -1.30 (-3.1) | +2.82 (+1.3) | -10.5 @1440m |
| fvg_m0p5 | 1,028,848 | 1,460 | -0.14 (-5.2) | -0.43 (-7.6) | -0.42 (-5.1) | -0.71 (-4.8) | -0.58 (-2.5) | +4.39 (+1.4) | -8.8 @1440m |
| fvg_m2 | 67,203 | 1,460 | -0.13 (-0.9) | -1.17 (-3.5) | -1.28 (-2.9) | -2.68 (-3.3) | -4.60 (-4.0) | +7.15 (+0.5) | -6.0 @1440m |
| fvgfill_m0p5_W60 | 802,013 | 1,460 | +0.08 (+3.1) | +0.10 (+1.6) | +0.04 (-0.3) | -0.08 (-1.4) | +0.22 (-0.3) | +3.52 (+1.7) | -9.7 @1440m |
| fvgfill_m2_W240 | 60,037 | 1,460 | +0.00 (-0.0) | -0.17 (-1.1) | -0.66 (-2.3) | -1.84 (-3.2) | -2.96 (-3.4) | +2.13 (-0.1) | -11.0 @1440m |
| fvglimit_m0p5_W60 | 924,696 | 1,460 | -1.29 (-31.6) | -1.29 (-23.6) | -1.28 (-15.7) | -1.31 (-8.8) | -1.02 (-4.1) | +1.82 (+0.6) | -11.4 @1440m |
| squeeze_M30_q0p1_v0 | 151,664 | 1,460 | +0.47 (+10.7) | +0.23 (+1.8) | -0.19 (-1.6) | -0.67 (-2.2) | +0.66 (+0.6) | +4.42 (+1.7) | -8.8 @1440m |
| squeeze_M30_q0p1_v2 | 70,307 | 1,460 | +0.74 (+10.3) | +0.50 (+2.9) | +0.15 (+0.2) | -0.41 (-1.4) | +1.18 (+1.0) | +4.90 (+1.7) | -8.3 @1440m |
| squeeze_M120_q0p1_v2 | 36,005 | 1,456 | +0.79 (+6.1) | +0.70 (+2.2) | +0.63 (+1.0) | -1.42 (-2.0) | -0.74 (-0.6) | -1.52 (-0.4) | -12.4 @1m |
| ob_W60 | 389,903 | 1,460 | +0.02 (+0.6) | +0.01 (+0.0) | -0.49 (-4.4) | -0.33 (-1.7) | -0.18 (-0.7) | +0.84 (+0.6) | -12.4 @1440m |
| oblimit_W60 | 440,131 | 1,460 | -0.86 (-17.3) | -0.87 (-10.9) | -1.37 (-11.3) | -1.12 (-5.5) | -0.99 (-2.9) | -0.10 (-0.1) | -13.4 @1440m |
| disp_m3 | 413,309 | 1,460 | +0.08 (+1.1) | +0.20 (+1.4) | -0.03 (-0.9) | -0.75 (-2.8) | -1.26 (-3.0) | +8.09 (+1.0) | -5.2 @1440m |
| disp_m6 | 37,682 | 1,460 | +0.39 (+1.0) | -0.57 (-1.1) | -0.67 (-1.1) | -2.92 (-2.4) | -5.81 (-2.8) | +16.31 (+0.9) | +3.0 @1440m |

The complete table (all 420 rows with events, days, gross, baseline, excess, clustered and naive t, hit rate, net taker, maker bound) is in `out/table_design_full.md` and `out/table_design.csv`; the splits by year, coin and side for every row are in `out/breakdown_design.csv`.

### 4.4 What the table says, family by family

* **Sweeps of the N-minute extreme (reversal reading).** Supported everywhere, tiny at short holds: `sweep_N30_x2_k3` earns +0.91 bp over 5 minutes (t 12.2, the strongest t of all 420 looks). The effect grows with the size of the level and the holding time: `sweep_N240_x10_k1` earns +12.7 bp over 60 minutes (t 6.2) and +19.4 bp over 240 minutes (t 4.0).
* **Sweeps of an established level** (untouched for 10 minutes, which is what traders mean by "resting liquidity"): the reversal disappears. At 1 minute the sign is even the other way (-0.66 bp, t -6.4); at every longer hold |t| is below 2.1. The reversal in the plain sweep therefore comes from levels that were set minutes ago by the same fast move, i.e. from short-term reversal after a thrust, not from old highs and lows being "raided".
* **Acceptance beyond the level (continuation reading).** Rejected in 55 of 72 looks: price that breaks out and holds comes back as well (-1.34 bp at 5 minutes, t -10.8; -10.0 bp at 60 minutes for the 4-hour level, t -4.8).
* **Previous day's high and low.** Nothing: none of the 48 looks reaches |t| = 3, for either reading.
* **Break of structure / change of character.** A continuation of 0.32 bp in the first minute (t 9.2), then the opposite. Breaks in the direction of the previous break (break of structure proper) have reversed by 15 minutes (-0.97 bp, t -6.6); breaks against it (change of character) are flat for 15 minutes and have reversed by 60 (-0.59 bp, t -4.3). Neither continues.
* **Fair value gaps.** The gap is followed by a move against it (-0.43 bp at 5 minutes, t -7.6); large gaps more so (-4.6 bp at 240 minutes, t -4.0). Entering on the return into the gap at the next open earns nothing (|gross| below 0.2 bp up to 60 minutes for the standard gap). As a resting limit order at the gap edge it loses 1.29 bp by the next open (t -32) and is still 1.31 bp down an hour later: the order is filled exactly when price is on its way through.
* **Compression then breakout.** +0.47 bp in the first minute (t 10.7), +0.74 bp with the volume filter; nothing from 15 minutes on. Even this has faded: by year, the volume-filtered version earned 2020: +1.20, 2021: +1.32, 2022: +0.45, 2023: +0.03 bp.
* **Order blocks.** Next-open entry: about zero (the largest figure is -0.49 bp at 15 minutes, t -4.4, against the bounce). Resting limit at the block edge: -0.86 bp at the next open (t -17).
* **Displacement candles.** No continuation at any holding time; over 240 minutes a small move against the candle (-1.26 bp, t -3.0; for the largest candles -5.8 bp, t -2.8).

### 4.5 The sweep pattern itself carries no information

The same breach of the 4-hour (or 30-minute) extreme, split by what happened next, and expressed as what a **contrarian** earned in excess of the baseline (a buyer after a new low, a seller after a new high):

| event | what happened after the breach | h | new-low events | buyer excess bp | t | new-high events | seller excess bp | t  |
|---|---|---|---|---|---|---|---|---|
| sweep_N240_x10_k1 | closed back inside | 15 | 26,877 | 9.35 | 6.72 | 22,080 | -0.33 | -0.22 |
| sweep_N240_x10_k1 | closed back inside | 60 | 26,876 | 21.18 | 7.61 | 22,079 | 3.28 | 0.97 |
| sweep_N240_x10_k1 | closed back inside | 240 | 26,872 | 26.33 | 4.18 | 22,075 | 9.32 | 1.17 |
| sweep_N240_x10_k3 | closed back inside | 15 | 50,744 | 7.04 | 6.60 | 46,621 | 0.55 | 0.59 |
| sweep_N240_x10_k3 | closed back inside | 60 | 50,743 | 15.40 | 6.66 | 46,620 | 2.69 | 1.17 |
| sweep_N240_x10_k3 | closed back inside | 240 | 50,730 | 19.18 | 3.64 | 46,611 | 7.33 | 1.28 |
| hold_N240_x10_k1 | closed beyond (held) | 15 | 67,403 | 10.10 | 7.71 | 76,663 | -0.59 | -0.70 |
| hold_N240_x10_k1 | closed beyond (held) | 60 | 67,403 | 19.79 | 7.19 | 76,663 | 0.09 | 0.03 |
| hold_N240_x10_k1 | closed beyond (held) | 240 | 67,376 | 26.06 | 4.82 | 76,648 | 4.43 | 0.82 |
| hold_N240_x10_k3 | closed beyond (held) | 15 | 40,708 | 11.21 | 7.37 | 49,367 | 0.75 | 0.81 |
| hold_N240_x10_k3 | closed beyond (held) | 60 | 40,708 | 22.15 | 6.88 | 49,367 | 1.46 | 0.49 |
| hold_N240_x10_k3 | closed beyond (held) | 240 | 40,689 | 29.86 | 5.18 | 49,357 | 5.56 | 0.97 |
| sweep_N30_x2_k3 | closed back inside | 15 | 383,205 | 2.02 | 6.30 | 389,571 | 0.97 | 3.85 |
| sweep_N30_x2_k3 | closed back inside | 60 | 383,196 | 2.41 | 2.83 | 389,565 | 0.95 | 1.25 |
| sweep_N30_x2_k3 | closed back inside | 240 | 383,145 | 3.67 | 1.38 | 389,508 | 1.30 | 0.52 |
| hold_N30_x2_k3 | closed beyond (held) | 15 | 232,402 | 3.06 | 6.74 | 264,681 | 0.77 | 2.52 |
| hold_N30_x2_k3 | closed beyond (held) | 60 | 232,401 | 4.71 | 4.82 | 264,681 | 0.49 | 0.55 |
| hold_N30_x2_k3 | closed beyond (held) | 240 | 232,336 | 6.98 | 2.72 | 264,625 | 0.47 | 0.19 |

After a push to a new 4-hour low a buyer earned 21.2 bp over the next hour when the bar closed back inside, 19.8 bp when it closed below, and 22.2 bp when price stayed below for three closes. After a push to a new high a seller earned nothing in any of the three cases. The information is "price has just fallen hard", not the shape of the candle.

### 4.6 Naive against clustered t-statistics

Median ratio of the naive t (events treated as independent) to the day-clustered t, by holding time: 1 min: 1.19, 5 min: 1.37, 15 min: 1.52, 60 min: 1.86, 240 min: 2.05, 1440 min: 2.32. A naive t of 4 at the one-day horizon is a clustered t of about 1.7.

### 4.7 Cross-checks

* **Spot, BTC and ETH.** The 420 t-statistics on BTC+ETH spot correlate 0.92 with those on BTC+ETH perpetuals; where the perpetual |t| exceeds 3 the sign agrees in 98% of the looks. With spot costs of 22 bp, 2 looks have a positive mean after costs, at the one-day horizon with |t| below 1.5 (noise).
* **Tokenised gold (PAXG spot), side check.** Its price moves in steps of about 5 bp and about a third of its design-segment minutes have no trade. 379 of 420 looks have |t| > 3, most of them "reversals" of 0.5 to 4 bp, which is what bid-ask bounce on a coarse tick looks like in bars. The largest are sweep reversals of 14 to 21 bp over 4 to 24 hours; none has a positive mean after the 24 bp spot round trip (best: -3.4 bp). One-minute structure trading in PAXG cannot be assessed from bars and is not attractive on its face.

## 5. Selection, frozen before validation

`selection.json` was written at 2026-11-12T14:22:50Z from the design tables, with the criteria for success, the diagnostics and the Part B protocol. Three (definition, holding time) pairs:

| # | pair | why |
|---|---|---|
| 1 | `sweep_N240_x10_k1`, 240 minutes | the only look with |t| >= 3.9 and a positive mean after taker costs (+6.0 bp) |
| 2 | `sweep_N240_x10_k1`, 60 minutes | the same structure where the evidence is strongest (t 6.2; positive in every design year and coin) |
| 3 | `sweep_N30_x2_k3`, 5 minutes | the most significant of all looks (t 12.2); carried forward to see whether the most robust statistical effect persists, not because 1 bp can be traded |

Known weaknesses were written into the file at selection time: the effect of pairs 1 and 2 sits on the long side and in 2020-2021; 2023 is about zero. A later check on design data only (made while the validation code was being debugged on 2022-07 to 2023-12) showed that in those 18 months it was already down to +2.7 bp at 60 minutes (t 1.3) and +8.8 bp at 240 minutes (t 1.5).

Pre-registered criteria: *replicates* = same sign and clustered t of the excess at least 2.4; *pays taker costs* = mean after the taker round trip above zero with t at least 2 in both segments; *edge only with maker fills* = the taker test fails but the resting-order simulation at 4 bp is above zero with t at least 2 in both segments.

## 6. Part A on the validation segment (2024-01 to 2025-06), looked at once

### 6.1 The three pairs

| definition | h | segment | events | days | gross | base | excess | t_excess | t_naive | boot_ci_lo | boot_ci_hi | cost_taker | net_taker | t_net_taker | maker_bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sweep_N240_x10_k1 | 240 | design | 48,947 | 1,458 | 19.37 | 0.71 | 18.66 | 3.99 | 10.85 | 9.79 | 27.28 | 13.35 | 6.02 | 1.26 | 15.37 |
| sweep_N240_x10_k1 | 240 | validation | 13,248 | 546 | 18.82 | 0.11 | 18.70 | 3.46 | 8.18 | 8.35 | 28.98 | 13.50 | 5.32 | 0.98 | 14.82 |
| sweep_N240_x10_k1 | 60 | design | 48,955 | 1,458 | 12.73 | -0.37 | 13.11 | 6.17 | 12.82 | 8.92 | 17.02 | 13.35 | -0.62 | -0.29 | 8.73 |
| sweep_N240_x10_k1 | 60 | validation | 13,249 | 546 | 9.66 | -0.47 | 10.13 | 3.82 | 7.30 | 5.02 | 15.31 | 13.50 | -3.84 | -1.45 | 5.66 |
| sweep_N30_x2_k3 | 5 | design | 772,787 | 1,460 | 0.91 | -0.07 | 0.97 | 12.24 | 19.36 | 0.82 | 1.13 | 13.35 | -12.45 | -156.62 | -3.09 |
| sweep_N30_x2_k3 | 5 | validation | 307,372 | 547 | 0.35 | -0.06 | 0.41 | 4.26 | 7.25 | 0.21 | 0.59 | 13.40 | -13.06 | -136.67 | -3.65 |

`boot_ci` = 95% interval of the excess from a day-block bootstrap (4,000 resamples). `t_net_taker` = (gross - cost) divided by the clustered standard error.

* **All three replicate** by the pre-registered criterion (t 3.46, 3.82, 4.26). The big-sweep effect has the same size as on design (18.8 against 19.4 bp at 240 minutes); the small-sweep effect has more than halved (0.35 against 0.91 bp).
* **None pays taker costs** by the pre-registered criterion. Pair 1 has a positive point estimate in both segments (+6.0 and +5.3 bp) with t 1.3 and 1.0; both segments together: +5.7 bp, t 1.6. Pair 2 is negative (-0.6, -3.8); pair 3 loses 13 bp per trade.

### 6.2 By year, by side, by coin

| definition | h | year | segment | events | days | gross | excess | t_excess | net_taker |
|---|---|---|---|---|---|---|---|---|---|
| sweep_N240_x10_k1 | 240 | 2020 | design | 8,655 | 365 | 18.15 | 15.61 | 1.41 | 4.97 |
| sweep_N240_x10_k1 | 240 | 2021 | design | 18,815 | 365 | 30.82 | 30.22 | 3.28 | 17.41 |
| sweep_N240_x10_k1 | 240 | 2022 | design | 12,754 | 365 | 14.73 | 15.39 | 1.96 | 1.38 |
| sweep_N240_x10_k1 | 240 | 2023 | design | 8,723 | 363 | 2.68 | 1.53 | 0.23 | -10.74 |
| sweep_N240_x10_k1 | 240 | 2024 | validation | 9,067 | 365 | 20.67 | 19.77 | 3.01 | 7.15 |
| sweep_N240_x10_k1 | 240 | 2025 | validation | 4,181 | 181 | 14.79 | 16.39 | 1.72 | 1.34 |
| sweep_N240_x10_k1 | 60 | 2020 | design | 8,655 | 365 | 16.72 | 17.04 | 2.88 | 3.55 |
| sweep_N240_x10_k1 | 60 | 2021 | design | 18,815 | 365 | 21.27 | 21.47 | 5.34 | 7.87 |
| sweep_N240_x10_k1 | 60 | 2022 | design | 12,758 | 365 | 5.36 | 5.97 | 2.03 | -7.99 |
| sweep_N240_x10_k1 | 60 | 2023 | design | 8,727 | 363 | 1.13 | 1.60 | 0.50 | -12.29 |
| sweep_N240_x10_k1 | 60 | 2024 | validation | 9,067 | 365 | 9.51 | 9.81 | 3.23 | -4.01 |
| sweep_N240_x10_k1 | 60 | 2025 | validation | 4,182 | 181 | 9.98 | 10.82 | 2.07 | -3.47 |
| sweep_N30_x2_k3 | 5 | 2020 | design | 146,631 | 365 | 0.59 | 0.65 | 4.21 | -12.57 |
| sweep_N30_x2_k3 | 5 | 2021 | design | 249,426 | 365 | 1.76 | 1.85 | 10.40 | -11.61 |
| sweep_N30_x2_k3 | 5 | 2022 | design | 208,692 | 365 | 0.58 | 0.63 | 4.62 | -12.81 |
| sweep_N30_x2_k3 | 5 | 2023 | design | 168,038 | 365 | 0.32 | 0.38 | 3.67 | -13.13 |
| sweep_N30_x2_k3 | 5 | 2024 | validation | 203,185 | 366 | 0.32 | 0.37 | 2.98 | -13.08 |
| sweep_N30_x2_k3 | 5 | 2025 | validation | 104,187 | 181 | 0.40 | 0.48 | 3.29 | -13.02 |

| definition | h | side | segment | events | gross | base | excess | t_excess | net_taker | excess minus cost | t (excess - cost) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sweep_N240_x10_k1 | 240 | long | design | 26,872 | 38.58 | 12.25 | 26.33 | 4.18 | 25.23 | 12.99 | 2.06 |
| sweep_N240_x10_k1 | 240 | long | validation | 7,576 | 34.75 | 9.41 | 25.35 | 2.84 | 21.26 | 11.86 | 1.33 |
| sweep_N240_x10_k1 | 240 | short | design | 22,075 | -4.01 | -13.33 | 9.32 | 1.17 | -17.37 | -4.04 | -0.51 |
| sweep_N240_x10_k1 | 240 | short | validation | 5,672 | -2.47 | -12.30 | 9.83 | 1.29 | -15.98 | -3.68 | -0.48 |
| sweep_N240_x10_k1 | 60 | long | design | 26,876 | 23.99 | 2.81 | 21.18 | 7.61 | 10.65 | 7.84 | 2.82 |
| sweep_N240_x10_k1 | 60 | long | validation | 7,577 | 15.28 | 1.44 | 13.84 | 3.90 | 1.79 | 0.35 | 0.10 |
| sweep_N240_x10_k1 | 60 | short | design | 22,079 | -0.98 | -4.25 | 3.28 | 0.97 | -14.34 | -10.09 | -3.00 |
| sweep_N240_x10_k1 | 60 | short | validation | 5,672 | 2.14 | -3.04 | 5.18 | 1.25 | -11.37 | -8.33 | -2.02 |
| sweep_N30_x2_k3 | 5 | long | design | 383,209 | 1.27 | 0.10 | 1.17 | 9.60 | -12.07 | -12.16 | -99.82 |
| sweep_N30_x2_k3 | 5 | long | validation | 155,165 | 0.62 | 0.03 | 0.59 | 3.84 | -12.79 | -12.81 | -83.47 |
| sweep_N30_x2_k3 | 5 | short | design | 389,578 | 0.55 | -0.23 | 0.78 | 6.78 | -12.82 | -12.59 | -109.36 |
| sweep_N30_x2_k3 | 5 | short | validation | 152,207 | 0.07 | -0.15 | 0.22 | 1.81 | -13.34 | -13.18 | -109.67 |

"excess minus cost" is the excess over the baseline after a taker round trip, i.e. what remains once the general rise of the market in comparable moments is removed. On the long side of pair 1 it is +13.0 bp on design (t 2.1) and +11.9 bp on validation (t 1.3). This split was seen on design before validation but was not pre-registered as a test of its own; it is a hypothesis for the hold-out, not a finding.

| definition | h | coin | segment | events | gross | excess | t_excess | net_taker |
|---|---|---|---|---|---|---|---|---|
| sweep_N240_x10_k1 | 240 | BNB | design | 6,586 | 20.70 | 18.24 | 2.63 | 6.70 |
| sweep_N240_x10_k1 | 240 | BTC | design | 7,129 | 9.43 | 8.36 | 1.39 | -2.57 |
| sweep_N240_x10_k1 | 240 | DOGE | design | 8,311 | 22.22 | 23.31 | 1.53 | 8.22 |
| sweep_N240_x10_k1 | 240 | ETH | design | 8,688 | 15.30 | 14.13 | 2.84 | 3.30 |
| sweep_N240_x10_k1 | 240 | SOL | design | 10,093 | 17.34 | 17.43 | 2.44 | 3.34 |
| sweep_N240_x10_k1 | 240 | XRP | design | 8,140 | 30.96 | 29.64 | 4.22 | 16.96 |
| sweep_N240_x10_k1 | 240 | BNB | validation | 1,067 | 19.11 | 15.56 | 1.75 | 5.11 |
| sweep_N240_x10_k1 | 240 | BTC | validation | 1,204 | 10.52 | 9.04 | 1.53 | -1.48 |
| sweep_N240_x10_k1 | 240 | DOGE | validation | 3,420 | 19.39 | 20.22 | 2.36 | 5.39 |
| sweep_N240_x10_k1 | 240 | ETH | validation | 2,125 | 0.31 | 0.29 | 0.04 | -11.69 |
| sweep_N240_x10_k1 | 240 | SOL | validation | 3,025 | 22.75 | 24.24 | 3.20 | 8.75 |
| sweep_N240_x10_k1 | 240 | XRP | validation | 2,407 | 33.41 | 32.08 | 3.40 | 19.41 |
| sweep_N240_x10_k1 | 60 | BNB | design | 6,586 | 15.05 | 15.05 | 4.48 | 1.05 |
| sweep_N240_x10_k1 | 60 | BTC | design | 7,130 | 8.66 | 8.74 | 2.85 | -3.34 |
| sweep_N240_x10_k1 | 60 | DOGE | design | 8,312 | 12.81 | 13.75 | 1.79 | -1.19 |
| sweep_N240_x10_k1 | 60 | ETH | design | 8,688 | 7.49 | 7.45 | 3.29 | -4.51 |
| sweep_N240_x10_k1 | 60 | SOL | design | 10,096 | 13.18 | 13.85 | 3.94 | -0.82 |
| sweep_N240_x10_k1 | 60 | XRP | design | 8,143 | 19.38 | 19.80 | 4.99 | 5.38 |
| sweep_N240_x10_k1 | 60 | BNB | validation | 1,067 | 10.45 | 10.78 | 2.83 | -3.55 |
| sweep_N240_x10_k1 | 60 | BTC | validation | 1,205 | 8.22 | 7.91 | 2.13 | -3.78 |
| sweep_N240_x10_k1 | 60 | DOGE | validation | 3,420 | 11.07 | 11.64 | 2.53 | -2.93 |
| sweep_N240_x10_k1 | 60 | ETH | validation | 2,125 | 3.07 | 3.50 | 1.18 | -8.93 |
| sweep_N240_x10_k1 | 60 | SOL | validation | 3,025 | 10.31 | 11.41 | 3.11 | -3.69 |
| sweep_N240_x10_k1 | 60 | XRP | validation | 2,407 | 13.00 | 13.04 | 2.40 | -1.00 |
| sweep_N30_x2_k3 | 5 | BNB | design | 129,184 | 0.46 | 0.52 | 4.46 | -13.54 |
| sweep_N30_x2_k3 | 5 | BTC | design | 117,116 | 1.22 | 1.25 | 13.42 | -10.78 |
| sweep_N30_x2_k3 | 5 | DOGE | design | 122,221 | 1.07 | 1.15 | 5.69 | -12.93 |
| sweep_N30_x2_k3 | 5 | ETH | design | 133,263 | 1.07 | 1.12 | 11.16 | -10.93 |
| sweep_N30_x2_k3 | 5 | SOL | design | 133,992 | 0.43 | 0.52 | 3.00 | -13.57 |
| sweep_N30_x2_k3 | 5 | XRP | design | 137,011 | 1.21 | 1.30 | 9.93 | -12.79 |
| sweep_N30_x2_k3 | 5 | BNB | validation | 44,693 | 0.34 | 0.39 | 2.95 | -13.66 |
| sweep_N30_x2_k3 | 5 | BTC | validation | 41,841 | 0.27 | 0.30 | 2.79 | -11.73 |
| sweep_N30_x2_k3 | 5 | DOGE | validation | 60,414 | 0.26 | 0.33 | 1.86 | -13.74 |
| sweep_N30_x2_k3 | 5 | ETH | validation | 49,928 | 0.29 | 0.35 | 2.95 | -11.71 |
| sweep_N30_x2_k3 | 5 | SOL | validation | 61,175 | 0.59 | 0.66 | 4.45 | -13.41 |
| sweep_N30_x2_k3 | 5 | XRP | validation | 49,321 | 0.27 | 0.35 | 1.99 | -13.73 |

BTC and ETH only, perpetuals against spot:

| definition | h | segment | perp events | perp gross | perp t | perp net taker | spot events | spot gross | spot t | spot net taker |
|---|---|---|---|---|---|---|---|---|---|---|
| sweep_N240_x10_k1 | 240 | design | 15,817 | 12.66 | 2.31 | 0.66 | 13,425 | 15.49 | 2.78 | -6.51 |
| sweep_N240_x10_k1 | 240 | validation | 3,329 | 4.01 | 0.59 | -7.99 | 2,759 | 4.28 | 0.58 | -17.72 |
| sweep_N240_x10_k1 | 60 | design | 15,818 | 8.02 | 3.42 | -3.98 | 13,443 | 9.92 | 3.70 | -12.08 |
| sweep_N240_x10_k1 | 60 | validation | 3,330 | 4.93 | 1.90 | -7.07 | 2,759 | 4.35 | 1.51 | -17.65 |
| sweep_N30_x2_k3 | 5 | design | 250,379 | 1.14 | 14.45 | -10.86 | 241,590 | 1.13 | 13.85 | -20.87 |
| sweep_N30_x2_k3 | 5 | validation | 91,769 | 0.28 | 3.44 | -11.72 | 85,990 | 0.27 | 3.19 | -21.73 |

For the two coins the trend rule trades, the big-sweep effect did not replicate at a useful size in validation (+4.0 bp at 240 minutes on perpetuals, +4.3 bp on spot, t below 1). The pooled result is carried by the four altcoins.

### 6.3 What a resting order really gets

The usual "maker" figure (gross minus 4 bp) pretends that a passive order is filled at the taker's price. The simulation in `maker_sim.py` places a limit order at the close of the event bar, leaves it for 5 minutes, and counts a fill only if a later bar trades strictly through the price.

| definition | h | segment | fill_rate | filled_taker_gross | missed_taker_gross | mark_next_open | maker_gross_per_fill | maker_net_4bp | t_maker_net_4bp | maker_in_taker_out | t_maker_in_taker_out |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sweep_N240_x10_k1 | 240 | design | 0.972 | 18.33 | 55.25 | -1.38 | 18.07 | 14.07 | 2.84 | 9.39 | 1.89 |
| sweep_N240_x10_k1 | 240 | validation | 0.973 | 17.90 | 52.42 | -0.73 | 18.13 | 14.13 | 2.57 | 9.38 | 1.71 |
| sweep_N240_x10_k1 | 60 | design | 0.972 | 11.41 | 58.04 | -1.38 | 11.40 | 7.40 | 3.30 | 2.73 | 1.22 |
| sweep_N240_x10_k1 | 60 | validation | 0.973 | 8.81 | 40.57 | -0.73 | 8.87 | 4.87 | 1.81 | 0.12 | 0.05 |
| sweep_N30_x2_k3 | 5 | design | 0.956 | -0.26 | 26.12 | -1.26 | -0.29 | -4.29 | -44.66 | -8.97 | -93.29 |
| sweep_N30_x2_k3 | 5 | validation | 0.959 | -0.60 | 22.46 | -1.02 | -0.52 | -4.52 | -41.60 | -9.22 | -84.87 |

* `missed_taker_gross` is what a taker earned on the events where the resting order was never filled: the orders a maker misses are the best trades (+26 bp for the small sweep, +55 bp for the big one).
* For the small sweep the whole effect lives in the 4% of events a maker does not get. On the fills the gross return is -0.29 bp (design) and -0.52 bp (validation): the "maker bound" of -3.1 bp is really -4.3 bp. This is adverse selection, measured.
* For the big sweep at 240 minutes a maker keeps +14.1 bp (design, t 2.8) and +14.1 bp (validation, t 2.6) per fill if the exit is also passive and free of slippage, which is the optimistic bound, and +9.4 / +9.4 bp (t 1.9 / 1.7; both segments together t 2.5) with a maker entry and a taker exit. By the pre-registered rule this pair is an "edge only with maker fills". At 60 minutes the 4 bp figure is +7.4 (t 3.3) on design and +4.9 (t 1.8) on validation, which misses the criterion.

## 7. Part B: does waiting for a structure improve the fill of a daily trend trade?

The rule is `blend_weights()` in `research/baseline/trend_baseline.py`: the average of three long / flat filters (close above its 50-, 100- and 200-day average) on BTC and ETH spot. A change decided at the daily close is executed at the next 00:00 UTC. In 2020-2023 the target weight changed 292 times (151 increases, 141 decreases), in 2024 to mid-2025 132 times (63 and 69); a typical trade moves 18% of the equity, which makes about 13 times the equity per year in design and 16 times in validation, one way. (A few trades are dropped below where the 1-minute data have a gap.)

For every such trade the price at the 00:00 open is compared with the price obtained by waiting, for up to 4 hours and for up to 24 hours, for the first occurrence of a structure in the direction of the trade (bullish for a buy, bearish for a sell), buying or selling at the open of the next minute, or at the end of the window if the structure never appears. Improvement is in bp of the 00:00 price, positive = better price. Structures: the two selected in Part A, and two companions that were not selected but are the entry triggers discretionary traders actually use (a break of structure as "confirmation", and the return into a fair value gap). Spot 1-minute bars; the perpetual version is in `out/part_b_*.csv` and tells the same story.

Columns: *occurred* = share of trades in which the structure appeared inside the window; *t* = day-clustered t of the mean; *share worse* = how often the wait gave a worse price; *just waiting* = improvement from trading at the end of the window regardless; *structure adds* = improvement minus what the same waiting times earn when applied without looking at the structure (time-matched placebo).

### Entries (the rule increases its position; wait for a bullish structure)

| structure | wait up to (min) | segment | trades | days | occurred | mean wait (min) | mean bp | median bp | t | share worse | mean if better | mean if worse | just waiting, bp | structure adds, bp | t (adds) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| big sweep (N=240, 10 bp, k=1) | 240 | design | 147 | 130 | 0.23 | 207 | 15.8 | 35.0 | 1.31 | 0.30 | 79.2 | -132.5 | 19.7 | -1.1 | -0.26 |
| big sweep (N=240, 10 bp, k=1) | 240 | validation | 63 | 60 | 0.06 | 233 | 7.7 | 16.6 | 0.73 | 0.38 | 48.2 | -58.1 | 10.9 | -3.3 | -0.82 |
| big sweep (N=240, 10 bp, k=1) | 1,440 | design | 147 | 130 | 0.75 | 768 | 12.7 | 66.4 | 0.71 | 0.34 | 126.8 | -208.5 | -31.3 | 15.6 | 1.25 |
| big sweep (N=240, 10 bp, k=1) | 1,440 | validation | 63 | 60 | 0.51 | 1,029 | -15.4 | 7.0 | -0.59 | 0.46 | 121.8 | -176.3 | -28.4 | -3.2 | -0.21 |
| break of structure (5/5 swings) | 240 | design | 147 | 130 | 1.00 | 31 | -1.5 | -11.0 | -0.27 | 0.69 | 59.5 | -29.3 | 19.7 | -2.9 | -0.65 |
| break of structure (5/5 swings) | 240 | validation | 63 | 60 | 1.00 | 23 | -9.5 | -9.4 | -2.84 | 0.78 | 22.9 | -18.7 | 10.9 | -9.2 | -3.64 |
| return into fair value gap | 240 | design | 147 | 130 | 1.00 | 29 | 0.9 | 2.5 | 0.18 | 0.48 | 36.9 | -37.6 | 19.7 | -0.0 | -0.01 |
| return into fair value gap | 240 | validation | 63 | 60 | 1.00 | 17 | -1.2 | 1.9 | -0.26 | 0.44 | 16.9 | -23.7 | 10.9 | -0.1 | -0.03 |
| small sweep (N=30, 2 bp, k=3) | 240 | design | 147 | 130 | 0.99 | 41 | 2.6 | 14.4 | 0.34 | 0.26 | 31.3 | -79.9 | 19.7 | -0.9 | -0.16 |
| small sweep (N=30, 2 bp, k=3) | 240 | validation | 63 | 60 | 1.00 | 43 | 8.4 | 12.7 | 2.04 | 0.24 | 22.7 | -37.1 | 10.9 | 8.3 | 2.59 |

### Exits (the rule reduces its position; wait for a bearish structure)

| structure | wait up to (min) | segment | trades | days | occurred | mean wait (min) | mean bp | median bp | t | share worse | mean if better | mean if worse | just waiting, bp | structure adds, bp | t (adds) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| big sweep (N=240, 10 bp, k=1) | 240 | design | 137 | 113 | 0.15 | 223 | 32.2 | 15.9 | 1.98 | 0.38 | 113.4 | -100.5 | 29.4 | 4.6 | 1.32 |
| big sweep (N=240, 10 bp, k=1) | 240 | validation | 69 | 63 | 0.06 | 232 | -28.0 | 21.6 | -0.96 | 0.39 | 73.2 | -185.5 | -26.5 | -2.9 | -1.97 |
| big sweep (N=240, 10 bp, k=1) | 1,440 | design | 136 | 112 | 0.64 | 861 | 15.7 | 67.2 | 0.44 | 0.34 | 172.9 | -291.9 | 84.1 | -45.7 | -2.11 |
| big sweep (N=240, 10 bp, k=1) | 1,440 | validation | 69 | 63 | 0.42 | 1,120 | -88.5 | -16.0 | -2.26 | 0.55 | 138.3 | -273.6 | -56.7 | -45.8 | -2.22 |
| break of structure (5/5 swings) | 240 | design | 137 | 113 | 1.00 | 29 | 3.2 | -10.4 | 0.56 | 0.63 | 61.3 | -31.3 | 29.4 | -2.1 | -0.44 |
| break of structure (5/5 swings) | 240 | validation | 69 | 63 | 1.00 | 27 | -9.3 | -9.0 | -1.31 | 0.72 | 36.8 | -26.8 | -26.5 | -4.6 | -1.45 |
| return into fair value gap | 240 | design | 137 | 113 | 1.00 | 27 | -15.1 | -3.8 | -1.22 | 0.55 | 52.8 | -69.5 | 29.4 | -20.2 | -1.81 |
| return into fair value gap | 240 | validation | 69 | 63 | 1.00 | 20 | 2.6 | 5.4 | 0.51 | 0.43 | 24.9 | -26.4 | -26.5 | 5.5 | 1.48 |
| small sweep (N=30, 2 bp, k=3) | 240 | design | 137 | 113 | 0.99 | 45 | 18.6 | 17.7 | 2.22 | 0.25 | 52.4 | -83.9 | 29.4 | 12.6 | 2.01 |
| small sweep (N=30, 2 bp, k=3) | 240 | validation | 69 | 63 | 0.99 | 40 | -1.9 | 13.6 | -0.18 | 0.26 | 24.2 | -75.8 | -26.5 | 1.0 | 0.17 |

### Entries and exits together

| structure | wait up to (min) | segment | trades | days | occurred | mean wait (min) | mean bp | median bp | t | share worse | mean if better | mean if worse | just waiting, bp | structure adds, bp | t (adds) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| big sweep (N=240, 10 bp, k=1) | 240 | design | 284 | 242 | 0.19 | 215 | 23.7 | 33.0 | 2.38 | 0.34 | 94.7 | -115.2 | 24.4 | 1.7 | 0.61 |
| big sweep (N=240, 10 bp, k=1) | 240 | validation | 132 | 123 | 0.06 | 233 | -11.0 | 20.4 | -0.68 | 0.39 | 61.1 | -125.5 | -8.6 | -3.1 | -1.51 |
| big sweep (N=240, 10 bp, k=1) | 1,440 | design | 283 | 241 | 0.70 | 812 | 14.2 | 66.4 | 0.74 | 0.34 | 149.0 | -248.4 | 24.1 | -13.8 | -1.11 |
| big sweep (N=240, 10 bp, k=1) | 1,440 | validation | 132 | 123 | 0.46 | 1,077 | -53.6 | -4.3 | -2.13 | 0.51 | 129.7 | -231.5 | -43.2 | -25.5 | -1.79 |
| break of structure (5/5 swings) | 240 | design | 284 | 242 | 1.00 | 30 | 0.8 | -10.5 | 0.19 | 0.66 | 60.5 | -30.2 | 24.4 | -2.5 | -0.77 |
| break of structure (5/5 swings) | 240 | validation | 132 | 123 | 1.00 | 25 | -9.4 | -9.2 | -2.34 | 0.75 | 30.9 | -22.8 | -8.6 | -6.8 | -3.30 |
| return into fair value gap | 240 | design | 284 | 242 | 1.00 | 28 | -6.8 | -1.0 | -1.05 | 0.52 | 44.0 | -54.1 | 24.4 | -9.8 | -1.64 |
| return into fair value gap | 240 | validation | 132 | 123 | 1.00 | 19 | 0.8 | 2.1 | 0.24 | 0.44 | 21.1 | -25.1 | -8.6 | 2.8 | 1.14 |
| small sweep (N=30, 2 bp, k=3) | 240 | design | 284 | 242 | 0.99 | 43 | 10.3 | 15.5 | 1.82 | 0.25 | 41.6 | -81.8 | 24.4 | 5.6 | 1.35 |
| small sweep (N=30, 2 bp, k=3) | 240 | validation | 132 | 123 | 0.99 | 41 | 3.0 | 13.0 | 0.53 | 0.25 | 23.4 | -58.2 | -8.6 | 4.5 | 1.31 |

### Supplement with more power: the same comparison on every calendar day, both sides

| structure | wait up to (min) | segment | trades | days | occurred | mean wait (min) | mean bp | median bp | t | share worse | mean if better | mean if worse | just waiting, bp | structure adds, bp | t (adds) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| big sweep (N=240, 10 bp, k=1) | 240 | design | 5,744 | 1,436 | 0.21 | 210 | 3.5 | 7.5 | 2.49 | 0.46 | 92.2 | -100.6 | 0.0 | 3.7 | 2.64 |
| big sweep (N=240, 10 bp, k=1) | 240 | validation | 2,188 | 547 | 0.11 | 225 | 1.1 | 4.2 | 1.12 | 0.48 | 77.2 | -80.9 | 0.0 | 1.1 | 1.08 |
| big sweep (N=240, 10 bp, k=1) | 1,440 | design | 5,712 | 1,428 | 0.71 | 807 | -1.1 | 41.9 | -0.33 | 0.38 | 135.3 | -221.6 | 0.0 | -2.1 | -0.69 |
| big sweep (N=240, 10 bp, k=1) | 1,440 | validation | 2,184 | 546 | 0.51 | 1,030 | -6.2 | 30.3 | -1.59 | 0.43 | 144.3 | -207.3 | 0.0 | -6.9 | -1.76 |
| break of structure (5/5 swings) | 240 | design | 5,744 | 1,436 | 1.00 | 30 | -1.1 | -10.5 | -1.17 | 0.70 | 63.2 | -28.6 | 0.0 | -1.1 | -1.13 |
| break of structure (5/5 swings) | 240 | validation | 2,188 | 547 | 1.00 | 29 | -0.5 | -9.0 | -0.38 | 0.73 | 47.1 | -18.2 | 0.0 | -0.4 | -0.34 |
| return into fair value gap | 240 | design | 5,744 | 1,436 | 1.00 | 27 | -2.4 | -3.7 | -3.34 | 0.57 | 42.5 | -35.6 | 0.0 | -2.4 | -3.31 |
| return into fair value gap | 240 | validation | 2,188 | 547 | 1.00 | 17 | 0.8 | -0.9 | 1.23 | 0.52 | 22.2 | -18.6 | 0.0 | 0.9 | 1.29 |
| small sweep (N=30, 2 bp, k=3) | 240 | design | 5,744 | 1,436 | 0.98 | 44 | 4.2 | 11.4 | 4.28 | 0.33 | 32.6 | -54.5 | 0.0 | 4.1 | 4.29 |
| small sweep (N=30, 2 bp, k=3) | 240 | validation | 2,188 | 547 | 0.99 | 41 | 1.3 | 8.1 | 1.05 | 0.32 | 20.9 | -41.0 | 0.0 | 1.3 | 1.04 |

### Reading

* **Small sweep-and-reclaim (occurs within about 40 minutes in 99% of cases).** Entries: +2.6 bp (t 0.3) on design, +8.4 bp (t 2.0) on validation. Exits: +18.6 bp (t 2.2) on design, -1.9 bp (t -0.2) on validation. Together: +10.3 bp (t 1.8) and +3.0 bp (t 0.5). The median is around +13 to +15 bp and the price is worse in only a quarter of the trades, but when it is worse it is worse by 82 bp (design) and 58 bp (validation) on average against a gain of 42 and 23 bp when it is better. No cell is significant in both segments. On all days the same wait is worth +4.2 bp (t 4.3) on design and +1.3 bp (t 1.0) on validation: a small positive number that is shrinking, like the pattern's own effect in Part A.
* **Big sweep.** It usually does not come: within 4 hours in 23% (design) and 6% (validation) of the entries. The 4-hour result is therefore just the result of trading 4 hours late (compare the "just waiting" column; the structure itself adds between -3 and +5 bp, never with |t| above 2). Waiting up to 24 hours gave +14.2 bp (t 0.7) on design and -53.6 bp (t -2.1) on validation, with the price worse in 51% of the validation trades.
* **Break of structure as confirmation.** The price is worse in 69% of the design entries and 78% of the validation entries; on average -1.5 bp (t -0.3) and -9.5 bp (t -2.8). On all days: -1.1 and -0.5 bp with the price worse in 70% and 73% of cases. Confirmation is paid for.
* **Return into a fair value gap.** Around zero in both segments, with the sign changing.
* **Scale.** With perfect hindsight the best price in the first 4 hours was about 82 bp (design) and 64 bp (validation) better than the open (median). The structures capture none of it reliably. With a turnover of 13 to 16 times the equity per year, one bp of average improvement is worth about 0.13 to 0.16% of the equity per year: the +3 bp point estimate of validation would be about half a percent a year (USD 10 on a USD 2,000 account), the -9 bp of the confirmation entry would cost about one and a half percent a year.
* **Power.** With 284 and 132 trades the standard error of the mean is 6 and 6 bp, so only improvements of 15 bp or more could have been detected on the trend trades themselves; the all-days supplement narrows the unconditional effect of the wait to a few bp.

**Answer to the owner's idea as stated:** on this evidence, waiting for a 1-minute structure before a daily trend trade is not a source of better prices. The most favourable variant (wait for a small sweep and reclaim) gives a better fill three times in four and loses it all in the fourth; the "wait for confirmation" variant gets a worse price in two trades out of three or more.

## 8. What survived

| claim | design | validation | survives? |
|---|---|---|---|
| The named structures predict returns at 1 to 15 minutes by enough to pay costs | largest gross effect 5.1 bp against 12-14 bp | small sweep +0.35 bp | no |
| Continuation readings (break of structure, fair value gap, breakout that holds, displacement) | mostly reversed | not carried forward | no |
| Limit orders at gap / block edges earn the bounce | -1.29 and -0.86 bp at the next open | not carried forward (resting orders on the small sweep: -0.52 bp per fill) | no (adverse selection) |
| Reversal after a small sweep, as a statistical effect | +0.91 bp, t 12.2 | +0.35 bp, t 4.3 | yes, and irrelevant at this size |
| Reversal after a sweep of the 4-hour extreme, 60 min | +12.7 bp, t 6.2; net taker -0.6 | +9.7 bp, t 3.8; net taker -3.8 | effect yes, costs no |
| The same, 240 min, taker | net +6.0 bp, t 1.3 | net +5.3 bp, t 1.0 | not shown |
| The same, 240 min, resting orders | +14.1 bp per fill, t 2.8 | +14.1 bp per fill, t 2.6 | yes, as an optimistic bound |
| Waiting for a structure improves trend-trade fills | best case +10.3 bp, t 1.8 | +3.0 bp, t 0.5 | no |
| Waiting for a break of structure usually gives a worse price | worse in 66% of trades, median -10.5 bp, mean +0.8 (t 0.2) | worse in 75%, median -9.2 bp, mean -9.4 (t -2.3) | frequency yes; mean cost significant in validation only |

### What follows for the project

* **Drop** as a basis for strategies or entry rules: break of structure, change of character, fair value gaps (as continuation and as a place to rest orders), order blocks, compression breakouts, displacement candles and sweeps of the previous day's high and low on 1-minute bars. They fail by a factor of ten to a hundred before any question of robustness arises.
* **Do not delay trend trades to wait for a 1-minute trigger.** Executing at the daily open (or spreading the order over a short window) is as good as any of the waits tested, and waiting for confirmation is worse.
* **One lead worth a dedicated, pre-registered follow-up**, and nothing more than a lead: buying after a sharp push to a new 4-hour low on the altcoin perpetuals, with resting orders, holding hours rather than minutes. A test of it has to be a portfolio simulation (one position per coin, a cap on simultaneous positions, funding, explicit non-fills), has to fix the rule from the design and validation numbers above, and has to be judged once on the hold-out. The pattern vocabulary is not needed for it: "new 4-hour low by at least 10 bp" is the whole signal.

## 9. Caveats

1. Verdict 'edge_only_with_maker_fills' rests on ONE pre-registered pair (sweep of the 4-hour extreme, 240-minute hold). Its taker net (+6.0 design, +5.3 validation) is not distinguishable from zero (t 1.3 and 1.0); the maker figure assumes a passive exit at the open price 240 minutes later (optimistic); with a taker exit it is +9.4 / +9.4 bp (t 1.9 / 1.7).
2. The effect is not a property of the 'liquidity sweep' pattern: on design, a thrust to a new 4-hour low was followed by the same rebound whether the bar closed back inside (sweep) or price stayed below for three closes (acceptance). It is a generic dip-rebound, on the long side only; the short side (sweeps of highs) earns nothing.
3. It is unstable in time and across coins: about zero in 2023 (+2.7 bp at 240 min, t 0.2) and weak in the last 18 months of design (2022-07..2023-12: +8.8 bp, t 1.5), back in 2024-25; carried by XRP, SOL, DOGE, BNB. For BTC and ETH alone the validation figure is +4.0 bp (t 0.6) on perpetuals and about the same on spot.
4. This is an event study, not a portfolio backtest: events come in bursts across correlated coins (many per sell-off), so a small account could take only a fraction of them; capital limits, overlapping positions, funding payments and liquidation risk while buying falling prices are not modelled.
5. Fills are assumed at the next 1-minute open with 1-2 bp slippage. These events occur in the fastest minutes of the day, when spreads are widest, so taker results are more likely over- than under-stated. Bars cannot show queue position; the resting-order simulation counts a fill only when price trades strictly through the order.
6. Multiple testing: 420 looks on design (plus 420 on PAXG, reported separately, and year/coin/side splits of about 15 looks). Design figures of the selected pairs are biased upward by selection; the validation figures are not. Validation was evaluated once, for the three pre-registered pairs and the pre-registered diagnostics only. The hold-out (from 2025-07-01) was never loaded.
7. The long-only split of the selected pair (excess over baseline minus taker cost about +12-13 bp, t 2.1 design / 1.3 validation) was noticed on design before validation but was not pre-registered as its own test; treat it as a hypothesis for the hold-out, not a finding.
8. Part B has little power on the actual trend trades (284 design, 132 validation; standard errors of 4 to 25 bp depending on the structure). The all-days supplement narrows the unconditional effect of waiting for the small sweep to +4.2 bp (design) and +1.3 bp (validation). All Part B trades start at 00:00 UTC, an unusual minute (daily close, funding), so the result may not carry to other clock times.
9. The matched baseline uses volatility quintiles formed within each calendar year (a benchmark, not a signal, so it is not causal by construction). Disclosure: at the very start a data-availability printout of the full files (row counts, date range, number of gaps; no prices) was seen before the segments were separated in code.
10. PAXG (tokenised gold) cannot be judged on 1-minute bars: its tick is about 5 bp of price and about a third of its bars have no trade, so 379 of 420 looks are 'significant' through bid-ask bounce alone and none clears the 24 bp spot round trip.

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
