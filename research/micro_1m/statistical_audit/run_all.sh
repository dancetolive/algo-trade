#!/usr/bin/env bash
# Re-run the whole audit (each step well under five minutes; reads the parquet store through the loader only,
# never sets ALGOTRADE_ALLOW_HOLDOUT, writes only under this directory).
set -euo pipefail
cd "$(dirname "$0")"
P="/home/user/algotrade/.venv/bin/python -I"
mkdir -p out
$P a00_inspect.py                               # what is in the 1-minute store (gaps, zero-volume bars)
for ms in "um BTCUSDT" "um DOGEUSDT" "um SOLUSDT" "spot ETHUSDT"; do $P a01_lookahead.py $ms; done   # truncation test of the researcher's flags
$P a01b_lookahead_exhaustive.py                 # bar-by-bar truncation around crash days
$P a02_build_events.py design um                # auditor's own events of the selected definitions
$P a02_build_events.py validation um
$P a02_build_events.py design spot BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT DOGEUSDT
$P a02_build_events.py validation spot BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT DOGEUSDT
$P a02_build_events.py validation um PAXGUSDT
$P a03_headline.py                              # headline numbers, by year / coin, later fills, event-by-event match
$P a04_inference.py                             # clustering by day / week / month, block bootstrap, looks
$P a05_placebo.py design                        # time-shift placebos
$P a05_placebo.py validation
$P a06_controls.py                              # design-only volatility-matched controls
$P a07_realism.py                               # costs, tails, anatomy, capital-constrained versions
$P a08_partb.py                                 # Part B re-implemented, fairness checks, planted effect
$P a09_planted.py                               # planted effects pushed through the researcher's own code
$P a10_unused_symbols.py                        # spot alts and the PAXG perpetual
$P a11_grid_spotcheck.py                        # 22 rows of the design grid, bounce and resting-order figures
$P a12_misc.py                                  # bar labelling (open time), extreme days
