#!/usr/bin/env bash
# Reproduce every number of this study from the parquet store. Each step runs in well under five minutes.
# The selection (SELECTION.json) already exists: select_candidates.py refuses to run again, and the
# validation steps are locked in code to the definitions and horizons recorded there.
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=".venv/bin/python -I"
D=research/micro_1m/statistical
PERPS="BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT DOGEUSDT"

$P -m pytest $D/test_eventstudy.py -q

# design segment (bars before 2024-01-01)
$P $D/run_events.py design um $PERPS
$P $D/run_events.py design spot BTCUSDT ETHUSDT PAXGUSDT
$P $D/event_level.py design um $PERPS
$P $D/event_level.py design spot BTCUSDT ETHUSDT
$P $D/autocorr.py design um
$P $D/design_tables.py
[ -f $D/SELECTION.json ] || $P $D/select_candidates.py

# validation segment (2024-01-01 .. 2025-06-30): selected definitions and horizons only
$P $D/run_events.py validation um $PERPS
$P $D/run_events.py validation spot BTCUSDT ETHUSDT
$P $D/event_level.py validation um $PERPS
$P $D/event_level.py validation spot BTCUSDT ETHUSDT
$P $D/selected_report.py design
$P $D/selected_report.py validation

# Part B and side analyses
$P $D/part_b.py design
$P $D/part_b.py validation
$P $D/design_extras.py
$P $D/survivorship_check.py
$P $D/independent_check.py design
$P $D/independent_check.py validation

$P $D/make_results.py --tables
