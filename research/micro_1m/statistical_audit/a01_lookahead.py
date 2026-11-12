"""Audit step 1: look-ahead test of the researcher's event code by truncation, on real bars.

    python -I a01_lookahead.py um BTCUSDT          (one market/symbol per run; each run < 5 minutes)

For the given symbol (design segment only):
  A. the whole design history is processed once by the researcher's own code (eventstudy.to_grid,
     eventstudy.trailing_features, definitions.generate: every definition, every feature);
  B. the bars after a cut point are REMOVED and everything is recomputed; every feature value and every
     event flag / trade direction at bars <= cut must be bit-identical (NaN == NaN);
  C. the bars after the cut are REPLACED by a distorted future (prices x 1.5 and reversed order of returns,
     volumes x 20, a halt, a hole) and the same comparison is made.
Cut points: random ones, plus adversarial ones (inside a zero-volume run, the bar before a hole in the index,
one bar after a selected event) when the symbol has them.
The auditor's own flags (auditlib) get the same treatment, so the reproduction cannot share a look-ahead.

The researcher's modules are imported read-only (no bytecode is written into their directory).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
THEIR = HERE.parent / "statistical"
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(THEIR))

import auditlib as al  # noqa: E402
import definitions as defs  # noqa: E402  (researcher's)
import eventstudy as es  # noqa: E402  (researcher's)
from algotrade.data import store  # noqa: E402


def same(a: np.ndarray, b: np.ndarray) -> bool:
    if a.dtype == bool or b.dtype == bool:
        return bool(np.array_equal(a, b))
    return bool(np.array_equal(a, b, equal_nan=True))


def run_theirs(frame: pd.DataFrame, market: str, symbol: str, funding):
    grid = es.to_grid(frame, market, symbol)
    feats = es.trailing_features(grid)
    return grid, feats, defs.generate(grid, feats, funding)


def compare_theirs(full, frame_cut: pd.DataFrame, n_keep: int, market: str, symbol: str, funding) -> dict:
    """``full`` = (grid, feats, list of (name, packed mask, direction)); compare the first n_keep grid bars."""
    grid_f, feats_f, defs_f = full
    grid_c, feats_c, gen_c = run_theirs(frame_cut, market, symbol, funding)
    bad = []
    for key in feats_f:
        if not same(feats_f[key][:n_keep], feats_c[key][:n_keep]):
            bad.append(f"feature {key}")
    for key in ("usable", "open", "high", "low", "close", "qv", "trades", "tbqv"):
        if not same(getattr(grid_f, key)[:n_keep], getattr(grid_c, key)[:n_keep]):
            bad.append(f"grid.{key}")
    n_defs = 0
    flags = 0
    for (name, packed, direction), d in zip(defs_f, gen_c):
        assert name == d.name
        n_defs += 1
        mask_full = np.unpackbits(packed, count=len(grid_f)).astype(bool)[:n_keep]
        flags += int(mask_full.sum())
        if not np.array_equal(mask_full, d.mask[:n_keep]):
            bad.append(f"mask {name}: {int((mask_full != d.mask[:n_keep]).sum())} bars differ")
        dir_c = d.direction[:n_keep].astype(np.int8) if isinstance(d.direction, np.ndarray) else d.direction
        dir_f = direction[:n_keep] if isinstance(direction, np.ndarray) else direction
        if isinstance(dir_f, np.ndarray):
            if not np.array_equal(dir_f[mask_full], dir_c[mask_full]):
                bad.append(f"direction {name}")
        elif dir_f != dir_c:
            bad.append(f"direction {name}")
    # positive controls: quantities that are NOT causal must be caught by the same comparison
    controls = {
        "grid.outage (uses the whole zero-volume run)": not same(grid_f.outage[:n_keep], grid_c.outage[:n_keep]),
        "next bar's close (shift -1)": not same(np.concatenate([grid_f.close[1:], [np.nan]])[:n_keep], np.concatenate([grid_c.close[1:], [np.nan]])[:n_keep]),
        "centred 5-bar mean of close": not same(pd.Series(grid_f.close).rolling(5, center=True).mean().to_numpy()[:n_keep],
                                                 pd.Series(grid_c.close).rolling(5, center=True).mean().to_numpy()[:n_keep]),
    }
    return {"definitions": n_defs, "flags_before_cut": flags, "differences": bad, "controls_caught": controls}


def distorted_future(frame: pd.DataFrame, cut_pos: int, seed: int) -> pd.DataFrame:
    """Same bars up to and including row cut_pos; afterwards a deliberately different future."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    tail = out.iloc[cut_pos + 1:].copy()
    if len(tail) < 50:
        return out
    for c in ("open", "high", "low", "close"):
        tail[c] = tail[c].to_numpy()[::-1] * 1.5
    for c in ("volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"):
        tail[c] = tail[c].to_numpy() * 20.0
    tail["trades"] = (tail["trades"].to_numpy() * 20).astype(tail["trades"].dtype)
    # a halt right after the cut and a hole a little later
    k = int(rng.integers(1, 4))
    halt_rows = tail.index[k:k + 12]
    tail.loc[halt_rows, ["volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]] = 0.0
    tail.loc[halt_rows, "trades"] = 0
    hole = tail.index[30:45]
    tail = tail.drop(index=hole)
    return pd.concat([out.iloc[: cut_pos + 1], tail])


def own_flags(frame: pd.DataFrame, market: str, symbol: str, funding) -> dict:
    b = al.Bars(market, symbol, frame=frame)
    falls, rises = al.flag_move5_abs2pct(b)
    dips, pops = al.flag_move1_z3(b)
    out = {"falls": falls, "rises": rises, "dips": dips, "pops": pops, "vol_state": al.vol_state(b)}
    if funding is not None:
        mask, side = al.flag_fund_post5_big(b, funding)
        out["fund"], out["fund_side"] = mask, np.where(mask, side, 0.0)
    return out


def main(argv):
    market, symbol = argv[0], argv[1]
    started = time.time()
    frame = store.load_klines(market, "1m", symbol, end=store.DESIGN_END)
    assert frame.index[-1] < store.DESIGN_END
    funding = None
    if market == "um":
        wide = store.load_funding([symbol], end=store.DESIGN_END)
        funding = wide[symbol] if symbol in wide.columns else None
    grid, feats, gen = run_theirs(frame, market, symbol, funding)
    # masks are stored bit-packed and directions as int8 (they are -1 / 0 / +1) to keep the memory small
    full_defs = [(d.name, np.packbits(d.mask), d.direction.astype(np.int8) if isinstance(d.direction, np.ndarray) else d.direction) for d in gen]
    full = (grid, feats, full_defs)
    n = len(grid)
    mine_full = own_flags(frame, market, symbol, funding)
    print(f"{market} {symbol}: {len(frame)} bars on a grid of {n}, {len(full_defs)} definitions, {len(feats)} features, full pass {time.time() - started:.0f}s", flush=True)

    # ---- cut points (grid positions) ------------------------------------------------------------
    rng = np.random.default_rng({"BTCUSDT": 11, "ETHUSDT": 12, "SOLUSDT": 13, "BNBUSDT": 14, "XRPUSDT": 15, "DOGEUSDT": 16}.get(symbol, 17))
    cuts = {f"random {i + 1}": int(rng.integers(n // 5, n - 5000)) for i in range(2)}
    so_far, total = es.run_lengths(grid.exists & (grid.raw_qv <= 0))
    inside_run = np.flatnonzero((total >= 8) & (so_far == 3))
    if len(inside_run):
        cuts["inside a zero-volume run (3rd bar of >= 8)"] = int(inside_run[len(inside_run) // 2])
    before_hole = np.flatnonzero(grid.exists[:-1] & ~grid.exists[1:])
    if len(before_hole):
        cuts["last bar before a hole in the index"] = int(before_hole[len(before_hole) // 2])
    sel = dict((name, np.unpackbits(packed, count=n).astype(bool)) for name, packed, _ in full_defs if name in ("move5_abs2pct_dn", "move1_z3_dn"))
    ev = np.flatnonzero(sel["move5_abs2pct_dn"])
    if len(ev):
        cuts["the bar of a selected 2% fall event"] = int(ev[len(ev) // 2])
        cuts["two bars after a selected 2% fall event"] = int(ev[len(ev) // 3]) + 2

    position_of = (frame.index.as_unit("ns").asi8 - frame.index.as_unit("ns").asi8[0]) // al.MIN_NS
    report = {"market": market, "symbol": symbol, "bars": int(len(frame)), "definitions": len(full_defs), "checks": []}
    for label, cut in cuts.items():
        rows = int(np.searchsorted(position_of, cut, side="right"))  # frame rows with grid position <= cut
        last_row_pos = int(position_of[rows - 1])
        n_keep = last_row_pos + 1  # compare the grid up to the last bar that is kept
        for mode in ("removed", "replaced"):
            t0 = time.time()
            cut_frame = frame.iloc[:rows] if mode == "removed" else distorted_future(frame, rows - 1, seed=rows % 1000)
            res = compare_theirs(full, cut_frame, n_keep, market, symbol, funding)
            mine_cut = own_flags(cut_frame, market, symbol, funding)
            own_bad = [k for k in mine_full if not same(mine_full[k][:n_keep], mine_cut[k][:n_keep])]
            entry = {"cut": label, "cut_time": str(grid.index[last_row_pos]), "later_bars": mode, "bars_compared": n_keep,
                     "researcher_differences": res["differences"], "researcher_flags_compared": res["flags_before_cut"],
                     "auditor_differences": own_bad, "non_causal_controls_caught": res["controls_caught"]}
            report["checks"].append(entry)
            print(f"  cut '{label}' at {grid.index[last_row_pos]} ({mode}): {res['definitions']} definitions, {res['flags_before_cut']} flags before the cut, "
                  f"researcher differences {len(res['differences'])} {res['differences'][:5]}, auditor differences {own_bad}, "
                  f"controls caught {[int(v) for v in res['controls_caught'].values()]}, {time.time() - t0:.0f}s", flush=True)
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    (out / f"lookahead_{market}_{symbol}.json").write_text(json.dumps(report, indent=1))
    print("total", f"{time.time() - started:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
