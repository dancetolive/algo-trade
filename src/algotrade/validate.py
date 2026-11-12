"""Guards against the classic ways a backtest lies."""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import pandas as pd

Panel = Mapping[str, pd.DataFrame]


def _as_simulated(weights: pd.DataFrame, index: pd.DatetimeIndex, columns: pd.Index) -> np.ndarray:
    """The decisions as ``simulate`` would read them on ``index``.

    A row that is absent or entirely NaN is "no decision" (kept as NaN); in a
    decided row a NaN or absent cell is a target of 0.
    """
    own = weights.reindex(index=index).to_numpy(dtype=float)
    decided = ~np.all(np.isnan(own), axis=1) if own.shape[1] else np.zeros(len(index), dtype=bool)
    values = weights.reindex(index=index, columns=columns).to_numpy(dtype=float)
    values = np.where(np.isnan(values), 0.0, values)
    values[~decided] = np.nan
    return values


def check_no_lookahead(
    build_weights: Callable[[Panel], pd.DataFrame],
    panel: Panel,
    cuts: int = 12,
    atol: float = 1e-9,
    seed: int = 0,
    warmup: int = 50,
) -> list[str]:
    """Re-run a strategy on truncated history and demand identical past decisions.

    ``build_weights(panel)`` must return target weights indexed by (a subset of)
    the panel's bar index. Every frame of the panel must share that index: a
    coarser input (daily data next to hourly bars) has to be aligned causally
    first, see :func:`algotrade.backtest.decisions_on`.

    The history is cut at two kinds of places: right after rows where the
    strategy's decisions change (a peek taken only on rebalance rows shows up
    only there), and at random rows. After each cut the decisions are compared
    over the whole truncated index and over the union of columns, read the way
    the simulator reads them, so a decision that disappears, or a column that is
    only chosen with hindsight, counts as a difference.

    Returns a list of human-readable problems; an empty list means the check
    passed. Passing is necessary, not sufficient: it cannot see leakage that is
    baked into the input data, and it samples cut points rather than trying all.
    """
    reference_index = next(iter(panel.values())).index
    for name, frame in panel.items():
        if not frame.index.equals(reference_index):
            raise ValueError(
                f"panel frame {name!r} is not on the common bar index; align coarser inputs "
                "causally before the check (a value is known at its bar's close)"
            )
    n = len(reference_index)
    if n < warmup + 10:
        return ["panel too short to test"]
    full = build_weights(panel)
    stray = full.index.difference(reference_index)
    if len(stray):
        return [f"weights contain {len(stray)} rows that are not bars of the panel (first: {stray[0]})"]
    full_values = _as_simulated(full, reference_index, full.columns)

    # Rows where a decision is made or changes: cut right after each of a sample of them.
    previous = np.vstack([np.full((1, full_values.shape[1]), np.nan), full_values[:-1]])
    same = np.isclose(full_values, previous, atol=atol, rtol=0.0) | (np.isnan(full_values) & np.isnan(previous))
    changed = np.flatnonzero(~same.all(axis=1)) if full_values.shape[1] else np.array([], dtype=int)
    changed = changed[(changed >= warmup) & (changed < n - 1)]
    rng = np.random.default_rng(seed)
    points: set[int] = set()
    if len(changed):
        chosen = rng.choice(changed, size=min(cuts, len(changed)), replace=False)
        points.update(int(r) + 1 for r in chosen)
    points.update(int(k) for k in rng.integers(warmup, n, size=cuts))

    problems: list[str] = []
    for cut in sorted(points):
        truncated = {name: frame.iloc[:cut] for name, frame in panel.items()}
        partial = build_weights(truncated)
        index = reference_index[:cut]
        outside = partial.index.difference(index)
        if len(outside):
            problems.append(f"cut {cut}: weights are indexed outside the truncated panel (first: {outside[0]})")
            continue
        columns = full.columns.union(partial.columns)
        a = _as_simulated(partial, index, columns)
        b = _as_simulated(full, index, columns)
        mismatch = ~(np.isclose(a, b, atol=atol, rtol=0.0) | (np.isnan(a) & np.isnan(b)))
        if mismatch.any():
            rows = np.flatnonzero(mismatch.any(axis=1))
            names = [str(c) for c in columns[mismatch.any(axis=0)][:4]]
            problems.append(
                f"cut {cut} ({reference_index[cut - 1]}): {int(mismatch.sum())} decisions differ; first differing bar "
                f"{index[rows[0]]} is {cut - 1 - int(rows[0])} bars before the cut; columns: {', '.join(names)}"
            )
    return problems
