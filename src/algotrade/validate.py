"""Guards against the classic ways a backtest lies."""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import pandas as pd

Panel = Mapping[str, pd.DataFrame]


def check_no_lookahead(
    build_weights: Callable[[Panel], pd.DataFrame],
    panel: Panel,
    cuts: int = 6,
    atol: float = 1e-9,
    seed: int = 0,
) -> list[str]:
    """Re-run a strategy on truncated history and demand identical past decisions.

    ``build_weights(panel)`` must return target weights indexed like the panel.
    If the weights for a bar change when later bars are removed, that decision
    used information from the future (centred windows, full-sample scaling,
    forward fills from later data, a shift in the wrong direction, ...).

    Returns a list of human-readable problems; an empty list means the check
    passed. Passing is necessary, not sufficient: it cannot see leakage that is
    baked into the input data itself.
    """
    reference_index = next(iter(panel.values())).index
    full = build_weights(panel)
    n = len(reference_index)
    if n < 50:
        return ["panel too short to test"]
    rng = np.random.default_rng(seed)
    points = sorted(set(rng.integers(n // 3, n - 1, size=cuts).tolist()))
    problems: list[str] = []
    for cut in points:
        truncated = {name: frame.iloc[:cut] for name, frame in panel.items()}
        partial = build_weights(truncated)
        if not partial.index.equals(reference_index[:cut].intersection(partial.index)):
            problems.append(f"cut {cut}: weights are indexed outside the truncated panel")
            continue
        expected = full.reindex(index=partial.index, columns=partial.columns)
        a = partial.to_numpy(dtype=float)
        b = expected.to_numpy(dtype=float)
        mismatch = ~(np.isclose(a, b, atol=atol, rtol=0.0) | (np.isnan(a) & np.isnan(b)))
        if mismatch.any():
            rows = np.where(mismatch.any(axis=1))[0]
            first = partial.index[rows[0]]
            problems.append(
                f"cut {cut} ({reference_index[cut - 1]}): {int(mismatch.sum())} weights differ; "
                f"first differing bar {first} is {cut - 1 - rows[0]} bars before the cut"
            )
    return problems
