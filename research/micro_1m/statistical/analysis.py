"""Tables from the per-day sums written by run_events.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eventstudy as es  # noqa: E402

OUT = HERE / "out"
H_INDEX = {h: j for j, h in enumerate(es.HORIZONS)}


class Run:
    """Per-day sums of one (segment, market, symbol)."""

    def __init__(self, segment: str, market: str, symbol: str, tag: str = ""):
        data = np.load(OUT / segment / f"{market}_{symbol}{tag}.npz", allow_pickle=False)
        self.segment, self.market, self.symbol = segment, market, symbol
        self.names = [str(x) for x in data["names"]]
        self.agg = data["agg"]  # (definition, field, horizon, day)
        self.meta = json.loads(str(data["meta"]))
        self.row = {name: i for i, name in enumerate(self.names)}
        self.definitions = {d["name"]: d for d in self.meta["definitions"]}
        self.day0, self.n_days = self.meta["day0"], self.meta["n_days"]
        self.dates = pd.to_datetime(np.arange(self.day0, self.day0 + self.n_days), unit="D")

    def block(self, name: str) -> np.ndarray | None:
        return self.agg[self.row[name]] if name in self.row else None


def load(segment: str, market: str, symbols, tag: str = "") -> list[Run]:
    return [Run(segment, market, symbol, tag) for symbol in symbols]


def _stat_row(block: np.ndarray, h: int, day_mask: np.ndarray | None = None) -> dict:
    j = H_INDEX[h]
    n, s, b, q, w = (block[k, j] for k in range(5))
    if day_mask is not None:
        n, s, b, q, w = (np.where(day_mask, x, 0.0) for x in (n, s, b, q, w))
    lags = es.HAC_LAGS[h]
    gross = es.cluster_mean(n, s, lags)
    excess = es.cluster_mean(n, s - b, lags)
    total = n.sum()
    return {
        "events": int(total),
        "days": gross["days"],
        "gross_bp": gross["mean"] * 1e4,
        "base_bp": b.sum() / total * 1e4 if total else float("nan"),
        "excess_bp": excess["mean"] * 1e4,
        "t_day": excess["t"],
        "t_gross": gross["t"],
        "t_naive": es.naive_t(total, s.sum(), q.sum()),
        "hit": w.sum() / total if total else float("nan"),
    }


def pooled(runs: list[Run], name: str, h: int, day_mask: np.ndarray | None = None) -> dict:
    """Statistics of one definition and horizon pooled over the runs' symbols (clusters = calendar days)."""
    blocks = [(run, run.block(name)) for run in runs if run.block(name) is not None]
    if not blocks:
        return {}
    total = sum(block for _, block in blocks)
    row = _stat_row(total, h, day_mask)
    j = H_INDEX[h]
    events = {run.symbol: float((block[0, j] if day_mask is None else np.where(day_mask, block[0, j], 0.0)).sum()) for run, block in blocks}
    n_all = sum(events.values())
    for mode in es.COST_MODES:
        cost = sum(events[sym] * es.round_trip_cost_bp(sym, mode) for sym in events) / n_all if n_all else float("nan")
        row[f"net_{mode}_bp"] = row["gross_bp"] - cost
        row[f"cost_{mode}_bp"] = cost
    return row


def table(runs: list[Run], names=None, all_horizons: bool = False) -> pd.DataFrame:
    """One row per (definition, horizon that counts as a look)."""
    reference = runs[0]
    rows = []
    for name in names or reference.names:
        d = next((run.definitions[name] for run in runs if name in run.definitions), None)
        if d is None:
            continue
        for h in (es.HORIZONS if all_horizons else d["horizons"]):
            row = {"definition": name, "family": d["family"], "h": h, "hypothesis": d["hypothesis"]}
            row.update(pooled(runs, name, h))
            rows.append(row)
    return pd.DataFrame(rows)


def by_year(runs: list[Run], name: str, h: int) -> pd.DataFrame:
    years = runs[0].dates.year
    rows = {}
    for year in sorted(set(years)):
        row = pooled(runs, name, h, np.asarray(years == year))
        rows[int(year)] = {k: row.get(k) for k in ("events", "days", "gross_bp", "excess_bp", "t_day", "net_perp_taker_bp")}
    return pd.DataFrame(rows).T


def by_symbol(runs: list[Run], name: str, h: int) -> pd.DataFrame:
    rows = {}
    for run in runs:
        if run.block(name) is None:
            continue
        row = pooled([run], name, h)
        rows[run.symbol] = {k: row.get(k) for k in ("events", "days", "gross_bp", "excess_bp", "t_day", "net_perp_taker_bp")}
    return pd.DataFrame(rows).T


def halves(runs: list[Run], name: str, h: int) -> pd.DataFrame:
    """First and second half of the segment's calendar days (a cheap stability check inside design)."""
    n_days = runs[0].n_days
    first = np.arange(n_days) < n_days // 2
    rows = {}
    for label, mask in (("first half", first), ("second half", ~first)):
        row = pooled(runs, name, h, mask)
        rows[label] = {k: row.get(k) for k in ("events", "days", "gross_bp", "excess_bp", "t_day")}
    return pd.DataFrame(rows).T
