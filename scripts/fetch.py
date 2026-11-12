"""Download Binance bulk market data into the local parquet store.

Examples (run from anywhere; paths are resolved from --root):

    python -I scripts/fetch.py symbols
    python -I scripts/fetch.py klines --market spot --interval 1d --symbols @usdt
    python -I scripts/fetch.py klines --market um --interval 1m --symbols BTCUSDT,ETHUSDT --since 2020-01
    python -I scripts/fetch.py funding --symbols @usdt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from algotrade.data import vision


def _resolve_symbols(spec: str, market: str, meta_dir: Path) -> list[str]:
    if spec.startswith("@"):
        name = spec[1:]
        if name == "usdt":
            catalog = json.loads((meta_dir / "symbols.json").read_text())
            return sorted(s for s in catalog[market] if s.endswith("USDT") and "_" not in s)
        listed = (meta_dir / f"{name}.txt").read_text().split()
        return sorted(set(listed))
    return sorted({s.strip().upper() for s in spec.split(",") if s.strip()})


def _month_of(remote: vision.RemoteFile) -> str:
    # ...-YYYY-MM.zip
    return remote.name[:-4][-7:]


def _run_pool(tasks, workers: int, label: str):
    """Run callables concurrently, yielding (index, result_or_exception)."""
    started = time.monotonic()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn): i for i, fn in enumerate(tasks)}
        for future in as_completed(futures):
            done += 1
            if done % 500 == 0 or done == len(futures):
                rate = done / max(time.monotonic() - started, 1e-9)
                print(f"  {label}: {done}/{len(futures)} ({rate:.0f}/s)", flush=True)
            try:
                yield futures[future], future.result()
            except Exception as exc:  # noqa: BLE001 - reported to the caller
                yield futures[future], exc


def cmd_symbols(args) -> int:
    meta_dir = args.root / "data" / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    catalog = {market: vision.list_symbols(market) for market in ("spot", "um")}
    (meta_dir / "symbols.json").write_text(json.dumps(catalog, indent=0))
    for market, symbols in catalog.items():
        usdt = [s for s in symbols if s.endswith("USDT") and "_" not in s]
        print(f"{market}: {len(symbols)} symbols, {len(usdt)} quoted in USDT")
    return 0


def _fetch_dataset(args, kind: str) -> int:
    root: Path = args.root
    raw_root = root / "data" / "raw"
    meta_dir = root / "data" / "meta"
    market = "um" if kind == "funding" else args.market
    interval = "funding" if kind == "funding" else args.interval
    out_dir = root / "data" / "parquet" / market / interval
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = _resolve_symbols(args.symbols, market, meta_dir)
    print(f"{kind} {market} {interval}: {len(symbols)} symbols", flush=True)

    def prefix(symbol: str) -> str:
        return vision.funding_prefix(symbol) if kind == "funding" else vision.klines_prefix(market, symbol, interval)

    # 1. list what exists for every symbol
    listing: dict[str, list[vision.RemoteFile]] = {}
    tasks = [lambda s=s: vision.list_files(prefix(s)) for s in symbols]
    for index, result in _run_pool(tasks, args.workers, "list"):
        symbol = symbols[index]
        if isinstance(result, Exception):
            print(f"  ! listing failed for {symbol}: {result}", file=sys.stderr)
            continue
        files = [f for f in result if args.since <= _month_of(f) <= args.until]
        if files:
            listing[symbol] = files

    # 2. download every archive
    flat = [(symbol, remote) for symbol, files in listing.items() for remote in files]
    total_bytes = sum(remote.size for _, remote in flat)
    print(f"  {len(flat)} archives, {total_bytes / 1e6:.1f} MB", flush=True)
    failed: set[str] = set()
    tasks = [lambda r=remote: vision.download(r, raw_root) for _, remote in flat]
    for index, result in _run_pool(tasks, args.workers, "download"):
        if isinstance(result, Exception):
            symbol, remote = flat[index]
            failed.add(symbol)
            print(f"  ! download failed for {remote.key}: {result}", file=sys.stderr)

    # 3. build one parquet file per symbol
    manifest = {}
    builder = vision.build_funding if kind == "funding" else vision.build_klines
    for symbol in sorted(listing):
        if symbol in failed:
            continue
        paths = [raw_root / remote.key for remote in listing[symbol]]
        try:
            frame = builder(paths)
        except Exception as exc:  # noqa: BLE001 - one bad archive must not stop the batch
            failed.add(symbol)
            print(f"  ! parse failed for {symbol}: {exc}", file=sys.stderr)
            continue
        if frame.empty:
            continue
        frame.to_parquet(out_dir / f"{symbol}.parquet")
        manifest[symbol] = {
            "rows": int(len(frame)),
            "first": frame.index[0].isoformat(),
            "last": frame.index[-1].isoformat(),
            "months": len(paths),
        }
    (meta_dir / f"manifest_{market}_{interval}.json").write_text(json.dumps(manifest, indent=0))
    print(f"  wrote {len(manifest)} parquet files to {out_dir}; {len(failed)} symbols failed", flush=True)
    if failed:
        print("  failed symbols: " + ", ".join(sorted(failed)), file=sys.stderr)
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("symbols", help="catalogue every symbol in the archive")

    klines = sub.add_parser("klines", help="download candlesticks")
    klines.add_argument("--market", choices=["spot", "um"], required=True)
    klines.add_argument("--interval", required=True)

    funding = sub.add_parser("funding", help="download perpetual funding-rate history")

    for p in (klines, funding):
        p.add_argument("--symbols", required=True, help="comma list, @usdt, or @name for data/meta/name.txt")
        p.add_argument("--since", default="2017-01", help="first month, YYYY-MM")
        p.add_argument("--until", default="2026-10", help="last month, YYYY-MM")
        p.add_argument("--workers", type=int, default=16)

    args = parser.parse_args()
    if args.command == "symbols":
        return cmd_symbols(args)
    return _fetch_dataset(args, "funding" if args.command == "funding" else "klines")


if __name__ == "__main__":
    raise SystemExit(main())
