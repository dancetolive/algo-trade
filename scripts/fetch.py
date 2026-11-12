"""Download Binance bulk market data into the local parquet store.

Examples (run from anywhere; paths are resolved from --root):

    python -I scripts/fetch.py symbols
    python -I scripts/fetch.py klines --market spot --interval 1d --symbols @usdt
    python -I scripts/fetch.py klines --market um --interval 1m --symbols BTCUSDT,ETHUSDT --since 2020-01
    python -I scripts/fetch.py funding --symbols @usdt
    python -I scripts/fetch.py rebuild --market um --interval 1d      # re-derive parquet from local archives

What "build" does for candles, per symbol: merge the monthly archives; snap
open times to the interval grid; look for short holes (the monthly files of some
symbols lack a few days that the daily files do contain) and patch them from the
daily archives; remove runs of zero-volume bars lasting a day or more (halted and
delisted markets), so that a market that is not trading has no bar at all; and
cut the history into separate instruments where the ticker was relisted,
redenominated or reused (SYMBOL keeps the latest one, SYMBOL__1, SYMBOL__2, ...
hold the earlier ones).
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


def _build_klines(market: str, interval: str, symbol: str, raw_root: Path, repair: bool):
    """Merge, patch, clean and segment one symbol's candles. Returns ({name: frame}, info)."""
    monthly = sorted((raw_root / vision.klines_prefix(market, symbol, interval)).glob("*.zip"))
    frame = vision.snap_to_grid(vision.build_klines(monthly), interval)
    holes = vision.missing_days(frame.index, interval)
    if repair:
        for day in holes:
            vision.download_unlisted(vision.daily_kline_key(market, symbol, interval, day), raw_root)
    daily = sorted((raw_root / vision.daily_klines_prefix(market, symbol, interval)).glob("*.zip"))
    if daily:
        frame = vision.snap_to_grid(vision.build_klines(monthly + daily), interval)
    still_missing = vision.missing_days(frame.index, interval)
    frame, dead = vision.drop_dead_runs(frame, interval)
    pieces = vision.split_instruments(frame, interval) if len(frame) else []
    named = {}
    for number, piece in enumerate(pieces, 1):
        named[symbol if number == len(pieces) else f"{symbol}__{number}"] = piece
    info = {
        "months": len(monthly),
        "daily_patches": len(daily),
        "hole_days_found": len(holes),
        "hole_days_left": len(still_missing),
        "dead_bars_removed": dead,
        "segments": len(pieces),
    }
    return named, info


def _build_funding(symbol: str, raw_root: Path):
    paths = sorted((raw_root / vision.funding_prefix(symbol)).glob("*.zip"))
    frame = vision.build_funding(paths)
    return ({symbol: frame} if len(frame) else {}), {"months": len(paths)}


def _write_outputs(args, kind: str, market: str, interval: str, symbols: list[str], skip: set[str]) -> int:
    """Build one parquet file per symbol from the local archives and update the manifest."""
    root: Path = args.root
    raw_root = root / "data" / "raw"
    meta_dir = root / "data" / "meta"
    out_dir = root / "data" / "parquet" / market / interval
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = meta_dir / f"manifest_{market}_{interval}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    failed = set(skip)
    totals = {"daily_patches": 0, "hole_days_left": 0, "dead_bars_removed": 0, "extra_segments": 0}
    written = 0
    todo = [s for s in symbols if s not in skip]

    def build(symbol: str):
        if kind == "funding":
            return _build_funding(symbol, raw_root)
        return _build_klines(market, interval, symbol, raw_root, repair=not args.no_repair)

    # Parsing and hole-patching run in threads; files and the manifest are written here.
    for position, result in _run_pool([lambda s=s: build(s) for s in todo], args.workers, "build"):
        symbol = todo[position]
        if isinstance(result, Exception):
            failed.add(symbol)
            print(f"  ! build failed for {symbol}: {result}", file=sys.stderr)
            continue
        named, info = result
        # drop whatever an earlier build left for this ticker, then write the current instruments
        for stale in [out_dir / f"{symbol}.parquet", *out_dir.glob(f"{symbol}__*.parquet")]:
            stale.unlink(missing_ok=True)
        for name in [k for k in manifest if k == symbol or k.startswith(f"{symbol}__")]:
            del manifest[name]
        for name, frame in named.items():
            frame.to_parquet(out_dir / f"{name}.parquet")
            written += 1
            manifest[name] = {"rows": int(len(frame)), "first": frame.index[0].isoformat(), "last": frame.index[-1].isoformat(), **info}
        for key in ("daily_patches", "hole_days_left", "dead_bars_removed"):
            totals[key] += info.get(key, 0)
        totals["extra_segments"] += max(info.get("segments", 1) - 1, 0)
    manifest_path.write_text(json.dumps(manifest, indent=0, sort_keys=True))
    print(f"  wrote {written} parquet files to {out_dir}; {len(failed)} symbols failed", flush=True)
    if kind != "funding":
        print(
            f"  patched from daily archives: {totals['daily_patches']} files; hole days still missing: "
            f"{totals['hole_days_left']}; zero-volume bars removed: {totals['dead_bars_removed']}; "
            f"earlier instruments split off: {totals['extra_segments']}",
            flush=True,
        )
    if failed:
        print("  failed symbols: " + ", ".join(sorted(failed)), file=sys.stderr)
    return 1 if failed else 0


def _fetch_dataset(args, kind: str) -> int:
    root: Path = args.root
    raw_root = root / "data" / "raw"
    meta_dir = root / "data" / "meta"
    market = "um" if kind == "funding" else args.market
    interval = "funding" if kind == "funding" else args.interval
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

    # 3. build
    return _write_outputs(args, kind, market, interval, sorted(listing), failed)


def cmd_rebuild(args) -> int:
    """Re-derive parquet from the archives already on disk (no listing, no monthly downloads)."""
    raw_root = args.root / "data" / "raw"
    kind = "funding" if args.interval == "funding" else "klines"
    market = "um" if kind == "funding" else args.market
    if kind == "funding":
        base = raw_root / vision.MARKET_ROOT["um"] / "fundingRate"
        symbols = sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
    else:
        base = raw_root / vision.MARKET_ROOT[market] / "klines"
        symbols = sorted(p.name for p in base.iterdir() if (p / args.interval).is_dir()) if base.is_dir() else []
    if args.symbols:
        wanted = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        symbols = [s for s in symbols if s in wanted]
    print(f"rebuild {market} {args.interval}: {len(symbols)} symbols", flush=True)
    return _write_outputs(args, kind, market, args.interval, symbols, set())


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
        p.add_argument("--no-repair", action="store_true", help="do not patch holes from the daily archives")

    rebuild = sub.add_parser("rebuild", help="rebuild parquet from local archives")
    rebuild.add_argument("--market", choices=["spot", "um"], default="um")
    rebuild.add_argument("--interval", required=True, help="1d, 1h, 1m, ... or 'funding'")
    rebuild.add_argument("--symbols", default="", help="optional comma list; default: everything on disk")
    rebuild.add_argument("--no-repair", action="store_true", help="do not patch holes from the daily archives")
    rebuild.add_argument("--workers", type=int, default=6)

    args = parser.parse_args()
    if args.command == "symbols":
        return cmd_symbols(args)
    if args.command == "rebuild":
        return cmd_rebuild(args)
    return _fetch_dataset(args, "funding" if args.command == "funding" else "klines")


if __name__ == "__main__":
    raise SystemExit(main())
