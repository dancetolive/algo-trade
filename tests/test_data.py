import zipfile

import numpy as np
import pandas as pd
import pytest

from algotrade.data import store, vision

ROW = "{t},100.0,110.0,90.0,105.0,12.5,{c},1300.0,42,6.0,620.0,0"


def write_zip(path, text, member="data.csv"):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, text)
    return path


def test_klines_without_header_in_milliseconds(tmp_path):
    t0 = 1_704_067_200_000  # 2024-01-01 00:00 UTC
    text = "\n".join(ROW.format(t=t0 + i * 86_400_000, c=t0 + (i + 1) * 86_400_000 - 1) for i in range(3))
    frame = vision.read_kline_zip(write_zip(tmp_path / "a.zip", text))
    assert list(frame["open_time"]) == list(pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC"))
    assert frame["close"].tolist() == [105.0] * 3
    assert frame["trades"].dtype == np.int64


def test_klines_with_header_and_microseconds_agree(tmp_path):
    t0 = 1_735_689_600_000  # 2025-01-01 00:00 UTC
    header = ",".join(vision.KLINE_COLUMNS)
    ms = header + "\n" + ROW.format(t=t0, c=t0 + 59_999)
    us = ROW.format(t=t0 * 1000, c=t0 * 1000 + 59_999_999)
    a = vision.read_kline_zip(write_zip(tmp_path / "ms.zip", ms))
    b = vision.read_kline_zip(write_zip(tmp_path / "us.zip", us))
    assert a["open_time"].iloc[0] == b["open_time"].iloc[0] == pd.Timestamp("2025-01-01", tz="UTC")
    assert len(a) == len(b) == 1


def test_build_klines_sorts_and_deduplicates(tmp_path):
    t0 = 1_704_067_200_000
    day = 86_400_000
    first = "\n".join(ROW.format(t=t0 + i * day, c=0) for i in (0, 1))
    second = "\n".join(ROW.format(t=t0 + i * day, c=0) for i in (1, 2))
    paths = [write_zip(tmp_path / "b.zip", second), write_zip(tmp_path / "a.zip", first)]
    frame = vision.build_klines(paths)
    assert frame.index.is_monotonic_increasing and not frame.index.has_duplicates
    assert len(frame) == 3
    assert list(frame.columns) == vision.KLINE_KEEP
    assert vision.build_klines([]).empty


def test_funding_archive(tmp_path):
    text = "calc_time,funding_interval_hours,last_funding_rate\n1704096000003,8,0.0001\n1704124800001,8,-0.0002"
    frame = vision.build_funding([write_zip(tmp_path / "f.zip", text)])
    assert frame["funding_rate"].tolist() == [0.0001, -0.0002]
    assert frame.index[0].floor("min") == pd.Timestamp("2024-01-01 08:00", tz="UTC")


def test_archives_with_unexpected_shape_are_rejected(tmp_path):
    path = tmp_path / "two.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("a.csv", "1")
        archive.writestr("b.csv", "2")
    with pytest.raises(ValueError):
        vision.read_kline_zip(path)


def test_download_rejects_keys_outside_the_archive(tmp_path):
    for key in ("../etc/passwd", "data/../../x.zip", "/abs/path.zip", "other/x.zip"):
        with pytest.raises(ValueError):
            vision.download(vision.RemoteFile(key=key, size=1, etag=""), tmp_path)


def _write_symbol(root, market, interval, symbol, index, close):
    folder = root / market / interval
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "quote_volume": 1.0},
        index=pd.DatetimeIndex(index, name="open_time"),
    )
    frame.to_parquet(folder / f"{symbol}.parquet")


@pytest.fixture
def tiny_store(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGOTRADE_DATA", str(tmp_path))
    monkeypatch.delenv("ALGOTRADE_ALLOW_HOLDOUT", raising=False)
    days = pd.date_range("2025-06-25", "2025-07-05", freq="1D", tz="UTC")
    _write_symbol(tmp_path, "spot", "1d", "AAAUSDT", days, np.arange(len(days), dtype=float))
    _write_symbol(tmp_path, "spot", "1d", "BBBUSDT", days[3:8], np.ones(5))
    return tmp_path


def test_holdout_is_locked_by_default(tiny_store):
    frame = store.load_klines("spot", "1d", "AAAUSDT")
    assert frame.index.max() == pd.Timestamp("2025-06-30", tz="UTC")
    # asking for it explicitly does not help either
    frame = store.load_klines("spot", "1d", "AAAUSDT", end="2026-01-01")
    assert frame.index.max() < store.HOLDOUT_START
    panel = store.load_panel("spot", "1d")
    assert panel["close"].index.max() < store.HOLDOUT_START


def test_holdout_opens_only_with_the_flag(tiny_store, monkeypatch):
    monkeypatch.setenv("ALGOTRADE_ALLOW_HOLDOUT", "1")
    frame = store.load_klines("spot", "1d", "AAAUSDT")
    assert frame.index.max() == pd.Timestamp("2025-07-05", tz="UTC")


def test_panel_is_union_of_times_with_nan_gaps(tiny_store):
    panel = store.load_panel("spot", "1d", start="2025-06-26", end="2025-06-30")
    close = panel["close"]
    assert list(close.columns) == ["AAAUSDT", "BBBUSDT"]
    assert close.index.min() == pd.Timestamp("2025-06-26", tz="UTC")
    assert close.index.max() == pd.Timestamp("2025-06-29", tz="UTC")  # end is exclusive
    assert close["BBBUSDT"].isna().tolist() == [True, True, False, False]
    assert store.symbols("spot", "1d") == ["AAAUSDT", "BBBUSDT"]
    assert store.symbols("spot", "4h") == []


def test_resample_uses_first_open_and_last_close():
    index = pd.date_range("2024-01-01", periods=48, freq="1h", tz="UTC")
    values = np.arange(48, dtype=float)
    hourly = pd.DataFrame(
        {"open": values, "high": values + 0.5, "low": values - 0.5, "close": values + 0.1, "volume": 1.0},
        index=index,
    )
    daily = store.resample_bars(hourly, "1D")
    assert len(daily) == 2
    assert daily["open"].tolist() == [0.0, 24.0]
    assert daily["close"].tolist() == [23.1, 47.1]
    assert daily["high"].tolist() == [23.5, 47.5]
    assert daily["low"].tolist() == [-0.5, 23.5]
    assert daily["volume"].tolist() == [24.0, 24.0]


def _bars(times, volume):
    index = pd.DatetimeIndex(times, tz="UTC", name="open_time")
    return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": volume}, index=index)


def test_missing_days_reports_short_holes_only():
    days = pd.date_range("2022-02-01", "2022-04-30", freq="1D")
    keep = [d for d in days if not (pd.Timestamp("2022-02-26") <= d <= pd.Timestamp("2022-02-28"))]
    keep = [d for d in keep if not (pd.Timestamp("2022-03-10") <= d <= pd.Timestamp("2022-04-05"))]  # 27 days: not a hole
    index = pd.DatetimeIndex(keep, tz="UTC")
    found = vision.missing_days(index, "1d")
    assert [str(d) for d in found] == ["2022-02-26", "2022-02-27", "2022-02-28"]
    hours = pd.date_range("2024-01-01", "2024-01-03 23:00", freq="1h", tz="UTC")
    hours = hours.delete(slice(30, 40))  # ten hours missing inside 2 January
    assert [str(d) for d in vision.missing_days(hours, "1h")] == ["2024-01-02"]
    assert vision.missing_days(hours[:1], "1h") == []


def test_dead_runs_are_removed_but_thin_trading_is_kept():
    hours = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    volume = np.ones(100)
    volume[10:13] = 0  # three quiet hours: a thin market, kept
    volume[40:70] = 0  # thirty hours without a trade: not trading, removed
    frame, removed = vision.drop_dead_runs(_bars(hours, volume), "1h")
    assert removed == 30 and len(frame) == 70
    assert frame.index[39] == hours[39] and frame.index[40] == hours[70]
    days = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    frame, removed = vision.drop_dead_runs(_bars(days, [5, 0, 3, 0, 0]), "1d")
    assert removed == 3 and list(frame["volume"]) == [5, 3]  # a whole day without a trade is always dead
    frame, removed = vision.drop_dead_runs(_bars(days, [1, 1, 1, 1, 1]), "1d")
    assert removed == 0 and len(frame) == 5


def test_unlisted_download_rejects_unsafe_keys(tmp_path):
    for key in ("../x.zip", "data/../x.zip", "/etc/passwd"):
        with pytest.raises(ValueError):
            vision.download_unlisted(key, tmp_path)
    assert vision.daily_kline_key("um", "BTCUSDT", "1d", pd.Timestamp("2022-02-26").date()) == (
        "data/futures/um/daily/klines/BTCUSDT/1d/BTCUSDT-1d-2022-02-26.zip"
    )


def test_holdout_lock_tests_the_close_of_the_bar(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGOTRADE_DATA", str(tmp_path))
    monkeypatch.delenv("ALGOTRADE_ALLOW_HOLDOUT", raising=False)
    weeks = pd.date_range("2025-06-02", periods=6, freq="7D", tz="UTC")  # Mondays; 30 June opens before the cut-off
    _write_symbol(tmp_path, "spot", "1w", "AAAUSDT", weeks, np.arange(6, dtype=float))
    frame = store.load_klines("spot", "1w", "AAAUSDT")
    assert frame.index.max() == pd.Timestamp("2025-06-23", tz="UTC")  # the 30 June bar closes on 7 July: withheld
    assert store.bar_length("4h") == pd.Timedelta(hours=4) and store.bar_length("1w") == pd.Timedelta(days=7)
    with pytest.raises(ValueError):
        store.bar_length("1M")


def test_funding_loader_maps_segments_and_includes_the_boundary_payment(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGOTRADE_DATA", str(tmp_path))
    monkeypatch.delenv("ALGOTRADE_ALLOW_HOLDOUT", raising=False)
    folder = tmp_path / "um" / "funding"
    folder.mkdir(parents=True)
    stamps = pd.to_datetime(["2025-06-30 16:00:00.004", "2025-07-01 00:00:00.010", "2025-07-01 08:00:00.002"], utc=True)
    pd.DataFrame(
        {"funding_interval_hours": 8, "funding_rate": [0.001, 0.002, 0.003]},
        index=pd.DatetimeIndex(stamps, name="calc_time"),
    ).to_parquet(folder / "AAAUSDT.parquet")
    rates = store.load_funding(["AAAUSDT", "AAAUSDT__1", "ZZZUSDT"])
    assert list(rates.columns) == ["AAAUSDT", "AAAUSDT__1"]  # the segment reads its ticker's file; ZZZ has none
    assert rates["AAAUSDT"].tolist() == [0.001, 0.002]  # the payment at the cut-off instant is included, later ones not
    assert store.base_symbol("AAAUSDT__12") == "AAAUSDT"


def test_resample_honours_offsets_and_a_fixed_grid():
    index = pd.date_range("2024-01-01", periods=96, freq="1h", tz="UTC")
    hourly = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": np.arange(96.0), "volume": 1.0}, index=index)
    shifted = store.resample_bars(hourly, "1D", offset="8h")
    assert shifted.index[1] == pd.Timestamp("2024-01-01 08:00", tz="UTC")
    assert shifted["close"].iloc[1] == 31.0  # the bar 08:00 -> 08:00 closes with the 07:00 hourly bar of the next day
    three_a = store.resample_bars(hourly, "3D")
    three_b = store.resample_bars(hourly.iloc[24:], "3D")
    assert set(three_b.index) <= set(three_a.index)  # same buckets whatever the first row is


def test_bars_stamped_off_the_grid_are_snapped():
    stamps = pd.to_datetime(["2018-02-09 08:00:00.000", "2018-02-09 09:28:14.789", "2018-02-09 10:28:14.789"], utc=True)
    frame = _bars(stamps, [1, 1, 1])
    snapped = vision.snap_to_grid(frame, "1h")
    assert list(snapped.index) == list(pd.to_datetime(["2018-02-09 08:00", "2018-02-09 09:00", "2018-02-09 10:00"], utc=True))
    assert vision.snap_to_grid(snapped, "1h") is snapped


def test_history_is_cut_into_instruments():
    days = pd.date_range("2022-01-01", periods=10, freq="1D", tz="UTC")
    frame = _bars(days, 1.0)
    assert len(vision.split_instruments(frame, "1d")) == 1

    relisted = frame.drop(frame.index[3:5])  # a two-day halt: still one instrument
    assert len(vision.split_instruments(relisted, "1d")) == 1
    long_gap = pd.concat([frame.iloc[:3], _bars(pd.date_range("2022-03-01", periods=4, freq="1D", tz="UTC"), 1.0)])
    pieces = vision.split_instruments(long_gap, "1d")
    assert [len(p) for p in pieces] == [3, 4]

    redenominated = frame.copy()
    redenominated.iloc[6:, redenominated.columns.get_indexer(["open", "high", "low", "close"])] = 0.001
    pieces = vision.split_instruments(redenominated, "1d")
    assert [len(p) for p in pieces] == [6, 4]
    assert pieces[1]["close"].iloc[0] == 0.001


def test_parser_rejects_unknown_time_units_and_handles_a_bom(tmp_path):
    t0 = 1_704_067_200_000
    rows = "\n".join(ROW.format(t=t0 + i * 86_400_000, c=0) for i in range(3))
    with_bom = vision.read_kline_zip(write_zip(tmp_path / "bom.zip", "﻿" + rows))
    assert len(with_bom) == 3 and with_bom["open_time"].iloc[0] == pd.Timestamp("2024-01-01", tz="UTC")
    seconds = vision.read_kline_zip(write_zip(tmp_path / "s.zip", ROW.format(t=t0 // 1000, c=0)))
    nanos = vision.read_kline_zip(write_zip(tmp_path / "ns.zip", ROW.format(t=t0 * 1_000_000, c=0)))
    assert seconds["open_time"].iloc[0] == nanos["open_time"].iloc[0] == pd.Timestamp("2024-01-01", tz="UTC")
    with pytest.raises(ValueError, match="outside 2017-2100"):
        vision.read_kline_zip(write_zip(tmp_path / "bad.zip", ROW.format(t=123456, c=0)))
    mixed = ROW.format(t=t0, c=0) + "\n" + ROW.format(t=(t0 + 86_400_000) * 1000, c=0)  # ms then us in one file
    frame = vision.read_kline_zip(write_zip(tmp_path / "mixed.zip", mixed))
    assert list(frame["open_time"]) == list(pd.date_range("2024-01-01", periods=2, freq="1D", tz="UTC"))


def test_two_funding_records_in_one_minute_are_both_paid(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGOTRADE_DATA", str(tmp_path))
    folder = tmp_path / "um" / "funding"
    folder.mkdir(parents=True)
    stamps = pd.to_datetime(["2024-05-11 08:00:00.001", "2024-05-11 08:00:00.004", "2024-05-11 16:00:00.002"], utc=True)
    pd.DataFrame(
        {"funding_interval_hours": 8, "funding_rate": [0.0001, -0.0009, 0.0001]},
        index=pd.DatetimeIndex(stamps, name="calc_time"),
    ).to_parquet(folder / "AAAUSDT.parquet")
    rates = store.load_funding(["AAAUSDT"])["AAAUSDT"]
    assert len(rates) == 2
    assert rates.iloc[0] == pytest.approx(-0.0008) and rates.sum() == pytest.approx(-0.0007)
