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
