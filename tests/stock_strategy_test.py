import json
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analyzers.market_analyzer import MarketAnalyzer


def load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prepare_dataframe(alias: str, period: str, interval: str, candles: int) -> pd.DataFrame:
    data = yf.download(
        tickers=alias,
        period=period,
        interval=interval,
        auto_adjust=False,
        prepost=False
    )

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(-1, axis=1)

    df = data.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "tick_volume"
    }).reset_index().rename(columns={"Datetime": "time"})

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    df = df.drop(columns=["adj_close"], errors="ignore")
    df["tick_volume"] = df["tick_volume"].fillna(0)

    if candles:
        df = df.tail(candles)

    return df


def safe_text(value) -> str:
    return str(value).encode("ascii", "ignore").decode("ascii")


def main() -> None:
    config = load_config()
    current = config.get("current", {})
    symbol = current.get("symbol", "")
    alias_map = current.get("symbol_aliases", {})
    alias = alias_map.get(symbol, f"{symbol}.JK")

    candles = current.get("candles", 100)
    period = "3mo"
    interval = "1h"

    print(f"Fetching {alias} ({period}/{interval}) data from yfinance...")
    df = prepare_dataframe(alias, period, interval, candles)

    if df.empty:
        print("❌ Data tidak tersedia dari yfinance")
        return

    print("Kolom DataFrame:", df.columns.tolist())
    print("Sample tail:\n", df.tail(3))

    analyzer = MarketAnalyzer()
    analysis = analyzer.analyze_market(df, symbol, config)

    stock_signal = analysis.get("stock_strategy", {})
    overall = analysis.get("overall", {})

    print("\n=== Stock Strategy Signal ===")
    for key, value in stock_signal.items():
        print(f"{key}: {safe_text(value)}")

    print("\n=== Overall Signal ===")
    for key, value in overall.items():
        print(f"{key}: {safe_text(value)}")


if __name__ == "__main__":
    main()
