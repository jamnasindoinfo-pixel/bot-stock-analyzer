import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yfinance as yf


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analyzers.market_analyzer import MarketAnalyzer


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def timeframe_to_yf_params(timeframe: str) -> Tuple[str, str]:
    mapping = {
        "M1": ("1m", "7d"),
        "M5": ("5m", "1mo"),
        "M15": ("15m", "2mo"),
        "M30": ("30m", "3mo"),
        "H1": ("60m", "6mo"),
        "H4": ("60m", "1y"),
        "D1": ("1d", "2y"),
    }
    return mapping.get(timeframe.upper(), ("60m", "6mo"))


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


def render_signal(symbol: str, stock_signal: Dict, overall: Dict) -> None:
    print(f"\n=== {symbol} ===")
    print(f"Stock signal : {safe_text(stock_signal.get('signal', 'WAIT'))}")
    reasons = stock_signal.get("reasons") or []
    print(f"Reasons      : {safe_text('; '.join(map(str, reasons)) or '-')}")
    print(f"Trend        : {safe_text(stock_signal.get('trend', 'N/A'))}")
    vol_ratio = stock_signal.get("volume_ratio")
    print(f"Volume ratio : {vol_ratio:.2f}" if isinstance(vol_ratio, (int, float)) else "Volume ratio : -")
    print(f"Breakout lvl : {safe_text(stock_signal.get('breakout_level', '-'))}")
    print(f"Breakdown lvl: {safe_text(stock_signal.get('breakdown_level', '-'))}")
    print(f"ATR          : {safe_text(stock_signal.get('atr', '-'))}")
    print(f"Overall      : {safe_text(overall.get('signal', 'WAIT'))} | Strength {overall.get('strength', 0):.2f}")
    overall_reasons = overall.get("reasons") or []
    print(f"Overall ctx  : {safe_text('; '.join(map(str, overall_reasons)) or '-')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stock trading signals using yfinance data.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "config.json",
        help="Path to config.json (default: project config.json)"
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Optional list of symbols to analyze (defaults to symbols_to_trade)"
    )
    parser.add_argument(
        "--timeframe",
        help="Override timeframe (e.g., M30). Defaults to config current timeframe"
    )
    parser.add_argument(
        "--candles",
        type=int,
        help="Override number of candles fetched"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    current = config.get("current", {})
    symbols = args.symbols or current.get("symbols_to_trade") or [current.get("symbol")]
    timeframe = args.timeframe or current.get("timeframe", "M30")
    candles = args.candles or current.get("candles", 100)
    alias_map = current.get("symbol_aliases", {})

    instrument_type = current.get("instrument_type", "").lower()
    if instrument_type != "stock":
        print("Warning: instrument_type is not set to 'stock'; stock strategy weights may be ignored.")

    interval, period = timeframe_to_yf_params(timeframe)
    analyzer = MarketAnalyzer()

    for symbol in symbols:
        alias = alias_map.get(symbol, f"{symbol}.JK")
        print(f"\nFetching {alias} ({period}/{interval})...")
        df = prepare_dataframe(alias, period, interval, candles)
        if df.empty:
            print("Data unavailable from yfinance.")
            continue

        analysis = analyzer.analyze_market(df, symbol, config)
        stock_signal = analysis.get("stock_strategy", {})
        overall = analysis.get("overall", {})
        render_signal(symbol, stock_signal, overall)


if __name__ == "__main__":
    main()
