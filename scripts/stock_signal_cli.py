"""Interactive CLI for fetching stock signals using yfinance with rich UI."""

from __future__ import annotations

import copy
import json
import os
import sys
from json import JSONDecodeError
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
import requests
from requests import RequestException

try:
    from google import genai
except ImportError:  # pragma: no cover - fallback when package unavailable
    genai = None
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.theme import Theme
from rich import box



IS_FROZEN = getattr(sys, "frozen", False)
BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
PROJECT_ROOT = BASE_DIR if IS_FROZEN else Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

APP_DATA_DIR = Path(os.getenv("APPDATA", Path.home())) / "BotForexMT5"
DEFAULT_CONFIG_PATH = BASE_DIR / "config.json"
USER_CONFIG_PATH = APP_DATA_DIR / "config.json"

load_dotenv()

from analyzers.market_analyzer import MarketAnalyzer  # noqa: E402
from utils.secure_store import (
    CredentialDecryptionError,
    SecureCredentialStore,
    filter_credentials,
)


DEFAULT_TIMEFRAME = "M5"
GEMINI_MODEL = "gemini-2.5-flash"

# Fallback to avoid NameError when legacy references expect a module-level status.
status: Optional[Any] = None

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/129.0.0.0 Safari/537.36"
)
os.environ.setdefault("YF_USER_AGENT", USER_AGENT)


STRATEGY_ALIASES: Dict[str, str] = {
    "jangka_panjang": "long_term",
    "jangka panjang": "long_term",
    "long term": "long_term",
    "long": "long_term",
    "jangka_pendek": "short_term",
    "jangka pendek": "short_term",
    "short term": "short_term",
    "short": "short_term",
    "agresif": "aggressive",
    "aggressive": "aggressive",
    "konservatif": "conservative",
    "conservative": "conservative",
    "momentum": "momentum",
    "scalping": "scalping",
    "dividen": "dividend",
    "value": "dividend",
    "dividend": "dividend",
    "event": "event_driven",
    "event-driven": "event_driven",
    "event_driven": "event_driven"
}


def normalize_strategy_key(raw: str) -> str:
    key = raw.strip().lower().replace('-', '_')
    return STRATEGY_ALIASES.get(key, key)


class DataFetchError(Exception):
    """Kesalahan saat mengambil data pasar eksternal."""


def _explain_status(code: str) -> str:
    if code.startswith("HTTP 401"):
        return " (akses ditolak Yahoo; biasanya karena signature permintaan berbeda. Coba ulang beberapa menit atau gunakan koneksi lain.)"
    if code.startswith("HTTP 429"):
        return " (permintaan terlalu sering. Tunggu beberapa menit sebelum mencoba lagi.)"
    return ""


def _build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return session


def diagnose_connectivity() -> Dict[str, str]:
    tests = {
        "Yahoo Finance": "https://query1.finance.yahoo.com/v7/finance/quote?symbols=BBCA.JK",
        "NewsAPI": "https://newsapi.org/v2/top-headlines?country=id&pageSize=1",
        "Internet": "https://www.google.com",
    }
    results: Dict[str, str] = {}
    news_api_key = os.getenv("NEWS_API_KEY") or os.getenv("news_api_key")
    for name, url in tests.items():
        if name == "NewsAPI" and not news_api_key:
            results[name] = "Lewati (API key belum diisi)"
            continue
        try:
            headers = {}
            if name == "NewsAPI" and news_api_key:
                headers["X-Api-Key"] = news_api_key
            headers.setdefault("User-Agent", USER_AGENT)
            resp = requests.get(url, timeout=5, headers=headers)
            if resp.status_code == 200:
                results[name] = "OK"
            else:
                explanation = ""
                if name == "Yahoo Finance" and resp.status_code == 401:
                    explanation = " - akses ditolak, kemungkinan perlu ganti jaringan atau tunggu"
                elif name == "Yahoo Finance" and resp.status_code == 429:
                    explanation = " - dibatasi sementara, coba lagi nanti"
                results[name] = f"HTTP {resp.status_code}{explanation}"
        except RequestException as exc:
            results[name] = f"Gagal: {exc}"
    return results


CREDENTIAL_STORE_PATH = (APP_DATA_DIR / "credentials" / "api_keys.enc")
REQUIRED_ENV_KEYS = ["GEMINI_API_KEY", "NEWS_API_KEY"]


def _collect_existing_credentials(config: dict) -> Dict[str, str]:
    values: Dict[str, str] = {}
    current = config.get("current", {}) if config else {}
    for key in REQUIRED_ENV_KEYS:
        lower = key.lower()
        value = (
            os.getenv(key)
            or os.getenv(lower)
            or current.get(lower)
            or current.get(key)
        )
        if value:
            values[key] = value
    return values


def ensure_api_credentials(console: Console, config: dict) -> Dict[str, str]:
    """Ensure encrypted credentials exist and are loaded into environment."""

    def has_all(creds: Dict[str, str]) -> bool:
        return all(creds.get(key) for key in REQUIRED_ENV_KEYS)

    existing = _collect_existing_credentials(config)
    if has_all(existing):
        return existing

    store = SecureCredentialStore(CREDENTIAL_STORE_PATH)

    if store.exists():
        console.print("[info]Kredensial terenkripsi ditemukan. Masukkan passphrase untuk membuka.[/info]")
        for attempt in range(3):
            passphrase = Prompt.ask(
                "Passphrase",
                password=True,
                console=console,
                default="",
            )
            if not passphrase:
                console.print("[error]Passphrase tidak boleh kosong.[/error]")
                continue
            try:
                decrypted = filter_credentials(store.load(passphrase))
            except CredentialDecryptionError as exc:
                remaining = 2 - attempt
                console.print(f"[error]{exc}[/error]")
                if remaining >= 0:
                    console.print(f"[warning]Kesempatan tersisa: {remaining + 1}[/warning]")
                continue
            for key, value in decrypted.items():
                os.environ[key] = value
            existing = _collect_existing_credentials(config)
            if has_all(existing):
                console.print("[success]Kredensial berhasil dimuat.[/success]")
                return existing
            console.print("[warning]Kredensial belum lengkap di file terenkripsi. Harap isi ulang.[/warning]")
            break
        console.print("[error]Gagal membuka kredensial terenkripsi. Program dihentikan.[/error]")
        sys.exit(1)

    console.print(
        Panel(
            "Belum ada kredensial tersimpan. Anda harus membuat passphrase dan memasukkan API key Gemini & News.",
            title="Setup Kredensial",
            border_style="bright_magenta",
        )
    )

    if not Confirm.ask("Lanjut membuat kredensial sekarang?", console=console, default=True):
        console.print("[error]Kredensial wajib untuk melanjutkan. Program dihentikan.[/error]")
        sys.exit(1)

    while True:
        passphrase = Prompt.ask("Buat passphrase", password=True, console=console)
        confirm_passphrase = Prompt.ask("Ulangi passphrase", password=True, console=console)
        if not passphrase:
            console.print("[error]Passphrase tidak boleh kosong.[/error]")
            continue
        if passphrase != confirm_passphrase:
            console.print("[error]Passphrase tidak sama, coba lagi.[/error]")
            continue

        creds: Dict[str, str] = {}
        for key in REQUIRED_ENV_KEYS:
            value = Prompt.ask(f"Masukkan {key}", password=True, console=console).strip()
            if not value:
                console.print("[error]Semua API key wajib diisi.[/error]")
                break
            creds[key] = value
        else:  # only runs if loop not broken
            filtered = filter_credentials(creds)
            if not has_all(filtered):
                console.print("[error]API key tidak lengkap. Ulangi kembali.")
                continue
            store.save(filtered, passphrase)
            for key, value in filtered.items():
                os.environ[key] = value
            console.print("[success]Kredensial terenkripsi tersimpan. Jaga passphrase Anda baik-baik![/success]")
            return filtered

        if not Confirm.ask("Ingin mencoba lagi?", console=console, default=True):
            console.print("[error]Kredensial belum tersimpan. Program dihentikan.[/error]")
            sys.exit(1)


TIMEFRAME_MAP = {
    "M1": ("1m", "7d"),
    "M5": ("5m", "1mo"),
    "M15": ("15m", "2mo"),
    "M30": ("30m", "3mo"),
    "H1": ("60m", "6mo"),
    "H4": ("60m", "1y"),
    "D1": ("1d", "2y"),
}


def load_config() -> dict:
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    source = DEFAULT_CONFIG_PATH
    target = USER_CONFIG_PATH if IS_FROZEN else DEFAULT_CONFIG_PATH
    if IS_FROZEN and not target.exists():
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        else:
            raise FileNotFoundError("config.json tidak ditemukan di paket.")
    with target.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: dict) -> None:
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = USER_CONFIG_PATH if IS_FROZEN else DEFAULT_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def get_strategy_profiles(config: dict) -> Dict[str, Dict]:
    return config.get("strategy_profiles", {})


def apply_strategy_profile(config: dict, profile_key: str) -> Optional[Dict]:
    profiles = get_strategy_profiles(config)
    if not profiles:
        return None

    key = profile_key.lower()
    profile = profiles.get(key)
    if not profile:
        return None

    params = copy.deepcopy(profile.get("params", {}))
    if not params:
        params = copy.deepcopy(config.get("current", {}).get("stock_strategy", {}))

    config.setdefault("current", {})
    config["current"]["stock_strategy"] = params
    config["current"]["strategy_profile"] = key

    if "min_threshold" in profile:
        config["current"]["min_signal_strength"] = profile["min_threshold"]

    return profile


def init_gemini() -> Optional["genai.Client"]:
    if genai is None:
        return None
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("gemini_api_key")
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def download_market_data(alias: str, period: str, interval: str) -> pd.DataFrame:
    try:
        return yf.download(
            tickers=alias,
            period=period,
            interval=interval,
            auto_adjust=False,
            prepost=False,
            progress=False,
            threads=False,
        )
    except (JSONDecodeError, ValueError) as exc:
        raise DataFetchError(
            f"Respons Yahoo Finance untuk {alias} tidak valid. Periksa koneksi internet atau coba lagi nanti."
        ) from exc
    except Exception as exc:  # pragma: no cover - defensif
        raise DataFetchError(f"Gagal mengambil data {alias}: {exc}") from exc


def fetch_from_chart_api(alias: str, period: str, interval: str) -> Tuple[pd.DataFrame, Optional[str]]:
    client = _build_session()
    try:
        try:
            from curl_cffi import requests as curl_requests  # type: ignore

            client = curl_requests.Session()
            client.headers.update({
                "User-Agent": USER_AGENT,
                "Accept": "application/json, text/plain, */*",
            })
            get_fn = client.get
        except ImportError:
            get_fn = client.get

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{alias}"
        params = {
            "range": period,
            "interval": interval,
            "includePrePost": "false",
            "events": "div,splits",
        }
        response = get_fn(url, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return pd.DataFrame(), f"Chart API error {exc}"

    chart_data = (payload.get("chart", {}) or {}).get("result")
    if not chart_data:
        error_info = (payload.get("chart", {}) or {}).get("error") or {}
        code = error_info.get("code") or ""
        description = error_info.get("description") or ""
        message = f"Chart API kosong {code} {description}".strip()
        return pd.DataFrame(), message

    result = chart_data[0]
    timestamps = result.get("timestamp") or []
    if not timestamps:
        return pd.DataFrame(), "Chart API tidak mengembalikan timestamp"

    quotes = result.get("indicators", {}).get("quote", [{}])[0]
    time_index = pd.to_datetime(timestamps, unit="s", utc=True)
    try:
        time_index = time_index.tz_convert("Asia/Jakarta")
    except Exception:
        time_index = time_index.tz_localize(None)

    df = pd.DataFrame({
        "time": time_index,
        "open": quotes.get("open"),
        "high": quotes.get("high"),
        "low": quotes.get("low"),
        "close": quotes.get("close"),
        "tick_volume": quotes.get("volume"),
    })
    df.dropna(subset=["close"], inplace=True)
    if df.empty:
        return pd.DataFrame(), "Chart API mengembalikan data kosong"
    df["tick_volume"] = df["tick_volume"].fillna(0)
    return df, None


def prepare_dataframe(alias: str, period: str, interval: str, candles: int) -> Tuple[pd.DataFrame, Optional[str]]:
    try:
        data = download_market_data(alias, period, interval)
    except DataFetchError as exc:
        fallback_df, fallback_error = fetch_from_chart_api(alias, period, interval)
        if fallback_df is not None and not fallback_df.empty:
            return fallback_df, None
        message = str(exc)
        if fallback_error:
            message = f"{message}. Upaya fallback chart API gagal: {fallback_error}"
        return pd.DataFrame(), message

    if data is None or data.empty:
        fallback_df, fallback_error = fetch_from_chart_api(alias, period, interval)
        if fallback_df is not None and not fallback_df.empty:
            return fallback_df, None
        detail = fallback_error or "Yahoo Finance mengembalikan data kosong."
        return pd.DataFrame(), (
            f"{detail} (EXE & CLI). Jalankan perintah 'test connection' dan coba lagi." 
        )

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
    return df, None


def safe_text(value) -> str:
    return str(value).encode("ascii", "ignore").decode("ascii")


def build_ai_prompt(entry: Dict) -> str:
    stock = entry.get("stock", {})
    overall = entry.get("overall", {})
    metrics = entry.get("metrics", {})
    analysis = entry.get("analysis", {})
    news = analysis.get("news", {})
    strategy_info = analysis.get("strategy", {})

    stock_reasons = "; ".join(map(str, stock.get("reasons", []))) or "-"
    overall_reasons = "; ".join(map(str, overall.get("reasons", []))) or "-"
    news_summary = news.get("summary") or "Tidak ada ringkasan berita."
    headlines = news.get("headlines", [])
    headlines_text = "\n".join([
        f"  - {safe_text(headline)}" for headline in headlines[:3]
    ]) or "  - (Tidak ada headline terkini)"
    strategy_label = strategy_info.get("label") or strategy_info.get("key", "-").title()
    strategy_desc = strategy_info.get("description") or "-"

    prompt = f"""
Analisislah saham Indonesia {entry.get('symbol')} dan jawab dalam bahasa Indonesia.

Ringkasan Pasar:
- Alias/Ticker: {entry.get('alias')}
- Harga penutupan terakhir: {metrics.get('last_close')}
- Perubahan harga 5 candle: {metrics.get('price_change_5'):+.2f}%
- Perubahan harga 20 candle: {metrics.get('price_change_20'):+.2f}%
- Tertinggi/Terendah 10 candle: {metrics.get('recent_high')} / {metrics.get('recent_low')}
- Level breakout / breakdown: {stock.get('breakout_level')} / {stock.get('breakdown_level')}
- Rasio volume: {stock.get('volume_ratio')}
- Tren: {stock.get('trend')}
- ATR (volatilitas): {stock.get('atr')}

Sinyal:
- Strategi saham: {stock.get('signal')} (alasan: {stock_reasons})
- Sinyal keseluruhan: {overall.get('signal')} dengan kekuatan {overall.get('strength')} (alasan: {overall_reasons})

Strategi Aktif:
- Profil: {safe_text(strategy_label)} ({safe_text(strategy_info.get('key', '-'))})
- Fokus: {safe_text(strategy_desc)}

Berita Terkini:
- Dampak keseluruhan: {news.get('impact', 'NEUTRAL')}
- Skor sentimen: {news.get('sentiment_score', 0)}
- Ringkasan: {safe_text(news_summary)}
- Headline penting:
{headlines_text}

Berikan hanya 3 poin bullet, masing-masing ringkas:
1. Outlook singkat sesi berikutnya (bullish/bearish/sideways) beserta dampak utama dari berita di atas; jelaskan juga jika tidak ada berita relevan.
2. Level teknikal penting yang perlu diperhatikan (support/resistance/breakout).
3. Tips praktis untuk trader manual (ide entry, konfirmasi, dan catatan manajemen risiko) yang mempertimbangkan sentimen berita.
"""
    return prompt


def generate_ai_summary(client: Optional["genai.Client"], entry: Dict) -> str:
    if not client:
        return "Gemini AI tidak tersedia (cek GEMINI_API_KEY)."
    try:
        prompt = build_ai_prompt(entry)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"temperature": 0.4},
        )
        text = response.text.strip() if response and response.text else "(Tidak ada respon AI)"
        return text
    except Exception as exc:
        return f"Gagal mendapatkan analisis AI: {exc}"


def fetch_signal(
    symbol: str,
    timeframe: str,
    candles: int,
    config: dict,
    analyzer: MarketAnalyzer,
    console: Optional[Console] = None,
    status: Optional[object] = None,
) -> Dict:
    alias_map = config.get("current", {}).get("symbol_aliases", {})
    alias = alias_map.get(symbol, f"{symbol}.JK")
    interval, period = TIMEFRAME_MAP.get(timeframe.upper(), TIMEFRAME_MAP[DEFAULT_TIMEFRAME])

    message = f"[bold cyan]Mengambil data {alias} ({period}/{interval})..."
    if status is not None:
        status.update(message)
    elif console:
        console.log(message)

    df, fetch_error = prepare_dataframe(alias, period, interval, candles)
    if fetch_error:
        return {
            "symbol": symbol,
            "alias": alias,
            "error": f"{fetch_error}{_explain_status(fetch_error)} Jalankan perintah 'test connection' untuk diagnosa jaringan.",
        }

    if df.empty:
        return {
            "symbol": symbol,
            "alias": alias,
            "error": (
                f"Data {alias} tidak tersedia. Pastikan koneksi ke Yahoo Finance dapat diakses (gunakan perintah 'test connection')."
                if alias != symbol
                else "Data tidak tersedia. Periksa koneksi dan coba lagi dengan perintah 'test connection'."
            ),
        }

    df = df.tail(candles)
    analysis = analyzer.analyze_market(df, symbol, config)
    stock_signal = analysis.get("stock_strategy", {})
    overall = analysis.get("overall", {})
    close_series = df["close"] if "close" in df else pd.Series(dtype=float)
    metrics = {
        "last_close": float(close_series.iloc[-1]) if not close_series.empty else None,
        "price_change_5": float(((close_series.iloc[-1] / close_series.iloc[-5]) - 1) * 100) if len(close_series) >= 5 else 0.0,
        "price_change_20": float(((close_series.iloc[-1] / close_series.iloc[-20]) - 1) * 100) if len(close_series) >= 20 else 0.0,
        "recent_high": float(df["high"].tail(10).max()) if "high" in df else None,
        "recent_low": float(df["low"].tail(10).min()) if "low" in df else None,
    }
    return {
        "symbol": symbol,
        "alias": alias,
        "stock": stock_signal,
        "overall": overall,
        "last_close": metrics["last_close"],
        "analysis": analysis,
        "metrics": metrics,
        "data": df.tail(60),
    }


def main() -> None:
    config = load_config()
    current = config.get("current", {})
    default_symbols = current.get("symbols_to_trade", [current.get("symbol")])
    default_timeframe = current.get("timeframe", DEFAULT_TIMEFRAME)
    default_candles = current.get("candles", 100)
    strategy_profiles = get_strategy_profiles(config)
    strategy_profile_key = current.get("strategy_profile")
    if strategy_profiles:
        normalized_key = None
        if strategy_profile_key and strategy_profile_key.lower() in strategy_profiles:
            normalized_key = strategy_profile_key.lower()
        elif strategy_profile_key:
            normalized_key = normalize_strategy_key(strategy_profile_key)
            if normalized_key not in strategy_profiles:
                normalized_key = None
        if normalized_key is None:
            normalized_key = "aggressive" if "aggressive" in strategy_profiles else next(iter(strategy_profiles))
            applied_profile = apply_strategy_profile(config, normalized_key)
            if applied_profile:
                current = config.get("current", {})
                strategy_profile_key = normalized_key
                try:
                    save_config(config)
                except Exception:
                    pass
        else:
            strategy_profile_key = normalized_key
    default_strategy = strategy_profile_key

    console = Console(theme=Theme({"info": "cyan", "error": "bold red", "success": "green"}))
    credentials = ensure_api_credentials(console, config)
    news_api_key = credentials.get("NEWS_API_KEY")
    analyzer = MarketAnalyzer(news_api_key=news_api_key)
    gemini_client = init_gemini()

    intro_panel = Panel(
        "[bold cyan]Ketik simbol[/] (contoh: [bold]BBCA[/], [bold]BBCA.JK[/]) pisahkan dengan spasi.\n"
        "Perintah: [yellow]set timeframe H1[/], [yellow]set candles 200[/], [yellow]set strategy aggressive[/], [yellow]strategies[/], [yellow]help[/], [yellow]ai BBCA[/], [yellow]default[/], [yellow]exit[/].",
        title="Stock Signal CLI",
        border_style="bright_magenta",
    )
    console.print(intro_panel)

    help_panel = Panel(
        """
[bold]Indikator utama[/]
- EMA 20/50 crossover → menentukan trend (UP/DOWN)
- Highest/Lowest lookback 20 candle → level breakout/breakdown
- Volume ratio ≥ 1.5 → filter breakout kuat
- ATR 14 × 1.5 → buffer stop-loss & take-profit
- RSI 14 (<=65 BUY, >=35 SELL) → hindari overbought/oversold

[bold]Sinyal[/]
- stock signal: khusus breakout saham (WAIT/BUY/SELL)
- overall: gabungan teknikal, pola, breakout, scalping
- reasons: pemicu tambahan (misal "Breakout", "Near Resistance")

[bold]Strategi[/]
- Gunakan [yellow]strategies[/] untuk melihat daftar profil.
- [yellow]set strategy <nama>[/] mengganti profil aktif dan menyimpan ke config.json.
- Profil strategi memengaruhi parameter EMA/RSI, ambang volume, bobot berita, serta batas kekuatan sinyal.

[bold]Diagnostik[/]
- Jalankan [yellow]test connection[/] untuk memeriksa koneksi ke Yahoo Finance, NewsAPI, dan internet umum.

[bold]Tips[/]
- Harga & level ditampilkan dalam rupiah, gunakan sebagai referensi
- Volume ratio < 1 → breakout lemah, tunggu konfirmasi
- Strength ≥ 0.3 → sinyal kuat; < 0.1 → lemah
\nPerintah tambahan:\n- [bold]help[/] menampilkan panel ini\n- [bold]ai BBCA[/] meminta ringkasan Gemini untuk sinyal terakhir (butuh GEMINI_API_KEY)
""",
        title="Help",
        border_style="green",
    )

    symbols = default_symbols
    timeframe = default_timeframe
    candles = default_candles
    last_results: Dict[str, Dict] = {}

    while True:
        try:
            current_strategy_key = config.get("current", {}).get("strategy_profile", strategy_profile_key)
            strategy_profile = strategy_profiles.get(current_strategy_key, {}) if strategy_profiles else {}
            strategy_label = strategy_profile.get("label", (current_strategy_key or "-").upper())
            raw = Prompt.ask(
                "[bold white]\n[/][bright_black]([/][magenta]{}[/] | candles [magenta]{}[/] | strategy [magenta]{}[/][bright_black])[/]".format(
                    timeframe,
                    candles,
                    strategy_label,
                ),
                console=console,
            )
        except (EOFError, KeyboardInterrupt):
            console.print("\n[info]Keluar.[/info]")
            break

        if not raw:
            continue

        if raw.lower() in {"exit", "quit"}:
            console.print("[info]Keluar.[/info]")
            break

        if raw.lower() == "default":
            symbols = default_symbols
            timeframe = default_timeframe
            candles = default_candles
            if default_strategy and strategy_profiles:
                profile = apply_strategy_profile(config, default_strategy)
                if profile:
                    current = config.get("current", {})
            console.print("[success]Pengaturan dikembalikan ke default.[/success]")
            continue

        if raw.lower() == "help":
            console.print(help_panel)
            continue

        if raw.lower() == "test connection":
            console.print("[info]Menjalankan diagnosa koneksi...[/info]")
            diagnoses = diagnose_connectivity()
            table = Table(title="Diagnosa Koneksi", box=box.SIMPLE_HEAD, style="white")
            table.add_column("Layanan", style="cyan")
            table.add_column("Status", style="bold")
            for service, status_text in diagnoses.items():
                style = "green" if status_text == "OK" else "red"
                table.add_row(service, f"[{style}]{status_text}[/{style}]")
            console.print(table)
            continue

        if raw.lower() == "strategies":
            if not strategy_profiles:
                console.print("[error]Strategi belum dikonfigurasi dalam config.json.[/error]")
                continue
            table = Table(title="Profil Strategi", box=box.MINIMAL_DOUBLE_HEAD, style="white")
            table.add_column("Key", style="cyan")
            table.add_column("Nama", style="bold cyan")
            table.add_column("Deskripsi", overflow="fold")
            for key, profile in strategy_profiles.items():
                table.add_row(key, profile.get("label", key.title()), profile.get("description", "-"))
            console.print(table)
            continue

        if raw.lower().startswith("set "):
            parts = raw.split()
            if len(parts) == 3 and parts[1].lower() == "timeframe":
                timeframe = parts[2].upper()
                console.print(f"[success]Timeframe diubah ke {timeframe}.[/success]")
            elif len(parts) == 3 and parts[1].lower() == "candles":
                try:
                    candles = int(parts[2])
                    console.print(f"[success]Candles diubah ke {candles}.[/success]")
                except ValueError:
                    console.print("[error]Candles harus berupa angka.[/error]")
            elif len(parts) >= 3 and parts[1].lower() == "strategy":
                desired_key_raw = " ".join(parts[2:])
                normalized = normalize_strategy_key(desired_key_raw)
                if not strategy_profiles or normalized not in strategy_profiles:
                    console.print("[error]Strategi tidak dikenal. Jalankan [yellow]strategies[/] untuk daftar lengkap.[/error]")
                    continue
                profile = apply_strategy_profile(config, normalized)
                if profile:
                    try:
                        save_config(config)
                    except Exception as exc:
                        console.log(f"[bright_black]Gagal menyimpan config: {exc}[/bright_black]")
                    strategy_profile_key = normalized
                    current = config.get("current", {})
                    default_strategy = normalized
                    console.print(f"[success]Strategi diubah ke {profile.get('label', normalized.title())}.[/success]")
                else:
                    console.print("[error]Gagal menerapkan strategi tersebut.[/error]")
            else:
                console.print("[error]Perintah set tidak dikenal.[/error]")
            continue

        if raw.lower().startswith("ai"):
            if not gemini_client:
                console.print("[error]Gemini AI tidak tersedia. Pastikan GEMINI_API_KEY sudah diset.[/error]")
                continue
            ai_symbols = raw.split()[1:] or list(last_results.keys())
            if not ai_symbols:
                console.print("[error]Belum ada data untuk dianalisis. Jalankan pencarian simbol terlebih dahulu.[/error]")
                continue
            for sym in ai_symbols:
                sym_upper = sym.upper()
                entry = last_results.get(sym_upper)
                if not entry:
                    entry = fetch_signal(
                        sym_upper,
                        timeframe,
                        candles,
                        config,
                        analyzer,
                        console=console,
                    )
                if entry.get("error"):
                    console.print(f"[error]{sym_upper}: {entry['error']}[/error]")
                    continue
                ai_text = generate_ai_summary(gemini_client, entry)
                console.print(Panel(ai_text, title=f"AI Insight {sym_upper}", border_style="magenta"))
            continue

        symbols = raw.upper().split()

        results: List[Dict] = []
        with console.status("[bold cyan]Mengambil data...") as status:
            for symbol in symbols:
                entry = fetch_signal(
                    symbol,
                    timeframe,
                    candles,
                    config,
                    analyzer,
                    console=console,
                    status=status,
                )
                results.append(entry)
                if not entry.get("error"):
                    last_results[symbol] = entry

        table = Table(title="Ringkasan Sinyal", box=box.MINIMAL_DOUBLE_HEAD, style="white")
        table.add_column("Symbol", style="bold cyan")
        table.add_column("Alias", style="cyan")
        table.add_column("Harga", justify="right")
        table.add_column("Stock Sig", style="magenta")
        table.add_column("Trend")
        table.add_column("Vol x", justify="right")
        table.add_column("Overall", style="green")
        table.add_column("Strength", justify="right")

        detail_panels: List[Panel] = []
        for entry in results:
            if entry.get("error"):
                console.print(f"[error]{entry['symbol']}: {entry['error']}[/error]")
                continue

            stock = entry.get("stock", {})
            overall = entry.get("overall", {})
            vol_ratio = stock.get("volume_ratio")
            strength = overall.get("strength", 0)

            table.add_row(
                entry["symbol"],
                entry.get("alias", "-"),
                f"{entry.get('last_close', '-'):,}" if entry.get("last_close") else "-",
                safe_text(stock.get("signal", "WAIT")),
                safe_text(stock.get("trend", "-")),
                f"{vol_ratio:.2f}" if isinstance(vol_ratio, (int, float)) else "-",
                safe_text(overall.get("signal", "WAIT")),
                f"{strength:.2f}",
            )

            reasons = stock.get("reasons") or []
            overall_reasons = overall.get("reasons") or []
            panel_text = (
                f"Breakout lvl : {safe_text(stock.get('breakout_level', '-'))}\n"
                f"Breakdown lvl: {safe_text(stock.get('breakdown_level', '-'))}\n"
                f"ATR          : {safe_text(stock.get('atr', '-'))}\n"
                f"Stock reasons: {safe_text('; '.join(map(str, reasons)) or '-')}\n"
                f"Overall ctx  : {safe_text('; '.join(map(str, overall_reasons)) or '-')}"
            )
            detail_panels.append(
                Panel(
                    panel_text,
                    title=f"{entry['symbol']} Detail",
                    border_style="bright_blue" if stock.get("signal") == "BUY" else "bright_yellow",
                )
            )

        console.print()
        console.print(table)
        for panel in detail_panels:
            console.print(panel)


if __name__ == "__main__":
    main()
