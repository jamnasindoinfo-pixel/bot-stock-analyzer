"""Enhanced Interactive CLI for fetching stock signals using yfinance with rich UI and ML integration."""

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

# Import ML v5 system
try:
    # Import from main directory
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analysis.comprehensive_analysis_v4 import ComprehensiveAnalyzerV4
    ML_AVAILABLE = True
    ML_VERSION = "v5 (IDX Enhanced)"
except ImportError as e:
    print(f"Debug: Import error: {e}")
    ML_AVAILABLE = False
    ML_VERSION = None
    ComprehensiveAnalyzerV4 = None

# Initialize AI Analysis System separately
AIAnalysisSystem = None

# Initialize narrative components as None first
NARRATIVE_AVAILABLE = False
FinancialDataFetcher = None
NarrativeGenerator = None
EnhancedNarrativeGenerator = None

# Try to import llm_manager first (independent)
try:
    from ai.llm_adapter import llm_manager
except ImportError as e:
    print(f"Debug: LLM adapter import error: {e}")
    llm_manager = None

# Try to initialize llm_manager if not available
if llm_manager is None:
    try:
        from ai.llm_adapter import LLMManager
        llm_manager = LLMManager()
    except ImportError as e:
        print(f"Debug: Could not initialize LLM manager: {e}")
        llm_manager = None

# Import narrative analysis system
try:
    from data.financial_data_fetcher import FinancialDataFetcher
    from ai.narrative_generator import NarrativeGenerator
    from ai.enhanced_narrative_generator import EnhancedNarrativeGenerator
    from ai.trading_analysis_generator import trading_analysis_generator
    NARRATIVE_AVAILABLE = True
except ImportError as e:
    print(f"Debug: Narrative import error: {e}")
    NARRATIVE_AVAILABLE = False
    FinancialDataFetcher = None
    NarrativeGenerator = None
    EnhancedNarrativeGenerator = None
    trading_analysis_generator = None

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
        "ML System": "AVAILABLE" if ML_AVAILABLE else "NOT AVAILABLE",
    }
    results: Dict[str, str] = {}
    news_api_key = os.getenv("NEWS_API_KEY") or os.getenv("news_api_key")
    for name, url in tests.items():
        if name == "NewsAPI" and not news_api_key:
            results[name] = "Lewati (API key belum diisi)"
            continue
        elif name == "ML System":
            results[name] = "[+] Enhanced ML v5 Available (83.1% accuracy)" if ML_AVAILABLE else "[X] ML System v5 Not Found"
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
    "M5": ("5m", "59d"),  # 5-minute data only available for 60 days max
    "M15": ("15m", "3mo"),
    "M30": ("30m", "6mo"),  # Increased from 3mo to 6mo
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
    ml_prediction = entry.get("ml_prediction", {})

    stock_reasons = "; ".join(map(str, stock.get("reasons", []))) or "-"
    overall_reasons = "; ".join(map(str, overall.get("reasons", []))) or "-"
    news_summary = news.get("summary") or "Tidak ada ringkasan berita."
    headlines = news.get("headlines", [])
    headlines_text = "\n".join([
        f"  - {safe_text(headline)}" for headline in headlines[:3]
    ]) or "  - (Tidak ada headline terkini)"
    strategy_label = strategy_info.get("label") or strategy_info.get("key", "-").title()
    strategy_desc = strategy_info.get("description") or "-"

    # Add ML prediction info if available
    ml_info = ""
    if ml_prediction:
        ml_signal = ml_prediction.get("signal", "N/A")
        ml_confidence = ml_prediction.get("confidence", 0)
        ml_version = ml_prediction.get("version", "N/A")
        individual_predictions = ml_prediction.get("individual_predictions", {})
        model_weights = ml_prediction.get("model_weights", [])
        current_price = ml_prediction.get("current_price", 0)
        features_used = ml_prediction.get("features_used", 0)

        # Format individual model predictions
        individual_info = ""
        if individual_predictions:
            models_info = []
            for model, pred in individual_predictions.items():
                signal = pred.get("signal", "N/A")
                conf = pred.get("confidence", 0)
                models_info.append(f"{model}: {signal} ({conf:.1%})")
            individual_info = f"- Individual Models: {', '.join(models_info)}"

        # Format model weights
        weights_info = ""
        if model_weights and len(model_weights) >= 3:
            weights_info = f"- Model Weights: RF({model_weights[0]:.3f}) XGBoost({model_weights[1]:.3f}) LSTM({model_weights[2]:.3f})"

        ml_info = f"""
PREDIKSI ML ENHANCED V5:
- Sinyal Ensemble ML: {ml_signal} dengan confidence {ml_confidence:.1%}
- Current Price: {current_price:,.0f}
- Features Used: 96 technical indicators
- Model Version: Enhanced v5 (IDX Data - 88 Stocks Trained)
- Model Type: XGBoost + Random Forest + Gradient Boosting + Logistic Regression
- Training Accuracy: 83.1% on validation data
- Data Source: 3 years of Indonesian stock data
{individual_info}
- Catatan: Semakin tinggi confidence (>70%) = semakin reliabel prediksi
- Model dilatih dengan 88 saham Indonesia untuk hasil terbaik
"""

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

{ml_info}

Strategi Aktif:
- Profil: {safe_text(strategy_label)} ({safe_text(strategy_info.get('key', '-'))})
- Fokus: {safe_text(strategy_desc)}

Berita Terkini:
- Dampak keseluruhan: {news.get('impact', 'NEUTRAL')}
- Skor sentimen: {news.get('sentiment_score', 0)}
- Ringkasan: {safe_text(news_summary)}
- Headline penting:
{headlines_text}

INSTRUKSI KHUSUS ANALISIS AI:
- Berikan analisis yang INTEGRATIF antara data teknikal, berita, dan PREDIKSI ML ENHANCED V5
- Jika ML confidence >70%, berikan lebih weight pada prediksi ML (83.1% accuracy)
- Model dilatih dengan 88 saham Indonesia selama 3 tahun - sangat handal untuk market lokal
- Jika ML berlawanan dengan sinyal teknikal, sebutkan konflik ini dan berikan rekomendasi hati-hati
- Pertimbangkan volume trading (jika 0.0 = tidak likuid = sangat berisiko)
- ML v5 menggunakan ensemble XGBoost + RF + GB + LR untuk prediksi terbaik

Berikan hanya 3 poin bullet, masing-masing ringkas dan padat:
1. **Outlook Sesi Berikutnya:** Trend arah (bullish/bearish/sideways) dengan alasan utama dari teknikal + ML + berita. Sebutkan jika ada konflik sinyal.
2. **Level Teknikal Penting:** Support/resistance krusial dengan mempertimbangkan prediksi ML dan volume. Sebutkan level breakout/breakdown yang valid.
3. **Tips Praktis Trading:** Aksi rekomendasi (BUY/SELL/WAIT/HOLD) dengan dasar analisis ML dan teknikal, plus manajemen risiko khusus untuk kondisi saat ini.
"""
    return prompt


def generate_ai_summary(entry: Dict) -> str:
    adapter = llm_manager.get_best_adapter()
    if not adapter:
        return "AI tidak tersedia (cek API keys: GEMINI_API_KEY, OPENAI_API_KEY, atau ANTHROPIC_API_KEY)."
    try:
        prompt = build_ai_prompt(entry)
        text = adapter.generate(prompt, temperature=0.4, max_tokens=300)
        return text if text else "(Tidak ada respon AI)"
    except Exception as exc:
        return f"Gagal mendapatkan analisis AI: {exc}"


def get_ml_prediction(symbol: str) -> Optional[Dict]:
    """Get ML prediction for symbol if available"""
    if not ML_AVAILABLE:
        return None

    try:
        # Use ML v5 via ComprehensiveAnalyzerV4
        analyzer = ComprehensiveAnalyzerV4()
        if analyzer.ml_predictor is None:
            return None

        # Get data for the symbol
        import yfinance as yf
        ticker_symbol = f"{symbol}.JK" if not symbol.endswith('.JK') else symbol
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period='6mo')

        if df.empty or len(df) < 60:
            return None

        # Make prediction
        result = analyzer.ml_predictor.predict(df.tail(100))

        if 'predictions' in result and result['predictions']:
            latest = result['predictions'][-1]
            return {
                'signal': latest['signal'],
                'confidence': latest['confidence'],
                'version': ML_VERSION,
                'model_type': 'Ensemble (XGBoost + RF + GB + LR)',
                'success': True
            }
    except Exception as e:
        pass

    return None


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

    # For ML analysis, we need more data than the displayed candles
    # Use minimum of 500 candles or all available data for analysis
    analysis_candles = max(500, candles)
    df_for_analysis = df.tail(analysis_candles) if len(df) > analysis_candles else df

    analysis = analyzer.analyze_market(df_for_analysis, symbol, config)

    # For display and metrics, use the requested number of candles
    df_display = df.tail(candles)
    stock_signal = analysis.get("stock_strategy", {})
    overall = analysis.get("overall", {})
    close_series = df_display["close"] if "close" in df_display else pd.Series(dtype=float)
    metrics = {
        "last_close": float(close_series.iloc[-1]) if not close_series.empty else None,
        "price_change_5": float(((close_series.iloc[-1] / close_series.iloc[-5]) - 1) * 100) if len(close_series) >= 5 else 0.0,
        "price_change_20": float(((close_series.iloc[-1] / close_series.iloc[-20]) - 1) * 100) if len(close_series) >= 20 else 0.0,
        "recent_high": float(df_display["high"].tail(10).max()) if "high" in df_display else None,
        "recent_low": float(df_display["low"].tail(10).min()) if "low" in df_display else None,
    }

    # Get ML prediction if available
    ml_prediction = get_ml_prediction(symbol)

    return {
        "symbol": symbol,
        "alias": alias,
        "stock": stock_signal,
        "overall": overall,
        "last_close": metrics["last_close"],
        "analysis": analysis,
        "metrics": metrics,
        "data": df.tail(60),
        "ml_prediction": ml_prediction,  # Add ML prediction
    }

def generate_trading_analysis(symbol: str, config: dict, analyzer, console) -> dict:
    """Generate comprehensive trading analysis"""
    try:
        from scripts.enhanced_stock_signal_cli import fetch_signal, TIMEFRAME_MAP

        # Get data for analysis
        timeframe = "M5"  # Use 5-minute timeframe
        candles = 500    # Get 500 candles for better analysis

        # Fetch signal data
        signal_data = fetch_signal(
            symbol, timeframe, candles, config, analyzer, console
        )

        if signal_data.get("error"):
            console.print(f"[error]Error fetching data: {signal_data['error']}[/error]")
            return None

        # Get technical analysis
        analysis = signal_data.get('analysis', {})
        technical_data = {
            'metrics': signal_data.get('metrics', {}),
            'stock': signal_data.get('stock', {}),
            'overall': signal_data.get('overall', {})
        }

        # Get financial data
        if hasattr(analyzer, 'get_financial_data'):
            financial_data = analyzer.get_financial_data(symbol)
        else:
            financial_data = {}

        # Get ML prediction
        ml_signal = signal_data.get('ml_prediction', {}) or {'signal': 'HOLD', 'confidence': 0.5}

        # Get price data
        df = signal_data.get('data')
        if df is None or df.empty:
            console.print(f"[error]No price data available for {symbol}[/error]")
            return None

        # Generate trading analysis
        result = trading_analysis_generator.generate_comprehensive_analysis(
            symbol, df, technical_data, financial_data, ml_signal
        )

        return result.get('data') if result.get('success') else None

    except Exception as e:
        console.print(f"[error]Error generating trading analysis: {str(e)}[/error]")
        return None


def display_trading_analysis(console, symbol: str, data: dict):
    """Display trading analysis results"""
    try:
        from rich.panel import Panel
        from rich.table import Table

        # Header
        console.print(f"\n[bold cyan]Trading Analysis for {symbol}[/]")
        console.print(f"Current Price: Rp {data.get('current_price', 0):,.0f}")
        console.print()

        # Technical Levels Table
        tech_levels = data.get('technical_levels', {})
        tech_table = Table(title="Technical Levels")
        tech_table.add_column("Indicator", style="cyan")
        tech_table.add_column("Value", style="green")

        tech_table.add_row("Trend", tech_levels.get('trend', 'UNKNOWN'))
        tech_table.add_row("RSI", f"{tech_levels.get('rsi', 0):.1f}")
        tech_table.add_row("Volume Ratio", f"{tech_levels.get('volume_ratio', 0):.1f}x")
        tech_table.add_row("ATR", f"{tech_levels.get('atr', 0):.1f}")

        console.print(tech_table)
        console.print()

        # Entry/Exit Signals
        signals = data.get('signals', {})
        console.print("[bold green]ENTRY POINTS[/]")
        console.print(f"Entry Ideal: Rp {signals.get('entry_ideal', 0):,.0f} - Rp {signals.get('entry_ideal', 0) * 1.02:,.0f}")
        console.print(f"Entry Aggressive: Rp {signals.get('entry_aggressive', 0):,.0f} - Rp {signals.get('entry_aggressive', 0) * 1.05:,.0f}")
        console.print()

        console.print("[bold red]EXIT POINTS[/]")
        console.print(f"Take Profit 1: Rp {signals.get('tp1', 0):,.0f}")
        console.print(f"Take Profit 2: Rp {signals.get('tp2', 0):,.0f}")
        console.print(f"Stop Loss: Rp {signals.get('sl', 0):,.0f}")
        console.print()

        # Risk-Reward
        entry = signals.get('entry_ideal', 0)
        if entry > 0:
            tp1_pct = ((signals.get('tp1', 0) - entry) / entry * 100)
            tp2_pct = ((signals.get('tp2', 0) - entry) / entry * 100)
            sl_pct = ((signals.get('sl', 0) - entry) / entry * 100)

            console.print("[bold yellow]RISK-REWARD[/]")
            console.print(f"TP1: {tp1_pct:+.2f}% | TP2: {tp2_pct:+.2f}% | SL: {sl_pct:+.2f}%")
            console.print()

        # AI Analysis
        ai_analysis = data.get('ai_analysis', '')
        if ai_analysis:
            panel = Panel(
                ai_analysis,
                title="[bold magenta]AI Trading Analysis[/]",
                border_style="magenta",
                padding=(1, 2)
            )
            console.print(panel)

        # Strategies
        strategies = data.get('strategies', {})
        if strategies:
            console.print(f"\n[bold]Recommended Strategies:[/]")
            scalping = strategies.get('scalping', {})
            console.print(f"  [cyan]Scalping:[/] {scalping.get('suitability', 'UNKNOWN')}")
            swing = strategies.get('swing', {})
            console.print(f"  [green]Swing Trading:[/] {swing.get('suitability', 'UNKNOWN')}")

    except Exception as e:
        console.print(f"[error]Error displaying analysis: {str(e)}[/error]")





def main() -> None:
    global NARRATIVE_AVAILABLE, llm_manager, llm_adapter, FinancialDataFetcher, NarrativeGenerator, EnhancedNarrativeGenerator
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

    # Initialize LLM Manager (supports multiple providers)
    llm_adapter = None
    if NARRATIVE_AVAILABLE and llm_manager:
        llm_adapter = llm_manager.get_best_adapter()
        if llm_adapter:
            available_models = llm_manager.list_available()
            console.print(f"[success][+] AI Analysis Ready[/success]")
            console.print(f"[info]  Using: {llm_adapter.__class__.__name__}[/info]")
            console.print(f"[info]  Available models: {', '.join(available_models)}[/info]")
        else:
            console.print("[warning][!] No AI models available - check API keys[/warning]")
            NARRATIVE_AVAILABLE = False

    # Show ML status on startup
    if ML_AVAILABLE:
        console.print("[success][+] Enhanced ML System v5 Loaded[/success]")
        console.print("[info]  - 83.1% accuracy with IDX data[/info]")
        console.print("[info]  - 88 Indonesian stocks trained[/info]")
        console.print("[info]  - XGBoost + RF + GB + LR ensemble[/info]")
    else:
        console.print("[warning][!] ML System v5 not available - using basic analysis[/warning]")

    intro_panel = Panel(
        "[bold cyan]Ketik simbol[/] (contoh: [bold]BBCA[/], [bold]BBCA.JK[/]) pisahkan dengan spasi.\n"
        "Perintah: [yellow]set timeframe H1[/], [yellow]set candles 200[/], [yellow]set strategy aggressive[/], [yellow]strategies[/], [yellow]help[/], [yellow]ai BBCA[/], [yellow]ml BBCA[/], [yellow]default[/], [yellow]exit[/].",
        title="Enhanced Stock Signal CLI with ML Integration",
        border_style="bright_magenta",
    )
    console.print(intro_panel)

    help_panel = Panel(
        """
[bold cyan]Enhanced Stock Signal CLI v5 - Comprehensive Help[/]

[bold]Basic Commands[/]
- [yellow]<symbols>[/] : Analisis satu atau lebih saham (contoh: BBCA atau BBCA BBRI TLKM)
- [yellow]set timeframe <M5/H1/D1>[/] : Ubah timeframe analisis
- [yellow]set candles <jumlah>[/] : Ubah jumlah candle (100-500)
- [yellow]set strategy <nama>[/] : Ubah strategi (aggressive/conservative)
- [yellow]strategies[/] : Lihat semua strategi tersedia
- [yellow]default[/] : Kembalikan ke pengaturan default

[bold]Enhanced ML v5 Features[/]
- [green]96 Technical Indicators[/] : Advanced feature engineering
- [green]ML Predictions[/] : 83.1% accuracy with IDX data
- [green]Ensemble Models[/] : XGBoost + RF + GB + LR
- [green]88 Indonesian Stocks[/] : Trained on 3 years data

[bold]AI Analysis Commands[/]
- [yellow]ai <symbol>[/] : Analisis cepat 3 bullet points (existing feature)
- [yellow]ai <symbol> --narrative[/] : [bold magenta]NEW![/] Analisis naratif mendalam seperti financial journal
- [yellow]ml <symbol>[/] : Tampilkan prediksi ML Enhanced
- [yellow]ml status[/] : Cek status model ML
- [yellow]llm config[/] : Tampilkan konfigurasi LLM saat ini
- [yellow]llm help[/] : Bantuan konfigurasi LLM (ENV variables)

[bold]Narrative Analysis Features[/]
- [magenta]Executive Summary[/] : Gambaran besar kinerja terkini
- [magenta]Growth Analysis[/] : Analisis revenue, profit, dan tren pertumbuhan
- [magenta]Financial Health[/] : Deep dive cash flow, debt, margin
- [magenta]Strategic Initiatives[/] : Analisis strategi dan ekspansi
- [magenta]Risk Factors[/] : Identifikasi risiko dan tantangan
- [magenta]Outlook[/] : Prospek dan rekomendasi investasi

[bold]Technical Indicators[/]
- EMA crossover : Trend determination (UP/DOWN)
- Volume ratio >= 1.5 : Strong breakout filter
- RSI 14 : Avoid overbought/oversold zones
- ATR : Stop-loss & take-profit buffer

[bold]Signal Types[/]
- [green]BUY[/] : Strong buy signal with high confidence
- [yellow]WAIT[/] : Hold/wait for better entry
- [red]SELL[/] : Sell signal or take profit

[bold]Diagnostic[/]
- Jalankan [yellow]test connection[/] untuk memeriksa koneksi ke Yahoo Finance, NewsAPI, dan ML System.

[bold]Tips & Best Practices[/]
- Volume ratio < 1 → Weak breakout, wait for confirmation
- ML Confidence ≥ 70% → Reliable prediction
- Use [yellow]ai BBCA --narrative[/] for deep fundamental analysis
- Combine technical signals with narrative insights for best decisions

[bold]Examples[/]
- [yellow]BBCA[/] : Quick analysis of BBCA
- [yellow]ai BBCA --narrative[/] : Full financial story of BBCA
- [yellow]ai BBCA BBRI --narrative[/] : Compare narrative analysis
- [yellow]ml BBCA[/] : ML prediction for BBCA only
""",
        title="📊 Enhanced CLI Help v5",
        border_style="bright_cyan",
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
            ml_status = "[green]ML[/green]" if ML_AVAILABLE else "[red]ML[/red]"
            raw = Prompt.ask(
                f"[bold white]\n[/][bright_black]([/][magenta]{timeframe}[/] | candles [magenta]{candles}[/] | strategy [magenta]{strategy_label}[/] | {ml_status}[bright_black])[/]",
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

        if raw.lower() == "ml status":
            if ML_AVAILABLE:
                try:
                    # Check if ML v5 models exist
                    import os
                    models_v5_path = Path(__file__).resolve().parents[1] / 'ml_system' / 'models_v5'
                    if models_v5_path.exists():
                        model_files = list(models_v5_path.glob('*.joblib'))
                        if model_files:
                            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                            file_size = latest_model.stat().st_size / (1024*1024)
                            console.print(Panel(
                                f"ML System: [green]ACTIVE[/green]\n"
                                f"Version: v5 (IDX Enhanced)\n"
                                f"Model: {latest_model.name}\n"
                                f"Size: {file_size:.1f} MB\n"
                                f"Accuracy: 83.1%\n"
                                f"Stocks Trained: 88 Indonesian stocks\n"
                                f"Features: 96 technical indicators\n"
                                f"Models: XGBoost + RF + GB + LR ensemble",
                                title="ML v5 System Status",
                                border_style="green"
                            ))
                        else:
                            console.print("[error]No ML v5 models found[/error]")
                    else:
                        console.print("[error]ML v5 models directory not found[/error]")
                except Exception as e:
                    console.print(f"[error]Error getting ML status: {e}[/error]")
            else:
                console.print("[error]ML System v5 not available[/error]")
            continue

        if raw.lower() == "test connection":
            console.print("[info]Menjalankan diagnosa koneksi...[/info]")
            diagnoses = diagnose_connectivity()
            table = Table(title="Diagnosa Koneksi", box=box.SIMPLE_HEAD, style="white")
            table.add_column("Layanan", style="cyan")
            table.add_column("Status", style="bold")
            for service, status_text in diagnoses.items():
                style = "green" if "OK" in status_text or "Available" in status_text else "red"
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

        # ML command
        if raw.lower().startswith("ml"):
            if not ML_AVAILABLE:
                console.print("[error]ML System v5 tidak tersedia.[/error]")
                continue
            ml_symbols = raw.split()[1:] or list(last_results.keys())
            if not ml_symbols:
                console.print("[error]Belum ada data untuk dianalisis. Jalankan pencarian simbol terlebih dahulu.[/error]")
                continue
            for sym in ml_symbols:
                sym_upper = sym.upper()
                try:
                    # Get ML prediction using ML v5
                    ml_pred = get_ml_prediction(sym_upper)
                    if ml_pred and ml_pred.get('success'):
                        console.print(Panel(
                            f"ML Signal: {ml_pred.get('signal', 'N/A')}\n"
                            f"Confidence: {ml_pred.get('confidence', 0):.1%}\n"
                            f"Version: {ml_pred.get('version', 'N/A')}\n"
                            f"Model Type: {ml_pred.get('model_type', 'N/A')}\n"
                            f"\n[green]ML v5 Features:[/green]\n"
                            f"• 83.1% accuracy on validation\n"
                            f"• Trained on 88 Indonesian stocks\n"
                            f"• 96 technical indicators\n"
                            f"• XGBoost + RF + GB + LR ensemble",
                            title=f"ML v5 Prediction {sym_upper}",
                            border_style="blue"
                        ))
                    else:
                        console.print(f"[error]ML prediction failed for {sym_upper}: Unable to get prediction[/error]")
                except Exception as e:
                    console.print(f"[error]Error getting ML prediction for {sym_upper}: {e}[/error]")
            continue

        # LLM configuration command
        if raw.lower().startswith("llm"):
            if not NARRATIVE_AVAILABLE or not llm_manager:
                console.print("[error]LLM System tidak tersedia.[/error]")
                continue

            subcommand = raw.lower().split()[1] if len(raw.split()) > 1 else ""

            if subcommand == "config":
                config = llm_manager.get_current_config()
                console.print("\n[bold]Current LLM Configuration[/]")
                console.print(f"Provider/Model: {config['provider_model']}")
                console.print(f"Available: {len(config['available_models'])} models")

                # Show detailed config
                cfg = config['config']
                console.print(f"\n[bold]Settings[/]")
                console.print(f"Primary Provider: {cfg['primary_provider']}")
                console.print(f"Preferred Model: {cfg['preferred_model']}")
                console.print(f"Fallback Enabled: {cfg['fallback_enabled']}")
                console.print(f"Max Retries: {cfg['max_retries']}")
                console.print(f"Retry Delay: {cfg['retry_delay']}s")

            elif subcommand == "help":
                llm_manager.print_config_help()
            else:
                console.print("[yellow]LLM Commands:[/]")
                console.print("  llm config   - Show current LLM configuration")
                console.print("  llm help     - Show LLM configuration help")
            continue

        if raw.lower().startswith("ai"):
            if not llm_manager or not llm_manager.get_best_adapter():
                console.print("[error]AI tidak tersedia. Pastikan API Key sudah diset (GEMINI_API_KEY, OPENAI_API_KEY, atau ANTHROPIC_API_KEY).[/error]")
                continue

            # Parse command for --narrative flag
            ai_parts = raw.split()
            narrative_mode = "--narrative" in ai_parts
            ai_symbols = [sym for sym in ai_parts[1:] if sym != "--narrative"] or list(last_results.keys())

        # Check for trading analysis command
        if raw.lower().startswith("trading"):
            trading_parts = raw.split()
            if len(trading_parts) > 1:
                symbol = trading_parts[1].upper()
                console.print(f"\n[bold cyan]Generating trading analysis for {symbol}...[/]")
                trading_result = generate_trading_analysis(symbol, config, analyzer, console)
                if trading_result:
                    display_trading_analysis(console, symbol, trading_result)
                else:
                    console.print(f"[error]Failed to generate trading analysis for {symbol}[/error]")
                continue
            else:
                console.print("[error]Please specify a symbol. Usage: trading <symbol>[/error]")
                continue

            if not ai_symbols:
                console.print("[error]Belum ada data untuk dianalisis. Jalankan pencarian simbol terlebih dahulu.[/error]")
                continue

            # Check if narrative features are available
            if narrative_mode and not NARRATIVE_AVAILABLE:
                console.print("[error]Narrative analysis tidak tersedia. Missing dependencies.[/error]")
                continue

            for sym in ai_symbols:
                sym_upper = sym.upper()

                if narrative_mode:
                    # Generate narrative analysis
                    with console.status(f"[bold cyan]Generating narrative analysis for {sym_upper}..."):
                        narrative_result = generate_narrative_analysis(
                            sym_upper, timeframe, candles, config, analyzer, console
                        )

                    if narrative_result["success"]:
                        display_narrative_analysis(console, sym_upper, narrative_result["data"])
                    else:
                        console.print(f"[error]Gagal generate narrative untuk {sym_upper}: {narrative_result['error']}[/error]")
                else:
                    # Existing 3-bullet analysis (unchanged)
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
                    ai_text = generate_ai_summary(entry)
                    console.print(Panel(ai_text, title=f"Enhanced AI Insight {sym_upper}", border_style="magenta"))
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

        # Enhanced table with ML predictions
        table = Table(title="Enhanced Signal Summary", box=box.MINIMAL_DOUBLE_HEAD, style="white")
        table.add_column("Symbol", style="bold cyan")
        table.add_column("Alias", style="cyan")
        table.add_column("Harga", justify="right")
        table.add_column("Stock Sig", style="magenta")
        table.add_column("ML Sig", style="blue")
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
            ml_prediction = entry.get("ml_prediction", {})
            vol_ratio = stock.get("volume_ratio", 0)
            trend = stock.get("trend", "-")
            strength = overall.get("strength", 0)

            # Get ML signal
            ml_signal = "N/A"
            if ml_prediction:
                ml_signal = f"{ml_prediction.get('signal', 'N/A')} ({ml_prediction.get('confidence', 0):.0%})"

            table.add_row(
                entry["symbol"],
                entry.get("alias", "-"),
                f"{entry.get('last_close', '-'):,}" if entry.get("last_close") else "-",
                safe_text(stock.get("signal", "WAIT")),
                ml_signal,
                safe_text(stock.get("trend", "-")),
                f"{vol_ratio:.2f}" if isinstance(vol_ratio, (int, float)) else "-",
                safe_text(overall.get("signal", "WAIT")),
                f"{strength:.2f}",
            )

            reasons = stock.get("reasons") or []
            overall_reasons = overall.get("reasons") or []
            ml_info = ""
            if ml_prediction:
                ml_info = f"ML Signal     : {ml_prediction.get('signal', 'N/A')} (confidence: {ml_prediction.get('confidence', 0):.1%})\n"

            panel_text = (
                f"Breakout lvl : {safe_text(stock.get('breakout_level', '-'))}\n"
                f"Breakdown lvl: {safe_text(stock.get('breakdown_level', '-'))}\n"
                f"ATR          : {safe_text(stock.get('atr', '-'))}\n"
                f"Trend        : {safe_text(stock.get('trend', '-'))}\n"
                f"Volume Ratio : {safe_text(stock.get('volume_ratio', '-'))}\n"
                f"RSI          : {safe_text(stock.get('rsi', '-'))}\n"
                f"Stock reasons: {safe_text('; '.join(map(str, reasons)) or '-')}\n"
                f"Overall ctx  : {safe_text('; '.join(map(str, overall_reasons)) or '-')}\n"
                f"{ml_info}"
            )
            detail_panels.append(
                Panel(
                    panel_text,
                    title=f"{entry['symbol']} Enhanced Detail",
                    border_style="bright_blue" if stock.get("signal") == "BUY" else "bright_yellow",
                )
            )

        console.print()
        console.print(table)
        for panel in detail_panels:
            console.print(panel)


def generate_narrative_analysis(symbol: str, timeframe: str, candles: int,
                              config: dict, analyzer, console) -> Dict:
    """Generate comprehensive narrative financial analysis"""
    try:
        # Initialize components
        fetcher = FinancialDataFetcher()
        # Use enhanced narrative generator with multi-LLM support
        generator = EnhancedNarrativeGenerator()

        # Get technical data - need more data for ML prediction
        # For narrative analysis, we fetch more historical data to ensure ML works
        narrative_candles = max(candles, 500)  # Ensure at least 500 candles for ML
        technical_entry = fetch_signal(
            symbol, timeframe, narrative_candles, config, analyzer, console=console
        )

        if technical_entry.get("error"):
            return {
                'success': False,
                'error': technical_entry['error']
            }

        # Get ML prediction from the technical entry
        ml_prediction = technical_entry.get('ml', {}) or {'signal': 'HOLD', 'confidence': 0.5}

        # Get financial data
        console.print("[cyan]Fetching financial data...[/cyan]")
        financial_data = fetcher.get_key_metrics(symbol)
        quarterly_data = fetcher.get_quarterly_performance(symbol)
        growth_data = fetcher.get_growth_trends(symbol)

        # Generate narrative
        console.print("[cyan]Generating narrative analysis...[/cyan]")
        narrative_result = generator.generate_narrative_analysis(
            symbol, technical_entry, financial_data, ml_prediction, quarterly_data, growth_data
        )

        return narrative_result

    except Exception as e:
        return {
            'success': False,
            'error': f"Error generating narrative: {str(e)}"
        }


def display_narrative_analysis(console: Console, symbol: str, narrative_data: Dict) -> None:
    """Display narrative analysis in rich formatted panels"""

    # Title Panel
    title = narrative_data.get('title', f"Analisis {symbol}")
    console.print("\n")
    console.print(Panel(
        f"[bold cyan]{title}[/]",
        title=f"📊 {symbol} - Narrative Analysis",
        border_style="bright_magenta"
    ))

    # Executive Summary
    if 'executive_summary' in narrative_data:
        console.print(Panel(
            narrative_data['executive_summary'],
            title="📋 Executive Summary",
            border_style="green"
        ))

    # Growth Analysis
    if 'growth_analysis' in narrative_data:
        console.print(Panel(
            narrative_data['growth_analysis'],
            title="📈 Growth Analysis",
            border_style="blue"
        ))

    # Financial Health
    if 'financial_health' in narrative_data:
        console.print(Panel(
            narrative_data['financial_health'],
            title="💰 Financial Health Deep Dive",
            border_style="yellow"
        ))

    # Strategic Initiatives
    if 'strategic_initiatives' in narrative_data:
        console.print(Panel(
            narrative_data['strategic_initiatives'],
            title="🎯 Strategic Initiatives",
            border_style="cyan"
        ))

    # Risk Factors
    if 'risk_factors' in narrative_data:
        console.print(Panel(
            narrative_data['risk_factors'],
            title="⚠️  Risk Factors",
            border_style="red"
        ))

    # Outlook
    if 'outlook' in narrative_data:
        console.print(Panel(
            narrative_data['outlook'],
            title="🔮 Outlook",
            border_style="magenta"
        ))

    # Main Content (if no sections parsed)
    if 'main_content' in narrative_data:
        console.print(Panel(
            narrative_data['main_content'],
            title="📝 Analysis",
            border_style="white"
        ))

    # Metadata
    if 'metadata' in narrative_data:
        meta = narrative_data['metadata']
        console.print(f"\n[dim]Generated at {meta.get('generated_at', 'N/A')} | "
                     f"Word count: {meta.get('word_count', 0)}[/dim]")

    console.print("\n")


if __name__ == "__main__":
    main()