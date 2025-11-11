import os
import json
import time
import threading
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import requests
import google.generativeai as genai
from analyzers.market_analyzer import MarketAnalyzer
from ml_system.cli.EnhancedProductionMLCLI import EnhancedProductionMLCLI
import yfinance as yf


# --------------------------
# 1. LOAD KONFIG & ENV
# --------------------------
def load_environment():
    """
    Load .env and return a dict with multiple key formats to avoid
    mismatches between UPPER and snake_case usages in the code.
    """
    load_dotenv()  # load .env into os.environ

    # Basic presence debug (masked) - will not print the full key
    raw_key = os.getenv("GEMINI_API_KEY") or os.getenv("gemini_api_key")
    if raw_key:
        masked = raw_key[:4] + "..." + raw_key[-4:] if len(raw_key) > 8 else "****"
        print(f"\n[GEMINI_API_KEY found in environment (masked): {masked}]")
    else:
        print("\n[WARNING] GEMINI_API_KEY not found in environment variables")

    env = {
        # Remove MT5 related keys
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "NEWS_API_KEY": os.getenv("NEWS_API_KEY"),
        "TRADING_ECONOMICS_KEY": os.getenv("TRADING_ECONOMICS_KEY", "guest:guest"),

        # snake_case keys
        "gemini_api_key": os.getenv("GEMINI_API_KEY"),
        "news_api_key": os.getenv("NEWS_API_KEY"),
        "trading_economics_key": os.getenv("TRADING_ECONOMICS_KEY", "guest:guest"),
    }

    # Remove None values to avoid TypeErrors later
    return {k: v for k, v in env.items() if v is not None}


def test_gemini_connection():
    """Test basic Gemini connection."""
    try:
        genai.configure(api_key=env["gemini_api_key"])
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content("Test")
        if response.text:
            return True
    except Exception as e:
        print(f"\n[WARNING] Gemini API Error: {e}")
        print("\n[INFO] Solusi umum:")
        print("1. Periksa API key di .env atau Google AI Studio")
        print("2. API key valid dan aktif")
        print("3. Package google-generativeai terinstall versi terbaru")
        print("4. Koneksi internet stabil")
    return None

# Inisialisasi global (akan diinisialisasi di main())
gemini_client = None

# --------------------------
# 2. YAHOO FINANCE DATA (Replacement for MT5)
# --------------------------
env = load_environment()


def get_yfinance_symbols():
    """Get list of Indonesian stock symbols"""
    # Common Indonesian stocks
    symbols = [
        "BBCA.JK",  # Bank Central Asia
        "BBRI.JK",  # Bank Rakyat Indonesia
        "BBNI.JK",  # Bank Negara Indonesia
        "BMRI.JK",  # Bank Mandiri
        "TLKM.JK",  # Telkom Indonesia
        "INDF.JK",  # Indofood Sukses Makmur
        "UNVR.JK",  # Unilever Indonesia
        "ASII.JK",  # Astra International
        "ICBP.JK",  # Indofood CBP Sukses Makmur
        "KLBF.JK",  # Kalbe Farma
    ]
    return symbols


def fetch_yfinance_data(symbol, period="1mo", interval="1d"):
    """Fetch data from Yahoo Finance"""
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        if not data.empty:
            return data
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
    return None


def get_current_price(symbol):
    """Get current price for a symbol"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d", interval="1m")
        if not data.empty:
            return data['Close'].iloc[-1]
    except Exception as e:
        print(f"Error getting price for {symbol}: {e}")
    return None


# --------------------------
# 3. GEMINI AI ANALYSIS
# --------------------------
def analyze_with_gemini(symbol: str, data: pd.DataFrame, timeframe: str = "1D") -> dict:
    """
    Analisis market dengan Gemini AI
    """
    try:
        # Setup client
        genai.configure(api_key=env["gemini_api_key"])
        model = genai.GenerativeModel("gemini-pro")

        # Format data for prompt
        latest_data = data.iloc[-5:]  # Last 5 rows
        prompt = f"""
        Sebagai ahli market Indonesia, analisis data untuk {symbol}:

        Data Terbaru ({timeframe}):
        {latest_data.to_string()}

        Berikan analisis:
        1. Trend saat ini (naik/turun/sideways)
        2. Support & resistance levels
        3. Rekomendasi (BUY/SELL/HOLD)
        4. Risk level (1-10)
        5. Target harga

        Format response sebagai JSON:
        {{
            "trend": "...",
            "support": "...",
            "resistance": "...",
            "recommendation": "BUY/SELL/HOLD",
            "risk_level": "...",
            "target_price": "...",
            "analysis": "..."
        }}
        """

        response = model.generate_content(prompt)

        # Try to parse JSON from response
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                result['symbol'] = symbol
                result['timestamp'] = datetime.now().isoformat()
                return result
        except:
            pass

        # Fallback if JSON parsing fails
        return {
            "symbol": symbol,
            "trend": "Tidak dapat menganalisis",
            "recommendation": "HOLD",
            "risk_level": "5",
            "target_price": "N/A",
            "analysis": response.text[:500],
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"\n[ERROR] Gemini error: {str(e)}")
        return {
            "symbol": symbol,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# --------------------------
# 4. ML PREDICTIONS
# --------------------------
def get_ml_predictions(symbols):
    """Get ML predictions for symbols"""
    try:
        ml_cli = EnhancedProductionMLCLI()
        results = ml_cli.predict_signals(symbols, verbose=True)
        return results
    except Exception as e:
        print(f"ML prediction error: {e}")
        return []


# --------------------------
# 5. MAIN PROGRAM
# --------------------------
def main():
    print("\n" + "=" * 60)
    print("BOT STOCK MARKET - NO MT5 VERSION")
    print("   Analisis Saham Indonesia dengan AI & Machine Learning")
    print("=" * 60)

    # Test Gemini connection
    print("\n[1] Mengecek koneksi Gemini AI...")
    if test_gemini_connection():
        print("[SUCCESS] Gemini AI terhubung")
        gemini_client = "connected"
    else:
        print("[ERROR] Gemini AI gagal terhubung")
        gemini_client = None

    # Get available symbols
    print("\n[2] Mendapatkan daftar saham...")
    symbols = get_yfinance_symbols()
    print(f"[INFO] {len(symbols)} saham tersedia:")
    for i, sym in enumerate(symbols[:5], 1):
        price = get_current_price(sym)
        print(f"   {i}. {sym}: Rp {price:,.0f}" if price else f"   {i}. {sym}: Data tidak tersedia")
    if len(symbols) > 5:
        print(f"   ... dan {len(symbols)-5} saham lainnya")

    # Get user input
    print("\n[3] Pilih saham untuk analisis:")
    for i, sym in enumerate(symbols, 1):
        print(f"   {i}. {sym}")

    try:
        choice = input("\nMasukkan nomor saham (1-10): ").strip()
        if not choice.isdigit():
            print("[ERROR] Input tidak valid")
            return

        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(symbols):
            selected_symbol = symbols[choice_idx]
            print(f"\n[SELECTED] Memilih: {selected_symbol}")
        else:
            print("[ERROR] Pilihan tidak valid")
            return
    except KeyboardInterrupt:
        print("\n\n[STOPPED] Program dihentikan user")
        return

    # Fetch data
    print(f"\n[4] Mengambil data {selected_symbol}...")
    data = fetch_yfinance_data(selected_symbol, period="1mo")

    if data is None or data.empty:
        print(f"[ERROR] Gagal mengambil data untuk {selected_symbol}")
        return

    print(f"[SUCCESS] Data berhasil diambil ({len(data)} bar)")
    print(f"   Range: {data.index[0].date()} hingga {data.index[-1].date()}")
    print(f"   Harga terakhir: Rp {data['Close'].iloc[-1]:,.0f}")

    # ML Analysis
    print(f"\n[5] Analisis Machine Learning...")
    ml_results = get_ml_predictions([selected_symbol.replace('.JK', '')])
    if ml_results and len(ml_results) > 0:
        result = ml_results[0]
        print(f"\n[ML PREDICTION RESULTS]")
        print(f"   Signal: {result.get('signal', 'N/A')}")
        print(f"   Confidence: {result.get('confidence', 0):.2%}")
        print(f"   Current Price: Rp {result.get('current_price', 0):,.0f}")

        individual = result.get('individual_predictions', {})
        if individual:
            print(f"\n   Individual Models:")
            for model, pred in individual.items():
                if pred:
                    print(f"   - {model}: {pred.get('signal', 'N/A')} ({pred.get('confidence', 0):.2%})")

    # Gemini AI Analysis
    if gemini_client:
        print(f"\n[6] Analisis dengan Gemini AI...")
        analysis = analyze_with_gemini(selected_symbol, data)

        if 'error' not in analysis:
            print(f"\n[AI ANALYSIS RESULTS]")
            print(f"   Trend: {analysis.get('trend', 'N/A')}")
            print(f"   Recommendation: {analysis.get('recommendation', 'N/A')}")
            print(f"   Risk Level: {analysis.get('risk_level', 'N/A')}/10")
            print(f"   Target Price: {analysis.get('target_price', 'N/A')}")
            print(f"   Support: {analysis.get('support', 'N/A')}")
            print(f"   Resistance: {analysis.get('resistance', 'N/A')}")
            print(f"\n   Analysis:\n   {analysis.get('analysis', 'N/A')[:300]}...")
        else:
            print(f"\n[ERROR] AI Analysis gagal: {analysis.get('error', 'Unknown error')}")

    # Market Analyzer
    try:
        print(f"\n[7] Analisis teknikal dengan Market Analyzer...")
        analyzer = MarketAnalyzer()
        tech_analysis = analyzer.analyze(selected_symbol, data)

        if tech_analysis:
            print(f"\n[TECHNICAL ANALYSIS]")
            for indicator, value in tech_analysis.items():
                if isinstance(value, (int, float)):
                    print(f"   {indicator}: {value:.2f}")
                else:
                    print(f"   {indicator}: {value}")
    except Exception as e:
        print(f"[WARNING] Technical analysis error: {e}")

    # Summary
    print(f"\n" + "=" * 60)
    print(f"[SUMMARY FOR {selected_symbol}]")
    print("=" * 60)

    if data is not None and not data.empty:
        latest = data.iloc[-1]
        prev_close = data['Close'].iloc[-2] if len(data) > 1 else latest['Close']
        change = latest['Close'] - prev_close
        change_pct = (change / prev_close) * 100

        print(f"Current Price:   Rp {latest['Close']:,.0f}")
        print(f"Change:          {change:+,.0f} ({change_pct:+.2f}%)")
        print(f"Volume:          {latest['Volume']:,.0f}")
        print(f"High:            Rp {latest['High']:,.0f}")
        print(f"Low:             Rp {latest['Low']:,.0f}")
        print(f"Open:            Rp {latest['Open']:,.0f}")

    print(f"\nNext Steps:")
    print(f"1. Monitor price action")
    print(f"2. Set stop loss at support level")
    print(f"3. Consider risk/reward ratio")
    print(f"4. Follow your trading strategy")

    print("\n[SUCCESS] Analisis selesai! Happy trading!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[STOPPED] Program dihentikan user")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()