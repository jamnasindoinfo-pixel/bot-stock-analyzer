#!/usr/bin/env python3
"""
Automated Stock Analysis Script for Short-Term Trading
Analyzes all stocks and provides top 5 recommendations
"""
import os
import json
import sys
from datetime import datetime
import pandas as pd
import numpy as np
from analyzers.market_analyzer import MarketAnalyzer
from ml_system.cli.EnhancedProductionMLCLI import EnhancedProductionMLCLI
import yfinance as yf

def get_indonesian_stocks():
    """Get list of Indonesian stocks to analyze"""
    return [
        'BBCA.JK',  # Bank Central Asia
        'BBRI.JK',  # Bank BRI
        'BBNI.JK',  # Bank BNI
        'BMRI.JK',  # Bank Mandiri
        'TLKM.JK',  # Telkom Indonesia
        'INDF.JK',  # Indofood Sukses Makmur
        'UNVR.JK',  # Unilever Indonesia
        'ASII.JK',  # Astra International
        'ICBP.JK',  # Indofood CBP Sukses Makmur
        'KLBF.JK'   # Kalbe Farma
    ]

def calculate_short_term_signals(df):
    """Calculate short-term trading signals"""
    if len(df) < 20:
        return None

    # Price indicators
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    price_change = (current_price - prev_price) / prev_price * 100

    # Moving averages
    sma_5 = df['Close'].rolling(5).mean().iloc[-1]
    sma_10 = df['Close'].rolling(10).mean().iloc[-1]
    sma_20 = df['Close'].rolling(20).mean().iloc[-1]

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]

    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9).mean()
    macd_histogram = macd - signal
    current_macd = macd.iloc[-1]
    current_signal = signal.iloc[-1]

    # Volume analysis
    avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
    current_volume = df['Volume'].iloc[-1]
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    bb_middle = df['Close'].rolling(bb_period).mean()
    bb_std_dev = df['Close'].rolling(bb_period).std()
    bb_upper = bb_middle + (bb_std_dev * bb_std)
    bb_lower = bb_middle - (bb_std_dev * bb_std)
    bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

    # Calculate signals
    signals = {
        'price_momentum': 1 if price_change > 0 else -1,
        'sma_trend': 1 if current_price > sma_5 > sma_10 else -1,
        'volume_strength': 1 if volume_ratio > 1.2 else 0,
        'rsi_signal': 1 if 30 < current_rsi < 70 else (0 if current_rsi < 30 else -1),
        'macd_signal': 1 if current_macd > current_signal else -1,
        'bb_signal': 1 if bb_position < 0.2 else (0 if bb_position > 0.8 else -1)
    }

    # Overall score
    total_score = sum(signals.values())

    return {
        'symbol': df['Symbol'].iloc[-1] if 'Symbol' in df.columns else 'UNKNOWN',
        'current_price': current_price,
        'price_change': price_change,
        'rsi': current_rsi,
        'macd': current_macd,
        'volume_ratio': volume_ratio,
        'bb_position': bb_position,
        'signals': signals,
        'total_score': total_score,
        'recommendation': 'STRONG_BUY' if total_score >= 4 else 'BUY' if total_score >= 2 else 'HOLD' if total_score >= 0 else 'SELL'
    }

def analyze_stocks():
    """Analyze all stocks and return recommendations"""
    stocks = get_indonesian_stocks()
    recommendations = []

    print("=" * 60)
    print("SHORT-TERM STOCK ANALYSIS - INDONESIAN MARKET")
    print("=" * 60)
    print(f"Analyzing {len(stocks)} stocks for short-term opportunities...\n")

    for stock in stocks:
        try:
            print(f"Analyzing {stock}...", end=" ")

            # Get stock data
            ticker = yf.Ticker(stock)
            df = ticker.history(period="1mo", interval="1d")

            if df.empty or len(df) < 20:
                print("❌ Insufficient data")
                continue

            # Add symbol to dataframe
            df['Symbol'] = stock

            # Calculate signals
            analysis = calculate_short_term_signals(df)
            if analysis:
                recommendations.append(analysis)

                # Print basic info
                status = "[OK]" if analysis['total_score'] >= 2 else "[!]"
                print(f"{status} Score: {analysis['total_score']}/6 | {analysis['recommendation']} | Price: Rp{analysis['current_price']:,.0f}")

        except Exception as e:
            print(f"[X] Error: {str(e)[:50]}")
            continue

    # Sort by score
    recommendations.sort(key=lambda x: x['total_score'], reverse=True)

    return recommendations

def print_recommendations(recommendations):
    """Print top 5 recommendations with detailed analysis"""
    print("\n" + "=" * 80)
    print("TOP 5 SHORT-TERM STOCK RECOMMENDATIONS")
    print("=" * 80)

    top_5 = recommendations[:5]

    for i, rec in enumerate(top_5, 1):
        print(f"\n{i}. {rec['symbol']} - {rec['recommendation']}")
        print("-" * 50)
        print(f"Current Price: Rp {rec['current_price']:,.0f}")
        print(f"Price Change: {rec['price_change']:+.2f}%")
        print(f"RSI: {rec['rsi']:.1f} ({'Oversold' if rec['rsi'] < 30 else 'Overbought' if rec['rsi'] > 70 else 'Neutral'})")
        print(f"Volume Ratio: {rec['volume_ratio']:.2f}x ({'High' if rec['volume_ratio'] > 1.2 else 'Normal'})")
        print(f"Bollinger Position: {rec['bb_position']:.1%} ({'Lower Band' if rec['bb_position'] < 0.2 else 'Upper Band' if rec['bb_position'] > 0.8 else 'Middle'})")

        print("\nSignal Breakdown:")
        for signal, value in rec['signals'].items():
            icon = "[+]" if value > 0 else "[-]" if value < 0 else "[0]"
            print(f"  {icon} {signal.replace('_', ' ').title()}: {value}")

        print(f"\nOverall Score: {rec['total_score']}/6")

        # Trading suggestion
        if rec['recommendation'] == 'STRONG_BUY':
            suggestion = "Strong buy signal - Multiple indicators aligned. Consider entry with stop-loss at recent support."
        elif rec['recommendation'] == 'BUY':
            suggestion = "Buy signal - Good risk-reward ratio. Enter on dips for better entry."
        else:
            suggestion = "Hold/Wait - Not optimal for entry. Monitor for better setup."

        print(f"Suggestion: {suggestion}")

    if len(recommendations) > 5:
        print(f"\n--- {len(recommendations) - 5} more stocks analyzed ---")
        print("Full analysis available upon request")

    # Market summary
    print("\n" + "=" * 80)
    print("MARKET SUMMARY")
    print("=" * 80)

    buy_signals = sum(1 for r in recommendations if r['recommendation'] in ['STRONG_BUY', 'BUY'])
    total = len(recommendations)

    print(f"Total Analyzed: {total} stocks")
    print(f"Buy Signals: {buy_signals} ({buy_signals/total*100:.1f}%)")
    print(f"Hold/Sell: {total - buy_signals} ({(total-buy_signals)/total*100:.1f}%)")

    if buy_signals > total * 0.6:
        print("\nMarket Sentiment: BULLISH - Good buying opportunities")
    elif buy_signals > total * 0.4:
        print("\nMarket Sentiment: NEUTRAL - Selective buying")
    else:
        print("\nMarket Sentiment: BEARISH - Caution advised")

if __name__ == "__main__":
    try:
        recommendations = analyze_stocks()
        print_recommendations(recommendations)

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"short_term_recommendations_{timestamp}.json", "w") as f:
            json.dump(recommendations, f, indent=2, default=str)

        print(f"\nFull analysis saved to: short_term_recommendations_{timestamp}.json")

    except Exception as e:
        print(f"\nError running analysis: {e}")
        import traceback
        traceback.print_exc()