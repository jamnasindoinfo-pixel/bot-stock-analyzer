#!/usr/bin/env python3
"""
Expanded Stock Analysis Script - Focus on Technical Indicators
Analyzes 30+ Indonesian stocks for short-term trading opportunities
"""
import os
import json
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf

def get_expanded_stock_list():
    """Get expanded list of Indonesian stocks across various sectors"""
    return {
        'Banking': ['BBCA.JK', 'BBRI.JK', 'BBNI.JK', 'BMRI.JK', 'BRIS.JK', 'BNGA.JK', 'MDKA.JK'],
        'Technology': ['TLKM.JK', 'EXCL.JK', 'ISAT.JK', 'FREN.JK', 'BKLA.JK', 'AGIS.JK'],
        'Consumer': ['INDF.JK', 'UNVR.JK', 'ICBP.JK', 'KLBF.JK', 'MYOR.JK', 'KAEF.JK', 'GGRM.JK', 'HMSP.JK'],
        'Infrastructure': ['ASII.JK', 'PTPP.JK', 'WIKA.JK', 'ADHI.JK', 'JSMR.JK', 'TOWR.JK'],
        'Mining': ['ANTM.JK', 'PTBA.JK', 'TINS.JK', 'SMGR.JK', 'INCO.JK'],
        'Property': ['BSDE.JK', 'CTRS.JK', 'PWON.JK', 'CTRA.JK', 'LPKR.JK'],
        'Energy': ['PGAS.JK', 'MEDC.JK', 'Energi.JK', 'HRUM.JK'],
        'Trading': ['ITMG.JK', 'SIDO.JK', 'KBLI.JK', 'DOID.JK']
    }

def calculate_advanced_signals(df):
    """Calculate comprehensive technical signals for short-term trading"""
    if len(df) < 30:
        return None

    # Basic price data
    current_price = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    price_change = (current_price - prev_close) / prev_close * 100

    # Volume analysis
    volume = df['Volume']
    current_volume = volume.iloc[-1]
    avg_volume_20 = volume.rolling(20).mean().iloc[-1]
    avg_volume_5 = volume.rolling(5).mean().iloc[-1]
    volume_surge = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1

    # Moving Averages
    sma_5 = df['Close'].rolling(5).mean()
    sma_10 = df['Close'].rolling(10).mean()
    sma_20 = df['Close'].rolling(20).mean()
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()

    current_sma_5 = sma_5.iloc[-1]
    current_sma_10 = sma_10.iloc[-1]
    current_sma_20 = sma_20.iloc[-1]
    current_ema_12 = ema_12.iloc[-1]
    current_ema_26 = ema_26.iloc[-1]

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    rsi_prev = rsi.iloc[-2]

    # MACD
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9).mean()
    histogram = macd - signal
    current_macd = macd.iloc[-1]
    current_signal = signal.iloc[-1]
    current_histogram = histogram.iloc[-1]
    prev_histogram = histogram.iloc[-2]

    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    bb_middle = df['Close'].rolling(bb_period).mean()
    bb_std_dev = df['Close'].rolling(bb_period).std()
    bb_upper = bb_middle + (bb_std_dev * bb_std)
    bb_lower = bb_middle - (bb_std_dev * bb_std)
    bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

    # Stochastic Oscillator
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    k_percent = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    d_percent = k_percent.rolling(3).mean()
    current_k = k_percent.iloc[-1]
    current_d = d_percent.iloc[-1]

    # Williams %R
    williams_r = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
    current_williams = williams_r.iloc[-1]

    # Price Action Patterns
    # Hammer pattern
    body_size = abs(df['Close'] - df['Open'])
    upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
    lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
    is_hammer = (body_size.iloc[-1] < (df['High'].iloc[-1] - df['Low'].iloc[-1]) * 0.3) and \
                (lower_shadow.iloc[-1] > body_size.iloc[-1] * 2)

    # Engulfing pattern
    prev_bullish = df['Close'].iloc[-2] > df['Open'].iloc[-2]
    current_bearish = df['Close'].iloc[-1] < df['Open'].iloc[-1]
    bearish_engulfing = prev_bullish and current_bearish and \
                       (df['Open'].iloc[-1] > df['Close'].iloc[-2]) and \
                       (df['Close'].iloc[-1] < df['Open'].iloc[-2])

    # Momentum indicators
    momentum_5 = (current_price / df['Close'].iloc[-5] - 1) * 100
    momentum_10 = (current_price / df['Close'].iloc[-10] - 1) * 100

    # Support/Resistance levels
    resistance_20 = df['High'].rolling(20).max().iloc[-1]
    support_20 = df['Low'].rolling(20).min().iloc[-1]
    price_position = (current_price - support_20) / (resistance_20 - support_20)

    # Calculate individual signals
    signals = {}

    # Price momentum signals
    signals['price_momentum_1d'] = 1 if price_change > 1 else (-1 if price_change < -1 else 0)
    signals['price_momentum_5d'] = 1 if momentum_5 > 3 else (-1 if momentum_5 < -3 else 0)
    signals['price_momentum_10d'] = 1 if momentum_10 > 5 else (-1 if momentum_10 < -5 else 0)

    # Moving average signals
    signals['sma_cross'] = 1 if current_price > current_sma_5 > current_sma_10 else (-1 if current_price < current_sma_5 < current_sma_10 else 0)
    signals['ema_trend'] = 1 if current_ema_12 > current_ema_26 else (-1 if current_ema_12 < current_ema_26 else 0)
    signals['ma_support'] = 1 if (current_price > current_sma_20 and abs(current_price - current_sma_20) / current_sma_20 < 0.02) else 0

    # RSI signals
    if current_rsi < 30:
        signals['rsi_oversold'] = 2  # Strong buy
    elif current_rsi < 35:
        signals['rsi_oversold'] = 1  # Buy
    elif current_rsi > 70:
        signals['rsi_oversold'] = -2  # Strong sell
    elif current_rsi > 65:
        signals['rsi_oversold'] = -1  # Sell
    else:
        signals['rsi_oversold'] = 0  # Neutral

    signals['rsi_divergence'] = 1 if (current_rsi > rsi_prev and price_change < 0) else (-1 if (current_rsi < rsi_prev and price_change > 0) else 0)

    # MACD signals
    signals['macd_crossover'] = 1 if (current_macd > current_signal and prev_histogram < 0) else (-1 if (current_macd < current_signal and prev_histogram > 0) else 0)
    signals['macd_momentum'] = 1 if current_histogram > 0 and current_histogram > prev_histogram else (-1 if current_histogram < 0 and current_histogram < prev_histogram else 0)

    # Volume signals
    signals['volume_surge'] = 2 if volume_surge > 2 else (1 if volume_surge > 1.5 else (0 if volume_surge > 0.5 else -1))
    signals['volume_price'] = 1 if (volume_surge > 1.2 and price_change > 0) else (-1 if (volume_surge > 1.2 and price_change < 0) else 0)

    # Bollinger Bands
    if bb_position < 0.1:
        signals['bb_squeeze'] = 2  # Strong buy at lower band
    elif bb_position < 0.2:
        signals['bb_squeeze'] = 1  # Buy near lower band
    elif bb_position > 0.9:
        signals['bb_squeeze'] = -2  # Strong sell at upper band
    elif bb_position > 0.8:
        signals['bb_squeeze'] = -1  # Sell near upper band
    else:
        signals['bb_squeeze'] = 0

    # Stochastic signals
    signals['stoch_oversold'] = 1 if current_k < 20 and current_d < 20 else (-1 if current_k > 80 and current_d > 80 else 0)
    signals['stoch_cross'] = 1 if (current_k > current_d and current_k < 80) else (-1 if (current_k < current_d and current_k > 20) else 0)

    # Williams %R
    signals['williams_oversold'] = 1 if current_williams < -80 else (-1 if current_williams > -20 else 0)

    # Pattern signals
    signals['hammer_pattern'] = 2 if is_hammer else 0
    signals['engulfing_pattern'] = -2 if bearish_engulfing else 0

    # Support/Resistance
    signals['near_support'] = 1 if price_position < 0.2 else (-1 if price_position > 0.8 else 0)
    signals['breaking_resistance'] = 2 if (current_price > resistance_20.iloc[-2] and current_price / resistance_20.iloc[-2] > 1.01) else 0

    # Calculate total score
    total_score = sum(signals.values())
    max_possible = sum(abs(v) for v in signals.values())

    # Normalize score to -10 to 10 range
    normalized_score = (total_score / max_possible) * 10 if max_possible > 0 else 0

    # Determine recommendation
    if normalized_score >= 6:
        recommendation = "STRONG_BUY"
    elif normalized_score >= 3:
        recommendation = "BUY"
    elif normalized_score >= -1:
        recommendation = "HOLD"
    elif normalized_score >= -4:
        recommendation = "SELL"
    else:
        recommendation = "STRONG_SELL"

    # Calculate risk level
    volatility = df['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
    if volatility > 0.4:
        risk_level = "HIGH"
    elif volatility > 0.25:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        'symbol': df['Symbol'].iloc[-1] if 'Symbol' in df.columns else 'UNKNOWN',
        'current_price': current_price,
        'price_change': price_change,
        'momentum_5d': momentum_5,
        'momentum_10d': momentum_10,
        'rsi': current_rsi,
        'macd': current_macd,
        'signal_line': current_signal,
        'volume_ratio': volume_surge,
        'bb_position': bb_position,
        'stoch_k': current_k,
        'stoch_d': current_d,
        'williams_r': current_williams,
        'volatility': volatility,
        'risk_level': risk_level,
        'support_level': support_20.iloc[-1],
        'resistance_level': resistance_20.iloc[-1],
        'signals': signals,
        'total_score': normalized_score,
        'recommendation': recommendation,
        'pattern_hammer': is_hammer,
        'pattern_engulfing': bearish_engulfing
    }

def analyze_expanded_stocks():
    """Analyze expanded list of stocks"""
    stock_dict = get_expanded_stock_list()
    all_recommendations = []
    sector_analysis = {}

    print("=" * 100)
    print("EXPANDED TECHNICAL ANALYSIS - INDONESIAN STOCK MARKET")
    print("=" * 100)
    print(f"Analyzing {sum(len(v) for v in stock_dict.values())} stocks across {len(stock_dict)} sectors...\n")

    for sector, stocks in stock_dict.items():
        print(f"\n{'='*20} {sector.upper()} SECTOR {'='*20}")
        sector_recommendations = []

        for stock in stocks:
            try:
                print(f"Analyzing {stock}...", end=" ")

                # Get stock data
                ticker = yf.Ticker(stock)
                df = ticker.history(period="3mo", interval="1d")

                if df.empty or len(df) < 30:
                    print("Insufficient data")
                    continue

                # Add symbol to dataframe
                df['Symbol'] = stock

                # Calculate signals
                analysis = calculate_advanced_signals(df)
                if analysis:
                    all_recommendations.append(analysis)
                    sector_recommendations.append(analysis)

                    # Print summary
                    rec_icon = "[STRONG BUY]" if analysis['recommendation'] == 'STRONG_BUY' else \
                              "[BUY]" if analysis['recommendation'] == 'BUY' else \
                              "[SELL]" if analysis['recommendation'] == 'SELL' else \
                              "[HOLD]"

                    print(f"{rec_icon} | Score: {analysis['total_score']:+.1f}/10 | Risk: {analysis['risk_level']} | Rp{analysis['current_price']:,.0f}")

            except Exception as e:
                print(f"Error: {str(e)[:50]}")
                continue

        # Sector summary
        if sector_recommendations:
            avg_score = sum(r['total_score'] for r in sector_recommendations) / len(sector_recommendations)
            buy_signals = sum(1 for r in sector_recommendations if r['recommendation'] in ['STRONG_BUY', 'BUY'])
            sector_analysis[sector] = {
                'avg_score': avg_score,
                'buy_signals': buy_signals,
                'total_stocks': len(sector_recommendations)
            }
            print(f"\nSector Avg Score: {avg_score:+.2f} | Buy Signals: {buy_signals}/{len(sector_recommendations)}")

    # Sort all recommendations by score
    all_recommendations.sort(key=lambda x: x['total_score'], reverse=True)

    return all_recommendations, sector_analysis

def print_technical_recommendations(recommendations, sector_analysis):
    """Print top recommendations with technical focus"""
    print("\n" + "=" * 100)
    print("TOP 5 TECHNICAL TRADING RECOMMENDATIONS")
    print("=" * 100)

    top_5 = recommendations[:5]

    for i, rec in enumerate(top_5, 1):
        print(f"\n{i}. {rec['symbol']} - {rec['recommendation']} | Risk: {rec['risk_level']}")
        print("-" * 80)
        print(f"Current Price: Rp {rec['current_price']:,.0f} ({rec['price_change']:+.2f}%)")
        print(f"Support: Rp {rec['support_level']:,.0f} | Resistance: Rp {rec['resistance_level']:,.0f}")
        print(f"Technical Score: {rec['total_score']:+.1f}/10 | Volatility: {rec['volatility']:.1%}")

        print("\nKey Technical Indicators:")
        print(f"  RSI: {rec['rsi']:.1f} ({'Oversold <30' if rec['rsi'] < 30 else 'Overbought >70' if rec['rsi'] > 70 else 'Neutral 30-70'})")
        print(f"  MACD: {rec['macd']:+.4f} vs Signal: {rec['signal_line']:+.4f}")
        print(f"  Volume: {rec['volume_ratio']:.2f}x average ({'Spike' if rec['volume_ratio'] > 1.5 else 'Normal'})")
        print(f"  Bollinger: {rec['bb_position']:.0%} position ({'Lower Band' if rec['bb_position'] < 0.2 else 'Upper Band' if rec['bb_position'] > 0.8 else 'Middle'})")
        print(f"  Stochastic: K={rec['stoch_k']:.0f} D={rec['stoch_d']:.0f}")
        print(f"  Williams %R: {rec['williams_r']:.0f}")

        print("\nTop 5 Signal Breakdown:")
        sorted_signals = sorted(rec['signals'].items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for signal, value in sorted_signals:
            icon = "[++]" if value >= 2 else "[+]" if value > 0 else "[--]" if value <= -2 else "[-]" if value < 0 else "[0]"
            signal_name = signal.replace('_', ' ').title()
            print(f"  {icon} {signal_name}: {value:+d}")

        if rec['pattern_hammer']:
            print("\n[!] HAMMER PATTERN DETECTED - Potential reversal signal")
        if rec['pattern_engulfing']:
            print("\n[!] BEARISH ENGULFING PATTERN - Potential downtrend continuation")

        print(f"\nTrading Strategy:")
        if rec['recommendation'] in ['STRONG_BUY', 'BUY']:
            if rec['risk_level'] == 'HIGH':
                strategy = "High volatility trade - Use tight stop-loss (3-4%). Target 5-8% gain."
            else:
                strategy = "Good risk-reward setup - Entry on dips. Target 3-6% gain with 2-3% stop-loss."
        else:
            strategy = "Avoid long positions - Consider shorting or wait for better setup."
        print(f"  {strategy}")

    # Sector Performance
    print("\n" + "=" * 100)
    print("SECTOR PERFORMANCE SUMMARY")
    print("=" * 100)

    sorted_sectors = sorted(sector_analysis.items(), key=lambda x: x[1]['avg_score'], reverse=True)

    for sector, data in sorted_sectors:
        sentiment = "[BULLISH]" if data['avg_score'] > 2 else "[BEARISH]" if data['avg_score'] < -2 else "[NEUTRAL]"
        print(f"{sector:15} | Score: {data['avg_score']:+.2f} | Buy Signals: {data['buy_signals']}/{data['total_stocks']} {sentiment}")

    # Additional opportunities
    print("\n" + "=" * 100)
    print("ADDITIONAL TRADING OPPORTUNITIES")
    print("=" * 100)

    # Oversold stocks
    oversold = [r for r in recommendations if r['rsi'] < 30][:3]
    if oversold:
        print("\n[OVERSOLD - Potential Bounce Plays]")
        for stock in oversold:
            print(f"  {stock['symbol']}: RSI {stock['rsi']:.1f} | Rp {stock['current_price']:,.0f}")

    # Volume spike stocks
    volume_spike = [r for r in recommendations if r['volume_ratio'] > 2][:3]
    if volume_spike:
        print("\n[VOLUME SPIKE - Breakout Candidates]")
        for stock in volume_spike:
            print(f"  {stock['symbol']}: {stock['volume_ratio']:.1f}x volume | {stock['price_change']:+.2f}%")

    # Near resistance
    near_resistance = [r for r in recommendations if r['bb_position'] > 0.85 and r['price_change'] > 0][:3]
    if near_resistance:
        print("\n[NEAR RESISTANCE - Watch for Breakout]")
        for stock in near_resistance:
            print(f"  {stock['symbol']}: {stock['bb_position']:.0%} BB | Rp {stock['current_price']:,.0f}")

    # Save detailed analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"expanded_technical_analysis_{timestamp}.json", "w") as f:
        json.dump({
            'top_recommendations': top_5,
            'sector_analysis': sector_analysis,
            'all_stocks': recommendations[:20]  # Save top 20
        }, f, indent=2, default=str)

    print(f"\nFull analysis saved to: expanded_technical_analysis_{timestamp}.json")

if __name__ == "__main__":
    try:
        recommendations, sector_analysis = analyze_expanded_stocks()
        print_technical_recommendations(recommendations, sector_analysis)

        print("\n" + "=" * 100)
        print("ANALYSIS COMPLETE")
        print("=" * 100)
        print(f"Total Stocks Analyzed: {len(recommendations)}")
        print(f"Strong Buy: {sum(1 for r in recommendations if r['recommendation'] == 'STRONG_BUY')}")
        print(f"Buy: {sum(1 for r in recommendations if r['recommendation'] == 'BUY')}")
        print(f"Hold: {sum(1 for r in recommendations if r['recommendation'] == 'HOLD')}")
        print(f"Sell: {sum(1 for r in recommendations if r['recommendation'] == 'SELL')}")
        print(f"Strong Sell: {sum(1 for r in recommendations if r['recommendation'] == 'STRONG_SELL')}")

    except Exception as e:
        print(f"\nError running analysis: {e}")
        import traceback
        traceback.print_exc()