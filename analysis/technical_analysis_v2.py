#!/usr/bin/env python3
"""
Technical Analysis v2 - Fixed version
Focus on short-term trading signals with technical indicators
"""
import os
import json
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf

def get_indonesian_stocks_technical():
    """Get list of Indonesian stocks for technical analysis"""
    return {
        'Blue Chip': ['BBCA.JK', 'BBRI.JK', 'BBNI.JK', 'BMRI.JK', 'TLKM.JK', 'UNVR.JK', 'ASII.JK', 'INDF.JK'],
        'Growth': ['EXCL.JK', 'ISAT.JK', 'KLBF.JK', 'ICBP.JK', 'PTPP.JK', 'WIKA.JK', 'WSKT.JK'],
        'Cyclical': ['ANTM.JK', 'PTBA.JK', 'SMGR.JK', 'INCO.JK', 'TINS.JK', 'ADHI.JK'],
        'Value': 'BRIS.JK BTPS.JK BBTN.JK BJBR.JK BDMN.JK BNII.JK BUKK.JK DIRA.JK ELTY.JK EMTK.JK GGRM.JK HMSP.JK ITMG.JK JPFA.JK KAEF.JK KLBF.JK MAPI.JK MCAS.JK MEDC.JK MNCN.JK MYOR.JK PWON.JK SIDO.JK SKBM.JK TBIG.JK TINS.JK TOWR.JK ULTJ.JK'.split(),
        'Momentum': 'DOID.JK GOTO.JK ARTO.JK BUKA.JK DKSH.JK EMTK.JK FORU.JK GOTO.JK HRUM.JK ICBP.JK INAF.JK JPFA.JK KBLI.JK KIOS.JK LABA.JK LPIN.JK MAIN.JK MNCN.JK MORE.JK MYOR.JK PGAS.JK PTBA.JK RALS.JK SAHAM.JK SIMP.JK SMRU.JK SRTG.JK TOWR.JK'.split()
    }

def calculate_technical_signals(df):
    """Calculate comprehensive technical signals"""
    if len(df) < 20:
        return None

    # Price data
    current_price = float(df['Close'].iloc[-1])
    prev_close = float(df['Close'].iloc[-2])
    price_change = (current_price - prev_close) / prev_close * 100

    # Volume
    current_volume = float(df['Volume'].iloc[-1])
    avg_volume = float(df['Volume'].rolling(20).mean().iloc[-1])
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

    # Moving averages
    sma_5 = float(df['Close'].rolling(5).mean().iloc[-1])
    sma_10 = float(df['Close'].rolling(10).mean().iloc[-1])
    sma_20 = float(df['Close'].rolling(20).mean().iloc[-1])
    ema_12 = float(df['Close'].ewm(span=12).mean().iloc[-1])
    ema_26 = float(df['Close'].ewm(span=26).mean().iloc[-1])

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = float(rsi.iloc[-1])

    # MACD
    macd_line = ema_12 - ema_26
    signal_line = df['Close'].ewm(span=9).mean().iloc[-1]  # Simplified
    macd_histogram = macd_line - signal_line

    # Bollinger Bands
    bb_middle = sma_20
    bb_std = float(df['Close'].rolling(20).std().iloc[-1])
    bb_upper = bb_middle + (2 * bb_std)
    bb_lower = bb_middle - (2 * bb_std)
    bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5

    # Stochastic
    low_14 = float(df['Low'].rolling(14).min().iloc[-1])
    high_14 = float(df['High'].rolling(14).max().iloc[-1])
    current_close = float(df['Close'].iloc[-1])
    stoch_k = 100 * ((current_close - low_14) / (high_14 - low_14)) if high_14 != low_14 else 50
    stoch_d = stoch_k  # Simplified

    # Price momentum
    momentum_5 = (current_price / float(df['Close'].iloc[-5]) - 1) * 100 if len(df) > 5 else 0
    momentum_10 = (current_price / float(df['Close'].iloc[-10]) - 1) * 100 if len(df) > 10 else 0

    # Volatility
    returns = df['Close'].pct_change()
    volatility = float(returns.rolling(20).std().iloc[-1] * np.sqrt(252)) if len(returns) > 20 else 0

    # Support/Resistance
    resistance_20 = float(df['High'].rolling(20).max().iloc[-1])
    support_20 = float(df['Low'].rolling(20).min().iloc[-1])

    # Calculate signals
    signals = {}

    # Price momentum signals
    signals['price_1d'] = 1 if price_change > 0.5 else (-1 if price_change < -0.5 else 0)
    signals['price_5d'] = 1 if momentum_5 > 2 else (-1 if momentum_5 < -2 else 0)
    signals['price_10d'] = 1 if momentum_10 > 3 else (-1 if momentum_10 < -3 else 0)

    # Moving averages
    signals['sma_trend'] = 1 if (current_price > sma_5 > sma_10) else (-1 if (current_price < sma_5 < sma_10) else 0)
    signals['ema_cross'] = 1 if ema_12 > ema_26 else (-1 if ema_12 < ema_26 else 0)
    signals['price_vs_sma20'] = 1 if current_price > sma_20 else (-1 if current_price < sma_20 else 0)

    # RSI signals
    if current_rsi < 30:
        signals['rsi'] = 2
    elif current_rsi < 35:
        signals['rsi'] = 1
    elif current_rsi > 70:
        signals['rsi'] = -2
    elif current_rsi > 65:
        signals['rsi'] = -1
    else:
        signals['rsi'] = 0

    # MACD signals
    signals['macd'] = 1 if macd_line > signal_line else (-1 if macd_line < signal_line else 0)

    # Volume signals
    signals['volume'] = 2 if volume_ratio > 2 else (1 if volume_ratio > 1.5 else (-1 if volume_ratio < 0.5 else 0))

    # Bollinger Bands
    if bb_position < 0.2:
        signals['bollinger'] = 2
    elif bb_position < 0.4:
        signals['bollinger'] = 1
    elif bb_position > 0.8:
        signals['bollinger'] = -2
    elif bb_position > 0.6:
        signals['bollinger'] = -1
    else:
        signals['bollinger'] = 0

    # Stochastic
    signals['stochastic'] = 1 if stoch_k < 20 else (-1 if stoch_k > 80 else 0)

    # Support/Resistance
    signals['support_resistance'] = 1 if (current_price - support_20) < (resistance_20 - current_price) * 0.3 else 0

    # Total score
    total_score = sum(signals.values())
    max_score = len(signals) * 2
    normalized_score = (total_score / max_score) * 10

    # Recommendation
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

    # Risk level
    if volatility > 0.4:
        risk = "HIGH"
    elif volatility > 0.25:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        'symbol': df['Symbol'].iloc[-1] if 'Symbol' in df.columns else 'UNKNOWN',
        'price': current_price,
        'change': price_change,
        'rsi': current_rsi,
        'volume_ratio': volume_ratio,
        'bb_position': bb_position,
        'momentum_5d': momentum_5,
        'momentum_10d': momentum_10,
        'volatility': volatility,
        'risk': risk,
        'support': support_20,
        'resistance': resistance_20,
        'signals': signals,
        'score': normalized_score,
        'recommendation': recommendation,
        'sma_5': sma_5,
        'sma_20': sma_20
    }

def analyze_all_stocks():
    """Analyze all stocks from different categories"""
    all_stocks = []
    stock_dict = get_indonesian_stocks_technical()

    # Create a comprehensive list
    comprehensive_list = []
    for category, stocks in stock_dict.items():
        if isinstance(stocks, list):
            comprehensive_list.extend(stocks)
        else:
            comprehensive_list.extend(stocks[:5])  # Limit value stocks to first 5

    # Remove duplicates
    comprehensive_list = list(set(comprehensive_list))

    print("=" * 80)
    print("COMPREHENSIVE TECHNICAL ANALYSIS")
    print("=" * 80)
    print(f"Analyzing {len(comprehensive_list)} stocks...\n")

    for i, stock in enumerate(comprehensive_list[:30], 1):  # Analyze first 30 to avoid timeouts
        try:
            print(f"[{i:2d}/30] {stock}...", end=" ")

            ticker = yf.Ticker(stock)
            df = ticker.history(period="2mo", interval="1d")

            if df.empty or len(df) < 20:
                print("No data")
                continue

            df['Symbol'] = stock
            analysis = calculate_technical_signals(df)

            if analysis:
                all_stocks.append(analysis)

                # Quick status
                if analysis['recommendation'] in ['STRONG_BUY', 'BUY']:
                    status = f"[BUY] Score:{analysis['score']:+.1f}"
                else:
                    status = f"[{analysis['recommendation']}] Score:{analysis['score']:+.1f}"
                print(f"{status} Rp{analysis['price']:,.0f}")

        except Exception as e:
            print(f"Error")
            continue

    # Sort by score
    all_stocks.sort(key=lambda x: x['score'], reverse=True)
    return all_stocks

def display_top_recommendations(stocks):
    """Display top 5 recommendations with detailed analysis"""
    print("\n" + "=" * 80)
    print("TOP 5 SHORT-TERM TECHNICAL TRADING RECOMMENDATIONS")
    print("=" * 80)

    top_5 = stocks[:5]

    for i, stock in enumerate(top_5, 1):
        print(f"\n{i}. {stock['symbol']} - {stock['recommendation']} (Risk: {stock['risk']})")
        print("-" * 60)
        print(f"Price: Rp {stock['price']:,.0f} ({stock['change']:+.2f}%)")
        print(f"Support: Rp {stock['support']:,.0f} | Resistance: Rp {stock['resistance']:,.0f}")
        print(f"Technical Score: {stock['score']:+.1f}/10 | Volatility: {stock['volatility']:.0%}")

        print("\nKey Indicators:")
        print(f"  RSI: {stock['rsi']:.1f} ({'Oversold' if stock['rsi'] < 30 else 'Overbought' if stock['rsi'] > 70 else 'Neutral'})")
        print(f"  Volume: {stock['volume_ratio']:.1f}x average ({'High' if stock['volume_ratio'] > 1.5 else 'Normal'})")
        print(f"  Bollinger: {stock['bb_position']:.0%} position ({'Lower' if stock['bb_position'] < 0.3 else 'Upper' if stock['bb_position'] > 0.7 else 'Middle'})")
        print(f"  5D Momentum: {stock['momentum_5d']:+.1f}% | 10D Momentum: {stock['momentum_10d']:+.1f}%")
        print(f"  SMA: Price {stock['sma_5']:,.0f} vs SMA5 {stock['sma_5']:,.0f} vs SMA20 {stock['sma_20']:,.0f}")

        print("\nSignal Strength:")
        sorted_signals = sorted(stock['signals'].items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for signal, value in sorted_signals:
            strength = "+++" if value >= 2 else "++" if value > 0 else "--" if value < 0 else "---" if value <= -2 else "0"
            print(f"  {strength:3} {signal.replace('_', ' ').title():20}: {value:+d}")

        # Trading strategy
        print(f"\nTrading Strategy:")
        if stock['recommendation'] == 'STRONG_BUY':
            print(f"  Multiple bullish signals aligned - Consider immediate entry")
            print(f"  Entry: Rp {stock['price']:,.0f} | Target: Rp {stock['resistance']:,.0f} ({((stock['resistance']/stock['price']-1)*100):+.0f}%)")
            print(f"  Stop Loss: Rp {stock['support']:,.0f} ({((stock['support']/stock['price']-1)*100):+.0f}%)")
        elif stock['recommendation'] == 'BUY':
            print(f"  Good setup - Wait for slight dip if possible")
            print(f"  Entry Zone: Rp {stock['price']*0.98:,.0f} - Rp {stock['price']:,.0f}")
            print(f"  Target: Rp {stock['resistance']:,.0f}")
        else:
            print(f"  Not optimal for buying - Consider shorting or wait")

    # Market overview
    print("\n" + "=" * 80)
    print("MARKET OVERVIEW")
    print("=" * 80)

    buy_count = sum(1 for s in stocks if s['recommendation'] in ['STRONG_BUY', 'BUY'])
    hold_count = sum(1 for s in stocks if s['recommendation'] == 'HOLD')
    sell_count = sum(1 for s in stocks if s['recommendation'] in ['SELL', 'STRONG_SELL'])

    print(f"Buy Signals: {buy_count} | Hold: {hold_count} | Sell: {sell_count}")
    print(f"Average Score: {np.mean([s['score'] for s in stocks]):+.2f}")

    # Best performers by category
    print("\nBest Performers:")

    # Most oversold (bounce candidates)
    oversold = sorted([s for s in stocks if s['rsi'] < 35], key=lambda x: x['rsi'])[:3]
    if oversold:
        print("\n  Oversold (Bounce Candidates):")
        for s in oversold:
            print(f"    {s['symbol']}: RSI {s['rsi']:.1f} at Rp {s['price']:,.0f}")

    # Volume spikes
    volume_spikes = sorted([s for s in stocks if s['volume_ratio'] > 1.5], key=lambda x: x['volume_ratio'], reverse=True)[:3]
    if volume_spikes:
        print("\n  Volume Spikes (Breakout Candidates):")
        for s in volume_spikes:
            print(f"    {s['symbol']}: {s['volume_ratio']:.1f}x volume at Rp {s['price']:,.0f}")

    # Strong momentum
    momentum = sorted([s for s in stocks if s['momentum_5d'] > 2], key=lambda x: x['momentum_5d'], reverse=True)[:3]
    if momentum:
        print("\n  Strong Momentum (Trend Following):")
        for s in momentum:
            print(f"    {s['symbol']}: {s['momentum_5d']:+.1f}% (5D) at Rp {s['price']:,.0f}")

    return top_5

if __name__ == "__main__":
    try:
        all_stocks = analyze_all_stocks()
        if all_stocks:
            top_recommendations = display_top_recommendations(all_stocks)

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"technical_analysis_results_{timestamp}.json"
            with open(filename, "w") as f:
                json.dump({
                    'top_5': top_recommendations,
                    'all_analyzed': all_stocks[:15],  # Save top 15
                    'analysis_time': datetime.now().isoformat()
                }, f, indent=2, default=str)

            print(f"\nAnalysis saved to: {filename}")
        else:
            print("\nNo stocks analyzed successfully")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()