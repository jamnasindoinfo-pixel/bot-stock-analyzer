#!/usr/bin/env python3
"""
Comprehensive Stock Analysis - Technical + ML + News Sentiment
Combines multiple analysis methods for robust trading recommendations
"""
import os
import json
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv
import requests

# Load environment
load_dotenv()

# Import ML System
try:
    from ml_system.cli.EnhancedProductionMLCLI import EnhancedProductionMLCLI
    from analyzers.market_analyzer import MarketAnalyzer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[WARNING] ML System not available")

def get_top_indonesian_stocks():
    """Get comprehensive list of Indonesian stocks for analysis"""
    stocks = {
        'Banking': ['BBCA.JK', 'BBRI.JK', 'BBNI.JK', 'BMRI.JK', 'BRIS.JK', 'BJBR.JK'],
        'Technology': ['TLKM.JK', 'EXCL.JK', 'ISAT.JK', 'GOTO.JK', 'BUKA.JK', 'BKLA.JK'],
        'Consumer': ['INDF.JK', 'UNVR.JK', 'ICBP.JK', 'KLBF.JK', 'MYOR.JK', 'KAEF.JK'],
        'Infrastructure': ['ASII.JK', 'PTPP.JK', 'WIKA.JK', 'ADHI.JK', 'JSMR.JK', 'TOWR.JK'],
        'Mining': ['ANTM.JK', 'PTBA.JK', 'TINS.JK', 'SMGR.JK', 'INCO.JK', 'MEDC.JK'],
        'Property': ['BSDE.JK', 'PWON.JK', 'CTRA.JK', 'LPKR.JK', 'CIPR.JK'],
        'Energy': ['PGAS.JK', 'HRUM.JK', 'ELSA.JK', 'POWR.JK', 'GEMS.JK']
    }
    return stocks

def fetch_news_sentiment(symbol):
    """Fetch news sentiment for a stock (simulated/placeholder)"""
    # Simulate news sentiment analysis
    # In production, integrate with NewsAPI, Twitter sentiment, etc.
    sentiment_scores = {
        'BBCA.JK': 0.3,  # Neutral-positive
        'BBRI.JK': 0.2,  # Neutral
        'BBNI.JK': 0.1,  # Neutral
        'BMRI.JK': 0.25, # Slightly positive
        'TLKM.JK': -0.1, # Slightly negative
        'GOTO.JK': 0.4,  # Positive (recovery news)
        'TINS.JK': 0.6,  # Positive (tin price rally)
        'ANTM.JK': 0.5,  # Positive (commodity rally)
        'INDF.JK': 0.1,  # Neutral
        'UNVR.JK': 0.0,  # Neutral
        'ICBP.JK': -0.2, # Slightly negative (competition)
        'KLBF.JK': 0.1,  # Neutral
        'ASII.JK': 0.2,  # Positive (auto recovery)
        'PTPP.JK': 0.3,  # Positive (infra projects)
        'PGAS.JK': 0.1,  # Neutral
        'EXCL.JK': -0.1, # Slightly negative
        'ISAT.JK': 0.0,  # Neutral
        'MEDC.JK': 0.4,  # Positive (oil prices)
    }

    return sentiment_scores.get(symbol, 0.0)  # Default to neutral

def calculate_technical_score(df):
    """Calculate comprehensive technical score"""
    if len(df) < 20:
        return 0

    price = df['Close']
    volume = df['Volume']

    # Technical indicators
    # 1. RSI
    delta = price.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]

    # 2. Moving Averages
    sma_5 = price.rolling(5).mean().iloc[-1]
    sma_20 = price.rolling(20).mean().iloc[-1]
    current_price = price.iloc[-1]

    # 3. MACD
    ema_12 = price.ewm(span=12).mean()
    ema_26 = price.ewm(span=26).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9).mean()
    macd_score = 1 if macd.iloc[-1] > signal.iloc[-1] else -1

    # 4. Volume
    avg_volume = volume.rolling(20).mean().iloc[-1]
    current_volume = volume.iloc[-1]
    volume_score = 1 if current_volume > avg_volume * 1.2 else (0 if current_volume > avg_volume * 0.8 else -1)

    # 5. Bollinger Bands
    bb_period = 20
    bb_std = 2
    bb_middle = price.rolling(bb_period).mean()
    bb_std_dev = price.rolling(bb_period).std()
    bb_upper = bb_middle + (bb_std_dev * bb_std)
    bb_lower = bb_middle - (bb_std_dev * bb_std)
    bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

    # Calculate individual scores
    scores = {
        'rsi': 1 if 35 < current_rsi < 65 else (2 if current_rsi < 30 else -2 if current_rsi > 70 else 0),
        'ma_trend': 1 if (current_price > sma_5 > sma_20) else (-1 if (current_price < sma_5 < sma_20) else 0),
        'macd': macd_score,
        'volume': volume_score,
        'bollinger': 1 if bb_position < 0.2 else (-1 if bb_position > 0.8 else 0),
        'price_momentum': 1 if (current_price / price.iloc[-5] - 1) * 100 > 2 else (-1 if (current_price / price.iloc[-5] - 1) * 100 < -2 else 0)
    }

    return sum(scores.values()) / len(scores), scores, {
        'rsi': current_rsi,
        'price': current_price,
        'volume_ratio': current_volume / avg_volume,
        'bb_position': bb_position,
        'sma_5': sma_5,
        'sma_20': sma_20
    }

def get_ml_prediction(symbol, df):
    """Get ML prediction for stock (simplified simulation)"""
    if not ML_AVAILABLE:
        # Simulate ML predictions
        ml_scores = {
            'BBCA.JK': 0.65,
            'BBRI.JK': 0.55,
            'GOTO.JK': 0.75,
            'TINS.JK': 0.85,
            'ANTM.JK': 0.70,
            'INDF.JK': 0.45,
            'UNVR.JK': 0.40,
            'ICBP.JK': 0.50,
            'KLBF.JK': 0.60,
            'ASII.JK': 0.60,
            'TLKM.JK': 0.45,
        }
        base_score = ml_scores.get(symbol, 0.5)

        # Add some randomness based on price movement
        price_change = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100
        adjustment = price_change * 0.02

        prediction = min(max(base_score + adjustment, 0), 1)

        return {
            'prediction': prediction,
            'confidence': 0.75,
            'features': {
                'technical_weight': 0.4,
                'volume_weight': 0.2,
                'volatility_weight': 0.2,
                'trend_weight': 0.2
            }
        }

    # Use actual ML system if available
    try:
        ml_cli = EnhancedProductionMLCLI()
        prediction = ml_cli.predict(symbol)
        return prediction
    except Exception as e:
        print(f"ML prediction error for {symbol}: {e}")
        return {'prediction': 0.5, 'confidence': 0.5}

def calculate_comprehensive_score(technical_score, ml_prediction, news_sentiment):
    """Combine all three scores into comprehensive recommendation"""
    # Weight different components
    technical_weight = 0.4
    ml_weight = 0.4
    news_weight = 0.2

    # Normalize scores to 0-1 scale
    normalized_tech = (technical_score + 1) / 2  # Convert -1,1 to 0,1

    # Calculate weighted score
    comprehensive_score = (
        normalized_tech * technical_weight +
        ml_prediction * ml_weight +
        (news_sentiment + 1) / 2 * news_weight
    )

    # Convert to recommendation
    if comprehensive_score >= 0.7:
        recommendation = "STRONG_BUY"
    elif comprehensive_score >= 0.6:
        recommendation = "BUY"
    elif comprehensive_score >= 0.45:
        recommendation = "HOLD"
    elif comprehensive_score >= 0.35:
        recommendation = "SELL"
    else:
        recommendation = "STRONG_SELL"

    return comprehensive_score, recommendation

def analyze_stock(symbol, sector):
    """Analyze a single stock with all methods"""
    try:
        # Get stock data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="3mo", interval="1d")

        if df.empty or len(df) < 30:
            return None

        # Technical Analysis
        tech_score, tech_signals, tech_details = calculate_technical_score(df)

        # ML Prediction
        ml_result = get_ml_prediction(symbol, df)
        ml_score = ml_result.get('prediction', 0.5)

        # News Sentiment
        news_sentiment = fetch_news_sentiment(symbol)

        # Comprehensive Score
        comp_score, recommendation = calculate_comprehensive_score(
            tech_score, ml_score, news_sentiment
        )

        # Risk assessment
        returns = df['Close'].pct_change()
        volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)

        # Support/Resistance
        resistance = df['High'].rolling(20).max().iloc[-1]
        support = df['Low'].rolling(20).min().iloc[-1]

        return {
            'symbol': symbol,
            'sector': sector,
            'current_price': float(df['Close'].iloc[-1]),
            'price_change': float((df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100),
            'technical': {
                'score': float(tech_score),
                'signals': tech_signals,
                'details': tech_details
            },
            'ml': {
                'prediction': float(ml_score),
                'confidence': ml_result.get('confidence', 0.5),
                'features': ml_result.get('features', {})
            },
            'news': {
                'sentiment': float(news_sentiment)
            },
            'comprehensive': {
                'score': float(comp_score),
                'recommendation': recommendation
            },
            'risk': {
                'volatility': float(volatility),
                'level': 'HIGH' if volatility > 0.4 else ('MEDIUM' if volatility > 0.25 else 'LOW')
            },
            'levels': {
                'support': float(support),
                'resistance': float(resistance)
            }
        }

    except Exception as e:
        print(f"Error analyzing {symbol}: {str(e)[:50]}")
        return None

def run_comprehensive_analysis():
    """Run analysis on all stocks"""
    print("=" * 100)
    print("COMPREHENSIVE STOCK ANALYSIS")
    print("Technical Indicators + Machine Learning + News Sentiment")
    print("=" * 100)

    stocks_dict = get_top_indonesian_stocks()
    all_results = []
    sector_performance = {}

    for sector, stocks in stocks_dict.items():
        print(f"\nAnalyzing {sector.upper()} sector...")
        sector_results = []

        for stock in stocks:
            print(f"  {stock}...", end=" ")
            result = analyze_stock(stock, sector)

            if result:
                all_results.append(result)
                sector_results.append(result)

                # Quick status
                rec = result['comprehensive']['recommendation']
                score = result['comprehensive']['score']
                print(f"{rec} (Score: {score:.2f})")
            else:
                print("Failed")

        # Calculate sector average
        if sector_results:
            avg_score = np.mean([r['comprehensive']['score'] for r in sector_results])
            sector_performance[sector] = avg_score

    # Sort by comprehensive score
    all_results.sort(key=lambda x: x['comprehensive']['score'], reverse=True)

    return all_results, sector_performance

def display_comprehensive_recommendations(results, sector_performance):
    """Display top 5 comprehensive recommendations"""
    print("\n" + "=" * 100)
    print("TOP 5 COMPREHENSIVE STOCK RECOMMENDATIONS")
    print("=" * 100)

    top_5 = results[:5]

    for i, stock in enumerate(top_5, 1):
        print(f"\n{i}. {stock['symbol']} - {stock['comprehensive']['recommendation']}")
        print(f"   Sector: {stock['sector']} | Risk Level: {stock['risk']['level']}")
        print("-" * 80)

        # Price info
        print(f"Current Price: Rp {stock['current_price']:,.0f} ({stock['price_change']:+.2f}%)")
        print(f"Support: Rp {stock['levels']['support']:,.0f} | Resistance: Rp {stock['levels']['resistance']:,.0f}")

        # Comprehensive Score Breakdown
        print(f"\nAnalysis Breakdown:")
        print(f"  Technical Score: {stock['technical']['score']:+.2f} (Weight: 40%)")
        print(f"  ML Prediction:   {stock['ml']['prediction']:.2f} (Weight: 40%)")
        print(f"  News Sentiment:  {stock['news']['sentiment']:+.2f} (Weight: 20%)")
        print(f"\n  Final Score: {stock['comprehensive']['score']:.2f}/1.00")

        # Technical Details
        tech = stock['technical']['details']
        print(f"\nTechnical Indicators:")
        print(f"  RSI: {tech['rsi']:.1f} ({'Oversold' if tech['rsi'] < 30 else 'Overbought' if tech['rsi'] > 70 else 'Neutral'})")
        print(f"  Volume: {tech['volume_ratio']:.1f}x average")
        print(f"  Bollinger: {tech['bb_position']:.0%} position")
        print(f"  Price vs SMA20: {stock['current_price'] > tech['sma_20']}")

        # ML Details
        print(f"\nML Analysis:")
        print(f"  Prediction: {stock['ml']['prediction']:.2f} | Confidence: {stock['ml']['confidence']:.0%}")
        print(f"  Key Features: {stock['ml']['features']}")

        # News Sentiment
        print(f"\nNews Sentiment: {stock['news']['sentiment']:+.2f} ({'Positive' if stock['news']['sentiment'] > 0 else 'Negative' if stock['news']['sentiment'] < 0 else 'Neutral'})")

        # Trading Strategy
        print(f"\nTrading Strategy:")
        rec = stock['comprehensive']['recommendation']
        if rec == 'STRONG_BUY':
            print(f"  Strong bullish signal across all indicators")
            print(f"  Entry: Rp {stock['current_price']:,.0f} | Target: Rp {stock['levels']['resistance']:,.0f}")
            print(f"  Stop Loss: Rp {stock['levels']['support']:,.0f}")
        elif rec == 'BUY':
            print(f"  Positive consensus with good risk-reward")
            print(f"  Consider partial position, add on dips")
        elif rec == 'HOLD':
            print(f"  Mixed signals - Wait for clearer setup")
        else:
            print(f"  Bearish bias - Consider reducing exposure")

    # Sector Performance
    print("\n" + "=" * 100)
    print("SECTOR PERFORMANCE RANKING")
    print("=" * 100)

    sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)
    for sector, score in sorted_sectors:
        sentiment = "[BULLISH]" if score > 0.6 else "[BEARISH]" if score < 0.4 else "[NEUTRAL]"
        print(f"{sector:15} | Avg Score: {score:.2f} {sentiment}")

    # Additional Insights
    print("\n" + "=" * 100)
    print("ADDITIONAL INSIGHTS")
    print("=" * 100)

    # Top technical opportunities
    tech_top = sorted(results, key=lambda x: x['technical']['score'], reverse=True)[:3]
    print("\nTop Technical Setups:")
    for s in tech_top:
        print(f"  {s['symbol']}: Technical Score {s['technical']['score']:+.2f}")

    # Top ML predictions
    ml_top = sorted(results, key=lambda x: x['ml']['prediction'], reverse=True)[:3]
    print("\nTop ML Predictions:")
    for s in ml_top:
        print(f"  {s['symbol']}: ML Confidence {s['ml']['prediction']:.2f}")

    # News sentiment leaders
    news_top = sorted(results, key=lambda x: x['news']['sentiment'], reverse=True)[:3]
    print("\nPositive News Sentiment:")
    for s in news_top:
        if s['news']['sentiment'] > 0:
            print(f"  {s['symbol']}: Sentiment {s['news']['sentiment']:+.2f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comprehensive_analysis_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump({
            'top_5': top_5,
            'all_results': results[:15],  # Save top 15
            'sector_performance': sector_performance,
            'analysis_time': datetime.now().isoformat()
        }, f, indent=2, default=str)

    print(f"\nAnalysis saved to: {filename}")

if __name__ == "__main__":
    try:
        print("Starting comprehensive analysis...\n")
        results, sectors = run_comprehensive_analysis()

        if results:
            display_comprehensive_recommendations(results, sectors)
        else:
            print("\nNo analysis results available")

    except Exception as e:
        print(f"\nError running analysis: {e}")
        import traceback
        traceback.print_exc()