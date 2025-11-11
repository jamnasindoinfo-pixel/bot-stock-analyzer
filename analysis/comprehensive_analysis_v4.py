#!/usr/bin/env python3
"""
Enhanced Comprehensive Stock Analysis using ML v4/v5
Combines Technical Analysis, ML Predictions (v4/v5), and AI Analysis
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
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Load environment
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import ML predictors
ML_V4_AVAILABLE = False

try:
    from ml_system.core.ml_predictor_v4 import MLPredictorV4
    ML_V4_AVAILABLE = True
    print("[INFO] ML v4 (Enhanced) available")
except ImportError:
    print("[WARN] ML v4 not available")

# ML v5 uses the same MLPredictorV4 class but with IDX-trained models
ML_V5_AVAILABLE = ML_V4_AVAILABLE  # v5 models are compatible with v4 predictor

try:
    from analyzers.market_analyzer import MarketAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False
    print("[WARN] Market Analyzer not available")

# Import Gemini AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    model = genai.GenerativeModel('gemini-pro')
except:
    GEMINI_AVAILABLE = False
    print("[WARN] Gemini AI not available")

class ComprehensiveAnalyzerV4:
    """Enhanced comprehensive analyzer with ML v4/v5 support"""

    def __init__(self):
        self.ml_predictor = None
        self.market_analyzer = None
        self.ml_version = None
        self.load_ml_model()

    def load_ml_model(self):
        """Load the best available ML model"""

        # Try v5 first (IDX data), then v4
        for version, path_suffix in [('v5', 'models_v5'), ('v4', 'models_v4')]:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml_system', path_suffix)
            if os.path.exists(model_path):
                # Find the latest model file
                model_files = [f for f in os.listdir(model_path) if f.endswith('.joblib')]
                if model_files:
                    model_files.sort()
                    latest_model = os.path.join(model_path, model_files[-1])

                    try:
                        if version == 'v4' and ML_V4_AVAILABLE:
                            self.ml_predictor = MLPredictorV4()
                            self.ml_predictor.load_models(latest_model)
                            self.ml_version = "v4 (Enhanced)"
                            print(f"[INFO] Loaded ML {self.ml_version} model")
                            return
                        elif version == 'v5':
                            # MLPredictorV5 would be similar to v4 but with IDX data
                            if ML_V4_AVAILABLE:
                                self.ml_predictor = MLPredictorV4()
                                self.ml_predictor.load_models(latest_model)
                                self.ml_version = "v5 (IDX Data)"
                                print(f"[INFO] Loaded ML {self.ml_version} model")
                                return
                    except Exception as e:
                        print(f"[WARN] Failed to load {version} model: {str(e)}")

        print("[WARN] No ML models loaded")
        self.ml_version = None

    def get_analysis_weight(self, method: str, confidence: float) -> float:
        """
        Calculate weight for each analysis method based on confidence
        """
        base_weights = {
            'technical': 0.35,
            'ml': 0.45,
            'ai': 0.20
        }

        # Adjust weights based on confidence
        if method == 'ml' and confidence > 0.8:
            base_weights['ml'] = 0.55
            base_weights['technical'] = 0.30
            base_weights['ai'] = 0.15
        elif method == 'ml' and confidence < 0.5:
            base_weights['ml'] = 0.30
            base_weights['technical'] = 0.45
            base_weights['ai'] = 0.25

        return base_weights

    def analyze_stock(self, symbol: str) -> Dict:
        """
        Comprehensive analysis of a stock

        Args:
            symbol: Stock symbol (e.g., 'BBCA.JK')

        Returns:
            Dictionary with complete analysis
        """
        print(f"\n[*] Analyzing {symbol}...")

        # Get stock data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period='6mo')

        if df.empty:
            return {'error': f'No data available for {symbol}'}

        # Initialize analysis result
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'ml_version': self.ml_version,
            'current_price': df['Close'][-1],
            'price_change': (df['Close'][-1] - df['Close'][-2]) / df['Close'][-2] * 100,
            'volume': df['Volume'][-1],
            'recommendations': {},
            'scores': {},
            'final_recommendation': 'HOLD',
            'confidence': 0.0
        }

        # 1. Technical Analysis
        technical_score, technical_signal = self.analyze_technical(df)
        analysis['scores']['technical'] = technical_score
        analysis['recommendations']['technical'] = technical_signal

        # 2. ML Prediction
        if self.ml_predictor:
            ml_score, ml_signal, ml_confidence = self.analyze_ml(df, symbol)
            analysis['scores']['ml'] = ml_score
            analysis['recommendations']['ml'] = ml_signal
            analysis['ml_confidence'] = ml_confidence
        else:
            analysis['scores']['ml'] = 0
            analysis['recommendations']['ml'] = 'HOLD'
            analysis['ml_confidence'] = 0

        # 3. AI Analysis (Gemini)
        if GEMINI_AVAILABLE:
            ai_score, ai_signal = self.analyze_ai(symbol, df)
            analysis['scores']['ai'] = ai_score
            analysis['recommendations']['ai'] = ai_signal
        else:
            analysis['scores']['ai'] = 0
            analysis['recommendations']['ai'] = 'HOLD'

        # 4. Combine recommendations with dynamic weighting
        weights = self.get_analysis_weight('ml', analysis.get('ml_confidence', 0.5))

        # Convert signals to scores
        signal_scores = {'BUY': 1, 'HOLD': 0, 'SELL': -1}

        final_score = (
            signal_scores[analysis['recommendations']['technical']] * weights['technical'] +
            signal_scores[analysis['recommendations']['ml']] * weights['ml'] +
            signal_scores[analysis['recommendations']['ai']] * weights['ai']
        )

        # Convert final score to recommendation
        if final_score > 0.3:
            analysis['final_recommendation'] = 'BUY'
        elif final_score < -0.3:
            analysis['final_recommendation'] = 'SELL'
        else:
            analysis['final_recommendation'] = 'HOLD'

        analysis['final_score'] = final_score
        analysis['confidence'] = abs(final_score)

        return analysis

    def analyze_technical(self, df: pd.DataFrame) -> Tuple[float, str]:
        """Technical analysis using MarketAnalyzer"""
        if not ANALYZER_AVAILABLE:
            return 0.0, 'HOLD'

        try:
            analyzer = MarketAnalyzer()
            result = analyzer.analyze_stock(df)

            # Convert result to score and signal
            if result['recommendation'] == 'BUY':
                score = min(result.get('confidence', 0.5), 1.0)
            elif result['recommendation'] == 'SELL':
                score = -min(result.get('confidence', 0.5), 1.0)
            else:
                score = 0.0

            return score, result['recommendation']
        except:
            return 0.0, 'HOLD'

    def analyze_ml(self, df: pd.DataFrame, symbol: str) -> Tuple[float, str, float]:
        """ML prediction using loaded model"""
        if not self.ml_predictor:
            return 0.0, 'HOLD', 0.0

        try:
            # Make prediction
            result = self.ml_predictor.predict(df.tail(100))

            if 'predictions' in result and result['predictions']:
                latest = result['predictions'][-1]
                signal = latest['signal']
                confidence = latest['confidence']

                # Convert to score
                if signal == 'BUY':
                    score = confidence
                elif signal == 'SELL':
                    score = -confidence
                else:
                    score = 0.0

                return score, signal, confidence
            else:
                return 0.0, 'HOLD', 0.0
        except Exception as e:
            print(f"  [ERROR] ML prediction failed: {str(e)}")
            return 0.0, 'HOLD', 0.0

    def analyze_ai(self, symbol: str, df: pd.DataFrame) -> Tuple[float, str]:
        """AI analysis using Gemini"""
        if not GEMINI_AVAILABLE:
            return 0.0, 'HOLD'

        try:
            # Prepare context for AI
            current_price = df['Close'][-1]
            price_change = (df['Close'][-1] - df['Close'][-5]) / df['Close'][-5] * 100
            volume = df['Volume'][-1]
            avg_volume = df['Volume'].mean()

            # Simple technical summary
            if len(df) >= 20:
                ma20 = df['Close'].rolling(20).mean()[-1]
                above_ma20 = current_price > ma20
                trend = "uptrend" if df['Close'][-1] > df['Close'][-20] else "downtrend"
            else:
                above_ma20 = True
                trend = "neutral"

            prompt = f"""
            Analyze {symbol} Indonesian stock with these data:
            - Current price: {current_price:.2f}
            - 5-day change: {price_change:+.2f}%
            - Volume: {volume:,.0f} (Avg: {avg_volume:,.0f})
            - Position vs MA20: {'Above' if above_ma20 else 'Below'}
            - Trend: {trend}

            Give a brief trading recommendation (BUY/HOLD/SELL) with reasoning.
            Response format: "RECOMMENDATION: BUY/HOLD/SELL | Reason: [brief reason]"
            """

            response = model.generate_content(prompt)
            text = response.text

            # Extract recommendation
            if 'BUY' in text:
                signal = 'BUY'
                score = 0.6  # Moderate confidence for AI
            elif 'SELL' in text:
                signal = 'SELL'
                score = -0.6
            else:
                signal = 'HOLD'
                score = 0.0

            return score, signal

        except Exception as e:
            print(f"  [ERROR] AI analysis failed: {str(e)}")
            return 0.0, 'HOLD'

    def analyze_multiple_stocks(self, symbols: List[str]) -> List[Dict]:
        """Analyze multiple stocks and rank them"""
        print(f"\n[*] Analyzing {len(symbols)} stocks...")
        print(f"    ML Model: {self.ml_version or 'Not loaded'}")
        print(f"    Gemini AI: {'Available' if GEMINI_AVAILABLE else 'Not available'}")

        results = []

        for symbol in symbols:
            try:
                analysis = self.analyze_stock(symbol)
                if 'error' not in analysis:
                    results.append(analysis)
                print(f"  {symbol}: {analysis['final_recommendation']} (conf: {analysis['confidence']:.2f})")
            except Exception as e:
                print(f"  {symbol}: Error - {str(e)}")

        # Sort by final score
        results.sort(key=lambda x: x['final_score'], reverse=True)

        return results

def main():
    """Main function to run comprehensive analysis"""
    print("="*70)
    print("COMPREHENSIVE STOCK ANALYSIS V4")
    print("Technical + ML (v4/v5) + AI Analysis")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Initialize analyzer
    analyzer = ComprehensiveAnalyzerV4()

    # Stock universe
    stock_universe = {
        'Top Blue Chips': ['BBCA.JK', 'BBRI.JK', 'BMRI.JK', 'TLKM.JK', 'UNVR.JK', 'ASII.JK'],
        'Banking': ['BBNI.JK', 'BRIS.JK', 'BTPN.JK', 'BJBR.JK'],
        'Technology': ['EXCL.JK', 'ISAT.JK', 'GOTO.JK'],
        'Consumer': ['INDF.JK', 'ICBP.JK', 'KLBF.JK'],
        'Infrastructure': ['JSMR.JK', 'WIKA.JK', 'ADHI.JK', 'PTPP.JK'],
        'Mining': ['ANTM.JK', 'PTBA.JK', 'TINS.JK'],
        'Property': ['BSDE.JK', 'PWON.JK', 'CTRA.JK']
    }

    # Analyze all
    all_results = []
    for sector, stocks in stock_universe.items():
        print(f"\n{'='*50}")
        print(f"SECTOR: {sector}")
        print(f"{'='*50}")
        results = analyzer.analyze_multiple_stocks(stocks)
        all_results.extend(results)

    # Overall ranking
    print(f"\n{'='*70}")
    print("OVERALL RANKING")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'Symbol':<10} {'Recommendation':<15} {'Score':<8} {'Price':<10} {'Change':<8}")
    print("-"*70)

    for i, result in enumerate(all_results[:20], 1):  # Top 20
        print(f"{i:<5} {result['symbol']:<10} {result['final_recommendation']:<15} "
              f"{result['final_score']:+.2f}    {result['current_price']:<10.2f} "
              f"{result['price_change']:+.1f}%")

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'analysis_logs')
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    detailed_file = os.path.join(output_dir, f'comprehensive_analysis_v4_{timestamp}.json')
    with open(detailed_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'ml_version': analyzer.ml_version,
            'total_analyzed': len(all_results),
            'results': all_results
        }, f, indent=2)

    # Save summary
    summary = []
    for result in all_results:
        summary.append({
            'symbol': result['symbol'],
            'recommendation': result['final_recommendation'],
            'score': result['final_score'],
            'confidence': result['confidence'],
            'price': result['current_price'],
            'change': result['price_change']
        })

    summary_file = os.path.join(output_dir, 'latest_analysis_v4.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'ml_version': analyzer.ml_version,
            'summary': summary
        }, f, indent=2)

    print(f"\n[*] Results saved:")
    print(f"  Detailed: {detailed_file}")
    print(f"  Summary: {summary_file}")

    # Top 5 picks
    print(f"\n{'='*70}")
    print("TOP 5 RECOMMENDATIONS")
    print(f"{'='*70}")
    for i, result in enumerate(all_results[:5], 1):
        print(f"\n{i}. {result['symbol']} - {result['final_recommendation']}")
        print(f"   Score: {result['final_score']:+.2f} | Confidence: {result['confidence']:.2f}")
        print(f"   Price: {result['current_price']:.2f} | Change: {result['price_change']:+.1f}%")
        print(f"   ML: {result['recommendations']['ml']} (conf: {result.get('ml_confidence', 0):.2f})")
        print(f"   Technical: {result['recommendations']['technical']}")
        print(f"   AI: {result['recommendations']['ai']}")

    print("\n✓ Analysis complete!")

if __name__ == "__main__":
    main()