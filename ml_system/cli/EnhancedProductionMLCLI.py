#!/usr/bin/env python3
"""
Enhanced Production CLI with v1/v2 Automatic Detection

Automatically detects and uses the best available ML predictor
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict

# Import ML components
try:
    from ml_system.core.ml_predictor_v2 import MLPredictorV2
    from ml_system.core.ml_predictor import MLPredictor
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    class MLPredictorV2:
        def __init__(self):
            self.trained = False
        def is_trained(self):
            return False

class EnhancedProductionMLCLI:
    """Enhanced production CLI with automatic version detection"""

    def __init__(self):
        self.predictor = None
        self.version = None
        self.model_type = None
        self._initialize_predictor()

    def _initialize_predictor(self):
        """Initialize the best available predictor"""
        try:
            # Try Enhanced v2 first
            if ENHANCED_AVAILABLE:
                self.predictor = MLPredictorV2()
                if self.predictor.is_trained():
                    self.version = "v2"
                    self.model_type = "Enhanced Ensemble (RF + XGBoost + LSTM)"
                    print("[INFO] Using Enhanced ML Predictor v2")
                    return

                # Try to load models
                if self.predictor.load_models():
                    self.version = "v2"
                    self.model_type = "Enhanced Ensemble (RF + XGBoost + LSTM)"
                    print("[INFO] Loaded Enhanced ML Predictor v2")
                    return
                else:
                    print("[INFO] Enhanced v2 models not trained, checking v1...")
            else:
                print("[INFO] Enhanced v2 not available, using v1...")

            # Fallback to v1
            self.predictor = MLPredictor()
            if self.predictor.enabled:
                self.version = "v1"
                self.model_type = "Single Random Forest"
                print("[INFO] Using Basic ML Predictor v1")
                return

        except Exception as e:
            print(f"[ERROR] Failed to initialize any ML predictor: {e}")

        # Final fallback
        self.version = "none"
        self.model_type = "No ML Available"
        print("[WARNING] No ML predictor available - Using fallback analysis")
        self.predictor = None

    def is_available(self):
        """Check if any ML predictor is available"""
        return self.predictor is not None and (
            (self.version == "v2" and self.predictor.is_trained()) or
            (self.version == "v1" and self.predictor.enabled)
        )

    def predict_signals(self, symbols: List[str], verbose: bool = False) -> List[Dict]:
        """Predict signals for multiple symbols"""
        results = []

        for symbol in symbols:
            if not symbol.endswith('.JK'):
                symbol += '.JK'

            try:
                if verbose:
                    print(f"Analyzing {symbol}...")

                # Get data
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="3mo")

                if data.empty:
                    print(f"[DEBUG] Data empty for {symbol}")
                    results.append({
                        'symbol': symbol,
                        'signal': 'WAIT',
                        'confidence': 0.0,
                        'error': 'No data available',
                        'success': False
                    })
                    continue

                print(f"[DEBUG] Got {len(data)} rows of data for {symbol}")

                # Preprocess data to handle Adj Close column
                # Handle multi-level columns from yfinance
                if hasattr(data.columns, 'levels') and len(data.columns.levels) > 1:
                    # Flatten multi-level columns: ('Close', 'AAPL') -> 'Close'
                    data.columns = data.columns.get_level_values(0)

                # Add Adj Close column if it doesn't exist (use Close as Adj Close)
                if 'Adj Close' not in data.columns and 'Close' in data.columns:
                    data = data.copy()
                    data['Adj Close'] = data['Close']

                # Make prediction
                if self.version == "v2":
                    print(f"[DEBUG] Making v2 prediction for {symbol}...")
                    result = self.predictor.predict_signal(data, symbol)
                    print(f"[DEBUG] Prediction result: {result}")

                    if result and result.get('success', False):
                        print(f"[DEBUG] Prediction successful for {symbol}")
                        # Format v2 results
                        enhanced_result = {
                            'symbol': symbol,
                            'signal': result['signal'],
                            'confidence': result['confidence'],
                            'individual_predictions': result.get('individual_predictions', {}),
                            'model_weights': result.get('model_weights', {}),
                            'feature_importance': result.get('feature_importance', {}),
                            'current_price': result.get('current_price', 0),
                            'features_used': result.get('features_used', 0),
                            'success': True,
                            'version': 'v2'
                        }

                        # Add version-specific fields
                        if result.get('model_confidence'):
                            enhanced_result['model_confidence'] = result['model_confidence']

                        print(f"[DEBUG] Appending result for {symbol}")
                        results.append(enhanced_result)
                    else:
                        print(f"[DEBUG] Prediction failed for {symbol}: {result.get('error')}")
                        enhanced_result = {
                            'symbol': symbol,
                            'signal': 'WAIT',
                            'confidence': 0.0,
                            'error': result.get('error', 'Unknown error'),
                            'success': False,
                            'version': 'v2'
                        }

                        print(f"[DEBUG] Appending failed result for {symbol}")
                        results.append(enhanced_result)

                elif self.version == "v1":
                    result = self.predictor.predict_signal(data, symbol)

                    if result and result.get('success', False):
                        # Format v1 results to match v2 format
                        results.append({
                            'symbol': symbol,
                            'signal': result['signal'],
                            'confidence': result['confidence'],
                            'probabilities': result.get('probabilities', {}),
                            'success': True,
                            'version': 'v1',
                            'model_type': 'Random Forest'
                        })
                    else:
                        # Fallback when v1 prediction fails
                        results.append({
                            'symbol': symbol,
                            'signal': 'WAIT',
                            'confidence': 0.0,
                            'error': result.get('error', 'Unknown error'),
                            'success': False,
                            'version': 'v1'
                        })

                else:
                    # No ML available - use simple technical analysis
                    results.append({
                        'symbol': symbol,
                        'signal': self._simple_analysis(data, symbol),
                        'confidence': 0.5,  # Medium confidence
                        'success': True,
                        'version': 'none',
                        'note': 'Simple technical analysis - No ML available'
                    })

            except Exception as e:
                results.append({
                    'symbol': symbol,
                    'signal': 'WAIT',
                    'confidence': 0.0,
                    'error': str(e),
                    'success': False,
                    'version': self.version
                })

        return results

    def _simple_analysis(self, data, symbol):
        """Simple technical analysis when no ML available"""
        try:
            # Calculate basic indicators
            close = data['Close']

            # RSI (14)
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # Simple moving averages
            sma_20 = close.rolling(20).mean()
            current_price = close.iloc[-1]
            sma_20_price = sma_20.iloc[-1]

            # Generate simple signal
            if current_price > sma_20_price * 1.02 and rsi.iloc[-1] < 70:
                return 'BUY'
            elif current_price < sma_20_price * 0.98 and rsi.iloc[-1] > 30:
                return 'SELL'
            else:
                return 'WAIT'

        except:
            return 'WAIT'

    def print_results(self, results: List[Dict], verbose: bool = False):
        """Print prediction results"""
        print("\n" + "=" * 80)
        print(f"Stock Signal Results - {self.model_type}")
        print("=" * 80)

        if verbose:
            # Detailed output
            print(f"{'SYMBOL':<12} {'SIGNAL':<8} {'CONFIDENCE':<12} {'PRICE':<15} {'INDIVIDUAL'}")
            print("-" * 80)

            for result in results:
                if result['success']:
                    signal = result['signal']
                    confidence = result['confidence']
                    price = result.get('current_price', 0)
                    individual = result.get('individual_predictions', {})

                    # Show individual models for v2
                    if result['version'] == 'v2' and individual:
                        models_str = ", ".join([
                            f"{k}:{v['signal']}" for k, v in individual.items()
                        ])
                        print(f"{result['symbol']:<12} {signal:<8} {confidence:<12.3f} {price:<15.2f} {models_str}")
                    else:
                        print(f"{result['symbol']:<12} {signal:<8} {confidence:<12.3f} {price:<15.2f} {'N/A'}")
                else:
                    print(f"{result['symbol']:<12} {'ERROR':<8} {'0.000':<12} {'N/A':<15} {result.get('error', 'Unknown error')}")

        else:
            # Simple output
            successful = sum(1 for r in results if r['success'])
            print(f"{'SYMBOL':<12} {'SIGNAL':<8} {'CONFIDENCE':<12} {'PRICE':<15} {'STATUS'}")
            print("-" * 80)

            for result in results:
                if result['success']:
                    signal = result['signal']
                    confidence = result['confidence']
                    price = result.get('current_price', 0)
                    status = "HIGH" if confidence > 0.6 else "MED" if confidence > 0.4 else "LOW"
                    print(f"{result['symbol']:<12} {signal:<8} {confidence:<12.3f} {price:<15.2f} {status}")
                else:
                    print(f"{result['symbol']:<12} {'ERROR':<8} {'0.000':<12} {'N/A':<15} {result.get('error', 'Unknown error')}")

        # Summary
        print("-" * 80)
        print(f"Analysis: {successful}/{len(results)} successful | Version: {self.version}")

    def get_status(self):
        """Get system status"""
        status = {
            'ml_available': self.is_available(),
            'version': self.version,
            'model_type': self.model_type,
            'features': 0
        }

        if self.version == "v2" and self.predictor.is_trained():
            info = self.predictor.get_model_info()
            status['features'] = info.get('feature_count', 0)
            status['trained_samples'] = info.get('training_info', {}).get('training_samples', 0)
            status['validation_accuracy'] = info.get('training_info', {}).get('validation_accuracy', 0)

        elif self.version == "v1" and self.predictor.enabled:
            status['features'] = 14
            status['trained_samples'] = "Unknown"

        return status

def main():
    """Main function for CLI usage"""
    cli = EnhancedProductionMLCLI()

    # Example usage
    results = cli.predict_signals(['BBCA', 'BBRI', 'TLKM'], verbose=True)
    cli.print_results(results, verbose=True)

if __name__ == "__main__":
    main()