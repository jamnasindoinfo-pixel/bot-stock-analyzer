#!/usr/bin/env python3
"""
ML Enhanced Stock Signal CLI v2 - Production Version

Enhanced CLI supporting both v1 and v2 ML predictors with:
- Automatic model detection (v2 preferred, v1 fallback)
- Ensemble model output display
- Individual model predictions and weights
- Feature importance display
- Verbose mode options
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import yfinance as yf
import joblib
import argparse

class ProductionMLAnalyzer:
    """Production-ready ML analyzer supporting both v1 and v2 predictors."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.v1_predictor = None
        self.v2_predictor = None
        self.active_version = None

        # Try to load v2 first, fallback to v1
        self._load_predictors()

    def _load_predictors(self):
        """Load ML predictors, preferring v2 over v1"""
        try:
            # Try to import v2 predictor
            from ml_system.core.ml_predictor_v2 import MLPredictorV2
            self.v2_predictor = MLPredictorV2()
            if self.v2_predictor.is_trained():
                self.active_version = 'v2'
                if self.verbose:
                    print(f"[SUCCESS] Enhanced ML Predictor v2 loaded")
                return True
        except ImportError as e:
            if self.verbose:
                print(f"[INFO] MLPredictorV2 not available: {e}")
        except Exception as e:
            if self.verbose:
                print(f"[WARNING] Error loading v2 predictor: {e}")

        # Fallback to v1
        try:
            self.v1_predictor = ProductionMLAnalyzerV1()
            if self.v1_predictor.is_loaded():
                self.active_version = 'v1'
                if self.verbose:
                    print(f"[SUCCESS] ML Predictor v1 loaded")
                return True
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Error loading v1 predictor: {e}")

        print("[WARNING] No ML predictors available - using fallback analysis")
        self.active_version = None
        return False

    def get_model_info(self):
        """Get information about the active ML model"""
        if self.active_version == 'v2':
            info = self.v2_predictor.get_model_info()
            info['active_version'] = 'v2'
            return info
        elif self.active_version == 'v1':
            info = {
                'active_version': 'v1',
                'feature_count': len(self.v1_predictor.features),
                'features': self.v1_predictor.features,
                'signal_classes': self.v1_predictor.reverse_mapping
            }
            return info
        else:
            return {'active_version': None, 'status': 'No ML models available'}

    def is_loaded(self):
        """Check if any ML model is loaded"""
        return self.active_version is not None

    def predict_signal(self, symbol):
        """Predict signal for a stock symbol using available ML predictor."""
        try:
            if self.verbose:
                print(f"[INFO] Analyzing {symbol} using ML {self.active_version if self.active_version else 'fallback'}...")

            ticker = yf.Ticker(symbol)
            data = ticker.history(period="3mo")

            if data.empty:
                return self._create_error_result(symbol, "No data available")

            # Use active predictor
            if self.active_version == 'v2':
                result = self._predict_v2(symbol, data)
            elif self.active_version == 'v1':
                result = self._predict_v1(symbol, data)
            else:
                result = self._create_fallback_result(symbol, data)

            return result

        except Exception as e:
            return self._create_error_result(symbol, str(e))

    def _predict_v2(self, symbol, data):
        """Predict using enhanced v2 predictor"""
        result = self.v2_predictor.predict_signal(data, symbol)

        if result['success'] and self.verbose:
            self._display_v2_details(result)

        return result

    def _predict_v1(self, symbol, data):
        """Predict using v1 predictor"""
        result = self.v1_predictor.predict_signal(symbol)

        if result['success'] and self.verbose:
            self._display_v1_details(result)

        return result

    def _create_fallback_result(self, symbol, data):
        """Create fallback result when no ML model is available"""
        current_price = data['Close'].iloc[-1]
        price_change = ((current_price / data['Close'].iloc[-5]) - 1) * 100 if len(data) >= 5 else 0

        # Simple logic: positive change = WAIT, negative large change = BUY, positive large change = SELL
        if price_change < -3:
            signal = 'BUY'
            confidence = min(0.6, abs(price_change) / 10)
        elif price_change > 3:
            signal = 'SELL'
            confidence = min(0.6, price_change / 10)
        else:
            signal = 'WAIT'
            confidence = 0.3

        return {
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'current_price': current_price,
            'success': True,
            'version': 'fallback',
            'analysis_type': 'price_momentum'
        }

    def _create_error_result(self, symbol, error_msg):
        """Create error result"""
        return {
            'symbol': symbol,
            'signal': 'WAIT',
            'confidence': 0.0,
            'error': error_msg,
            'success': False,
            'version': 'error'
        }

    def _display_v2_details(self, result):
        """Display detailed v2 prediction information"""
        print(f"\n[ENSEMBLE DETAILS for {result['symbol']}]")
        print("-" * 50)

        # Display model weights
        if 'model_weights' in result:
            print("Model Weights:")
            for model, weight in result['model_weights'].items():
                print(f"  {model:<15}: {weight:.3f}")

        # Display individual predictions
        if 'individual_predictions' in result:
            print("\nIndividual Model Predictions:")
            for model, pred in result['individual_predictions'].items():
                print(f"  {model:<15}: {pred['signal']:<6} (confidence: {pred['confidence']:.3f})")

        # Display model confidence stats
        if 'model_confidence' in result:
            conf = result['model_confidence']
            print(f"\nModel Agreement:")
            print(f"  Min Confidence:  {conf['min']:.3f}")
            print(f"  Max Confidence:  {conf['max']:.3f}")
            print(f"  Avg Confidence:  {conf['mean']:.3f}")
            print(f"  Confidence Std:  {conf['std']:.3f}")

        # Display feature importance
        if 'feature_importance' in result and result['feature_importance']:
            print(f"\nTop 5 Feature Importance:")
            for i, (feature, importance) in enumerate(list(result['feature_importance'].items())[:5]):
                print(f"  {i+1}. {feature:<20}: {importance:.4f}")

        print(f"\nFeatures Used: {result.get('features_used', 'N/A')}")

    def _display_v1_details(self, result):
        """Display v1 prediction information"""
        print(f"\n[V1 MODEL DETAILS for {result['symbol']}]")
        print("-" * 40)
        print(f"Signal: {result['signal']}")
        print(f"Confidence: {result['confidence']:.3f}")

        if 'probabilities' in result:
            print("Signal Probabilities:")
            for signal, prob in result['probabilities'].items():
                print(f"  {signal:<6}: {prob:.3f}")


class ProductionMLAnalyzerV1:
    """Original v1 ML analyzer (kept for backward compatibility)."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = []
        self.reverse_mapping = {}
        self.load_model()

    def load_model(self):
        """Load the trained Random Forest model."""
        try:
            if os.path.exists('models/working_rf_model.pkl'):
                self.model = joblib.load('models/working_rf_model.pkl')
                self.scaler = joblib.load('models/working_rf_scaler.pkl')
                model_info = joblib.load('models/working_rf_info.pkl')
                self.features = model_info['features']
                self.reverse_mapping = model_info['reverse_mapping']
                return True
        except Exception as e:
            pass
        return False

    def is_loaded(self):
        return self.model is not None

    def predict_signal(self, symbol):
        """Predict signal for a stock symbol."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="3mo")

            if data.empty:
                return {
                    'symbol': symbol,
                    'signal': 'WAIT',
                    'confidence': 0.0,
                    'error': 'No data available',
                    'success': False,
                    'version': 'v1'
                }

            # Create features
            features = self.create_features(data)

            if features.empty:
                return {
                    'symbol': symbol,
                    'signal': 'WAIT',
                    'confidence': 0.0,
                    'error': 'Could not create features',
                    'success': False,
                    'version': 'v1'
                }

            # Get latest features
            latest_features = features.iloc[[-1]][self.features].fillna(0)
            current_price = data['Close'].iloc[-1]

            # Scale and predict
            scaled_features = self.scaler.transform(latest_features)
            prediction = self.model.predict(scaled_features)[0]
            probabilities = self.model.predict_proba(scaled_features)[0]

            # Get signal and confidence
            signal = self.reverse_mapping.get(prediction, 'WAIT')
            confidence = probabilities[np.where(self.model.classes_ == prediction)[0][0]]

            return {
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence,
                'current_price': current_price,
                'success': True,
                'version': 'v1',
                'probabilities': {
                    self.reverse_mapping.get(cls, 'UNKNOWN'): float(prob)
                    for cls, prob in zip(self.model.classes_, probabilities)
                }
            }

        except Exception as e:
            return {
                'symbol': symbol,
                'signal': 'WAIT',
                'confidence': 0.0,
                'error': str(e),
                'success': False,
                'version': 'v1'
            }

    def create_features(self, df):
        """Create features for prediction."""
        if df.empty or len(df) < 30:
            return pd.DataFrame()

        features = df.copy()
        close = features['Close']

        try:
            # Basic features
            for period in [5, 10, 20]:
                features[f'return_{period}d'] = close.pct_change(period)
                features[f'sma_{period}'] = close.rolling(window=period).mean()
                features[f'price_to_sma_{period}'] = close / features[f'sma_{period}']
                features[f'volatility_{period}d'] = features[f'return_{period}d'].rolling(window=period).std()

            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi_14'] = 100 - (100 / (1 + rs))

            # Momentum
            features['momentum_5'] = (close / close.shift(5) - 1) * 100

            # Clean data
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(0)

            return features
        except Exception as e:
            return pd.DataFrame()


def print_header():
    print("=" * 60)
    print("         ML ENHANCED STOCK SIGNAL ANALYZER v2")
    print("=" * 60)


def print_results(results, show_details=False):
    """Print analysis results in clean format."""
    print("\n" + "=" * 80)
    print(f"{'SYMBOL':<12} {'SIGNAL':<8} {'CONFIDENCE':<12} {'PRICE':<15} {'VERSION':<8} {'STATUS'}")
    print("-" * 80)

    for result in results:
        if result['success']:
            conf_color = "HIGH" if result['confidence'] > 0.6 else "MED" if result['confidence'] > 0.4 else "LOW"
            version = result.get('version', 'N/A')

            print(f"{result['symbol']:<12} {result['signal']:<8} {result['confidence']:<12.3f} "
                  f"{result['current_price']:<15.2f} {version:<8} {conf_color}")
        else:
            version = result.get('version', 'ERROR')
            print(f"{result['symbol']:<12} {'ERROR':<8} {'0.000':<12} {'-':<15} {version:<8} FAILED")

    if show_details:
        for result in results:
            if result['success'] and result.get('version') == 'v2':
                analyzer = ProductionMLAnalyzer(verbose=False)
                analyzer._display_v2_details(result)


def print_status(ml_analyzer):
    """Print detailed status information"""
    model_info = ml_analyzer.get_model_info()

    print("\n[ML SYSTEM STATUS]")
    print("-" * 40)

    if model_info['active_version']:
        version = model_info['active_version']
        print(f"Active Version: {version}")

        if version == 'v2':
            print(f"Model Types: {model_info.get('model_types', 'N/A')}")
            print(f"Feature Count: {model_info.get('feature_count', 'N/A')}")
            print(f"Feature Groups: {model_info.get('feature_groups', 'N/A')}")
            if 'model_weights' in model_info:
                print("Model Weights:")
                for model, weight in model_info['model_weights'].items():
                    print(f"  {model}: {weight:.3f}")
        elif version == 'v1':
            print(f"Feature Count: {model_info.get('feature_count', 'N/A')}")
            print(f"Signal Classes: {model_info.get('signal_classes', 'N/A')}")

        training_info = model_info.get('training_info', {})
        if training_info:
            print(f"Training Samples: {training_info.get('training_samples', 'N/A')}")
            print(f"Validation Samples: {training_info.get('validation_samples', 'N/A')}")
            print(f"Validation Accuracy: {training_info.get('validation_accuracy', 'N/A'):.3f}")
    else:
        print("Status: No ML models available")
        print("Mode: Using fallback analysis")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='ML Enhanced Stock Signal Analyzer v2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python production_ml_cli.py --symbols BBCA BBRI TLKM --verbose
  python production_ml_cli.py --interactive
  python production_ml_cli.py --status
        """
    )

    parser.add_argument('--symbols', nargs='+', help='Stock symbols to analyze (e.g., BBCA BBRI)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output with detailed model information')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--status', '-s', action='store_true', help='Show ML system status and exit')

    return parser.parse_args()


def main():
    """Main production CLI function."""
    args = parse_arguments()

    print_header()

    # Initialize ML analyzer
    ml_analyzer = ProductionMLAnalyzer(verbose=args.verbose)

    # Show status if requested
    if args.status:
        print_status(ml_analyzer)
        return

    # Show brief status
    if ml_analyzer.is_loaded():
        model_info = ml_analyzer.get_model_info()
        version = model_info['active_version']
        print(f"[STATUS] ML System: ACTIVE (v{version})")
        if args.verbose:
            print_status(ml_analyzer)
    else:
        print("[STATUS] ML System: INACTIVE - Using fallback analysis")

    # Handle different modes
    if args.symbols:
        # Batch mode - analyze provided symbols
        symbols = [s.upper() if s.endswith('.JK') else s.upper() + '.JK' for s in args.symbols]

        print(f"\n[PROCESSING] Analyzing {len(symbols)} symbols...")

        results = []
        for symbol in symbols:
            result = ml_analyzer.predict_signal(symbol)
            results.append(result)

        # Display results
        print_results(results, show_details=args.verbose)

        # Summary
        successful = sum(1 for r in results if r['success'])
        print(f"\n[SUMMARY] {successful}/{len(results)} analyses completed successfully")

    elif args.interactive or not args.symbols:
        # Interactive mode
        print("\nCommands:")
        print("  - Enter stock symbols (e.g., BBCA BBRI TLKM)")
        print("  - 'status' for detailed ML system status")
        print("  - 'verbose' toggle verbose mode")
        print("  - 'exit' to quit")
        print("-" * 60)

        while True:
            try:
                user_input = input("\nEnter symbols or command: ").strip()

                if not user_input:
                    continue

                cmd = user_input.lower()

                if cmd in {"exit", "quit", "q"}:
                    print("[INFO] Exiting program...")
                    break

                if cmd == "status":
                    print_status(ml_analyzer)
                    continue

                if cmd == "verbose":
                    ml_analyzer.verbose = not ml_analyzer.verbose
                    print(f"[INFO] Verbose mode: {'ON' if ml_analyzer.verbose else 'OFF'}")
                    continue

                # Treat everything else as stock symbols
                symbols = user_input.upper().split()
                symbols = [s if s.endswith('.JK') else s + '.JK' for s in symbols]
                results = []

                print(f"\n[PROCESSING] Analyzing {len(symbols)} symbols...")

                for symbol in symbols:
                    result = ml_analyzer.predict_signal(symbol)
                    results.append(result)

                # Display results
                print_results(results, show_details=ml_analyzer.verbose)

                # Summary
                successful = sum(1 for r in results if r['success'])
                print(f"\n[SUMMARY] {successful}/{len(results)} analyses completed successfully")

            except KeyboardInterrupt:
                print("\n[INFO] Exiting program...")
                break
            except Exception as e:
                print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()