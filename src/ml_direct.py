#!/usr/bin/env python3
"""
Direct ML Test - No Rich dependencies
Usage: python ml_direct.py BBCA
"""

import sys
import yfinance as yf

sys.path.insert(0, '.')

def main():
    if len(sys.argv) < 2:
        print("Usage: python ml_direct.py BBCA")
        return

    symbol = sys.argv[1]
    if not symbol.endswith('.JK'):
        symbol += '.JK'

    print(f"Testing direct ML prediction for {symbol}...")

    try:
        # Test ML Predictor v2 directly
        from ml_system.core.ml_predictor_v2 import MLPredictorV2

        predictor = MLPredictorV2()

        if predictor.load_models():
            print("[+] Models loaded successfully")

            # Get data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="3mo")

            # Add Adj Close if missing
            if 'Adj Close' not in data.columns:
                data['Adj Close'] = data['Close']
                print("[+] Added Adj Close column")

            print(f"[+] Got data: {len(data)} rows")

            # Predict
            result = predictor.predict_signal(data, symbol)

            if result and result.get('success'):
                print("\n=== ML PREDICTION RESULT ===")
                print(f"Signal: {result.get('signal')}")
                print(f"Confidence: {result.get('confidence', 0):.1%}")
                print(f"Current Price: {result.get('current_price', 0):,.0f}")
                print(f"Version: {result.get('version')}")

                # Individual predictions
                individual = result.get('individual_predictions', {})
                if individual:
                    print("\nIndividual Models:")
                    for model, pred in individual.items():
                        signal = pred.get('signal')
                        conf = pred.get('confidence', 0)
                        print(f"  {model}: {signal} ({conf:.1%})")

                weights = result.get('model_weights', [])
                if weights and isinstance(weights, list):
                    print("\nModel Weights:")
                    model_names = ['Random Forest', 'XGBoost', 'LSTM']
                    for i, (name, weight) in enumerate(zip(model_names, weights)):
                        print(f"  {name}: {weight:.3f}")
            else:
                print(f"[-] Prediction failed: {result.get('error') if result else 'Unknown'}")
        else:
            print("[-] Failed to load ML models")

    except Exception as e:
        print(f"[-] Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()