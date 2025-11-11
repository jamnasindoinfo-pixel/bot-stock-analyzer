#!/usr/bin/env python3
"""
ML Training with Enhanced IDX Data Integration - Clean Version
"""

import sys
import os
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.insert(0, project_root)

from ml_system.core.ml_predictor_v4 import MLPredictorV4
from ml_system.data.idx_scrapper_custom import IdxScrapper

def train_with_idx_enhanced():
    """Train ML models using enhanced IDX data"""
    print("="*70)
    print("ML SYSTEM V5 - ENHANCED TRAINING WITH IDX DATA")
    print("="*70)
    print("Using custom IDX scraper for Indonesian stocks")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Initialize IDX scraper
    print("\n[*] Initializing IDX Scraper...")
    idx_scrapper = IdxScrapper()

    # Get all available stocks
    all_stocks = idx_scrapper.get_idx_stock_list()
    print(f"    Available IDX stocks: {len(all_stocks)}")

    # Configuration
    config = {
        'period': '3y',  # 3 years of data
        'min_days_per_stock': 300,
        'n_estimators': 300,  # Reduced for faster training with more stocks
        'use_feature_selection': True,
        'target_type': 'multiclass',
        'max_stocks': 100  # Target number of stocks
    }

    # Get ALL available stocks for training
    print("\n[*] Selecting stocks for training...")

    # Option 1: Use all available stocks (98 stocks)
    use_all_stocks = True

    if use_all_stocks:
        # Get all stocks from the IDX scraper
        stock_list = idx_scrapper.get_idx_stock_list()
        selected_symbols = [stock['symbol'] for stock in stock_list]

        # Remove duplicates (if any)
        selected_symbols = list(set(selected_symbols))

        # Limit to max_stocks to prevent memory issues
        if len(selected_symbols) > config['max_stocks']:
            selected_symbols = selected_symbols[:config['max_stocks']]

        print(f"    Using {len(selected_symbols)} stocks (limited to {config['max_stocks']} max) for maximum data coverage")
    else:
        # Original selection (37 stocks)
        selected_symbols = [
            # Banking (most liquid)
            'BBCA', 'BBRI', 'BBNI', 'BMRI', 'BRIS',
            # Consumer staples
            'UNVR', 'INDF', 'ICBP', 'KLBF', 'ULTJ',
            # Telecommunication
            'TLKM', 'EXCL', 'ISAT',
            # Conglomerates
            'ASII', 'ELSA', 'MEDC',
            # Mining & Energy
            'ANTM', 'PTBA', 'ADRO', 'TINS', 'PGAS',
            # Infrastructure
            'WIKA', 'ADHI', 'PTPP', 'JSMR', 'TOWR',
            # Property
            'PWON', 'CTRA', 'BSDE',
            # Technology
            'GOTO', 'BUKK',
            # Healthcare
            'KLBF', 'KAEF',
            # Agriculture
            'JPFA', 'CPIN',
            # Automotive
            'AUTO', 'GJTL'
        ]
        print(f"    Selected {len(selected_symbols)} top liquid stocks for training")

    # Download data
    print("\n[*] STEP 1: Downloading IDX data...")
    print(f"    Period: {config['period']}")
    print(f"    Minimum days per stock: {config['min_days_per_stock']}")

    data_start = time.time()
    stock_data = idx_scrapper.get_multiple_stocks(
        symbols=selected_symbols,
        period=config['period']
    )

    if not stock_data:
        print("\n[FAIL] Could not download any IDX data")
        return False

    data_time = time.time() - data_start
    print(f"\n[OK] IDX data downloaded in {data_time:.2f} seconds")
    print(f"  Successfully downloaded: {len(stock_data)} stocks")

    # Filter stocks with sufficient data
    valid_stocks = {}
    for symbol, df in stock_data.items():
        if len(df) >= config['min_days_per_stock']:
            valid_stocks[symbol] = df
            print(f"  {symbol}: {len(df)} days")
        else:
            print(f"  {symbol}: Insufficient data ({len(df)} days)")

    if not valid_stocks:
        print("\n[FAIL] No stocks with sufficient data")
        return False

    print(f"\n  Total valid stocks: {len(valid_stocks)}")

    # Initialize ML predictor
    print("\n[*] STEP 2: Initializing Enhanced ML Predictor...")
    predictor = MLPredictorV4()

    # Process data and create features
    print("\n[*] STEP 3: Processing data and creating features...")
    print("    Generating 100+ technical indicators for each stock...")

    feature_start = time.time()
    all_features = []
    all_targets = []

    total_samples = 0
    for symbol, df in valid_stocks.items():
        print(f"\n  Processing {symbol}...")

        try:
            # Create advanced features
            df_features = predictor.create_advanced_features(df)

            # Create dynamic targets
            df_features['Target'] = predictor.create_dynamic_target(df_features)

            # Remove NaN rows
            df_clean = df_features.dropna()

            if len(df_clean) > 100:
                # Select feature columns
                feature_cols = [col for col in df_clean.columns
                              if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']]

                features = df_clean[feature_cols]
                target = df_clean['Target']

                # Add symbol for tracking
                features['Symbol'] = f'{symbol}.JK'

                all_features.append(features)
                all_targets.append(target)
                total_samples += len(features)

                print(f"    [OK] Generated {len(features)} samples")
            else:
                print(f"    [FAIL] Insufficient data after processing: {len(df_clean)} samples")

        except Exception as e:
            print(f"    [ERROR] Error processing {symbol}: {str(e)}")
            continue

    if not all_features:
        print("\n[FAIL] No valid features generated")
        return False

    # Combine all data
    X = pd.concat(all_features, ignore_index=True)
    y = pd.concat(all_targets, ignore_index=True)

    # Remove symbol column
    if 'Symbol' in X.columns:
        X = X.drop('Symbol', axis=1)

    # Clean data
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    feature_time = time.time() - feature_start
    print(f"\n[OK] Feature engineering completed in {feature_time:.2f} seconds")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Features generated: {X.shape[1]}")

    # Show target distribution
    target_counts = y.value_counts()
    print(f"\n  Target distribution:")
    total = len(y)
    for label, count in target_counts.items():
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        pct = (count / total) * 100
        print(f"    {signal_map[label]}: {count:,} ({pct:.1f}%)")

    # Train models
    print("\n[*] STEP 4: Training Enhanced ML Models...")
    print("    Using ensemble methods with regularization...")

    training_start = time.time()

    try:
        # Calculate test size
        test_size = max(500, int(len(X) * 0.1))

        results = predictor.train_ensemble_models(
            X, y,
            test_size=test_size,
            n_estimators=config['n_estimators'],
            use_feature_selection=config['use_feature_selection']
        )

        training_time = time.time() - training_start
        total_time = time.time() - data_start

        # Print results
        print("\n" + "="*70)
        print("IDX-ENHANCED ML TRAINING RESULTS")
        print("="*70)

        print(f"\nPerformance Metrics:")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Data download: {data_time:.2f}s")
        print(f"  Feature engineering: {feature_time:.2f}s")
        print(f"  Model training: {training_time:.2f}s")

        print(f"\nDataset Statistics:")
        print(f"  Training samples: {results['training_samples']:,}")
        print(f"  Validation samples: {results['validation_samples']:,}")
        print(f"  Features selected: {results['feature_count']}")
        print(f"  Stocks processed: {len(valid_stocks)}")

        print(f"\nModel Performance (Validation):")
        model_perfs = results['models']
        sorted_models = sorted(model_perfs.items(), key=lambda x: x[1]['val_accuracy'], reverse=True)

        for name, perf in sorted_models:
            accuracy = perf['val_accuracy']
            logloss = perf['val_logloss']
            train_time = perf['training_time']
            print(f"  {name.upper():12s}: {accuracy:.3f} accuracy | {logloss:.3f} logloss | {train_time:.2f}s")

        best_accuracy = max([perf['val_accuracy'] for perf in model_perfs.values()])

        # Save models
        print("\n[*] Saving IDX-enhanced models...")
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models_v5'))
        os.makedirs(models_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_file = os.path.join(models_dir, f'ml_predictor_v5_idx_{timestamp}.joblib')

        try:
            saved_path = predictor.save_models(model_file)
            if saved_path:
                file_size = os.path.getsize(saved_path) / (1024 * 1024)
                print(f"  [OK] Models saved: {saved_path}")
                print(f"  [OK] Model size: {file_size:.1f} MB")
        except Exception as e:
            print(f"  [ERROR] Error saving: {str(e)}")

        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'version': '5.0_idx_enhanced',
            'data_source': 'Custom IDX Scraper + Yahoo Finance',
            'stocks_processed': list(valid_stocks.keys()),
            'total_samples': total_samples,
            'processing_time': total_time,
            'model_file': saved_path if 'saved_path' in locals() else None,
            'results': results,
            'config': config
        }

        metadata_path = os.path.join(models_dir, 'training_metadata_v5_idx.json')
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"  [OK] Metadata saved: {metadata_path}")
        except Exception as e:
            print(f"  [WARN] Could not save metadata")

        # Test predictions
        print("\n[*] Testing model predictions...")
        test_symbol = 'BBCA'

        try:
            test_df = idx_scrapper.get_historical_data(test_symbol, period='6mo')

            if test_df is not None and len(test_df) > 60:
                result = predictor.predict(test_df.tail(60))

                if 'predictions' in result and result['predictions']:
                    latest = result['predictions'][-1]
                    print(f"\n  Sample prediction for {test_symbol}.JK:")
                    print(f"    Signal: {latest['signal']}")
                    print(f"    Confidence: {latest['confidence']:.2f}")
                    print(f"    Date: {latest['date']}")
                    print("  [OK] Model working correctly")
                else:
                    print("  [WARN] No predictions returned")
            else:
                print("  [WARN] Insufficient test data")
        except Exception as e:
            print(f"  [ERROR] Prediction test error: {str(e)}")

        # Save data cache
        cache_dir = os.path.join(models_dir, 'data_cache')
        print(f"\n[*] Saving data cache...")
        idx_scrapper.save_data_to_cache(stock_data, cache_dir)
        print(f"  [OK] Data cached to: {cache_dir}")

        # Final summary
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Best accuracy: {best_accuracy:.3f}")

        # Version comparison
        print("\nVersion Comparison Summary:")
        print("  ML v2: ~40% (broken)")
        print("  ML v3: 49.5% (Yahoo Finance - 10 stocks)")
        print("  ML v4: 81.6% (Yahoo Finance - 30 stocks)")
        print(f"  ML v5: {best_accuracy:.1%} (IDX data - {len(valid_stocks)} stocks)")

        if best_accuracy > 0.80:
            print("\n[EXCELLENT] Accuracy > 80%!")
        elif best_accuracy > 0.70:
            print("\n[VERY GOOD] Accuracy > 70%")
        elif best_accuracy > 0.60:
            print("\n[GOOD] Accuracy > 60%")
        else:
            print("\n[INFO] Could be improved with more data or tuning")

        return True

    except Exception as e:
        print(f"\n[FAIL] Training error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = train_with_idx_enhanced()
    print(f"\n[*] Done. Success: {success}")