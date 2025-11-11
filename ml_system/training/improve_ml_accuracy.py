#!/usr/bin/env python3
"""
Script to improve ML model accuracy by fixing data preprocessing and training issues
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

def fix_data_preprocessing(df):
    """Fix data preprocessing to avoid symbol conversion errors"""
    # Remove non-numeric columns before scaling
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    # Handle NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')

    # Ensure all features are numeric
    for col in df.columns:
        if col not in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def create_better_features(df):
    """Create improved features for better prediction"""
    # Price-based features
    df['returns_1d'] = df['Close'].pct_change()
    df['returns_5d'] = df['Close'].pct_change(5)
    df['returns_10d'] = df['Close'].pct_change(10)

    # Volatility
    df['volatility_5d'] = df['returns_1d'].rolling(5).std()
    df['volatility_20d'] = df['returns_1d'].rolling(20).std()

    # Moving averages
    df['sma_5'] = df['Close'].rolling(5).mean()
    df['sma_10'] = df['Close'].rolling(10).mean()
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['ema_12'] = df['Close'].ewm(span=12).mean()
    df['ema_26'] = df['Close'].ewm(span=26).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    # Volume features
    df['volume_sma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma']

    # Bollinger Bands
    df['bb_middle'] = df['sma_20']
    df['bb_std'] = df['Close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # Price position relative to moving averages
    df['price_vs_sma5'] = (df['Close'] - df['sma_5']) / df['sma_5']
    df['price_vs_sma20'] = (df['Close'] - df['sma_20']) / df['sma_20']

    # Create target variable (predict if price will increase in next 5 days)
    future_return = df['Close'].shift(-5) / df['Close'] - 1
    df['target'] = np.where(future_return > 0.03, 2,  # Buy if >3% gain
                           np.where(future_return < -0.03, 0,  # Sell if >3% loss
                                   1))  # Hold otherwise

    # Create features for next day prediction
    df['target_1d'] = np.where(df['returns_1d'].shift(-1) > 0.01, 2,
                              np.where(df['returns_1d'].shift(-1) < -0.01, 0, 1))

    return df

def balance_dataset(X, y):
    """Balance the dataset to reduce bias"""
    from sklearn.utils import resample

    # Convert to DataFrame and Series
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y, name='target')

    # Combine features and target
    df = pd.concat([X, y], axis=1)

    # Get target column name
    target_col = y.name

    # Separate classes
    df_0 = df[df[target_col] == 0]  # Sell
    df_1 = df[df[target_col] == 1]  # Hold
    df_2 = df[df[target_col] == 2]  # Buy

    # Determine the class with minimum samples
    min_samples = min(len(df_0), len(df_1), len(df_2))

    # Downsample majority classes
    if len(df_0) > min_samples:
        df_0_downsampled = resample(df_0, replace=False, n_samples=min_samples, random_state=42)
    else:
        df_0_downsampled = df_0

    if len(df_1) > min_samples:
        df_1_downsampled = resample(df_1, replace=False, n_samples=min_samples, random_state=42)
    else:
        df_1_downsampled = df_1

    if len(df_2) > min_samples:
        df_2_downsampled = resample(df_2, replace=False, n_samples=min_samples, random_state=42)
    else:
        df_2_downsampled = df_2

    # Combine
    df_balanced = pd.concat([df_0_downsampled, df_1_downsampled, df_2_downsampled])

    # Shuffle
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split back
    X_balanced = df_balanced.drop(target_col, axis=1)
    y_balanced = df_balanced[target_col]

    return X_balanced, y_balanced

def train_improved_models():
    """Train improved ML models with better techniques"""
    print("=" * 80)
    print("IMPROVING ML MODEL ACCURACY")
    print("=" * 80)

    # Get sample data for training
    symbols = ['BBCA.JK', 'BBRI.JK', 'TLKM.JK', 'UNVR.JK', 'ASII.JK',
              'ANTM.JK', 'BBNI.JK', 'INDF.JK', 'KLBF.JK', 'BMRI.JK']

    all_data = []

    print("\n1. Collecting and preprocessing data...")
    for symbol in symbols:
        try:
            print(f"   Processing {symbol}...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="2y", interval="1d")

            if len(df) > 100:
                # Create features
                df = create_better_features(df)
                df['symbol'] = symbol

                # Remove rows with NaN
                df = df.dropna()

                if len(df) > 50:
                    all_data.append(df)

        except Exception as e:
            print(f"   Error with {symbol}: {e}")
            continue

    # Combine all data
    if not all_data:
        print("No data collected!")
        return

    df_all = pd.concat(all_data, ignore_index=True)
    print(f"\n   Total samples collected: {len(df_all)}")

    # Select features
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'returns_1d', 'returns_5d', 'returns_10d',
        'volatility_5d', 'volatility_20d',
        'sma_5', 'sma_10', 'sma_20',
        'rsi', 'macd', 'volume_ratio',
        'bb_position', 'price_vs_sma5', 'price_vs_sma20'
    ]

    # Prepare features and target
    X = df_all[feature_columns].copy()
    y = df_all['target_1d'].copy()  # Use 1-day target for short-term trading

    # Fix data preprocessing
    X = fix_data_preprocessing(X)

    # Remove any remaining NaN values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]

    print(f"   Clean samples: {len(X)}")
    print(f"   Target distribution: {y.value_counts().to_dict()}")

    # Convert y to Series if it's a numpy array
    if isinstance(y, np.ndarray):
        y = pd.Series(y, name='target')

    # Balance the dataset
    print("\n2. Balancing dataset...")
    X_balanced, y_balanced = balance_dataset(X, y)
    print(f"   Balanced samples: {len(X_balanced)}")
    print(f"   Balanced distribution: {y_balanced.value_counts().to_dict()}")

    # Time series split for more realistic validation
    print("\n3. Splitting data with TimeSeriesSplit...")
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X_balanced))

    # Use last split for validation
    train_idx, val_idx = splits[-1]
    X_train, X_val = X_balanced.iloc[train_idx], X_balanced.iloc[val_idx]
    y_train, y_val = y_balanced.iloc[train_idx], y_balanced.iloc[val_idx]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")

    # Train models with better hyperparameters
    print("\n4. Training improved models...")

    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        ),
        'xgboost': XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )
    }

    results = {}

    for name, model in models.items():
        print(f"\n   Training {name}...")

        # Train
        model.fit(X_train_scaled, y_train)

        # Predictions
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)

        # Calculate accuracy
        train_acc = (train_pred == y_train).mean()
        val_acc = (val_pred == y_val).mean()

        results[name] = {
            'model': model,
            'train_accuracy': train_acc,
            'validation_accuracy': val_acc,
            'predictions': val_pred
        }

        print(f"   Training Accuracy: {train_acc:.2%}")
        print(f"   Validation Accuracy: {val_acc:.2%}")

        # Classification report
        print(f"\n   Classification Report for {name}:")
        print(classification_report(y_val, val_pred, target_names=['Sell', 'Hold', 'Buy']))

    # Ensemble predictions
    print("\n5. Creating ensemble model...")
    ensemble_preds = np.zeros(len(X_val_scaled))
    for name, result in results.items():
        ensemble_preds += result['predictions']
    ensemble_preds /= len(results)
    ensemble_pred_labels = np.round(ensemble_preds).astype(int)

    ensemble_acc = (ensemble_pred_labels == y_val).mean()
    print(f"\n   Ensemble Validation Accuracy: {ensemble_acc:.2%}")

    # Feature importance
    print("\n6. Top 10 Feature Importances (Random Forest):")
    rf_model = results['random_forest']['model']
    importances = rf_model.feature_importances_
    feature_importance = list(zip(feature_columns, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for feature, importance in feature_importance[:10]:
        print(f"   {feature}: {importance:.4f}")

    # Save improved models
    print("\n7. Saving improved models...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save models and scaler
    import joblib
    save_path = f"ml_system/models/improved_models_{timestamp}"
    os.makedirs(save_path, exist_ok=True)

    for name, result in results.items():
        joblib.dump(result['model'], f"{save_path}/{name}.pkl")
    joblib.dump(scaler, f"{save_path}/scaler.pkl")

    # Save results
    results_summary = {
        'timestamp': timestamp,
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'feature_count': len(feature_columns),
        'models': {
            name: {
                'train_accuracy': result['train_accuracy'],
                'validation_accuracy': result['validation_accuracy']
            }
            for name, result in results.items()
        },
        'ensemble_accuracy': ensemble_acc,
        'feature_importances': dict(feature_importance[:10]),
        'target_distribution': y_balanced.value_counts().to_dict()
    }

    with open(f"{save_path}/results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nModels saved to: {save_path}")

    # Recommendations for improvement
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR FURTHER IMPROVEMENT:")
    print("=" * 80)
    print("""
1. Add more features:
   - Market sentiment scores
   - Economic indicators (inflation, interest rates)
   - Sector performance
   - News sentiment analysis
   - Social media sentiment

2. Use more sophisticated models:
   - LSTM/GRU for sequence data
   - Transformer models for multivariate time series
   - Neural networks with attention mechanisms

3. Improve data quality:
   - Include more historical data (5+ years)
   - Add more stocks from different sectors
   - Use intraday data for better signals

4. Advanced techniques:
   - Walk-forward validation
   - Bayesian optimization for hyperparameters
   - Ensemble of different model types
   - Calibration for probability outputs

5. Real-time updates:
   - Retrain models monthly
   - Online learning for new patterns
   - Feature engineering for current market conditions
    """)

if __name__ == "__main__":
    train_improved_models()