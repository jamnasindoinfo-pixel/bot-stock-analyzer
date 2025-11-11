#!/usr/bin/env python3
"""
ML Predictor v4 - Enhanced version with accuracy improvements
Key improvements:
1. More sophisticated features
2. Better target creation with dynamic thresholds
3. Advanced ensemble methods
4. Feature selection
5. Hyperparameter optimization
6. Regularization to prevent overfitting
7. Multi-timeframe features
"""

import os
import json
import logging
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from scipy import stats

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available. Install with: pip install xgboost")

# Try to import LightGBM
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not available. Install with: pip install lightgbm")

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class MLPredictorV4:
    """Enhanced ML Predictor with accuracy improvements"""

    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_columns = None
        self.feature_selector = None
        self.model_info = {}
        self.trained = False
        self.target_thresholds = None

    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced technical features

        Returns:
            DataFrame with enhanced features
        """
        df = df.copy()

        # Ensure we have enough data
        if len(df) < 60:
            logger.warning("Insufficient data for advanced features")
            return df

        # Basic price features
        df['returns_1d'] = df['Close'].pct_change(1)
        df['returns_2d'] = df['Close'].pct_change(2)
        df['returns_3d'] = df['Close'].pct_change(3)
        df['returns_5d'] = df['Close'].pct_change(5)
        df['returns_10d'] = df['Close'].pct_change(10)
        df['returns_20d'] = df['Close'].pct_change(20)

        # Log returns for better distribution
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # High-Low spread (volatility indicator)
        df['hl_spread'] = (df['High'] - df['Low']) / df['Close']
        df['oc_spread'] = (df['Close'] - df['Open']) / df['Open']

        # Moving averages - multiple periods
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
            df[f'price_to_sma_{period}'] = df['Close'] / df[f'sma_{period}'] - 1
            df[f'price_to_ema_{period}'] = df['Close'] / df[f'ema_{period}'] - 1

        # Moving average crossovers
        df['sma_cross_5_20'] = (df['sma_5'] > df['sma_20']).astype(int)
        df['ema_cross_10_20'] = (df['ema_10'] > df['ema_20']).astype(int)

        # RSI with multiple periods
        for period in [7, 14, 21]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # Stochastic Oscillator
        low_min = df['Low'].rolling(14).min()
        high_max = df['High'].rolling(14).max()
        df['stoch_k'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        # Williams %R
        df['williams_r'] = -100 * (high_max - df['Close']) / (high_max - low_min)

        # Bollinger Bands with multiple std deviations
        for period, std in [(20, 2), (20, 2.5), (10, 1.5)]:
            bb_mid = df['Close'].rolling(period).mean()
            bb_std = df['Close'].rolling(period).std()
            df[f'bb_upper_{period}_{std}'] = bb_mid + (bb_std * std)
            df[f'bb_lower_{period}_{std}'] = bb_mid - (bb_std * std)
            df[f'bb_position_{period}_{std}'] = (df['Close'] - df[f'bb_lower_{period}_{std}']) / (df[f'bb_upper_{period}_{std}'] - df[f'bb_lower_{period}_{std}'])
            df[f'bb_width_{period}_{std}'] = (df[f'bb_upper_{period}_{std}'] - df[f'bb_lower_{period}_{std}']) / df['Close']

        # MACD with multiple signal periods
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # ADX (Average Directional Index) - trend strength
        high_diff = df['High'].diff()
        low_diff = -df['Low'].diff()
        df['plus_dm'] = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        df['minus_dm'] = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        atr1 = pd.DataFrame({
            'high': df['High'],
            'low': df['Low'],
            'close': df['Close']
        }).assign(
            tr=lambda x: np.maximum(
                x['high'] - x['low'],
                np.maximum(
                    abs(x['high'] - x['close'].shift(1)),
                    abs(x['low'] - x['close'].shift(1))
                )
            )
        )['tr'].rolling(14).mean()

        df['plus_di'] = 100 * (df['plus_dm'].rolling(14).mean() / atr1)
        df['minus_di'] = 100 * (df['minus_dm'].rolling(14).mean() / atr1)
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].rolling(14).mean()

        # Volume features
        df['volume_sma_10'] = df['Volume'].rolling(10).mean()
        df['volume_sma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio_10'] = df['Volume'] / df['volume_sma_10']
        df['volume_ratio_20'] = df['Volume'] / df['volume_sma_20']

        # Price-Volume indicators
        df['pvi'] = df['returns_1d'] * df['Volume']
        df['vwap'] = (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
        df['price_to_vwap'] = df['Close'] / df['vwap'] - 1

        # Volatility features
        df['volatility_5d'] = df['returns_1d'].rolling(5).std()
        df['volatility_10d'] = df['returns_1d'].rolling(10).std()
        df['volatility_20d'] = df['returns_1d'].rolling(20).std()
        df['atr_ratio'] = atr1 / df['Close']

        # Momentum indicators
        df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['roc_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)) * 100
        df['roc_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100

        # Rate of change of indicators
        df['rsi_14_roc'] = df['rsi_14'].diff(1)
        df['macd_roc'] = df['macd'].diff(1)

        # Price patterns
        df['higher_high'] = (df['High'] > df['High'].shift(1)).rolling(3).sum()
        df['lower_low'] = (df['Low'] < df['Low'].shift(1)).rolling(3).sum()

        # Fibonacci retracements (simplified)
        df['fib_236'] = df['High'].rolling(20).max() - 0.236 * (df['High'].rolling(20).max() - df['Low'].rolling(20).min())
        df['fib_382'] = df['High'].rolling(20).max() - 0.382 * (df['High'].rolling(20).max() - df['Low'].rolling(20).min())
        df['fib_618'] = df['High'].rolling(20).max() - 0.618 * (df['High'].rolling(20).max() - df['Low'].rolling(20).min())

        # Support and Resistance levels
        df['resistance_10'] = df['High'].rolling(10).max()
        df['support_10'] = df['Low'].rolling(10).min()
        df['resistance_20'] = df['High'].rolling(20).max()
        df['support_20'] = df['Low'].rolling(20).min()

        # Distance from support/resistance
        df['dist_to_resistance_10'] = (df['resistance_10'] - df['Close']) / df['Close']
        df['dist_to_support_10'] = (df['Close'] - df['support_10']) / df['Close']

        # Gap features
        df['gap_up'] = (df['Low'] > df['High'].shift(1)).astype(int)
        df['gap_down'] = (df['High'] < df['Low'].shift(1)).astype(int)
        df['gap_size'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

        # Time-based features
        df['day_of_week'] = pd.to_datetime(df.index).dayofweek
        df['month'] = pd.to_datetime(df.index).month
        df['quarter'] = pd.to_datetime(df.index).quarter

        # Cyclical encoding for time features
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Market regime detection
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['sma_200'] = df['Close'].rolling(200).mean()
        df['bull_market'] = (df['sma_50'] > df['sma_200']).astype(int)
        df['above_sma_20'] = (df['Close'] > df['sma_20']).astype(int)
        df['above_sma_50'] = (df['Close'] > df['sma_50']).astype(int)

        return df

    def create_dynamic_target(self, df: pd.DataFrame,
                            lookforward_days: int = 5,
                            volatility_window: int = 20) -> pd.Series:
        """
        Create dynamic targets based on volatility-adjusted thresholds

        Args:
            df: DataFrame with price data
            lookforward_days: Days to look forward for target
            volatility_window: Window for volatility calculation

        Returns:
            Series with target values (0=SELL, 1=HOLD, 2=BUY)
        """
        df = df.copy()

        # Calculate future returns
        df['future_return'] = df['Close'].shift(-lookforward_days) / df['Close'] - 1

        # Calculate rolling volatility
        df['volatility'] = df['Close'].pct_change().rolling(volatility_window).std()

        # Dynamic thresholds based on volatility
        # Higher volatility = higher threshold for signals
        base_threshold = 0.02  # 2% base
        volatility_multiplier = 2.0  # Multiply by 2x volatility

        df['buy_threshold'] = base_threshold + (volatility_multiplier * df['volatility'])
        df['sell_threshold'] = -(base_threshold + (volatility_multiplier * df['volatility']))

        # Create targets
        df['target'] = 1  # Default to HOLD
        df.loc[df['future_return'] > df['buy_threshold'], 'target'] = 2  # BUY
        df.loc[df['future_return'] < df['sell_threshold'], 'target'] = 0  # SELL

        # Store thresholds for model info
        self.target_thresholds = {
            'buy_threshold_mean': df['buy_threshold'].mean(),
            'sell_threshold_mean': df['sell_threshold'].mean(),
            'volatility_mean': df['volatility'].mean()
        }

        return df['target']

    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 50) -> Tuple[pd.DataFrame, SelectKBest]:
        """
        Select top k features using univariate statistical tests

        Args:
            X: Feature DataFrame
            y: Target Series
            k: Number of features to select

        Returns:
            Tuple of (selected features DataFrame, selector object)
        """
        logger.info(f"Selecting top {k} features from {X.shape[1]} total features...")

        # Remove non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]

        # Handle any remaining NaN or inf values
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Feature selection
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X_numeric, y)

        # Get selected feature names
        selected_features = X_numeric.columns[selector.get_support()].tolist()

        logger.info(f"Selected {len(selected_features)} features")

        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selector

    def train_ensemble_models(self, X: pd.DataFrame, y: pd.Series,
                            test_size: int = 100,
                            n_estimators: int = 300,
                            use_feature_selection: bool = True) -> Dict:
        """
        Train ensemble models with advanced techniques

        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Number of samples for test set
            n_estimators: Number of trees for ensemble models
            use_feature_selection: Whether to use feature selection

        Returns:
            Dictionary with training results
        """
        logger.info(f"Training enhanced ensemble models with {len(X)} samples...")

        # Feature selection
        if use_feature_selection:
            X, self.feature_selector = self.select_features(X, y, k=min(50, X.shape[1]))
        else:
            self.feature_selector = None

        self.feature_columns = X.columns.tolist()

        # Time series split
        tscv = TimeSeriesSplit(n_splits=3, test_size=test_size)
        splits = list(tscv.split(X, y))

        # Use the last split for validation
        train_idx, val_idx = splits[-1]
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")

        # Use RobustScaler for better handling of outliers
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Define models with optimized hyperparameters
        models = {}

        # Random Forest with regularization
        models['rf'] = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=15,  # Prevent overfitting
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        )

        # XGBoost (if available)
        if HAS_XGBOOST:
            models['xgb'] = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=6,
                learning_rate=0.05,  # Lower learning rate
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=1,  # Regularization
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1,  # L2 regularization
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )

        # LightGBM (if available)
        if HAS_LGBM:
            models['lgb'] = LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )

        # Gradient Boosting
        models['gb'] = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            max_features='sqrt',
            random_state=42
        )

        # Logistic Regression (as baseline)
        models['lr'] = LogisticRegression(
            C=1.0,  # Inverse regularization strength
            penalty='l2',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )

        # Train models and collect results
        results = {}
        trained_models = {}

        for name, model in models.items():
            logger.info(f"Training {name}...")
            start_time = time.time()

            model.fit(X_train_scaled, y_train)

            # Predictions
            y_train_pred = model.predict(X_train_scaled)
            y_val_pred = model.predict(X_val_scaled)

            # Probabilities
            y_train_proba = model.predict_proba(X_train_scaled)
            y_val_proba = model.predict_proba(X_val_scaled)

            # Accuracy
            train_acc = (y_train_pred == y_train).mean()
            val_acc = (y_val_pred == y_val).mean()

            # Log loss for probabilistic evaluation
            from sklearn.metrics import log_loss
            train_logloss = log_loss(y_train, y_train_proba)
            val_logloss = log_loss(y_val, y_val_proba)

            training_time = time.time() - start_time

            logger.info(f"{name.upper()} - Train: {train_acc:.3f}, Val: {val_acc:.3f}, LogLoss: {val_logloss:.3f}, Time: {training_time:.2f}s")

            results[name] = {
                'model': model,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'train_logloss': train_logloss,
                'val_logloss': val_logloss,
                'training_time': training_time
            }

            trained_models[name] = model

        # Create ensemble using VotingClassifier
        logger.info("Creating ensemble model...")

        # Select top 3 models based on validation accuracy
        top_models = sorted(results.items(), key=lambda x: x[1]['val_accuracy'], reverse=True)[:3]
        ensemble_estimators = [(name, result['model']) for name, result in top_models]

        # Weighted voting based on validation performance
        weights = [result['val_accuracy'] for _, result in top_models]

        ensemble = VotingClassifier(
            estimators=ensemble_estimators,
            voting='soft',
            weights=weights
        )

        # Train ensemble
        start_time = time.time()
        ensemble.fit(X_train_scaled, y_train)
        ensemble_time = time.time() - start_time

        # Ensemble predictions
        y_train_ensemble = ensemble.predict(X_train_scaled)
        y_val_ensemble = ensemble.predict(X_val_scaled)
        y_val_ensemble_proba = ensemble.predict_proba(X_val_scaled)

        ensemble_train_acc = (y_train_ensemble == y_train).mean()
        ensemble_val_acc = (y_val_ensemble == y_val).mean()
        ensemble_logloss = log_loss(y_val, y_val_ensemble_proba)

        logger.info(f"ENSEMBLE - Train: {ensemble_train_acc:.3f}, Val: {ensemble_val_acc:.3f}, LogLoss: {ensemble_logloss:.3f}, Time: {ensemble_time:.2f}s")

        # Store models
        self.models = trained_models
        self.models['ensemble'] = ensemble

        # Model info
        self.model_info = {
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'feature_count': len(self.feature_columns),
            'target_distribution': y_train.value_counts().to_dict(),
            'models': {},
            'target_thresholds': self.target_thresholds,
            'feature_importance': {},
            'trained_at': datetime.now().isoformat(),
            'version': 'v4'
        }

        # Add model results to info
        for name, result in results.items():
            self.model_info['models'][name] = {
                'val_accuracy': result['val_accuracy'],
                'val_logloss': result['val_logloss'],
                'training_time': result['training_time']
            }

        self.model_info['models']['ensemble'] = {
            'val_accuracy': ensemble_val_acc,
            'val_logloss': ensemble_logloss,
            'training_time': ensemble_time
        }

        # Feature importance from Random Forest
        if 'rf' in self.models:
            rf_importance = self.models['rf'].feature_importances_
            feature_importance = dict(zip(self.feature_columns, rf_importance))
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            self.model_info['feature_importance'] = feature_importance

        self.trained = True

        logger.info("Enhanced training completed successfully!")

        return self.model_info

    def prepare_training_data(self, symbols: List[str] = None,
                            period: str = "3y",
                            min_samples: int = 100) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data with enhanced features

        Args:
            symbols: List of stock symbols
            period: Historical period to fetch
            min_samples: Minimum samples required

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if symbols is None:
            # Default Indonesian stocks - expanded list
            symbols = [
                'BBCA.JK', 'BBRI.JK', 'BBNI.JK', 'BMRI.JK', 'TLKM.JK',
                'UNVR.JK', 'INDF.JK', 'ICBP.JK', 'ASII.JK', 'KLBF.JK',
                'EXCL.JK', 'ISAT.JK', 'GOTO.JK', 'ANTM.JK', 'PTBA.JK',
                'TINS.JK', 'SMGR.JK', 'ADHI.JK', 'PTPP.JK', 'WIKA.JK',
                'JPFA.JK', 'CPIN.JK', 'SRIL.JK', 'ELSA.JK', 'MEDC.JK'
            ]

        all_data = []

        logger.info(f"Downloading data for {len(symbols)} symbols...")

        for i, symbol in enumerate(symbols):
            try:
                logger.info(f"  Downloading {symbol} ({i+1}/{len(symbols)})...")

                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval="1d")

                if len(df) < min_samples:
                    logger.warning(f"  {symbol}: Insufficient data ({len(df)} < {min_samples})")
                    continue

                # Add symbol column
                df['Symbol'] = symbol

                # Create advanced features
                df = self.create_advanced_features(df)

                # Create dynamic target
                df['Target'] = self.create_dynamic_target(df)

                # Remove NaN rows
                df = df.dropna()

                if len(df) < 50:
                    continue

                all_data.append(df)

            except Exception as e:
                logger.error(f"  Error processing {symbol}: {str(e)}")
                continue

        if not all_data:
            raise ValueError("No valid data obtained")

        logger.info(f"Combining data from {len(all_data)} symbols...")

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Remove non-feature columns
        feature_cols = [col for col in combined_df.columns
                       if col not in ['Symbol', 'Target', 'Open', 'High', 'Low', 'Close', 'Volume',
                                     'Dividends', 'Stock Splits', 'future_return', 'volatility',
                                     'buy_threshold', 'sell_threshold']]

        X = combined_df[feature_cols]
        y = combined_df['Target']

        # Remove any remaining NaN or infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        logger.info(f"Total training samples: {len(X)}")
        logger.info(f"Target distribution: {dict(y.value_counts())}")

        return X, y

    def predict(self, df: pd.DataFrame,
                ensemble_method: str = 'ensemble',
                verbose: bool = False) -> Dict:
        """
        Make predictions with enhanced models

        Args:
            df: DataFrame with OHLCV data
            ensemble_method: Which model to use for prediction
            verbose: Whether to show detailed info

        Returns:
            Dictionary with predictions
        """
        if not self.trained:
            return {'error': 'Model not trained', 'predictions': []}

        if len(df) < 50:
            logger.warning(f"Insufficient data for prediction: {len(df)}")
            return {'error': 'Insufficient data', 'predictions': []}

        try:
            # Create features
            df_features = self.create_advanced_features(df)
            df_features = df_features.dropna()

            if len(df_features) == 0:
                logger.warning("No valid data after feature creation")
                return {'error': 'No valid data', 'predictions': []}

            # Select features
            available_features = [col for col in self.feature_columns
                                 if col in df_features.columns]

            if len(available_features) < len(self.feature_columns):
                missing_features = set(self.feature_columns) - set(available_features)
                logger.warning(f"Missing features: {missing_features}")

            X = df_features[available_features]

            # Adjust features if needed
            if len(available_features) != len(self.feature_columns):
                X_adjusted = pd.DataFrame(0, index=X.index, columns=self.feature_columns)
                X_adjusted[available_features] = X.values
                X = X_adjusted

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Make predictions
            if ensemble_method in self.models:
                model = self.models[ensemble_method]
                predictions = model.predict(X_scaled)
                probabilities = model.predict_proba(X_scaled)
            else:
                logger.warning(f"Model {ensemble_method} not found, using ensemble")
                model = self.models['ensemble']
                predictions = model.predict(X_scaled)
                probabilities = model.predict_proba(X_scaled)

            # Convert to signal names
            signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            signals = [signal_map[p] for p in predictions]

            # Create results
            results = []
            for i in range(len(signals)):
                proba_dict = dict(zip(signal_map.values(), probabilities[i]))

                # Safely get date - handle both datetime and numeric indices
                try:
                    if hasattr(df_features.index[i], 'strftime'):
                        date_str = df_features.index[i].strftime('%Y-%m-%d')
                    else:
                        # If index is not datetime, use a default date
                        date_str = datetime.now().strftime('%Y-%m-%d')
                except Exception:
                    date_str = datetime.now().strftime('%Y-%m-%d')

                result = {
                    'date': date_str,
                    'signal': signals[i],
                    'probabilities': proba_dict,
                    'confidence': np.max(probabilities[i])
                }
                results.append(result)

            if verbose:
                logger.info(f"Prediction completed for {len(results)} samples")
                logger.info(f"Signal distribution: {dict(pd.Series(signals).value_counts())}")

            return {
                'success': True,
                'predictions': results,
                'model_info': self.model_info,
                'features_used': available_features
            }

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {'error': str(e), 'predictions': []}

    def save_models(self, filepath: str):
        """Save enhanced models to disk"""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'feature_selector': self.feature_selector,
            'model_info': self.model_info,
            'trained': self.trained,
            'target_thresholds': self.target_thresholds,
            'version': 'v4'
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)

        logger.info(f"Models saved to: {filepath}")
        return filepath

    def load_models(self, filepath: str):
        """Load enhanced models from disk"""
        model_data = joblib.load(filepath)

        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.feature_selector = model_data.get('feature_selector')
        self.model_info = model_data['model_info']
        self.trained = model_data['trained']
        self.target_thresholds = model_data.get('target_thresholds')

        logger.info(f"Models loaded from: {filepath}")
        return True

    def get_model_summary(self) -> Dict:
        """Get comprehensive model summary"""
        if not self.trained:
            return {'error': 'Model not trained'}

        summary = {
            'version': 'v4',
            'trained_at': self.model_info.get('trained_at'),
            'training_samples': self.model_info.get('training_samples'),
            'validation_samples': self.model_info.get('validation_samples'),
            'features': {
                'total': len(self.feature_columns),
                'selected': len(self.feature_columns) if self.feature_selector else 'All'
            },
            'target_distribution': self.model_info.get('target_distribution'),
            'target_thresholds': self.model_info.get('target_thresholds'),
            'model_performance': {}
        }

        # Add model performances
        for name, info in self.model_info.get('models', {}).items():
            summary['model_performance'][name] = {
                'accuracy': f"{info['val_accuracy']:.3f}",
                'log_loss': f"{info['val_logloss']:.3f}",
                'training_time': f"{info['training_time']:.2f}s"
            }

        # Add top features
        if 'feature_importance' in self.model_info:
            top_features = dict(list(self.model_info['feature_importance'].items())[:10])
            summary['top_features'] = top_features

        return summary