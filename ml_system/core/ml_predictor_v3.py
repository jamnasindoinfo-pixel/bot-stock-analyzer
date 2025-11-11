#!/usr/bin/env python3
"""
ML Predictor v3 - Complete rewrite with proper functionality
Fixed version that actually works correctly
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

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLPredictorV3:
    """
    Enhanced ML Predictor with proper data handling
    """

    def __init__(self):
        """Initialize the predictor"""
        self.trained = False
        self.models = {}
        self.scaler = None
        self.feature_columns = []
        self.model_info = {}

        # Try to load existing models
        self.load_models()

    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive technical features from OHLCV data

        Args:
            df: DataFrame with OHLCV data (index should be datetime)

        Returns:
            DataFrame with technical features
        """
        df = df.copy()

        # 1. Price-based features
        # Returns
        df['returns_1d'] = df['Close'].pct_change()
        df['returns_3d'] = df['Close'].pct_change(3)
        df['returns_5d'] = df['Close'].pct_change(5)
        df['returns_10d'] = df['Close'].pct_change(10)
        df['returns_20d'] = df['Close'].pct_change(20)

        # Log returns
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # 2. Volatility features
        df['volatility_5d'] = df['returns_1d'].rolling(5).std()
        df['volatility_10d'] = df['returns_1d'].rolling(10).std()
        df['volatility_20d'] = df['returns_1d'].rolling(20).std()
        df['volatility_ratio'] = df['volatility_5d'] / df['volatility_20d']

        # 3. Moving averages
        df['sma_5'] = df['Close'].rolling(5).mean()
        df['sma_10'] = df['Close'].rolling(10).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()

        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()

        # Distance from moving averages
        df['price_sma5_ratio'] = df['Close'] / df['sma_5']
        df['price_sma20_ratio'] = df['Close'] / df['sma_20']
        df['price_sma50_ratio'] = df['Close'] / df['sma_50']

        # Moving average crossovers
        df['sma_cross_5_20'] = np.where(df['sma_5'] > df['sma_20'], 1, 0)
        df['ema_cross_12_26'] = np.where(df['ema_12'] > df['ema_26'], 1, 0)

        # 4. RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # RSI levels
        df['rsi_overbought'] = np.where(df['rsi'] > 70, 1, 0)
        df['rsi_oversold'] = np.where(df['rsi'] < 30, 1, 0)
        df['rsi_neutral'] = np.where((df['rsi'] >= 30) & (df['rsi'] <= 70), 1, 0)

        # 5. MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # MACD signals
        df['macd_bullish'] = np.where(df['macd'] > df['macd_signal'], 1, 0)
        df['macd_zero_cross'] = np.where(np.sign(df['macd'].shift(1)) != np.sign(df['macd']), 1, 0)

        # 6. Bollinger Bands
        df['bb_middle'] = df['sma_20']
        bb_std = df['Close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
        df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Bollinger Bands signals
        df['bb_squeeze'] = np.where(df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.8, 1, 0)
        df['bb_breakout_upper'] = np.where(df['Close'] > df['bb_upper'], 1, 0)
        df['bb_breakout_lower'] = np.where(df['Close'] < df['bb_lower'], 1, 0)

        # 7. Stochastic Oscillator
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        df['stoch_k'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        # Stochastic signals
        df['stoch_overbought'] = np.where(df['stoch_k'] > 80, 1, 0)
        df['stoch_oversold'] = np.where(df['stoch_k'] < 20, 1, 0)
        df['stoch_cross'] = np.where(df['stoch_k'] > df['stoch_d'], 1, 0)

        # 8. Williams %R
        df['williams_r'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
        df['williams_overbought'] = np.where(df['williams_r'] > -20, 1, 0)
        df['williams_oversold'] = np.where(df['williams_r'] < -80, 1, 0)

        # 9. Volume indicators
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        df['volume_price_trend'] = np.where((df['Close'] > df['Close'].shift(1)) &
                                      (df['Volume'] > df['volume_sma']), 1, 0)

        # 10. Price patterns
        # Higher Highs and Lower Lows
        df['hh_5d'] = df['High'].rolling(5).max()
        df['ll_5d'] = df['Low'].rolling(5).min()
        df['hh_10d'] = df['High'].rolling(10).max()
        df['ll_10d'] = df['Low'].rolling(10).min()
        df['hh_20d'] = df['High'].rolling(20).max()
        df['ll_20d'] = df['Low'].rolling(20).min()

        # Position within range
        df['position_5d'] = (df['Close'] - df['ll_5d']) / (df['hh_5d'] - df['ll_5d'])
        df['position_10d'] = (df['Close'] - df['ll_10d']) / (df['hh_10d'] - df['ll_10d'])
        df['position_20d'] = (df['Close'] - df['ll_20d']) / (df['hh_20d'] - df['ll_20d'])

        # 11. Trend indicators
        # ADX (simplified version)
        up = df['High'] - df['High'].shift(1)
        down = df['Low'].shift(1) - df['Low']
        df['plus_dm'] = np.where(up > down, up, 0)
        df['minus_dm'] = np.where(down > up, down, 0)
        df['tr'] = np.maximum(df['High'] - df['Low'],
                               np.abs(df['High'] - df['Close'].shift(1)),
                               np.abs(df['Low'] - df['Close'].shift(1)))

        df['plus_di'] = df['plus_dm'].rolling(14).mean()
        df['minus_di'] = df['minus_dm'].rolling(14).mean()
        df['dx'] = 100 * (np.abs(df['plus_di'] - df['minus_di']) /
                       (df['plus_di'] + df['minus_di']))
        df['adx'] = df['dx'].rolling(14).mean()

        df['adx_trending'] = np.where(df['adx'] > 25, 1, 0)

        # 12. Momentum indicators
        # Rate of Change (ROC)
        df['roc_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)) * 100
        df['roc_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        df['roc_20'] = ((df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)) * 100

        # Momentum
        df['momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['momentum_10'] = df['Close'] - df['Close'].shift(10)

        # Rate of Change (rate of change of rate of change)
        df['roc_roc_5'] = df['roc_5'] - df['roc_5'].shift(1)
        df['roc_roc_10'] = df['roc_10'] - df['roc_10'].shift(1)

        # 13. Lag features
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns_1d'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
            df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)

        # 14. Statistical features
        # Rolling statistics
        df['close_mean_20'] = df['Close'].rolling(20).mean()
        df['close_std_20'] = df['Close'].rolling(20).std()
        df['close_zscore'] = (df['Close'] - df['close_mean_20']) / df['close_std_20']

        df['volume_mean_20'] = df['Volume'].rolling(20).mean()
        df['volume_std_20'] = df['Volume'].rolling(20).std()
        df['volume_zscore'] = (df['Volume'] - df['volume_mean_20']) / df['volume_std_20']

        return df

    def create_target(self, df: pd.DataFrame,
                      target_type: str = 'multiclass',
                      forward_days: int = 5,
                      profit_threshold: float = 0.02) -> pd.Series:
        """
        Create target variable for training

        Args:
            df: DataFrame with OHLCV data
            target_type: 'binary', 'multiclass', or 'regression'
            forward_days: Days ahead to predict
            profit_threshold: Profit threshold for buy/sell signals

        Returns:
            Series with target values
        """
        if target_type == 'binary':
            # Binary: Buy (1) or Don't Buy (0)
            future_return = df['Close'].shift(-forward_days) / df['Close'] - 1
            target = (future_return > profit_threshold).astype(int)

        elif target_type == 'multiclass':
            # Multiclass: Buy (2), Hold (1), Sell (0)
            future_return = df['Close'].shift(-forward_days) / df['Close'] - 1
            target = np.where(future_return > profit_threshold, 2,
                           np.where(future_return < -profit_threshold, 0, 1))

        elif target_type == 'regression':
            # Regression: Predict actual return
            target = df['Close'].shift(-forward_days) / df['Close'] - 1

        else:
            raise ValueError(f"Invalid target_type: {target_type}")

        return target

    def prepare_training_data(self,
                              symbols: List[str] = None,
                              period: str = "2y",
                              target_type: str = 'multiclass',
                              min_samples: int = 50) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from multiple symbols

        Returns:
            X: Feature DataFrame
            y: Target Series
        """
        if symbols is None:
            # Default Indonesian stocks
            symbols = [
                'BBCA.JK', 'BBRI.JK', 'BBNI.JK', 'BMRI.JK', 'TLKM.JK',
                'UNVR.JK', 'INDF.JK', 'ICBP.JK', 'ASII.JK', 'KLBF.JK',
                'EXCL.JK', 'ISAT.JK', 'GOTO.JK', 'ANTM.JK', 'PTBA.JK',
                'TINS.JK', 'SMGR.JK', 'ADHI.JK', 'PTPP.JK', 'WIKA.JK'
            ]

        all_data = []

        logger.info(f"Downloading data for {len(symbols)} symbols...")

        for i, symbol in enumerate(symbols):
            try:
                logger.info(f"  Downloading {symbol} ({i+1}/{len(symbols)})...")

                # Download data
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval="1d")

                if len(df) < min_samples:
                    logger.warning(f"  {symbol}: Insufficient data ({len(df)} < {min_samples})")
                    continue

                # Add symbol column BEFORE creating features
                df['Symbol'] = symbol

                # Create features
                df = self.create_technical_features(df)

                # Create target
                df['Target'] = self.create_target(df, target_type=target_type)

                # Keep only complete rows
                df = df.dropna()

                # Only keep if we have target
                if not df['Target'].isna().all():
                    all_data.append(df)
                    logger.info(f"  {symbol}: {len(df)} training samples")

            except Exception as e:
                logger.error(f"  Error processing {symbol}: {str(e)}")
                continue

        if not all_data:
            raise ValueError("No valid data available for training")

        # Combine all data
        logger.info(f"Combining data from {len(all_data)} symbols...")
        combined_df = pd.concat(all_data, ignore_index=True)

        # Remove Symbol column (it's not a feature)
        feature_cols = [col for col in combined_df.columns
                       if col not in ['Symbol', 'Target']]

        X = combined_df[feature_cols].copy()
        y = combined_df['Target'].copy()

        logger.info(f"Total training samples: {len(X)}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")

        return X, y

    def train_models(self,
                    X: pd.DataFrame,
                    y: pd.Series,
                    test_size: float = 0.2,
                    n_estimators: int = 100,
                    random_state: int = 42) -> Dict:
        """
        Train ML models with proper validation

        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion for testing
            n_estimators: Number of trees for ensemble models

        Returns:
            Dictionary with training results
        """
        logger.info(f"Training models with {len(X)} samples...")

        # Time series split for proper validation
        # Use fewer splits to avoid error with smaller datasets
        n_splits = min(3, max(2, len(X) // (test_size * 4)))
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        splits = list(tscv.split(X, y))

        # Use the last split for validation (most recent data)
        train_idx, val_idx = splits[-1]
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Store feature columns
        self.feature_columns = X.columns.tolist()

        # Train Random Forest
        logger.info("Training Random Forest...")
        rf_start = time.time()

        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,  # Use all CPU cores
            class_weight='balanced'
        )

        rf_model.fit(X_train_scaled, y_train)
        rf_time = time.time() - rf_start

        rf_train_score = rf_model.score(X_train_scaled, y_train)
        rf_val_score = rf_model.score(X_val_scaled, y_val)

        logger.info(f"Random Forest trained in {rf_time:.2f}s")
        logger.info(f"Random Forest - Train: {rf_train_score:.3f}, Val: {rf_val_score:.3f}")

        # Train XGBoost
        logger.info("Training XGBoost...")
        xgb_start = time.time()

        xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            eval_metric='mlogloss',
            use_label_encoder=False
        )

        xgb_model.fit(X_train_scaled, y_train)
        xgb_time = time.time() - xgb_start

        xgb_train_score = xgb_model.score(X_train_scaled, y_train)
        xgb_val_score = xgb_model.score(X_val_scaled, y_val)

        logger.info(f"XGBoost trained in {xgb_time:.2f}s")
        logger.info(f"XGBoost - Train: {xgb_train_score:.3f}, Val: {xgb_val_score:.3f}")

        # Store models
        self.models = {
            'random_forest': rf_model,
            'xgboost': xgb_model
        }

        # Store model info
        self.model_info = {
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'feature_count': len(X.columns),
            'target_distribution': y_train.value_counts().to_dict(),
            'models': {
                'random_forest': {
                    'train_accuracy': rf_train_score,
                    'val_accuracy': rf_val_score,
                    'training_time': rf_time
                },
                'xgboost': {
                    'train_accuracy': xgb_train_score,
                    'val_accuracy': xgb_val_score,
                    'training_time': xgb_time
                }
            },
            'trained_at': datetime.now().isoformat(),
            'version': 'v3'
        }

        self.trained = True

        # Feature importance
        rf_importance = rf_model.feature_importances_
        feature_importance = dict(zip(self.feature_columns, rf_importance))
        feature_importance = dict(sorted(feature_importance.items(),
                                        key=lambda x: x[1], reverse=True))

        self.model_info['feature_importance'] = feature_importance

        logger.info("Training completed successfully!")

        return self.model_info

    def predict(self,
                df: pd.DataFrame,
                ensemble: str = 'weighted_average',
                verbose: bool = False) -> Dict:
        """
        Make predictions on new data

        Args:
            df: DataFrame with OHLCV data
            ensemble: 'voting', 'weighted_average', or 'best'
            verbose: Whether to show detailed info

        Returns:
            Dictionary with predictions
        """
        if not self.trained:
            logger.error("Model not trained!")
            return {'error': 'Model not trained', 'predictions': []}

        # Ensure we have enough data
        if len(df) < 50:
            logger.warning(f"Insufficient data for prediction: {len(df)}")
            return {'error': 'Insufficient data', 'predictions': []}

        # Create features
        try:
            # Create technical features
            df_features = self.create_technical_features(df)

            # Drop any NaN rows
            df_features = df_features.dropna()

            if len(df_features) == 0:
                logger.warning("No valid data after feature creation")
                return {'error': 'No valid data', 'predictions': []}

            # Select only the features used in training
            available_features = [col for col in self.feature_columns
                                  if col in df_features.columns]

            if len(available_features) < len(self.feature_columns):
                missing_features = set(self.feature_columns) - set(available_features)
                logger.warning(f"Missing features: {missing_features}")

            X = df_features[available_features]

            # If features don't match exactly, adjust
            if len(available_features) != len(self.feature_columns):
                # Create empty DataFrame with correct columns
                X_adjusted = pd.DataFrame(0, index=X.index, columns=self.feature_columns)
                X_adjusted[available_features] = X.values
                X = X_adjusted

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Get predictions from each model
            rf_probs = self.models['random_forest'].predict_proba(X_scaled)
            xgb_probs = self.models['xgboost'].predict_proba(X_scaled)

            # Combine predictions
            if ensemble == 'voting':
                # Simple voting: use Random Forest if uncertain
                rf_preds = self.models['random_forest'].predict(X_scaled)
                xgb_preds = self.models['xgboost'].predict(X_scaled)

                predictions = np.where(rf_preds == xgb_preds, rf_preds, rf_preds)
                probabilities = rf_probs

            elif ensemble == 'weighted_average':
                # Weighted average of probabilities
                # Give higher weight to better performing model
                rf_weight = 0.6 if self.model_info['models']['random_forest']['val_accuracy'] > \
                              self.model_info['models']['xgboost']['val_accuracy'] else 0.4
                xgb_weight = 1 - rf_weight

                probabilities = (rf_probs * rf_weight + xgb_probs * xgb_weight)
                predictions = np.argmax(probabilities, axis=1)

            else:  # best
                # Use the model with better validation accuracy
                best_model = 'random_forest' if self.model_info['models']['random_forest']['val_accuracy'] > \
                                  self.model_info['models']['xgboost']['val_accuracy'] else 'xgboost'

                probabilities = self.models[best_model].predict_proba(X_scaled)
                predictions = self.models[best_model].predict(X_scaled)

            # Create result
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                result = {
                    'symbol': df['Symbol'].iloc[i] if 'Symbol' in df.columns else 'UNKNOWN',
                    'prediction': int(pred),
                    'probabilities': {
                        'sell': float(prob[0]),
                        'hold': float(prob[1]) if len(prob) > 2 else 0.0,
                        'buy': float(prob[2]) if len(prob) > 2 else 0.0
                    },
                    'confidence': float(np.max(prob)),
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)

            if verbose:
                logger.info(f"Generated {len(results)} predictions")
                logger.info(f"Average confidence: {np.mean([r['confidence'] for r in results]):.3f}")

            return {
                'success': True,
                'predictions': results,
                'ensemble': ensemble,
                'model_info': self.model_info
            }

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'predictions': []}

    def predict_signal(self, symbol: str,
                        days_back: int = 100,
                        ensemble: str = 'weighted_average') -> Dict:
        """
        Predict signal for a single symbol

        Args:
            symbol: Stock symbol (e.g., 'BBCA.JK')
            days_back: Number of days of historical data
            ensemble: Ensemble method

        Returns:
            Dictionary with prediction signal
        """
        try:
            # Get data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days_back}d", interval="1d")

            if len(df) == 0:
                return {
                    'symbol': symbol,
                    'signal': 'WAIT',
                    'confidence': 0.0,
                    'error': 'No data available',
                    'success': False
                }

            # Add symbol
            df['Symbol'] = symbol

            # Predict
            result = self.predict(df, ensemble=ensemble, verbose=False)

            if 'error' in result:
                return {
                    'symbol': symbol,
                    'signal': 'WAIT',
                    'confidence': 0.0,
                    'error': result['error'],
                    'success': False
                }

            # Convert prediction to signal
            signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            latest = result['predictions'][-1]

            return {
                'symbol': symbol,
                'signal': signal_map.get(latest['prediction'], 'WAIT'),
                'confidence': latest['confidence'],
                'probabilities': latest['probabilities'],
                'success': True,
                'timestamp': latest['timestamp']
            }

        except Exception as e:
            logger.error(f"Error predicting signal for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'signal': 'WAIT',
                'confidence': 0.0,
                'error': str(e),
                'success': False
            }

    def save_models(self, filepath: str = None):
        """Save models to disk"""
        if filepath is None:
            filepath = f"ml_system/models/ml_predictor_v3_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_info': self.model_info,
            'trained': self.trained,
            'version': 'v3'
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)

        logger.info(f"Models saved to: {filepath}")
        return filepath

    def load_models(self, filepath: str = None):
        """Load models from disk"""
        if filepath is None:
            # Try to find latest model file
            model_dir = "ml_system/models"
            if os.path.exists(model_dir):
                files = [f for f in os.listdir(model_dir) if f.startswith('ml_predictor_v3_') and f.endswith('.pkl')]
                if files:
                    files.sort()
                    filepath = os.path.join(model_dir, files[-1])

        if filepath is None or not os.path.exists(filepath):
            logger.info("No pre-trained models found, starting fresh")
            return False

        try:
            model_data = joblib.load(filepath)

            self.models = model_data.get('models', {})
            self.scaler = model_data.get('scaler')
            self.feature_columns = model_data.get('feature_columns', [])
            self.model_info = model_data.get('model_info', {})
            self.trained = model_data.get('trained', False)

            if self.trained:
                logger.info(f"Models loaded from: {filepath}")
                logger.info(f"Model version: {self.model_info.get('version', 'unknown')}")
                logger.info(f"Features: {len(self.feature_columns)}")
                return True

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False

        return False

    def get_feature_importance(self, top_n: int = 10) -> Dict:
        """Get feature importance from the trained model"""
        if not self.trained or 'feature_importance' not in self.model_info:
            return {}

        # Return top N features
        all_features = self.model_info['feature_importance']
        return dict(list(all_features.items())[:top_n])

    def is_trained(self) -> bool:
        """Check if models are trained"""
        return self.trained

    def get_model_info(self) -> Dict:
        """Get information about the trained models"""
        return self.model_info