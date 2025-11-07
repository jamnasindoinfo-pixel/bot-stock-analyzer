"""
Enhanced ML Predictor v2 - Ensemble model with advanced features

Integrates EnhancedFeatureEngineer (37 features) and EnsembleModel (RF+XGBoost+LSTM)
for improved prediction accuracy targeting 68-70% accuracy.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import logging

# Import enhanced components
from ml_system.core.feature_engineering_v2 import EnhancedFeatureEngineer
from ml_system.core.ensemble_model import EnsembleModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLPredictorV2:
    """
    Enhanced ML predictor v2 using ensemble models and advanced feature engineering

    Target accuracy: 68-70% (vs 64.5% current)
    Features: 37 (vs 14 current)
    Models: 3-Model Ensemble (vs single RF)
    """

    def __init__(self):
        """Initialize enhanced ML predictor"""
        self.ensemble = EnsembleModel()
        self.feature_engineer = EnhancedFeatureEngineer()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # Model state
        self.trained = False
        self.feature_columns = []
        self.model_info = {}

        # Try to load existing models
        self.load_models()

    def predict_signal(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        Make enhanced prediction using ensemble model and advanced features

        Args:
            df: Historical price data
            symbol: Stock symbol (e.g., 'BBCA.JK')

        Returns:
            Dict with comprehensive prediction information
        """
        if not self.trained:
            return {
                'symbol': symbol,
                'signal': 'WAIT',
                'confidence': 0.0,
                'error': 'Enhanced ML models not trained yet',
                'success': False,
                'version': 'v2'
            }

        try:
            # Create enhanced features
            logger.info(f"Creating enhanced features for {symbol}...")
            features_df = self.feature_engineer.create_features(df)

            if features_df.empty or len(features_df) < 10:
                return {
                    'symbol': symbol,
                    'signal': 'WAIT',
                    'confidence': 0.0,
                    'error': 'Insufficient data for feature creation',
                    'success': False,
                    'version': 'v2'
                }

            # Get latest features
            latest_features = features_df.iloc[[-1]][self.feature_columns].fillna(0)
            current_price = df['Close'].iloc[-1]

            # Scale features
            scaled_features = self.scaler.transform(latest_features)

            # Get individual model predictions
            individual_predictions = {}
            all_probabilities = []
            model_confidences = []

            for model_type in self.ensemble.model_types:
                model = self.ensemble.models[model_type]

                # Get prediction
                pred = model.predict(scaled_features)[0]

                # Get probabilities
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(scaled_features)[0]
                else:
                    # Handle models without predict_proba
                    n_classes = len(self.ensemble.label_encoder.classes_)
                    proba = np.zeros(n_classes)
                    proba[pred] = 1.0

                # Convert to signal and confidence
                signal_map = {0: 'BUY', 1: 'SELL', 2: 'WAIT'}
                signal = signal_map.get(pred, 'WAIT')
                confidence = np.max(proba)

                individual_predictions[model_type] = {
                    'signal': signal,
                    'confidence': confidence,
                    'probabilities': proba.tolist()
                }
                all_probabilities.append(proba)
                model_confidences.append(confidence)

            # Get ensemble prediction
            ensemble_pred = self.ensemble.predict(scaled_features)[0]
            ensemble_proba = self.ensemble.predict_proba(scaled_features)[0]
            ensemble_signal = {0: 'BUY', 1: 'SELL', 2: 'WAIT'}.get(ensemble_pred, 'WAIT')
            ensemble_confidence = np.max(ensemble_proba)

            # Get feature importance
            feature_importance = self._get_feature_importance()

            return {
                'symbol': symbol,
                'signal': ensemble_signal,
                'confidence': ensemble_confidence,
                'individual_predictions': individual_predictions,
                'model_weights': getattr(self.ensemble, 'weights', {}) or getattr(self.ensemble, 'model_weights', {}),
                'feature_importance': feature_importance,
                'current_price': current_price,
                'features_used': len(self.feature_columns),
                'model_confidence': {
                    'min': min(model_confidences),
                    'max': max(model_confidences),
                    'mean': np.mean(model_confidences),
                    'std': np.std(model_confidences)
                },
                'success': True,
                'version': 'v2'
            }

        except Exception as e:
            logger.error(f"Error predicting signal for {symbol}: {e}")
            return {
                'symbol': symbol,
                'signal': 'WAIT',
                'confidence': 0.0,
                'error': str(e),
                'success': False,
                'version': 'v2'
            }

    def train_models(self, training_data: pd.DataFrame, labels: pd.Series,
                    validation_split: float = 0.2) -> Dict:
        """
        Train enhanced models with advanced features

        Args:
            training_data: Historical OHLCV data
            labels: Signal labels (0=BUY, 1=SELL, 2=WAIT)
            validation_split: Fraction for validation

        Returns:
            Dict with training metrics
        """
        logger.info("Starting enhanced model training...")

        # Create enhanced features
        logger.info("Creating enhanced features...")
        features_df = self.feature_engineer.create_features(training_data)

        # Clean data
        features_df = features_df.fillna(0)
        features_df = features_df.replace([np.inf, -np.inf], 0)

        # Remove any remaining NaN
        features_df = features_df.dropna()
        labels = labels.loc[features_df.index]

        # Store feature columns
        self.feature_columns = features_df.columns.tolist()
        logger.info(f"Features created: {len(self.feature_columns)}")

        # Split data
        split_idx = int(len(features_df) * (1 - validation_split))
        X_train = features_df.iloc[:split_idx]
        y_train = labels.iloc[:split_idx]
        X_val = features_df.iloc[split_idx:]
        y_val = labels.iloc[split_idx:]

        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Train ensemble
        logger.info("Training ensemble models...")
        self.ensemble.fit(X_train_scaled, y_train, X_val_scaled, y_val)

        # Calculate training metrics
        train_pred = self.ensemble.predict(X_train_scaled)
        val_pred = self.ensemble.predict(X_val_scaled)

        train_accuracy = np.mean(train_pred == y_train.values)
        val_accuracy = np.mean(val_pred == y_val.values)

        # Mark as trained
        self.trained = True

        # Store model info
        self.model_info = {
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'feature_count': len(self.feature_columns),
            'train_accuracy': train_accuracy,
            'validation_accuracy': val_accuracy,
            'model_weights': getattr(self.ensemble, 'weights', {}) or getattr(self.ensemble, 'model_weights', {}),
            'feature_groups': {}
        }

        # Save models
        self.save_models()

        logger.info(f"Training complete!")
        logger.info(f"Training Accuracy: {train_accuracy:.3f}")
        logger.info(f"Validation Accuracy: {val_accuracy:.3f}")
        logger.info(f"Features: {len(self.feature_columns)}")
        logger.info(f"Model Weights: {getattr(self.ensemble, 'weights', {}) or getattr(self.ensemble, 'model_weights', {})}")

        return self.model_info

    def save_models(self):
        """Save trained models and components"""
        model_data = {
            'ensemble': self.ensemble,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'model_info': self.model_info,
            'trained': self.trained,
            'version': 'v2'
        }

        model_path = 'ml_system/models/enhanced_v2_models.pkl'
        joblib.dump(model_data, model_path)
        logger.info(f"Enhanced models saved to {model_path}")

    def load_models(self):
        """Load trained models"""
        model_path = 'ml_system/models/enhanced_v2_models.pkl'

        try:
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                self.ensemble = model_data['ensemble']
                self.scaler = model_data['scaler']
                self.label_encoder = model_data['label_encoder']
                self.feature_columns = model_data['feature_columns']
                self.model_info = model_data.get('model_info', {})
                self.trained = model_data['trained']

                logger.info(f"Enhanced models loaded from {model_path}")
                logger.info(f"Features: {len(self.feature_columns)}")
                logger.info(f"Last validation accuracy: {self.model_info.get('validation_accuracy', 'N/A')}")
                return True
        except Exception as e:
            logger.warning(f"Error loading enhanced models: {e}")

        logger.info("Enhanced models not found - need to train first")
        return False

    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from ensemble models"""
        importance_dict = {}

        try:
            # Get importance from Random Forest model
            rf_model = self.ensemble.models.get('rf')
            if rf_model and hasattr(rf_model, 'feature_importances_'):
                for i, feature in enumerate(self.feature_columns):
                    importance_dict[feature] = rf_model.feature_importances_[i]

            # Get importance from XGBoost if available
            xgb_model = self.ensemble.models.get('xgb')
            if xgb_model and hasattr(xgb_model, 'feature_importances_'):
                for i, feature in enumerate(self.feature_columns):
                    # Average with RF importance
                    if feature in importance_dict:
                        importance_dict[feature] = (importance_dict[feature] +
                                                   xgb_model.feature_importances_[i]) / 2
                    else:
                        importance_dict[feature] = xgb_model.feature_importances_[i]

            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(),
                                        key=lambda x: x[1], reverse=True))

            # Return top 10
            return dict(list(importance_dict.items())[:10])

        except Exception as e:
            logger.warning(f"Error getting feature importance: {e}")
            return {}

    def get_model_info(self) -> Dict:
        """Get information about trained models"""
        return {
            'trained': self.trained,
            'version': 'v2',
            'model_types': self.ensemble.model_types,
            'feature_count': len(self.feature_columns),
            'feature_groups': {},
            'model_weights': getattr(self.ensemble, 'weights', {}) or getattr(self.ensemble, 'model_weights', {}),
            'training_info': self.model_info
        }

    def is_trained(self) -> bool:
        """Check if models are trained"""
        return self.trained

    def get_feature_summary(self) -> Dict:
        """Get summary of features being used"""
        return {
            'total_features': len(self.feature_columns),
            'feature_groups': {},
            'top_features': list(self._get_feature_importance().keys())[:5]
        }