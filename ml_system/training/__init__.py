"""
ML System Training Module

Enhanced model training pipeline for MLPredictorV2 with support for:
- Indonesian stock data loading using yfinance
- Trading label creation based on future returns
- Enhanced feature engineering with 37 features
- Ensemble model training (RF + XGBoost + LSTM)
- Comprehensive metrics calculation and validation
"""

from .train_enhanced_models import ModelTrainer

__all__ = ['ModelTrainer']

__version__ = '2.0.0'
__author__ = 'Enhanced ML System'
__description__ = 'Enhanced training pipeline for stock market prediction models'