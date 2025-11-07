#!/usr/bin/env python3
"""
Model Trainer Wrapper for ML System

Provides a simplified interface to the existing MLModelTrainer
for use with the CLI training interface.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add paths to access the ML trainer
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / '.worktrees' / 'ml-enhancement'))

try:
    from ml_models.training.trainer import MLModelTrainer
    TRAINER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"MLModelTrainer not available: {e}")
    TRAINER_AVAILABLE = False


class ModelTrainer:
    """
    Simplified interface for training ML models.
    Wraps the existing MLModelTrainer functionality.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the model trainer.

        Args:
            config_path: Path to training configuration file
        """
        self.config_path = config_path
        self.trainer = None
        self.logger = logging.getLogger(__name__)

        if TRAINER_AVAILABLE:
            try:
                self.trainer = MLModelTrainer(config_path or "ml_system/configs/training_config.yaml")
                self.logger.info("MLModelTrainer initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize MLModelTrainer: {e}")
                self.trainer = None
        else:
            self.logger.warning("MLModelTrainer not available")

    def is_available(self) -> bool:
        """Check if the ML trainer is available."""
        return TRAINER_AVAILABLE and self.trainer is not None

    def train_models(
        self,
        symbols: List[str],
        period: str = '2y',
        test_split: float = 0.2,
        model_types: List[str] = None,
        force_retrain: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train ML models for the given symbols.

        Args:
            symbols: List of stock symbols to train on
            period: Historical data period
            test_split: Test set split ratio
            model_types: Types of models to train ('lstm', 'rf', 'both')
            force_retrain: Force retraining even if models exist
            **kwargs: Additional training parameters

        Returns:
            Dictionary containing training results
        """
        if not self.is_available():
            return {
                'status': 'failed',
                'error': 'MLModelTrainer not available',
                'timestamp': self._get_timestamp()
            }

        try:
            # Update trainer configuration
            self._update_trainer_config(symbols, period, test_split, model_types, **kwargs)

            # Run training
            self.logger.info(f"Starting training for {len(symbols)} symbols")
            results = self.trainer.run_full_training_pipeline()

            return results

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': self._get_timestamp()
            }

    def _update_trainer_config(
        self,
        symbols: List[str],
        period: str,
        test_split: float,
        model_types: List[str],
        **kwargs
    ) -> None:
        """Update trainer configuration with provided parameters."""
        if not self.trainer:
            return

        # Update data configuration
        self.trainer.config['data']['symbols'] = symbols
        self.trainer.config['data']['years_of_data'] = self._period_to_years(period)
        self.trainer.config['data']['train_ratio'] = 1 - test_split

        # Update model types
        if model_types:
            # Configure which models to train
            train_lstm = 'lstm' in model_types or 'both' in model_types
            train_rf = 'rf' in model_types or 'both' in model_types

            # Note: This would require extending the original trainer
            # For now, we'll train both as that's the default behavior
            pass

        # Update additional parameters
        for key, value in kwargs.items():
            if key in self.trainer.config:
                self.trainer.config[key] = value

    def _period_to_years(self, period: str) -> float:
        """Convert period string to years."""
        if period.endswith('y'):
            return float(period[:-1])
        elif period.endswith('mo'):
            return float(period[:-2]) / 12
        else:
            return 2.0  # Default to 2 years

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models."""
        if not self.is_available():
            return {
                'available': False,
                'error': 'MLModelTrainer not available'
            }

        # Check for existing model files
        model_files = {
            'lstm_model': self.trainer.lstm_predictor.model_path if self.trainer.lstm_predictor else None,
            'lstm_scaler': self.trainer.lstm_predictor.scaler_path if self.trainer.lstm_predictor else None,
            'rf_model': self.trainer.rf_classifier.model_path if self.trainer.rf_classifier else None,
            'rf_scaler': self.trainer.rf_classifier.scaler_path if self.trainer.rf_classifier else None,
        }

        model_status = {}
        for name, path in model_files.items():
            if path and os.path.exists(path):
                model_status[name] = {
                    'exists': True,
                    'path': path,
                    'modified': os.path.getmtime(path)
                }
            else:
                model_status[name] = {
                    'exists': False,
                    'path': path,
                    'modified': None
                }

        return {
            'available': True,
            'model_files': model_status,
            'config': self.trainer.config if self.trainer else None
        }

    def should_retrain(self, model_type: str) -> bool:
        """
        Check if a model should be retrained.

        Args:
            model_type: Type of model ('lstm' or 'rf')

        Returns:
            True if model should be retrained
        """
        if not self.is_available():
            return True

        return self.trainer.should_retrain(model_type)

    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get history of recent training runs."""
        # This would require extending the original trainer to maintain history
        # For now, return empty list
        return []