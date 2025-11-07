"""
Enhanced Model Training Pipeline

Training pipeline for MLPredictorV2 with:
- Data loading for Indonesian stocks using yfinance
- Label creation based on future returns (>2% = BUY, < -2% = SELL, else WAIT)
- Training with enhanced features and ensemble models
- Comprehensive metrics calculation and validation
- Model saving and loading functionality
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import os
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import warnings

# Import ML components
from ml_system.core.ml_predictor_v2 import MLPredictorV2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Enhanced Model Trainer for MLPredictorV2

    Features:
    - Load training data for multiple Indonesian stocks
    - Create labels based on future returns (>2% = BUY, < -2% = SELL, else WAIT)
    - Train MLPredictorV2 with enhanced features (37 features)
    - Calculate comprehensive training and validation metrics
    - Save trained models and training results
    - Validate models with test data
    """

    def __init__(self):
        """Initialize the enhanced model trainer"""
        self.predictor = MLPredictorV2()

        # Indonesian stock symbols for training
        self.indonesian_stocks = [
            # Banking stocks
            'BBCA.JK',  # Bank Central Asia
            'BMRI.JK',  # Bank Mandiri
            'BBNI.JK',  # Bank BNI
            'BBRI.JK',  # Bank BRI

            # Telecommunication stocks
            'TLKM.JK',  # Telkom Indonesia
            'ISAT.JK',  # Indosat
            'EXCL.JK',  # XL Axiata

            # Consumer stocks
            'UNVR.JK',  # Unilever Indonesia
            'INDF.JK',  # Indofood
            'ICBP.JK',  # Indofood CBP

            # Technology stocks
            'GOTO.JK',  # GoTo Gojek Tokopedia
            'BUKA.JK',  # Bukalapak

            # Infrastructure and utilities
            'PGAS.JK',  # Perusahaan Gas Negara
            'JSMR.JK',  # Jasa Marga
            'ANTM.JK',  # Aneka Tambang

            # Automotive stocks
            'AUTO.JK',  # Astra International
            'ASII.JK',  # Astra International

            # Property stocks
            'BSDE.JK',  # Bumi Serpong Damai
            'LPKR.JK',  # Lippo Karawaci
        ]

        # Training parameters
        self.default_params = {
            'future_days': 5,          # Days to look ahead for returns
            'buy_threshold': 0.02,     # 2% return for BUY signal
            'sell_threshold': -0.02,   # -2% return for SELL signal
            'validation_split': 0.2,   # 20% for validation
            'min_data_points': 252,    # Minimum 1 year of trading days
        }

        logger.info(f"ModelTrainer initialized with {len(self.indonesian_stocks)} Indonesian stocks")

    def load_stock_data(self, symbol: str, period: str = '2y') -> pd.DataFrame:
        """
        Load historical stock data using yfinance

        Args:
            symbol: Stock symbol (e.g., 'BBCA.JK')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')

        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Loading data for {symbol} (period: {period})...")

            # Download data
            data = yf.download(symbol, period=period, progress=False, auto_adjust=False)

            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()

            # Check minimum data requirements
            if len(data) < self.default_params['min_data_points']:
                logger.warning(f"Insufficient data for {symbol}: {len(data)} days (need {self.default_params['min_data_points']})")
                return data  # Return available data, let caller decide

            # Clean and validate data
            data = self._clean_data(data)

            logger.info(f"Loaded {len(data)} days of data for {symbol}")
            logger.info(f"Date range: {data.index.min().date()} to {data.index.max().date()}")

            return data

        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate stock data

        Args:
            data: Raw OHLCV data

        Returns:
            Cleaned DataFrame
        """
        # Handle multi-level columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten multi-level columns by taking the first level
            data.columns = [col[0] for col in data.columns.values]

        # Reset index if it's a MultiIndex to ensure clean DatetimeIndex
        if isinstance(data.index, pd.MultiIndex):
            data = data.reset_index(level=0, drop=True)

        # Ensure index is a proper DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except:
                pass  # Keep as-is if conversion fails

        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in data.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. Available: {data.columns.tolist()}")

        # Remove rows with missing critical data
        data = data.dropna(subset=required_cols)

        # Remove zero or negative prices
        for col in required_cols:
            data = data[data[col] > 0]

        # Sort by date
        data = data.sort_index()

        # Remove duplicates
        data = data[~data.index.duplicated(keep='first')]

        # Fill missing volume with median
        if 'Volume' in data.columns:
            median_volume = data['Volume'].median()
            data['Volume'] = data['Volume'].fillna(median_volume)

        return data

    def create_trading_labels(self, data: pd.DataFrame,
                            future_days: int = None,
                            buy_threshold: float = None,
                            sell_threshold: float = None) -> pd.Series:
        """
        Create trading labels based on future returns

        Args:
            data: DataFrame with Close prices
            future_days: Days to look ahead for return calculation
            buy_threshold: Return threshold for BUY signal (decimal)
            sell_threshold: Return threshold for SELL signal (decimal)

        Returns:
            Series with labels (0=BUY, 1=SELL, 2=WAIT)
        """
        if future_days is None:
            future_days = self.default_params['future_days']
        if buy_threshold is None:
            buy_threshold = self.default_params['buy_threshold']
        if sell_threshold is None:
            sell_threshold = self.default_params['sell_threshold']

        if 'Close' not in data.columns:
            # Return empty series if Close column is missing
            return pd.Series(dtype=int)

        if len(data) < future_days + 1:
            logger.warning(f"Insufficient data for label creation: {len(data)} days, need {future_days + 1}")
            # Return all WAIT labels if insufficient data
            return pd.Series([2] * len(data), index=data.index)

        # Calculate future returns
        future_returns = data['Close'].shift(-future_days) / data['Close'] - 1

        # Create labels using explicit initialization
        labels = pd.Series(2, index=data.index, dtype=int)  # Default to WAIT

        # Create boolean masks for each signal type
        buy_mask = future_returns > buy_threshold
        sell_mask = future_returns < sell_threshold

        # Use .loc for safe assignment
        labels.loc[buy_mask] = 0  # BUY
        labels.loc[sell_mask] = 1  # SELL
        # WAIT signal is already set as default

        # Handle NaN values (at the end where we can't calculate future returns)
        labels = labels.fillna(2)  # Default to WAIT

        logger.info(f"Created labels for {len(labels)} data points")
        label_counts = labels.value_counts().sort_index()
        logger.info(f"Label distribution: BUY={int(label_counts.get(0, 0))}, "
                   f"SELL={int(label_counts.get(1, 0))}, WAIT={int(label_counts.get(2, 0))}")

        return labels

    def train_single_stock(self, symbol: str,
                         period: str = '2y',
                         custom_params: Dict = None) -> Dict:
        """
        Train model on a single stock

        Args:
            symbol: Stock symbol to train on (e.g., 'BBCA' or 'BBCA.JK')
            period: Historical data period
            custom_params: Custom training parameters

        Returns:
            Dictionary with training results
        """
        # Ensure symbol has .JK suffix for Indonesian stocks
        if not symbol.endswith('.JK'):
            symbol = f"{symbol}.JK"

        logger.info(f"Starting training for {symbol}...")

        # Merge custom parameters with defaults
        params = self.default_params.copy()
        if custom_params:
            params.update(custom_params)

        # Load data
        data = self.load_stock_data(symbol, period)
        if data.empty:
            return {
                'symbol': symbol,
                'error': 'Failed to load training data',
                'training_samples': 0,
                'success': False
            }

        # Create labels
        labels = self.create_trading_labels(
            data,
            future_days=params['future_days'],
            buy_threshold=params['buy_threshold'],
            sell_threshold=params['sell_threshold']
        )

        # Check if we have sufficient labeled data
        if len(data) < 50:  # Minimum 50 samples for meaningful training
            return {
                'symbol': symbol,
                'error': f'Insufficient training data: {len(data)} samples',
                'training_samples': len(data),
                'success': False
            }

        try:
            # Train the enhanced model
            training_results = self.predictor.train_models(
                data,
                labels,
                validation_split=params['validation_split']
            )

            # Add additional metadata
            training_results.update({
                'symbol': symbol,
                'data_period': period,
                'data_range': {
                    'start': str(data.index.min()),
                    'end': str(data.index.max())
                },
                'training_samples': len(data),
                'label_distribution': {int(k): int(v) for k, v in labels.value_counts().to_dict().items()},
                'training_params': params,
                'training_date': datetime.now().isoformat(),
                'success': True
            })

            logger.info(f"Training completed for {symbol}")
            logger.info(f"Validation accuracy: {training_results['validation_accuracy']:.3f}")
            logger.info(f"Features used: {training_results['feature_count']}")

            return training_results

        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'training_samples': len(data),
                'success': False
            }

    def train_multiple_stocks(self, symbols: List[str] = None,
                            period: str = '2y',
                            save_results: bool = True) -> List[Dict]:
        """
        Train models on multiple stocks

        Args:
            symbols: List of stock symbols (default: Indonesian stocks)
            period: Historical data period
            save_results: Whether to save training results

        Returns:
            List of training results for each stock
        """
        if symbols is None:
            symbols = self.indonesian_stocks

        logger.info(f"Starting training on {len(symbols)} stocks...")

        all_results = []
        successful_trainings = 0

        for i, symbol in enumerate(symbols, 1):
            logger.info(f"Training {i}/{len(symbols)}: {symbol}")

            result = self.train_single_stock(symbol, period)
            all_results.append(result)

            if result.get('success', False):
                successful_trainings += 1

            # Progress logging
            progress = (i / len(symbols)) * 100
            logger.info(f"Progress: {progress:.1f}% ({successful_trainings} successful)")

        # Summary statistics
        summary = {
            'total_stocks': len(symbols),
            'successful_trainings': successful_trainings,
            'success_rate': successful_trainings / len(symbols) if symbols else 0,
            'training_date': datetime.now().isoformat()
        }

        # Calculate average metrics for successful trainings
        successful_results = [r for r in all_results if r.get('success', False)]
        if successful_results:
            val_accuracies = [r['validation_accuracy'] for r in successful_results]
            summary['average_validation_accuracy'] = np.mean(val_accuracies)
            summary['max_validation_accuracy'] = np.max(val_accuracies)
            summary['min_validation_accuracy'] = np.min(val_accuracies)

        logger.info(f"Training completed: {successful_trainings}/{len(symbols)} successful")
        if successful_results:
            logger.info(f"Average validation accuracy: {summary['average_validation_accuracy']:.3f}")

        # Save results if requested
        if save_results:
            self.save_training_results(all_results, 'ml_system/models/training_results.json')
            self.save_training_results(summary, 'ml_system/models/training_summary.json')

        return all_results

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate comprehensive classification metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary with various metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)

        # Precision, recall, F1 (handle cases where classes might be missing)
        try:
            precision = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

            # Convert to dictionaries with class names
            class_names = ['BUY', 'SELL', 'WAIT']
            precision_dict = {class_names[i]: precision[i] if i < len(precision) else 0
                            for i in range(3)}
            recall_dict = {class_names[i]: recall[i] if i < len(recall) else 0
                         for i in range(3)}
            f1_dict = {class_names[i]: f1[i] if i < len(f1) else 0
                      for i in range(3)}

            # Weighted averages
            precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        except Exception as e:
            logger.warning(f"Error calculating detailed metrics: {e}")
            precision_dict = recall_dict = f1_dict = {}
            precision_weighted = recall_weighted = f1_weighted = 0

        # Confusion matrix
        try:
            cm = confusion_matrix(y_true, y_pred)
            cm_dict = {
                'matrix': cm.tolist(),
                'shape': cm.shape
            }
        except Exception as e:
            logger.warning(f"Error calculating confusion matrix: {e}")
            cm_dict = {'matrix': [], 'shape': (0, 0)}

        # Classification report
        try:
            # Get unique labels present
            unique_labels = np.unique(np.concatenate([y_true, y_pred]))
            if len(unique_labels) == 3:
                target_names = ['BUY', 'SELL', 'WAIT']
            else:
                # Use only the labels that are present
                label_names = ['BUY', 'SELL', 'WAIT']
                target_names = [label_names[i] for i in unique_labels]

            class_report = classification_report(y_true, y_pred,
                                                labels=unique_labels,
                                                target_names=target_names,
                                                zero_division=0,
                                                output_dict=True)
        except Exception as e:
            logger.warning(f"Error generating classification report: {e}")
            class_report = {}

        # Additional metrics
        label_counts = np.bincount(y_true, minlength=3)
        pred_counts = np.bincount(y_pred, minlength=3)

        return {
            'accuracy': float(accuracy),
            'precision': precision_dict,
            'recall': recall_dict,
            'f1_score': f1_dict,
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'confusion_matrix': cm_dict,
            'classification_report': class_report,
            'true_label_distribution': {
                'BUY': int(label_counts[0]),
                'SELL': int(label_counts[1]),
                'WAIT': int(label_counts[2])
            },
            'predicted_label_distribution': {
                'BUY': int(pred_counts[0]),
                'SELL': int(pred_counts[1]),
                'WAIT': int(pred_counts[2])
            },
            'total_samples': int(len(y_true))
        }

    def validate_trained_model(self, symbol: str, test_data: pd.DataFrame) -> Dict:
        """
        Validate trained model with test data

        Args:
            symbol: Stock symbol for validation
            test_data: Test data for validation

        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating model for {symbol}...")

        if not self.predictor.is_trained():
            return {
                'symbol': symbol,
                'error': 'Model not trained yet',
                'validation_samples': 0,
                'success': False
            }

        if test_data.empty:
            return {
                'symbol': symbol,
                'error': 'Empty test data',
                'validation_samples': 0,
                'success': False
            }

        try:
            # Create labels for test data
            test_labels = self.create_trading_labels(test_data)

            # Make predictions
            predictions = []
            confidences = []

            for i in range(len(test_data)):
                # Get data up to current point
                current_data = test_data.iloc[:i+1]

                if len(current_data) < 10:  # Need minimum data for prediction
                    predictions.append(2)  # WAIT
                    confidences.append(0.0)
                    continue

                # Make prediction
                pred_result = self.predictor.predict_signal(current_data, symbol)

                if pred_result.get('success', False):
                    signal_map = {'BUY': 0, 'SELL': 1, 'WAIT': 2}
                    predictions.append(signal_map.get(pred_result['signal'], 2))
                    confidences.append(pred_result['confidence'])
                else:
                    predictions.append(2)  # WAIT on error
                    confidences.append(0.0)

            # Convert to numpy arrays and ensure integer type
            y_true = test_labels.values.astype(int)
            y_pred = np.array(predictions, dtype=int)

            # Calculate metrics
            metrics = self.calculate_metrics(y_true, y_pred)

            # Additional validation statistics
            avg_confidence = np.mean(confidences) if confidences else 0.0

            validation_results = {
                'symbol': symbol,
                'validation_samples': len(test_data),
                'validation_accuracy': metrics['accuracy'],
                'validation_metrics': metrics,
                'avg_confidence': avg_confidence,
                'confidence_distribution': {
                    'min': float(np.min(confidences)) if confidences else 0.0,
                    'max': float(np.max(confidences)) if confidences else 0.0,
                    'mean': float(np.mean(confidences)) if confidences else 0.0,
                    'std': float(np.std(confidences)) if confidences else 0.0
                },
                'validation_date': datetime.now().isoformat(),
                'success': True
            }

            logger.info(f"Validation completed for {symbol}")
            logger.info(f"Validation accuracy: {validation_results['validation_accuracy']:.3f}")
            logger.info(f"Average confidence: {avg_confidence:.3f}")

            return validation_results

        except Exception as e:
            logger.error(f"Error validating model for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'validation_samples': len(test_data),
                'success': False
            }

    def save_training_results(self, results, output_path: str):
        """
        Save training results to JSON file

        Args:
            results: Training results (dict or list)
            output_path: Output file path
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save to JSON
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Training results saved to {output_path}")

        except Exception as e:
            logger.error(f"Error saving training results: {e}")

    def load_training_results(self, input_path: str) -> Dict:
        """
        Load training results from JSON file

        Args:
            input_path: Input file path

        Returns:
            Dictionary with training results
        """
        try:
            with open(input_path, 'r') as f:
                results = json.load(f)

            logger.info(f"Training results loaded from {input_path}")
            return results

        except Exception as e:
            logger.error(f"Error loading training results: {e}")
            return {}

    def get_training_summary(self, results: List[Dict]) -> Dict:
        """
        Generate summary statistics from training results

        Args:
            results: List of training results

        Returns:
            Summary statistics
        """
        if not results:
            return {
                'total_trainings': 0,
                'successful_trainings': 0,
                'success_rate': 0.0,
                'error': 'No training results provided'
            }

        successful_results = [r for r in results if r.get('success', False)]

        if not successful_results:
            return {
                'total_trainings': len(results),
                'successful_trainings': 0,
                'success_rate': 0.0,
                'error': 'No successful trainings',
                'failed_trainings': len(results)
            }

        # Extract metrics
        val_accuracies = [r['validation_accuracy'] for r in successful_results]
        train_accuracies = [r.get('train_accuracy', 0) for r in successful_results]
        feature_counts = [r.get('feature_count', 0) for r in successful_results]

        summary = {
            'total_trainings': len(results),
            'successful_trainings': len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'validation_accuracy': {
                'mean': np.mean(val_accuracies),
                'std': np.std(val_accuracies),
                'min': np.min(val_accuracies),
                'max': np.max(val_accuracies),
                'median': np.median(val_accuracies)
            },
            'training_accuracy': {
                'mean': np.mean(train_accuracies),
                'std': np.std(train_accuracies),
                'min': np.min(train_accuracies),
                'max': np.max(train_accuracies),
                'median': np.median(train_accuracies)
            },
            'feature_count': {
                'mean': np.mean(feature_counts),
                'std': np.std(feature_counts),
                'min': np.min(feature_counts),
                'max': np.max(feature_counts)
            },
            'best_performing_stocks': sorted(
                successful_results,
                key=lambda x: x['validation_accuracy'],
                reverse=True
            )[:5]
        }

        return summary


def main():
    """Main function for running training pipeline"""
    trainer = ModelTrainer()

    logger.info("Starting Enhanced Model Training Pipeline")
    logger.info("=" * 60)

    # Train on a subset of Indonesian stocks for demonstration
    sample_stocks = trainer.indonesian_stocks[:5]  # Train on first 5 stocks

    logger.info(f"Training on sample stocks: {sample_stocks}")

    # Train models
    results = trainer.train_multiple_stocks(sample_stocks, period='2y')

    # Print summary
    summary = trainer.get_training_summary(results)

    logger.info("\nTraining Summary:")
    logger.info(f"Success rate: {summary['success_rate']:.1%}")
    logger.info(f"Average validation accuracy: {summary['validation_accuracy']['mean']:.3f}")
    logger.info(f"Best validation accuracy: {summary['validation_accuracy']['max']:.3f}")

    logger.info("\nTop performing stocks:")
    for i, stock in enumerate(summary['best_performing_stocks'], 1):
        logger.info(f"{i}. {stock['symbol']}: {stock['validation_accuracy']:.3f}")

    logger.info("=" * 60)
    logger.info("Training pipeline completed!")


if __name__ == '__main__':
    main()