#!/usr/bin/env python3
"""
Demonstration script for running Indonesian stocks training pipeline

This script shows how to use the comprehensive Indonesian stocks trainer
with a smaller dataset for demonstration purposes.
"""

import sys
import os
from datetime import datetime

# Add paths
sys.path.append('.')
sys.path.append('./.worktrees/ml-accuracy-enhancement')

def run_demo():
    """Run a demonstration of the Indonesian stocks training pipeline"""
    print("=" * 60)
    print("INDONESIAN STOCKS COMPREHENSIVE TRAINING DEMO")
    print("=" * 60)
    print()

    # Configuration for demo (smaller dataset for quick demonstration)
    demo_config = {
        'min_data_years': 1.0,    # Require only 1 year of data for demo
        'max_workers': 3,         # Fewer workers for demo
        'batch_size': 10,         # Smaller batches
        'progress_update_interval': 5  # More frequent updates
    }

    print("Demo Configuration:")
    for key, value in demo_config.items():
        print(f"  {key}: {value}")
    print()

    print("Starting demo training pipeline...")
    print("Note: This will use a subset of Indonesian stocks for demonstration")
    print()

    try:
        # Import trainer
        from ml_system.training.train_all_indonesian_stocks import IndonesianStockTrainer

        # Initialize trainer with demo config
        trainer = IndonesianStockTrainer(**demo_config)

        # Override with a smaller subset for demo
        demo_symbols = [
            # Major banking stocks
            'BBCA.JK', 'BBRI.JK', 'BMRI.JK', 'BBNI.JK',
            # Major consumer stocks
            'UNVR.JK', 'INDF.JK', 'GGRM.JK', 'HMSP.JK',
            # Major telecom stocks
            'TLKM.JK', 'EXCL.JK', 'ISAT.JK',
            # Major automotive stocks
            'ASII.JK', 'AUTO.JK',
            # Major property stocks
            'PWON.JK', 'BSDE.JK'
        ]

        print(f"Using demo subset of {len(demo_symbols)} stocks:")
        print(", ".join(demo_symbols))
        print()

        # Override the symbols list
        trainer.all_symbols = demo_symbols

        # Run the pipeline
        start_time = datetime.now()
        print(f"Starting training at {start_time}")
        print()

        # Run a modified version with demo symbols
        results = run_demo_pipeline(trainer, demo_symbols)

        duration = datetime.now() - start_time

        print()
        print("=" * 60)
        print("DEMO TRAINING COMPLETED")
        print("=" * 60)
        print(f"Total Duration: {duration}")
        print(f"Status: {results.get('status', 'unknown')}")
        print()

        if results.get('status') == 'failed':
            print(f"Error: {results.get('error', 'unknown error')}")
        else:
            print("Training Results:")
            print(f"  Symbols processed: {len(trainer.processed_count) if hasattr(trainer, 'processed_count') else 'N/A'}")
            print(f"  Symbols with sufficient data: {len(trainer.filtered_symbols)}")
            print(f"  Training samples: {results.get('training_samples', 'N/A')}")
            print(f"  Validation accuracy: {results.get('validation_accuracy', 'N/A')}")
            print(f"  Features used: {results.get('feature_count', 'N/A')}")

        print()
        print("Files created:")
        print("  - ml_system/training/results/latest_indonesian_training.json")
        print("  - ml_system/training/results/training_report.txt")
        print("  - ml_system/training/progress.json")

        return results

    except ImportError as e:
        print(f"Error importing trainer: {e}")
        print("Make sure the training script is in the correct location.")
        return None
    except Exception as e:
        print(f"Demo failed with error: {e}")
        return None

def run_demo_pipeline(trainer, symbols):
    """Run a simplified version of the training pipeline for demo"""
    try:
        # Step 1: Filter stocks with sufficient data
        print("Step 1: Checking data availability...")
        sufficient_symbols = trainer.filter_stocks_with_sufficient_data(symbols)

        if not sufficient_symbols:
            return {
                'status': 'failed',
                'error': 'No stocks with sufficient data found'
            }

        print(f"Found {len(sufficient_symbols)} stocks with sufficient data:")
        for symbol in sufficient_symbols:
            data_days = len(trainer.downloaded_data.get(symbol, []))
            print(f"  {symbol}: {data_days} days of data")
        print()

        # Step 2: Create training dataset
        print("Step 2: Creating training dataset...")
        features, labels = trainer.create_training_dataset(sufficient_symbols)

        if features.empty:
            return {
                'status': 'failed',
                'error': 'No valid training data created'
            }

        print(f"Created dataset with {len(features)} samples and {len(features.columns)} features")
        print(f"Label distribution: BUY={labels.value_counts().get(0, 0)}, "
              f"SELL={labels.value_counts().get(1, 0)}, WAIT={labels.value_counts().get(2, 0)}")
        print()

        # Step 3: Train models (if enhanced ML is available)
        if trainer.ml_predictor:
            print("Step 3: Training enhanced ML models...")
            training_results = trainer.train_models(features, labels)
        else:
            print("Step 3: Enhanced ML not available, creating basic model...")
            # Create a basic model for demo
            training_results = train_basic_model(features, labels, trainer)

        # Step 4: Save results
        print("Step 4: Saving results...")
        trainer.save_results(training_results)

        # Step 5: Generate simplified report
        print("Step 5: Generating report...")
        generate_demo_report(trainer, training_results)

        return training_results

    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e)
        }

def train_basic_model(features, labels, trainer):
    """Train a basic model when enhanced ML is not available"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        import joblib

        # Prepare data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )

        # Train basic Random Forest
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)

        # Save model
        model_data = {
            'model': model,
            'feature_columns': features.columns.tolist(),
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
        }

        model_path = 'ml_system/training/models/basic_demo_model.pkl'
        joblib.dump(model_data, model_path)

        print(f"Basic model trained and saved to {model_path}")
        print(f"Train accuracy: {train_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")

        return {
            'status': 'success',
            'model_type': 'BasicRandomForest',
            'train_accuracy': train_accuracy,
            'validation_accuracy': test_accuracy,
            'training_samples': len(X_train),
            'validation_samples': len(X_test),
            'feature_count': len(features.columns),
            'model_path': model_path
        }

    except Exception as e:
        return {
            'status': 'failed',
            'error': f'Basic model training failed: {str(e)}'
        }

def generate_demo_report(trainer, results):
    """Generate a demo training report"""
    report = f"""
INDONESIAN STOCKS TRAINING DEMO REPORT
=====================================

Training Summary:
- Start Time: {trainer.start_time if trainer.start_time else 'N/A'}
- End Time: {datetime.now()}
- Total Symbols Tested: {len(trainer.all_symbols)}
- Symbols with Sufficient Data: {len(trainer.filtered_symbols)}
- Failed Downloads: {len(trainer.failed_symbols)}

Training Results:
- Status: {results.get('status', 'unknown')}
- Model Type: {results.get('model_type', 'N/A')}
- Training Accuracy: {results.get('train_accuracy', 'N/A'):.4f}
- Validation Accuracy: {results.get('validation_accuracy', 'N/A'):.4f}
- Features Used: {results.get('feature_count', 'N/A')}

Symbols Used for Training:
{', '.join(trainer.filtered_symbols)}

Failed Symbols (if any):
{', '.join(trainer.failed_symbols[:10])}

Files Created:
- Training results: ml_system/training/results/latest_indonesian_training.json
- Model file: ml_system/training/models/ (if training was successful)
- Progress file: ml_system/training/progress.json

Next Steps:
1. To train on ALL Indonesian stocks, run the full training script
2. Monitor model performance on test data
3. Consider tuning hyperparameters for better performance
4. Set up periodic retraining with updated data

Generated: {datetime.now()}
    """

    # Save report
    report_file = 'ml_system/training/results/demo_training_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"Demo report saved to {report_file}")
    print(report)

if __name__ == "__main__":
    run_demo()