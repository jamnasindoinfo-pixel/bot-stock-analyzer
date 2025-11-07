#!/usr/bin/env python3
"""
Run Training Module - Entry point for training ML models

This module provides command-line interface for training ML models
using the enhanced training pipeline.
"""

import sys
import argparse
import logging
from ml_system.training.train_enhanced_models import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for training"""
    parser = argparse.ArgumentParser(description='Train Enhanced ML Models')
    parser.add_argument('--demo', action='store_true',
                      help='Run demo training with 15 popular Indonesian stocks')
    parser.add_argument('--full', action='store_true',
                      help='Train with all Indonesian stocks')
    parser.add_argument('--symbols', nargs='+',
                      help='Train with specific stock symbols')
    parser.add_argument('--years', type=float, default=2.0,
                      help='Years of historical data to use (default: 2.0)')

    args = parser.parse_args()

    try:
        # Initialize trainer
        trainer = ModelTrainer()

        logger.info("Starting Enhanced Model Training Pipeline")
        logger.info("=" * 60)

        # Determine which stocks to train
        if args.symbols:
            stocks_to_train = args.symbols
            logger.info(f"Training custom stocks: {stocks_to_train}")
        elif args.demo:
            # Use first 15 stocks for demo
            stocks_to_train = trainer.indonesian_stocks[:15]
            logger.info(f"Demo training with 15 popular stocks")
        elif args.full:
            # Train with all Indonesian stocks
            stocks_to_train = trainer.indonesian_stocks
            logger.info(f"Full training with {len(stocks_to_train)} Indonesian stocks")
        else:
            # Default to demo
            stocks_to_train = trainer.indonesian_stocks[:15]
            logger.info(f"Default: Demo training with 15 stocks")

        # Convert years to period string
        if args.years >= 1:
            period = f"{int(args.years)}y"
        else:
            period = f"{int(args.years * 12)}mo"

        logger.info(f"Using {args.years} years of data (period: {period})")

        # Train models
        results = trainer.train_multiple_stocks(stocks_to_train, period=period)

        # Print summary
        summary = trainer.get_training_summary(results)

        logger.info("\nTraining Summary:")
        logger.info(f"Stocks attempted: {len(results)}")

        if 'error' in summary and summary.get('success_rate', 0) == 0:
            # All trainings failed
            logger.error(f"All {len(results)} trainings failed")
            logger.error("Possible reasons:")
            logger.error("- No internet connection")
            logger.error("- Invalid stock symbols")
            logger.error("- yfinance API issues")
            logger.info("=" * 60)
            logger.error("Training pipeline completed with failures!")
            return 1
        else:
            # Some or all trainings succeeded
            successful_count = summary.get('successful_trainings', 0)
            logger.info(f"Stocks successful: {successful_count}")
            logger.info(f"Success rate: {summary['success_rate']:.1%}")

            if successful_count > 0:
                logger.info(f"Average validation accuracy: {summary['validation_accuracy']['mean']:.3f}")
                logger.info(f"Best validation accuracy: {summary['validation_accuracy']['max']:.3f}")

                if summary['success_rate'] > 0:
                    logger.info("\nTop performing stocks:")
                    for i, stock in enumerate(summary['best_performing_stocks'][:5], 1):
                        logger.info(f"{i}. {stock['symbol']}: {stock['validation_accuracy']:.3f}")

            logger.info("=" * 60)
            logger.info("Training pipeline completed!")
            return 0 if summary['success_rate'] > 0 else 1

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())