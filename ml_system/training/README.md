# Indonesian Stocks Comprehensive Training Pipeline

This directory contains the comprehensive training pipeline for all Indonesian stocks from yfinance, designed to train MLPredictorV2 on a massive dataset of Indonesian market data.

## Overview

The training pipeline:

1. **Scrapes ALL Indonesian stocks** from yfinance (JK market) - over 200+ symbols
2. **Filters stocks with sufficient data** (minimum 1.5-2 years by default)
3. **Downloads historical data** with robust error handling
4. **Creates comprehensive dataset** with advanced feature engineering
5. **Trains MLPredictorV2** ensemble models (RF + XGBoost + LSTM)
6. **Saves models and generates reports** with detailed performance metrics

## Files

### Main Scripts

- **`train_all_indonesian_stocks.py`** - Complete training pipeline for all Indonesian stocks
- **`run_training_demo.py`** - Demonstration script with smaller dataset
- **`test_indonesian_trainer.py`** - Core functionality test script

### Generated Files

- **`ml_system/training/results/`** - Training results and reports
- **`ml_system/training/models/`** - Trained model files
- **`ml_system/training/data/`** - Training datasets
- **`ml_system/training/logs/`** - Training logs

## Quick Start

### 1. Run Demo (Recommended First)

Test the pipeline with a small subset of stocks:

```bash
cd "C:\Users\jamna\OneDrive\Desktop\File Coding JNI\Bot-Stock-Market"
python ml_system/training/run_training_demo.py
```

This uses 15 major Indonesian stocks and completes in ~2-3 minutes.

### 2. Run Full Training Pipeline

Train on ALL available Indonesian stocks:

```bash
python ml_system/training/train_all_indonesian_stocks.py
```

**Expected Duration:** 30-60 minutes depending on network speed
**Expected Output:** Trained models from 100+ stocks with 50,000+ training samples

### 3. Test Core Functionality

Verify the system works:

```bash
python ml_system/training/test_indonesian_trainer.py
```

## Configuration

### Main Training Parameters

```python
config = {
    'min_data_years': 1.5,        # Minimum years of historical data required
    'max_workers': 10,            # Concurrent download workers
    'batch_size': 50,             # Stocks processed per batch
    'progress_update_interval': 20 # Progress update frequency
}
```

### Indonesian Stock Coverage

The pipeline includes stocks from all major sectors:

- **Banking**: BBCA.JK, BBRI.JK, BMRI.JK, BBNI.JK, BTPN.JK, BRIS.JK
- **Telecom**: TLKM.JK, ISAT.JK, EXCL.JK, FREN.JK
- **Consumer**: UNVR.JK, INDF.JK, ICBP.JK, GGRM.JK, HMSP.JK
- **Mining**: ADRO.JK, ANTM.JK, PTBA.JK, TINS.JK, ITMG.JK
- **Property**: BSDE.JK, LPKR.JK, PWON.JK, ASRI.JK
- **Automotive**: ASII.JK, AUTO.JK, GJTL.JK
- **Technology**: GOTO.JK, BUKA.JK, EXCL.JK
- **Infrastructure**: WIKA.JK, ADHI.JK, PTPP.JK, JSMR.JK
- **Agriculture**: AALI.JK, LSIP.JK, SGRO.JK
- **Healthcare**: KLBF.JK, KAEF.JK, HEAL.JK

### Data Requirements

- **Minimum History**: 1.5 years (adjustable via `min_data_years`)
- **Data Quality**: <10% missing values required
- **Validation**: Basic OHLCV data integrity checks

## Features

### Robust Data Collection

- **Error Handling**: Graceful handling of network timeouts and API errors
- **Rate Limiting**: Built-in delays to avoid overwhelming yfinance API
- **Data Validation**: Quality checks and filtering for reliable data
- **Progress Tracking**: Real-time progress updates with detailed statistics

### Advanced Feature Engineering

When MLPredictorV2 is available, the system creates 37 enhanced features:

- **Price Features**: Returns, momentum, volatility, price ratios
- **Volume Features**: Volume moving averages, volume price trends
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ADX
- **Statistical Features**: Z-scores, rolling statistics
- **Market Structure**: Support/resistance levels, trend indicators

### Ensemble Model Training

- **Multiple Models**: Random Forest + XGBoost + LSTM ensemble
- **Cross-Validation**: Proper train/validation splits
- **Hyperparameter Tuning**: Automated optimization (optional)
- **Model Persistence**: Automatic saving of trained models

### Memory Efficiency

- **Batch Processing**: Large datasets processed in manageable chunks
- **Memory Monitoring**: Tracks memory usage during processing
- **Garbage Collection**: Automatic cleanup of intermediate data

## Output Files

### Training Results

```
ml_system/training/results/
├── latest_indonesian_training.json          # Latest training results
├── indonesian_stocks_training_YYYYMMDD_HHMMSS.json  # Timestamped results
├── training_report.txt                      # Human-readable report
└── demo_training_report.txt                 # Demo run report
```

### Model Files

```
ml_system/training/models/
├── enhanced_v2_models.pkl                   # MLPredictorV2 models (if available)
└── basic_demo_model.pkl                     # Basic model fallback
```

### Training Data

```
ml_system/training/data/
└── training_data_YYYYMMDD_HHMMSS.pkl       # Combined training dataset
```

## Monitoring and Logs

### Progress Tracking

The system provides detailed progress updates:

```
Processing batch 1/5...
Processed 50/250 symbols (Success: 45)
BBCA.JK: ✓ 717 days of data
BBRI.JK: ✓ 717 days of data
...
```

### Error Handling

Failed downloads are logged and tracked:

```
Failed symbols: ['SYMBOL1.JK', 'SYMBOL2.JK']
Errors tracked in: ml_system/training/progress.json
```

### Performance Metrics

Training results include comprehensive metrics:

```json
{
  "training_samples": 50000,
  "validation_accuracy": 0.6845,
  "feature_count": 37,
  "ensemble_models": ["rf", "xgb", "lstm"],
  "symbols_used": 156
}
```

## Troubleshooting

### Common Issues

1. **"Enhanced MLPredictorV2 not available"**
   - The system falls back to basic Random Forest training
   - Models still work but with fewer features

2. **Network Timeouts**
   - System automatically retries failed downloads
   - Increase `timeout` in yfinance calls if needed

3. **Memory Issues**
   - Reduce `max_workers` and `batch_size`
   - The system processes data in batches to manage memory

4. **Insufficient Data**
   - Check if yfinance is accessible from your network
   - Some stocks may have limited historical data

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Next Steps

### After Training

1. **Model Evaluation**: Test on held-out data
2. **Performance Monitoring**: Track live trading performance
3. **Periodic Retraining**: Schedule regular updates with new data
4. **Feature Expansion**: Add more technical or fundamental features

### Integration

Trained models can be used with the main bot:

```python
from ml_system.core.ml_predictor import MLPredictor
predictor = MLPredictor()
signal = predictor.predict_signal(data, 'BBCA.JK')
```

## Requirements

- Python 3.7+
- yfinance
- pandas, numpy
- scikit-learn
- joblib
- Standard ML libraries (tensorflow, xgboost for enhanced features)

## Performance Expectations

- **Demo Run**: 2-3 minutes, 15 stocks, ~10K samples
- **Full Run**: 30-60 minutes, 200+ stocks, 50K+ samples
- **Expected Accuracy**: 65-70% with enhanced features
- **Memory Usage**: 1-2GB during full training

## Support

For issues or questions:

1. Check the troubleshooting section
2. Review the logs in `ml_system/training/logs/`
3. Run the test script to verify basic functionality
4. Check network connectivity to yfinance API