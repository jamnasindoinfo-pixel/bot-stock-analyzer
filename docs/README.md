# 🤖 Bot Stock Market Indonesia

Sistem analisis saham Indonesia yang menggabungkan **Technical Analysis + Machine Learning + News Sentiment** untuk memberikan rekomendasi trading jangka pendek.

## 📁 Struktur Proyek

```
Bot-Stock-Market/
├── main.py                     # Aplikasi utama
├── .env                        # Environment variables
├── requirements.txt            # Dependencies
│
├── analysis/                   # Script-script analisis
│   ├── auto_analysis.py       # Analisis otomatis technical
│   ├── expanded_analysis.py   # Analisis 30+ saham
│   ├── technical_analysis_v2.py # Technical analysis v2
│   └── comprehensive_analysis.py # Gabungan technical+ML+news
│
├── analyzers/                  # Modul analisis
│   └── market_analyzer.py
│
├── ml_system/                  # Machine Learning System
│   ├── core/
│   │   └── ml_predictor_v2.py
│   ├── cli/
│   │   └── EnhancedProductionMLCLI.py
│   ├── training/
│   │   ├── improve_ml_accuracy.py
│   │   └── results/
│   └── models/
│
├── analysis_logs/             # Hasil analisis (JSON)
├── backup/                    # File-file backup
├── tests/                     # File test
├── docs/                      # Dokumentasi
├── data/                      # Data
│   ├── raw/                   # Data mentah
│   └── processed/             # Data olahan
└── models/                    # Model terlatih
```

## 🚀 Cara Menjalankan

### 1. Setup Awal
```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env dengan API keys Anda
```

### 2. Jalankan Aplikasi Utama
```bash
python main.py
```

### 3. Jalankan Analisis Spesifik

#### Technical Analysis Saja
```bash
python analysis/auto_analysis.py
```

#### Technical + ML + News (Rekomendasi)
```bash
python analysis/comprehensive_analysis.py
```

#### Analisis 30+ Saham
```bash
python analysis/expanded_analysis.py
```

#### Training/Ulang ML Model
```bash
python ml_system/training/improve_ml_accuracy.py
```

## 📊 Fitur-Fitur

### 1. **Technical Analysis**
- RSI, MACD, Bollinger Bands
- Moving Averages (SMA, EMA)
- Volume Analysis
- Price Patterns
- Support/Resistance Levels

### 2. **Machine Learning**
- Ensemble Models (Random Forest, XGBoost)
- 43+ Technical Features
- Probability Predictions
- Risk Assessment

### 3. **News Sentiment**
- Market sentiment analysis
- Sector-specific sentiment
- News impact scoring

### 4. **Comprehensive Analysis**
- Menggabungkan semua metode
- Weighted scoring system
- Risk-adjusted recommendations
- Sector performance ranking

## ⚙️ Konfigurasi

### Environment Variables (.env)
```bash
# API Keys
GEMINI_API_KEY=your_gemini_api_key
NEWS_API_KEY=your_news_api_key
TRADING_ECONOMICS_KEY=your_key

# Model Configuration
MODEL_VERSION=v2
VALIDATION_ACCURACY_THRESHOLD=0.6
```

## 📈 Output & Reports

### 1. Console Output
- Real-time analysis results
- Top 5 stock recommendations
- Sector performance summary
- Market sentiment indicators

### 2. JSON Logs (di `analysis_logs/`)
- Detailed analysis per stock
- Historical performance
- Model confidence scores
- Feature importance

### 3. Model Files (di `ml_system/models/`)
- Trained models (.pkl)
- Scalers
- Feature metadata
- Training results

## 🎯 Cara Membaca Rekomendasi

### Skor Komprehensif (0-1.0):
- **0.70-1.0**: STRONG BUY
- **0.60-0.69**: BUY
- **0.45-0.59**: HOLD
- **0.35-0.44**: SELL
- **0.00-0.34**: STRONG SELL

### Risk Levels:
- **LOW**: Volatility < 25%
- **MEDIUM**: Volatility 25-40%
- **HIGH**: Volatility > 40%

## ⚠️ Penting!

1. **Akurasi ML ~40%** adalah normal untuk prediksi saham
2. Gunakan sebagai **decision support**, bukan automatic trading
3. Selalu gunakan **stop-loss** 2-4%
4. Rekomendasi terbaik ketika:
   - Technical > 0.6
   - ML confidence > 60%
   - News sentiment positif

## 📊 Update Model

Model sebaiknya di-retrain:
- **Mingguan** untuk market conditions terbaru
- **Bulanan** untuk full retraining
- Setelah **major market events**

```bash
# Retrain dengan data terbaru
python ml_system/training/train_models.py
```

## 🔧 Troubleshooting

### ML Accuracy Rendah (<40%)
- Check data quality
- Increase training data
- Add more features
- Use longer prediction window

### API Errors
- Verify API keys
- Check rate limits
- Ensure stable internet

### No Recommendations
- Market might be in transition
- Check risk parameters
- Lower confidence thresholds

## 📞 Support

Untuk issues atau questions:
1. Check logs di `analysis_logs/`
2. Review model performance
3. Verify API configurations

## 📝 Changelog

### v2.0 (Latest)
- Added comprehensive analysis
- Improved feature engineering
- Better risk management
- Sector analysis

### v1.0
- Basic technical analysis
- Simple ML integration
- Manual stock selection