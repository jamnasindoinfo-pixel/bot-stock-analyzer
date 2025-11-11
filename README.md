# 📊 Bot Analisis Saham Indonesia

**Sistem analisis saham Indonesia** yang menggabungkan Technical Analysis, AI (Gemini), dan Machine Learning untuk memberikan rekomendasi trading.

> ⚠️ **PENTING**: Ini adalah **sistem analisis**, BUKAN trading bot otomatis. Program memberikan rekomendasi untuk keputusan trading Anda.

## 🎯 Fungsi Utama

1. **Analisis Teknikal**
   - RSI, MACD, Bollinger Bands
   - Moving Averages
   - Volume analysis

2. **AI Analysis (Gemini)**
   - Analisis qualitative dengan Google AI
   - Interpretasi market sentiment

3. **Machine Learning**
   - Ensemble models (Random Forest, XGBoost, LSTM)
   - Prediksi berdasarkan 43+ fitur teknikal

4. **Saham Indonesia Focus**
   - Data dari Yahoo Finance (yfinance)
   - Top 10 saham: BBCA.JK, BBRI.JK, BBNI.JK, dll

## 🚀 Cara Menggunakan

### 1. Jalankan Program Utama
```bash
python main.py
```

Program akan:
1. Mengecek koneksi Gemini AI
2. Menampilkan daftar 10 saham Indonesia
3. Meminta user memilih saham (1-10)
4. Memberikan analisis komprehensif

### 2. Jalankan Analisis Otomatis
```bash
# Analisis semua 10 saham
python analysis/auto_analysis.py

# Analisis 30+ saham
python analysis/expanded_analysis.py

# Analisis gabungan (Technical + ML + News)
python analysis/comprehensive_analysis.py
```

### 3. Train/Ulang ML Model
```bash
python ml_system/training/improve_ml_accuracy.py
```

## 📊 Output yang Dihasilkan

### Console Output:
- Harga saham terkini
- Technical indicators (RSI, MACD, dll)
- Rekomendasi: BUY/HOLD/SELL
- Risk level (LOW/MEDIUM/HIGH)
- Support & resistance levels

### JSON Logs (di `analysis_logs/`):
- Detailed analysis per stock
- ML prediction scores
- Historical performance

## ⚙️ Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup API Keys
Buat file `.env`:
```bash
GEMINI_API_KEY=your_gemini_key
NEWS_API_KEY=your_news_key
```

### 3. Run the Program
```bash
python main.py
```

## 📈 Contoh Output

```
============================================================
BOT STOCK MARKET - NO MT5 VERSION
   Analisis Saham Indonesia dengan AI & Machine Learning
============================================================

[1] Mengecek koneksi Gemini AI...
[SUCCESS] Gemini AI terhubung

[2] Mendapatkan daftar saham...
[INFO] 10 saham tersedia:
   1. BBCA.JK: Rp 8,575
   2. BBRI.JK: Rp 3,930
   3. BBNI.JK: Rp 4,420
   ... dan 7 saham lainnya

[3] Pilih saham untuk analisis:
   1. BBCA.JK
   2. BBRI.JK
   ...

Masukkan nomor saham (1-10): 1

[ANALYSIS] BBCA.JK - Bank Central Asia
Current Price: Rp 8,575 (-1.15%)
RSI: 53.2 (Neutral)
Volume: 0.41x average
Recommendation: HOLD
Risk Level: MEDIUM

Technical Signals:
  RSI: Neutral
  MACD: Bearish crossover
  Volume: Below average
  Position: Above SMA20

ML Prediction: 0.55 (Confidence: 65%)
Gemini Analysis: "Saham sedang konsolidasi..."

Overall Recommendation: WAIT FOR BETTER ENTRY
```

## 📁 Struktur Proyek

```
Bot-Stock-Market/
├── main.py                     # Program utama (interactive)
├── analysis/                   # Script-script analisis
├── ml_system/                  # Machine Learning system
├── analyzers/                  # Market analyzer
├── analysis_logs/            # Hasil analisis (JSON)
└── config/                    # Konfigurasi
```

## ⚠️ Limitations & Disclaimer

1. **Akurasi ML ~40%** - Normal untuk prediksi pasar
2. **Bukan financial advice** - Gunakan untuk research saja
3. **Market risk** - Saham memiliki risiko tinggi
4. **Data delay** - Data dari yfinance ada delay

## 🔧 Troubleshooting

### Gemini API Error:
- Check API key di .env
- Pastikan quota masih ada

### No Stock Data:
- Check internet connection
- Yahoo Finance sedang maintenance

### ML Model Not Loaded:
- Jalankan training script dulu
- Check model files di ml_system/models/

## 💡 Tips Penggunaan

1. **Best Time**: Jam 09:00-15:30 WIB (market hours)
2. **Combine Methods**: Gunakan semua 3 metode analisis
3. **Risk Management**: Selalu gunakan stop-loss
4. **Diversify**: Jangan fokus ke 1 saham saja

## 📞 Support

Check `analysis_logs/` untuk detail hasil analisis.
Lihat `docs/` untuk dokumentasi teknis.

---

**© 2024 - Bot Analisis Saham Indonesia**
*Untuk edukasi dan research purposes only*