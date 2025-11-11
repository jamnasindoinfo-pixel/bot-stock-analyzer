# Trading Analysis Guide

## Overview
The Enhanced Stock Signal CLI now includes comprehensive trading analysis that generates detailed reports similar to professional trading analysis, including entry/exit points, risk management, and trading strategies.

## Features

### 1. Technical Analysis
- **Support & Resistance Levels**: Automatically calculated from price action
- **EMA Alignment**: 9, 21, and 50-period EMAs for trend confirmation
- **RSI Momentum**: Overbought/oversold identification
- **Volume Analysis**: Volume ratio compared to 20-day average
- **ATR Calculation**: For dynamic stop losses and position sizing

### 2. Entry/Exit Signals
- **Entry Ideal Zone**: Conservative entry based on pullbacks
- **Entry Aggressive Zone**: Breakout entries for momentum trades
- **Take Profit Levels**: TP1 and TP2 based on resistance levels
- **Stop Loss**: ATR-based with support level confirmation
- **Risk-Reward Ratios**: Calculated for each target

### 3. Trading Strategies
- **Scalping**: Short-term 5m/15m timeframe strategies
- **Swing Trading**: Medium-term Daily/4H timeframe strategies
- **Position Sizing**: Recommendations based on volatility
- **ML Integration**: 83.1% accuracy predictions with confidence levels

### 4. AI Analysis
- Narrative analysis similar to conceptAI.md style
- Fundamental insights integration
- News sentiment analysis
- Catalyst identification

## Usage

### Via CLI Command
```bash
# Start the CLI
python scripts/enhanced_stock_signal_cli.py

# Generate trading analysis for a stock
trading BBCA

# Multiple stocks
trading BBCA BBRI TLKM
```

### Direct Script
```bash
# Generate analysis for specific stocks
python generate_trading_analysis.py BBCA BBRI TLKM

# Or run the demo with sample data
python trading_analysis_demo.py
```

## Example Output

```
================================================================================
TRADING ANALYSIS FOR BBCA
================================================================================

Current Price: Rp 9,750

Technical Levels:
  Trend: UPTREND
  RSI: 65.3
  Volume Ratio: 1.5x
  ATR: 124.3

ENTRY POINTS
Entry Ideal: Rp 9,600 - Rp 9,792
  (Uptrend confirmed with EMA alignment; RSI 65.3 shows room for upside)

Entry Aggressive: Rp 9,750 - Rp 10,238
  (Breakout confirmation required; Volume spike 1.5x confirms interest)

EXIT POINTS
Take Profit 1: Rp 9,854 (Nearby resistance at 9854)
Take Profit 2: Rp 9,958 (Major resistance at 9958)
Stop Loss: Rp 9,476 (Strong support at 9476; ATR buffer of 1.2x (149.2 points))

Risk-Reward Percentages:
TP1: +2.67%
TP2: +3.75%
SL: -1.29%

ML Signal: BUY (75% confidence)

AI Trading Analysis
┌─────────────────────────────────────────┐
│           AI Trading Analysis           │
├─────────────────────────────────────────┤
│ BBCA sedang berada di fase uptrend yang │
│ solid dengan EMA 9, 21, dan 50 yang     │
│ terkonfirmasi...                       │
└─────────────────────────────────────────┘

Recommended Strategies:
  Scalping: HIGHLY SUITABLE
    - Timeframe: 5m/15m
    - Entry: Pullback to 9600-9792 with volume confirmation

  Swing Trading: SUITABLE
    - Timeframe: Daily/4H
    - Entry: Breakout confirmation above 9750
```

## Risk Management

### Position Sizing Rules
1. **High Volatility** (ATR > 2% of price): Use 1% risk
2. **Medium Volatility** (ATR 1-2% of price): Use 2% risk
3. **Low Volatility** (ATR < 1% of price): Use 3% risk

### Stop Loss Rules
- Always use ATR-based stops (1.2x - 1.5x ATR)
- Combine with support/resistance levels
- Adjust for news catalysts

### Take Profit Rules
- TP1: First resistance or 1.5x ATR
- TP2: Second resistance or 3x ATR
- Scale out: 50% at TP1, 50% at TP2

## Integration with Existing Features

### ML Signals
- Trading analysis incorporates ML v5 predictions
- 83.1% accuracy on Indonesian stocks
- Confidence levels guide entry decisions

### Narrative Analysis
- Combined with technical signals
- Provides fundamental context
- Identifies catalysts and risks

### Multiple Timeframes
- 5m: Entry timing
- 15m: Confirmation
- Daily: Trend direction

## Best Practices

1. **Always confirm with volume** - Entries without volume confirmation have lower success rates
2. **Respect ATR** - High volatility stocks require wider stops
3. **ML confidence matters** - Signals with >70% confidence have higher win rates
4. **Check news catalysts** - Earnings, dividend announcements, sector news
5. **Use proper position sizing** - Never risk more than 2% per trade

## Troubleshooting

### Error: "No 'close' column found"
- This usually means no data was fetched
- Check if the symbol is correct (BBCA vs BBCA.JK)
- Verify internet connection

### ML Signal Not Available
- Ensure you have 500+ candles of data
- Check if the stock is in the trained list (88 Indonesian stocks)
- Try refreshing the data

### AI Analysis Missing
- Check if Gemini API key is configured
- Verify rate limits haven't been exceeded
- Ensure internet connection is stable

## Configuration

The system uses the configuration from `config.json` and environment variables:

```bash
# LLM Provider (for AI analysis)
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.0-flash
GEMINI_API_KEY=your_key_here

# Enable fallback for reliability
LLM_FALLBACK_ENABLED=true
```

## Support

For issues or questions:
1. Check the logs for error messages
2. Verify all dependencies are installed
3. Ensure configuration is correct
4. Test with known symbols (BBCA, TLKM, BBRI)