#!/usr/bin/env python3
"""Generate narrative analysis for stocks"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.enhanced_narrative_generator import EnhancedNarrativeGenerator
from analyzers.market_analyzer import MarketAnalyzer
from analyzers.config_loader import ConfigLoader
import yfinance as yf
from datetime import datetime, timedelta

def get_stock_data(symbol, days=59):
    """Get stock data for analysis"""
    ticker = yf.Ticker(f"{symbol}.JK")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    df = ticker.history(start=start_date, end=end_date, interval="5m")
    return df, ticker

def main():
    symbols = ["NRCA", "MTFN", "MOLI"]

    # Initialize components
    config = ConfigLoader("config.json")
    analyzer = MarketAnalyzer(config)
    narrative_gen = EnhancedNarrativeGenerator()

    print("=" * 80)
    print("STOCK NARRATIVE ANALYSIS REPORT")
    print("=" * 80)
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    for symbol in symbols:
        print(f"\n{'=' * 60}")
        print(f"ANALYSIS FOR {symbol}")
        print(f"{'=' * 60}")

        try:
            # Get data
            df, ticker = get_stock_data(symbol)

            if df.empty:
                print(f"❌ No data available for {symbol}")
                continue

            # Get technical analysis
            analysis = analyzer.analyze(df, symbol, config.get_strategy())

            # Get financial data
            financial_data = analyzer.get_financial_data(symbol)

            # Get ML prediction
            ml_data = analyzer.get_ml_prediction(symbol, df)

            # Prepare quarterly data (mock for now)
            quarterly_data = {
                "latest_quarter": {},
                "quarterly_comparison": {}
            }

            # Prepare growth data (mock for now)
            growth_data = {
                "revenue_trend": [],
                "profit_trend": []
            }

            # Generate narrative
            print(f"Generating narrative for {symbol}...")
            result = narrative_gen.generate_narrative_analysis(
                symbol, analysis, financial_data, ml_data,
                quarterly_data, growth_data
            )

            if result['success']:
                data = result['data']

                # Print title
                if 'title' in data:
                    print(f"\n📊 {data['title']}")
                else:
                    print(f"\n📊 {symbol} Narrative Analysis")

                # Print content
                if 'content' in data:
                    print(f"\n{data['content']}")

                # Print metadata
                if 'metadata' in data:
                    meta = data['metadata']
                    print(f"\n---")
                    print(f"Model: {meta.get('model_used', 'N/A')}")
                    print(f"Generated: {meta.get('generated_at', 'N/A')}")
                    print(f"Word count: {meta.get('word_count', 0)}")
            else:
                print(f"❌ Failed to generate narrative: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)

if __name__ == "__main__":
    main()