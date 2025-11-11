#!/usr/bin/env python3
"""Generate narrative analysis for stocks using the existing CLI infrastructure"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.enhanced_stock_signal_cli import fetch_signal, generate_narrative_analysis, load_config, TIMEFRAME_MAP
from analyzers.market_analyzer import MarketAnalyzer
from rich.console import Console
import json

def main():
    symbols = ["NRCA", "MTFN", "MOLI"]

    # Initialize
    config = load_config()
    analyzer = MarketAnalyzer(config)
    console = Console()

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
            # Generate narrative analysis using existing function
            print(f"[+] Processing {symbol}...")

            # Use M5 timeframe with 500 candles for ML analysis
            timeframe = "M5"
            candles = 500

            # Generate narrative
            result = generate_narrative_analysis(
                symbol, timeframe, candles, config, analyzer, console
            )

            if result['success']:
                data = result['data']

                # Print title
                if 'title' in data:
                    print(f"\n{data['title']}")
                else:
                    print(f"\n{symbol} Narrative Analysis")

                # Print content
                if 'content' in data:
                    print(f"\n{data['content']}")

                # Print metadata if available
                if 'metadata' in data:
                    meta = data['metadata']
                    print(f"\n---")
                    print(f"Model: {meta.get('model_used', 'N/A')}")
                    print(f"Generated: {meta.get('generated_at', 'N/A')}")
                    print(f"Word count: {meta.get('word_count', 0)}")
            else:
                print(f"[X] Failed to generate narrative: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"[X] Error analyzing {symbol}: {str(e)}")

    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)

if __name__ == "__main__":
    from datetime import datetime
    main()