#!/usr/bin/env python3
"""Generate comprehensive trading analysis reports"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.trading_analysis_generator import trading_analysis_generator
from scripts.enhanced_stock_signal_cli import fetch_signal, load_config, FinancialDataFetcher
from analyzers.market_analyzer import MarketAnalyzer
from rich.console import Console
from rich.panel import Panel
import json

def main():
    # Get symbols from command line or use defaults
    if len(sys.argv) > 1:
        symbols = sys.argv[1:]
    else:
        symbols = ["NRCA", "MTFN", "MOLI"]

    # Initialize
    config = load_config()
    analyzer = MarketAnalyzer(config)
    fetcher = FinancialDataFetcher() if FinancialDataFetcher else None
    console = Console()

    print("=" * 100)
    print("COMPREHENSIVE TRADING ANALYSIS REPORT")
    print("=" * 100)
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    for symbol in symbols:
        print(f"\n{'=' * 100}")
        print(f"TRADING ANALYSIS FOR {symbol}")
        print(f"{'=' * 100}")

        try:
            # Get comprehensive data
            print(f"[+] Processing {symbol}...")

            # Use M5 timeframe with 500 candles for complete analysis
            timeframe = "M5"
            candles = 500

            # Fetch signal data
            signal_data = fetch_signal(
                symbol, timeframe, candles, config, analyzer, console
            )

            if signal_data.get("error"):
                print(f"[X] Error fetching data: {signal_data['error']}")
                continue

            # Get technical analysis
            analysis = signal_data.get('analysis', {})
            technical_data = {
                'metrics': signal_data.get('metrics', {}),
                'stock': signal_data.get('stock', {}),
                'overall': signal_data.get('overall', {})
            }

            # Get financial data
            print("[+] Fetching financial data...")
            if fetcher:
                financial_data = fetcher.get_key_metrics(symbol)
            else:
                financial_data = {}

            # Get ML prediction
            ml_signal = signal_data.get('ml_prediction', {}) or {'signal': 'HOLD', 'confidence': 0.5}

            # Get price data for technical calculations
            df = signal_data.get('data')
            if df is None or df.empty:
                print(f"[X] No price data available for {symbol}")
                continue

            # Generate comprehensive trading analysis
            print("[+] Generating trading analysis...")
            result = trading_analysis_generator.generate_comprehensive_analysis(
                symbol, df, technical_data, financial_data, ml_signal
            )

            if result['success']:
                data = result['data']

                # Display comprehensive analysis
                console.print(f"\n[bold cyan]{symbol} - Current Price: Rp {data['current_price']:,.0f}[/]")

                # Technical levels
                tech_levels = data.get('technical_levels', {})
                console.print("\n[bold]Technical Levels:[/]")
                console.print(f"  Trend: {tech_levels.get('trend', 'UNKNOWN')}")
                console.print(f"  RSI: {tech_levels.get('rsi', 0):.1f}")
                console.print(f"  Volume Ratio: {tech_levels.get('volume_ratio', 0):.1f}x")
                console.print(f"  ATR: {tech_levels.get('atr', 0):.1f}")

                # Entry/Exit signals
                signals = data.get('signals', {})
                console.print(f"\n[bold green]ENTRY IDEAL:[/] Rp {signals.get('entry_ideal', 0):,.0f}–Rp {signals.get('entry_ideal', 0) * 1.02:,.0f}")
                console.print(f"  ({signals.get('entry_reasoning', '')})")

                console.print(f"\n[bold yellow]ENTRY AGGRESIF:[/] Rp {signals.get('entry_aggressive', 0):,.0f}–Rp {signals.get('entry_aggressive', 0) * 1.05:,.0f}")
                console.print(f"  (Breakout confirmation required)")

                # TP & SL
                console.print(f"\n[bold green]TAKE PROFIT:[/]")
                console.print(f"  TP1: Rp {signals.get('tp1', 0):,.0f} ({signals.get('risk_reward_tp1', 0):.1f} R/R)")
                console.print(f"  TP2: Rp {signals.get('tp2', 0):,.0f} ({signals.get('risk_reward_tp2', 0):.1f} R/R)")

                console.print(f"\n[bold red]STOP LOSS:[/] Rp {signals.get('sl', 0):,.0f}")
                console.print(f"  ({signals.get('sl_reasoning', '')})")

                # Risk percentages
                console.print(f"\n[bold]Risk-Reward Percentages:[/]")
                entry = signals.get('entry_ideal', 0)
                if entry > 0:
                    tp1_pct = ((signals.get('tp1', 0) - entry) / entry * 100)
                    tp2_pct = ((signals.get('tp2', 0) - entry) / entry * 100)
                    sl_pct = ((signals.get('sl', 0) - entry) / entry * 100)
                    console.print(f"  TP1: {tp1_pct:+.2f}%")
                    console.print(f"  TP2: {tp2_pct:+.2f}%")
                    console.print(f"  SL: {sl_pct:+.2f}%")

                # Financial highlights
                financial = data.get('financial_highlights', {})
                if financial:
                    console.print(f"\n[bold]Financial Highlights:[/]")
                    if financial.get('revenue_growth'):
                        console.print(f"  Revenue Growth: {financial['revenue_growth']:.1f}%")
                    if financial.get('net_margin'):
                        console.print(f"  Net Margin: {financial['net_margin']:.1f}%")
                    if financial.get('roe'):
                        console.print(f"  ROE: {financial['roe']:.1f}%")

                # ML Signal
                ml = data.get('ml_signal', {})
                console.print(f"\n[bold]ML Signal:[/] {ml.get('signal', 'HOLD')} ({ml.get('confidence', 0):.0%} confidence)")

                # AI Analysis
                ai_analysis = data.get('ai_analysis', '')
                if ai_analysis:
                    from rich.panel import Panel
                    console.print("\n")
                    panel = Panel(
                        ai_analysis,
                        title="[bold magenta]AI Trading Analysis[/]",
                        border_style="magenta",
                        padding=(1, 2)
                    )
                    console.print(panel)

                # Trading Strategies
                strategies = data.get('strategies', {})
                if strategies:
                    console.print(f"\n[bold]Trading Strategies:[/]")

                    scalping = strategies.get('scalping', {})
                    console.print(f"  [cyan]Scalping:[/] {scalping.get('suitability', 'UNKNOWN')}")
                    console.print(f"    Timeframe: {scalping.get('timeframe', 'N/A')}")
                    console.print(f"    Entry: {scalping.get('entry_rules', 'N/A')}")

                    swing = strategies.get('swing', {})
                    console.print(f"  [green]Swing Trading:[/] {swing.get('suitability', 'UNKNOWN')}")
                    console.print(f"    Timeframe: {swing.get('timeframe', 'N/A')}")
                    console.print(f"    Entry: {swing.get('entry_rules', 'N/A')}")

                # Risk Metrics
                risk = data.get('risk_metrics', {})
                if risk:
                    console.print(f"\n[bold red]Risk Metrics:[/]")
                    console.print(f"  Max Loss: {risk.get('max_loss_percent', 0):.1f}%")
                    console.print(f"  Max Gain: {risk.get('max_gain_percent', 0):.1f}%")
                    console.print(f"  Volatility: {risk.get('volatility', 'UNKNOWN')}")

            else:
                console.print(f"[X] Failed to generate analysis: {result.get('error', 'Unknown error')}")

        except Exception as e:
            console.print(f"[X] Error analyzing {symbol}: {str(e)}")

    print("\n" + "=" * 100)
    print("END OF TRADING ANALYSIS REPORT")
    print("=" * 100)

if __name__ == "__main__":
    from datetime import datetime
    main()