#!/usr/bin/env python3
"""
Integration script to add trading analysis to the CLI
This modifies the enhanced_stock_signal_cli.py to include trading analysis commands
"""

import re

def add_trading_analysis_command():
    """Add trading analysis command to CLI"""

    cli_file = "scripts/enhanced_stock_signal_cli.py"

    # Read the current CLI file
    with open(cli_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if trading analysis is already integrated
    if 'trading_analysis' in content:
        print("[+] Trading analysis already integrated in CLI")
        return

    # Find the help text section to add new command
    help_pattern = r'(\n(\s+)- \[yellow\]ai.*?--narrative.*?: \[bold magenta\]NEW!\[/\] Analisis naratif mendalam seperti financial journal\n)'

    # Add trading analysis command to help
    new_help = r'\1\2- [yellow]trading <symbol>[/]: [bold cyan]NEW![/] Generate comprehensive trading analysis with entry/exit points\n'

    content = re.sub(help_pattern, new_help, content)

    # Find the command parsing section
    command_pattern = r'(# Parse command for --narrative flag\s+ai_parts = raw\.split\(\)\s+narrative_mode = "--narrative" in ai_parts\s+ai_symbols = \[sym for sym in ai_parts\[1:\] if sym != "--narrative"\] or list\(last_results\.keys\(\)\))'

    # Add trading command parsing
    trading_command = r'\1\n\n        # Check for trading analysis command\n        if raw.lower().startswith("trading"):\n            trading_parts = raw.split()\n            if len(trading_parts) > 1:\n                symbol = trading_parts[1].upper()\n                console.print(f"\\n[bold cyan]Generating trading analysis for {symbol}...[/]")\n                trading_result = generate_trading_analysis(symbol, config, analyzer, console)\n                if trading_result:\n                    display_trading_analysis(console, symbol, trading_result)\n                else:\n                    console.print(f"[error]Failed to generate trading analysis for {symbol}[/error]")\n                continue\n            else:\n                console.print("[error]Please specify a symbol. Usage: trading <symbol>[/error]")\n                continue'

    content = re.sub(command_pattern, trading_command, content, flags=re.MULTILINE | re.DOTALL)

    # Add import and function definitions at the end of imports
    import_pattern = r'(from ai\.enhanced_narrative_generator import EnhancedNarrativeGenerator)'
    new_import = r'\1\nfrom ai.trading_analysis_generator import trading_analysis_generator'
    content = re.sub(import_pattern, new_import, content)

    # Add trading analysis functions before main()
    function_pattern = r'(\n\ndef main\(\) -> None:)'

    trading_functions = '''
def generate_trading_analysis(symbol: str, config: dict, analyzer, console) -> dict:
    """Generate comprehensive trading analysis"""
    try:
        from scripts.enhanced_stock_signal_cli import fetch_signal, TIMEFRAME_MAP

        # Get data for analysis
        timeframe = "M5"  # Use 5-minute timeframe
        candles = 500    # Get 500 candles for better analysis

        # Fetch signal data
        signal_data = fetch_signal(
            symbol, timeframe, candles, config, analyzer, console
        )

        if signal_data.get("error"):
            console.print(f"[error]Error fetching data: {signal_data['error']}[/error]")
            return None

        # Get technical analysis
        analysis = signal_data.get('analysis', {})
        technical_data = {
            'metrics': signal_data.get('metrics', {}),
            'stock': signal_data.get('stock', {}),
            'overall': signal_data.get('overall', {})
        }

        # Get financial data
        if hasattr(analyzer, 'get_financial_data'):
            financial_data = analyzer.get_financial_data(symbol)
        else:
            financial_data = {}

        # Get ML prediction
        ml_signal = signal_data.get('ml_prediction', {}) or {'signal': 'HOLD', 'confidence': 0.5}

        # Get price data
        df = signal_data.get('data')
        if df is None or df.empty:
            console.print(f"[error]No price data available for {symbol}[/error]")
            return None

        # Generate trading analysis
        result = trading_analysis_generator.generate_comprehensive_analysis(
            symbol, df, technical_data, financial_data, ml_signal
        )

        return result.get('data') if result.get('success') else None

    except Exception as e:
        console.print(f"[error]Error generating trading analysis: {str(e)}[/error]")
        return None


def display_trading_analysis(console, symbol: str, data: dict):
    """Display trading analysis results"""
    try:
        from rich.panel import Panel
        from rich.table import Table

        # Header
        console.print(f"\\n[bold cyan]Trading Analysis for {symbol}[/]")
        console.print(f"Current Price: Rp {data.get('current_price', 0):,.0f}")
        console.print()

        # Technical Levels Table
        tech_levels = data.get('technical_levels', {})
        tech_table = Table(title="Technical Levels")
        tech_table.add_column("Indicator", style="cyan")
        tech_table.add_column("Value", style="green")

        tech_table.add_row("Trend", tech_levels.get('trend', 'UNKNOWN'))
        tech_table.add_row("RSI", f"{tech_levels.get('rsi', 0):.1f}")
        tech_table.add_row("Volume Ratio", f"{tech_levels.get('volume_ratio', 0):.1f}x")
        tech_table.add_row("ATR", f"{tech_levels.get('atr', 0):.1f}")

        console.print(tech_table)
        console.print()

        # Entry/Exit Signals
        signals = data.get('signals', {})
        console.print("[bold green]ENTRY POINTS[/]")
        console.print(f"Entry Ideal: Rp {signals.get('entry_ideal', 0):,.0f} - Rp {signals.get('entry_ideal', 0) * 1.02:,.0f}")
        console.print(f"Entry Aggressive: Rp {signals.get('entry_aggressive', 0):,.0f} - Rp {signals.get('entry_aggressive', 0) * 1.05:,.0f}")
        console.print()

        console.print("[bold red]EXIT POINTS[/]")
        console.print(f"Take Profit 1: Rp {signals.get('tp1', 0):,.0f}")
        console.print(f"Take Profit 2: Rp {signals.get('tp2', 0):,.0f}")
        console.print(f"Stop Loss: Rp {signals.get('sl', 0):,.0f}")
        console.print()

        # Risk-Reward
        entry = signals.get('entry_ideal', 0)
        if entry > 0:
            tp1_pct = ((signals.get('tp1', 0) - entry) / entry * 100)
            tp2_pct = ((signals.get('tp2', 0) - entry) / entry * 100)
            sl_pct = ((signals.get('sl', 0) - entry) / entry * 100)

            console.print("[bold yellow]RISK-REWARD[/]")
            console.print(f"TP1: {tp1_pct:+.2f}% | TP2: {tp2_pct:+.2f}% | SL: {sl_pct:+.2f}%")
            console.print()

        # AI Analysis
        ai_analysis = data.get('ai_analysis', '')
        if ai_analysis:
            panel = Panel(
                ai_analysis,
                title="[bold magenta]AI Trading Analysis[/]",
                border_style="magenta",
                padding=(1, 2)
            )
            console.print(panel)

        # Strategies
        strategies = data.get('strategies', {})
        if strategies:
            console.print(f"\\n[bold]Recommended Strategies:[/]")
            scalping = strategies.get('scalping', {})
            console.print(f"  [cyan]Scalping:[/] {scalping.get('suitability', 'UNKNOWN')}")
            swing = strategies.get('swing', {})
            console.print(f"  [green]Swing Trading:[/] {swing.get('suitability', 'UNKNOWN')}")

    except Exception as e:
        console.print(f"[error]Error displaying analysis: {str(e)}[/error]")

\n
'''

    content = re.sub(function_pattern, trading_functions + r'\1', content)

    # Write the updated content back
    with open(cli_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print("[+] Successfully integrated trading analysis into CLI")
    print("[+] You can now use: trading <symbol> command in the CLI")

if __name__ == "__main__":
    print("Integrating Trading Analysis into Enhanced Stock Signal CLI...")
    print("-" * 60)
    add_trading_analysis_command()