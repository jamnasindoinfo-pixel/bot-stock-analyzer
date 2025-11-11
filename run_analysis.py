#!/usr/bin/env python3
"""
Runner script untuk memudahkan menjalankan berbagai jenis analisis
"""
import os
import sys
import json
import argparse
from datetime import datetime

def load_config():
    """Load configuration from config file"""
    try:
        with open('config/config.json', 'r') as f:
            return json.load(f)
    except:
        return {}

def run_technical_analysis():
    """Jalankan technical analysis sederhana"""
    print("=" * 80)
    print("RUNNING TECHNICAL ANALYSIS")
    print("=" * 80)
    os.system('python analysis/auto_analysis.py')

def run_comprehensive_analysis():
    """Jalankan comprehensive analysis (Technical + ML + News)"""
    print("=" * 80)
    print("RUNNING COMPREHENSIVE ANALYSIS (Technical + ML + News)")
    print("=" * 80)
    os.system('python analysis/comprehensive_analysis.py')

def run_expanded_analysis():
    """Jalankan analisis 30+ saham"""
    print("=" * 80)
    print("RUNNING EXPANDED ANALYSIS (30+ Stocks)")
    print("=" * 80)
    os.system('python analysis/expanded_analysis.py')

def run_ml_training():
    """Jalankan training ML model"""
    print("=" * 80)
    print("TRAINING ML MODELS")
    print("=" * 80)
    os.system('python ml_system/training/improve_ml_accuracy.py')

def run_main_app():
    """Jalankan aplikasi utama"""
    print("=" * 80)
    print("RUNNING MAIN APPLICATION")
    print("=" * 80)
    os.system('python main.py')

def clean_logs():
    """Bersihkan file log lama"""
    import glob
    logs_dir = 'analysis_logs'
    if os.path.exists(logs_dir):
        log_files = glob.glob(f'{logs_dir}/*.json')
        print(f"Found {len(log_files)} log files")

        # Keep last 10 files
        log_files.sort()
        files_to_delete = log_files[:-10]

        for file in files_to_delete:
            os.remove(file)
            print(f"Deleted: {file}")
        print("Cleanup complete!")

def show_recent_analysis():
    """Tampilkan hasil analisis terbaru"""
    import glob
    logs_dir = 'analysis_logs'
    if not os.path.exists(logs_dir):
        print("No analysis logs found!")
        return

    log_files = glob.glob(f'{logs_dir}/*.json')
    if not log_files:
        print("No analysis logs found!")
        return

    # Get the most recent file
    log_files.sort(key=os.path.getmtime, reverse=True)
    latest_file = log_files[0]

    print(f"\nLatest Analysis: {latest_file}")
    print(f"Modified: {datetime.fromtimestamp(os.path.getmtime(latest_file))}")

    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)

        # Show summary
        if 'top_5' in data:
            print("\nTop 5 Recommendations:")
            for i, stock in enumerate(data['top_5'], 1):
                symbol = stock.get('symbol', 'N/A')
                rec = stock.get('comprehensive', {}).get('recommendation', 'N/A')
                score = stock.get('comprehensive', {}).get('score', 0)
                price = stock.get('current_price', 0)
                print(f"{i}. {symbol} - {rec} (Score: {score:.2f}) - Rp {price:,.0f}")

        if 'sector_performance' in data:
            print("\nSector Performance:")
            for sector, score in data['sector_performance'].items():
                print(f"  {sector}: {score:.2f}")

    except Exception as e:
        print(f"Error reading file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Bot Stock Market Analysis Runner')
    parser.add_argument('--type', '-t',
                       choices=['technical', 'comprehensive', 'expanded', 'ml', 'main'],
                       help='Type of analysis to run')
    parser.add_argument('--clean', action='store_true',
                       help='Clean old log files')
    parser.add_argument('--show', action='store_true',
                       help='Show recent analysis results')
    parser.add_argument('--all', action='store_true',
                       help='Run all analysis types')

    args = parser.parse_args()

    # Load config
    config = load_config()
    print(f"\nBot {config.get('app', {}).get('name', 'Stock Market')} v{config.get('app', {}).get('version', '1.0')}")
    print("=" * 80)

    if args.clean:
        clean_logs()
        return

    if args.show:
        show_recent_analysis()
        return

    if args.all:
        print("\nRunning all analysis types...\n")
        run_technical_analysis()
        print("\n" + "="*80 + "\n")
        run_comprehensive_analysis()
        print("\n" + "="*80 + "\n")
        run_expanded_analysis()
        print("\nAll analysis complete!")
        return

    if args.type == 'technical':
        run_technical_analysis()
    elif args.type == 'comprehensive':
        run_comprehensive_analysis()
    elif args.type == 'expanded':
        run_expanded_analysis()
    elif args.type == 'ml':
        run_ml_training()
    elif args.type == 'main':
        run_main_app()
    else:
        # Interactive mode
        print("\nPilih jenis analisis:")
        print("1. Technical Analysis (sederhana)")
        print("2. Comprehensive Analysis (Technical + ML + News)")
        print("3. Expanded Analysis (30+ saham)")
        print("4. ML Model Training")
        print("5. Main Application")
        print("6. Show Recent Results")
        print("7. Clean Old Logs")
        print("0. Exit")

        choice = input("\nPilihan (0-7): ").strip()

        if choice == '1':
            run_technical_analysis()
        elif choice == '2':
            run_comprehensive_analysis()
        elif choice == '3':
            run_expanded_analysis()
        elif choice == '4':
            run_ml_training()
        elif choice == '5':
            run_main_app()
        elif choice == '6':
            show_recent_analysis()
        elif choice == '7':
            clean_logs()
        elif choice == '0':
            print("Goodbye!")
        else:
            print("Pilihan tidak valid!")

if __name__ == "__main__":
    main()