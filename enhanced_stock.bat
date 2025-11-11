@echo off
REM Enhanced Stock Signal CLI Launcher v5 with ML v5 Integration
REM Uses latest ML v5 models (83.1% accuracy)
REM Usage: enhanced_stock.bat [options]
REM
REM Examples:
REM   enhanced_stock.bat                    - Run comprehensive analysis
REM   enhanced_stock.bat -s BBCA.JK         - Analyze single stock
REM   enhanced_stock.bat --train           - Train ML models
REM   enhanced_stock.bat --status          - Show system status

setlocal enabledelayedexpansion

REM Check if python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python and add it to your PATH
    pause
    exit /b 1
)

echo Enhanced Stock Signal Programs
echo Using Machine Learning
echo PT Jaminan Nasional Indonesia
echo Developed by: Muklis
echo Version: 1.0.0
echo.

REM Execute the enhanced CLI
python "scripts\enhanced_stock_signal_cli.py" %*

exit /b %errorlevel%