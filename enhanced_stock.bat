@echo off
REM Enhanced Stock Signal CLI Launcher with ML Integration
REM Usage: enhanced_stock.bat

setlocal enabledelayedexpansion

REM Check if python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python and add it to your PATH
    pause
    exit /b 1
)

REM Execute the enhanced CLI
python "scripts\enhanced_stock_signal_cli.py" %*

exit /b %errorlevel%