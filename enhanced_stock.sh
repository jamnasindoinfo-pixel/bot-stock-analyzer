#!/bin/bash
# Enhanced Stock Signal CLI Launcher with ML Integration
# Usage: ./enhanced_stock.sh [command] [symbols...]
# Examples:
#   ./enhanced_stock.sh              # Show menu
#   ./enhanced_stock.sh ml BBCA     # ML prediction for BBCA
#   ./enhanced_stock.sh train demo  # Train ML models

# Check if python is available
if ! command -v python &> /dev/null; then
    echo "[ERROR] Python is not installed or not in PATH"
    echo "Please install Python and add it to your PATH"
    exit 1
fi

# No arguments - show menu
if [ $# -eq 0 ]; then
    echo "=== Enhanced Stock Signal CLI with ML v2 ==="
    echo ""
    echo "Usage: ./enhanced_stock.sh [command] [symbols...]"
    echo ""
    echo "Commands:"
    echo "  ml [symbol]      - ML prediction (e.g., ./enhanced_stock.sh ml BBCA)"
    echo "  train demo       - Train ML models with demo data"
    echo "  train [symbols]  - Train specific symbols"
    echo "  status          - Check ML system status"
    echo ""
    echo "Examples:"
    echo "  ./enhanced_stock.sh ml BBCA"
    echo "  ./enhanced_stock.sh ml BBCA BBRI TLKM"
    echo "  ./enhanced_stock.sh status"
    echo ""
    exit 0
fi

# Handle commands
command=$1
shift

case $command in
    "ml")
        if [ $# -eq 0 ]; then
            echo "[ERROR] Please provide symbol(s) for ML prediction"
            echo "Example: ./enhanced_stock.sh ml BBCA"
            exit 1
        fi
        echo "=== ML Prediction Results ==="
        for symbol in "$@"; do
            echo ""
            echo "Analyzing: $symbol"
            python ml_direct.py "$symbol"
        done
        ;;
    "status")
        echo "=== ML System Status ==="
        python -c "
import sys
sys.path.insert(0, '.')
from ml_system.core.ml_predictor_v2 import MLPredictorV2
predictor = MLPredictorV2()
if predictor.load_models():
    info = predictor.get_model_info()
    print(f'ML System: ACTIVE')
    print(f'Version: v2')
    print(f'Model Type: {info.get(\"model_types\", \"Unknown\")}')
    print(f'Features: {info.get(\"feature_count\", 0)}')
    print(f'Training Info: {info.get(\"training_info\", {})}')
else:
    print('ML System: NOT TRAINED')
    print('Run: ./enhanced_stock.sh train demo')
"
        ;;
    "train")
        if [ $# -eq 0 ] || [ "$1" = "demo" ]; then
            echo "=== Training ML Models (Demo) ==="
            echo "This will train ML models with 15 popular Indonesian stocks..."
            echo "Note: This may take 5-10 minutes"
            read -p "Continue? (y/N): " confirm
            if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
                python run_simple.py train --demo
            fi
        else
            echo "=== Training ML Models ==="
            echo "Training for: $@"
            python run_simple.py train "$@"
        fi
        ;;
    "help"|"-h"|"--help")
        echo "=== Enhanced Stock Signal CLI Help ==="
        echo ""
        echo "Commands:"
        echo "  ml [symbols]     - ML prediction for specified symbols"
        echo "  train [demo|symbols] - Train ML models"
        echo "  status          - Check ML system status"
        echo "  help            - Show this help"
        echo ""
        echo "Examples:"
        echo "  ./enhanced_stock.sh ml BBCA"
        echo "  ./enhanced_stock.sh ml BBCA BBRI TLKM"
        echo "  ./enhanced_stock.sh train demo"
        echo "  ./enhanced_stock.sh status"
        echo ""
        ;;
    *)
        echo "[ERROR] Unknown command: $command"
        echo "Use './enhanced_stock.sh help' for available commands"
        exit 1
        ;;
esac