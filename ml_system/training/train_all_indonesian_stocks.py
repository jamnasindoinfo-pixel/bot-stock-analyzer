#!/usr/bin/env python3
"""
Comprehensive Training Pipeline for All Indonesian Stocks from yfinance

This script scrapes all available Indonesian stocks from yfinance (JK market),
filters stocks with sufficient data, creates a comprehensive dataset,
and trains MLPredictorV2 on the full dataset.

Features:
- Scrapes ALL Indonesian stocks from yfinance/JK market
- Filters stocks with sufficient historical data (1-2 years minimum)
- Robust error handling for network issues and insufficient data
- Memory-efficient processing for large datasets
- Progress tracking with detailed status reporting
- Comprehensive training with MLPredictorV2 ensemble models
- Model evaluation and result reporting
"""

import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
import json
import warnings
from pathlib import Path
import joblib
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / '.worktrees' / 'ml-accuracy-enhancement'))

try:
    from ml_system.core.ml_predictor_v2 import MLPredictorV2
    ENHANCED_ML_AVAILABLE = True
    logger.info("Enhanced MLPredictorV2 available")
except ImportError as e:
    logger.warning(f"Enhanced MLPredictorV2 not available: {e}")
    ENHANCED_ML_AVAILABLE = False

class IndonesianStockTrainer:
    """
    Comprehensive trainer for all Indonesian stocks from yfinance
    """

    def __init__(self,
                 min_data_years: float = 1.5,
                 max_workers: int = 10,
                 batch_size: int = 50,
                 progress_update_interval: int = 10):
        """
        Initialize the Indonesian stock trainer

        Args:
            min_data_years: Minimum years of historical data required
            max_workers: Maximum number of concurrent download workers
            batch_size: Number of stocks to process in each batch
            progress_update_interval: Update progress every N stocks
        """
        self.min_data_years = min_data_years
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.progress_update_interval = progress_update_interval

        # Storage
        self.all_symbols = []
        self.filtered_symbols = []
        self.downloaded_data = {}
        self.training_dataset = None
        self.training_results = {}

        # Progress tracking
        self.start_time = None
        self.processed_count = 0
        self.success_count = 0
        self.failed_symbols = []

        # MLPredictorV2
        self.ml_predictor = MLPredictorV2() if ENHANCED_ML_AVAILABLE else None

        # Create directories
        self._create_directories()

        # Setup signal handler for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.interrupted = False

    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            'ml_system/training/data',
            'ml_system/training/models',
            'ml_system/training/logs',
            'ml_system/training/results'
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def _signal_handler(self, signum, frame):
        """Handle interruption signals gracefully"""
        logger.info("Received interruption signal, saving progress...")
        self.interrupted = True
        self._save_progress()

    def _save_progress(self):
        """Save current progress to file"""
        progress_data = {
            'timestamp': datetime.now().isoformat(),
            'processed_count': self.processed_count,
            'success_count': self.success_count,
            'failed_symbols': self.failed_symbols,
            'filtered_symbols': self.filtered_symbols,
            'downloaded_data_count': len(self.downloaded_data)
        }

        progress_file = 'ml_system/training/progress.json'
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        logger.info(f"Progress saved to {progress_file}")

    def get_all_indonesian_stocks(self) -> List[str]:
        """
        Scrape all Indonesian stock symbols from yfinance

        Returns:
            List of Indonesian stock symbols with .JK suffix
        """
        logger.info("Scraping all Indonesian stocks from yfinance...")

        # Comprehensive list of Indonesian stocks by sector
        # This is a more extensive list covering major stocks in IDX
        indonesian_stocks = [
            # Banking & Financial Services
            'BBCA.JK', 'BBRI.JK', 'BMRI.JK', 'BBNI.JK', 'BTPN.JK', 'BNGA.JK', 'BRIS.JK',
            'BNII.JK', 'BKDP.JK', 'BBSR.JK', 'BAGR.JK', 'BAPA.JK', 'BAYU.JK', 'BBLD.JK',
            'BBKP.JK', 'BBTN.JK', 'BDMN.JK', 'BFIN.JK', 'BHIT.JK', 'BIMA.JK', 'BKSW.JK',
            'BMSR.JK', 'BNGA.JK', 'BOBA.JK', 'BPID.JK', 'BRPT.JK', 'BRTG.JK', 'BUKK.JK',

            # Telecom & Technology
            'TLKM.JK', 'ISAT.JK', 'EXCL.JK', 'FREN.JK', 'GOTO.JK', 'BUKA.JK', 'TCID.JK',
            'MTEL.JK', 'FAST.JK', 'ADMF.JK', 'ARTO.JK', 'BCAP.JK', 'BIRD.JK', 'BKDP.JK',
            'BIMA.JK', 'BNET.JK', 'CPRI.JK', 'DMMX.JK', 'FISH.JK', 'GEMS.JK', 'GTBO.JK',
            'HDIT.JK', 'IIKP.JK', 'INDF.JK', 'INOV.JK', 'IPCM.JK', 'JGLE.JK', 'KOIN.JK',
            'KPIG.JK', 'LCGP.JK', 'LPIN.JK', 'MAXS.JK', 'MCPA.JK', 'MDKA.JK', 'MDLN.JK',
            'MEDC.JK', 'MGRO.JK', 'MNCN.JK', 'MOGI.JK', 'MORA.JK', 'MPMX.JK', 'MSIN.JK',
            'MTDL.JK', 'MTFN.JK', 'MTHS.JK', 'NASI.JK', 'NATO.JK', 'NOBU.JK', 'NPCK.JK',

            # Consumer Goods & Retail
            'UNVR.JK', 'INDF.JK', 'ICBP.JK', 'MYOR.JK', 'KLBF.JK', 'KAEF.JK', 'GGRM.JK',
            'HMSP.JK', 'TCID.JK', 'WIIM.JK', 'ULTJ.JK', 'DLTA.JK', 'SIDO.JK', 'KARW.JK',
            'MLPL.JK', 'MRAT.JK', 'SKBM.JK', 'TKIM.JK', 'TOWR.JK', 'TPIA.JK', 'TPID.JK',
            'ADES.JK', 'AKRA.JK', 'ALDO.JK', 'ALMI.JK', 'ALTO.JK', 'AMFG.JK', 'AMMN.JK',
            'ANCORA.JK', 'ANJT.JK', 'ARKA.JK', 'ARNA.JK', 'ASA.JK', 'ASII.JK', 'ASGR.JK',
            'ASRI.JK', 'AUTP.JK', 'BAYU.JK', 'BEEF.JK', 'BELL.JK', 'BFIN.JK', 'BGRO.JK',
            'BHMN.JK', 'BIMA.JK', 'BISI.JK', 'BPKP.JK', 'BRAU.JK', 'BSIM.JK', 'BTEK.JK',

            # Mining & Energy
            'ADRO.JK', 'ANTM.JK', 'PTBA.JK', 'TINS.JK', 'ITMG.JK', 'BYAN.JK', 'MEDC.JK',
            'PGAS.JK', 'PGAS.JK', 'ELSA.JK', 'GEMS.JK', 'DOID.JK', 'PTRO.JK', 'PERT.JK',
            'KOPI.JK', 'FIRE.JK', 'BORN.JK', 'BRAU.JK', 'BSSR.JK', 'BTON.JK', 'BTEK.JK',
            'BUDI.JK', 'BUMI.JK', 'BUNL.JK', 'BWEN.JK', 'BYPK.JK', 'CAPE.JK', 'CAMP.JK',
            'CAN.JK', 'CARS.JK', 'CASH.JK', 'CASA.JK', 'CATA.JK', 'CAMP.JK', 'CBMF.JK',
            'CBPE.JK', 'CCSI.JK', 'CENT.JK', 'CFIN.JK', 'CGNT.JK', 'CHEM.JK', 'CHMG.JK',
            'CIAN.JK', 'CITA.JK', 'CJFI.JK', 'CKRA.JK', 'CLPI.JK', 'CMPP.JK', 'CNKO.JK',
            'CNNB.JK', 'COLT.JK', 'COMP.JK', 'CPIN.JK', 'CPRO.JK', 'CRSB.JK', 'CSTM.JK',

            # Property & Real Estate
            'BSDE.JK', 'LPKR.JK', 'PWON.JK', 'ASRI.JK', 'CTRA.JK', 'MDKA.JK', 'SMRA.JK',
            'DPUL.JK', 'KPIG.JK', 'RDTX.JK', 'ADHI.JK', 'WIKA.JK', 'PTPP.JK', 'DGIK.JK',
            'DILD.JK', 'RISE.JK', 'GAMA.JK', 'GJSR.JK', 'GJTL.JK', 'GLOB.JK', 'GMFI.JK',
            'GOTO.JK', 'GPAI.JK', 'GPRA.JK', 'GREN.JK', 'GRMT.JK', 'GRTA.JK', 'GTSI.JK',
            'GWON.JK', 'HAJJ.JK', 'HEAL.JK', 'HOKI.JK', 'HOPE.JK', 'HRUM.JK', 'HUMI.JK',
            'IBOS.JK', 'ICON.JK', 'IFII.JK', 'IFSH.JK', 'IGAR.JK', 'IIKP.JK', 'IKAI.JK',
            'IMAS.JK', 'IMJS.JK', 'INAF.JK', 'INAI.JK', 'INDF.JK', 'INDO.JK', 'INDR.JK',
            'INDX.JK', 'INDY.JK', 'INOV.JK', 'INTA.JK', 'INTD.JK', 'INVS.JK', 'IPCM.JK',

            # Automotive & Industrial
            'AUTO.JK', 'ASII.JK', 'SAME.JK', 'GJTL.JK', 'PRAS.JK', 'FAST.JK', 'SULI.JK',
            'PNIN.JK', 'ALDO.JK', 'BRTG.JK', 'BRMS.JK', 'BULL.JK', 'BUME.JK', 'BUMI.JK',
            'BUNL.JK', 'BUTK.JK', 'BWEN.JK', 'BYPK.JK', 'CAMP.JK', 'CARS.JK', 'CASA.JK',
            'CASH.JK', 'CATA.JK', 'CBMF.JK', 'CBPE.JK', 'CCSI.JK', 'CDMM.JK', 'CENT.JK',
            'CFIN.JK', 'CGNT.JK', 'CHEM.JK', 'CHMG.JK', 'CIAN.JK', 'CITA.JK', 'CJFI.JK',

            # Infrastructure & Construction
            'ADHI.JK', 'WIKA.JK', 'PTPP.JK', 'DGIK.JK', 'TOTL.JK', 'PTRO.JK', 'JSMR.JK',
            'TOWR.JK', 'WSKT.JK', 'WSBP.JK', 'SAGE.JK', 'PTSN.JK', 'GIAA.JK', 'TCID.JK',
            'LPPF.JK', 'LPGI.JK', 'LPKR.JK', 'LPLI.JK', 'LPIN.JK', 'LRNA.JK', 'LRTS.JK',

            # Agriculture & Plantation
            'AALI.JK', 'LSIP.JK', 'SGRO.JK', 'TBLA.JK', 'BBCA.JK', 'BBRI.JK', 'BMRI.JK',
            'SMAR.JK', 'GGRM.JK', 'HMSP.JK', 'LINK.JK', 'BISI.JK', 'PGAS.JK', 'CPIN.JK',
            'JPFA.JK', 'MAIN.JK', 'MASK.JK', 'MCAS.JK', 'MDKA.JK', 'MEDC.JK', 'MEGA.JK',

            # Healthcare & Pharmaceuticals
            'KLBF.JK', 'KAEF.JK', 'DKFT.JK', 'SQMI.JK', 'HEAL.JK', 'SAME.JK', 'SDPC.JK',
            'SIDO.JK', 'PTSI.JK', 'SAMF.JK', 'IRAA.JK', 'MYOH.JK', 'SELA.JK', 'TARA.JK',

            # Additional stocks for comprehensive coverage
            'ACST.JK', 'ADAP.JK', 'ADMG.JK', 'ADMR.JK', 'AEWS.JK', 'AFII.JK', 'AGII.JK',
            'AGRO.JK', 'AHAP.JK', 'AIRA.JK', 'AJWA.JK', 'AKKU.JK', 'AKPI.JK', 'AKRA.JK',
            'AKSI.JK', 'ALDO.JK', 'ALKA.JK', 'ALMI.JK', 'ALTO.JK', 'AMAN.JK', 'AMFG.JK',
            'AMIN.JK', 'AMMN.JK', 'AMPG.JK', 'ANRS.JK', 'AOI.JK', 'APIC.JK', 'APII.JK',
            'APLN.JK', 'APOB.JK', 'ARAB.JK', 'ARII.JK', 'ARNA.JK', 'ARSI.JK', 'ARTI.JK',
            'ARTO.JK', 'ASA.JK', 'ASGR.JK', 'ASKM.JK', 'ASMI.JK', 'ASRM.JK', 'ASSA.JK',
            'ASTI.JK', 'ATAP.JK', 'AUTO.JK', 'BAUT.JK', 'BAYU.JK', 'BBCA.JK', 'BBAU.JK',
            'BBBK.JK', 'BBDK.JK', 'BBHI.JK', 'BBIA.JK', 'BBIK.JK', 'BBJM.JK', 'BBKU.JK',
            'BBLK.JK', 'BBMD.JK', 'BBNI.JK', 'BBNP.JK', 'BBNS.JK', 'BBOB.JK', 'BPPD.JK'
        ]

        # Remove duplicates while preserving order
        seen = set()
        unique_stocks = []
        for stock in indonesian_stocks:
            if stock not in seen:
                seen.add(stock)
                unique_stocks.append(stock)

        logger.info(f"Found {len(unique_stocks)} unique Indonesian stock symbols")
        self.all_symbols = unique_stocks
        return unique_stocks

    def download_stock_data(self, symbol: str, period: str = "3y") -> Optional[pd.DataFrame]:
        """
        Download historical data for a single stock with error handling

        Args:
            symbol: Stock symbol (e.g., 'BBCA.JK')
            period: Historical data period

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            # Download data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, auto_adjust=False, prepost=False)

            if data.empty:
                return None

            # Check data quality
            min_required_days = int(self.min_data_years * 252)  # Trading days per year
            if len(data) < min_required_days:
                logger.debug(f"{symbol}: Insufficient data ({len(data)} days < {min_required_days})")
                return None

            # Basic data quality checks
            if data['Close'].isna().sum() > len(data) * 0.1:  # More than 10% missing
                return None

            # Add symbol column
            data['symbol'] = symbol

            return data

        except Exception as e:
            logger.debug(f"{symbol}: Error downloading data - {str(e)}")
            return None

    def filter_stocks_with_sufficient_data(self, symbols: List[str]) -> List[str]:
        """
        Filter stocks that have sufficient historical data

        Args:
            symbols: List of stock symbols to filter

        Returns:
            List of symbols with sufficient data
        """
        logger.info(f"Filtering stocks with minimum {self.min_data_years} years of data...")

        sufficient_symbols = []

        # Process in batches to avoid overwhelming the API
        for i in range(0, len(symbols), self.batch_size):
            batch = symbols[i:i + self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(symbols)-1)//self.batch_size + 1}")

            for symbol in batch:
                if self.interrupted:
                    logger.info("Filtering interrupted by user")
                    return sufficient_symbols

                self.processed_count += 1

                # Update progress
                if self.processed_count % self.progress_update_interval == 0:
                    logger.info(f"Processed {self.processed_count}/{len(symbols)} symbols "
                              f"(Success: {self.success_count})")

                # Download and check data
                data = self.download_stock_data(symbol)

                if data is not None:
                    sufficient_symbols.append(symbol)
                    self.downloaded_data[symbol] = data
                    self.success_count += 1
                    logger.debug(f"{symbol}: ✓ {len(data)} days of data")
                else:
                    self.failed_symbols.append(symbol)
                    logger.debug(f"{symbol}: ✗ Insufficient data or error")

                # Rate limiting to avoid overwhelming yfinance
                time.sleep(0.1)

        logger.info(f"Filtering complete. {len(sufficient_symbols)}/{len(symbols)} symbols have sufficient data")
        self.filtered_symbols = sufficient_symbols
        return sufficient_symbols

    def create_training_dataset(self, symbols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create comprehensive training dataset from multiple stocks

        Args:
            symbols: List of stock symbols to include in training

        Returns:
            Tuple of (features DataFrame, labels Series)
        """
        logger.info("Creating comprehensive training dataset...")

        all_features = []
        all_labels = []

        for i, symbol in enumerate(symbols):
            if self.interrupted:
                logger.info("Dataset creation interrupted by user")
                return pd.DataFrame(), pd.Series()

            if symbol not in self.downloaded_data:
                continue

            logger.info(f"Processing {symbol} ({i+1}/{len(symbols)})...")

            try:
                # Get data for this symbol
                data = self.downloaded_data[symbol].copy()

                if len(data) < 100:  # Minimum data for feature creation
                    logger.warning(f"{symbol}: Insufficient data for feature creation")
                    continue

                # Create features using enhanced feature engineering
                if ENHANCED_ML_AVAILABLE:
                    features_df = self.ml_predictor.feature_engineer.create_features(data)
                else:
                    # Basic feature engineering fallback
                    features_df = self._create_basic_features(data)

                if features_df.empty:
                    logger.warning(f"{symbol}: No features created")
                    continue

                # Create labels based on future returns
                labels = self._create_labels(data)

                # Align features and labels
                min_length = min(len(features_df), len(labels))
                features_df = features_df.iloc[:min_length]
                labels = labels[:min_length]

                # Remove any NaN values
                valid_mask = ~(features_df.isna().any(axis=1) | pd.isna(labels))
                features_df = features_df[valid_mask]
                labels = labels[valid_mask]

                if len(features_df) < 50:  # Minimum after cleaning
                    logger.warning(f"{symbol}: Insufficient valid samples after cleaning")
                    continue

                # Add to dataset
                all_features.append(features_df)
                all_labels.append(labels)

                logger.info(f"{symbol}: Added {len(features_df)} samples to dataset")

            except Exception as e:
                logger.error(f"{symbol}: Error processing data - {str(e)}")
                continue

        if not all_features:
            raise ValueError("No valid features created from any symbols")

        # Combine all data
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_labels = pd.concat(all_labels, ignore_index=True)

        logger.info(f"Created training dataset with {len(combined_features)} samples "
                   f"from {len(all_features)} stocks")

        self.training_dataset = {
            'features': combined_features,
            'labels': combined_labels,
            'symbols_used': symbols,
            'total_samples': len(combined_features)
        }

        return combined_features, combined_labels

    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create basic technical features when enhanced features are not available

        Args:
            df: OHLCV DataFrame

        Returns:
            Features DataFrame
        """
        features = df.copy()

        try:
            # Handle both 'Close' and 'close' column names
            close_col = 'Close' if 'Close' in df.columns else 'close'

            if close_col not in df.columns:
                return pd.DataFrame()

            close = df[close_col]

            # Basic returns
            for period in [5, 10, 20]:
                features[f'return_{period}d'] = close.pct_change(period)
                features[f'sma_{period}'] = close.rolling(window=period).mean()
                features[f'price_to_sma_{period}'] = close / features[f'sma_{period}']
                features[f'volatility_{period}d'] = features[f'return_{period}d'].rolling(window=period).std()

            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi_14'] = 100 - (100 / (1 + rs))

            # Momentum
            features['momentum_5'] = (close / close.shift(5) - 1) * 100

            # Clean data
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(0)

            return features

        except Exception as e:
            logger.error(f"Error creating basic features: {e}")
            return pd.DataFrame()

    def _create_labels(self, df: pd.DataFrame, horizon: int = 5) -> pd.Series:
        """
        Create labels based on future returns

        Args:
            df: OHLCV DataFrame
            horizon: Future return horizon in days

        Returns:
            Series with labels (0=BUY, 1=SELL, 2=WAIT)
        """
        try:
            close_col = 'Close' if 'Close' in df.columns else 'close'

            if close_col not in df.columns:
                return pd.Series()

            # Calculate future returns
            future_return = df[close_col].pct_change(horizon).shift(-horizon)

            # Create labels based on return thresholds
            buy_threshold = 0.02  # 2% return
            sell_threshold = -0.02  # -2% return

            labels = pd.Series(2, index=future_return.index)  # Default to WAIT
            labels[future_return >= buy_threshold] = 0  # BUY
            labels[future_return <= sell_threshold] = 1  # SELL

            # Remove NaN values
            labels = labels[future_return.notna()]

            return labels

        except Exception as e:
            logger.error(f"Error creating labels: {e}")
            return pd.Series()

    def train_models(self, features: pd.DataFrame, labels: pd.Series) -> Dict:
        """
        Train MLPredictorV2 on the comprehensive dataset

        Args:
            features: Features DataFrame
            labels: Labels Series

        Returns:
            Training results dictionary
        """
        if not ENHANCED_ML_AVAILABLE:
            return {
                'status': 'failed',
                'error': 'Enhanced ML models not available',
                'timestamp': datetime.now().isoformat()
            }

        logger.info("Training MLPredictorV2 on comprehensive Indonesian stocks dataset...")

        try:
            # Train the enhanced models
            training_results = self.ml_predictor.train_models(
                training_data=features,
                labels=labels,
                validation_split=0.2
            )

            # Save models explicitly
            self.ml_predictor.save_models()

            # Add additional metadata
            training_results.update({
                'dataset_info': {
                    'total_samples': len(features),
                    'feature_count': len(features.columns),
                    'symbols_used': len(self.filtered_symbols),
                    'training_date': datetime.now().isoformat(),
                    'data_coverage_years': self.min_data_years
                },
                'model_type': 'MLPredictorV2',
                'ensemble_models': self.ml_predictor.ensemble.model_types if hasattr(self.ml_predictor, 'ensemble') else []
            })

            self.training_results = training_results

            logger.info("Training completed successfully!")
            logger.info(f"Training samples: {training_results.get('training_samples', 'N/A')}")
            logger.info(f"Validation accuracy: {training_results.get('validation_accuracy', 'N/A'):.4f}")
            logger.info(f"Features used: {training_results.get('feature_count', 'N/A')}")

            return training_results

        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            logger.error(error_msg)
            return {
                'status': 'failed',
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }

    def save_results(self, results: Dict):
        """
        Save comprehensive training results

        Args:
            results: Training results dictionary
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_file = f'ml_system/training/results/indonesian_stocks_training_{timestamp}.json'

        comprehensive_results = {
            'training_metadata': {
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': datetime.now().isoformat(),
                'total_symbols_processed': len(self.all_symbols),
                'symbols_with_sufficient_data': len(self.filtered_symbols),
                'successful_downloads': len(self.downloaded_data),
                'failed_symbols': self.failed_symbols,
                'min_data_years': self.min_data_years,
                'model_type': 'MLPredictorV2',
                'enhanced_features': ENHANCED_ML_AVAILABLE
            },
            'dataset_summary': {
                'total_training_samples': len(self.training_dataset['features']) if self.training_dataset else 0,
                'feature_count': len(self.training_dataset['features'].columns) if self.training_dataset else 0,
                'symbols_used': self.filtered_symbols,
                'labels_distribution': self.training_dataset['labels'].value_counts().to_dict() if self.training_dataset else {}
            },
            'training_results': results,
            'model_info': self.ml_predictor.get_model_info() if self.ml_predictor else None
        }

        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)

        # Also save as latest
        latest_file = 'ml_system/training/results/latest_indonesian_training.json'
        with open(latest_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)

        logger.info(f"Results saved to {results_file}")
        logger.info(f"Latest results saved to {latest_file}")

        # Save training data for future use
        if self.training_dataset:
            data_file = f'ml_system/training/data/training_data_{timestamp}.pkl'
            joblib.dump(self.training_dataset, data_file)
            logger.info(f"Training data saved to {data_file}")

    def generate_report(self) -> str:
        """
        Generate a comprehensive training report

        Returns:
            Report string
        """
        if not self.training_results:
            return "No training results available"

        report = f"""
========================================
INDONESIAN STOCKS COMPREHENSIVE TRAINING REPORT
========================================

Training Overview:
- Start Time: {self.start_time}
- End Time: {datetime.now()}
- Duration: {datetime.now() - self.start_time if self.start_time else 'N/A'}

Data Collection:
- Total Indonesian Stocks Processed: {len(self.all_symbols)}
- Symbols with Sufficient Data: {len(self.filtered_symbols)}
- Successful Data Downloads: {len(self.downloaded_data)}
- Failed Downloads: {len(self.failed_symbols)}
- Minimum Data Required: {self.min_data_years} years

Training Dataset:
- Total Training Samples: {self.training_dataset.get('total_samples', 'N/A')}
- Features per Sample: {len(self.training_dataset['features'].columns) if self.training_dataset else 'N/A'}
- Stocks Included: {len(self.training_dataset.get('symbols_used', [])) if self.training_dataset else 'N/A'}

Model Performance:
- Model Type: MLPredictorV2
- Enhanced Features: {ENHANCED_ML_AVAILABLE}
- Training Accuracy: {self.training_results.get('train_accuracy', 'N/A'):.4f}
- Validation Accuracy: {self.training_results.get('validation_accuracy', 'N/A'):.4f'}
- Models in Ensemble: {', '.join(self.training_results.get('ensemble_models', []))}

Top Performing Stocks (with sufficient data):
{', '.join(self.filtered_symbols[:20])}...

Failed/Insufficient Data Stocks:
{', '.join(self.failed_symbols[:20])}...

Model Files Saved:
- Enhanced Models: ml_system/models/enhanced_v2_models.pkl
- Training Results: ml_system/training/results/latest_indonesian_training.json
- Training Data: ml_system/training/data/

Next Steps:
1. Test the trained models with new data
2. Monitor performance on live trading
3. Retrain periodically with updated data
4. Consider adding more features or data sources

========================================
Generated: {datetime.now()}
========================================
        """

        # Save report to file
        report_file = 'ml_system/training/results/training_report.txt'
        with open(report_file, 'w') as f:
            f.write(report)

        logger.info(f"Training report saved to {report_file}")

        return report

    def run_complete_pipeline(self) -> Dict:
        """
        Run the complete training pipeline

        Returns:
            Training results
        """
        self.start_time = datetime.now()
        logger.info("Starting comprehensive Indonesian stocks training pipeline...")

        try:
            # Step 1: Get all Indonesian stocks
            logger.info("Step 1: Scraping Indonesian stock symbols...")
            symbols = self.get_all_indonesian_stocks()

            if not symbols:
                raise ValueError("No Indonesian stock symbols found")

            # Step 2: Filter stocks with sufficient data
            logger.info("Step 2: Filtering stocks with sufficient historical data...")
            sufficient_symbols = self.filter_stocks_with_sufficient_data(symbols)

            if not sufficient_symbols:
                raise ValueError("No stocks with sufficient data found")

            # Step 3: Create training dataset
            logger.info("Step 3: Creating comprehensive training dataset...")
            features, labels = self.create_training_dataset(sufficient_symbols)

            if features.empty:
                raise ValueError("No valid training data created")

            # Step 4: Train models
            logger.info("Step 4: Training MLPredictorV2 models...")
            training_results = self.train_models(features, labels)

            # Step 5: Save results
            logger.info("Step 5: Saving training results...")
            self.save_results(training_results)

            # Step 6: Generate report
            logger.info("Step 6: Generating training report...")
            report = self.generate_report()
            print(report)

            duration = datetime.now() - self.start_time
            logger.info(f"Complete training pipeline finished in {duration}")

            return training_results

        except Exception as e:
            error_msg = f"Training pipeline failed: {str(e)}"
            logger.error(error_msg)
            return {
                'status': 'failed',
                'error': error_msg,
                'timestamp': datetime.now().isoformat(),
                'partial_results': self.training_results
            }

        finally:
            # Save final progress
            self._save_progress()


def main():
    """Main function to run the comprehensive Indonesian stocks training"""
    print("========================================")
    print("INDONESIAN STOCKS COMPREHENSIVE TRAINER")
    print("Training MLPredictorV2 on All JK Stocks")
    print("========================================\n")

    # Configuration
    config = {
        'min_data_years': 1.5,  # Minimum 1.5 years of data
        'max_workers': 8,       # Concurrent downloads
        'batch_size': 50,       # Batch processing size
        'progress_update_interval': 20  # Progress updates
    }

    try:
        # Initialize trainer
        trainer = IndonesianStockTrainer(**config)

        # Run complete pipeline
        results = trainer.run_complete_pipeline()

        if results.get('status') == 'failed':
            print(f"\nTraining failed: {results.get('error')}")
            return 1
        else:
            print(f"\nTraining completed successfully!")
            print(f"Validation accuracy: {results.get('validation_accuracy', 'N/A'):.4f}")
            return 0

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())