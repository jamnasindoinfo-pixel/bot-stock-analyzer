"""
IDX Data Loader using IDX-Scrapper
Load historical data from Indonesian Stock Exchange (IDX) for ML training
"""

import os
import sys
import json
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IDXDataLoader:
    """Load and manage IDX stock data for ML training"""

    def __init__(self, data_dir: str = None):
        """
        Initialize IDX data loader

        Args:
            data_dir: Directory to store scraped data
        """
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), 'idx_data')

        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # Load IDX stock listings
        self.idx_stocks = self._load_idx_listings()

    def _load_idx_listings(self) -> Dict[str, str]:
        """
        Load IDX stock listings with sectors
        Returns dict of {symbol: name}
        """
        # Common IDX stocks by sector
        idx_stocks = {
            # Banking
            'BBCA': 'Bank Central Asia',
            'BBRI': 'Bank Rakyat Indonesia',
            'BBNI': 'Bank Negara Indonesia',
            'BMRI': 'Bank Mandiri',
            'BRIS': 'Bank Syariah Indonesia',
            'BTPN': 'Bank Tabungan Pensiunan Nasional',
            'MNCN': 'Bank MNC Internasional',

            # Consumer Goods
            'UNVR': 'Unilever Indonesia',
            'INDF': 'Indofood Sukses Makmur',
            'ICBP': 'Indofood CBP Sukses Makmur',
            'KLBF': 'Kalbe Farma',
            'ULTJ': 'Ultra Jaya Milk',
            'MYOR': 'Mayora Indah',
            'TKIM': 'Pabrik Kertas Tjiandi Kimia',

            # Telecommunication
            'TLKM': 'Telkom Indonesia',
            'EXCL': 'XL Axiata',
            'ISAT': 'Indosat Ooredoo',
            'FREN': 'Smartfren Telecom',

            # Energy & Mining
            'ANTM': 'Aneka Tambang',
            'PTBA': 'Tambang Batubara Bukit Asam',
            'ADRO': 'Adaro Energy',
            'TINS': 'Timah',
            'PGAS': 'Perusahaan Gas Negara',
            'MEDC': 'Medco Energi',
            'Elnusa': 'ELSA',

            # Property & Construction
            'ASII': 'Astra International',
            'AUTO': 'Astra Otoparts',
            'GJTL': 'Gajah Tunggal',
            'TPIA': 'Chandra Asri',
            'SMGR': 'Semen Indonesia',
            'INTP': 'Indocement Tunggal Prakasa',
            'WIKA': 'Waskita Karya',
            'ADHI': 'Adhi Karya',
            'PTPP': 'PP (Persero)',
            'PWON': 'Pakuwon Jati',
            'CTRA': 'Ciputra Development',
            'BSDE': 'Bumi Serpong Damai',

            # Retail
            'LPPF': 'Matahari Department Store',
            'MAP': 'Mitra Adiperkasa',
            'ACES': 'Ace Hardware Indonesia',
            'RANC': 'Ramayana Lestari Sentosa',
            'KIPA': 'Kipindo',

            # Agriculture & Plantation
            'JPFA': 'Japfa Comfeed',
            'CPIN': 'Charoen Pokphand',
            'SIDO': 'Sido Mulyo',
            'KLBF': 'Kalbe Farma',

            # Infrastructure
            'JSMR': 'Jasa Marga',
            'TOWR': 'Sarana Menara Nusantara',
            'PGAS': 'Perusahaan Gas Negara',

            # Finance
            'BBRI': 'Bank Rakyat Indonesia',
            'BBCA': 'Bank Central Asia',
            'BBNI': 'Bank Negara Indonesia',
            'BMRI': 'Bank Mandiri',
            'AGRO': 'Bank Raya',

            # Healthcare
            'KLBF': 'Kalbe Farma',
            'PRDA': 'Pyridam Farma',
            'KAEF': 'Kimia Farma',

            # Technology
            'GOTO': 'GoTo Gojek Tokopedia',
            'BUKK': 'Bukalapak',
            'ARSA': 'MNC Digital Entertainment',
        }

        # Add .JK suffix for Yahoo Finance compatibility
        return {k + '.JK': v for k, v in idx_stocks.items()}

    def get_all_stocks(self) -> List[str]:
        """Get all available IDX stock symbols"""
        return list(self.idx_stocks.keys())

    def get_stocks_by_sector(self, sectors: List[str] = None) -> List[str]:
        """
        Get stocks by sector

        Args:
            sectors: List of sectors to include. If None, returns all

        Returns:
            List of stock symbols
        """
        if sectors is None:
            return self.get_all_stocks()

        # Sector mapping would need to be implemented
        return self.get_all_stocks()

    def load_historical_data(self, symbol: str, period_years: int = 5) -> Optional[pd.DataFrame]:
        """
        Load historical data for a symbol

        Args:
            symbol: Stock symbol (with .JK suffix)
            period_years: Number of years of historical data

        Returns:
            DataFrame with OHLCV data or None if not found
        """
        # Try to load from cached file first
        cache_file = os.path.join(self.data_dir, f"{symbol.replace('.JK', '')}.csv")

        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                logger.info(f"Loaded {len(df)} days of data for {symbol} from cache")

                # Filter by period
                cutoff_date = datetime.now() - timedelta(days=period_years * 365)
                df = df[df.index >= cutoff_date]

                return df
            except Exception as e:
                logger.error(f"Error loading cached data for {symbol}: {str(e)}")

        # If not in cache, try to scrape
        try:
            # Import IDX scraper
            from idx_scrapper import IdxScrapper

            # Initialize scraper
            scraper = IdxScrapper()

            # Get stock data (convert symbol to IDX format)
            idx_symbol = symbol.replace('.JK', '')
            logger.info(f"Scraping data for {idx_symbol}...")

            # Get historical data
            data = scraper.get_historical_data(idx_symbol)

            if data and len(data) > 0:
                # Convert to DataFrame
                df = pd.DataFrame(data)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

                # Rename columns to standard format
                df.rename(columns={
                    'Open': 'Open',
                    'High': 'High',
                    'Low': 'Low',
                    'Close': 'Close',
                    'Volume': 'Volume'
                }, inplace=True)

                # Save to cache
                df.to_csv(cache_file)
                logger.info(f"Scraped and saved {len(df)} days of data for {symbol}")

                # Filter by period
                cutoff_date = datetime.now() - timedelta(days=period_years * 365)
                df = df[df.index >= cutoff_date]

                return df
            else:
                logger.warning(f"No data found for {symbol}")
                return None

        except ImportError:
            logger.error("IDX-Scrapper not installed. Install with: pip install idx-scrapper")
            return None
        except Exception as e:
            logger.error(f"Error scraping data for {symbol}: {str(e)}")
            return None

    def load_multiple_stocks(self, symbols: List[str], period_years: int = 3,
                           min_days: int = 200, max_stocks: int = None) -> pd.DataFrame:
        """
        Load data for multiple stocks and combine them

        Args:
            symbols: List of stock symbols
            period_years: Years of historical data
            min_days: Minimum days of data required
            max_stocks: Maximum number of stocks to process

        Returns:
            Combined DataFrame with all stocks
        """
        if max_stocks:
            symbols = symbols[:max_stocks]

        all_data = []

        logger.info(f"Loading data for {len(symbols)} stocks...")

        for i, symbol in enumerate(symbols):
            try:
                logger.info(f"  [{i+1}/{len(symbols)}] Loading {symbol}...")

                df = self.load_historical_data(symbol, period_years)

                if df is not None and len(df) >= min_days:
                    # Add symbol column
                    df = df.copy()
                    df['Symbol'] = symbol

                    # Add stock name
                    if symbol in self.idx_stocks:
                        df['StockName'] = self.idx_stocks[symbol]

                    all_data.append(df)
                    logger.info(f"    Loaded {len(df)} days")
                else:
                    logger.warning(f"    Insufficient data for {symbol}")

            except Exception as e:
                logger.error(f"    Error loading {symbol}: {str(e)}")
                continue

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Combined data: {len(all_data)} stocks, {len(combined_df)} total rows")
            return combined_df
        else:
            logger.error("No data loaded")
            return pd.DataFrame()

    def download_all_idx_data(self, period_years: int = 3):
        """
        Download all available IDX data

        Args:
            period_years: Years of historical data to download
        """
        symbols = self.get_all_stocks()
        logger.info(f"Starting download of all IDX data ({len(symbols)} stocks)")

        success_count = 0
        fail_count = 0

        for i, symbol in enumerate(symbols):
            try:
                logger.info(f"[{i+1}/{len(symbols)}] Downloading {symbol}...")

                df = self.load_historical_data(symbol, period_years)

                if df is not None and len(df) > 0:
                    success_count += 1
                    logger.info(f"  ✓ Success: {len(df)} days")
                else:
                    fail_count += 1
                    logger.warning(f"  ✗ Failed")

            except Exception as e:
                fail_count += 1
                logger.error(f"  ✗ Error: {str(e)}")

        logger.info(f"\nDownload complete:")
        logger.info(f"  Success: {success_count} stocks")
        logger.info(f"  Failed: {fail_count} stocks")
        logger.info(f"  Data saved to: {self.data_dir}")

    def get_market_indices(self) -> Dict[str, pd.DataFrame]:
        """
        Load major IDX market indices

        Returns:
            Dictionary of {index_name: DataFrame}
        """
        indices = {
            'IHSG': 'Indonesia Composite Index',
            'JII': 'Jakarta Islamic Index',
            'LQ45': 'LQ45 Index',
            'KOMPAS100': 'Kompas 100 Index'
        }

        # Would need to implement index data fetching
        # This is a placeholder
        logger.info("Market indices data not yet implemented")
        return {}

    def save_data_summary(self):
        """Save a summary of all downloaded data"""
        summary = {
            'total_stocks': len(self.idx_stocks),
            'download_date': datetime.now().isoformat(),
            'stocks': {}
        }

        for symbol in self.idx_stocks:
            file_path = os.path.join(self.data_dir, f"{symbol.replace('.JK', '')}.csv")
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    summary['stocks'][symbol] = {
                        'name': self.idx_stocks[symbol],
                        'data_points': len(df),
                        'date_range': {
                            'start': df.index[0].strftime('%Y-%m-%d'),
                            'end': df.index[-1].strftime('%Y-%m-%d')
                        }
                    }
                except:
                    summary['stocks'][symbol] = {'error': 'Failed to read'}

        # Save summary
        summary_file = os.path.join(self.data_dir, 'data_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Data summary saved to: {summary_file}")


def main():
    """Example usage of IDX Data Loader"""
    print("="*60)
    print("IDX DATA LOADER")
    print("="*60)

    # Initialize loader
    loader = IDXDataLoader()

    # Download all data (commented out by default)
    # print("\n[*] Downloading all IDX data...")
    # loader.download_all_idx_data(period_years=3)

    # Load specific stocks
    print("\n[*] Loading top IDX stocks...")
    top_stocks = [
        'BBCA.JK', 'BBRI.JK', 'BBNI.JK', 'BMRI.JK',  # Banks
        'UNVR.JK', 'INDF.JK', 'ICBP.JK',             # Consumer
        'TLKM.JK', 'EXCL.JK',                       # Telecom
        'ANTM.JK', 'PTBA.JK', 'TINS.JK',            # Mining
        'ASII.JK', 'AUTO.JK',                       # Automotive
    ]

    df = loader.load_multiple_stocks(top_stocks, period_years=3, min_days=200)

    if not df.empty:
        print(f"\n✓ Successfully loaded data:")
        print(f"  Total rows: {len(df):,}")
        print(f"  Unique stocks: {df['Symbol'].nunique()}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")

        # Save data summary
        loader.save_data_summary()
    else:
        print("\n✗ No data loaded")


if __name__ == "__main__":
    main()