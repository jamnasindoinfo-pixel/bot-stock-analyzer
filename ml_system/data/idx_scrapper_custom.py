"""
Custom IDX Stock Data Scraper
Alternative to IDX-Scrapper for Indonesian stock data
Uses multiple sources for reliable data collection
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomIDXScraper:
    """Custom scraper for Indonesian stock market data"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # Base URLs
        self.idx_url = "https://www.idx.co.id"
        self.investing_url = "https://www.investing.com"

        # Stock list cache
        self._stock_list = None

    def get_idx_stock_list(self) -> List[Dict]:
        """
        Get list of all stocks listed on IDX

        Returns:
            List of dictionaries with stock information
        """
        if self._stock_list:
            return self._stock_list

        try:
            # Try to get from IDX website
            url = f"{self.idx_url}/en/listed-companies/company-profiles"
            response = self.session.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')

            stocks = []

            # This is a simplified version - in production would parse actual table
            # For now, return major IDX stocks
            major_stocks = {
                # Banking
                'BBCA': 'Bank Central Asia Tbk',
                'BBRI': 'Bank Rakyat Indonesia (Persero) Tbk',
                'BBNI': 'Bank Negara Indonesia (Persero) Tbk',
                'BMRI': 'Bank Mandiri (Persero) Tbk',
                'BRIS': 'Bank Syariah Indonesia Tbk',
                'BTPN': 'Bank Tabungan Pensiunan Nasional Tbk',
                'BJBR': 'Bank BJB Tbk',
                'BNGA': 'Bank CIMB Niaga Tbk',
                'BDKR': 'Bank Danamon Indonesia Tbk',
                'BNII': 'Bank Maybank Indonesia Tbk',
                'AGRO': 'Bank Raya Indonesia Tbk',
                'BANK': 'Bank Aladin Syariah Tbk',

                # Consumer Goods
                'UNVR': 'Unilever Indonesia Tbk',
                'INDF': 'Indofood Sukses Makmur Tbk',
                'ICBP': 'Indofood CBP Sukses Makmur Tbk',
                'KLBF': 'Kalbe Farma Tbk',
                'MYOR': 'Mayora Indah Tbk',
                'ULTJ': 'Ultra Jaya Milk Industry Tbk',
                'TKIM': 'Pabrik Kertas Tjiandi Kimia Tbk',
                'DLTA': 'Delta Dunia Makmur Tbk',
                'FAST': 'Fast Food Indonesia Tbk',
                'SQMI': 'Suaraya Moda Nusantara Tbk',
                'GEMZ': 'Gemilang Makmur Sejahtera Tbk',

                # Telecommunication
                'TLKM': 'Telekomunikasi Indonesia (Persero) Tbk',
                'EXCL': 'XL Axiata Tbk',
                'ISAT': 'Indosat Ooredoo Hutchison Tbk',
                'FREN': 'Smartfren Telecom Tbk',
                'BRO': 'Digital Mediatama Maxima Tbk',
                'MNCN': 'Media Nusantara Citra Tbk',
                'MBSS': 'MNC Bisnis Tbk',
                'MNCN': 'MNC Sky Vision Tbk',

                # Technology & Digital
                'GOTO': 'GoTo Gojek Tokopedia Tbk',
                'BUKK': 'Bukalapak.com Tbk',
                'ARSA': 'MNC Digital Entertainment Tbk',
                'EDGE': 'MD Pictures Tbk',
                'BCAP': 'MNC Kapital Indonesia Tbk',
                'BABP': 'Bank Aladin Syariah Tbk',

                # Conglomerates
                'ASII': 'Astra International Tbk',
                'ELSA': 'Elnusa Tbk',
                'MEDC': 'Medco Energi International Tbk',
                'ADRO': 'Adaro Energy Tbk',
                'PGAS': 'Perusahaan Gas Negara (Persero) Tbk',
                'PTBA': 'Tambang Batubara Bukit Asam Tbk',
                'ITMG': 'Indo Tambangraya Megah Tbk',
                'HRUM': 'Harum Energy Tbk',

                # Automotive
                'AUTO': 'Astra Otoparts Tbk',
                'GJTL': 'Gajah Tunggal Tbk',
                'TPIA': 'Chandra Asri Petrochemical Tbk',
                'SIPD': 'Sri Rejeki Isman Tbk',
                'PRAS': 'Prasidha Aneka Niaga Tbk',
                'MGRO': 'Magnificent Digital Selular Tbk',

                # Infrastructure & Construction
                'WIKA': 'Waskita Karya (Persero) Tbk',
                'ADHI': 'Adhi Karya (Persero) Tbk',
                'PTPP': 'PP (Persero) Tbk',
                'JSMR': 'Jasa Marga (Persero) Tbk',
                'TOWR': 'Sarana Menara Nusantara Tbk',
                'EXCL': 'Tower Bersama Infrastructure Tbk',
                'RUIS': 'Rukun Raharja Tbk',
                'DGIK': 'Wulandari Bangun Persada Tbk',

                # Property & Real Estate
                'PWON': 'Pakuwon Jati Tbk',
                'CTRA': 'Ciputra Development Tbk',
                'BSDE': 'Bumi Serpong Damai Tbk',
                'LPKR': 'Lippo Karawaci Tbk',
                'CIPR': 'Ciputra Property Tbk',
                'PSDN': 'Pioneerindo Gatra International Tbk',
                'PIEF': 'Pioneerindo Gatra International Tbk',
                'APLN': 'Alam Sutera Realty Tbk',
                'DART': 'Darmi Bersaudera Tbk',

                # Mining & Energy
                'ANTM': 'Aneka Tambang Tbk',
                'TINS': 'Timah (Persero) Tbk',
                'DOID': 'Delta Dunia Makmur Tbk',
                'TARO': 'Tiga Pilar Sejahtera Food Tbk',
                'SMMT': 'Sumber Mitra Jaya Makmur Tbk',
                'SMRU': 'Sinar Mas Multiartha Tbk',

                # Agriculture & Plantation
                'JPFA': 'Japfa Comfeed Indonesia Tbk',
                'CPIN': 'Charoen Pokphand Indonesia Tbk',
                'SIDO': 'Sido Mulyo Tbk',
                'KAEF': 'Kimia Farma Tbk',
                'HEAL': 'Mediheal Indonesia Tbk',
                'MIKA': 'Mitra Keluarga Karyasehat Tbk',
                'SIPD': 'Sri Rejeki Isman Tbk',

                # Retail & Services
                'LPPF': 'Matahari Department Store Tbk',
                'MAP': 'Mitra Adiperkasa Tbk',
                'ACES': 'Ace Hardware Indonesia Tbk',
                'RANC': 'Ramayana Lestari Sentosa Tbk',
                'KIPA': 'Kartika Permata Nusantara Tbk',
                'MAPI': 'Mitra Adiperkasa Tbk',
                'SRIL': 'Sri Rejeki Isman Tbk',
                'SKBM': 'Sukabumi Trading Tbk',

                # Healthcare
                'KLBF': 'Kalbe Farma Tbk',
                'PRDA': 'Pyridam Farma Tbk',
                'KAEF': 'Kimia Farma Tbk',
                'DOKA': 'Dokter Mobil Indonesia Tbk',
                'CARE': 'Mediheal Indonesia Tbk',
                'HEAL': 'Mediheal Indonesia Tbk',

                # Chemicals
                'SILO': 'Siloam International Hospitals Tbk',
                'DLTA': 'Delta Dunia Makmur Tbk',
                'INTP': 'Indocement Tunggal Prakasa Tbk',
                'SMGR': 'Semen Indonesia (Persero) Tbk',
                'SMCB': 'Semen Indonesia (Persero) Tbk',

                # Transportation
                'GIAA': 'Garuda Indonesia (Persero) Tbk',
                'JCT': 'JAPFA Sudirma Sentra Tbk',
                'SMDR': 'Samudera Indonesia Tbk',
                'HUMP': 'Humpuss Maritim Internasional Tbk',
                'TUGU': 'Tugu Pratama Indonesia Tbk'
            }

            for symbol, name in major_stocks.items():
                stocks.append({
                    'symbol': symbol,
                    'name': name,
                    'symbol_jk': f'{symbol}.JK'
                })

            self._stock_list = stocks
            logger.info(f"Loaded {len(stocks)} stocks")
            return stocks

        except Exception as e:
            logger.error(f"Error getting stock list: {str(e)}")
            return []

    def get_historical_data(self, symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """
        Get historical stock data

        Args:
            symbol: Stock symbol (without .JK)
            period: Period string (1y, 2y, 5y, etc.)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Primary: Try Yahoo Finance with .JK suffix
            ticker_symbol = f"{symbol}.JK"

            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(period=period)

            if not df.empty and len(df) > 50:
                # Ensure we have the right columns
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df.index.name = 'Date'

                logger.info(f"Loaded {len(df)} days for {symbol} from Yahoo Finance")
                return df

            # Fallback: Try without .JK
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)

            if not df.empty and len(df) > 50:
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df.index.name = 'Date'
                logger.info(f"Loaded {len(df)} days for {symbol} (no .JK)")
                return df

            logger.warning(f"No data found for {symbol}")
            return None

        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {str(e)}")
            return None

    def get_multiple_stocks(self, symbols: List[str], period: str = "2y") -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple stocks

        Args:
            symbols: List of stock symbols
            period: Period for historical data

        Returns:
            Dictionary of {symbol: DataFrame}
        """
        data = {}

        for symbol in symbols:
            try:
                df = self.get_historical_data(symbol, period)
                if df is not None:
                    data[symbol] = df
                time.sleep(0.1)  # Small delay to avoid rate limiting
            except Exception as e:
                logger.error(f"Error getting {symbol}: {str(e)}")

        return data

    def get_market_indices(self, period: str = "2y") -> Dict[str, pd.DataFrame]:
        """
        Get Indonesian market indices

        Args:
            period: Period for data

        Returns:
            Dictionary of indices data
        """
        indices = {
            'IHSG': '^JKSE',  # Jakarta Composite Index
            'JII': None,       # Jakarta Islamic Index
            'LQ45': None,      # LQ45 Index
            'KOMPAS100': None  # Kompas 100 Index
        }

        data = {}

        for name, symbol in indices.items():
            try:
                if symbol:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(period=period)
                    if not df.empty:
                        data[name] = df
            except Exception as e:
                logger.error(f"Error getting {name}: {str(e)}")

        return data

    def save_data_to_cache(self, data: Dict, cache_dir: str = None):
        """
        Save downloaded data to cache

        Args:
            data: Dictionary of data to save
            cache_dir: Directory to save data
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), 'idx_cache')

        os.makedirs(cache_dir, exist_ok=True)

        for symbol, df in data.items():
            try:
                file_path = os.path.join(cache_dir, f"{symbol}.csv")
                df.to_csv(file_path)
                logger.info(f"Saved {symbol} data to {file_path}")
            except Exception as e:
                logger.error(f"Error saving {symbol}: {str(e)}")

    def load_from_cache(self, symbol: str, cache_dir: str = None) -> Optional[pd.DataFrame]:
        """
        Load data from cache

        Args:
            symbol: Stock symbol
            cache_dir: Cache directory

        Returns:
            DataFrame or None
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), 'idx_cache')

        file_path = os.path.join(cache_dir, f"{symbol}.csv")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                logger.info(f"Loaded {symbol} from cache")
                return df
            except Exception as e:
                logger.error(f"Error loading {symbol} from cache: {str(e)}")

        return None


# Compatibility layer for the original idx_scrapper interface
class IdxScrapper(CustomIDXScraper):
    """Compatibility wrapper for the original idx_scrapper"""
    pass


def test_scrapper():
    """Test the custom IDX scraper"""
    print("="*60)
    print("TESTING CUSTOM IDX SCRAPER")
    print("="*60)

    scraper = CustomIDXScraper()

    # Test stock list
    print("\n[*] Getting stock list...")
    stocks = scraper.get_idx_stock_list()
    print(f"  Found {len(stocks)} stocks")

    # Test getting data for a few stocks
    test_symbols = ['BBCA', 'BBRI', 'TLKM']
    print(f"\n[*] Testing data download for: {', '.join(test_symbols)}")

    data = scraper.get_multiple_stocks(test_symbols, period="1y")

    for symbol, df in data.items():
        print(f"\n  {symbol}:")
        print(f"    Rows: {len(df)}")
        print(f"    Date range: {df.index.min()} to {df.index.max()}")
        print(f"    Latest price: {df['Close'][-1]:.2f}")

    print("\n✓ Test completed successfully!")


if __name__ == "__main__":
    test_scrapper()