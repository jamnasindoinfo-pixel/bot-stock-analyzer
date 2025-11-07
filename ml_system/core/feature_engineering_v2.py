"""
Enhanced Feature Engineering v2
Comprehensive feature set for ML-based stock signal analysis

Technical Features: 8+ indicators (VWAP, MFI, Aroon, OBV, RoC, Bollinger Bands)
Sentiment Features: 6 indicators (placeholders for FREE APIs)
Economic Features: 4 indicators (placeholders for FRED API)
Temporal Features: 4 indicators (momentum, gap, autocorr, seasonal)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')


class EnhancedFeatureEngineer:
    """
    Enhanced feature engineering for ML-based stock signal analysis

    Generates comprehensive feature set including:
    - Technical indicators (8+ features)
    - Sentiment indicators (6 features)
    - Economic indicators (4 features)
    - Temporal features (4+ features)
    """

    def __init__(self):
        """Initialize EnhancedFeatureEngineer"""
        self.required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']

    def create_features(self, df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Create comprehensive feature set from OHLCV data

        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol for external data fetching

        Returns:
            DataFrame with enhanced features
        """
        if df.empty:
            return pd.DataFrame()

        # Handle multi-level columns from yfinance
        if hasattr(df.columns, 'levels') and len(df.columns.levels) > 1:
            # Flatten multi-level columns: ('Close', 'AAPL') -> 'Close'
            df.columns = df.columns.get_level_values(0)

        # Add Adj Close column if missing (use Close as Adj Close for recent data)
        if 'Adj Close' not in df.columns and 'Close' in df.columns:
            df = df.copy()
            df['Adj Close'] = df['Close']

        # Ensure standard column names
        column_mapping = {
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume',
            'Adj Close': 'Adj Close'
        }

        # Rename columns if needed (case insensitive)
        for old_col, new_col in column_mapping.items():
            for col in df.columns:
                if col.lower() == old_col.lower() and col != new_col:
                    df = df.rename(columns={col: new_col})
                    break

        # Validate input data
        if not all(col in df.columns for col in self.required_columns):
            missing = [col for col in self.required_columns if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing}")

        features = df.copy()

        try:
            # Technical Features (8+ indicators)
            features = self._add_technical_features(features)

            # Sentiment Features (6 indicators - placeholders)
            features = self._add_sentiment_features(features, symbol)

            # Economic Features (4 indicators - placeholders for FRED API)
            features = self._add_economic_features(features, symbol)

            # Temporal Features (4+ indicators)
            features = self._add_temporal_features(features)

            # Clean data
            features = self._clean_features(features)

            return features

        except Exception as e:
            print(f"[EnhancedFeatureEngineer] Error creating features: {e}")
            return df.copy()

    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 8+ technical indicators"""

        # 1. Volume Weighted Average Price (VWAP)
        df['vwap'] = self._calculate_vwap(df)

        # 2. Money Flow Index (MFI)
        df['mfi_14'] = self._calculate_mfi(df, period=14)

        # 3. Aroon Indicator
        df['aroon_up_25'], df['aroon_down_25'] = self._calculate_aroon(df, period=25)

        # 4. On-Balance Volume (OBV)
        df['obv'] = self._calculate_obv(df)

        # 5. Rate of Change (RoC)
        df['roc_10'] = self._calculate_roc(df['Close'], period=10)
        df['roc_20'] = self._calculate_roc(df['Close'], period=20)

        # 6. Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['Close'], period=20, std_dev=2)
        df['bollinger_upper'] = bb_upper
        df['bollinger_middle'] = bb_middle
        df['bollinger_lower'] = bb_lower
        df['bollinger_width'] = (bb_upper - bb_lower) / bb_middle
        df['bollinger_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)

        # 7. Additional RSI for confirmation
        df['rsi_14'] = self._calculate_rsi(df['Close'], period=14)
        df['rsi_7'] = self._calculate_rsi(df['Close'], period=7)

        # 8. MACD
        df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df['Close'])

        return df

    def _add_sentiment_features(self, df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
        """Add 6 sentiment features (placeholders for FREE APIs)"""

        # Placeholder values - in real implementation, these would fetch from free APIs
        # Using default values of 0.0 as placeholders

        # 1. News Sentiment Score (-1 to 1)
        df['news_sentiment'] = 0.0

        # 2. Social Media Sentiment (-1 to 1)
        df['social_sentiment'] = 0.0

        # 3. Fear & Greed Index (0 to 100)
        df['fear_greed_index'] = 50.0

        # 4. Put/Call Ratio
        df['put_call_ratio'] = 1.0

        # 5. Volatility Index (VIX) proxy
        df['volatility_index'] = 20.0

        # 6. Insider Trading Sentiment (-1 to 1)
        df['insider_trading_sentiment'] = 0.0

        # In a real implementation, you would fetch these from:
        # - NewsAPI.org for news sentiment
        # - Reddit/Twitter APIs for social sentiment
        # - CNN Fear & Greed Index API
        # - CBOE Data for Put/Call ratio
        # - Yahoo Finance or CBOE for VIX
        # - SEC filings for insider trading data

        return df

    def _add_economic_features(self, df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
        """Add 4 economic features (placeholders for FRED API)"""

        # Placeholder values - in real implementation, these would fetch from FRED API
        # Using realistic default values as placeholders

        # 1. Federal Funds Rate (percentage)
        df['interest_rate'] = 5.25

        # 2. CPI Inflation Rate (percentage)
        df['inflation_rate'] = 3.2

        # 3. GDP Growth Rate (percentage)
        df['gdp_growth'] = 2.1

        # 4. Unemployment Rate (percentage)
        df['unemployment_rate'] = 3.8

        # In a real implementation, you would fetch these from:
        # - FRED API: https://fred.stlouisfed.org/docs/api/fred/
        # - Series IDs: FEDFUNDS, CPIAUCSL, GDP, UNRATE

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 4+ temporal features"""

        # 1. Momentum indicators
        df['momentum_5'] = (df['Close'] / df['Close'].shift(5) - 1) * 100
        df['momentum_10'] = (df['Close'] / df['Close'].shift(10) - 1) * 100

        # 2. Gap ratio (overnight gap)
        df['gap_ratio'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

        # 3. Autocorrelation
        df['autocorr_5'] = df['Close'].rolling(window=5).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
        )

        # 4. Day/Month/Quarter seasonal effects
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter

        # 5. Time-based features
        df['days_since_start'] = (df.index - df.index[0]).days

        # 6. Volatility clustering
        df['volatility_regime'] = df['Close'].pct_change().rolling(window=20).std().rank(pct=True)

        return df

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and handle infinite/missing values"""

        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Forward fill then backward fill to handle missing values
        df = df.ffill().bfill()

        # Fill any remaining NaN with 0
        df = df.fillna(0)

        return df

    # Technical Indicator Calculation Methods
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        return vwap

    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']

        # Positive and negative money flow
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

        # Money flow ratio
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()

        mfr = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + mfr))

        return mfi

    def _calculate_aroon(self, df: pd.DataFrame, period: int = 25) -> tuple:
        """Calculate Aroon Up and Down"""
        # Aroon Up: ((period - days since highest high) / period) * 100
        # Aroon Down: ((period - days since lowest low) / period) * 100

        high_periods = df['High'].rolling(window=period + 1).apply(
            lambda x: x.argmax(), raw=False
        )
        low_periods = df['Low'].rolling(window=period + 1).apply(
            lambda x: x.argmin(), raw=False
        )

        aroon_up = ((period - high_periods) / period) * 100
        aroon_down = ((period - low_periods) / period) * 100

        return aroon_up, aroon_down

    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = []
        obv.append(0)

        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])

        return pd.Series(obv, index=df.index)

    def _calculate_roc(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Rate of Change"""
        return ((series - series.shift(period)) / series.shift(period)) * 100

    def _calculate_bollinger_bands(self, series: pd.Series, period: int = 20, std_dev: float = 2) -> tuple:
        """Calculate Bollinger Bands"""
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        exp1 = series.ewm(span=fast).mean()
        exp2 = series.ewm(span=slow).mean()

        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line

        return macd, signal_line, histogram

    def get_feature_info(self) -> Dict:
        """Get information about all features"""
        return {
            'technical_features': [
                'vwap', 'mfi_14', 'aroon_up_25', 'aroon_down_25', 'obv',
                'roc_10', 'roc_20', 'bollinger_upper', 'bollinger_middle',
                'bollinger_lower', 'bollinger_width', 'bollinger_position',
                'rsi_14', 'rsi_7', 'macd', 'macd_signal', 'macd_histogram'
            ],
            'sentiment_features': [
                'news_sentiment', 'social_sentiment', 'fear_greed_index',
                'put_call_ratio', 'volatility_index', 'insider_trading_sentiment'
            ],
            'economic_features': [
                'interest_rate', 'inflation_rate', 'gdp_growth', 'unemployment_rate'
            ],
            'temporal_features': [
                'momentum_5', 'momentum_10', 'gap_ratio', 'autocorr_5',
                'day_of_week', 'day_of_month', 'month', 'quarter',
                'days_since_start', 'volatility_regime'
            ],
            'total_features': 37  # Total number of features created
        }