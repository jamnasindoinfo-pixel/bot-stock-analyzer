"""
Financial Data Fetcher for Indonesian Stocks
Fetches fundamental financial data from yfinance for narrative analysis
"""

import yfinance as yf
import pandas as pd
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class FinancialDataFetcher:
    """Fetches fundamental financial data for Indonesian stocks"""

    def __init__(self):
        self.session = None
        self.cache = {}
        self.cache_duration = timedelta(hours=1)  # Cache data for 1 hour

    def _get_ticker(self, symbol: str) -> yf.Ticker:
        """Get yfinance ticker object with proper Indonesian stock format"""
        # Add .JK suffix if not present
        ticker_symbol = f"{symbol}.JK" if not symbol.endswith('.JK') else symbol
        return yf.Ticker(ticker_symbol)

    def _is_cached(self, symbol: str, data_type: str) -> bool:
        """Check if data is cached and still valid"""
        cache_key = f"{symbol}_{data_type}"
        if cache_key not in self.cache:
            return False

        cached_time = self.cache[cache_key].get('timestamp')
        if cached_time and (datetime.now() - cached_time) < self.cache_duration:
            return True

        return False

    def _get_cached_data(self, symbol: str, data_type: str) -> Optional[Dict]:
        """Get cached data if available"""
        cache_key = f"{symbol}_{data_type}"
        if self._is_cached(symbol, data_type):
            return self.cache[cache_key]['data']
        return None

    def _cache_data(self, symbol: str, data_type: str, data: Dict) -> None:
        """Cache data with timestamp"""
        cache_key = f"{symbol}_{data_type}"
        self.cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }

    def get_financial_statements(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive financial statements"""
        # Check cache first
        cached = self._get_cached_data(symbol, 'financials')
        if cached:
            return cached

        try:
            ticker = self._get_ticker(symbol)

            # Get financial statements
            financials = ticker.financials
            quarterly_financials = ticker.quarterly_financials
            balance_sheet = ticker.balance_sheet
            quarterly_balance_sheet = ticker.quarterly_balance_sheet
            cashflow = ticker.cashflow
            quarterly_cashflow = ticker.quarterly_cashflow

            # Process data - check if it's already a dict
            result = {
                'annual': {
                    'income_statement': self._process_dataframe(financials) if financials is not None else {},
                    'balance_sheet': self._process_dataframe(balance_sheet) if balance_sheet is not None else {},
                    'cash_flow': self._process_dataframe(cashflow) if cashflow is not None else {}
                },
                'quarterly': {
                    'income_statement': self._process_dataframe(quarterly_financials) if quarterly_financials is not None else {},
                    'balance_sheet': self._process_dataframe(quarterly_balance_sheet) if quarterly_balance_sheet is not None else {},
                    'cash_flow': self._process_dataframe(quarterly_cashflow) if quarterly_cashflow is not None else {}
                }
            }

            # Cache the result
            self._cache_data(symbol, 'financials', result)
            return result

        except Exception as e:
            logger.error(f"Error fetching financial statements for {symbol}: {e}")
            return {}

    def get_key_metrics(self, symbol: str) -> Dict[str, Any]:
        """Calculate key financial metrics and ratios"""
        cached = self._get_cached_data(symbol, 'metrics')
        if cached:
            return cached

        try:
            ticker = self._get_ticker(symbol)
            info = ticker.info

            # Get financial data for calculations
            financials = self.get_financial_statements(symbol)

            # Extract key metrics
            result = {
                'market_data': {
                    'market_cap': info.get('marketCap'),
                    'enterprise_value': info.get('enterpriseValue'),
                    'pe_ratio': info.get('trailingPE'),
                    'pb_ratio': info.get('priceToBook'),
                    'ps_ratio': info.get('priceToSalesTrailing12Months'),
                    'ev_ebitda': info.get('enterpriseToEbitda'),
                    'dividend_yield': info.get('dividendYield') * 100 if info.get('dividendYield') else None
                },
                'profitability': {
                    'roe': info.get('returnOnEquity') * 100 if info.get('returnOnEquity') else None,
                    'roa': info.get('returnOnAssets') * 100 if info.get('returnOnAssets') else None,
                    'roic': info.get('returnOnCapital') * 100 if info.get('returnOnCapital') else None,
                    'gross_margin': info.get('grossMargins') * 100 if info.get('grossMargins') else None,
                    'operating_margin': info.get('operatingMargins') * 100 if info.get('operatingMargins') else None,
                    'net_margin': info.get('profitMargins') * 100 if info.get('profitMargins') else None
                },
                'financial_health': {
                    'debt_to_equity': info.get('debtToEquity'),
                    'current_ratio': info.get('currentRatio'),
                    'quick_ratio': info.get('quickRatio'),
                    'interest_coverage': None  # Calculate from financials
                },
                'growth': {
                    'revenue_growth': info.get('revenueGrowth') * 100 if info.get('revenueGrowth') else None,
                    'earnings_growth': info.get('earningsGrowth') * 100 if info.get('earningsGrowth') else None,
                    'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth') * 100 if info.get('earningsQuarterlyGrowth') else None
                },
                'company_info': {
                    'sector': info.get('sector'),
                    'industry': info.get('industry'),
                    'description': info.get('longBusinessSummary'),
                    'employees': info.get('fullTimeEmployees'),
                    'website': info.get('website'),
                    'country': info.get('country')
                }
            }

            # Calculate additional metrics from financial statements
            if financials:
                result = self._calculate_additional_metrics(result, financials)

            # Cache the result
            self._cache_data(symbol, 'metrics', result)
            return result

        except Exception as e:
            logger.error(f"Error calculating metrics for {symbol}: {e}")
            return {}

    def get_growth_trends(self, symbol: str, years: int = 3) -> Dict[str, Any]:
        """Analyze revenue and profit growth trends"""
        cached = self._get_cached_data(symbol, 'growth')
        if cached:
            return cached

        try:
            financials = self.get_financial_statements(symbol)
            result = {
                'revenue_trend': [],
                'profit_trend': [],
                'eps_trend': [],
                'dividend_trend': []
            }

            if financials and 'annual' in financials:
                income_stmt = financials['annual']['income_statement']

                # Extract trends from historical data - handle dict format
                if isinstance(income_stmt, dict) and 'Total Revenue' in income_stmt:
                    revenue_data = income_stmt['Total Revenue']
                    if isinstance(revenue_data, dict):
                        periods = sorted(revenue_data.keys(), reverse=True)[:years]
                        prev_value = None
                        for period in periods:
                            value = revenue_data[period]
                            if prev_value is not None and prev_value > 0:
                                growth = ((value - prev_value) / prev_value) * 100
                                result['revenue_trend'].append({
                                    'year': str(period)[:4],  # Extract year
                                    'value': value,
                                    'growth_yoy': growth
                                })
                            prev_value = value

                if isinstance(income_stmt, dict) and 'Net Income' in income_stmt:
                    profit_data = income_stmt['Net Income']
                    if isinstance(profit_data, dict):
                        periods = sorted(profit_data.keys(), reverse=True)[:years]
                        prev_value = None
                        for period in periods:
                            value = profit_data[period]
                            if prev_value is not None and prev_value > 0:
                                growth = ((value - prev_value) / prev_value) * 100
                                result['profit_trend'].append({
                                    'year': str(period)[:4],  # Extract year
                                    'value': value,
                                    'growth_yoy': growth
                                })
                            prev_value = value

            # Get dividend history
            try:
                ticker = self._get_ticker(symbol)
                dividends = ticker.dividends
                if dividends is not None and not dividends.empty:
                    yearly_dividends = dividends.groupby(dividends.index.year).sum()
                    for year, dividend in yearly_dividends.items():
                        result['dividend_trend'].append({
                            'year': year,
                            'dividend': dividend
                        })
            except:
                pass  # Ignore dividend errors

            # Cache the result
            self._cache_data(symbol, 'growth', result)
            return result

        except Exception as e:
            logger.error(f"Error analyzing growth trends for {symbol}: {e}")
            return {}

    def get_quarterly_performance(self, symbol: str) -> Dict[str, Any]:
        """Extract and analyze quarterly performance"""
        cached = self._get_cached_data(symbol, 'quarterly')
        if cached:
            return cached

        try:
            financials = self.get_financial_statements(symbol)
            result = {
                'latest_quarter': {},
                'quarterly_comparison': [],
                'key_metrics': {}
            }

            if financials and 'quarterly' in financials:
                q_income = financials['quarterly']['income_statement']

                # Get latest quarter data - handle dict format
                if isinstance(q_income, dict) and q_income:
                    # Get the first (latest) period
                    latest_period = list(q_income.keys())[0]
                    latest_data = q_income[latest_period]

                    result['latest_quarter'] = {
                        'period': str(latest_period),
                        'revenue': latest_data.get('Total Revenue', 0),
                        'gross_profit': latest_data.get('Gross Profit', 0),
                        'operating_income': latest_data.get('Operating Income', 0),
                        'net_income': latest_data.get('Net Income', 0),
                        'eps': latest_data.get('Diluted EPS', 0)
                    }

                    # Compare with previous quarter if available
                    if len(q_income) > 1:
                        prev_period = list(q_income.keys())[1]
                        prev_data = q_income[prev_period]

                        result['quarterly_comparison'] = {
                            'revenue_change': self._calculate_change(
                                prev_data.get('Total Revenue', 0),
                                latest_data.get('Total Revenue', 0)
                            ),
                            'net_income_change': self._calculate_change(
                                prev_data.get('Net Income', 0),
                                latest_data.get('Net Income', 0)
                            )
                        }

                # Calculate quarterly metrics
                if result['latest_quarter']:
                    latest = result['latest_quarter']
                    if latest.get('revenue') and latest.get('gross_profit'):
                        result['key_metrics']['gross_margin'] = (latest['gross_profit'] / latest['revenue']) * 100
                    if latest.get('revenue') and latest.get('net_income'):
                        result['key_metrics']['net_margin'] = (latest['net_income'] / latest['revenue']) * 100

            # Cache the result
            self._cache_data(symbol, 'quarterly', result)
            return result

        except Exception as e:
            logger.error(f"Error getting quarterly performance for {symbol}: {e}")
            return {}

    def _process_dataframe(self, df: pd.DataFrame) -> Dict:
        """Convert pandas DataFrame to dictionary"""
        if df is None or df.empty:
            return {}

        # Convert to dictionary with proper handling of MultiIndex
        result = {}
        for index, row in df.iterrows():
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                # For now, just use the first level
                col_dict = {}
                for col in df.columns:
                    col_key = col[0] if isinstance(col, tuple) else col
                    col_dict[col_key] = row[col]
                result[str(index)] = col_dict
            else:
                result[str(index)] = row.to_dict()

        return result

    def _calculate_additional_metrics(self, metrics: Dict, financials: Dict) -> Dict:
        """Calculate additional metrics from financial statements"""
        try:
            # Get recent balance sheet and income statement
            if 'annual' in financials and financials['annual']['balance_sheet']:
                balance = financials['annual']['balance_sheet']
                income = financials['annual']['income_statement']

                # Calculate debt ratios if data available
                if balance and 'Total Debt' in balance and 'Total Stockholder Equity' in balance:
                    total_debt = list(balance['Total Debt'].values())[0] if balance['Total Debt'] else 0
                    total_equity = list(balance['Total Stockholder Equity'].values())[0] if balance['Total Stockholder Equity'] else 0

                    if total_equity > 0:
                        metrics['financial_health']['debt_to_equity'] = total_debt / total_equity

                # Calculate interest coverage if data available
                if income and 'Operating Income' in income and 'Interest Expense' in income:
                    operating_income = list(income['Operating Income'].values())[0] if income['Operating Income'] else 0
                    interest_expense = abs(list(income['Interest Expense'].values())[0]) if income['Interest Expense'] else 0

                    if interest_expense > 0:
                        metrics['financial_health']['interest_coverage'] = operating_income / interest_expense

        except Exception as e:
            logger.error(f"Error calculating additional metrics: {e}")

        return metrics

    def _calculate_change(self, previous: float, current: float) -> Dict[str, float]:
        """Calculate percentage and absolute change"""
        if previous == 0:
            return {
                'absolute': current,
                'percentage': 0
            }

        change = current - previous
        pct_change = (change / abs(previous)) * 100

        return {
            'absolute': change,
            'percentage': pct_change
        }