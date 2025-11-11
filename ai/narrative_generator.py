"""
Narrative Generator for Financial Analysis
Generates comprehensive financial narrative analysis using Gemini AI
"""

from typing import Dict, Any, Optional, List
from google import generativeai as genai
import json
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class NarrativeGenerator:
    """Generates comprehensive narrative financial analysis"""

    def __init__(self, gemini_client: genai.GenerativeModel):
        self.client = gemini_client
        # Use gemini-1.5-flash which has higher free tier limits
        self.model_name = "gemini-1.5-flash"
        self.model = genai.GenerativeModel(self.model_name)
        self.max_retries = 3
        self.base_delay = 60  # Base delay in seconds for rate limits

    def build_narrative_prompt(self, symbol: str, technical_data: Dict,
                             financial_data: Dict, ml_data: Dict,
                             quarterly_data: Dict, growth_data: Dict) -> str:
        """Build comprehensive prompt for narrative generation"""

        # Extract key information
        current_price = technical_data.get('metrics', {}).get('last_close', 0)
        price_change = technical_data.get('metrics', {}).get('price_change_5', 0)
        volume_ratio = technical_data.get('stock', {}).get('volume_ratio', 0)

        # Financial metrics
        metrics = financial_data.get('market_data', {})
        profitability = financial_data.get('profitability', {})
        financial_health = financial_data.get('financial_health', {})
        growth = financial_data.get('growth', {})
        company_info = financial_data.get('company_info', {})

        # Quarterly performance
        latest_qtr = quarterly_data.get('latest_quarter', {})
        qtr_comparison = quarterly_data.get('quarterly_comparison', {})

        # Growth trends
        revenue_trend = growth_data.get('revenue_trend', [])[-3:]  # Last 3 years
        profit_trend = growth_data.get('profit_trend', [])[-3:]   # Last 3 years

        # ML prediction
        ml_signal = ml_data.get('signal', 'N/A')
        ml_confidence = ml_data.get('confidence', 0)

        # Prepare values with proper handling
        price_str = f"Rp {current_price:,.0f}" if current_price else "N/A"
        price_change_str = f"{price_change:+.2f}%" if price_change is not None else "0.00%"
        volume_ratio_str = f"{volume_ratio:.1f}x" if volume_ratio is not None else "0.0x"

        # Build the narrative prompt
        prompt = f"""
Kamu adalah analis saham profesional yang sedang menulis laporan analisis mendalam untuk saham Indonesia {symbol}.
Tulislah analisis naratif dalam format gaya jurnalisme keuangan seperti contoh conceptAI.md, dengan cerita yang menarik dan data yang mendalam.

INFORMASI PERUSAHAAN:
- Nama Saham: {symbol}
- Sektor: {company_info.get('sector', 'N/A')}
- Industri: {company_info.get('industry', 'N/A')}
- Deskripsi: {str(company_info.get('description', 'N/A'))[:300]}...

DATA PASAR SAAT INI:
- Harga Terakhir: {price_str}
- Perubahan 5 Hari: {price_change_str}
- Rasio Volume: {volume_ratio_str}
- Signal ML v5: {ml_signal} (confidence: {ml_confidence:.1%})

METRIK FINANSIAL KUNCI:
- Market Cap: Rp {metrics.get('market_cap', 0):,.0f}
- P/E Ratio: {metrics.get('pe_ratio', 'N/A')}
- P/B Ratio: {metrics.get('pb_ratio', 'N/A')}
- Dividend Yield: {profitability.get('dividend_yield', 0) or 0:.1f}%

PROFITABILITAS:
- ROE: {profitability.get('roe', 0) or 0:.1f}%
- ROA: {profitability.get('roa', 0) or 0:.1f}%
- Gross Margin: {profitability.get('gross_margin', 0) or 0:.1f}%
- Net Margin: {profitability.get('net_margin', 0) or 0:.1f}%

KESEHATAN FINANSIAL:
- Debt to Equity: {financial_health.get('debt_to_equity', 'N/A')}
- Current Ratio: {financial_health.get('current_ratio', 'N/A')}
- Interest Coverage: {financial_health.get('interest_coverage', 'N/A')}

PERTUMBUHAN:
- Revenue Growth: {growth.get('revenue_growth', 0) or 0:.1f}%
- Earnings Growth: {growth.get('earnings_growth', 0) or 0:.1f}%

KINERJA QUARTER TERAKHIR:
- Periode: {latest_qtr.get('period', 'N/A')}
- Revenue: Rp {latest_qtr.get('revenue', 0) or 0:,.0f}
- Net Income: Rp {latest_qtr.get('net_income', 0) or 0:,.0f}
- EPS: Rp {latest_qtr.get('eps', 0) or 0:,.0f}

TREN PERTUMBUHAN HISTORIS:
{self._format_growth_trends(revenue_trend, profit_trend)}

INSTRUKSI TULISAN:
1. Buat judul yang menarik yang merangkum tema utama analisis (misal: "BBCA Q3 2024: Digital Banking Momentum Amid Margin Compression")
2. Tulis dalam bahasa Indonesia dengan gaya yang menarik dan mudah dipahami
3. Fokus pada cerita di balik angka, bukan hanya daftar data
4. Buat 5-7 paragraf dengan alur yang logis:
   - Paragraf 1: Executive summary - gambaran besar kinerja terkini
   - Paragraf 2-3: Analisis pertumbuhan (revenue, profit, strategi ekspansi)
   - Paragraf 4: Kesehatan finansial mendalam (cash flow, debt, efisiensi)
   - Paragraf 5: Risiko dan tantangan yang dihadapi
   - Paragraf 6: Outlook dan prospek masa depan
5. Gunakan data konkret untuk mendukung narasi
6. Sebutkan insight dari ML prediction jika confidence > 70%
7. Berikan kesimpulan investasi yang seimbang (bull vs bear case)

Tulis analisis lengkap dalam format naratif yang mengalir, bukan bullet points. Jadikan seperti artikel analisis saham yang diterbitkan di media finansial terkemuka.
"""
        return prompt

    def generate_narrative_analysis(self, symbol: str, technical_data: Dict,
                                  financial_data: Dict, ml_data: Dict,
                                  quarterly_data: Dict, growth_data: Dict) -> Dict:
        """Generate full narrative analysis with sections"""

        try:
            # Build the prompt
            prompt = self.build_narrative_prompt(
                symbol, technical_data, financial_data,
                ml_data, quarterly_data, growth_data
            )

            # Generate response from Gemini with retry logic
            response = self._generate_with_retry(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.5,
                    max_output_tokens=2000,
                    top_p=0.8,
                    top_k=40
                )
            )

            if response and response.text:
                # Parse and structure the response
                narrative_text = response.text.strip()

                # Try to extract sections
                sections = self._parse_narrative_response(narrative_text)

                # Create a catchy title if not present
                if 'title' not in sections:
                    title = self._generate_title(symbol, technical_data, financial_data, ml_data)
                    sections['title'] = title

                # Add metadata
                sections['metadata'] = {
                    'symbol': symbol,
                    'generated_at': datetime.now().isoformat(),
                    'data_sources': ['yfinance', 'technical_indicators', 'ml_predictions_v5'],
                    'word_count': len(narrative_text.split())
                }

                return {
                    'success': True,
                    'data': sections,
                    'full_text': narrative_text
                }
            else:
                return {
                    'success': False,
                    'error': 'No response from AI model'
                }

        except Exception as e:
            logger.error(f"Error generating narrative analysis for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _parse_narrative_response(self, response_text: str) -> Dict[str, str]:
        """Parse AI response into structured sections"""
        sections = {}

        # Try to identify sections based on common patterns
        lines = response_text.split('\n')
        current_section = 'content'
        current_content = []

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Check for title (usually the first line)
            if not sections and ':' in line and any(keyword in line.upper() for keyword in ['Q1', 'Q2', 'Q3', 'Q4', '2023', '2024', '2025']):
                sections['title'] = line
                continue

            # Check for section headers
            if any(keyword in line.lower() for keyword in ['executive', 'ringkasan', 'summary']):
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'executive_summary'
                current_content = []
            elif any(keyword in line.lower() for keyword in ['pertumbuhan', 'growth', 'revenue']):
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'growth_analysis'
                current_content = []
            elif any(keyword in line.lower() for keyword in ['kesehatan', 'financial', 'kas', 'cash flow']):
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'financial_health'
                current_content = []
            elif any(keyword in line.lower() for keyword in ['risiko', 'risk', 'tantangan']):
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'risk_factors'
                current_content = []
            elif any(keyword in line.lower() for keyword in ['outlook', 'prospek', 'masa depan', 'kesimpulan']):
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'outlook'
                current_content = []
            else:
                current_content.append(line)

        # Don't forget the last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)

        # If no sections were identified, put everything in main content
        if 'content' in sections and len(sections) == 1:
            sections['main_content'] = sections['content']
            del sections['content']

        return sections

    def _generate_title(self, symbol: str, technical_data: Dict,
                       financial_data: Dict, ml_data: Dict) -> str:
        """Generate a catchy title for the analysis"""

        # Get key metrics
        price_change = technical_data.get('metrics', {}).get('price_change_5', 0)
        revenue_growth = financial_data.get('growth', {}).get('revenue_growth', 0)
        ml_signal = ml_data.get('signal', 'HOLD')

        # Generate title based on key themes
        current_quarter = self._get_current_quarter()
        year = datetime.now().year

        if price_change > 5 and revenue_growth > 10:
            return f"{symbol} {current_quarter} {year}: Strong Growth Momentum Continues"
        elif price_change < -5:
            return f"{symbol} {current_quarter} {year}: Navigating Through Challenging Times"
        elif ml_signal == 'BUY' and ml_data.get('confidence', 0) > 0.7:
            return f"{symbol} {current_quarter} {year}: poised for Rebound Amid Market Uncertainty"
        elif revenue_growth > 20:
            return f"{symbol} {current_quarter} {year}: Growth Acceleration Drives Investor Confidence"
        else:
            return f"{symbol} {current_quarter} {year}: Strategic Positioning for Long-term Value"

    def _format_growth_trends(self, revenue_trend: List[Dict], profit_trend: List[Dict]) -> str:
        """Format growth trends for the prompt"""
        trends_text = []

        if revenue_trend:
            trends_text.append("Tren Revenue:")
            for trend in revenue_trend:
                trends_text.append(f"  - {trend['year']}: Rp {trend['value']:,.0f} ({trend['growth_yoy']:+.1f}%)")

        if profit_trend:
            trends_text.append("Tren Laba:")
            for trend in profit_trend:
                trends_text.append(f"  - {trend['year']}: Rp {trend['value']:,.0f} ({trend['growth_yoy']:+.1f}%)")

        return '\n'.join(trends_text) if trends_text else "Data tren tidak tersedia"

    def _get_current_quarter(self) -> str:
        """Get current quarter string"""
        month = datetime.now().month
        if month <= 3:
            return "Q1"
        elif month <= 6:
            return "Q2"
        elif month <= 9:
            return "Q3"
        else:
            return "Q4"

    def generate_quick_insight(self, symbol: str, technical_data: Dict,
                             financial_data: Dict, ml_data: Dict) -> Dict:
        """Generate quick insight for display in summary"""

        try:
            # Build a shorter prompt for quick insights
            prompt = f"""
Berdasarkan data berikut untuk saham {symbol}, berikan 3 insight kunci dalam bahasa Indonesia:

Data Teknikal:
- Harga: {technical_data.get('metrics', {}).get('last_close', 0)}
- Perubahan: {technical_data.get('metrics', {}).get('price_change_5', 0):+.2f}%
- Volume Ratio: {technical_data.get('stock', {}).get('volume_ratio', 0):.1f}x

Data Finansial:
- ROE: {financial_data.get('profitability', {}).get('roe', 0):.1f}%
- Debt to Equity: {financial_data.get('financial_health', {}).get('debt_to_equity', 'N/A')}
- Revenue Growth: {financial_data.get('growth', {}).get('revenue_growth', 0):.1f}%

ML Signal: {ml_data.get('signal', 'N/A')} (confidence: {ml_data.get('confidence', 0):.1%})

Berikan 3 insight singkat (maksimal 50 kata per insight):
1. Tentang kinerja terkini
2. Tentang kesehatan finansial
3. Tentang outlook/rekomendasi
"""

            response = self._generate_with_retry(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.4,
                    max_output_tokens=300
                )
            )

            if response and response.text:
                return {
                    'success': True,
                    'insights': response.text.strip()
                }
            else:
                return {
                    'success': False,
                    'error': 'No response from AI model'
                }

        except Exception as e:
            logger.error(f"Error generating quick insight for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _generate_with_retry(self, prompt: str, **kwargs) -> Optional[genai.types.GenerateContentResponse]:
        """Generate content with retry logic for rate limits"""
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(prompt, **kwargs)
                return response
            except Exception as e:
                error_str = str(e).lower()

                # Check if it's a rate limit error
                if 'quota' in error_str or 'rate limit' in error_str or '429' in error_str:
                    if attempt < self.max_retries - 1:
                        # Extract retry delay if provided by API
                        retry_delay = self.base_delay * (2 ** attempt)  # Exponential backoff

                        # Try to extract suggested retry time from error message
                        if 'retry in' in error_str:
                            import re
                            match = re.search(r'retry in (\d+(?:\.\d+)?)s', error_str)
                            if match:
                                retry_delay = float(match.group(1))

                        logger.warning(f"Rate limit hit, retrying in {retry_delay:.0f} seconds... (attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"Max retries exceeded for rate limit: {e}")
                        raise

                # For other errors, don't retry
                logger.error(f"Error generating content: {e}")
                raise

        return None