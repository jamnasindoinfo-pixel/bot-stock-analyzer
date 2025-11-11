"""
Enhanced Narrative Generator that uses LLM Adapter
Supports multiple AI providers for stock analysis
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from .llm_adapter import llm_manager

logger = logging.getLogger(__name__)


class EnhancedNarrativeGenerator:
    """Enhanced narrative generator using LLM adapter"""

    def __init__(self):
        self.llm_adapter = llm_manager.get_best_adapter()
        self.model_name = ""
        if self.llm_adapter:
            # Get model name from adapter
            if hasattr(self.llm_adapter, 'model_name'):
                self.model_name = self.llm_adapter.model_name
            else:
                self.model_name = self.llm_adapter.__class__.__name__

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

        # Build the narrative prompt optimized for Indonesian stocks with exact conceptAI.md style
        prompt = f"""Anda adalah analis saham profesional yang menulis untuk komunitas investor Indonesia. Tulislah analisis saham {symbol} persis seperti gaya tulisan di conceptAI.md.

CONTOH STRUKTUR YANG DIHARAPKAN:
Judul Format: "$SYMBOL Q3 2025: [Phrase 1] With [Phrase 2]"
Contoh: "$ERAL Q3 2025: Growth With Bleeding Cash"

GAYA TULISAN YANG WAJIB:
- Narasi yang mengalir seperti bercerita, bukan laporan formal
- Bahasa Indonesia yang kasual namun tajam
- Menghubungkan angka dengan strategi bisnis
- Memberikan konteks "kenapa" dibalik angka
- Menggunakan analogi dan metafora yang mudah dipahami
- Tidak ada heading atau bullet points, hanya paragraf yang mengalir
- Sering menggunakan frasa seperti "Kalau dilihat dari angka...", "Masalahnya...", "Kalau dilihat dari tiga sudut pandang..."

DATA YANG TERSEDIA:
{symbol} | Sektor: {company_info.get('sector', 'N/A')} | Harga: {price_str} ({price_change_str})
Market Cap: Rp {metrics.get('market_cap', 0):,.0f} | P/E: {metrics.get('pe_ratio', 'N/A')}
Gross Margin: {profitability.get('gross_margin', 0) or 0} | Net Margin: {profitability.get('net_margin', 0) or 0}%
ROE: {profitability.get('roe', 0) or 0}% | Debt/Equity: {financial_health.get('debt_to_equity', 'N/A')}
Revenue Growth: {growth.get('revenue_growth', 0) or 0}% | Earnings Growth: {growth.get('earnings_growth', 0) or 0}%
Revenue QTerakhir: Rp {latest_qtr.get('revenue', 0) or 0:,.0f} | Net Profit: Rp {latest_qtr.get('net_income', 0) or 0:,.0f}

STRUKTUR TULISAN:
1. Judul dengan format "$SYMBOL [Periode]: [2-3 kata yang mendeskripsikan esensi]"
2. Paragraf pembuka (2-3 kalimat) - deskripsi posisi perusahaan saat ini
3. Paragraf analisis pertumbuhan - cerita revenue, ekspansi, diversifikasi
4. Paragraf analisis profitabilitas - dinamika margin, efisiensi, kompetisi
5. Paragraf analisis keuangan - struktur utang, cash flow, capex
6. Paragraf dengan 3 perspektif - "Dari sisi optimis...", "Dari sudut pandang pesimis...", "Dari sudut pandang realistis..."
7. Paragraf penutup - kesimpulan tajam tentang masa depan perusahaan

Tulis dalam format narasi tunggal tanpa headers. Gunakan transisi yang natural antar paragraf. Berikan insight unik yang tidak terlihat dari data. ML signal {ml_signal} dengan confidence {ml_confidence:.0%} harus menjadi bagian dari analisis jika confidence > 70%.

Contoh kalimat pembuka: "{symbol} sedang berada di fase yang bisa disebut..." atau "Ini adalah cerita klasik perusahaan yang..."""

        return prompt

    def generate_narrative_analysis(self, symbol: str, technical_data: Dict,
                                  financial_data: Dict, ml_data: Dict,
                                  quarterly_data: Dict, growth_data: Dict) -> Dict:
        """Generate full narrative analysis with sections"""

        if not self.llm_adapter:
            return {
                'success': False,
                'error': 'Tidak ada model AI yang tersedia'
            }

        try:
            # Build the prompt
            prompt = self.build_narrative_prompt(
                symbol, technical_data, financial_data,
                ml_data, quarterly_data, growth_data
            )

            # Generate response
            response = self.llm_adapter.generate(
                prompt,
                temperature=0.5,
                max_tokens=3000  # Increased for longer, detailed analysis
            )

            if response:
                narrative_text = response.strip()

                # Try to parse sections
                sections = self._parse_response(narrative_text)

                # Create title if not present
                if 'title' not in sections:
                    title = self._generate_title(symbol, technical_data, financial_data, ml_data)
                    sections['title'] = title

                # Add metadata
                sections['metadata'] = {
                    'symbol': symbol,
                    'model_used': self.model_name,
                    'generated_at': datetime.now().isoformat(),
                    'data_sources': ['yfinance', 'technical_indicators', 'ml_predictions_v5'],
                    'word_count': len(narrative_text.split())
                }

                return {
                    'success': True,
                    'data': sections
                }
            else:
                return {
                    'success': False,
                    'error': 'Tidak ada respons dari model AI'
                }

        except Exception as e:
            logger.error(f"Error generating narrative for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response into structured sections"""
        sections = {}

        # Try to extract title (first line or line with $ symbol)
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('$') or ('Q' in line and '20' in line):
                sections['title'] = line
                break

        # Store full text as content
        sections['content'] = response

        # Try to identify key sections
        content_lower = response.lower()

        # Common section markers
        if 'dari sisi optimis' in content_lower:
            start_idx = content_lower.find('dari sisi optimis')
            sections['optimistic_view'] = response[start_idx:start_idx+500]

        if 'dari sudut pandang pesimis' in content_lower:
            start_idx = content_lower.find('dari sudut pandang pesimis')
            sections['pessimistic_view'] = response[start_idx:start_idx+500]

        if 'dari sudut pandang realistis' in content_lower:
            start_idx = content_lower.find('dari sudut pandang realistis')
            sections['realistic_view'] = response[start_idx:start_idx+500]

        return sections

    def _generate_title(self, symbol: str, technical_data: Dict,
                       financial_data: Dict, ml_data: Dict) -> str:
        """Generate a catchy title for the analysis"""

        # Get key metrics
        price_change = technical_data.get('metrics', {}).get('price_change_5', 0)
        volume_ratio = technical_data.get('stock', {}).get('volume_ratio', 1)
        revenue_growth = financial_data.get('growth', {}).get('revenue_growth', 0)
        ml_signal = ml_data.get('signal', 'HOLD')
        ml_confidence = ml_data.get('confidence', 0)

        # Determine key themes
        themes = []

        if abs(price_change) > 5:
            themes.append("Volatility")
        if revenue_growth > 15:
            themes.append("Growth")
        elif revenue_growth < -5:
            themes.append("Contraction")
        if volume_ratio > 2:
            themes.append("High Volume")
        elif volume_ratio < 0.5:
            themes.append("Low Volume")
        if ml_confidence > 0.7:
            themes.append(f"Strong {ml_signal}")

        # Generate title based on current quarter
        quarter = self._get_current_quarter()

        if len(themes) >= 2:
            title = f"${symbol} {quarter} 2025: {themes[0]} With {themes[1]}"
        elif len(themes) == 1:
            title = f"${symbol} {quarter} 2025: {themes[0]} In Focus"
        else:
            title = f"${symbol} {quarter} 2025: Steady Performance"

        return title

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

        if not self.llm_adapter:
            return {
                'success': False,
                'error': 'Tidak ada model AI yang tersedia'
            }

        try:
            # Extract values to avoid f-string formatting issues
            price_change = technical_data.get('metrics', {}).get('price_change_5', 0)
            price_change_str = f"{price_change:+.2f}%" if price_change is not None else "0.00%"
            volume_ratio = technical_data.get('stock', {}).get('volume_ratio', 0)
            volume_str = f"{volume_ratio:.1f}x" if volume_ratio is not None else "0.0x"
            roe = financial_data.get('profitability', {}).get('roe', 0)
            roe_str = f"{roe:.1f}%" if roe is not None else "0.0%"
            revenue_growth = financial_data.get('growth', {}).get('revenue_growth', 0)
            revenue_str = f"{revenue_growth:.1f}%" if revenue_growth is not None else "0.0%"
            ml_conf = ml_data.get('confidence', 0)
            ml_str = f"{ml_conf:.1%}" if ml_conf is not None else "0.0%"

            prompt = f"""Berdasarkan data saham Indonesia {symbol}, berikan 3 insight utama:

Data Teknikal:
- Harga: {technical_data.get('metrics', {}).get('last_close', 0)}
- Perubahan: {price_change_str}
- Volume: {volume_str} rata-rata

Data Finansial:
- ROE: {roe_str}
- Debt to Equity: {financial_data.get('financial_health', {}).get('debt_to_equity', 'N/A')}
- Revenue Growth: {revenue_str}

Sinyal ML: {ml_data.get('signal', 'N/A')} (confidence: {ml_str})

Berikan 3 insight singkat (maksimal 50 kata per insight):
1. Kinerja terkini
2. Kesehatan finansial
3. Outlook/rekomendasi"""

            response = self.llm_adapter.generate(
                prompt,
                temperature=0.4,
                max_tokens=300
            )

            if response:
                return {
                    'success': True,
                    'insights': response.strip()
                }
            else:
                return {
                    'success': False,
                    'error': 'Tidak ada respons dari model AI'
                }

        except Exception as e:
            logger.error(f"Error generating quick insight for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }