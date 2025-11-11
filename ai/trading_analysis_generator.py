"""
Advanced Trading Analysis Generator
Produces detailed trading analysis reports with entry/exit points, risk management, and strategies
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from .llm_adapter import llm_manager

logger = logging.getLogger(__name__)


class TradingAnalysisGenerator:
    """Generate comprehensive trading analysis reports"""

    def __init__(self):
        self.llm_adapter = llm_manager.get_best_adapter()
        if self.llm_adapter:
            if hasattr(self.llm_adapter, 'model_name'):
                self.model_name = self.llm_adapter.model_name
            else:
                self.model_name = self.llm_adapter.__class__.__name__

    def calculate_technical_levels(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Calculate key technical levels for trading"""
        try:
            # Handle both uppercase and lowercase column names
            df_columns = df.columns.tolist()

            # Standardize column names to lowercase
            df_std = df.copy()

            # Check what columns we have and map them
            column_mapping = {}
            for col in df_columns:
                if col.lower() == 'close' or col == 'Close':
                    column_mapping[col] = 'close'
                elif col.lower() == 'high' or col == 'High':
                    column_mapping[col] = 'high'
                elif col.lower() == 'low' or col == 'Low':
                    column_mapping[col] = 'low'
                elif 'volume' in col.lower():
                    column_mapping[col] = 'volume'

            # Apply the mapping
            if column_mapping:
                df_std = df_std.rename(columns=column_mapping)

            # Check if we have the required columns
            if 'close' not in df_std.columns:
                logger.error(f"No 'close' column found. Available columns: {df_columns}")
                return {}

            # Get latest price data
            current_price = float(df_std['close'].iloc[-1])
            high_20 = float(df_std['high'].tail(20).max()) if 'high' in df_std.columns else current_price * 1.02
            low_20 = float(df_std['low'].tail(20).min()) if 'low' in df_std.columns else current_price * 0.98
            high_50 = float(df_std['high'].tail(50).max()) if 'high' in df_std.columns else current_price * 1.05
            low_50 = float(df_std['low'].tail(50).min()) if 'low' in df_std.columns else current_price * 0.95

            # Calculate EMAs
            df_std['EMA9'] = df_std['close'].ewm(span=9).mean()
            df_std['EMA21'] = df_std['close'].ewm(span=21).mean()
            df_std['EMA50'] = df_std['close'].ewm(span=50).mean()

            ema9 = float(df_std['EMA9'].iloc[-1])
            ema21 = float(df_std['EMA21'].iloc[-1])
            ema50 = float(df_std['EMA50'].iloc[-1])

            # Calculate RSI
            delta = df_std['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = float(100 - (100 / (1 + rs.iloc[-1])))

            # Calculate ATR
            high_low = df_std['high'] - df_std['low']
            high_close = np.abs(df_std['high'] - df_std['close'].shift())
            low_close = np.abs(df_std['low'] - df_std['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = float(true_range.rolling(14).mean().iloc[-1])

            # Volume analysis
            if 'volume' in df_std.columns:
                avg_volume = float(df_std['volume'].tail(20).mean())
                current_volume = float(df_std['volume'].iloc[-1])
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            else:
                avg_volume = 1000000  # Default volume
                current_volume = 1000000
                volume_ratio = 1.0

            # Support and Resistance levels
            supports = [low_50, low_20, ema50, ema21]
            resistances = [high_20, high_50, ema9, ema21]

            # Filter levels near current price
            nearby_supports = sorted([s for s in supports if s < current_price], reverse=True)[:2]
            nearby_resistances = sorted([r for r in resistances if r > current_price])[:2]

            return {
                'current_price': current_price,
                'supports': nearby_supports,
                'resistances': nearby_resistances,
                'ema9': ema9,
                'ema21': ema21,
                'ema50': ema50,
                'rsi': rsi,
                'atr': atr,
                'volume_ratio': volume_ratio,
                'avg_volume': avg_volume,
                'trend': self._determine_trend(df_std),
                'momentum': self._calculate_momentum(df_std)
            }
        except Exception as e:
            logger.error(f"Error calculating technical levels: {e}")
            return {}

    def _determine_trend(self, df: pd.DataFrame) -> str:
        """Determine the current trend"""
        try:
            df['EMA9'] = df['close'].ewm(span=9).mean()
            df['EMA21'] = df['close'].ewm(span=21).mean()
            df['EMA50'] = df['close'].ewm(span=50).mean()

            current = df['close'].iloc[-1]
            ema9 = df['EMA9'].iloc[-1]
            ema21 = df['EMA21'].iloc[-1]
            ema50 = df['EMA50'].iloc[-1]

            if current > ema9 > ema21 > ema50:
                return "UPTREND"
            elif current < ema9 < ema21 < ema50:
                return "DOWNTREND"
            else:
                return "SIDEWAYS"
        except:
            return "UNKNOWN"

    def _calculate_momentum(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum indicators"""
        try:
            # Price momentum
            price_change_5 = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) * 100 if len(df) >= 5 else 0
            price_change_10 = (df['close'].iloc[-1] / df['close'].iloc[-10] - 1) * 100 if len(df) >= 10 else 0

            # Volume momentum
            vol_ma = df['volume'].rolling(20).mean()
            vol_change = (df['volume'].iloc[-1] / vol_ma.iloc[-1] - 1) * 100 if len(vol_ma) >= 20 else 0

            return {
                'price_5d': price_change_5,
                'price_10d': price_change_10,
                'volume_change': vol_change
            }
        except:
            return {'price_5d': 0, 'price_10d': 0, 'volume_change': 0}

    def generate_entry_exit_signals(self, technical: Dict, ml_signal: Dict, financial: Dict) -> Dict[str, Any]:
        """Generate entry and exit signals"""
        current_price = technical.get('current_price', 0)
        atr = technical.get('atr', 0)
        trend = technical.get('trend', 'SIDEWAYS')
        rsi = technical.get('rsi', 50)

        # Determine entry zones
        if trend == "UPTREND" and rsi < 70:
            # Pullback entry
            entry_ideal = technical.get('supports', [current_price * 0.98])[0]
            entry_aggressive = current_price
        elif trend == "DOWNTREND" and rsi > 30:
            # Reversal entry
            entry_ideal = current_price * 0.98
            entry_aggressive = current_price * 0.99
        else:
            # Range trading
            entry_ideal = current_price * 0.98
            entry_aggressive = current_price

        # Calculate TP and SL based on ATR
        tp1 = entry_ideal + (atr * 1.5)
        tp2 = entry_ideal + (atr * 3)
        sl = entry_ideal - (atr * 1.2)

        # Adjust based on support/resistance
        supports = technical.get('supports', [])
        resistances = technical.get('resistances', [])

        if supports:
            sl = max(sl, supports[0] * 0.99)
        if resistances:
            tp1 = min(tp1, resistances[0])
            if len(resistances) > 1:
                tp2 = min(tp2, resistances[1])

        # Risk-reward calculations
        risk_reward_1 = ((tp1 - entry_ideal) / (entry_ideal - sl) * 100) if entry_ideal != sl else 0
        risk_reward_2 = ((tp2 - entry_ideal) / (entry_ideal - sl) * 100) if entry_ideal != sl else 0

        return {
            'entry_ideal': entry_ideal,
            'entry_aggressive': entry_aggressive,
            'tp1': tp1,
            'tp2': tp2,
            'sl': sl,
            'risk_reward_tp1': risk_reward_1,
            'risk_reward_tp2': risk_reward_2,
            'entry_reasoning': self._generate_entry_reasoning(technical, ml_signal),
            'tp_reasoning': self._generate_tp_reasoning(technical, resistances),
            'sl_reasoning': self._generate_sl_reasoning(technical, supports, atr)
        }

    def _generate_entry_reasoning(self, technical: Dict, ml_signal: Dict) -> str:
        """Generate reasoning for entry points"""
        reasons = []

        trend = technical.get('trend', '')
        rsi = technical.get('rsi', 50)
        volume_ratio = technical.get('volume_ratio', 1)

        if trend == "UPTREND":
            reasons.append(f"Uptrend confirmed with EMA alignment")
            if rsi < 70:
                reasons.append(f"RSI {rsi:.1f} shows room for upside")
        elif trend == "DOWNTREND":
            reasons.append(f"Downtrend reversal potential")
            if rsi > 30:
                reasons.append(f"RSI {rsi:.1f} oversold recovery")

        if volume_ratio > 1.5:
            reasons.append(f"Volume spike {volume_ratio:.1f}x confirms interest")

        ml_conf = ml_signal.get('confidence', 0)
        if ml_conf > 0.7:
            reasons.append(f"ML signal {ml_signal.get('signal', 'HOLD')} with {ml_conf:.0%} confidence")

        return "; ".join(reasons) if reasons else "Technical setup forming"

    def _generate_tp_reasoning(self, technical: Dict, resistances: List[float]) -> str:
        """Generate reasoning for take profit levels"""
        reasons = []

        if resistances:
            reasons.append(f"Next resistance at {resistances[0]:.0f}")
            if len(resistances) > 1:
                reasons.append(f"Major resistance at {resistances[1]:.0f}")

        atr = technical.get('atr', 0)
        if atr > 0:
            reasons.append(f"ATR-based targets at 1.5x and 3x ATR")

        return "; ".join(reasons) if reasons else "Based on price structure"

    def _generate_sl_reasoning(self, technical: Dict, supports: List[float], atr: float) -> str:
        """Generate reasoning for stop loss"""
        reasons = []

        if supports:
            reasons.append(f"Strong support at {supports[0]:.0f}")

        if atr > 0:
            reasons.append(f"ATR buffer of 1.2x ({atr * 1.2:.1f} points)")

        return "; ".join(reasons) if reasons else "Volatility-based protection"

    def fetch_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Fetch and analyze news sentiment (mock implementation)"""
        # In a real implementation, this would fetch from news APIs
        # For now, return mock data based on symbol
        return {
            'recent_news': [
                "Q3 2025 earnings beat expectations",
                "New expansion project announced",
                "Analysts upgrade price target"
            ],
            'sentiment_score': 0.7,  # Scale -1 to 1
            'key_catalysts': [
                "Strong earnings growth",
                "Dividend announcement",
                "Sector tailwinds"
            ],
            'risks': [
                "Market volatility",
                "Commodity price fluctuations",
                "Regulatory changes"
            ]
        }

    def generate_comprehensive_analysis(self, symbol: str, df: pd.DataFrame,
                                     technical_analysis: Dict, financial_data: Dict,
                                     ml_signal: Dict) -> Dict[str, Any]:
        """Generate comprehensive trading analysis report"""

        try:
            # Calculate technical levels
            technical_levels = self.calculate_technical_levels(df, symbol)

            # Generate entry/exit signals
            signals = self.generate_entry_exit_signals(technical_levels, ml_signal, financial_data)

            # Fetch news sentiment
            news_sentiment = self.fetch_news_sentiment(symbol)

            # Build comprehensive prompt
            prompt = self._build_comprehensive_prompt(
                symbol, technical_levels, signals, financial_data,
                ml_signal, news_sentiment
            )

            # Generate AI analysis
            ai_analysis = ""
            if self.llm_adapter:
                ai_analysis = self.llm_adapter.generate(
                    prompt,
                    temperature=0.3,
                    max_tokens=4000
                ) or ""

            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(signals, technical_levels)

            # Determine trading strategies
            strategies = self._generate_trading_strategies(
                technical_levels, signals, ml_signal
            )

            return {
                'success': True,
                'data': {
                    'symbol': symbol,
                    'current_price': technical_levels.get('current_price', 0),
                    'technical_levels': technical_levels,
                    'signals': signals,
                    'financial_highlights': self._extract_financial_highlights(financial_data),
                    'ml_signal': ml_signal,
                    'news_sentiment': news_sentiment,
                    'ai_analysis': ai_analysis,
                    'risk_metrics': risk_metrics,
                    'strategies': strategies,
                    'generated_at': datetime.now().isoformat(),
                    'model_used': self.model_name
                }
            }

        except Exception as e:
            logger.error(f"Error generating comprehensive analysis: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _build_comprehensive_prompt(self, symbol: str, technical: Dict,
                                  signals: Dict, financial: Dict, ml_signal: Dict,
                                  news: Dict) -> str:
        """Build comprehensive prompt for AI analysis"""

        current_price = technical.get('current_price', 0)
        entry_ideal = signals.get('entry_ideal', 0)
        tp1 = signals.get('tp1', 0)
        tp2 = signals.get('tp2', 0)
        sl = signals.get('sl', 0)

        # Calculate percentages
        tp1_pct = ((tp1 - entry_ideal) / entry_ideal * 100) if entry_ideal > 0 else 0
        tp2_pct = ((tp2 - entry_ideal) / entry_ideal * 100) if entry_ideal > 0 else 0
        sl_pct = ((sl - entry_ideal) / entry_ideal * 100) if entry_ideal > 0 else 0

        # Get current date
        tanggal_hari_ini = datetime.now().strftime('%d %b %Y')

        prompt = f"""Anda adalah analis trading profesional yang ahli dalam pasar saham Indonesia. Buatlah analisis trading komprehensif untuk {symbol} dengan format seperti berikut:

✅ ENTRY IDEAL: Rp{entry_ideal:,.0f}–Rp{entry_ideal*1.02:,.0f}
(Alasan teknikal singkat: {signals.get('entry_reasoning', '')})

🔰 ENTRY AGRESIF: Rp{entry_ideal*1.02:,.0f}–Rp{entry_ideal*1.05:,.0f}
(Alasan teknikal, risiko, kondisi konfirmasi yang dibutuhkan: breakout confirm, volume requirement, dll)

TAKE PROFIT (TP) & STOP LOSS (SL)

🔥 TAKE PROFIT (TP):
TP1: Rp{tp1:,.0f}
(Alasan singkat: {signals.get('tp_reasoning', '')})

TP2: Rp{tp2:,.0f}
(Alasan singkat: {signals.get('tp_reasoning', '')})

🚫 STOP LOSS (SL):
SL: Rp{sl:,.0f}
(Alasan: {signals.get('sl_reasoning', '')})

📊 RISK-REWARD PERSENTASE
Dari Entry Ideal (Rp{entry_ideal:,.0f}):
TP1: {tp1_pct:+.2f}%
TP2: {tp2_pct:+.2f}%
SL: {sl_pct:+.2f}%

📰 ISU, BERITA & SENTIMEN PASAR
Fundamental & Katalis:
{self._format_financial_highlights(financial)}

Risiko & Sentimen Negatif:
{self._format_risks(news)}

Sentimen Komunitas/Market:
Volume analysis, broker interest, community sentiment

🔥 TEKNIKAL & MOMENTUM (per-timeframe)
Timeframe utama: 15m & Daily
5m: [analisis momentum jangka pendek]
15m: [analisis konfirmasi trend]
Daily: [analisis struktur jangka panjang]

✨ STRATEGI TRADING (Scalping)
Timeframe: 5m/15m
Rules entry: [aturan entry spesifik]
Entry agresif: [kondisi breakout]
Sizing: [ukuran posisi dan manajemen risiko]

✨ STRATEGI TRADING (Swing/Hold)
Timeframe: Daily/4H
[strategi swing trading dengan detail]

📈 MOMENTUM NAIK/TIDAK?
Kesimpulan: [analisis momentum dengan alasan]

SINYAL:
🚀 Sinyal (Scalping / Swing): [sinyal spesifik]
✅ Cocok Untuk: [jenis trading yang cocok]
✨ Confident: [persentase keyakinan]
✨ TIMEFRAME UTAMA: [timeframe rekomendasi]
📊 INDIKATOR UTAMA: [indikator kunci]
📚 RINGKASAN – {tanggal_hari_ini}
[simpulan singkat dengan rekomendasi jelas]

Data Teknikal:
- Harga saat ini: Rp{current_price:,.0f}
- Trend: {technical.get('trend', 'UNKNOWN')}
- RSI: {technical.get('rsi', 50):.1f}
- Volume Ratio: {technical.get('volume_ratio', 1):.1f}x
- ATR: {technical.get('atr', 0):.1f}
- Support terdekat: {technical.get('supports', [])}
- Resistensi terdekat: {technical.get('resistances', [])}
- ML Signal: {ml_signal.get('signal', 'HOLD')} ({ml_signal.get('confidence', 0):.0%} confidence)

Buat analisis yang tajam, insightfull, dan actionable dengan bahasa Indonesia yang profesional namun mudah dipahami trader.""".format(
            entry_ideal=entry_ideal,
            tp1=tp1,
            tp2=tp2,
            sl=sl,
            tp1_pct=tp1_pct,
            tp2_pct=tp2_pct,
            sl_pct=sl_pct,
            tanggal_hari_ini=tanggal_hari_ini,
            current_price=current_price,
            technical=technical,
            ml_signal=ml_signal,
            signals=signals
        )

        return prompt

    def _format_financial_highlights(self, financial: Dict) -> str:
        """Format financial highlights"""
        highlights = []

        profitability = financial.get('profitability', {})
        if profitability.get('revenue_growth'):
            highlights.append(f"Revenue growth {profitability['revenue_growth']:.1f}%")
        if profitability.get('net_margin'):
            highlights.append(f"Net margin {profitability['net_margin']:.1f}%")
        if profitability.get('roe'):
            highlights.append(f"ROE {profitability['roe']:.1f}%")

        financial_health = financial.get('financial_health', {})
        if financial_health.get('debt_to_equity'):
            highlights.append(f"Debt/Equity {financial_health['debt_to_equity']}")

        return "; ".join(highlights) if highlights else "Data finansial tersedia"

    def _format_risks(self, news: Dict) -> str:
        """Format risk factors"""
        risks = news.get('risks', [])
        return "; ".join(risks[:3]) if risks else "Market risks apply"

    def _extract_financial_highlights(self, financial: Dict) -> Dict[str, Any]:
        """Extract key financial highlights"""
        return {
            'revenue_growth': financial.get('growth', {}).get('revenue_growth', 0),
            'net_margin': financial.get('profitability', {}).get('net_margin', 0),
            'roe': financial.get('profitability', {}).get('roe', 0),
            'debt_to_equity': financial.get('financial_health', {}).get('debt_to_equity', 'N/A'),
            'market_cap': financial.get('market_data', {}).get('market_cap', 0)
        }

    def _calculate_risk_metrics(self, signals: Dict, technical: Dict) -> Dict[str, Any]:
        """Calculate risk metrics"""
        entry = signals.get('entry_ideal', 0)
        tp1 = signals.get('tp1', 0)
        sl = signals.get('sl', 0)

        if entry > 0:
            max_loss = abs(entry - sl) / entry * 100
            max_gain = abs(tp1 - entry) / entry * 100
            risk_reward = max_gain / max_loss if max_loss > 0 else 0
        else:
            max_loss = max_gain = risk_reward = 0

        return {
            'max_loss_percent': max_loss,
            'max_gain_percent': max_gain,
            'risk_reward_ratio': risk_reward,
            'atr': technical.get('atr', 0),
            'volatility': 'High' if technical.get('atr', 0) > entry * 0.03 else 'Medium' if technical.get('atr', 0) > entry * 0.015 else 'Low'
        }

    def _generate_trading_strategies(self, technical: Dict, signals: Dict, ml_signal: Dict) -> Dict[str, Any]:
        """Generate trading strategies"""
        trend = technical.get('trend', 'SIDEWAYS')
        volume_ratio = technical.get('volume_ratio', 1)
        rsi = technical.get('rsi', 50)

        # Determine suitability
        if trend == "UPTREND" and rsi < 70 and volume_ratio > 1.2:
            scalping_suitability = "HIGHLY SUITABLE"
            swing_suitability = "SUITABLE"
        elif trend == "SIDEWAYS" and 30 < rsi < 70:
            scalping_suitability = "SUITABLE"
            swing_suitability = "MODERATE"
        else:
            scalping_suitability = "CAUTION"
            swing_suitability = "CAUTION"

        return {
            'scalping': {
                'suitability': scalping_suitability,
                'timeframe': '5m/15m',
                'entry_rules': f"Pullback to {signals.get('entry_ideal', 0):.0f} with volume confirmation",
                'exit_rules': f"Take profit at {signals.get('tp1', 0):.0f} or {signals.get('tp2', 0):.0f}",
                'position_sizing': 'Moderate risk, scaling with momentum'
            },
            'swing': {
                'suitability': swing_suitability,
                'timeframe': 'Daily/4H',
                'entry_rules': f"Breakout confirmation above {signals.get('entry_aggressive', 0):.0f}",
                'exit_rules': f"Trail stop with ATR, target {signals.get('tp2', 0):.0f}",
                'position_sizing': 'Core position with optional scaling'
            },
            'ml_integration': {
                'signal': ml_signal.get('signal', 'HOLD'),
                'confidence': ml_signal.get('confidence', 0),
                'action': 'Follow ML signal if confidence > 70%'
            }
        }


# Global instance
trading_analysis_generator = TradingAnalysisGenerator()