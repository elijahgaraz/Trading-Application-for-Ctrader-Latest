from abc import ABC, abstractmethod
from typing import Any, Dict, TYPE_CHECKING, Union
import pandas as pd
from datetime import time, datetime
from zoneinfo import ZoneInfo  # Python 3.9+

from indicators import calculate_ema, calculate_atr, calculate_rsi, calculate_adx

if TYPE_CHECKING:
    from trading import Trader


class Strategy(ABC):
    """Abstract base class for trading strategies."""
    NAME: str = "Base Strategy"

    @abstractmethod
    def decide(self, symbol: str, data: Dict[str, Any], trader: "Trader") -> Dict[str, Any]:
        return {
            'action': 'hold',
            'comment': f'{self.NAME} not implemented',
            'sl_offset': None,
            'tp_offset': None
        }

    @abstractmethod
    def get_required_bars(self) -> Dict[str, int]:
        """Returns a dict of {'timeframe_str': count} required by the strategy."""
        return {}


class SafeStrategy(Strategy):
    """
    Enhanced Safe (Low-Risk) Trend-Following Scalper with configurable indicators and filters.
    """
    NAME = "Safe (Low-Risk) Trend-Following Scalper"

    def __init__(self, settings):
        self.settings = settings
        # Trailing stop state is instance-specific, not part of saveable settings
        self.trailing_activated = False

    def get_required_bars(self) -> Dict[str, int]:
        # Determine required bars based on the longest period needed by indicators
        strat_settings = self.settings.strategy
        required_bars = max(
            strat_settings.ema_period,
            strat_settings.atr_period,
            strat_settings.adx_period,
            strat_settings.rsi_period
        )
        return {'1m': max(required_bars, self.settings.general.min_bars_for_trading)}

    def _hold(self, reason: str) -> Dict[str, Any]:
        return {'action': 'hold', 'comment': f"{self.NAME}: {reason}", 'sl_offset': None, 'tp_offset': None}

    def decide(self, symbol: str, data: Dict[str, Any], trader: "Trader") -> Dict[str, Any]:
        df: pd.DataFrame = data.get('ohlc_1m')
        if df is None or len(df) < self.get_required_bars()['1m']:
            return self._hold("insufficient data")

        # --- Settings ---
        strat_settings = self.settings.strategy

        # --- Indicator Calculations ---
        close = df['close']
        price = close.iloc[-1]
        pip_factor = 10000

        # Base strategy indicators
        ema_long = calculate_ema(df, strat_settings.ema_period).iloc[-1]
        atr = calculate_atr(df, strat_settings.atr_period).iloc[-1]

        # Filter indicators
        adx_series = calculate_adx(df, strat_settings.adx_period)[f'ADX_{strat_settings.adx_period}']
        adx = adx_series.iloc[-1] if not adx_series.empty else 0
        rsi = calculate_rsi(df, strat_settings.rsi_period).iloc[-1]

        # --- Strategy Filters ---

        # 1. Buffer Zone Filter
        buffer = atr * strat_settings.buffer_mult
        if abs(price - ema_long) < buffer:
            return self._hold("within buffer zone")

        # 2. ADX Trend Strength Filter
        if adx < strat_settings.adx_threshold:
            return self._hold(f"ADX {adx:.2f} < {strat_settings.adx_threshold}, avoiding ranging market")

        # 3. RSI Momentum Filter & Direction
        if price > ema_long and rsi < strat_settings.rsi_overbought:
            action = 'buy'
            comment = f"price {price:.5f} > EMA & RSI {rsi:.2f} < {strat_settings.rsi_overbought}"
        elif price < ema_long and rsi > strat_settings.rsi_oversold:
            action = 'sell'
            comment = f"price {price:.5f} < EMA & RSI {rsi:.2f} > {strat_settings.rsi_oversold}"
        else:
            return self._hold(f"RSI out of bounds ({rsi:.2f})")

        # --- AI Overseer Confirmation ---
        if trader.settings.ai.use_ai_overseer:
            # AI snapshot uses fixed short-term indicators, not the configurable ones
            ema_fast_ai = calculate_ema(df, 9).iloc[-1]
            ema_slow_ai = calculate_ema(df, 21).iloc[-1]
            atr_pips_ai = atr * pip_factor

            snapshot = {
                "symbol": symbol.replace("/", ""),
                "bot_intent": action.upper(),
                "timeframe": "m1",
                "price": price,
                "spread_pips": 0.5,
                "ema_fast": ema_fast_ai,
                "ema_slow": ema_slow_ai,
                "rsi": rsi,
                "adx": adx,
                "atr_pips": atr_pips_ai
            }
            ai_advice = trader.get_ai_advice(snapshot)

            if ai_advice:
                ai_action_map = {'BUY': 'buy', 'SELL': 'sell'}
                if ai_advice.confidence < trader.settings.ai.advisor_min_confidence:
                    return self._hold(f"AI conf {ai_advice.confidence:.2%} below threshold. Reason: {ai_advice.reason}")
                if ai_action_map.get(ai_advice.direction) != action:
                    return self._hold(f"AI action '{ai_advice.direction}' contradicts strategy '{action}'. Reason: {ai_advice.reason}")
                comment += f" | AI Confirmed (Conf: {ai_advice.confidence:.2%})"
            else:
                return self._hold("AI advisor failed to provide a valid response.")

        # --- Stop-Loss and Take-Profit Calculation ---
        sl_pips = atr * strat_settings.stop_mult * pip_factor
        tp_pips = atr * strat_settings.target_mult * pip_factor

        # Trailing stop logic
        if not self.trailing_activated and (
            (action == 'buy' and price > ema_long + 2 * buffer) or
            (action == 'sell' and price < ema_long - 2 * buffer)
        ):
            self.trailing_activated = True
            comment += "; trailing stop activated"

        return {
            'action': action,
            'comment': f"{self.NAME}: {comment}",
            'sl_offset': sl_pips,
            'tp_offset': tp_pips,
            'risk_percentage': self.settings.general.risk_percentage
        }


