"""
RsiMeanReversionStrategy
========================
Mean-reversion strategy for Bybit spot, adapted for small stake sizes
($10-20 per trade) and backtest-honest sentiment filtering.

Logic (LONG only, spot):
  - RSI(14) on 15m below `rsi_buy` (oversold)
  - Price below lower Bollinger Band (BB(20, bb_std))
  - Higher timeframe (4h) trend not strongly bearish:
        4h EMA50 > EMA200 OR |close/ema200 - 1| < 5%
  - Volume above `volume_factor` * 20-bar volume SMA
  - Macro sentiment (Fear&Greed) >= `min_macro_score`
        Loaded from cached CSV in bot_start() and merged onto every candle,
        so this filter works in BACKTEST as well as live.

Exit:
  - RSI(14) > `rsi_sell` AND close > BB middle
  - OR ROI table
  - OR stoploss / trailing stoploss

Sentiment gate (live/dry-run only):
  - In confirm_trade_entry, also queries CryptoPanic news component.
  - News has no free historical API, so backtest cannot replay it.
    Backtest still respects the macro F&G filter via populate_indicators,
    which is the dominant component of the live blend.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import (
    DecimalParameter,
    IntParameter,
    IStrategy,
    informative,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_providers.sentiment_provider import SentimentProvider  # noqa: E402

logger = logging.getLogger(__name__)


class RsiMeanReversionStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short = False

    minimal_roi = {
        "0": 0.025,
        "30": 0.015,
        "90": 0.008,
        "240": 0.0,
    }
    stoploss = -0.05

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False

    startup_candle_count: int = 400

    # ---- Protections (moved from config; config-side is deprecated) ----
    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 5},
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 200,
                "trade_limit": 10,
                "stop_duration_candles": 50,
                "max_allowed_drawdown": 0.10,
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 100,
                "trade_limit": 3,
                "stop_duration_candles": 50,
                "only_per_pair": False,
            },
        ]

    # ---- Hyperoptable parameters ----
    rsi_buy = IntParameter(15, 35, default=25, space="buy")
    rsi_sell = IntParameter(55, 80, default=65, space="sell")
    bb_std = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, space="buy")
    volume_factor = DecimalParameter(1.0, 3.0, default=1.2, decimals=1, space="buy")
    min_macro_score = DecimalParameter(-1.0, 0.5, default=-0.3, decimals=1, space="buy")
    min_sentiment_score = DecimalParameter(-1.0, 0.5, default=-0.3, decimals=1, space="buy")

    # ---- Sentiment (lazy init) ----
    _sentiment: Optional[SentimentProvider] = None
    _fng_history: Optional[pd.DataFrame] = None

    FNG_CACHE_PATH = Path("user_data/data/sentiment/fng_history.csv")

    def bot_start(self, **kwargs) -> None:
        token = os.getenv("CRYPTOPANIC_TOKEN")
        self._sentiment = SentimentProvider(cryptopanic_token=token)
        if token:
            logger.info("Sentiment provider enabled (CryptoPanic + Fear&Greed).")
        else:
            logger.info(
                "Sentiment provider in macro-only mode "
                "(set CRYPTOPANIC_TOKEN env var to enable news component)."
            )
        self._load_fng_history()

    def bot_loop_start(self, **kwargs) -> None:
        if self._fng_history is None:
            self._load_fng_history()

    def _load_fng_history(self) -> None:
        try:
            provider = self._sentiment or SentimentProvider()
            self._fng_history = provider.load_or_fetch_historical_fng(self.FNG_CACHE_PATH)
            logger.info(
                "Loaded %d days of Fear&Greed history for backtest filter.",
                len(self._fng_history),
            )
        except Exception as e:
            logger.warning(
                "Could not load historical Fear&Greed (%s). "
                "Backtest will treat macro filter as neutral.",
                e,
            )
            self._fng_history = None

    # ---- 4h trend filter via @informative decorator ----
    @informative("4h")
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["trend_ok"] = (
            (dataframe["ema50"] > dataframe["ema200"])
            | ((dataframe["close"] / dataframe["ema200"] - 1).abs() < 0.05)
        ).astype(int)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        upper, middle, lower = ta.BBANDS(
            dataframe["close"],
            timeperiod=20,
            nbdevup=self.bb_std.value,
            nbdevdn=self.bb_std.value,
            matype=0,
        )
        dataframe["bb_upper"] = upper
        dataframe["bb_middle"] = middle
        dataframe["bb_lower"] = lower

        dataframe["volume_sma20"] = dataframe["volume"].rolling(20).mean()
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        dataframe["fng_score"] = self._map_fng(dataframe)

        return dataframe

    def _map_fng(self, dataframe: DataFrame) -> pd.Series:
        """Map daily F&G index onto each 15m candle by date. 0.0 if no history."""
        if self._fng_history is None or self._fng_history.empty:
            return pd.Series(0.0, index=dataframe.index)

        candles = dataframe[["date"]].copy()
        candles["candle_day"] = pd.to_datetime(candles["date"]).dt.tz_convert("UTC").dt.normalize()

        fng = self._fng_history.copy()
        if fng.index.tz is None:
            fng.index = fng.index.tz_localize("UTC")
        else:
            fng.index = fng.index.tz_convert("UTC")

        merged = candles.merge(
            fng[["fng_score"]],
            left_on="candle_day",
            right_index=True,
            how="left",
        )
        return merged["fng_score"].ffill().fillna(0.0).to_numpy()

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        cond = (
            (dataframe["rsi"] < self.rsi_buy.value)
            & (dataframe["close"] < dataframe["bb_lower"])
            & (dataframe["volume"] > dataframe["volume_sma20"] * self.volume_factor.value)
            & (dataframe["trend_ok_4h"] == 1)
            & (dataframe["fng_score"] >= self.min_macro_score.value)
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[cond, ["enter_long", "enter_tag"]] = (1, "rsi_bb_oversold")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        cond = (
            (dataframe["rsi"] > self.rsi_sell.value)
            & (dataframe["close"] > dataframe["bb_middle"])
        )
        dataframe.loc[cond, ["exit_long", "exit_tag"]] = (1, "rsi_meanrev_done")
        return dataframe

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        """Live-only blended sentiment gate. Backtest skips this hook."""
        if self._sentiment is None:
            return True

        try:
            sent = self._sentiment.get_score(pair)
        except Exception as e:
            logger.warning("Sentiment lookup failed for %s, allowing trade: %s", pair, e)
            return True

        threshold = self.min_sentiment_score.value
        if sent.score < threshold:
            logger.info(
                "BLOCKED %s entry: sentiment=%.2f (news=%.2f, macro=%.2f) below %.2f",
                pair, sent.score, sent.news_component, sent.macro_component, threshold,
            )
            return False

        logger.info(
            "ALLOWED %s entry: sentiment=%.2f (news=%.2f, macro=%.2f)",
            pair, sent.score, sent.news_component, sent.macro_component,
        )
        return True
