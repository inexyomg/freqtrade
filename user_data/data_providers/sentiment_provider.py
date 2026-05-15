"""
Sentiment Provider for Freqtrade strategies.

Aggregates two free signals into a single normalized score in [-1.0, +1.0]:
  - CryptoPanic news votes (per coin)         -> short-term news mood (live only)
  - alternative.me Fear & Greed Index (market) -> macro market sentiment

Two modes:
  1. Live / dry-run: real-time score via get_score(pair). Used in confirm_trade_entry.
  2. Backtest: historical Fear&Greed series via load_or_fetch_historical_fng().
     Returned as a pandas DataFrame indexed by date, ready to merge into candles.
     CryptoPanic has no free historical endpoint, so news component is omitted
     from backtest by design.

Designed to be safe in live trading:
  - All HTTP calls are wrapped in try/except. On any failure -> neutral 0.0.
  - Results are cached in-memory with a TTL so we do not hammer APIs.
  - No hard dependency on Freqtrade -- can be unit-tested standalone.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


@dataclass
class SentimentScore:
    """Aggregated sentiment for a single base coin."""

    coin: str
    score: float
    news_component: float
    macro_component: float
    ts: float


class SentimentProvider:
    """Sentiment provider. One instance per strategy, reused across pairs."""

    CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/"
    FNG_URL = "https://api.alternative.me/fng/"

    def __init__(
        self,
        cryptopanic_token: Optional[str] = None,
        news_cache_ttl: int = 600,
        macro_cache_ttl: int = 3600,
        http_timeout: int = 5,
    ):
        self.cryptopanic_token = cryptopanic_token
        self.news_cache_ttl = news_cache_ttl
        self.macro_cache_ttl = macro_cache_ttl
        self.http_timeout = http_timeout

        self._news_cache: dict[str, tuple[float, float]] = {}
        self._macro_cache: Optional[tuple[float, float]] = None

    # ---------- Public API: live ----------

    def get_score(self, pair: str) -> SentimentScore:
        """Aggregated sentiment for a pair like 'BTC/USDT'. Neutral on failure."""
        coin = pair.split("/")[0].upper()

        news = self._get_news_sentiment(coin)
        macro = self._get_macro_sentiment()

        combined = 0.6 * news + 0.4 * macro
        combined = max(-1.0, min(1.0, combined))

        return SentimentScore(
            coin=coin,
            score=combined,
            news_component=news,
            macro_component=macro,
            ts=time.time(),
        )

    # ---------- Public API: historical (backtest) ----------

    def load_or_fetch_historical_fng(
        self,
        cache_path: Path,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Returns historical Fear & Greed values as a DataFrame indexed by date.

        Columns:
          - fng_raw:   0..100 (original index value)
          - fng_score: -1..+1 (normalized: (raw - 50) / 50)

        Persists CSV cache at cache_path. Refetches whenever the cache is
        missing, malformed, or older than 24h (or force_refresh=True).
        """
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists() and not force_refresh:
            age = time.time() - cache_path.stat().st_mtime
            if age < 24 * 3600:
                try:
                    df = pd.read_csv(cache_path, parse_dates=["date"]).set_index("date")
                    if {"fng_raw", "fng_score"}.issubset(df.columns) and len(df) > 30:
                        return df.sort_index()
                except Exception as e:
                    logger.warning("F&G cache unreadable, refetching: %s", e)

        df = self._fetch_historical_fng()
        df.to_csv(cache_path, index_label="date")
        logger.info("Fetched %d days of Fear&Greed history -> %s", len(df), cache_path)
        return df

    def _fetch_historical_fng(self) -> pd.DataFrame:
        r = requests.get(
            self.FNG_URL,
            params={"limit": 0, "format": "json"},
            timeout=self.http_timeout * 4,
        )
        r.raise_for_status()
        rows = r.json().get("data", [])
        if not rows:
            raise RuntimeError("alternative.me returned empty F&G history")

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
        df["fng_raw"] = df["value"].astype(int)
        df["fng_score"] = ((df["fng_raw"] - 50) / 50.0).clip(-1.0, 1.0)
        return df[["date", "fng_raw", "fng_score"]].set_index("date").sort_index()

    # ---------- CryptoPanic ----------

    def _get_news_sentiment(self, coin: str) -> float:
        cached = self._news_cache.get(coin)
        if cached and (time.time() - cached[1]) < self.news_cache_ttl:
            return cached[0]

        if not self.cryptopanic_token:
            return 0.0

        try:
            params = {
                "auth_token": self.cryptopanic_token,
                "currencies": coin,
                "kind": "news",
                "public": "true",
                "filter": "hot",
            }
            r = requests.get(self.CRYPTOPANIC_URL, params=params, timeout=self.http_timeout)
            r.raise_for_status()
            posts = r.json().get("results", [])
        except Exception as e:
            logger.warning("CryptoPanic fetch failed for %s: %s", coin, e)
            return 0.0

        if not posts:
            self._news_cache[coin] = (0.0, time.time())
            return 0.0

        bull, bear, total = 0, 0, 0
        for p in posts[:20]:
            votes = p.get("votes", {}) or {}
            b = votes.get("positive", 0) + votes.get("lol", 0)
            s = votes.get("negative", 0) + votes.get("toxic", 0)
            bull += b
            bear += s
            total += b + s

        if total == 0:
            score = 0.0
        else:
            score = (bull - bear) / total
            score *= min(1.0, total / 20.0)

        score = max(-1.0, min(1.0, score))
        self._news_cache[coin] = (score, time.time())
        return score

    # ---------- Fear & Greed (live) ----------

    def _get_macro_sentiment(self) -> float:
        if self._macro_cache and (time.time() - self._macro_cache[1]) < self.macro_cache_ttl:
            return self._macro_cache[0]

        try:
            r = requests.get(self.FNG_URL, timeout=self.http_timeout)
            r.raise_for_status()
            data = r.json().get("data", [])
            if not data:
                return 0.0
            value = int(data[0]["value"])
        except Exception as e:
            logger.warning("Fear & Greed fetch failed: %s", e)
            return 0.0

        score = (value - 50) / 50.0
        score = max(-1.0, min(1.0, score))
        self._macro_cache = (score, time.time())
        return score


if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO)
    token = os.getenv("CRYPTOPANIC_TOKEN")
    provider = SentimentProvider(cryptopanic_token=token)

    print("Live scores:")
    for pair in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
        s = provider.get_score(pair)
        print(f"  {pair}: total={s.score:+.2f}  news={s.news_component:+.2f}  macro={s.macro_component:+.2f}")

    print("\nFetching historical F&G...")
    hist = provider.load_or_fetch_historical_fng(Path("user_data/data/sentiment/fng_history.csv"))
    print(hist.tail(5))
