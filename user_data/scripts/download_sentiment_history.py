"""
One-off helper: fetch all available Fear&Greed history and cache it for
backtests. Run this once before your first backtest, and re-run periodically
(e.g. weekly) to refresh.

Usage:
    python user_data/scripts/download_sentiment_history.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "user_data"))

from data_providers.sentiment_provider import SentimentProvider  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cache_path = REPO_ROOT / "user_data" / "data" / "sentiment" / "fng_history.csv"
    provider = SentimentProvider()
    df = provider.load_or_fetch_historical_fng(cache_path, force_refresh=True)
    print(f"Saved {len(df)} rows to {cache_path}")
    print(df.tail(10))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
