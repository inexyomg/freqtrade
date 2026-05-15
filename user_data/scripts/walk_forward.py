"""
Walk-forward validation runner.

Cycles through the timerange with a fixed-size train window and a smaller
out-of-sample (OOS) test window. For each cycle:
  1. Run hyperopt on train window (writes best params to user_data/hyperopt_results).
  2. Run backtesting on OOS window using those params.
  3. Collect per-window OOS profit/drawdown into a summary.

Aggregated OOS metrics are the honest measure of strategy quality. If your
in-sample hyperopt looks great but OOS averages near zero or negative, the
strategy is overfit -- adjust complexity or feature set, not parameters.

Usage example:
    python user_data/scripts/walk_forward.py \\
        --config user_data/config_bybit_spot.json \\
        --strategy RsiMeanReversionStrategy \\
        --start 20240101 --end 20250501 \\
        --train-days 120 --test-days 30 \\
        --epochs 200 --hyperopt-loss SortinoHyperOptLoss

Run it after `freqtrade download-data` is done for the full range.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger("walk_forward")


def daterange(start: datetime, end: datetime, train_days: int, test_days: int):
    """Yield (train_start, train_end, test_start, test_end) windows."""
    cur = start
    step = timedelta(days=test_days)
    while True:
        train_start = cur
        train_end = cur + timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + timedelta(days=test_days)
        if test_end > end:
            return
        yield train_start, train_end, test_start, test_end
        cur += step


def fmt(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    logger.info("$ %s", " ".join(cmd))
    return subprocess.run(cmd, check=False)  # noqa: S603


def parse_last_backtest_summary(results_dir: Path) -> dict:
    """Read the most recent backtest result JSON and pull out headline metrics."""
    candidates = sorted(results_dir.glob("backtest-result-*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return {}
    try:
        data = json.loads(candidates[-1].read_text())
    except Exception as e:
        logger.warning("Failed to parse %s: %s", candidates[-1], e)
        return {}
    strat = next(iter(data.get("strategy", {}).values()), {})
    return {
        "file": candidates[-1].name,
        "total_profit_pct": strat.get("profit_total_pct"),
        "trades": strat.get("total_trades"),
        "max_drawdown_pct": strat.get("max_drawdown_account"),
        "sharpe": strat.get("sharpe"),
        "sortino": strat.get("sortino"),
        "win_rate": (strat.get("wins", 0) / strat["total_trades"]) if strat.get("total_trades") else None,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--strategy", required=True)
    p.add_argument("--start", required=True, help="YYYYMMDD")
    p.add_argument("--end", required=True, help="YYYYMMDD")
    p.add_argument("--train-days", type=int, default=120)
    p.add_argument("--test-days", type=int, default=30)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hyperopt-loss", default="SortinoHyperOptLoss")
    p.add_argument("--spaces", default="buy sell")
    p.add_argument("--results-dir", default="user_data/backtest_results")
    p.add_argument("--summary-out", default="user_data/backtest_results/walk_forward_summary.json")
    p.add_argument("--skip-hyperopt", action="store_true", help="Use existing params (faster smoke test)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    start = datetime.strptime(args.start, "%Y%m%d")
    end = datetime.strptime(args.end, "%Y%m%d")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(
        daterange(start, end, args.train_days, args.test_days), start=1
    ):
        logger.info(
            "=== Window %d ===  TRAIN %s-%s   TEST %s-%s",
            i, fmt(tr_s), fmt(tr_e), fmt(te_s), fmt(te_e),
        )

        if not args.skip_hyperopt:
            run([
                "freqtrade", "hyperopt",
                "--config", args.config,
                "--strategy", args.strategy,
                "--hyperopt-loss", args.hyperopt_loss,
                "--spaces", *args.spaces.split(),
                "--timerange", f"{fmt(tr_s)}-{fmt(tr_e)}",
                "--epochs", str(args.epochs),
            ])

        run([
            "freqtrade", "backtesting",
            "--config", args.config,
            "--strategy", args.strategy,
            "--timerange", f"{fmt(te_s)}-{fmt(te_e)}",
            "--enable-protections",
        ])

        metrics = parse_last_backtest_summary(results_dir)
        metrics["window"] = i
        metrics["train"] = f"{fmt(tr_s)}-{fmt(tr_e)}"
        metrics["test"] = f"{fmt(te_s)}-{fmt(te_e)}"
        summary.append(metrics)
        logger.info("OOS metrics: %s", metrics)

    out = Path(args.summary_out)
    out.write_text(json.dumps(summary, indent=2, default=str))

    profits = [s.get("total_profit_pct") for s in summary if s.get("total_profit_pct") is not None]
    if profits:
        avg = sum(profits) / len(profits)
        logger.info("Avg OOS profit across %d windows: %.2f%%", len(profits), avg)
        positive = sum(1 for x in profits if x > 0)
        logger.info("Profitable windows: %d / %d", positive, len(profits))
    logger.info("Summary -> %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
