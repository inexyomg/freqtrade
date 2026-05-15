"""
Streamlit dashboard for the RSI Mean Reversion strategy.

Run from the freqtrade repo root:

    .venv/bin/streamlit run user_data/scripts/dashboard.py

Has three tabs:
  1. Chart & signals — load downloaded OHLCV, overlay BB/RSI, mark entry/exit
                       candles for the parameter values you pick in the sidebar.
  2. Sentiment      — historical Fear&Greed chart + live per-coin scoring
                       (CryptoPanic+F&G blend, requires CRYPTOPANIC_TOKEN).
  3. Run freqtrade  — buttons that shell out to download-data / lookahead /
                       backtesting / hyperopt and stream stdout into the page.

The dashboard computes indicators independently of the actual strategy file
so you can scrub parameters and see signals instantly. The committed strategy
in strategies/RsiMeanReversionStrategy.py is the source of truth for live
trading; this tool is for exploration.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import talib.abstract as ta
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "user_data"))

from data_providers.sentiment_provider import SentimentProvider  # noqa: E402

DATA_ROOT = REPO_ROOT / "user_data" / "data"
CONFIG_PATH = REPO_ROOT / "user_data" / "config_bybit_spot.json"
STRATEGY_NAME = "RsiMeanReversionStrategy"
FNG_CACHE = REPO_ROOT / "user_data" / "data" / "sentiment" / "fng_history.csv"

st.set_page_config(page_title="RSI MeanRev Dashboard", layout="wide")


# ---------- helpers ----------


def list_pairs(exchange: str, timeframe: str) -> list[str]:
    folder = DATA_ROOT / exchange
    if not folder.exists():
        return []
    pairs = []
    suffix = f"-{timeframe}"
    for f in folder.iterdir():
        name = f.name
        for ext in (".feather", ".json", ".json.gz", ".parquet"):
            if name.endswith(suffix + ext):
                stem = name[: -(len(suffix) + len(ext))]
                pairs.append(stem.replace("_", "/"))
                break
    return sorted(set(pairs))


def load_pair(exchange: str, pair: str, timeframe: str) -> pd.DataFrame | None:
    base = DATA_ROOT / exchange / f"{pair.replace('/', '_')}-{timeframe}"
    for ext in (".feather", ".parquet", ".json", ".json.gz"):
        path = base.with_name(base.name + ext)
        if not path.exists():
            continue
        try:
            if ext == ".feather":
                df = pd.read_feather(path)
            elif ext == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_json(path)
                df.columns = ["date", "open", "high", "low", "close", "volume"]
                df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
        except Exception as e:
            st.error(f"Не смог прочитать {path}: {e}")
            return None
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df = df.sort_values("date").reset_index(drop=True)
        return df
    return None


def compute_indicators(
    df: pd.DataFrame,
    bb_std: float,
    fng_df: pd.DataFrame | None,
) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = ta.RSI(df, timeperiod=14)
    upper, middle, lower = ta.BBANDS(
        df["close"], timeperiod=20, nbdevup=bb_std, nbdevdn=bb_std, matype=0
    )
    df["bb_upper"] = upper
    df["bb_middle"] = middle
    df["bb_lower"] = lower
    df["volume_sma20"] = df["volume"].rolling(20).mean()

    if fng_df is not None and not fng_df.empty:
        candle_day = pd.to_datetime(df["date"]).dt.tz_convert("UTC").dt.normalize()
        fng = fng_df.copy()
        if fng.index.tz is None:
            fng.index = fng.index.tz_localize("UTC")
        else:
            fng.index = fng.index.tz_convert("UTC")
        m = pd.DataFrame({"candle_day": candle_day}).merge(
            fng[["fng_score"]], left_on="candle_day", right_index=True, how="left"
        )
        df["fng_score"] = m["fng_score"].ffill().fillna(0.0).to_numpy()
    else:
        df["fng_score"] = 0.0
    return df


def compute_signals(
    df: pd.DataFrame,
    rsi_buy: int,
    rsi_sell: int,
    volume_factor: float,
    min_macro: float,
) -> pd.DataFrame:
    df = df.copy()
    df["enter_long"] = (
        (df["rsi"] < rsi_buy)
        & (df["close"] < df["bb_lower"])
        & (df["volume"] > df["volume_sma20"] * volume_factor)
        & (df["fng_score"] >= min_macro)
        & (df["volume"] > 0)
    )
    df["exit_long"] = (df["rsi"] > rsi_sell) & (df["close"] > df["bb_middle"])
    return df


def load_fng_history() -> pd.DataFrame | None:
    if not FNG_CACHE.exists():
        return None
    try:
        df = pd.read_csv(FNG_CACHE, parse_dates=["date"]).set_index("date").sort_index()
        return df
    except Exception:
        return None


def stream_command(cmd: list[str], placeholder) -> int:
    """Run cmd, stream stdout into the placeholder as it arrives."""
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line.rstrip())
        tail = "\n".join(lines[-200:])
        placeholder.code(tail, language="bash")
    proc.wait()
    return proc.returncode


# ---------- sidebar ----------

st.sidebar.title("Параметры")
exchange = st.sidebar.selectbox("Exchange", ["bybit", "binance", "okx"], index=0)
timeframe = st.sidebar.selectbox("Таймфрейм", ["15m", "5m", "1h", "4h"], index=0)

pairs = list_pairs(exchange, timeframe)
pair = st.sidebar.selectbox(
    "Пара",
    pairs or ["— нет данных, скачай во вкладке Run —"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption("Параметры стратегии")
rsi_buy = st.sidebar.slider("RSI buy <", 10, 40, 25)
rsi_sell = st.sidebar.slider("RSI sell >", 50, 85, 65)
bb_std = st.sidebar.slider("BB std", 1.0, 3.5, 2.0, 0.1)
volume_factor = st.sidebar.slider("Volume × SMA20", 0.5, 3.0, 1.2, 0.1)
min_macro = st.sidebar.slider("Min Fear&Greed score", -1.0, 0.5, -0.3, 0.05)

st.sidebar.markdown("---")
n_show = st.sidebar.number_input("Последние N свечей", 200, 20000, 2000, step=500)


# ---------- main tabs ----------

tab_chart, tab_sent, tab_run = st.tabs(["📈 График и сигналы", "🌡️ Sentiment", "⚙️ Run freqtrade"])


# ===== Tab 1: chart =====
with tab_chart:
    st.subheader(f"{pair} · {timeframe} · {exchange}")
    df = load_pair(exchange, pair, timeframe) if pairs else None
    fng_hist = load_fng_history()

    if df is None or df.empty:
        st.info(
            "Данных нет. Перейди во вкладку **Run freqtrade** и нажми "
            "*Download data*, чтобы скачать свечи."
        )
    else:
        df_ind = compute_indicators(df, bb_std=bb_std, fng_df=fng_hist)
        df_sig = compute_signals(
            df_ind,
            rsi_buy=rsi_buy,
            rsi_sell=rsi_sell,
            volume_factor=volume_factor,
            min_macro=min_macro,
        )
        view = df_sig.tail(n_show).reset_index(drop=True)

        n_entries = int(view["enter_long"].sum())
        n_exits = int(view["exit_long"].sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Свечей в окне", len(view))
        c2.metric("Сигналов входа", n_entries)
        c3.metric("Сигналов выхода", n_exits)
        c4.metric(
            "Текущий F&G",
            f"{view['fng_score'].iloc[-1]:+.2f}"
            if "fng_score" in view and not view["fng_score"].empty
            else "—",
        )

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.55, 0.2, 0.25],
            vertical_spacing=0.03,
            subplot_titles=("Price + Bollinger", "RSI", "Fear & Greed"),
        )
        fig.add_trace(
            go.Candlestick(
                x=view["date"], open=view["open"], high=view["high"],
                low=view["low"], close=view["close"], name="OHLC",
                showlegend=False,
            ),
            row=1, col=1,
        )
        for col, color in [("bb_upper", "#888"), ("bb_middle", "#bbb"), ("bb_lower", "#888")]:
            fig.add_trace(
                go.Scatter(x=view["date"], y=view[col], mode="lines",
                           line=dict(width=1, color=color), name=col, showlegend=False),
                row=1, col=1,
            )

        entries = view[view["enter_long"]]
        exits = view[view["exit_long"]]
        if not entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=entries["date"], y=entries["low"] * 0.995,
                    mode="markers", marker=dict(symbol="triangle-up", size=11, color="#22c55e"),
                    name="entry", hovertext=[f"RSI {r:.1f}" for r in entries["rsi"]],
                ),
                row=1, col=1,
            )
        if not exits.empty:
            fig.add_trace(
                go.Scatter(
                    x=exits["date"], y=exits["high"] * 1.005,
                    mode="markers", marker=dict(symbol="triangle-down", size=11, color="#ef4444"),
                    name="exit", hovertext=[f"RSI {r:.1f}" for r in exits["rsi"]],
                ),
                row=1, col=1,
            )

        fig.add_trace(
            go.Scatter(x=view["date"], y=view["rsi"], mode="lines",
                       line=dict(width=1.5, color="#3b82f6"), name="RSI"),
            row=2, col=1,
        )
        fig.add_hline(y=rsi_buy, line=dict(dash="dash", color="#22c55e", width=1), row=2, col=1)
        fig.add_hline(y=rsi_sell, line=dict(dash="dash", color="#ef4444", width=1), row=2, col=1)

        fig.add_trace(
            go.Scatter(x=view["date"], y=view["fng_score"], mode="lines",
                       line=dict(width=1.5, color="#a855f7"), name="F&G",
                       fill="tozeroy", fillcolor="rgba(168,85,247,0.15)"),
            row=3, col=1,
        )
        fig.add_hline(y=min_macro, line=dict(dash="dash", color="#f59e0b", width=1), row=3, col=1)

        fig.update_layout(
            height=780, margin=dict(l=10, r=10, t=40, b=10),
            xaxis_rangeslider_visible=False, hovermode="x unified",
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Последние 20 сигналов входа"):
            cols = ["date", "close", "rsi", "bb_lower", "volume", "fng_score"]
            recent = entries[cols].tail(20)
            st.dataframe(recent, use_container_width=True)


# ===== Tab 2: sentiment =====
with tab_sent:
    st.subheader("Fear & Greed (исторический)")
    fng = load_fng_history()
    if fng is None or fng.empty:
        st.warning(
            "Нет кэша F&G. Скачай его кнопкой ниже или командой "
            "`python user_data/scripts/download_sentiment_history.py`."
        )
        if st.button("Скачать историю Fear&Greed сейчас"):
            with st.spinner("Загружаю..."):
                provider = SentimentProvider()
                try:
                    df = provider.load_or_fetch_historical_fng(FNG_CACHE, force_refresh=True)
                    st.success(f"OK, {len(df)} строк сохранено в {FNG_CACHE}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Не получилось: {e}")
    else:
        col1, col2, col3 = st.columns(3)
        last = fng.iloc[-1]
        col1.metric("Последнее значение", f"{last['fng_raw']:.0f}/100", f"{last['fng_score']:+.2f}")
        col2.metric("Среднее за 30д", f"{fng['fng_score'].tail(30).mean():+.2f}")
        col3.metric("История", f"{len(fng)} дней")

        # Show last ~2 years
        recent = fng.tail(730).reset_index()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=recent["date"], y=recent["fng_score"], mode="lines",
                line=dict(color="#a855f7"), fill="tozeroy",
                fillcolor="rgba(168,85,247,0.15)", name="F&G score",
            )
        )
        fig.add_hline(y=0, line=dict(color="#aaa", dash="dot"))
        fig.add_hline(y=min_macro, line=dict(color="#f59e0b", dash="dash"),
                      annotation_text=f"порог {min_macro:.2f}")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                          yaxis_range=[-1.05, 1.05])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Live sentiment (требует сеть)")
    token = os.getenv("CRYPTOPANIC_TOKEN")
    if not token:
        st.caption("CRYPTOPANIC_TOKEN не задан — будет только Fear&Greed без новостной части.")
    coins_input = st.text_input("Пары через запятую", "BTC/USDT, ETH/USDT, SOL/USDT")
    if st.button("Опросить sentiment"):
        provider = SentimentProvider(cryptopanic_token=token)
        rows = []
        for p in [c.strip() for c in coins_input.split(",") if c.strip()]:
            try:
                s = provider.get_score(p)
                rows.append({
                    "pair": p,
                    "score": round(s.score, 2),
                    "news": round(s.news_component, 2),
                    "macro": round(s.macro_component, 2),
                })
            except Exception as e:
                rows.append({"pair": p, "score": "err", "news": "err", "macro": str(e)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ===== Tab 3: run freqtrade =====
with tab_run:
    st.subheader("Запуск команд freqtrade")
    st.caption(
        f"Конфиг: `{CONFIG_PATH.relative_to(REPO_ROOT)}` · "
        f"Стратегия: `{STRATEGY_NAME}` · CWD: `{REPO_ROOT}`"
    )

    col1, col2 = st.columns(2)
    days = col1.number_input("Days для download-data", 30, 1500, 540, step=30)
    timerange = col2.text_input("Timerange (для backtest/lookahead)", "20240101-20250601")

    cmd_holder_label = st.empty()
    log_holder = st.empty()

    btn1, btn2, btn3, btn4 = st.columns(4)
    fp = [sys.executable, "-m", "freqtrade"]

    if btn1.button("⬇️ Download data"):
        cmd = [
            *fp, "download-data",
            "--config", str(CONFIG_PATH.relative_to(REPO_ROOT)),
            "--timeframes", "15m", "4h",
            "--days", str(days),
            "--exchange", exchange,
        ]
        cmd_holder_label.markdown(f"**Команда:** `{' '.join(cmd)}`")
        rc = stream_command(cmd, log_holder)
        (st.success if rc == 0 else st.error)(f"Done (rc={rc})")

    if btn2.button("🔍 Lookahead analysis"):
        cmd = [
            *fp, "lookahead-analysis",
            "--config", str(CONFIG_PATH.relative_to(REPO_ROOT)),
            "--strategy", STRATEGY_NAME,
            "--timerange", timerange,
        ]
        cmd_holder_label.markdown(f"**Команда:** `{' '.join(cmd)}`")
        rc = stream_command(cmd, log_holder)
        (st.success if rc == 0 else st.error)(f"Done (rc={rc})")

    if btn3.button("🧪 Backtest"):
        cmd = [
            *fp, "backtesting",
            "--config", str(CONFIG_PATH.relative_to(REPO_ROOT)),
            "--strategy", STRATEGY_NAME,
            "--timerange", timerange,
            "--enable-protections",
        ]
        cmd_holder_label.markdown(f"**Команда:** `{' '.join(cmd)}`")
        rc = stream_command(cmd, log_holder)
        (st.success if rc == 0 else st.error)(f"Done (rc={rc})")

    if btn4.button("🎯 Hyperopt (200 epochs)"):
        cmd = [
            *fp, "hyperopt",
            "--config", str(CONFIG_PATH.relative_to(REPO_ROOT)),
            "--strategy", STRATEGY_NAME,
            "--hyperopt-loss", "SortinoHyperOptLoss",
            "--spaces", "buy", "sell",
            "--timerange", timerange,
            "--epochs", "200",
        ]
        cmd_holder_label.markdown(f"**Команда:** `{' '.join(cmd)}`")
        rc = stream_command(cmd, log_holder)
        (st.success if rc == 0 else st.error)(f"Done (rc={rc})")

    st.markdown("---")
    st.caption(
        "⚠️ Hyperopt и backtest могут занимать минуты. Окно остаётся открытым, "
        "лог обновляется онлайн. Если что-то зависло — перезапусти Streamlit (Ctrl+C в терминале)."
    )
