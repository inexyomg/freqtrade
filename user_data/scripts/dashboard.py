"""
Streamlit dashboard for the RSI Mean Reversion strategy.

Run from the freqtrade repo root:

    streamlit run user_data/scripts/dashboard.py

Tabs:
  1. 💡 Сигналы сейчас — для каждой пары: купить / ждать, с разбором условий
                          и подсказкой "по какой цене входить, где брать прибыль,
                          где ставить стоп"
  2. 📋 История сделок — симулированные сделки за всю историю с P&L
  3. 📈 График         — свечи + индикаторы + сигналы на одной паре
  4. 🌡️ Sentiment      — Fear&Greed history + live sentiment per coin
  5. ⚙️ Запуск          — кнопки download-data / lookahead / backtest / hyperopt

The dashboard computes indicators independently of the actual strategy file
so you can scrub parameters and see signals instantly. The committed strategy
in strategies/RsiMeanReversionStrategy.py is the source of truth for live.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

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

# Strategy constants (must mirror RsiMeanReversionStrategy.py)
STOPLOSS = -0.05
ROI_TABLE = [(0, 0.025), (30, 0.015), (90, 0.008), (240, 0.0)]
TRAIL_OFFSET = 0.015
TRAIL_DROP = 0.01

st.set_page_config(page_title="RSI MeanRev — простой режим", layout="wide")


# ---------- helpers ----------


def load_config_whitelist() -> tuple[str, list[str]]:
    try:
        cfg = json.loads(CONFIG_PATH.read_text())
        return cfg["exchange"]["name"], cfg["exchange"]["pair_whitelist"]
    except Exception:
        return "bybit", ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


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
        except Exception:
            return None
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df = df.sort_values("date").reset_index(drop=True)
        return df
    return None


def compute_indicators(df: pd.DataFrame, bb_std: float, fng_df: pd.DataFrame | None) -> pd.DataFrame:
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


def compute_4h_trend_ok(df_4h: pd.DataFrame | None) -> pd.Series | None:
    if df_4h is None or len(df_4h) < 200:
        return None
    df_4h = df_4h.copy()
    df_4h["ema50"] = ta.EMA(df_4h, timeperiod=50)
    df_4h["ema200"] = ta.EMA(df_4h, timeperiod=200)
    df_4h["trend_ok"] = (
        (df_4h["ema50"] > df_4h["ema200"])
        | ((df_4h["close"] / df_4h["ema200"] - 1).abs() < 0.05)
    )
    return df_4h.set_index("date")["trend_ok"]


def merge_4h_into_15m(df_15m: pd.DataFrame, trend_4h: pd.Series | None) -> pd.DataFrame:
    df = df_15m.copy()
    if trend_4h is None:
        df["trend_ok_4h"] = True
        return df
    trend_4h = trend_4h.sort_index()
    merged = pd.merge_asof(
        df.sort_values("date"),
        trend_4h.rename("trend_ok_4h").reset_index().sort_values("date"),
        on="date",
        direction="backward",
    )
    merged["trend_ok_4h"] = merged["trend_ok_4h"].fillna(False)
    return merged


def compute_signals(df: pd.DataFrame, rsi_buy, rsi_sell, volume_factor, min_macro) -> pd.DataFrame:
    df = df.copy()
    df["enter_long"] = (
        (df["rsi"] < rsi_buy)
        & (df["close"] < df["bb_lower"])
        & (df["volume"] > df["volume_sma20"] * volume_factor)
        & df["trend_ok_4h"]
        & (df["fng_score"] >= min_macro)
        & (df["volume"] > 0)
    )
    df["exit_long"] = (df["rsi"] > rsi_sell) & (df["close"] > df["bb_middle"])
    return df


def load_fng_history() -> pd.DataFrame | None:
    if not FNG_CACHE.exists():
        return None
    try:
        return pd.read_csv(FNG_CACHE, parse_dates=["date"]).set_index("date").sort_index()
    except Exception:
        return None


def stream_command(cmd: list[str], placeholder) -> int:
    proc = subprocess.Popen(
        cmd, cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line.rstrip())
        placeholder.code("\n".join(lines[-200:]), language="bash")
    proc.wait()
    return proc.returncode


# ---------- signal diagnosis ----------


def diagnose_latest(
    row: pd.Series,
    rsi_buy: int,
    rsi_sell: int,
    volume_factor: float,
    min_macro: float,
) -> dict:
    """Build a human-readable check-list for the latest candle.

    Returns:
        {"verdict": "BUY" | "WAIT", "checks": [{"ok", "label", "detail"}], ...}
    """
    if pd.isna(row.get("rsi")) or pd.isna(row.get("bb_lower")):
        return {"verdict": "WAIT", "checks": [], "reason_summary": "Недостаточно данных для расчёта индикаторов"}

    rsi = float(row["rsi"])
    close = float(row["close"])
    bb_lower = float(row["bb_lower"])
    vol = float(row["volume"])
    vol_avg = float(row["volume_sma20"]) if not pd.isna(row["volume_sma20"]) else 0.0
    trend_ok = bool(row.get("trend_ok_4h", True))
    fng = float(row.get("fng_score", 0.0))

    checks = [
        {
            "ok": rsi < rsi_buy,
            "label": "Перепроданность (RSI)",
            "detail": (
                f"RSI = {rsi:.1f}, нужно <{rsi_buy}. "
                + ("Покупатели выдохлись, статистически шанс отскока выше." if rsi < rsi_buy
                   else "Пока недостаточно глубокая просадка — рынок ещё не перепродан.")
            ),
        },
        {
            "ok": close < bb_lower,
            "label": "Цена под нижней Bollinger",
            "detail": (
                f"Цена ${close:,.2f}, нижняя BB ${bb_lower:,.2f}. "
                + ("Цена ушла ниже статистического коридора — повышенный шанс возврата к среднему."
                   if close < bb_lower else "Цена ещё внутри обычного коридора, статистической аномалии нет.")
            ),
        },
        {
            "ok": vol > vol_avg * volume_factor,
            "label": "Повышенный объём",
            "detail": (
                f"Объём свечи {vol:,.0f}, в {(vol / vol_avg) if vol_avg else 0:.1f}× от среднего за 20 свечей. "
                + ("Есть реальный интерес — не пустой рынок."
                   if vol > vol_avg * volume_factor
                   else f"Слишком тонкий рынок — нужно ≥{volume_factor:.1f}× от среднего.")
            ),
        },
        {
            "ok": trend_ok,
            "label": "4h-тренд не падающий",
            "detail": (
                "На 4h-графике EMA50≥EMA200 или цена близко к EMA200. Безопасный фон."
                if trend_ok
                else "На 4h явный нисходящий тренд — опасно ловить нож, ждём разворота на старшем ТФ."
            ),
        },
        {
            "ok": fng >= min_macro,
            "label": "Рыночный настрой не катастрофический",
            "detail": (
                f"Fear&Greed = {fng:+.2f}, порог {min_macro:+.2f}. "
                + ("Настрой в норме." if fng >= min_macro
                   else "Рынок в панике — фильтр блокирует вход, чтобы не попасть в каскад продаж.")
            ),
        },
    ]

    all_ok = all(c["ok"] for c in checks)
    n_ok = sum(c["ok"] for c in checks)

    if all_ok:
        reason = "🟢 Все 5 условий выполнены — стратегия покупает."
    else:
        missing = [c["label"] for c in checks if not c["ok"]]
        reason = f"⚪ Выполнено {n_ok}/5. Не хватает: " + ", ".join(missing).lower() + "."

    return {
        "verdict": "BUY" if all_ok else "WAIT",
        "checks": checks,
        "reason_summary": reason,
        "rsi": rsi,
        "close": close,
        "bb_lower": bb_lower,
        "fng": fng,
        "n_ok": n_ok,
    }


def trade_plan(entry_price: float) -> dict:
    """Return human-readable entry/TP/SL plan based on strategy constants."""
    return {
        "entry": entry_price,
        "tp_30min": entry_price * (1 + 0.025),
        "tp_90min": entry_price * (1 + 0.015),
        "tp_240min": entry_price * (1 + 0.008),
        "stoploss": entry_price * (1 + STOPLOSS),
    }


# ---------- trade simulation (for history table) ----------


def _roi_for(elapsed_min: float) -> float:
    target = ROI_TABLE[0][1]
    for mins, val in ROI_TABLE:
        if elapsed_min >= mins:
            target = val
    return target


def simulate_trades(df: pd.DataFrame, rsi_sell: int) -> list[dict]:
    """Walk through df, simulate trades following strategy exit rules.

    Approximates the full freqtrade backtest:
      stoploss -> ROI table -> trailing stop -> RSI exit signal (in this priority).
    Real backtest is still the authoritative number — this is for the UI table.
    """
    trades: list[dict] = []
    in_pos = False
    entry_price = 0.0
    entry_time = None
    peak = 0.0
    trail_active = False

    arr = df.to_dict("records")
    for row in arr:
        if pd.isna(row.get("close")):
            continue
        if not in_pos:
            if row.get("enter_long"):
                in_pos = True
                entry_price = float(row["close"])
                entry_time = row["date"]
                peak = float(row["high"]) if row.get("high") else entry_price
                trail_active = False
            continue

        elapsed_min = (row["date"] - entry_time).total_seconds() / 60.0
        low = float(row["low"])
        high = float(row["high"])
        close = float(row["close"])
        sl_price = entry_price * (1 + STOPLOSS)

        # 1. Stoploss
        if low <= sl_price:
            trades.append({
                "entry_time": entry_time, "entry_price": entry_price,
                "exit_time": row["date"], "exit_price": sl_price,
                "pnl_pct": -5.0, "reason": "🛑 Стоп-лосс",
                "duration_min": elapsed_min,
            })
            in_pos = False
            continue

        # 2. ROI
        roi_target = _roi_for(elapsed_min)
        if (high - entry_price) / entry_price >= roi_target:
            exit_p = entry_price * (1 + roi_target)
            trades.append({
                "entry_time": entry_time, "entry_price": entry_price,
                "exit_time": row["date"], "exit_price": exit_p,
                "pnl_pct": roi_target * 100, "reason": "🎯 Цель прибыли (ROI)",
                "duration_min": elapsed_min,
            })
            in_pos = False
            continue

        # 3. Trailing
        if high > peak:
            peak = high
        if not trail_active and (peak - entry_price) / entry_price >= TRAIL_OFFSET:
            trail_active = True
        if trail_active:
            trail_stop = peak * (1 - TRAIL_DROP)
            if low <= trail_stop:
                trades.append({
                    "entry_time": entry_time, "entry_price": entry_price,
                    "exit_time": row["date"], "exit_price": trail_stop,
                    "pnl_pct": (trail_stop - entry_price) / entry_price * 100,
                    "reason": "📉 Trailing-stop", "duration_min": elapsed_min,
                })
                in_pos = False
                continue

        # 4. RSI signal exit
        if row.get("exit_long"):
            trades.append({
                "entry_time": entry_time, "entry_price": entry_price,
                "exit_time": row["date"], "exit_price": close,
                "pnl_pct": (close - entry_price) / entry_price * 100,
                "reason": "📊 Сигнал RSI", "duration_min": elapsed_min,
            })
            in_pos = False
    return trades


def load_pair_with_signals(exchange, pair, fng_hist, rsi_buy, rsi_sell, bb_std, volume_factor, min_macro):
    df15 = load_pair(exchange, pair, "15m")
    if df15 is None or df15.empty:
        return None
    df4 = load_pair(exchange, pair, "4h")
    df15 = compute_indicators(df15, bb_std=bb_std, fng_df=fng_hist)
    trend_4h = compute_4h_trend_ok(df4)
    df15 = merge_4h_into_15m(df15, trend_4h)
    df15 = compute_signals(df15, rsi_buy, rsi_sell, volume_factor, min_macro)
    return df15


# ---------- sidebar ----------

st.sidebar.title("⚙️ Параметры")
cfg_exchange, cfg_whitelist = load_config_whitelist()
exchange = st.sidebar.selectbox("Биржа", ["bybit", "binance", "okx"],
                                index=["bybit", "binance", "okx"].index(cfg_exchange) if cfg_exchange in ("bybit", "binance", "okx") else 0)

st.sidebar.markdown("**Пороги стратегии**")
st.sidebar.caption("Не трогай, если не понимаешь — дефолты разумные")
rsi_buy = st.sidebar.slider("Перепроданность RSI <", 10, 40, 25,
                            help="Чем меньше — тем строже фильтр, сигналов меньше но качественнее")
rsi_sell = st.sidebar.slider("Перекупленность RSI >", 50, 85, 65,
                             help="При каком RSI бот видит «куплено достаточно, можно фиксировать»")
bb_std = st.sidebar.slider("Ширина Bollinger (σ)", 1.0, 3.5, 2.0, 0.1,
                           help="Сколько стандартных отклонений = «дёшево»")
volume_factor = st.sidebar.slider("Объём × среднего", 0.5, 3.0, 1.2, 0.1)
min_macro = st.sidebar.slider("Мин. Fear&Greed", -1.0, 0.5, -0.3, 0.05,
                              help="Ниже этого порога — общий рынок паникует, бот не входит")
st.sidebar.markdown("---")
st.sidebar.caption(
    "Цены входа и выходов считаются как:\n"
    "- Вход: цена закрытия сигнальной свечи\n"
    "- TP1: +2.5% (до 30 мин)\n"
    "- TP2: +1.5% (до 90 мин)\n"
    "- TP3: +0.8% (до 4ч)\n"
    "- Стоп-лосс: −5%"
)


# ---------- tabs ----------

tab_now, tab_history, tab_chart, tab_sent, tab_run = st.tabs([
    "💡 Сигналы сейчас",
    "📋 История сделок",
    "📈 График",
    "🌡️ Sentiment",
    "⚙️ Запуск",
])


# ===== Tab 1: signals now =====

with tab_now:
    st.title("💡 Стоит ли покупать прямо сейчас?")
    st.caption(
        "Для каждой пары из конфига показываем, выполняются ли сейчас условия "
        "стратегии. **Зелёная карточка** = бот купил бы прямо сейчас. "
        "**Серая** = ждём, и видно почему."
    )

    fng_hist = load_fng_history()

    if not list_pairs(exchange, "15m"):
        st.warning(
            "🚧 Нет скачанных данных. Перейди во вкладку **⚙️ Запуск** и "
            "нажми «Download data», потом возвращайся сюда."
        )
    else:
        pairs_to_show = [p for p in cfg_whitelist if p in list_pairs(exchange, "15m")]
        if not pairs_to_show:
            st.warning("Ни одна из пар в конфиге не имеет скачанных свечей. Скачай данные.")

        # Render in 2 columns
        for i in range(0, len(pairs_to_show), 2):
            row_pairs = pairs_to_show[i:i + 2]
            cols = st.columns(len(row_pairs))
            for col, pair in zip(cols, row_pairs):
                with col:
                    df_sig = load_pair_with_signals(
                        exchange, pair, fng_hist,
                        rsi_buy, rsi_sell, bb_std, volume_factor, min_macro,
                    )
                    if df_sig is None or df_sig.empty:
                        st.error(f"{pair}: нет данных")
                        continue
                    last = df_sig.iloc[-1]
                    diag = diagnose_latest(last, rsi_buy, rsi_sell, volume_factor, min_macro)
                    plan = trade_plan(float(last["close"]))

                    if diag["verdict"] == "BUY":
                        st.markdown(f"### 🟢 {pair}  —  **ПОКУПАТЬ**")
                    else:
                        st.markdown(f"### ⚪ {pair}  —  ждём ({diag['n_ok']}/5)")

                    c1, c2 = st.columns([1, 1])
                    c1.metric("Текущая цена", f"${last['close']:,.4f}".rstrip("0").rstrip("."))
                    c2.metric("RSI / F&G", f"{diag['rsi']:.0f} / {diag['fng']:+.2f}")

                    if diag["verdict"] == "BUY":
                        st.success(diag["reason_summary"])
                        st.markdown("**📍 План сделки**")
                        st.markdown(
                            f"- **Купить по:** ~${plan['entry']:,.4f}\n"
                            f"- **🎯 Продать (до 30 мин, +2.5%):** ~${plan['tp_30min']:,.4f}\n"
                            f"- **🎯 Продать (до 90 мин, +1.5%):** ~${plan['tp_90min']:,.4f}\n"
                            f"- **🎯 Продать (до 4ч, +0.8%):** ~${plan['tp_240min']:,.4f}\n"
                            f"- **🛑 Стоп-лосс (−5%):** ~${plan['stoploss']:,.4f}"
                        )
                    else:
                        st.info(diag["reason_summary"])

                    with st.expander("🔍 Разбор условий"):
                        for c in diag["checks"]:
                            mark = "✅" if c["ok"] else "❌"
                            st.markdown(f"**{mark} {c['label']}**  \n{c['detail']}")

                    st.markdown("---")

        st.caption(
            f"📅 Данные актуальны на момент последнего скачивания. "
            f"Чтобы обновить — вкладка **⚙️ Запуск** → «Download data»."
        )


# ===== Tab 2: trade history =====

with tab_history:
    st.title("📋 Сделки на исторических данных")
    st.caption(
        "Симуляция: для каждого сигнала в прошлом я прохожу вперёд по свечам и "
        "выхожу по правилам стратегии (стоп-лосс, цель прибыли, trailing, сигнал RSI). "
        "Это **приблизительная** оценка для понимания, как стратегия себя вела. "
        "Точный отчёт даёт кнопка «Backtest» во вкладке Запуск."
    )

    fng_hist = load_fng_history()
    available = [p for p in cfg_whitelist if p in list_pairs(exchange, "15m")]
    if not available:
        st.warning("Нет скачанных данных. Запусти Download data.")
    else:
        sel = st.multiselect("Пары", available, default=available)
        if sel:
            all_trades = []
            for pair in sel:
                df_sig = load_pair_with_signals(
                    exchange, pair, fng_hist,
                    rsi_buy, rsi_sell, bb_std, volume_factor, min_macro,
                )
                if df_sig is None:
                    continue
                trades = simulate_trades(df_sig, rsi_sell=rsi_sell)
                for t in trades:
                    t["pair"] = pair
                all_trades.extend(trades)

            if not all_trades:
                st.info("Сигналов не нашлось. Попробуй ослабить пороги в сайдбаре.")
            else:
                tdf = pd.DataFrame(all_trades).sort_values("entry_time", ascending=False)

                wins = (tdf["pnl_pct"] > 0).sum()
                losses = (tdf["pnl_pct"] <= 0).sum()
                total = len(tdf)
                cum_pnl = tdf["pnl_pct"].sum()
                avg_pnl = tdf["pnl_pct"].mean()
                win_rate = wins / total * 100 if total else 0

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Сделок", total)
                m2.metric("Прибыльных", wins, f"{win_rate:.0f}%")
                m3.metric("Убыточных", losses)
                m4.metric("Средняя сделка", f"{avg_pnl:+.2f}%")
                m5.metric("Сумма (без реинвеста)", f"{cum_pnl:+.1f}%")

                fig = go.Figure()
                tdf_sorted = tdf.sort_values("entry_time")
                fig.add_trace(go.Scatter(
                    x=tdf_sorted["entry_time"],
                    y=tdf_sorted["pnl_pct"].cumsum(),
                    mode="lines", line=dict(color="#22c55e"),
                    name="Накопительный P&L (%)",
                ))
                fig.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10),
                                  yaxis_title="Накопительный P&L, %")
                st.plotly_chart(fig, use_container_width=True)

                # nice table
                display = tdf.copy()
                display["entry_time"] = display["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
                display["exit_time"] = display["exit_time"].dt.strftime("%Y-%m-%d %H:%M")
                display["entry_price"] = display["entry_price"].map(lambda x: f"${x:,.4f}".rstrip("0").rstrip("."))
                display["exit_price"] = display["exit_price"].map(lambda x: f"${x:,.4f}".rstrip("0").rstrip("."))
                display["pnl_pct"] = display["pnl_pct"].map(lambda x: f"{x:+.2f}%")
                display["duration"] = display["duration_min"].map(
                    lambda x: f"{int(x // 60)}ч {int(x % 60)}м" if x >= 60 else f"{int(x)}м"
                )
                display = display[["pair", "entry_time", "entry_price",
                                   "exit_time", "exit_price", "pnl_pct", "duration", "reason"]]
                display.columns = ["Пара", "Вход", "Цена входа",
                                   "Выход", "Цена выхода", "P&L", "Длит.", "Причина выхода"]
                st.dataframe(display, use_container_width=True, height=500)


# ===== Tab 3: chart =====

with tab_chart:
    st.subheader("📈 Подробный график (для тех, кто хочет копаться)")
    pairs_avail = list_pairs(exchange, "15m")
    if not pairs_avail:
        st.info("Нет данных. Скачай через вкладку ⚙️ Запуск.")
    else:
        pair = st.selectbox("Пара", pairs_avail, key="chart_pair")
        n_show = st.number_input("Последние N свечей", 200, 20000, 2000, step=500)
        fng_hist = load_fng_history()
        df_sig = load_pair_with_signals(
            exchange, pair, fng_hist,
            rsi_buy, rsi_sell, bb_std, volume_factor, min_macro,
        )
        if df_sig is None or df_sig.empty:
            st.error("Не удалось загрузить данные")
        else:
            view = df_sig.tail(n_show).reset_index(drop=True)
            n_entries = int(view["enter_long"].sum())
            n_exits = int(view["exit_long"].sum())
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Свечей", len(view))
            c2.metric("Сигналов входа", n_entries)
            c3.metric("Сигналов выхода", n_exits)
            c4.metric("Текущий F&G",
                      f"{view['fng_score'].iloc[-1]:+.2f}" if "fng_score" in view else "—")

            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                row_heights=[0.55, 0.2, 0.25], vertical_spacing=0.03,
                subplot_titles=("Цена + Bollinger", "RSI", "Fear & Greed"),
            )
            fig.add_trace(
                go.Candlestick(
                    x=view["date"], open=view["open"], high=view["high"],
                    low=view["low"], close=view["close"], showlegend=False,
                ), row=1, col=1,
            )
            for col, color in [("bb_upper", "#888"), ("bb_middle", "#bbb"), ("bb_lower", "#888")]:
                fig.add_trace(
                    go.Scatter(x=view["date"], y=view[col], mode="lines",
                               line=dict(width=1, color=color), showlegend=False),
                    row=1, col=1,
                )
            entries = view[view["enter_long"]]
            exits = view[view["exit_long"]]
            if not entries.empty:
                fig.add_trace(go.Scatter(
                    x=entries["date"], y=entries["low"] * 0.995, mode="markers",
                    marker=dict(symbol="triangle-up", size=11, color="#22c55e"),
                    name="вход",
                ), row=1, col=1)
            if not exits.empty:
                fig.add_trace(go.Scatter(
                    x=exits["date"], y=exits["high"] * 1.005, mode="markers",
                    marker=dict(symbol="triangle-down", size=11, color="#ef4444"),
                    name="выход",
                ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=view["date"], y=view["rsi"], mode="lines",
                line=dict(width=1.5, color="#3b82f6"), name="RSI",
            ), row=2, col=1)
            fig.add_hline(y=rsi_buy, line=dict(dash="dash", color="#22c55e", width=1), row=2, col=1)
            fig.add_hline(y=rsi_sell, line=dict(dash="dash", color="#ef4444", width=1), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=view["date"], y=view["fng_score"], mode="lines",
                line=dict(width=1.5, color="#a855f7"), name="F&G",
                fill="tozeroy", fillcolor="rgba(168,85,247,0.15)",
            ), row=3, col=1)
            fig.add_hline(y=min_macro, line=dict(dash="dash", color="#f59e0b", width=1), row=3, col=1)
            fig.update_layout(height=780, margin=dict(l=10, r=10, t=40, b=10),
                              xaxis_rangeslider_visible=False, hovermode="x unified",
                              legend=dict(orientation="h", y=1.05))
            st.plotly_chart(fig, use_container_width=True)


# ===== Tab 4: sentiment =====

with tab_sent:
    st.subheader("🌡️ Fear & Greed — настрой рынка")
    fng = load_fng_history()
    if fng is None or fng.empty:
        st.warning("Нет кэша F&G.")
        if st.button("Скачать историю"):
            try:
                df = SentimentProvider().load_or_fetch_historical_fng(FNG_CACHE, force_refresh=True)
                st.success(f"Сохранено {len(df)} строк")
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка: {e}")
    else:
        c1, c2, c3 = st.columns(3)
        last = fng.iloc[-1]
        c1.metric("Сейчас", f"{last['fng_raw']:.0f}/100", f"{last['fng_score']:+.2f}")
        c2.metric("Среднее за 30д", f"{fng['fng_score'].tail(30).mean():+.2f}")
        c3.metric("Дней истории", f"{len(fng)}")
        recent = fng.tail(730).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent["date"], y=recent["fng_score"], mode="lines",
            line=dict(color="#a855f7"), fill="tozeroy",
            fillcolor="rgba(168,85,247,0.15)", name="F&G",
        ))
        fig.add_hline(y=0, line=dict(color="#aaa", dash="dot"))
        fig.add_hline(y=min_macro, line=dict(color="#f59e0b", dash="dash"),
                      annotation_text=f"порог {min_macro:.2f}")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                          yaxis_range=[-1.05, 1.05])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Live sentiment")
    token = os.getenv("CRYPTOPANIC_TOKEN")
    if not token:
        st.caption("CRYPTOPANIC_TOKEN не задан — только F&G компонент.")
    coins_input = st.text_input("Пары через запятую", "BTC/USDT, ETH/USDT, SOL/USDT")
    if st.button("Опросить sentiment"):
        provider = SentimentProvider(cryptopanic_token=token)
        rows = []
        for p in [c.strip() for c in coins_input.split(",") if c.strip()]:
            try:
                s = provider.get_score(p)
                rows.append({"pair": p, "score": round(s.score, 2),
                             "news": round(s.news_component, 2),
                             "macro": round(s.macro_component, 2)})
            except Exception as e:
                rows.append({"pair": p, "score": "err", "news": "err", "macro": str(e)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ===== Tab 5: run =====

with tab_run:
    st.subheader("⚙️ Запуск команд freqtrade")
    st.caption(
        f"Конфиг: `{CONFIG_PATH.relative_to(REPO_ROOT)}` · "
        f"Стратегия: `{STRATEGY_NAME}` · CWD: `{REPO_ROOT}`"
    )

    c1, c2 = st.columns(2)
    days = c1.number_input("Days для download-data", 30, 1500, 730, step=30)
    timerange = c2.text_input("Timerange (для backtest/lookahead)", "20240601-20251101")

    cmd_holder = st.empty()
    log_holder = st.empty()
    b1, b2, b3, b4 = st.columns(4)
    fp = [sys.executable, "-m", "freqtrade"]

    if b1.button("⬇️ Download data"):
        cmd = [*fp, "download-data",
               "--config", str(CONFIG_PATH.relative_to(REPO_ROOT)),
               "--timeframes", "15m", "4h",
               "--days", str(days),
               "--exchange", exchange]
        cmd_holder.markdown(f"**Команда:** `{' '.join(cmd)}`")
        rc = stream_command(cmd, log_holder)
        (st.success if rc == 0 else st.error)(f"Done (rc={rc})")

    if b2.button("🔍 Lookahead analysis"):
        cmd = [*fp, "lookahead-analysis",
               "--config", str(CONFIG_PATH.relative_to(REPO_ROOT)),
               "--strategy", STRATEGY_NAME, "--timerange", timerange]
        cmd_holder.markdown(f"**Команда:** `{' '.join(cmd)}`")
        rc = stream_command(cmd, log_holder)
        (st.success if rc == 0 else st.error)(f"Done (rc={rc})")

    if b3.button("🧪 Backtest"):
        cmd = [*fp, "backtesting",
               "--config", str(CONFIG_PATH.relative_to(REPO_ROOT)),
               "--strategy", STRATEGY_NAME, "--timerange", timerange,
               "--enable-protections"]
        cmd_holder.markdown(f"**Команда:** `{' '.join(cmd)}`")
        rc = stream_command(cmd, log_holder)
        (st.success if rc == 0 else st.error)(f"Done (rc={rc})")

    if b4.button("🎯 Hyperopt (200 epochs)"):
        cmd = [*fp, "hyperopt",
               "--config", str(CONFIG_PATH.relative_to(REPO_ROOT)),
               "--strategy", STRATEGY_NAME,
               "--hyperopt-loss", "SortinoHyperOptLoss",
               "--spaces", "buy", "sell",
               "--timerange", timerange, "--epochs", "200"]
        cmd_holder.markdown(f"**Команда:** `{' '.join(cmd)}`")
        rc = stream_command(cmd, log_holder)
        (st.success if rc == 0 else st.error)(f"Done (rc={rc})")
