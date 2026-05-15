# RSI Mean Reversion + Sentiment (Bybit Spot, малые суммы)

Адаптировано под Bybit spot и стейк $10–20 на сделку. Включает:

- `strategies/RsiMeanReversionStrategy.py` — стратегия с macro-фильтром Fear&Greed,
  работающим и в **бэктесте**, и в live.
- `data_providers/sentiment_provider.py` — live-провайдер CryptoPanic + Fear&Greed
  и загрузчик исторического F&G в CSV для бэктеста.
- `config_bybit_spot.json` — конфиг: `stake_amount=15`, `max_open_trades=2`,
  `dry_run_wallet=200`, 6 ликвидных пар. Dry-run по умолчанию.
- `scripts/download_sentiment_history.py` — выкачать историю Fear&Greed.
- `scripts/walk_forward.py` — walk-forward валидация против переоптимизации.

## Полная последовательность

```bash
# 0. Один раз — установка
./setup.sh -i && source .venv/bin/activate

# 1. История F&G в кэш (используется бэктестом)
python user_data/scripts/download_sentiment_history.py

# 2. Свечи (15m + 4h для informative)
freqtrade download-data \
  --config user_data/config_bybit_spot.json \
  --timeframes 15m 4h --days 540 --exchange bybit

# 3. Look-ahead bias чек
freqtrade lookahead-analysis \
  --config user_data/config_bybit_spot.json \
  --strategy RsiMeanReversionStrategy \
  --timerange 20240101-20250101

# 4. Бэктест с защитами
freqtrade backtesting \
  --config user_data/config_bybit_spot.json \
  --strategy RsiMeanReversionStrategy \
  --timerange 20240101-20250601 \
  --enable-protections

# 5. Walk-forward (главная защита от overfit)
python user_data/scripts/walk_forward.py \
  --config user_data/config_bybit_spot.json \
  --strategy RsiMeanReversionStrategy \
  --start 20240101 --end 20250501 \
  --train-days 120 --test-days 30 --epochs 200

# 6. Dry-run 4+ недели
export CRYPTOPANIC_TOKEN=...    # опционально, добавит новостной компонент
freqtrade trade \
  --config user_data/config_bybit_spot.json \
  --strategy RsiMeanReversionStrategy
```

## Sentiment

Два слоя:

1. **Macro (Fear&Greed)** — работает в бэктесте и в live. Свежий CSV из
   `user_data/data/sentiment/fng_history.csv` мерджится в свечи в
   `populate_indicators`. Фильтр `fng_score >= min_macro_score` (`-0.3` по
   умолчанию) применяется в `populate_entry_trend`.
2. **News (CryptoPanic)** — только live, через `confirm_trade_entry`. Включается,
   если задана переменная окружения `CRYPTOPANIC_TOKEN`. Бесплатный API CryptoPanic
   исторических данных не отдаёт, поэтому новостной компонент не воспроизводится
   в бэктесте — но он и не доминирует в скоринге (вес 60% на текущей дате;
   macro 40% всё равно учтён).

## Переход на live ($10–20)

1. Минимум 4 недели dry-run.
2. На Bybit отдельный API-ключ: Spot Trade + Read, **без Withdraw**, IP whitelist.
3. В конфиге: `"dry_run": false`, ключи в `exchange.key/secret`. Поменять пароли в `api_server`.
4. Для $15-стейка с BTC учти, что Bybit имеет минимальный размер ордера. Если
   `BTC/USDT` стабильно даёт ошибку min-order — убери его из `pair_whitelist` или
   подними `stake_amount` до $25.

## Что НЕ делать

- Не запускать live без walk-forward с положительными OOS-результатами в большинстве окон.
- Не докручивать параметры под прошедший месяц — почти всегда это overfit.
- Не вкладывать больше, чем готов потерять. Любой бот может слить депозит.
