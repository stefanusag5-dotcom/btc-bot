import ccxt
import pandas as pd

exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

print("Проверяем соединение с Binance Futures...")

try:
    # Простая проверка — получаем тикер BTC
    ticker = exchange.fetch_ticker('BTC/USDT')
    print(f"✅ BTC цена: {ticker['last']}")

    # Проверяем OHLCV (то, что использует бот)
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '15m', limit=100)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    print(f"✅ OHLCV получен успешно! Строк: {len(df)}")
    print(f"Последняя цена: {df['close'].iloc[-1]}")

except Exception as e:
    print(f"❌ Ошибка: {e}")
