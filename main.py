import asyncio
import os
import pandas as pd
from binance import AsyncClient, BinanceSocketManager
from telegram import Bot
from smartmoneyconcepts import smc
import logging

# ================== НАСТРОЙКИ ==================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("INTERVAL", "15m")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not TELEGRAM_TOKEN or not CHAT_ID:
    logger.error("❌ TELEGRAM_TOKEN и CHAT_ID обязательны!")
    exit(1)

# ================== SMC ЛОГИКА ==================
def get_smc_signals(df: pd.DataFrame):
    try:
        ob = smc.order_blocks(df, swing_length=5)
        fvg = smc.fvg(df)
        bos = smc.bos(df)

        price = float(df['close'].iloc[-1])

        if (not ob[ob['type'] == 'bullish'].empty and 
            not fvg[fvg['type'] == 'bullish'].empty and 
            bos.iloc[-1].get('bos') == 'bullish'):
            return f"🚀 <b>СИЛЬНЫЙ ЛОНГ {SYMBOL}</b>\nЦена: <b>{price:.2f}</b>\nВремя: {pd.Timestamp.now().strftime('%H:%M')}"

        if (not ob[ob['type'] == 'bearish'].empty and 
            not fvg[fvg['type'] == 'bearish'].empty and 
            bos.iloc[-1].get('bos') == 'bearish'):
            return f"🔻 <b>СИЛЬНЫЙ ШОРТ {SYMBOL}</b>\nЦена: <b>{price:.2f}</b>\nВремя: {pd.Timestamp.now().strftime('%H:%M')}"

        if not ob[ob['type'] == 'bullish'].empty or not fvg[fvg['type'] == 'bullish'].empty:
            return f"📈 Средний ЛОНГ {SYMBOL}\nЦена: {price:.2f}\nВремя: {pd.Timestamp.now().strftime('%H:%M')}"
        if not ob[ob['type'] == 'bearish'].empty or not fvg[fvg['type'] == 'bearish'].empty:
            return f"📉 Средний ШОРТ {SYMBOL}\nЦена: {price:.2f}\nВремя: {pd.Timestamp.now().strftime('%H:%M')}"

        return None
    except Exception as e:
        logger.error(f"SMC error: {e}")
        return None


async def main():
    bot = Bot(token=TELEGRAM_TOKEN)
    
    # Пробуем создать клиент с минимальным пингом + обработкой ошибок
    try:
        client = await AsyncClient.create(testnet=True)
        logger.info("✅ Подключение к Binance Testnet")
    except Exception as e:
        logger.error(f"Не удалось подключиться к Binance: {e}")
        # Продолжаем без клиента (только для теста)
        client = None

    bsm = BinanceSocketManager(client) if client else None

    logger.info(f"🚀 Бот запущен | {SYMBOL} | {INTERVAL}")

    # Загружаем историю (публичный метод)
    try:
        klines = await client.get_historical_klines(SYMBOL, INTERVAL, "7 days ago UTC") if client else []
        df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','qav','num_trades','tbbav','tbqav','ignore'])
        df = df[['timestamp','open','high','low','close','volume']].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    except Exception as e:
        logger.error(f"Ошибка загрузки истории: {e}")
        df = pd.DataFrame(columns=['timestamp','open','high','low','close','volume'])

    if not bsm:
        logger.warning("Работаем без websocket (только тестовый режим)")
        await asyncio.sleep(3600)  # держим контейнер живым
        return

    async with bsm.kline_socket(symbol=SYMBOL, interval=INTERVAL) as ts:
        async for msg in ts:
            if msg and msg.get('k', {}).get('x'):
                new_row = {
                    'timestamp': pd.to_datetime(msg['k']['t'], unit='ms'),
                    'open': float(msg['k']['o']),
                    'high': float(msg['k']['h']),
                    'low': float(msg['k']['l']),
                    'close': float(msg['k']['c']),
                    'volume': float(msg['k']['v'])
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True).tail(1000)

                signal = get_smc_signals(df)
                if signal:
                    try:
                        await bot.send_message(chat_id=CHAT_ID, text=signal, parse_mode='HTML')
                        logger.info("✅ Сигнал отправлен")
                    except Exception as e:
                        logger.error(f"Telegram error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
