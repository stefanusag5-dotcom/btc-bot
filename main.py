import asyncio
import os
import pandas as pd
from binance import AsyncClient, BinanceSocketManager
from telegram import Bot
from smartmoneyconcepts import smc
import logging

# ================== НАСТРОЙКИ ИЗ ENVIRONMENT VARIABLES (Railway) ==================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("INTERVAL", "15m")

if not TELEGRAM_TOKEN or not CHAT_ID:
    raise ValueError("❌ TELEGRAM_TOKEN и CHAT_ID обязательны! Добавь их в Variables на Railway.")

# Логирование (удобно смотреть в Railway Logs)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info(f"🚀 Запуск DBS Pro SMC Bot | {SYMBOL} | {INTERVAL}")

# ================== SMC ЛОГИКА (улучшенная) ==================
def get_smc_signals(df: pd.DataFrame):
    try:
        ob = smc.order_blocks(df, swing_length=5)
        fvg = smc.fvg(df)
        bos = smc.bos(df)

        price = float(df['close'].iloc[-1])

        # Сильный Long
        if (not ob[ob['type'] == 'bullish'].empty and 
            not fvg[fvg['type'] == 'bullish'].empty and 
            bos.iloc[-1].get('bos') == 'bullish'):
            
            return (f"🚀 <b>СИЛЬНЫЙ ЛОНГ BTC</b>\n"
                    f"Цена: <b>{price:.2f}</b> USDT\n"
                    f"Время: {pd.Timestamp.now().strftime('%H:%M:%S')}\n"
                    f"Тип: OB + FVG + BOS")

        # Сильный Short
        if (not ob[ob['type'] == 'bearish'].empty and 
            not fvg[fvg['type'] == 'bearish'].empty and 
            bos.iloc[-1].get('bos') == 'bearish'):
            
            return (f"🔻 <b>СИЛЬНЫЙ ШОРТ BTC</b>\n"
                    f"Цена: <b>{price:.2f}</b> USDT\n"
                    f"Время: {pd.Timestamp.now().strftime('%H:%M:%S')}\n"
                    f"Тип: OB + FVG + BOS")

        # Слабые сигналы (только OB или FVG)
        if not ob[ob['type'] == 'bullish'].empty or not fvg[fvg['type'] == 'bullish'].empty:
            return f"📈 Средний ЛОНГ BTC\nЦена: {price:.2f}\nВремя: {pd.Timestamp.now().strftime('%H:%M:%S')}"

        if not ob[ob['type'] == 'bearish'].empty or not fvg[fvg['type'] == 'bearish'].empty:
            return f"📉 Средний ШОРТ BTC\nЦена: {price:.2f}\nВремя: {pd.Timestamp.now().strftime('%H:%M:%S')}"

        return None

    except Exception as e:
        logger.error(f"SMC error: {e}")
        return None


# ================== ОСНОВНОЙ ЦИКЛ ==================
async def main():
    bot = Bot(token=TELEGRAM_TOKEN)
    client = await AsyncClient.create()
    bsm = BinanceSocketManager(client)

    # Загружаем историю
    logger.info("Загружаем исторические данные...")
    klines = await client.get_historical_klines(SYMBOL, INTERVAL, "10 days ago UTC")
    
    df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','qav','num_trades','tbbav','tbqav','ignore'])
    df = df[['timestamp','open','high','low','close','volume']].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    logger.info("✅ Бот успешно запущен на Railway. Ожидаем сигналы...")

    async with bsm.kline_socket(symbol=SYMBOL, interval=INTERVAL) as ts:
        async for msg in ts:
            if msg['k']['x']:  # свеча закрылась
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
                        logger.info("📤 Сигнал отправлен в Telegram")
                    except Exception as e:
                        logger.error(f"Ошибка Telegram: {e}")

if __name__ == "__main__":
    asyncio.run(main())
