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
    raise ValueError("❌ TELEGRAM_TOKEN и CHAT_ID обязательны в Variables Railway!")

# ================== SMC ЛОГИКА ==================
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
            return f"🚀 <b>СИЛЬНЫЙ ЛОНГ {SYMBOL}</b>\nЦена: <b>{price:.2f}</b>\nВремя: {pd.Timestamp.now().strftime('%H:%M')}"

        # Сильный Short
        if (not ob[ob['type'] == 'bearish'].empty and 
            not fvg[fvg['type'] == 'bearish'].empty and 
            bos.iloc[-1].get('bos') == 'bearish'):
            return f"🔻 <b>СИЛЬНЫЙ ШОРТ {SYMBOL}</b>\nЦена: <b>{price:.2f}</b>\nВремя: {pd.Timestamp.now().strftime('%H:%M')}"

        # Слабые сигналы
        if not ob[ob['type'] == 'bullish'].empty or not fvg[fvg['type'] == 'bullish'].empty:
            return f"📈 Средний ЛОНГ {SYMBOL}\nЦена: {price:.2f}\nВремя: {pd.Timestamp.now().strftime('%H:%M')}"
        if not ob[ob['type'] == 'bearish'].empty or not fvg[fvg['type'] == 'bearish'].empty:
            return f"📉 Средний ШОРТ {SYMBOL}\nЦена: {price:.2f}\nВремя: {pd.Timestamp.now().strftime('%H:%M')}"

        return None
    except Exception as e:
        logger.error(f"SMC error: {e}")
        return None


# ================== ОСНОВНОЙ ЦИКЛ ==================
async def main():
    bot = Bot(token=TELEGRAM_TOKEN)
    
    # ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
    # Используем TESTNET — чтобы обойти блокировку Binance
    client = await AsyncClient.create(testnet=True)
    # ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←

    bsm = BinanceSocketManager(client)

    logger.info(f"🚀 Бот запущен на Railway | {SYMBOL} | {INTERVAL} | TESTNET")

    # Исторические данные
    klines = await client.get_historical_klines(SYMBOL, INTERVAL, "7 days ago UTC")
    df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','qav','num_trades','tbbav','tbqav','ignore'])
    df = df[['timestamp','open','high','low','close','volume']].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

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
                        logger.info("✅ Сигнал отправлен в Telegram")
                    except Exception as e:
                        logger.error(f"Telegram error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
