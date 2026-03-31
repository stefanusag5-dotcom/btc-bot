import logging
import asyncio
import aiohttp
from datetime import datetime
import pandas as pd
import numpy as np
import pandas_ta as ta
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
import os
from google import genai

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    gemini_client = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================== DATA SOURCES ==================

async def fetch_binance_futures(ticker: str, session: aiohttp.ClientSession):
    url = f"https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": ticker, "interval": "15m", "limit": 500}
    async with session.get(url, params=params) as resp:
        if resp.status != 200:
            raise Exception(f"Binance Futures HTTP {resp.status}")
        return await resp.json()


async def fetch_binance_spot(ticker: str, session: aiohttp.ClientSession):
    url = f"https://api.binance.com/api/v3/klines"
    params = {"symbol": ticker, "interval": "15m", "limit": 500}
    async with session.get(url, params=params) as resp:
        if resp.status != 200:
            raise Exception(f"Binance Spot HTTP {resp.status}")
        return await resp.json()


async def fetch_bybit(ticker: str, session: aiohttp.ClientSession):
    # Bybit V5 API - Linear (USDT Perpetual)
    url = "https://api.bybit.com/v5/market/kline"
    params = {"category": "linear", "symbol": ticker, "interval": "15", "limit": 500}
    async with session.get(url, params=params) as resp:
        if resp.status != 200:
            raise Exception(f"Bybit HTTP {resp.status}")
        data = await resp.json()
        if data.get("retCode") != 0:
            raise Exception(f"Bybit error: {data.get('retMsg')}")
        # Bybit возвращает [startTime, open, high, low, close, volume, turnover]
        # и в обратном порядке (новые сначала)
        raw = data["result"]["list"]
        raw.reverse()
        return raw, "bybit"


def parse_binance_klines(data):
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def parse_bybit_klines(data):
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
    return df


async def fetch_ohlcv(symbol: str):
    """
    Пробует источники по порядку:
    1. Binance Futures
    2. Binance Spot
    3. Bybit Linear
    Возвращает (df, source_name) или (None, None)
    """
    ticker = symbol.replace("/", "")

    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        # 1. Binance Futures
        try:
            data = await fetch_binance_futures(ticker, session)
            df = parse_binance_klines(data)
            if len(df) >= 100:
                logger.info(f"{symbol}: got data from Binance Futures")
                return df, "Binance Futures"
        except Exception as e:
            logger.warning(f"{symbol} Binance Futures failed: {e}")

        # 2. Binance Spot
        try:
            data = await fetch_binance_spot(ticker, session)
            df = parse_binance_klines(data)
            if len(df) >= 100:
                logger.info(f"{symbol}: got data from Binance Spot")
                return df, "Binance Spot"
        except Exception as e:
            logger.warning(f"{symbol} Binance Spot failed: {e}")

        # 3. Bybit
        try:
            raw, _ = await fetch_bybit(ticker, session)
            df = parse_bybit_klines(raw)
            if len(df) >= 100:
                logger.info(f"{symbol}: got data from Bybit")
                return df, "Bybit"
        except Exception as e:
            logger.warning(f"{symbol} Bybit failed: {e}")

    logger.error(f"{symbol}: ALL sources failed")
    return None, None


# ================== VOLUME PROFILE ==================

def calculate_volume_profile(df: pd.DataFrame, num_bins: int = 80):
    price_min = df['low'].min()
    price_max = df['high'].max()
    bins = np.linspace(price_min, price_max, num_bins + 1)
    volume_profile = np.zeros(num_bins)

    for _, row in df.iterrows():
        low_bin = max(0, np.searchsorted(bins, row['low']) - 1)
        high_bin = min(num_bins - 1, np.searchsorted(bins, row['high']) - 1)
        if low_bin == high_bin:
            volume_profile[low_bin] += row['volume']
        else:
            bins_count = high_bin - low_bin + 1
            volume_profile[low_bin:high_bin + 1] += row['volume'] / bins_count

    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers, volume_profile


def find_hvn(volume_profile, bin_centers, current_price):
    threshold = np.percentile(volume_profile, 75)
    peaks = []
    for i in range(1, len(volume_profile) - 1):
        if (volume_profile[i] > threshold and
                volume_profile[i] > volume_profile[i - 1] and
                volume_profile[i] > volume_profile[i + 1]):
            peaks.append(i)

    hv_nodes = []
    for p in peaks:
        price = bin_centers[p]
        strength = volume_profile[p]
        dist = abs(price - current_price) / current_price * 100
        if dist < 20:
            hv_nodes.append({
                "price": round(float(price), 6),
                "strength": round(float(strength), 2),
                "distance_pct": round(float(dist), 2),
                "is_above": price > current_price
            })
    hv_nodes.sort(key=lambda x: x['strength'], reverse=True)
    return hv_nodes[:10]


async def ask_gemini(data: dict) -> str:
    if not gemini_client:
        return "AI отключён"
    prompt = (
        f"Кратко оцени сделку:\n"
        f"Монета: {data['symbol']}\nСигнал: {data['signal']}\n"
        f"Причина: {data['reason']}\nRSI: {data['rsi']}\n"
        f"POC: {data['poc']}\nBTC тренд: {data.get('btc_trend_text')}"
    )
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: gemini_client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return "Gemini ошибка"


def determine_btc_trend(df):
    if df is None or len(df) < 50:
        return "UNKNOWN", "Не удалось определить тренд BTC"
    close = df['close']
    ema20 = ta.ema(close, length=20).iloc[-1]
    ema50 = ta.ema(close, length=50).iloc[-1]
    current = close.iloc[-1]
    if current > ema20 > ema50:
        return "UPTREND", "🟢 Бычий тренд BTC"
    elif current < ema20 < ema50:
        return "DOWNTREND", "🔴 Медвежий тренд BTC"
    return "SIDEWAYS", "⚪ Боковик BTC"


async def analyze_symbol(symbol: str):
    df, source = await fetch_ohlcv(symbol)
    if df is None:
        return None

    price = df['close'].iloc[-1]
    df['rsi'] = ta.rsi(df['close'], length=14)
    rsi = round(df['rsi'].iloc[-1], 1)

    bin_centers, vol_profile = calculate_volume_profile(df)
    poc_idx = np.argmax(vol_profile)
    poc = round(float(bin_centers[poc_idx]), 6)

    hv_nodes = find_hvn(vol_profile, bin_centers, price)
    strong_above = [z for z in hv_nodes if z['is_above'] and z['distance_pct'] < 12]
    top_hvn = strong_above[0] if strong_above else None

    signal = "НЕТ СИГНАЛА"
    reason = "Нейтральная структура"

    if top_hvn and top_hvn['strength'] > 2.0 * np.mean(vol_profile):
        if rsi > 67:
            signal = "🟥 SHORT"
            reason = f"Огромная полка тепла сверху на {top_hvn['price']}"
        elif rsi < 35:
            signal = "🟩 LONG"
            reason = "Полка тепла сверху + перепроданность"
        else:
            signal = "⚠️ WATCH"
            reason = f"Сильная полка тепла сверху ({top_hvn['price']})"
    elif rsi < 38:
        signal = "🟩 LONG"
        reason = "Перепроданность"

    return {
        "symbol": symbol,
        "price": round(price, 6),
        "signal": signal,
        "reason": reason,
        "rsi": rsi,
        "poc": poc,
        "top_hvn": top_hvn,
        "source": source,
        "time": datetime.now().strftime("%H:%M"),
        "btc_trend_text": ""
    }


# ================== TELEGRAM ==================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ <b>Signal Volume Bot</b> запущен!\n\nПиши /btc /eth /sol /fartcoin",
        parse_mode='HTML'
    )


async def handle_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.lower().strip()
    if not text.startswith('/'):
        return

    cmd = text[1:].split()[0]
    base = cmd.upper().replace("USDT", "").replace("/", "")
    symbol = f"{base}/USDT"

    msg = await update.message.reply_text(f"🔄 Анализирую {symbol}...")

    try:
        btc_task = asyncio.create_task(fetch_ohlcv("BTC/USDT"))
        result_task = asyncio.create_task(analyze_symbol(symbol))
        (df_btc, _), result = await asyncio.gather(btc_task, result_task)

        btc_trend, btc_text = determine_btc_trend(df_btc)

        if not result:
            await msg.edit_text(
                f"❌ Не удалось получить данные для <b>{symbol}</b>.\n\n"
                f"Все источники недоступны (Binance Futures, Spot, Bybit).\n"
                f"Проверь логи Railway для деталей.",
                parse_mode='HTML'
            )
            return

        warning = ""
        if result["signal"] == "🟩 LONG" and btc_trend == "DOWNTREND":
            warning = "\n⚠️ ВНИМАНИЕ: LONG, но BTC в даунтренде!"
        elif result["signal"] == "🟥 SHORT" and btc_trend == "UPTREND":
            warning = "\n⚠️ ВНИМАНИЕ: SHORT, но BTC в аптренде!"

        result["btc_trend_text"] = btc_text
        gemini_text = await ask_gemini(result) if gemini_client else "AI отключён"

        response = (
            f"📊 <b>{result['symbol']}</b> • {result['time']}\n"
            f"<i>Источник: {result['source']}</i>\n\n"
            f"Цена: <b>{result['price']}</b>\n"
            f"Сигнал: <b>{result['signal']}</b>\n\n"
            f"Причина: {result['reason']}\n"
            f"RSI: {result['rsi']}\n"
            f"POC: {result['poc']}\n\n"
            f"BTC: {btc_text}"
            f"{warning}\n\n"
            f"🧠 Gemini: {gemini_text}"
        )

        await msg.edit_text(response, parse_mode='HTML')

    except Exception as e:
        logger.error(f"handle_command error: {e}", exc_info=True)
        await msg.edit_text(f"❌ Ошибка: {e}")


def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("btc", handle_command))
    app.add_handler(CommandHandler("eth", handle_command))
    app.add_handler(CommandHandler("sol", handle_command))
    app.add_handler(CommandHandler("fartcoin", handle_command))
    app.add_handler(MessageHandler(filters.COMMAND, handle_command))

    print("🚀 Signal Volume Bot запущен на Railway")
    app.run_polling()


if __name__ == '__main__':
    main()
    
