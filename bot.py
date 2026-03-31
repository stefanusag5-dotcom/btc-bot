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

# Binance Futures публичный API (не требует ключей)
BINANCE_FUTURES_URL = "https://fapi.binance.com"
# Фолбэк: Binance Spot
BINANCE_SPOT_URL = "https://api.binance.com"


# ================== ПОЛУЧЕНИЕ ДАННЫХ ==================
async def fetch_ohlcv(symbol: str, use_spot: bool = False) -> pd.DataFrame | None:
    """
    Загружает 15m свечи с Binance Futures (или Spot как фолбэк).
    symbol формат: BTC/USDT -> BTCUSDT
    """
    ticker = symbol.replace("/", "")
    base_url = BINANCE_SPOT_URL if use_spot else BINANCE_FUTURES_URL
    endpoint = "/api/v3/klines" if use_spot else "/fapi/v1/klines"
    url = f"{base_url}{endpoint}"
    params = {
        "symbol": ticker,
        "interval": "15m",
        "limit": 500
    }

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15)
        ) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"Binance API {resp.status} for {ticker}: {text[:200]}")
                    return None
                data = await resp.json()

        if not data:
            return None

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

    except aiohttp.ClientConnectorError as e:
        logger.error(f"Connection error fetching {ticker}: {e}")
        return None
    except Exception as e:
        logger.error(f"fetch_ohlcv error for {ticker}: {e}", exc_info=True)
        return None


async def fetch_ohlcv_with_fallback(symbol: str) -> tuple[pd.DataFrame | None, str]:
    """
    Пробует Futures, если не вышло — пробует Spot.
    Возвращает (df, source) где source = 'futures' | 'spot' | 'none'
    """
    df = await fetch_ohlcv(symbol, use_spot=False)
    if df is not None and len(df) >= 100:
        return df, "futures"

    logger.warning(f"{symbol}: futures failed, trying spot...")
    df = await fetch_ohlcv(symbol, use_spot=True)
    if df is not None and len(df) >= 100:
        return df, "spot"

    return None, "none"


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
        f"Монета: {data['symbol']}\n"
        f"Сигнал: {data['signal']}\n"
        f"Причина: {data['reason']}\n"
        f"RSI: {data['rsi']}\n"
        f"POC: {data['poc']}\n"
        f"BTC тренд: {data.get('btc_trend_text')}"
    )
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return "Gemini ошибка"


def determine_btc_trend(df: pd.DataFrame | None) -> tuple[str, str]:
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


async def analyze_symbol(symbol: str) -> dict | None:
    df, source = await fetch_ohlcv_with_fallback(symbol)
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


# ================== TELEGRAM HANDLERS ==================
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
        # Параллельно грузим BTC и нужную монету
        btc_task = asyncio.create_task(fetch_ohlcv_with_fallback("BTC/USDT"))
        result_task = asyncio.create_task(analyze_symbol(symbol))

        (df_btc, _), result = await asyncio.gather(btc_task, result_task)

        btc_trend, btc_text = determine_btc_trend(df_btc)

        if not result:
            await msg.edit_text(
                f"❌ Не удалось получить данные для <b>{symbol}</b>.\n\n"
                f"Проверь: пара существует на Binance Futures или Spot?",
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

        source_label = "📈 Futures" if result["source"] == "futures" else "📊 Spot"

        response = (
            f"📊 <b>{result['symbol']}</b> • {result['time']} • {source_label}\n\n"
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
    
