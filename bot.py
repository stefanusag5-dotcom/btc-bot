import asyncio
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import pandas_ta as ta
import ccxt
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

exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

logging.basicConfig(level=logging.INFO)

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
            volume_profile[low_bin:high_bin+1] += row['volume'] / bins_count
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers, volume_profile


def find_hvn(volume_profile, bin_centers, current_price):
    threshold = np.percentile(volume_profile, 75)
    peaks = []
    for i in range(1, len(volume_profile)-1):
        if (volume_profile[i] > threshold and 
            volume_profile[i] > volume_profile[i-1] and 
            volume_profile[i] > volume_profile[i+1]):
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


async def ask_gemini(data):
    if not gemini_client:
        return "AI подтверждение отключено"
    prompt = f"""
Ты опытный трейдер. Кратко ответь:
Это классическая сделка? (Да / Нет / С осторожностью)
Рекомендация: LONG / SHORT / HOLD

Монета: {data['symbol']}
Сигнал: {data['signal']}
Причина: {data['reason']}
RSI: {data['rsi']}
POC: {data['poc']}
BTC тренд: {data.get('btc_trend_text', '—')}
"""
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Gemini ошибка: {str(e)[:80]}"


async def fetch_ohlcv(symbol):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, '15m', limit=500)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except:
        return None


def determine_btc_trend(df):
    if not df or len(df) < 50:
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


async def analyze_symbol(symbol):
    df = await fetch_ohlcv(symbol)
    if not df or len(df) < 100:
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
        "time": datetime.now().strftime("%H:%M"),
        "btc_trend_text": ""
    }


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Бот запущен на Railway с AI!\n\nПиши /btc /eth /sol /fartcoin и т.д.")


async def handle_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.lower().strip()
    if not text.startswith('/'):
        return

    cmd = text[1:]
    base = cmd.upper().replace("USDT", "").replace("/", "")
    symbol = f"{base}/USDT"

    msg = await update.message.reply_text(f"🔄 Анализирую {symbol}...")

    df_btc = await fetch_ohlcv("BTC/USDT")
    btc_trend, btc_text = determine_btc_trend(df_btc)

    result = await analyze_symbol(symbol)
    if not result:
        await msg.edit_text("❌ Не удалось получить данные")
        return

    warning = ""
    if result["signal"] == "🟩 LONG" and btc_trend == "DOWNTREND":
        warning = "⚠️ ВНИМАНИЕ: LONG, но BTC в даунтренде!"
    elif result["signal"] == "🟥 SHORT" and btc_trend == "UPTREND":
        warning = "⚠️ ВНИМАНИЕ: SHORT, но BTC в аптренде!"

    gemini_text = await ask_gemini(result) if gemini_client else "AI отключён"

    response = f"""
📊 <b>{result['symbol']}</b> • {result['time']}

Цена: <b>{result['price']}</b>
Сигнал: <b>{result['signal']}</b>

Причина: {result['reason']}
RSI: {result['rsi']}
POC: {result['poc']}

BTC: {btc_text}

{warning}

🧠 Gemini: {gemini_text}
""".strip()

    await msg.edit_text(response, parse_mode='HTML')


def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_command))

    print("🚀 Бот запущен на Railway с Gemini AI")
    app.run_polling()


if __name__ == '__main__':
    main()
