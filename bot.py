import asyncio
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import pandas_ta as ta
import ccxt
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
else:
    gemini_model = None

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
            volume_profile[low_bin:high_bin+1] += row['volume'] / (high_bin - low_bin + 1)
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers, volume_profile

def find_hvn(volume_profile, bin_centers, current_price):
    threshold = np.percentile(volume_profile, 75)
    peaks = []
    for i in range(1, len(volume_profile)-1):
        if volume_profile[i] > threshold and volume_profile[i] > volume_profile[i-1] and volume_profile[i] > volume_profile[i+1]:
            peaks.append(i)
    hv_nodes = []
    for p in peaks:
        price = bin_centers[p]
        strength = volume_profile[p]
        dist = abs(price - current_price) / current_price * 100
        if dist < 20:
            hv_nodes.append({"price": round(float(price),6), "strength": round(float(strength),2), "distance_pct": round(float(dist),2), "is_above": price > current_price})
    hv_nodes.sort(key=lambda x: x['strength'], reverse=True)
    return hv_nodes[:10]

# ================== Gemini ==================
async def ask_gemini(data):
    if not gemini_model:
        return "AI подтверждение недоступно"
    prompt = f"Кратко: это классическая сделка? LONG/SHORT/HOLD?\nМонета: {data['symbol']}\nСигнал: {data['signal']}\nПричина: {data['reason']}\nRSI: {data['rsi']}\nPOC: {data['poc']}\nVAH/VAL: {data['vah']}/{data['val']}\nBTC: {data.get('btc_trend_text')}"
    try:
        resp = gemini_model.generate_content(prompt)
        return resp.text.strip()
    except:
        return "Ошибка Gemini"

# ================== Анализ ==================
async def fetch_ohlcv(symbol):
    try:
        data = exchange.fetch_ohlcv(symbol, '15m', limit=500)
        df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except:
        return None

def determine_btc_trend(df):
    if not df or len(df) < 50:
        return "UNKNOWN", "Не удалось определить тренд BTC"
    close = df['close']
    ema20 = ta.ema(close, 20).iloc[-1]
    ema50 = ta.ema(close, 50).iloc[-1]
    curr = close.iloc[-1]
    if curr > ema20 > ema50:
        return "UPTREND", "🟢 Бычий тренд BTC"
    elif curr < ema20 < ema50:
        return "DOWNTREND", "🔴 Медвежий тренд BTC"
    return "SIDEWAYS", "⚪ Боковик"

async def analyze_symbol(symbol):
    df = await fetch_ohlcv(symbol)
    if not df or len(df) < 100:
        return None
    price = df['close'].iloc[-1]
    df['rsi'] = ta.rsi(df['close'], 14)
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
            reason = f"Мощная полка тепла сверху ({top_hvn['price']})"
        elif rsi < 35:
            signal = "🟩 LONG"
            reason = "Полка сверху + перепроданность"
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
        "vah": "—",
        "val": "—",
        "top_hvn": top_hvn,
        "time": datetime.now().strftime("%H:%M")
    }

# ================== Команды ==================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Бот запущен!\nПиши /btc /eth /sol /fartcoin и т.д.")

async def handle_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cmd = update.message.text.lower().strip()
    if not cmd.startswith('/'):
        return
    base = cmd[1:].upper().replace("USDT", "").replace("/", "")
    symbol = f"{base}/USDT"

    await update.message.reply_text(f"🔄 Анализирую {symbol}...")

    df_btc = await fetch_ohlcv("BTC/USDT")
    btc_trend, btc_text = determine_btc_trend(df_btc)

    result = await analyze_symbol(symbol)
    if not result:
        await update.message.reply_text("❌ Не удалось получить данные")
        return

    warning = ""
    if result["signal"] == "🟩 LONG" and btc_trend == "DOWNTREND":
        warning = "⚠️ ВНИМАНИЕ: LONG, но BTC в даунтренде!"
    elif result["signal"] == "🟥 SHORT" and btc_trend == "UPTREND":
        warning = "⚠️ ВНИМАНИЕ: SHORT, но BTC в аптренде!"

    gemini_text = await ask_gemini(result) if gemini_model else ""

    text = f"""
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

    await update.message.reply_text(text, parse_mode='HTML')

# ================== Фоновый сканер BTC ==================
async def btc_scanner(context: ContextTypes.DEFAULT_TYPE):
    while True:
        try:
            df_btc = await fetch_ohlcv("BTC/USDT")
            btc_trend, btc_text = determine_btc_trend(df_btc)
            result = await analyze_symbol("BTC/USDT")
            if result and result["signal"] != "НЕТ СИГНАЛА":
                print(f"[AUTO] BTC {result['signal']} — {result['reason']}")
                # Можно отправлять в твой чат, если добавишь chat_id
        except Exception as e:
            print(f"Scanner error: {e}")
        await asyncio.sleep(900)  # 15 минут

# ================== ЗАПУСК ==================
async def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & \~filters.COMMAND, handle_command))

    # Запускаем фоновый сканер
    job_queue = app.job_queue
    if job_queue:
        job_queue.run_repeating(btc_scanner, interval=900, first=10)

    print("🚀 Бот запущен на Railway")
    await app.run_polling()

if __name__ == '__main__':
    asyncio.run(main())
