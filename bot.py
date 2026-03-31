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


# ================== DATA FETCHING ==================

async def fetch_binance_futures(ticker: str, session: aiohttp.ClientSession):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": ticker, "interval": "15m", "limit": 500}
    async with session.get(url, params=params) as resp:
        if resp.status != 200:
            raise Exception(f"Binance Futures HTTP {resp.status}")
        return await resp.json()

async def fetch_binance_spot(ticker: str, session: aiohttp.ClientSession):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": ticker, "interval": "15m", "limit": 500}
    async with session.get(url, params=params) as resp:
        if resp.status != 200:
            raise Exception(f"Binance Spot HTTP {resp.status}")
        return await resp.json()

async def fetch_bybit(ticker: str, session: aiohttp.ClientSession):
    url = "https://api.bybit.com/v5/market/kline"
    params = {"category": "linear", "symbol": ticker, "interval": "15", "limit": 500}
    async with session.get(url, params=params) as resp:
        if resp.status != 200:
            raise Exception(f"Bybit HTTP {resp.status}")
        data = await resp.json()
        if data.get("retCode") != 0:
            raise Exception(f"Bybit error: {data.get('retMsg')}")
        raw = data["result"]["list"]
        raw.reverse()
        return raw

async def fetch_funding_rate(ticker: str, session: aiohttp.ClientSession) -> float | None:
    """Фандинг рейт с Binance Futures"""
    try:
        url = "https://fapi.binance.com/fapi/v1/premiumIndex"
        params = {"symbol": ticker}
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            return float(data.get("lastFundingRate", 0)) * 100  # в процентах
    except:
        return None

async def fetch_open_interest(ticker: str, session: aiohttp.ClientSession) -> float | None:
    """Открытый интерес с Binance Futures"""
    try:
        url = "https://fapi.binance.com/fapi/v1/openInterest"
        params = {"symbol": ticker}
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            return float(data.get("openInterest", 0))
    except:
        return None

def parse_binance_klines(data) -> pd.DataFrame:
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume',
             'taker_buy_base', 'quote_volume']].copy()
    for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base', 'quote_volume']:
        df[col] = pd.to_numeric(df[col])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def parse_bybit_klines(data) -> pd.DataFrame:
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
    df['taker_buy_base'] = df['volume'] / 2  # Bybit не даёт дельту, ставим нейтральное
    df['quote_volume'] = df['volume'] * df['close']
    return df

async def fetch_ohlcv(symbol: str):
    ticker = symbol.replace("/", "")
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            data = await fetch_binance_futures(ticker, session)
            df = parse_binance_klines(data)
            if len(df) >= 100:
                fr = await fetch_funding_rate(ticker, session)
                oi = await fetch_open_interest(ticker, session)
                return df, "Binance Futures", fr, oi
        except Exception as e:
            logger.warning(f"{symbol} Binance Futures: {e}")

        try:
            data = await fetch_binance_spot(ticker, session)
            df = parse_binance_klines(data)
            if len(df) >= 100:
                return df, "Binance Spot", None, None
        except Exception as e:
            logger.warning(f"{symbol} Binance Spot: {e}")

        try:
            data = await fetch_bybit(ticker, session)
            df = parse_bybit_klines(data)
            if len(df) >= 100:
                return df, "Bybit", None, None
        except Exception as e:
            logger.warning(f"{symbol} Bybit: {e}")

    return None, None, None, None


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
            volume_profile[low_bin:high_bin + 1] += row['volume'] / (high_bin - low_bin + 1)
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


# ================== TECHNICAL ANALYSIS ==================

def detect_candle_pattern(df: pd.DataFrame) -> str:
    """Определяет паттерн последних 2 свечей"""
    c = df.iloc[-1]
    p = df.iloc[-2]
    body = abs(c['close'] - c['open'])
    full_range = c['high'] - c['low']
    if full_range == 0:
        return "Дожи"

    upper_wick = c['high'] - max(c['close'], c['open'])
    lower_wick = min(c['close'], c['open']) - c['low']

    # Пин-бар бычий
    if lower_wick > body * 2 and upper_wick < body * 0.5:
        return "📌 Бычий пин-бар"
    # Пин-бар медвежий
    if upper_wick > body * 2 and lower_wick < body * 0.5:
        return "📌 Медвежий пин-бар"
    # Бычье поглощение
    if (c['close'] > c['open'] and p['close'] < p['open'] and
            c['close'] > p['open'] and c['open'] < p['close']):
        return "🟢 Бычье поглощение"
    # Медвежье поглощение
    if (c['close'] < c['open'] and p['close'] > p['open'] and
            c['close'] < p['open'] and c['open'] > p['close']):
        return "🔴 Медвежье поглощение"
    # Дожи
    if body < full_range * 0.1:
        return "〰️ Дожи (нерешительность)"

    return "Обычная свеча"

def calculate_delta(df: pd.DataFrame) -> str:
    """Дельта объёма (покупатели vs продавцы) за последние 5 свечей"""
    recent = df.tail(5).copy()
    if 'taker_buy_base' not in recent.columns:
        return "N/A"
    buy_vol = recent['taker_buy_base'].sum()
    total_vol = recent['volume'].sum()
    if total_vol == 0:
        return "N/A"
    buy_pct = buy_vol / total_vol * 100
    sell_pct = 100 - buy_pct
    emoji = "🟢" if buy_pct > 55 else ("🔴" if sell_pct > 55 else "⚪")
    return f"{emoji} Покупки {buy_pct:.0f}% / Продажи {sell_pct:.0f}%"

def find_support_resistance(df: pd.DataFrame, current_price: float):
    """Простые уровни по swing high/low за последние 100 свечей"""
    recent = df.tail(100)
    highs = []
    lows = []
    for i in range(2, len(recent) - 2):
        h = recent.iloc[i]['high']
        l = recent.iloc[i]['low']
        if h > recent.iloc[i-1]['high'] and h > recent.iloc[i+1]['high']:
            highs.append(h)
        if l < recent.iloc[i-1]['low'] and l < recent.iloc[i+1]['low']:
            lows.append(l)

    # Ближайшие уровни
    resistances = sorted([h for h in highs if h > current_price])[:3]
    supports = sorted([l for l in lows if l < current_price], reverse=True)[:3]
    return supports, resistances

def calculate_sl_tp(signal: str, price: float, atr: float,
                    hv_nodes: list, vol_profile, bin_centers) -> dict:
    """
    Рассчитывает СЛ и 3 ТП на основе ATR и HVN.
    Логика ведения сделки:
    - ТП1 достигнут → переносим СЛ в безубыток
    - ТП2 достигнут → переносим СЛ за ТП1
    - ТП3 — финальная цель
    """
    if signal not in ["🟩 LONG", "🟥 SHORT"]:
        return {}

    is_long = signal == "🟩 LONG"
    atr_mult_sl = 1.5  # буфер для СЛ

    # СЛ — за ближайший HVN в противоположную сторону или ATR
    nodes_below = [n for n in hv_nodes if not n['is_above']]
    nodes_above = [n for n in hv_nodes if n['is_above']]

    if is_long:
        # СЛ под ближайшим HVN снизу или ATR*1.5
        if nodes_below:
            sl_hvn = min(nodes_below, key=lambda x: x['distance_pct'])
            sl = min(price - atr * atr_mult_sl, sl_hvn['price'] - atr * 0.3)
        else:
            sl = price - atr * atr_mult_sl
        sl = round(sl, 6)

        risk = price - sl
        tp1 = round(price + risk * 1.5, 6)   # RR 1.5
        tp2 = round(price + risk * 2.5, 6)   # RR 2.5
        # ТП3 — следующий сильный HVN сверху или RR 4
        if nodes_above:
            strong_above = sorted(nodes_above, key=lambda x: -x['strength'])
            tp3_hvn = strong_above[0]['price']
            tp3 = round(max(tp3_hvn, price + risk * 3.5), 6)
        else:
            tp3 = round(price + risk * 4.0, 6)

        rr1 = round(risk * 1.5 / risk, 1)
        rr2 = round(risk * 2.5 / risk, 1)

    else:  # SHORT
        if nodes_above:
            sl_hvn = min(nodes_above, key=lambda x: x['distance_pct'])
            sl = max(price + atr * atr_mult_sl, sl_hvn['price'] + atr * 0.3)
        else:
            sl = price + atr * atr_mult_sl
        sl = round(sl, 6)

        risk = sl - price
        tp1 = round(price - risk * 1.5, 6)
        tp2 = round(price - risk * 2.5, 6)
        if nodes_below:
            strong_below = sorted(nodes_below, key=lambda x: -x['strength'])
            tp3_hvn = strong_below[0]['price']
            tp3 = round(min(tp3_hvn, price - risk * 3.5), 6)
        else:
            tp3 = round(price - risk * 4.0, 6)

        rr1 = 1.5
        rr2 = 2.5

    rr_ratio = round((abs(tp2 - price)) / (abs(sl - price)), 2) if abs(sl - price) > 0 else 0

    return {
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "risk_pct": round(abs(price - sl) / price * 100, 2),
        "rr_ratio": rr_ratio,
        "be_trigger": tp1,  # после ТП1 → перенос в безубыток
    }

def determine_btc_trend(df) -> tuple:
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


# ================== AI PROMPT ==================

async def ask_gemini(data: dict) -> str:
    if not gemini_client:
        return "AI отключён"

    sl_tp = data.get("sl_tp", {})
    supports = data.get("supports", [])
    resistances = data.get("resistances", [])

    prompt = f"""Ты опытный крипто-трейдер. Проанализируй сделку и дай КОНКРЕТНЫЙ вывод.

=== МОНЕТА: {data['symbol']} ({data['source']}) ===
Цена входа: {data['price']}
Таймфрейм: 15m (интрадей)
Сигнал: {data['signal']}
Причина сигнала: {data['reason']}

=== ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ ===
RSI(14): {data['rsi']} {'(перекупленность)' if data['rsi'] > 70 else '(перепроданность)' if data['rsi'] < 30 else '(нейтрально)'}
ATR(14): {data.get('atr', 'N/A')} ({data.get('atr_pct', 'N/A')}% от цены)
EMA20: {data.get('ema20', 'N/A')} | EMA50: {data.get('ema50', 'N/A')} | EMA200: {data.get('ema200', 'N/A')}
Тренд EMA: {data.get('ema_trend', 'N/A')}
Паттерн свечи: {data.get('candle_pattern', 'N/A')}
Дельта объёма (5 свечей): {data.get('delta', 'N/A')}

=== VOLUME PROFILE ===
POC (точка контроля): {data['poc']}
Ближайшие HVN сверху: {[n['price'] for n in data.get('hvn_above', [])][:3]}
Ближайшие HVN снизу: {[n['price'] for n in data.get('hvn_below', [])][:3]}

=== УРОВНИ РЫНКА ===
Сопротивления (swing high): {resistances[:3]}
Поддержки (swing low): {supports[:3]}

=== ДАННЫЕ ФЬЮЧЕРСОВ ===
Фандинг рейт: {data.get('funding_rate', 'N/A')}% {'(лонги платят — давление вниз)' if data.get('funding_rate') and data['funding_rate'] > 0.05 else '(шорты платят — давление вверх)' if data.get('funding_rate') and data['funding_rate'] < -0.05 else ''}
Открытый интерес: {data.get('open_interest', 'N/A')}

=== BTC КОНТЕКСТ ===
BTC тренд: {data.get('btc_trend_text', 'N/A')}

=== ПЛАН СДЕЛКИ ===
СЛ: {sl_tp.get('sl', 'N/A')} (риск {sl_tp.get('risk_pct', 'N/A')}% от цены)
ТП1: {sl_tp.get('tp1', 'N/A')} → после достижения перенос СЛ в безубыток ({data['price']})
ТП2: {sl_tp.get('tp2', 'N/A')} → после достижения СЛ на уровень ТП1
ТП3: {sl_tp.get('tp3', 'N/A')} (финальная цель)
R/R итоговый: 1:{sl_tp.get('rr_ratio', 'N/A')}

Ответь СТРОГО по этому формату (кратко, по делу):
✅ ВЫВОД: [войти / не войти / подождать подтверждения]
💪 СИЛА СИГНАЛА: [слабый / средний / сильный] — одним словом
⚠️ ГЛАВНЫЙ РИСК: [одно предложение]
💡 СОВЕТ: [одно конкретное действие трейдеру]"""

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


# ================== MAIN ANALYSIS ==================

async def analyze_symbol(symbol: str) -> dict | None:
    df, source, funding_rate, open_interest = await fetch_ohlcv(symbol)
    if df is None:
        return None

    price = df['close'].iloc[-1]

    # Индикаторы
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    rsi = round(df['rsi'].iloc[-1], 1)
    atr = round(df['atr'].iloc[-1], 6)
    atr_pct = round(atr / price * 100, 2)

    ema20 = round(ta.ema(df['close'], length=20).iloc[-1], 6)
    ema50 = round(ta.ema(df['close'], length=50).iloc[-1], 6)
    ema200_series = ta.ema(df['close'], length=200)
    ema200 = round(ema200_series.iloc[-1], 6) if ema200_series is not None else None

    if price > ema20 > ema50:
        ema_trend = "📈 Восходящий (цена > EMA20 > EMA50)"
    elif price < ema20 < ema50:
        ema_trend = "📉 Нисходящий (цена < EMA20 < EMA50)"
    else:
        ema_trend = "↔️ Боковик / смешанный"

    # Volume Profile
    bin_centers, vol_profile = calculate_volume_profile(df)
    poc_idx = np.argmax(vol_profile)
    poc = round(float(bin_centers[poc_idx]), 6)
    hv_nodes = find_hvn(vol_profile, bin_centers, price)
    hvn_above = [n for n in hv_nodes if n['is_above']]
    hvn_below = [n for n in hv_nodes if not n['is_above']]

    # Уровни S/R
    supports, resistances = find_support_resistance(df, price)

    # Паттерн и дельта
    candle_pattern = detect_candle_pattern(df)
    delta = calculate_delta(df)

    # Сигнал
    strong_above = [z for z in hvn_above if z['distance_pct'] < 12]
    top_hvn = strong_above[0] if strong_above else None

    signal = "НЕТ СИГНАЛА"
    reason = "Нейтральная структура"

    if top_hvn and top_hvn['strength'] > 2.0 * np.mean(vol_profile):
        if rsi > 67:
            signal = "🟥 SHORT"
            reason = f"Сильная полка тепла сверху на {top_hvn['price']} + RSI перекуплен"
        elif rsi < 35:
            signal = "🟩 LONG"
            reason = f"Полка тепла сверху на {top_hvn['price']} + RSI перепродан"
        else:
            signal = "⚠️ WATCH"
            reason = f"Сильная полка тепла сверху ({top_hvn['price']}), ждём подтверждения"
    elif rsi < 38:
        signal = "🟩 LONG"
        reason = "Перепроданность RSI"
    elif rsi > 67:
        signal = "🟥 SHORT"
        reason = "Перекупленность RSI"

    # ТП/СЛ
    sl_tp = calculate_sl_tp(signal, price, atr, hv_nodes, vol_profile, bin_centers)

    return {
        "symbol": symbol,
        "price": round(price, 6),
        "signal": signal,
        "reason": reason,
        "rsi": rsi,
        "atr": atr,
        "atr_pct": atr_pct,
        "poc": poc,
        "hvn_above": hvn_above,
        "hvn_below": hvn_below,
        "supports": [round(s, 6) for s in supports],
        "resistances": [round(r, 6) for r in resistances],
        "candle_pattern": candle_pattern,
        "delta": delta,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "ema_trend": ema_trend,
        "sl_tp": sl_tp,
        "source": source,
        "funding_rate": round(funding_rate, 4) if funding_rate is not None else None,
        "open_interest": round(open_interest, 0) if open_interest is not None else None,
        "time": datetime.now().strftime("%H:%M"),
        "btc_trend_text": ""
    }


# ================== TELEGRAM ==================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ <b>Signal Volume Bot</b> запущен!\n\nПиши /btc /eth /sol /fartcoin или любую монету: /bnb /xrp /doge",
        parse_mode='HTML'
    )

def format_message(result: dict, btc_text: str, warning: str, gemini_text: str) -> str:
    sl_tp = result.get("sl_tp", {})
    p = result['price']

    # Блок ТП/СЛ
    trade_block = ""
    if sl_tp:
        def pct(target):
            return round(abs(target - p) / p * 100, 2)

        trade_block = (
            f"\n📐 <b>ПЛАН СДЕЛКИ</b>\n"
            f"├ 🛑 СЛ: <b>{sl_tp['sl']}</b> (-{pct(sl_tp['sl'])}%)\n"
            f"├ 🎯 ТП1: <b>{sl_tp['tp1']}</b> (+{pct(sl_tp['tp1'])}%) → <i>перенос в БУ</i>\n"
            f"├ 🎯 ТП2: <b>{sl_tp['tp2']}</b> (+{pct(sl_tp['tp2'])}%) → <i>СЛ на ТП1</i>\n"
            f"└ 🏆 ТП3: <b>{sl_tp['tp3']}</b> (+{pct(sl_tp['tp3'])}%) → <i>финал</i>\n"
            f"R/R: 1:{sl_tp['rr_ratio']} | Риск: {sl_tp['risk_pct']}% от цены\n"
        )

    # Фандинг
    funding_str = ""
    if result.get("funding_rate") is not None:
        fr = result["funding_rate"]
        fr_emoji = "🔴" if fr > 0.05 else ("🟢" if fr < -0.05 else "⚪")
        funding_str = f"\nФандинг: {fr_emoji} {fr}%"

    # OI
    oi_str = ""
    if result.get("open_interest") is not None:
        oi_val = int(r
