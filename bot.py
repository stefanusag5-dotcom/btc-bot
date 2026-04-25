import logging
import asyncio
import aiohttp
import json
import os
import re
import hmac
import hashlib
import time
from html import escape as html_escape
from datetime import datetime
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import pandas_ta as ta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]

MEXC_API_KEY    = os.getenv("MEXC_API_KEY", "")
MEXC_SECRET_KEY = os.getenv("MEXC_SECRET_KEY", "")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRADES_FILE = Path("open_trades.json")
SCANNER_FILE = Path("scanner_state.json")

TF_MAP = {
    "15m": ("15m", "15",  "15m", 1000),
    "1h":  ("1h",  "60",  "1h",  1000),
    "4h":  ("4h",  "240", "4h",  500),
    "1d":  ("1d",  "D",   "1d",  365),
}
DEFAULT_TF = "15m"

TRADE_MODES = {
    "low": {
        "label": "🟢 LOW",
        "rsi_long": 32, "rsi_short": 72, "hvn_mult": 2.5,
        "personality": "Ты консервативный трейдер. Торгуешь только очень чёткие сигналы. При слабом — говори ПРОПУСТИТЬ."
    },
    "mid": {
        "label": "🟡 MID",
        "rsi_long": 38, "rsi_short": 67, "hvn_mult": 2.0,
        "personality": "Ты сбалансированный интрадей трейдер. Торгуй средние и сильные сигналы."
    },
    "hard": {
        "label": "🔴 HARD",
        "rsi_long": 45, "rsi_short": 58, "hvn_mult": 1.5,
        "personality": "Ты агрессивный скальпер. НИКОГДА не говори 'дождитесь подтверждения'. Давай конкретный вход прямо сейчас."
    },
}

# ================== СДЕЛКИ ==================
def load_trades() -> dict:
    try:
        return json.loads(TRADES_FILE.read_text()) if TRADES_FILE.exists() else {}
    except: return {}

def save_trades(trades: dict):
    TRADES_FILE.write_text(json.dumps(trades, indent=2, ensure_ascii=False))

def open_trade(symbol, tf, data, chat_id):
    trades = load_trades()
    key = f"{symbol.replace('/','')}{tf}"
    trades[key] = {
        "symbol": symbol, "tf": tf, "chat_id": chat_id,
        "entry": data["price"], "signal": data["signal"],
        "sl": data["sl_tp"]["sl"],
        "tp1": data["sl_tp"]["tp1"],
        "tp2": data["sl_tp"]["tp2"],
        "tp3": data["sl_tp"]["tp3"],
        "sl_moved_be": False, "sl_moved_tp1": False,
        "tp1_hit": False, "tp2_hit": False,
        "opened_at": datetime.now().isoformat(),
    }
    save_trades(trades)
    return key

def close_trade(key):
    trades = load_trades()
    trades.pop(key, None)
    save_trades(trades)

# ================== СКАНЕР ==================
def load_scanner_state() -> dict:
    try:
        return json.loads(SCANNER_FILE.read_text()) if SCANNER_FILE.exists() else {}
    except: return {}

def save_scanner_state(state: dict):
    SCANNER_FILE.write_text(json.dumps(state, ensure_ascii=False))

# ================== DATA FETCHING ==================
async def _fetch_klines(ticker, interval_bn, interval_bb, limit, session):
    # Binance Futures
    try:
        async with session.get("https://fapi.binance.com/fapi/v1/klines",
                params={"symbol": ticker, "interval": interval_bn, "limit": limit}) as r:
            if r.status == 200:
                data = await r.json()
                if data: return _parse_binance(data), "Binance Futures"
    except Exception as e: logger.warning(f"BF {ticker}: {e}")

    # Binance Spot
    try:
        async with session.get("https://api.binance.com/api/v3/klines",
                params={"symbol": ticker, "interval": interval_bn, "limit": limit}) as r:
            if r.status == 200:
                data = await r.json()
                if data: return _parse_binance(data), "Binance Spot"
    except Exception as e: logger.warning(f"BS {ticker}: {e}")

    # Bybit fallback
    try:
        async with session.get("https://api.bybit.com/v5/market/kline",
                params={"category": "linear", "symbol": ticker,
                        "interval": interval_bb, "limit": limit}) as r:
            if r.status == 200:
                data = await r.json()
                if data.get("retCode") == 0:
                    raw = list(reversed(data["result"]["list"]))
                    return _parse_bybit(raw), "Bybit"
    except Exception as e: logger.warning(f"Bybit {ticker}: {e}")

    return None, None

def _parse_binance(data):
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume','ct','qv','trades','tbb','tbq','ignore'])
    df = df[['ts','open','high','low','close','volume','tbb']].copy()
    for c in ['open','high','low','close','volume','tbb']: df[c] = pd.to_numeric(df[c])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    return df.rename(columns={'ts':'timestamp','tbb':'taker_buy_base'})

def _parse_bybit(data):
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume','turnover'])
    df = df[['ts','open','high','low','close','volume']].copy()
    for c in ['open','high','low','close','volume']: df[c] = pd.to_numeric(df[c])
    df['ts'] = pd.to_datetime(pd.to_numeric(df['ts']), unit='ms')
    df = df.rename(columns={'ts':'timestamp'})
    df['taker_buy_base'] = df['volume'] / 2
    return df

async def fetch_ohlcv(symbol, tf="15m"):
    ticker = symbol.replace("/", "")
    cfg = TF_MAP.get(tf, TF_MAP[DEFAULT_TF])
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        df, source = await _fetch_klines(ticker, cfg[0], cfg[1], cfg[3], session)
        fr, oi = None, None
        if source == "Binance Futures":
            try:
                async with session.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                                        params={"symbol": ticker}) as r:
                    if r.status == 200:
                        d = await r.json()
                        fr = float(d.get("lastFundingRate", 0)) * 100
            except: pass
            try:
                async with session.get("https://fapi.binance.com/fapi/v1/openInterest",
                                        params={"symbol": ticker}) as r:
                    if r.status == 200:
                        d = await r.json()
                        oi = float(d.get("openInterest", 0))
            except: pass
    return df, source, fr, oi

async def fetch_binance_futures_symbols(limit: int = 150) -> list:
    """Топ символы с Binance Futures по объёму, фильтр $10M суточного объёма"""
    try:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get("https://fapi.binance.com/fapi/v1/ticker/24hr") as r:
                if r.status == 200:
                    data = await r.json()
                    pairs = [
                        d for d in data
                        if d['symbol'].endswith('USDT')
                        and float(d.get('quoteVolume', 0)) >= 10_000_000  # фильтр $10M+
                    ]
                    pairs.sort(key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)
                    return [p['symbol'].replace('USDT', '/USDT') for p in pairs[:limit]]
    except Exception as e:
        logger.error(f"fetch_symbols: {e}")
    return []

async def fetch_higher_tf(symbol, tf):
    higher = {"15m": "1h", "1h": "4h", "4h": "1d", "1d": "1d"}
    htf = higher.get(tf, "1h")
    df, _, _, _ = await fetch_ohlcv(symbol, htf)
    return df, htf

async def fetch_weekly_trend(symbol: str) -> str:
    """Недельный тренд как контекст — только подсказка, не фильтр"""
    cache_key = f"weekly_{symbol}"
    now = datetime.now().timestamp()
    if cache_key in _cache and now - _cache[cache_key]['ts'] < 3600:  # кеш 1 час
        return _cache[cache_key]['val']
    try:
        ticker = symbol.replace("/", "")
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                "https://fapi.binance.com/fapi/v1/klines",
                params={"symbol": ticker, "interval": "1w", "limit": 26}
            ) as r:
                if r.status != 200:
                    raise Exception(f"HTTP {r.status}")
                data = await r.json()
        if not data or len(data) < 10:
            return ""
        closes = [float(d[4]) for d in data]
        cur = closes[-1]
        ema10 = float(pd.Series(closes).ewm(span=10).mean().iloc[-1])
        ema20 = float(pd.Series(closes).ewm(span=20).mean().iloc[-1])
        change_4w = round((closes[-1] - closes[-4]) / closes[-4] * 100, 1)

        if cur > ema10 > ema20 and change_4w > 5:
            label = f"📅 Неделя: сильный аптренд (+{change_4w}% за 4 нед)"
        elif cur > ema10 > ema20:
            label = f"📅 Неделя: аптренд (+{change_4w}% за 4 нед)"
        elif cur < ema10 < ema20 and change_4w < -5:
            label = f"📅 Неделя: сильный даунтренд ({change_4w}% за 4 нед)"
        elif cur < ema10 < ema20:
            label = f"📅 Неделя: даунтренд ({change_4w}% за 4 нед)"
        else:
            label = f"📅 Неделя: боковик ({change_4w:+.1f}% за 4 нед)"

        _cache[cache_key] = {'val': label, 'ts': now}
        return label
    except Exception as e:
        logger.warning(f"weekly trend {symbol}: {e}")
        return ""

async def fetch_daily_vp(symbol):
    ticker = symbol.replace("/", "")
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        df, _ = await _fetch_klines(ticker, "1d", "D", 1000, session)
    return df

# ================== VOLUME PROFILE (векторизованный) ==================
def calculate_volume_profile(df, num_bins=120):
    """
    Полностью векторизованный Volume Profile на numpy.
    Быстрее iterrows в 15-20x на 1000 свечах.
    """
    price_min = df['low'].min()
    price_max = df['high'].max()
    if price_min == price_max:
        return np.array([price_min]), np.array([df['volume'].sum()])

    bins = np.linspace(price_min, price_max, num_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    vp = np.zeros(num_bins)

    lows    = df['low'].values
    highs   = df['high'].values
    volumes = df['volume'].values

    # Индексы бинов для каждой свечи — векторизованно
    lo_idx = np.searchsorted(bins, lows,  side='left')  - 1
    hi_idx = np.searchsorted(bins, highs, side='right') - 1
    lo_idx = np.clip(lo_idx, 0, num_bins - 1)
    hi_idx = np.clip(hi_idx, 0, num_bins - 1)

    for i in range(len(volumes)):
        lo, hi = lo_idx[i], hi_idx[i]
        if lo == hi:
            vp[lo] += volumes[i]
        else:
            spread = hi - lo + 1
            vp[lo:hi+1] += volumes[i] / spread

    return centers, vp

def find_hvn(vp, centers, price, dist_limit=25):
    threshold = np.percentile(vp, 70)
    nodes = []
    for i in range(1, len(vp)-1):
        if vp[i] > threshold and vp[i] > vp[i-1] and vp[i] > vp[i+1]:
            dist = abs(centers[i] - price) / price * 100
            if dist < dist_limit:
                nodes.append({
                    "price": round(float(centers[i]), 6),
                    "strength": round(float(vp[i]), 2),
                    "distance_pct": round(float(dist), 2),
                    "is_above": centers[i] > price,
                    "type": "local"
                })
    nodes.sort(key=lambda x: -x['strength'])
    return nodes[:12]

def merge_hvn_levels(local_nodes, daily_nodes):
    for n in daily_nodes: n['type'] = 'daily'
    merged = []
    for n in sorted(local_nodes + daily_nodes, key=lambda x: -x['strength']):
        if not any(abs(n['price'] - m['price']) / max(n['price'], 0.0001) * 100 < 0.5 for m in merged):
            merged.append(n)
    return sorted(merged, key=lambda x: x['price'])

# ================== ТЕХНИЧЕСКИЙ АНАЛИЗ ==================
def detect_candle_pattern(df):
    c, p = df.iloc[-1], df.iloc[-2]
    body = abs(c['close'] - c['open'])
    rng = c['high'] - c['low']
    if rng == 0: return "Дожи"
    uw = c['high'] - max(c['close'], c['open'])
    lw = min(c['close'], c['open']) - c['low']
    if lw > body*2 and uw < body*0.5: return "📌 Бычий пин-бар"
    if uw > body*2 and lw < body*0.5: return "📌 Медвежий пин-бар"
    if c['close']>c['open'] and p['close']<p['open'] and c['close']>p['open'] and c['open']<p['close']: return "🟢 Бычье поглощение"
    if c['close']<c['open'] and p['close']>p['open'] and c['close']<p['open'] and c['open']>p['close']: return "🔴 Медвежье поглощение"
    if body < rng*0.1: return "〰️ Дожи"
    return "Обычная свеча"

def calculate_delta(df):
    r = df.tail(5)
    if 'taker_buy_base' not in r.columns: return "N/A"
    bv, tv = r['taker_buy_base'].sum(), r['volume'].sum()
    if tv == 0: return "N/A"
    bp = bv / tv * 100
    e = "🟢" if bp > 55 else ("🔴" if bp < 45 else "⚪")
    return f"{e} {bp:.0f}% покупок / {100-bp:.0f}% продаж"

# ================== РЕЖИМ РЫНКА ==================
def detect_market_regime(df: pd.DataFrame, atr: float) -> dict:
    """
    Определяет режим рынка: trending_up / trending_down / ranging / volatile.

    Логика:
    - EMA slope (угол наклона EMA50 за 10 свечей) — насколько силён тренд
    - ATR% vs среднее ATR% за 50 свечей — волатильность выше нормы?
    - EMA расстояние — цена далеко от EMA20?

    Возвращает режим и корректирующие коэффициенты для скоринга.
    """
    if len(df) < 60:
        return {"regime": "unknown", "label": "❓ Режим неизвестен",
                "trend_score_mult": 1.0, "hvn_score_mult": 1.0}

    close  = df['close'].values
    price  = close[-1]

    # EMA slopes
    ema20_arr = ta.ema(pd.Series(close), length=20).values
    ema50_arr = ta.ema(pd.Series(close), length=50).values

    # Убираем NaN
    ema20_valid = ema20_arr[~np.isnan(ema20_arr)]
    ema50_valid = ema50_arr[~np.isnan(ema50_arr)]
    if len(ema20_valid) < 10 or len(ema50_valid) < 10:
        return {"regime": "unknown", "label": "❓ Режим неизвестен",
                "trend_score_mult": 1.0, "hvn_score_mult": 1.0}

    # Наклон EMA50 за последние 10 свечей (в % от цены)
    ema50_slope = (ema50_valid[-1] - ema50_valid[-10]) / ema50_valid[-10] * 100

    # ATR% сейчас vs среднее за 50 свечей
    atr_series = ta.atr(df['high'], df['low'], df['close'], length=14).dropna()
    if len(atr_series) < 20:
        atr_mean = float(atr_series.mean())
    else:
        atr_mean = float(atr_series.iloc[-50:].mean())
    atr_now   = float(atr_series.iloc[-1]) if len(atr_series) > 0 else atr
    vol_ratio = atr_now / atr_mean if atr_mean > 0 else 1.0

    # Расстояние цены от EMA20 в %
    ema20_dist = abs(price - ema20_valid[-1]) / price * 100

    # Определяем режим
    is_volatile = vol_ratio > 1.5
    is_trending_up   = ema50_slope >  0.3 and price > ema20_valid[-1] > ema50_valid[-1]
    is_trending_down = ema50_slope < -0.3 and price < ema20_valid[-1] < ema50_valid[-1]
    is_ranging = abs(ema50_slope) < 0.15 and ema20_dist < 1.5

    if is_volatile and not (is_trending_up or is_trending_down):
        regime = "volatile"
        label  = "⚡ Высокая волатильность"
        trend_mult = 0.8   # тренд менее надёжен
        hvn_mult   = 1.3   # HVN важнее в волатильности
    elif is_trending_up:
        regime = "trending_up"
        label  = "📈 Тренд вверх"
        trend_mult = 1.4   # трендовые сигналы ценнее
        hvn_mult   = 0.8   # HVN как поддержка, не разворот
    elif is_trending_down:
        regime = "trending_down"
        label  = "📉 Тренд вниз"
        trend_mult = 1.4
        hvn_mult   = 0.8
    elif is_ranging:
        regime = "ranging"
        label  = "↔️ Флэт/боковик"
        trend_mult = 0.6   # тренд ненадёжен в боковике
        hvn_mult   = 1.5   # HVN работает лучше в боковике
    else:
        regime = "mixed"
        label  = "🔀 Смешанный"
        trend_mult = 1.0
        hvn_mult   = 1.0

    return {
        "regime": regime,
        "label": label,
        "ema50_slope": round(ema50_slope, 3),
        "vol_ratio": round(vol_ratio, 2),
        "trend_score_mult": trend_mult,
        "hvn_score_mult": hvn_mult,
    }

def find_sr_levels(df, price):
    r = df.tail(200)
    highs, lows = [], []
    for i in range(2, len(r)-2):
        h, l = r.iloc[i]['high'], r.iloc[i]['low']
        if h > r.iloc[i-1]['high'] and h > r.iloc[i+1]['high']: highs.append(float(round(h, 6)))
        if l < r.iloc[i-1]['low'] and l < r.iloc[i+1]['low']: lows.append(float(round(l, 6)))
    return sorted([l for l in lows if l < price], reverse=True)[:3], sorted([h for h in highs if h > price])[:3]

def get_trend(df, label):
    if df is None or len(df) < 50: return "UNKNOWN", f"Нет данных {label}"
    close = df['close']
    e20 = ta.ema(close, length=20).iloc[-1]
    e50 = ta.ema(close, length=50).iloc[-1]
    cur = close.iloc[-1]
    if cur > e20 > e50: return "UPTREND", f"🟢 {label} аптренд"
    if cur < e20 < e50: return "DOWNTREND", f"🔴 {label} даунтренд"
    return "SIDEWAYS", f"⚪ {label} боковик"

# ================== RSI ДИВЕРГЕНЦИЯ (настоящие экстремумы) ==================
def _find_extrema(arr: np.ndarray, order: int = 3):
    """
    Находит локальные максимумы и минимумы в массиве.
    order — сколько соседей с каждой стороны должны быть меньше/больше.
    Возвращает (indices_max, indices_min).
    """
    n = len(arr)
    maxima, minima = [], []
    for i in range(order, n - order):
        window = arr[i-order:i+order+1]
        if arr[i] == window.max() and arr[i] > window.mean():
            maxima.append(i)
        if arr[i] == window.min() and arr[i] < window.mean():
            minima.append(i)
    return np.array(maxima), np.array(minima)

def detect_rsi_divergence(df: pd.DataFrame) -> str:
    """
    Настоящая дивергенция RSI через поиск локальных экстремумов.
    Бычья: цена — новый лоу, RSI — не новый лоу (разворот вверх).
    Медвежья: цена — новый хай, RSI — не новый хай (разворот вниз).
    """
    if len(df) < 40 or 'rsi' not in df.columns:
        return ""

    recent = df.tail(50).copy()
    prices   = recent['low'].values    # для бычьей — лоу свечей
    highs    = recent['high'].values   # для медвежьей — хай свечей
    rsi_vals = recent['rsi'].values

    valid = ~(np.isnan(prices) | np.isnan(rsi_vals))
    if valid.sum() < 30:
        return ""

    prices   = prices[valid]
    highs    = highs[valid]
    rsi_vals = rsi_vals[valid]

    _, minima = _find_extrema(prices, order=3)
    maxima, _ = _find_extrema(highs, order=3)

    # Бычья дивергенция: два последних минимума цены и RSI
    if len(minima) >= 2:
        i1, i2 = minima[-2], minima[-1]
        # Цена делает более низкий лоу, RSI — нет
        if prices[i2] < prices[i1] * 0.998 and rsi_vals[i2] > rsi_vals[i1] + 2:
            strength = round(rsi_vals[i2] - rsi_vals[i1], 1)
            return f"🔄 Бычья дивергенция RSI (+{strength} пунктов)"

    # Медвежья дивергенция: два последних максимума цены и RSI
    if len(maxima) >= 2:
        i1, i2 = maxima[-2], maxima[-1]
        # Цена делает более высокий хай, RSI — нет
        if highs[i2] > highs[i1] * 1.002 and rsi_vals[i2] < rsi_vals[i1] - 2:
            strength = round(rsi_vals[i1] - rsi_vals[i2], 1)
            return f"🔄 Медвежья дивергенция RSI (-{strength} пунктов)"

    return ""

# ================== ПРОБОЙ HVN С ОБЪЁМОМ ==================
def detect_hvn_breakout(df: pd.DataFrame, hv_nodes: list, price: float) -> str:
    """
    Проверяет пробил ли цена HVN с объёмом 2x от среднего.
    Подтверждённый пробой = сильный сигнал продолжения.
    """
    if len(df) < 20 or not hv_nodes:
        return ""

    avg_vol = df['volume'].tail(20).mean()
    last_vol = df['volume'].iloc[-1]
    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]

    if avg_vol == 0:
        return ""

    vol_ratio = last_vol / avg_vol

    for node in hv_nodes[:5]:
        node_price = node['price']
        # Пробой снизу вверх (бычий)
        if (prev_candle['close'] < node_price < last_candle['close'] and
                vol_ratio >= 1.8):
            return f"💥 Пробой HVN {node_price} вверх (объём x{vol_ratio:.1f})"
        # Пробой сверху вниз (медвежий)
        if (prev_candle['close'] > node_price > last_candle['close'] and
                vol_ratio >= 1.8):
            return f"💥 Пробой HVN {node_price} вниз (объём x{vol_ratio:.1f})"

    return ""

# ================== ВНЕШНИЕ ДАННЫЕ (кеш) ==================
_cache: dict = {}

async def fetch_btc_dominance() -> str:
    """BTC доминанс с CoinGecko — бесплатно, кеш 15 мин"""
    cache_key = "btc_dominance"
    now = datetime.now().timestamp()
    if cache_key in _cache and now - _cache[cache_key]['ts'] < 900:
        return _cache[cache_key]['val']
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get("https://api.coingecko.com/api/v3/global") as r:
                if r.status == 200:
                    data = await r.json()
                    dom = data['data']['market_cap_percentage'].get('btc', 0)
                    dom = round(dom, 1)
                    # Интерпретация
                    if dom > 58:
                        txt = f"📊 BTC доминанс: {dom}% (высокий — альты под давлением)"
                    elif dom < 48:
                        txt = f"📊 BTC доминанс: {dom}% (низкий — альт-сезон)"
                    else:
                        txt = f"📊 BTC доминанс: {dom}% (нейтральный)"
                    _cache[cache_key] = {'val': txt, 'ts': now}
                    return txt
    except Exception as e:
        logger.warning(f"BTC dominance fetch: {e}")
    return ""

async def fetch_crypto_news(symbol: str) -> str:
    """
    Новости через RSS CoinDesk + Cointelegraph — бесплатно, без ключей.
    Фильтрует по монете. Кеш 15 мин.
    """
    import xml.etree.ElementTree as ET
    coin = symbol.replace("/USDT","").replace("/","").upper()
    cache_key = f"news_{coin}"
    now = datetime.now().timestamp()
    if cache_key in _cache and now - _cache[cache_key]['ts'] < 900:
        return _cache[cache_key]['val']

    feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
    ]
    headlines = []
    timeout = aiohttp.ClientTimeout(total=8)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for feed_url in feeds:
                if len(headlines) >= 3:
                    break
                try:
                    async with session.get(feed_url, headers={"User-Agent": "Mozilla/5.0"}) as r:
                        if r.status != 200:
                            continue
                        text = await r.text()
                        root = ET.fromstring(text)
                        items = root.findall('.//item')
                        for item in items:
                            title = item.findtext('title', '')
                            # Ищем упоминание монеты в заголовке
                            if coin.lower() in title.lower() or 'bitcoin' in title.lower() and coin == 'BTC':
                                headlines.append(f"📰 {title[:75]}")
                                if len(headlines) >= 3:
                                    break
                except Exception as e:
                    logger.warning(f"RSS {feed_url}: {e}")
    except Exception as e:
        logger.warning(f"fetch_crypto_news: {e}")

    if not headlines:
        _cache[cache_key] = {'val': '', 'ts': now}
        return ''

    txt = "📰 <b>Новости:</b>\n" + "\n".join(headlines)
    _cache[cache_key] = {'val': txt, 'ts': now}
    return txt

async def fetch_liq_levels(symbol: str, price: float) -> str:
    """
    Приближённые уровни ликвидаций через Binance OI история.
    Считаем зоны где скопились позиции = потенциальные цели.
    """
    try:
        ticker = symbol.replace("/", "")
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Берём историю OI за последние 30 периодов
            async with session.get(
                "https://fapi.binance.com/futures/data/openInterestHist",
                params={"symbol": ticker, "period": "1h", "limit": 24}
            ) as r:
                if r.status != 200:
                    return ""
                data = await r.json()
                if not data:
                    return ""

                oi_values = [float(d['sumOpenInterest']) for d in data]
                oi_usdt   = [float(d['sumOpenInterestValue']) for d in data]

                max_oi = max(oi_usdt)
                min_oi = min(oi_usdt)
                curr_oi = oi_usdt[-1]
                oi_change_24h = round((curr_oi - oi_usdt[0]) / max(oi_usdt[0], 1) * 100, 1)

                if oi_change_24h > 10:
                    oi_txt = f"📈 OI +{oi_change_24h}% за 24ч (накопление позиций)"
                elif oi_change_24h < -10:
                    oi_txt = f"📉 OI {oi_change_24h}% за 24ч (ликвидации/закрытия)"
                else:
                    oi_txt = f"➡️ OI {oi_change_24h:+.1f}% за 24ч (стабильно)"
                return oi_txt
    except Exception as e:
        logger.warning(f"OI history: {e}")
    return ""

# ================== ТП/СЛ ==================
def _snap_to_level(target, levels, tolerance_pct=0.8):
    best, best_dist = target, float('inf')
    for lvl in levels:
        dist = abs(lvl - target) / max(target, 0.0001) * 100
        if dist < tolerance_pct and dist < best_dist:
            best, best_dist = lvl, dist
    return round(best, 6)

# Лимиты риска по таймфрейму
TF_MAX_RISK = {"15m": 0.04, "1h": 0.07, "4h": 0.12, "1d": 0.22}

def calculate_sl_tp(signal, price, atr, hv_nodes, supports=None, resistances=None, tf="15m"):
    if signal not in ("🟩 LONG", "🟥 SHORT"): return {}
    is_long = signal == "🟩 LONG"
    below = [n for n in hv_nodes if not n['is_above']]
    above = [n for n in hv_nodes if n['is_above']]
    hvn_above_p = [n['price'] for n in above]
    hvn_below_p = [n['price'] for n in below]
    res = list(resistances or [])
    sup = list(supports or [])
    max_risk = TF_MAX_RISK.get(tf, 0.07)  # лимит риска по таймфрейму

    if is_long:
        sl = (min(below, key=lambda x: x['distance_pct'])['price'] - atr*0.3 if below else price - atr*1.5)
        sl = min(sl, price - atr*1.2)
        near_sup = [s for s in sup if s < price and s > sl - atr]
        if near_sup: sl = min(sl, min(near_sup) - atr*0.2)
        sl = round(max(sl, price * 0.001), 6)
        # Жёсткий лимит по таймфрейму
        sl = round(max(sl, price * (1 - max_risk)), 6)
        risk = price - sl
        tp1 = _snap_to_level(max(price + risk*1.5, price + atr*0.5), hvn_above_p + res)
        tp2 = round(max(_snap_to_level(price + risk*2.5, hvn_above_p + res, 1.0), tp1 + atr*0.5), 6)
        far = [n for n in above if n['price'] > tp2]
        tp3 = round(max(
            max(far, key=lambda x: x['strength'])['price'] if far else price + risk*4.0,
            price + risk*3.5, tp2 + atr), 6)
    else:
        sl = (max(above, key=lambda x: x['distance_pct'])['price'] + atr*0.3 if above else price + atr*1.5)
        sl = max(sl, price + atr*1.2)
        near_res = [r for r in res if r > price and r < sl + atr]
        if near_res: sl = max(sl, max(near_res) + atr*0.2)
        # Жёсткий лимит по таймфрейму
        sl = round(min(sl, price * (1 + max_risk)), 6)
        risk = sl - price
        tp1 = _snap_to_level(min(price - risk*1.5, price - atr*0.5), hvn_below_p + sup)
        tp2 = round(min(_snap_to_level(price - risk*2.5, hvn_below_p + sup, 1.0), tp1 - atr*0.5), 6)
        far = [n for n in below if n['price'] < tp2]
        tp3_raw = min(far, key=lambda x: x['strength'])['price'] if far else price - risk*4.0
        tp3 = round(min(tp3_raw, price - risk*3.5, tp2 - atr), 6)
        # ТП3 не может быть отрицательным или нулём
        tp3 = round(max(tp3, price * 0.01), 6)

    rr = round(abs(tp2 - price) / max(abs(price - sl), 0.0001), 2)
    risk_pct = round(abs(price - sl) / price * 100, 2)

    warns = []
    if rr < 1.5: warns.append("⚠️ R/R ниже 1.5")
    if risk_pct > 5: warns.append(f"⚠️ Риск {risk_pct}% — уменьши размер позиции")

    return {
        "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "risk_pct": risk_pct,
        "rr_ratio": rr,
        "rr_warn": " | ".join(warns),
    }

# ================== СКОРИНГ → СИГНАЛ ==================
def compute_score_and_signal(
        rsi: float, price: float, atr: float,
        ema20: float, ema50: float,
        top_hvn, vp_mean: float,
        delta_str: str, trend_l: str, trend_h: str,
        candle: str, rsi_div: str, hvn_break: str,
        regime: dict, mode_cfg: dict
) -> tuple:
    """
    Скоринг-ориентированный подход:
    1. Считаем баллы ЛОНГ и ШОРТ независимо по всем факторам
    2. Победитель определяет направление сигнала
    3. Итоговый скор = балл победителя, нормированный до 100

    Возвращает (signal, reason, score, detail)
    """
    rsi_long  = mode_cfg["rsi_long"]
    rsi_short = mode_cfg["rsi_short"]
    hvn_mult  = mode_cfg["hvn_mult"]
    is_hard   = mode_cfg.get("label", "") == "🔴 HARD"

    trend_mult = regime.get("trend_score_mult", 1.0)
    hvn_w      = regime.get("hvn_score_mult", 1.0)

    try:
        buy_pct = float(delta_str.split('%')[0].split()[-1])
    except:
        buy_pct = 50.0

    long_score  = 0.0
    short_score = 0.0
    long_reasons  = []
    short_reasons = []

    # --- HVN / Volume Profile (самый важный фактор) ---
    if top_hvn:
        strength_ratio = top_hvn['strength'] / max(vp_mean, 0.0001)
        hvn_pts = min(30, strength_ratio * 10) * hvn_w
        # HVN сверху — давит на цену → SHORT сигнал
        if top_hvn['is_above'] and strength_ratio > hvn_mult:
            short_score += hvn_pts
            short_reasons.append(f"Полка тепла {top_hvn['price']} сверху ({strength_ratio:.1f}x)")
        # HVN снизу — поддержка → LONG сигнал
        elif not top_hvn['is_above'] and strength_ratio > hvn_mult:
            long_score += hvn_pts
            long_reasons.append(f"Полка тепла {top_hvn['price']} снизу ({strength_ratio:.1f}x)")

    # --- EMA тренд (второй по важности) ---
    if trend_l == "UPTREND":
        long_score  += 20 * trend_mult
        long_reasons.append("EMA аптренд")
    elif trend_l == "DOWNTREND":
        short_score += 20 * trend_mult
        short_reasons.append("EMA даунтренд")

    # --- HTF тренд (подтверждение) ---
    if trend_h == "UPTREND":
        long_score  += 15 * trend_mult
        long_reasons.append("HTF аптренд")
    elif trend_h == "DOWNTREND":
        short_score += 15 * trend_mult
        short_reasons.append("HTF даунтренд")

    # --- RSI как ФИЛЬТР, не основной сигнал ---
    # RSI перекупленность/перепроданность только добавляет баллы,
    # не создаёт сигнал сам по себе (кроме экстремумов)
    if rsi < rsi_long:
        pts = 15 if rsi < 30 else 8
        long_score += pts
        long_reasons.append(f"RSI {rsi} (перепроданность)")
    elif rsi > rsi_short:
        pts = 15 if rsi > 70 else 8
        short_score += pts
        short_reasons.append(f"RSI {rsi} (перекупленность)")
    # RSI в тренде — небольшой бонус трендовому направлению
    elif 45 < rsi < 60 and trend_l == "UPTREND":
        long_score += 5
    elif 40 < rsi < 55 and trend_l == "DOWNTREND":
        short_score += 5

    # --- Дивергенция RSI ---
    if rsi_div:
        if "Бычья" in rsi_div:
            long_score  += 15
            long_reasons.append(rsi_div)
        elif "Медвежья" in rsi_div:
            short_score += 15
            short_reasons.append(rsi_div)

    # --- Пробой HVN с объёмом ---
    if hvn_break:
        if "вверх" in hvn_break:
            long_score  += 20
            long_reasons.append(hvn_break)
        elif "вниз" in hvn_break:
            short_score += 20
            short_reasons.append(hvn_break)

    # --- Свечной паттерн ---
    if "Бычье поглощение" in candle or ("пин-бар" in candle and "Бычий" in candle):
        long_score  += 10
        long_reasons.append(candle)
    elif "Медвежье поглощение" in candle or ("пин-бар" in candle and "Медвежий" in candle):
        short_score += 10
        short_reasons.append(candle)

    # --- Дельта объёма ---
    if buy_pct > 62:
        long_score  += 8
        long_reasons.append(f"Дельта бычья ({buy_pct:.0f}%)")
    elif buy_pct < 38:
        short_score += 8
        short_reasons.append(f"Дельта медвежья ({buy_pct:.0f}%)")

    # --- Режим флэт штрафует RSI-сигналы ---
    if regime.get("regime") == "ranging":
        # В боковике HVN уже усилен, но RSI сигналы менее надёжны
        pass

    # --- Определяем победителя ---
    max_possible = 103.0  # сумма всех максимальных баллов
    diff = long_score - short_score
    abs_winner = max(long_score, short_score)
    normalized_score = min(100, int(abs_winner / max_possible * 100))

    # Минимальный разрыв для сигнала: не HARD = 10 pts, HARD = 5 pts
    min_diff = 5 if is_hard else 10

    if long_score > short_score and diff >= min_diff:
        signal = "🟩 LONG"
        reason = " + ".join(long_reasons[:3]) if long_reasons else "Совокупность факторов"
    elif short_score > long_score and diff >= min_diff * -1 and (short_score - long_score) >= min_diff:
        signal = "🟥 SHORT"
        reason = " + ".join(short_reasons[:3]) if short_reasons else "Совокупность факторов"
    elif is_hard:
        # HARD: при равных баллах — по тренду
        if trend_l == "UPTREND":
            signal, reason = "🟩 LONG", f"HARD: {trend_l} (равные баллы)"
        elif trend_l == "DOWNTREND":
            signal, reason = "🟥 SHORT", f"HARD: {trend_l} (равные баллы)"
        else:
            signal = "🟩 LONG" if buy_pct > 50 else "🟥 SHORT"
            reason = f"HARD: дельта {buy_pct:.0f}%"
    else:
        signal = "⚠️ WATCH"
        reason = "Нет доминирующего направления"

    # Скоринг < 30 → НЕТ СИГНАЛА (кроме HARD)
    if normalized_score < 30 and not is_hard:
        signal = "НЕТ СИГНАЛА"
        reason = f"Скоринг {normalized_score}/100 — слишком слабо"

    detail = {
        "long_score": round(long_score, 1),
        "short_score": round(short_score, 1),
        "long_reasons": long_reasons,
        "short_reasons": short_reasons,
    }

    return signal, reason, normalized_score, detail

# ================== АНАЛИЗ ==================
async def analyze_symbol(symbol, tf="15m", mode_cfg=None):
    if mode_cfg is None: mode_cfg = TRADE_MODES["mid"]
    (df, source, fr, oi), (df_htf, htf_label), df_daily, btc_dom, news, weekly_trend = await asyncio.gather(
        fetch_ohlcv(symbol, tf),
        fetch_higher_tf(symbol, tf),
        fetch_daily_vp(symbol),
        fetch_btc_dominance(),
        fetch_crypto_news(symbol),
        fetch_weekly_trend(symbol),
    )
    if df is None or len(df) < 100: return None

    # Используем ПРЕДПОСЛЕДНЮЮ (закрытую) свечу для сигнала
    # Последняя свеча ещё не закрыта — её данные нестабильны
    df_closed = df.iloc[:-1].copy()  # все закрытые свечи
    price = df_closed['close'].iloc[-1]  # цена закрытия последней закрытой свечи
    current_price = df['close'].iloc[-1]  # текущая цена для отображения

    df_closed['rsi'] = ta.rsi(df_closed['close'], length=14)
    df_closed['atr'] = ta.atr(df_closed['high'], df_closed['low'], df_closed['close'], length=14)
    rsi  = round(df_closed['rsi'].iloc[-1], 1)
    atr  = round(df_closed['atr'].iloc[-1], 6)
    atr_pct = round(atr / price * 100, 2)
    ema20 = round(ta.ema(df_closed['close'], length=20).iloc[-1], 6)
    ema50 = round(ta.ema(df_closed['close'], length=50).iloc[-1], 6)
    df = df_closed  # дальше работаем с закрытыми свечами

    if price > ema20 > ema50:   ema_trend = "📈 Восходящий"
    elif price < ema20 < ema50: ema_trend = "📉 Нисходящий"
    else:                        ema_trend = "↔️ Боковик"

    centers_l, vp_l = calculate_volume_profile(df)
    poc = round(float(centers_l[np.argmax(vp_l)]), 6)
    local_nodes = find_hvn(vp_l, centers_l, price)

    daily_nodes = []
    if df_daily is not None and len(df_daily) > 50:
        centers_d, vp_d = calculate_volume_profile(df_daily, num_bins=150)
        daily_nodes = find_hvn(vp_d, centers_d, price, dist_limit=30)

    all_nodes = merge_hvn_levels(local_nodes, daily_nodes)
    hvn_above = [n for n in all_nodes if n['is_above']]
    hvn_below = [n for n in all_nodes if not n['is_above']]

    supports, resistances = find_sr_levels(df, price)
    candle = detect_candle_pattern(df)
    delta  = calculate_delta(df)
    rsi_div   = detect_rsi_divergence(df)
    hvn_break = detect_hvn_breakout(df, local_nodes, price)
    oi_trend  = await fetch_liq_levels(symbol, price)
    trend_l, trend_l_txt = get_trend(df, tf)
    trend_h, trend_h_txt = get_trend(df_htf, htf_label) if df_htf is not None else ("UNKNOWN", "Нет данных HTF")

    vp_mean  = float(np.mean(vp_l))
    strong_above = [n for n in hvn_above if n['distance_pct'] < 12]
    top_hvn = strong_above[0] if strong_above else None

    # Определяем режим рынка
    regime = detect_market_regime(df, atr)

    # Скоринг-ориентированный сигнал
    signal, reason, score, score_detail = compute_score_and_signal(
        rsi=rsi, price=price, atr=atr,
        ema20=ema20, ema50=ema50,
        top_hvn=top_hvn, vp_mean=vp_mean,
        delta_str=delta, trend_l=trend_l, trend_h=trend_h,
        candle=candle, rsi_div=rsi_div, hvn_break=hvn_break,
        regime=regime, mode_cfg=mode_cfg
    )

    htf_conflict = ""
    if signal == "🟩 LONG"  and trend_h == "DOWNTREND": htf_conflict = f"⚠️ LONG против тренда {htf_label}!"
    if signal == "🟥 SHORT" and trend_h == "UPTREND":   htf_conflict = f"⚠️ SHORT против тренда {htf_label}!"

    sl_tp = calculate_sl_tp(signal, price, atr, all_nodes, supports, resistances, tf)

    return {
        "symbol": symbol, "tf": tf,
        "price": round(price, 6),
        "current_price": round(current_price, 6),
        "signal": signal, "reason": reason, "score": score,
        "rsi": rsi, "atr": atr, "atr_pct": atr_pct,
        "ema_trend": ema_trend, "poc": poc,
        "hvn_above": hvn_above, "hvn_below": hvn_below,
        "supports": supports, "resistances": resistances,
        "candle_pattern": candle, "delta": delta,
        "trend_local": trend_l_txt, "trend_higher": trend_h_txt,
        "htf_conflict": htf_conflict, "sl_tp": sl_tp, "source": source,
        "funding_rate": round(fr, 4) if fr is not None else None,
        "open_interest": int(oi) if oi is not None else None,
        "rsi_divergence": rsi_div,
        "hvn_breakout": hvn_break,
        "oi_trend": oi_trend,
        "btc_dominance": btc_dom,
        "news": news,
        "weekly_trend": weekly_trend,
        "time": datetime.now().strftime("%H:%M"),
        "btc_trend_text": "",
        "mode_label": mode_cfg["label"],
        "mode_personality": mode_cfg["personality"],
        "regime": regime,
        "score_detail": score_detail,
    }

# ================== GROQ AI — СЦЕНАРНЫЙ АНАЛИЗ ==================
async def ask_ai(data: dict) -> str:
    if not groq_client: return "AI отключён (нет GROQ_API_KEY)"

    sl_tp   = data.get("sl_tp", {})
    tf      = data['tf']
    score   = data.get('score', 0)
    signal  = data['signal']
    price   = data['price']

    # Ключевые уровни для сценариев
    hvn_a = [n['price'] for n in data['hvn_above'][:3]]
    hvn_b = [n['price'] for n in data['hvn_below'][:3]]
    poc   = data['poc']
    sup   = data.get('supports', [])[:2]
    res   = data.get('resistances', [])[:2]
    conflict = data.get('htf_conflict', '')

    tf_ctx = {
        "15m": "скальпинг, сделка живёт 1-4 часа",
        "1h":  "интрадей, сделка живёт 4-24 часа",
        "4h":  "свинг, сделка живёт 2-7 дней",
        "1d":  "позиция, сделка живёт 1-4 недели"
    }.get(tf, "интрадей")

    # Вычисляем качество сигнала для контекста
    rr = float(sl_tp.get('rr_ratio', 0)) if sl_tp else 0
    risk_pct = float(sl_tp.get('risk_pct', 0)) if sl_tp else 0

    # Лучший уровень для лимитного ордера — POC или ближайший HVN
    best_limit = None
    if hvn_b and abs(hvn_b[0] - price) / price < 0.05:
        best_limit = hvn_b[0]
    elif abs(poc - price) / price < 0.05:
        best_limit = poc
    elif sup:
        best_limit = sup[0]

    system = """Ты — профессиональный трейдер с 10 годами опыта торговли криптовалютами.
Специализация: анализ рисков, точные точки входа, управление позицией.
Твой стиль: конкретный, без воды, с уклоном к защите капитала.

ПРАВИЛА:
- Используй ТОЛЬКО цифры из данных — никаких выдуманных уровней
- Если R/R хуже 1.5 — рекомендуй пропустить или ждать лучшей точки
- Если риск > 5% — обязательно предупреди об уменьшении размера позиции
- Лимитный ордер предпочтительнее входа по рынку когда цена далеко от уровня
- Отвечай по-русски"""

    user = f"""=== АНАЛИЗ {data['symbol']} / {tf} ===
Таймфрейм: {tf_ctx}
Цена свечи: {price} | Сейчас: {data.get('current_price', price)}
Сигнал: {signal} | Скоринг: {score}/100
Режим: {data.get('regime', {}).get('label', 'неизвестно')}
Неделя: {data.get('weekly_trend', 'нет данных')}

=== ТЕХНИЧЕСКИЕ ДАННЫЕ ===
RSI: {data['rsi']} | ATR: {data['atr_pct']}% от цены
Тренд {tf}: {data['trend_local']}
Тренд HTF: {data['trend_higher']}
{'⚠️ ' + conflict if conflict else ''}
Дельта объёма: {data['delta']}
Дивергенция RSI: {data.get('rsi_divergence') or 'нет'}
Пробой уровня: {data.get('hvn_breakout') or 'нет'}
Фандинг: {data.get('funding_rate', 'N/A')}%

=== УРОВНИ ===
POC (главный магнит): {poc}
HVN выше цены: {hvn_a}
HVN ниже цены: {hvn_b}
Сопротивления: {res}
Поддержки: {sup}

=== ПЛАН БОТА ===
{'СЛ: ' + str(sl_tp.get('sl')) + ' | ТП1: ' + str(sl_tp.get('tp1')) + ' | ТП2: ' + str(sl_tp.get('tp2')) + ' | R/R: 1:' + str(rr) + ' | Риск: ' + str(risk_pct) + '%' if sl_tp else 'нет'}
Лучший уровень для лимитки: {best_limit or 'определи сам из уровней выше'}

Дай профессиональный анализ СТРОГО в формате:

🎯 ВХОД: [по рынку на X / лимит на X — конкретная цена и почему именно она]
⛔ ОТМЕНА СДЕЛКИ: [точный уровень цены при котором идея сломана]
💰 СООТНОШЕНИЕ: [R/R и стоит ли оно того при данном риске {risk_pct}%]
⚠️ ГЛАВНЫЙ РИСК: [одна техническая причина, конкретно]
✅ РЕШЕНИЕ: [войти / лимит на [цена] / пропустить — и одна фраза почему]"""

    for model in GROQ_MODELS:
        for attempt in range(2):
            try:
                r = await asyncio.get_event_loop().run_in_executor(None,
                    lambda m=model: groq_client.chat.completions.create(
                        model=m,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user",   "content": user}
                        ],
                        max_tokens=320, temperature=0.4))
                logger.info(f"Groq OK: {model}")
                return r.choices[0].message.content.strip()
            except Exception as ex:
                err = str(ex)
                logger.error(f"Groq {model} attempt {attempt+1}: {err}")
                if "404" in err or "not found" in err.lower() or "decommissioned" in err.lower():
                    break
                if "429" in err or "rate" in err.lower():
                    m2 = re.search(r'retryDelay[^0-9]+(\d+)', err)
                    await asyncio.sleep(min(int(m2.group(1))+2 if m2 else 12, 30))
                elif attempt == 0:
                    await asyncio.sleep(3)
    return "⏳ AI: лимит запросов"


# ================== ЛИМИТНЫЕ ОРДЕРА ==================
def calculate_limit_orders(result: dict) -> dict:
    """
    Лимитные ордера строго по направлению сигнала.
    LONG сигнал → только LONG лимитки (от поддержки или на пробой вверх)
    SHORT сигнал → только SHORT лимитки (от сопротивления или на пробой вниз)
    WATCH/НЕТ → показываем оба варианта
    """
    price       = result['price']
    signal      = result['signal']
    hvn_above   = result.get('hvn_above', [])
    hvn_below   = result.get('hvn_below', [])
    supports    = result.get('supports', [])
    resistances = result.get('resistances', [])
    atr         = result.get('atr', price * 0.01)
    poc         = result.get('poc', price)

    is_long  = "LONG"  in signal
    is_short = "SHORT" in signal
    is_watch = not is_long and not is_short

    orders = []

    levels_below = sorted(
        [n['price'] for n in hvn_below if n['distance_pct'] < 8] +
        [s for s in supports if s > price * 0.92],
        reverse=True
    )
    levels_above = sorted(
        [n['price'] for n in hvn_above if n['distance_pct'] < 8] +
        [r for r in resistances if r < price * 1.08]
    )

    # LONG лимитки — только если сигнал LONG или WATCH
    if (is_long or is_watch) and levels_below:
        buy_level = levels_below[0]
        dist_pct  = round((price - buy_level) / price * 100, 2)
        sl_limit  = round(buy_level - atr * 1.2, 6)
        tp_limit  = round(buy_level + (buy_level - sl_limit) * 2.0, 6)
        orders.append({
            "type":   "📥 ЛИМИТ LONG (от поддержки)",
            "entry":  round(buy_level, 6),
            "sl":     sl_limit,
            "tp1":    tp_limit,
            "dist":   dist_pct,
            "reason": f"HVN/поддержка {buy_level} (-{dist_pct}% от цены)"
        })

    if (is_long or is_watch) and levels_above:
        break_level = levels_above[0]
        entry_break = round(break_level * 1.002, 6)
        dist_pct    = round((break_level - price) / price * 100, 2)
        sl_break    = round(break_level - atr * 0.8, 6)
        tp_break    = round(entry_break + (entry_break - sl_break) * 2.0, 6)
        orders.append({
            "type":   "📈 ЛИМИТ LONG (пробой вверх)",
            "entry":  entry_break,
            "sl":     sl_break,
            "tp1":    tp_break,
            "dist":   dist_pct,
            "reason": f"Пробой {break_level} (+{dist_pct}% от цены)"
        })

    # SHORT лимитки — только если сигнал SHORT или WATCH
    if (is_short or is_watch) and levels_above:
        sell_level = levels_above[0]
        dist_pct   = round((sell_level - price) / price * 100, 2)
        sl_sell    = round(sell_level + atr * 1.2, 6)
        tp_sell    = round(sell_level - (sl_sell - sell_level) * 2.0, 6)
        orders.append({
            "type":   "📤 ЛИМИТ SHORT (от сопротивления)",
            "entry":  round(sell_level, 6),
            "sl":     sl_sell,
            "tp1":    tp_sell,
            "dist":   dist_pct,
            "reason": f"HVN/сопротивление {sell_level} (+{dist_pct}% от цены)"
        })

    if (is_short or is_watch) and levels_below:
        break_dn    = levels_below[0]
        entry_dn    = round(break_dn * 0.998, 6)
        dist_pct    = round((price - break_dn) / price * 100, 2)
        sl_dn       = round(break_dn + atr * 0.8, 6)
        tp_dn       = round(entry_dn - (sl_dn - entry_dn) * 2.0, 6)
        orders.append({
            "type":   "📉 ЛИМИТ SHORT (пробой вниз)",
            "entry":  entry_dn,
            "sl":     sl_dn,
            "tp1":    tp_dn,
            "dist":   dist_pct,
            "reason": f"Пробой поддержки {break_dn} (-{dist_pct}% от цены)"
        })

    return {"orders": orders, "poc": poc}

def format_limit_orders(limit_data: dict) -> str:
    """Форматирует блок лимитных ордеров для сообщения"""
    if not limit_data:
        return ""
    orders = limit_data.get('orders', [])
    if not orders:
        return ""
    e = html_escape
    lines = ["\n📋 <b>ЛИМИТНЫЕ ОРДЕРА:</b>"]
    for o in orders[:1]:  # показываем только лучший вариант
        lines.append(
            f"\n{e(o['type'])}\n"
            f"├ Вход: <b>{o['entry']}</b> ({e(o['reason'])})\n"
            f"├ СЛ: {o['sl']}\n"
            f"└ ТП: {o['tp1']}"
        )
    return "\n".join(lines)

# ================== ФОРМАТИРОВАНИЕ ==================
def format_message(result, ai_text, is_scanner=False, limit_data=None):
    e = html_escape
    sl_tp = result.get("sl_tp", {})
    p = result['price']
    def pct(t): return round(abs(t - p) / p * 100, 2)

    trade_block = ""
    if sl_tp and result['signal'] in ("🟩 LONG", "🟥 SHORT"):
        rr_warn = f"\n{e(sl_tp['rr_warn'])}" if sl_tp.get('rr_warn') else ""
        trade_block = (
            f"\n📐 <b>ПЛАН СДЕЛКИ</b>\n"
            f"├ 🛑 СЛ: <b>{sl_tp['sl']}</b> (-{pct(sl_tp['sl'])}%)\n"
            f"├ 🎯 ТП1: <b>{sl_tp['tp1']}</b> (+{pct(sl_tp['tp1'])}%) → <i>БУ</i>\n"
            f"├ 🎯 ТП2: <b>{sl_tp['tp2']}</b> (+{pct(sl_tp['tp2'])}%) → <i>СЛ на ТП1</i>\n"
            f"└ 🏆 ТП3: <b>{sl_tp['tp3']}</b> (+{pct(sl_tp['tp3'])}%) → <i>финал</i>\n"
            f"R/R: 1:{sl_tp['rr_ratio']} | Риск: {sl_tp['risk_pct']}%{rr_warn}\n"
        )

    score = result.get('score', 0)
    score_bar = "🟩"*(score//20) + "⬜"*(5-score//20)
    fr_str = ""
    if result.get('funding_rate') is not None:
        fr = result['funding_rate']
        fe = "🔴" if fr > 0.05 else ("🟢" if fr < -0.05 else "⚪")
        fr_str = f"\nФандинг: {fe} {fr}%"
    oi_str = f" | OI: {result['open_interest']:,}" if result.get('open_interest') else ""
    conflict  = f"\n{e(result['htf_conflict'])}"   if result.get('htf_conflict') else ""
    rsi_div   = f"\n{e(result['rsi_divergence'])}" if result.get('rsi_divergence') else ""
    hvn_break = f"\n{e(result['hvn_breakout'])}"   if result.get('hvn_breakout') else ""
    oi_trend  = f"\n{e(result['oi_trend'])}"        if result.get('oi_trend') else ""
    btc_dom   = f"\n{e(result['btc_dominance'])}"   if result.get('btc_dominance') else ""
    # news убраны из отображения — только в AI промпте

    def fmt_nodes(nodes):
        out = [f"{'🌍' if n.get('type')=='daily' else '📍'}{n['price']}" for n in nodes[:3]]
        return " ".join(out) if out else "—"

    header = "🔔 <b>АВТОСИГНАЛ</b>\n" if is_scanner else ""
    return (
        f"{header}📊 <b>{e(result['symbol'])}</b> {result['tf']} • {result['time']} • {e(result['mode_label'])}\n"
        f"<i>{e(result['source'])}</i>\n\n"
        f"Цена свечи: <b>{p}</b>"
        + (f" | Сейчас: <b>{result.get('current_price', p)}</b>" if result.get('current_price', p) != p else "")
        + "\n"
        + f"Сигнал: <b>{e(result['signal'])}</b>\n"
        f"Причина: {e(result['reason'])}\n"
        f"Скоринг: {score_bar} <b>{score}/100</b>\n\n"
        f"RSI: {result['rsi']} | ATR: {result['atr']} ({result['atr_pct']}%)\n"
        f"EMA: {e(result['ema_trend'])} | Свеча: {e(result['candle_pattern'])}\n"
        f"Дельта: {e(result['delta'])}"
        f"{fr_str}{oi_str}\n\n"
        f"🌡️ Режим: {e(result.get('regime', {}).get('label', ''))}\n"
        f"📈 Тренд {result['tf']}: {e(result['trend_local'])}\n"
        f"📊 Тренд HTF: {e(result['trend_higher'])}\n"
        f"{e(result.get('weekly_trend',''))}"
        f"{conflict}{rsi_div}{hvn_break}{oi_trend}{btc_dom}\n\n"
        f"POC: {result['poc']}\n"
        f"HVN↑: {fmt_nodes(result['hvn_above'])}\n"
        f"HVN↓: {fmt_nodes(result['hvn_below'])}\n"
        f"Сопр: {result['resistances'][:2]} | Подд: {result['supports'][:2]}\n"
        f"{trade_block}\n"
        f"{format_limit_orders(limit_data)}\n"
        f"🧠 <b>AI:</b>\n{e(ai_text[:1200])}"
    )


# ================== MEXC API ==================
async def mexc_request(method: str, path: str, params: dict = None) -> dict | None:
    """Подписанный запрос к MEXC Futures API"""
    if not MEXC_API_KEY or not MEXC_SECRET_KEY:
        return None
    params = params or {}
    ts = str(int(time.time() * 1000))
    params['timestamp'] = ts
    query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    sig = hmac.new(MEXC_SECRET_KEY.encode(), query.encode(), hashlib.sha256).hexdigest()
    params['signature'] = sig
    url = f"https://contract.mexc.com{path}"
    headers = {"ApiKey": MEXC_API_KEY, "Content-Type": "application/json"}
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if method == "GET":
                async with session.get(url, params=params, headers=headers) as r:
                    return await r.json()
            else:
                async with session.post(url, json=params, headers=headers) as r:
                    return await r.json()
    except Exception as e:
        logger.error(f"MEXC request {path}: {e}")
        return None

async def fetch_mexc_positions() -> list:
    """Получает открытые позиции с MEXC Futures"""
    data = await mexc_request("GET", "/api/v1/private/position/open_positions")
    if not data or data.get('code') != 200:
        return []
    return data.get('data', [])

async def fetch_mexc_position(symbol: str) -> dict | None:
    """Позиция по конкретной монете (MEXC формат: BTC_USDT)"""
    mexc_sym = symbol.replace("/", "_")
    positions = await fetch_mexc_positions()
    for p in positions:
        if p.get('symbol') == mexc_sym:
            return p
    return None

async def sync_mexc_trades(app=None):
    """Синхронизирует с MEXC и отправляет уведомление если позиция закрылась"""
    if not MEXC_API_KEY:
        return
    trades = load_trades()
    if not trades:
        return
    mexc_positions = await fetch_mexc_positions()
    mexc_symbols = {p.get('symbol') for p in mexc_positions}

    for key, t in list(trades.items()):
        mexc_sym = t['symbol'].replace("/", "_")
        if mexc_sym not in mexc_symbols:
            logger.info(f"MEXC: позиция {mexc_sym} закрыта на бирже")
            # Уведомляем пользователя
            if app and t.get('chat_id'):
                try:
                    await app.bot.send_message(
                        chat_id=t['chat_id'],
                        text=(
                            f"🛑 <b>Позиция закрыта на MEXC</b>\n"
                            f"{t['symbol']} {t['tf']} {t['signal']}\n"
                            f"Вход был: {t['entry']}\n"
                            f"Бот убрал из мониторинга."
                        ),
                        parse_mode='HTML'
                    )
                except Exception as e:
                    logger.error(f"MEXC notify {key}: {e}")
            close_trade(key)

async def get_mexc_sl_tp(symbol: str) -> dict | None:
    """
    Получает реальные СЛ и ТП установленные на MEXC.
    Возвращает dict с sl, tp если найдены.
    """
    mexc_sym = symbol.replace("/", "_")
    data = await mexc_request("GET", "/api/v1/private/stoporder/list/orders",
                               {"symbol": mexc_sym, "states": "1"})  # 1 = активные
    if not data or data.get('code') != 200:
        return None
    orders = data.get('data', {}).get('resultList', [])
    result = {}
    for o in orders:
        price = float(o.get('triggerPrice', 0))
        otype = o.get('orderType')  # 3=стоп, 1=тп
        if otype == 3 and price > 0:
            result['sl'] = price
        elif otype == 1 and price > 0:
            result['tp'] = price
    return result if result else None

# ================== МОНИТОРИНГ СДЕЛОК ==================
async def check_trades(app):
    """Одна итерация проверки сделок — вызывается JobQueue каждые 5 минут"""
    # MEXC синхронизация отключена — следим только за ценой
    trades = load_trades()
    if not trades: return
    for key, t in list(trades.items()):
        try:
            # Используем TF сделки для мониторинга — не хардкод 15m
            monitor_tf = t.get('tf', '15m')
            df_monitor, _, _, _ = await fetch_ohlcv(t['symbol'], monitor_tf)
            if df_monitor is None:
                continue

            # Берём закрытую свечу для надёжности
            last        = df_monitor.iloc[-2] if len(df_monitor) > 1 else df_monitor.iloc[-1]
            candle_high = float(last['high'])
            candle_low  = float(last['low'])
            price       = float(last['close'])
            is_long     = "LONG" in t['signal']
            msgs        = []
            sl_just_moved = False  # флаг: СЛ только что переместился → пропустить проверку СЛ

            check_tp = candle_high if is_long else candle_low
            check_sl = candle_low  if is_long else candle_high

            if not t['tp1_hit']:
                if (is_long and check_tp >= t['tp1']) or (not is_long and check_tp <= t['tp1']):
                    t['tp1_hit']      = True
                    t['sl']           = t['entry']
                    t['sl_moved_be']  = True
                    sl_just_moved     = True  # не проверяем СЛ в этом тике
                    msgs.append(
                        f"🎯 <b>ТП1 достигнут!</b>\n"
                        f"{t['symbol']} {t['tf']} | Цена: {round(price,6)}\n"
                        f"СЛ перенесён в безубыток → <b>{t['entry']}</b>\n"
                        f"Следующая цель ТП2: {t['tp2']}"
                    )
            elif not t['tp2_hit']:
                if (is_long and check_tp >= t['tp2']) or (not is_long and check_tp <= t['tp2']):
                    t['tp2_hit']      = True
                    t['sl']           = t['tp1']
                    t['sl_moved_tp1'] = True
                    sl_just_moved     = True
                    record_trade_result(t['symbol'],t['tf'],t['signal'],t['entry'],
                                       t['sl'],t['tp1'],t['tp2'],t['tp3'],t['tp2'],"tp2",0)
                    msgs.append(
                        f"🎯🎯 <b>ТП2 достигнут!</b>\n"
                        f"{t['symbol']} {t['tf']} | Цена: {round(price,6)}\n"
                        f"СЛ перенесён на ТП1 → <b>{t['tp1']}</b>\n"
                        f"Финальная цель ТП3: {t['tp3']}"
                    )
            else:
                if (is_long and check_tp >= t['tp3']) or (not is_long and check_tp <= t['tp3']):
                    record_trade_result(t['symbol'],t['tf'],t['signal'],t['entry'],
                                       t['sl'],t['tp1'],t['tp2'],t['tp3'],t['tp3'],"tp3",0)
                    msgs.append(
                        f"🏆 <b>ТП3 достигнут! Полная цель!</b>\n"
                        f"{t['symbol']} {t['tf']} | Цена: {round(price,6)}\n"
                        f"Сделка успешно закрыта 🎉"
                    )
                    close_trade(key)
                    trades.pop(key, None)

            # Проверка СЛ — пропускаем если СЛ только что переместился
            if key in trades and not sl_just_moved:
                if (is_long and check_sl <= t['sl']) or (not is_long and check_sl >= t['sl']):
                    tag     = "безубыток" if t.get('sl_moved_be') else "стоп-лосс"
                    exit_p  = round(t['sl'], 6)
                    pnl     = round((exit_p - t['entry']) / t['entry'] * 100 * (1 if is_long else -1), 2)
                    pnl_str = f"+{pnl}%" if pnl > 0 else f"{pnl}%"
                    record_trade_result(t['symbol'],t['tf'],t['signal'],t['entry'],
                                       t['sl'],t['tp1'],t['tp2'],t['tp3'],exit_p,tag,0)
                    msgs.append(
                        f"🛑 <b>Закрыто по {tag}</b>\n"
                        f"{t['symbol']} {t['tf']} | Цена: {exit_p}\n"
                        f"P&L: <b>{pnl_str}</b>"
                    )
                    close_trade(key)
                    trades.pop(key, None)

            # Отправляем все уведомления
            chat_id = t.get('chat_id')
            if msgs and chat_id:
                for m in msgs:
                    try:
                        await app.bot.send_message(
                            chat_id=chat_id, text=m, parse_mode='HTML'
                        )
                    except Exception as send_err:
                        logger.error(f"notify {key}: {send_err}")
            elif msgs and not chat_id:
                logger.warning(f"check_trades {key}: no chat_id, cant notify. msgs={msgs}")

            if key in trades:
                save_trades(trades)

        except Exception as ex:
            logger.error(f"check_trades {key}: {ex}", exc_info=True)

# ================== АВТОСКАНЕР ==================
scanner_threshold = 75
_scanner_running  = False  # защита от параллельного запуска

def _load_scanner_active() -> dict:
    """Загружает активные чаты сканера из файла"""
    state = load_scanner_state()
    raw = state.get("active_chats", {})
    return {int(k): v for k, v in raw.items()}

def _save_scanner_active(active: dict):
    """Сохраняет активные чаты в файл"""
    state = load_scanner_state()
    state["active_chats"] = {str(k): v for k, v in active.items()}
    save_scanner_state(state)

# Загружаем при старте из файла
scanner_active = _load_scanner_active()

async def run_scanner(app):
    """Одна итерация сканера — вызывается JobQueue каждую минуту"""
    global _scanner_running
    if _scanner_running:
        logger.info("Scanner: already running, skip")
        return
    _scanner_running = True
    try:
        active_chats = [cid for cid, active in scanner_active.items() if active]
        if not active_chats:
            return

        # Запускаем раз в 15 минут
        state = load_scanner_state()
        last_run = state.get("last_run", 0)
        now = datetime.now().timestamp()
        if (now - last_run) < 900:
            return

        # Обновляем время последнего запуска СРАЗУ
        state["last_run"] = now
        save_scanner_state(state)

        logger.info("Scanner: starting scan...")
        symbols = await fetch_binance_futures_symbols()
        if not symbols:
            logger.warning("Scanner: no symbols fetched")
            return

        mode_cfg = TRADE_MODES["mid"]
        sent_this_run = []

        for i in range(0, len(symbols), 5):
            batch = symbols[i:i+5]
            results = await asyncio.gather(
                *[analyze_symbol(s, "15m", mode_cfg) for s in batch],
                return_exceptions=True
            )

            for result in results:
                if isinstance(result, Exception) or result is None:
                    continue
                if result['signal'] not in ("🟩 LONG", "🟥 SHORT"):
                    continue
                if result['score'] < scanner_threshold:
                    continue
                # HTF конфликт не блокирует — только снижает порог
                # Если конфликт есть — требуем скоринг выше на 15 очков
                if result.get('htf_conflict') and result['score'] < scanner_threshold + 15:
                    continue

                symbol = result['symbol']
                # Антиспам: одна монета не чаще раза в 2 часа
                last_sent = state.get(f"sent_{symbol}", 0)
                if (now - last_sent) < 7200:
                    continue

                state[f"sent_{symbol}"] = now
                save_scanner_state(state)
                sent_this_run.append(symbol)

                result['btc_trend_text'] = ""
                ai_text = await ask_ai(result) if groq_client else "AI отключён"
                limit_data = calculate_limit_orders(result)
                msg = format_message(result, ai_text, is_scanner=True, limit_data=limit_data)

                TG_LIMIT = 4000
                ai_split = msg.find("🧠 <b>AI:</b>")
                for chat_id in active_chats:
                    try:
                        if ai_split > 0 and len(msg) > TG_LIMIT:
                            await app.bot.send_message(chat_id=chat_id, text=msg[:ai_split].rstrip()[:TG_LIMIT], parse_mode='HTML')
                            await app.bot.send_message(chat_id=chat_id, text=msg[ai_split:][:TG_LIMIT], parse_mode='HTML')
                        else:
                            await app.bot.send_message(chat_id=chat_id, text=msg[:TG_LIMIT], parse_mode='HTML')
                        if result.get('sl_tp') and result['signal'] in ("🟩 LONG", "🟥 SHORT"):
                            trade_key = open_trade(symbol, "15m", result, chat_id)
                            await app.bot.send_message(
                                chat_id=chat_id,
                                text=f"📌 Сделка добавлена в мониторинг\n/close {trade_key}"
                            )
                    except Exception as ex:
                        logger.error(f"Scanner send to {chat_id}: {ex}")

            await asyncio.sleep(2)

        logger.info(f"Scanner done. Signals sent: {sent_this_run}")

    except Exception as e:
        logger.error(f"run_scanner error: {e}", exc_info=True)
    finally:
        _scanner_running = False

# ================== КОМАНДЫ ==================
MODES = {"low","mid","hard"}
TFS   = set(TF_MAP.keys())

def parse_args(text):
    parts = text.lower().strip().lstrip("/").split()
    coin = parts[0].upper().replace("USDT","").replace("/","")
    symbol = f"{coin}/USDT"
    tf   = next((p for p in parts[1:] if p in TFS), DEFAULT_TF)
    mode = next((p for p in parts[1:] if p in MODES), "mid")
    return symbol, tf, mode

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ <b>Signal Volume Bot v3</b>\n\n"
        "<b>Анализ:</b>\n"
        "/btc — 15m mid\n"
        "/eth 4h — ETH на 4h\n"
        "/siren hard — агрессив\n"
        "/btc 1h low — консерватив\n\n"
        "<b>Таймфреймы:</b> 15m 1h 4h 1d\n"
        "<b>Режимы:</b> 🟢low 🟡mid 🔴hard\n\n"
        "<b>Сканер (авто-сигналы):</b>\n"
        "/scan on — включить\n"
        "/scan off — выключить\n"
        "/scan status — статус\n\n"
        "<b>Сделки:</b>\n"
        "/trades — открытые\n"
        "/close BTCUSDT15m — закрыть\n\n"
        "🌍 = дневная полка | 📍 = локальная",
        parse_mode='HTML'
    )

async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args = context.args
    cmd = args[0].lower() if args else "status"

    if cmd == "on":
        scanner_active[chat_id] = True
        _save_scanner_active(scanner_active)
        # Сбрасываем last_run чтобы первый скан запустился быстро
        state = load_scanner_state()
        state["last_run"] = 0
        save_scanner_state(state)
        await update.message.reply_text(
            "🔍 <b>Сканер включён</b>\n"
            f"Порог скоринга: {scanner_threshold}/100 (сигналы от 55+)\n"
            "Первый скан запустится через ~1 минуту\n"
            "Далее каждые 15 минут\n"
            "Сигналы: скоринг ≥75, без конфликта HTF",
            parse_mode='HTML'
        )
    elif cmd == "off":
        scanner_active[chat_id] = False
        _save_scanner_active(scanner_active)
        await update.message.reply_text("⏹ Сканер выключен")

    elif cmd == "top":
        # Немедленный скан топ-20 монет и показ лучших
        msg = await update.message.reply_text("🔍 Сканирую топ-50 монет, подожди ~60 сек...")
        symbols = await fetch_binance_futures_symbols(limit=50)
        mode_cfg = TRADE_MODES["mid"]
        results = []
        for i in range(0, len(symbols), 5):
            batch = symbols[i:i+5]
            batch_results = await asyncio.gather(
                *[analyze_symbol(s, "15m", mode_cfg) for s in batch],
                return_exceptions=True
            )
            for r in batch_results:
                if isinstance(r, Exception) or r is None: continue
                if r['signal'] in ("🟩 LONG", "🟥 SHORT"):
                    results.append(r)
            await asyncio.sleep(1)

        if not results:
            await msg.edit_text("😶 Нет сигналов в топ-20 прямо сейчас")
            return

        results.sort(key=lambda x: -x['score'])
        lines = [f"📊 <b>Топ сигналы ({len(results)} найдено):</b>\n"]
        for r in results[:30]:
            bar = "🟩"*(r['score']//20) + "⬜"*(5-r['score']//20)
            lines.append(
                f"{r['signal']} <b>{r['symbol']}</b> {r['tf']}\n"
                f"  {bar} {r['score']}/100 | RSI:{r['rsi']} | {r['reason']}"
            )
        await msg.edit_text("\n".join(lines), parse_mode='HTML')

    elif cmd == "debug":
        # Показываем почему сигналы не проходят
        msg = await update.message.reply_text("🔧 Анализирую топ-10 для диагностики...")
        symbols = await fetch_binance_futures_symbols()
        symbols = symbols[:10]
        mode_cfg = TRADE_MODES["mid"]
        lines = ["🔧 <b>Диагностика сканера:</b>\n"]
        for i in range(0, len(symbols), 5):
            batch = symbols[i:i+5]
            batch_results = await asyncio.gather(
                *[analyze_symbol(s, "15m", mode_cfg) for s in batch],
                return_exceptions=True
            )
            for r in batch_results:
                if isinstance(r, Exception) or r is None:
                    lines.append(f"❌ ошибка данных")
                    continue
                has_conflict = "⚠️" in r.get('htf_conflict','')
                reason_skip = []
                if r['signal'] not in ("🟩 LONG", "🟥 SHORT"):
                    reason_skip.append(f"сигнал={r['signal']}")
                if r['score'] < scanner_threshold:
                    reason_skip.append(f"скор={r['score']}<{scanner_threshold}")
                if has_conflict:
                    reason_skip.append("HTF конфликт")
                status = "✅ прошёл бы" if not reason_skip else f"⛔ {', '.join(reason_skip)}"
                lines.append(f"{r['symbol']}: {r['signal']} скор={r['score']} — {status}")
        await msg.edit_text("\n".join(lines), parse_mode='HTML')

    else:
        active = scanner_active.get(chat_id, False)
        state = load_scanner_state()
        last = state.get("last_run", 0)
        last_str = datetime.fromtimestamp(last).strftime("%H:%M:%S") if last else "никогда"
        # Считаем сколько монет было заблокировано антиспамом
        now = datetime.now().timestamp()
        blocked = sum(1 for k,v in state.items()
                      if k.startswith("sent_") and now - v < 7200)
        await update.message.reply_text(
            f"📡 Сканер: {'🟢 активен' if active else '🔴 выключен'}\n"
            f"Последний скан: {last_str}\n"
            f"Порог скоринга: {scanner_threshold}/100 (сигналы от 55+)\n"
            f"Антиспам заблокировал: {blocked} монет\n\n"
            f"Команды:\n"
            f"/scan on — включить\n"
            f"/scan off — выключить\n"
            f"/scan top — топ сигналы прямо сейчас\n"
            f"/scan debug — почему молчит сканер",
        )

async def cmd_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    trades = load_trades()
    if not trades:
        await update.message.reply_text("📭 Нет открытых сделок")
        return
    for k, t in trades.items():
        is_long = "LONG" in t['signal']
        direction = "🟩" if is_long else "🟥"
        text = (
            f"📊 <b>{t['symbol']}</b> {t['tf']} {direction} {t['signal']}\n"
            f"Вход: <b>{t['entry']}</b> | СЛ: {t['sl']}\n"
            f"ТП1: {t['tp1']} {'✅' if t['tp1_hit'] else '⏳'} | "
            f"ТП2: {t['tp2']} {'✅' if t['tp2_hit'] else '⏳'} | "
            f"ТП3: {t['tp3']}\n"
            f"Открыта: {t['opened_at'][:16]}"
        )
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("🛑 Закрыть сделку", callback_data=f"close_{k}")
        ]])
        await update.message.reply_text(text, parse_mode='HTML', reply_markup=keyboard)


async def cmd_mexc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает реальные позиции с MEXC и сравнивает с ботом"""
    if not MEXC_API_KEY:
        await update.message.reply_text(
            "❌ MEXC API не настроен\n\n"
            "Добавь в Railway Variables:\n"
            "MEXC_API_KEY = твой_ключ\n"
            "MEXC_SECRET_KEY = твой_секрет"
        )
        return

    msg = await update.message.reply_text("🔄 Загружаю позиции с MEXC...")
    positions = await fetch_mexc_positions()

    if not positions:
        await msg.edit_text("📭 Нет открытых позиций на MEXC")
        return

    bot_trades = load_trades()
    lines = [f"📊 <b>Позиции MEXC ({len(positions)}):</b>\n"]

    for p in positions:
        sym = p.get('symbol', '').replace('_', '/')
        side = "🟩 LONG" if p.get('positionType') == 1 else "🟥 SHORT"
        vol = p.get('vol', 0)
        entry = p.get('openAvgPrice', 0)
        upnl = round(float(p.get('unrealisedPnl', 0)), 2)
        upnl_emoji = "✅" if upnl >= 0 else "❌"
        liq = p.get('liquidatePrice', 0)

        # Проверяем есть ли в боте
        bot_key = sym.replace('/','') + '15m'
        in_bot = "📌 в боте" if bot_key in bot_trades else "⚠️ нет в боте"

        lines.append(
            f"{side} <b>{sym}</b> {in_bot}\n"
            f"Вход: {entry} | Объём: {vol}\n"
            f"P&L: {upnl_emoji} {upnl} USDT\n"
            f"Ликвидация: {liq}\n"
        )

    await msg.edit_text("\n".join(lines), parse_mode='HTML')

async def cmd_close(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "Использование:\n"
            "/close КЛЮЧ — закрыть одну\n"
            "/close all — закрыть все\n"
            "Ключи смотри в /trades"
        )
        return

    key = context.args[0]
    trades = load_trades()

    # Закрыть все
    if key.lower() == "all":
        if not trades:
            await update.message.reply_text("📭 Нет открытых сделок")
            return
        count = len(trades)
        save_trades({})  # очищаем всё
        await update.message.reply_text(f"✅ Закрыто всех сделок: {count}")
        return

    # Закрыть одну
    if key in trades:
        close_trade(key)
        await update.message.reply_text(f"✅ Сделка {key} закрыта вручную")
    else:
        keys = list(trades.keys())
        if keys:
            await update.message.reply_text(
                "❌ Сделка не найдена\nОткрытые: " + ", ".join(keys)
            )
        else:
            await update.message.reply_text("📭 Нет открытых сделок")

async def callback_close(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик кнопки Закрыть в /trades"""
    query = update.callback_query
    await query.answer()
    key = query.data.replace("close_", "", 1)
    trades = load_trades()
    if key in trades:
        t = trades[key]
        close_trade(key)
        await query.edit_message_text(
            f"✅ <b>Сделка закрыта вручную</b>\n"
            f"{t['symbol']} {t['tf']} | Вход: {t['entry']}",
            parse_mode='HTML'
        )
    else:
        await query.edit_message_text("❌ Сделка уже закрыта или не найдена")

async def handle_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if not text.startswith('/'): return
    symbol, tf, mode_key = parse_args(text)
    mode_cfg = TRADE_MODES[mode_key]
    chat_id = update.effective_chat.id

    msg = await update.message.reply_text(
        f"🔄 Анализирую <b>{symbol}</b> {tf} [{mode_cfg['label']}]...",
        parse_mode='HTML'
    )
    try:
        (df_btc, _, _, _), result = await asyncio.gather(
            fetch_ohlcv("BTC/USDT", "1h"),
            analyze_symbol(symbol, tf, mode_cfg),
        )

        if result is None:
            await msg.edit_text(f"❌ Нет данных для <b>{symbol}</b>", parse_mode='HTML')
            return

        _, btc_txt = get_trend(df_btc, "BTC")
        result["btc_trend_text"] = btc_txt

        ai_text = await ask_ai(result) if groq_client else "AI отключён"

        # Добавляем лимитные ордера если сигнал слабый или WATCH
        limit_data = None
        if result['signal'] in ("⚠️ WATCH", "НЕТ СИГНАЛА") or result.get('score', 0) < 55:
            limit_data = calculate_limit_orders(result)

        main_text = format_message(result, ai_text, limit_data=limit_data)

        # Telegram лимит 4096 — всегда разбиваем на tech + AI
        TG_LIMIT = 4000
        ai_split = main_text.find("🧠 <b>AI:</b>")
        if ai_split > 0 and len(main_text) > TG_LIMIT:
            part1 = main_text[:ai_split].rstrip()[:TG_LIMIT]
            part2 = main_text[ai_split:][:TG_LIMIT]
            await msg.edit_text(part1, parse_mode='HTML')
            await update.message.reply_text(part2, parse_mode='HTML')
        else:
            await msg.edit_text(main_text[:TG_LIMIT], parse_mode='HTML')

        # Открываем мониторинг только при реальном сигнале
        # НЕ открываем если: ИИ сказал пропустить, скоринг < 35, риск > 15%
        sl_tp = result.get('sl_tp', {})
        risk_pct = sl_tp.get('risk_pct', 99)
        score = result.get('score', 0)
        # Точная проверка - ищем строку ВЫВОД: пропустить
        import re as _re
        vyvod_match = _re.search(r'ВЫВОД[: ]+(\S+)', ai_text, _re.IGNORECASE)
        ai_verdict = vyvod_match.group(1).lower().strip('.,!') if vyvod_match else ""
        ai_skip = ai_verdict == "пропустить"

        if result['signal'] in ("🟩 LONG", "🟥 SHORT") and sl_tp:
            skip_reasons = []
            if score < 35: skip_reasons.append(f"скоринг {score}/100 слишком низкий")
            if risk_pct > 15: skip_reasons.append(f"риск {risk_pct}% слишком высокий для мониторинга")
            if ai_skip: skip_reasons.append("AI рекомендует пропустить")

            if skip_reasons:
                await update.message.reply_text(
                    f"⚠️ <b>Сделка НЕ добавлена в мониторинг:</b>\n"
                    + "\n".join(f"• {r}" for r in skip_reasons),
                    parse_mode='HTML'
                )
            else:
                trade_key = open_trade(symbol, tf, result, chat_id)
                await update.message.reply_text(
                    f"📌 <b>Сделка добавлена в мониторинг</b>\n"
                    f"Бот уведомит при ТП/СЛ\n"
                    f"/trades — все сделки\n"
                    f"/close {trade_key} — закрыть",
                    parse_mode='HTML'
                )

    except Exception as ex:
        logger.error(f"handle_command: {ex}", exc_info=True)
        await msg.edit_text(f"❌ Ошибка: {html_escape(str(ex))}", parse_mode='HTML')

# ================== MAIN ==================
async def _check_trades_job(context):
    try:
        await check_trades(context.application)
    except Exception as e:
        logger.error(f"check_trades_job: {e}", exc_info=True)

async def _scanner_job(context):
    await run_scanner(context.application)

async def post_init(app):
    # JobQueue надёжнее create_task на Railway — перезапускает при сбое
    app.job_queue.run_repeating(_check_trades_job, interval=300, first=15)
    app.job_queue.run_repeating(_scanner_job,      interval=60,  first=30)
    logger.info("JobQueue tasks started: check_trades every 5min, scanner every 1min")

# ================== СТАТИСТИКА ==================
STATS_FILE = Path("bot_stats.json")

def load_stats() -> dict:
    try:
        return json.loads(STATS_FILE.read_text()) if STATS_FILE.exists() else {}
    except: return {}

def save_stats(stats: dict):
    STATS_FILE.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

def record_trade_result(symbol: str, tf: str, signal: str, entry: float,
                        sl: float, tp1: float, tp2: float, tp3: float,
                        exit_price: float, exit_reason: str, score: int,
                        mode: str = "mid"):
    """Записывает результат сделки. mode = mid/low/hard/scanner"""
    stats = load_stats()
    is_long = "LONG" in signal
    pnl_pct = round(((exit_price - entry) / entry * 100) if is_long
                    else ((entry - exit_price) / entry * 100), 2)
    trade = {
        "symbol": symbol, "tf": tf, "signal": signal,
        "entry": entry, "exit": exit_price,
        "pnl_pct": pnl_pct, "exit_reason": exit_reason,
        "score": score, "won": pnl_pct > 0,
        "mode": mode,
        "time": datetime.now().isoformat()
    }
    if "trades" not in stats:
        stats["trades"] = []
    stats["trades"].append(trade)
    save_stats(stats)

def _calc_stats_block(trades: list) -> str:
    """Считает статистику для списка сделок"""
    if not trades:
        return "нет данных"
    total = len(trades)
    wins = sum(1 for t in trades if t['won'])
    losses = total - wins
    winrate = round(wins / total * 100, 1)
    avg_win  = round(sum(t['pnl_pct'] for t in trades if t['won'])  / max(wins, 1), 2)
    avg_loss = round(sum(t['pnl_pct'] for t in trades if not t['won']) / max(losses, 1), 2)
    gross_profit = sum(t['pnl_pct'] for t in trades if t['pnl_pct'] > 0)
    gross_loss   = abs(sum(t['pnl_pct'] for t in trades if t['pnl_pct'] < 0))
    pf = round(gross_profit / max(gross_loss, 0.001), 2)
    expectancy = round(winrate/100 * avg_win + (1 - winrate/100) * avg_loss, 2)
    pf_e = "✅" if pf >= 1.5 else ("⚠️" if pf >= 1.0 else "❌")
    wr_bar = "🟩"*(int(winrate)//20) + "⬜"*(5-int(winrate)//20)
    return (
        f"Сделок: {total} | Винрейт: {wr_bar} {winrate}%\n"
        f"Avg win: +{avg_win}% | Avg loss: {avg_loss}%\n"
        f"Профит-фактор: {pf_e} {pf} | Expectancy: {expectancy:+.2f}%"
    )

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats = load_stats()
    trades = stats.get("trades", [])

    if not trades:
        await update.message.reply_text(
            "📭 Нет записанных сделок\n"
            "Статистика копится при закрытии сделок по ТП/СЛ."
        )
        return

    # Разделяем по режиму
    hard_trades    = [t for t in trades if t.get('mode') == 'hard']
    scanner_trades = [t for t in trades if t.get('mode') == 'scanner']
    clean_trades   = [t for t in trades if t.get('mode') not in ('hard', 'scanner')]
    all_trades     = trades

    best  = max(trades, key=lambda x: x['pnl_pct'])
    worst = min(trades, key=lambda x: x['pnl_pct'])

    sym_wins = Counter(t['symbol'] for t in trades if t['won'])
    top_sym  = sym_wins.most_common(3)

    recent_lines = []
    for t in reversed(trades[-5:]):
        mode_tag = f"[{t.get('mode','?')}]"
        emoji = "✅" if t['won'] else "❌"
        recent_lines.append(f"{emoji} {t['symbol']} {t['signal']} {t['pnl_pct']:+.2f}% {mode_tag}")

    first_date = trades[0]['time'][:10]

    text = (
        f"📈 <b>СТАТИСТИКА БОТА</b>\n"
        f"<i>С {first_date} | Всего: {len(all_trades)} сделок</i>\n\n"
        f"<b>🟡 MID/LOW (честная база — {len(clean_trades)} сделок):</b>\n"
        f"{_calc_stats_block(clean_trades)}\n\n"
        f"<b>🔴 HARD режим ({len(hard_trades)} сделок):</b>\n"
        f"{_calc_stats_block(hard_trades)}\n\n"
        f"<b>🔍 Сканер ({len(scanner_trades)} сделок):</b>\n"
        f"{_calc_stats_block(scanner_trades)}\n\n"
        f"<b>📊 Всего:</b>\n"
        f"{_calc_stats_block(all_trades)}\n\n"
        f"🏆 Лучшая: {best['symbol']} {best['pnl_pct']:+.2f}%\n"
        f"💀 Худшая: {worst['symbol']} {worst['pnl_pct']:+.2f}%\n\n"
        f"<b>Топ монеты:</b> " + " | ".join(f"{s}({c})" for s,c in top_sym) +
        f"\n\n<b>Последние 5:</b>\n" + "\n".join(recent_lines)
    )
    await update.message.reply_text(text[:4000], parse_mode='HTML')

def main():
    app = (Application.builder()
           .token(TELEGRAM_TOKEN)
           .post_init(post_init)
           .build())

    app.add_handler(CommandHandler("start",     cmd_start))
    app.add_handler(CommandHandler("scan",      cmd_scan))
    app.add_handler(CommandHandler("trades",    cmd_trades))
    app.add_handler(CommandHandler("mexc",      cmd_mexc))
    app.add_handler(CommandHandler("close",     cmd_close))
    app.add_handler(CommandHandler("stats",     cmd_stats))
    app.add_handler(CallbackQueryHandler(callback_close, pattern="^close_"))
    app.add_handler(MessageHandler(filters.COMMAND, handle_command))

    print("🚀 Signal Volume Bot v4 запущен")
    app.run_polling()

if __name__ == '__main__':
    main()

