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

import xml.etree.ElementTree as ET
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
DATABASE_URL    = os.getenv("DATABASE_URL", "")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRADES_FILE  = Path("open_trades.json")
SCANNER_FILE = Path("scanner_state.json")
STATS_FILE   = Path("bot_stats.json")

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

# ================== КЕШИ ==================
_cache: dict = {}
_macro_cache: dict = {}

# ================== СДЕЛКИ ==================
def load_trades() -> dict:
    try:
        return json.loads(TRADES_FILE.read_text()) if TRADES_FILE.exists() else {}
    except:
        return {}

def save_trades(trades: dict):
    TRADES_FILE.write_text(json.dumps(trades, indent=2, ensure_ascii=False))

def open_trade(symbol, tf, data, chat_id, mode="mid"):
    trades = load_trades()
    key = f"{symbol.replace('/','')}{tf}"
    trades[key] = {
        "symbol": symbol, "tf": tf, "chat_id": chat_id,
        "entry": data["price"], "signal": data["signal"],
        "sl":  data["sl_tp"]["sl"],
        "tp1": data["sl_tp"]["tp1"],
        "tp2": data["sl_tp"]["tp2"],
        "tp3": data["sl_tp"]["tp3"],
        "sl_moved_be": False, "sl_moved_tp1": False,
        "tp1_hit": False, "tp2_hit": False,
        "mode": mode,
        "opened_at": datetime.now().isoformat(),
    }
    save_trades(trades)
    return key

def close_trade(key):
    trades = load_trades()
    trades.pop(key, None)
    save_trades(trades)

# ================== СКАНЕР STATE ==================
def load_scanner_state() -> dict:
    try:
        return json.loads(SCANNER_FILE.read_text()) if SCANNER_FILE.exists() else {}
    except:
        return {}

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
                if data:
                    return _parse_binance(data), "Binance Futures"
    except Exception as e:
        logger.warning(f"BF {ticker}: {e}")

    # Binance Spot
    try:
        async with session.get("https://api.binance.com/api/v3/klines",
                params={"symbol": ticker, "interval": interval_bn, "limit": limit}) as r:
            if r.status == 200:
                data = await r.json()
                if data:
                    return _parse_binance(data), "Binance Spot"
    except Exception as e:
        logger.warning(f"BS {ticker}: {e}")

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
    except Exception as e:
        logger.warning(f"Bybit {ticker}: {e}")

    return None, None

def _parse_binance(data):
    df = pd.DataFrame(data, columns=[
        'ts','open','high','low','close','volume','ct','qv','trades','tbb','tbq','ignore'
    ])
    df = df[['ts','open','high','low','close','volume','tbb']].copy()
    for c in ['open','high','low','close','volume','tbb']:
        df[c] = pd.to_numeric(df[c])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    return df.rename(columns={'ts': 'timestamp', 'tbb': 'taker_buy_base'})

def _parse_bybit(data):
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume','turnover'])
    df = df[['ts','open','high','low','close','volume']].copy()
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c])
    df['ts'] = pd.to_datetime(pd.to_numeric(df['ts']), unit='ms')
    df = df.rename(columns={'ts': 'timestamp'})
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
            except:
                pass
            try:
                async with session.get("https://fapi.binance.com/fapi/v1/openInterest",
                                        params={"symbol": ticker}) as r:
                    if r.status == 200:
                        d = await r.json()
                        oi = float(d.get("openInterest", 0))
            except:
                pass
    return df, source, fr, oi

async def fetch_binance_futures_symbols(limit: int = 150) -> list:
    """Топ символы с Binance Futures по объёму, фильтр $5M суточного объёма"""
    try:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get("https://fapi.binance.com/fapi/v1/ticker/24hr") as r:
                if r.status == 200:
                    data = await r.json()
                    pairs = [
                        d for d in data
                        if d['symbol'].endswith('USDT')
                        and float(d.get('quoteVolume', 0)) >= 5_000_000
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
    if cache_key in _cache and now - _cache[cache_key]['ts'] < 3600:
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
        cur    = closes[-1]
        ema10  = float(pd.Series(closes).ewm(span=10).mean().iloc[-1])
        ema20  = float(pd.Series(closes).ewm(span=20).mean().iloc[-1])
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

# ================== МАКРО И НОВОСТИ ==================
async def fetch_macro_events() -> list:
    """RSS-ленты с макро событиями (ФРС, CPI, FOMC). Кеш 1 час."""
    cache_key = "macro_events"
    now = datetime.now().timestamp()
    if cache_key in _macro_cache and now - _macro_cache[cache_key].get('ts', 0) < 3600:
        return _macro_cache[cache_key]['val']

    events = []
    keywords = ['fed', 'fomc', 'cpi', 'powell', 'inflation', 'rate decision', 'nonfarm']
    urls = [
        "https://rss.investing.com/rss/news_14.rss",
        "https://cointelegraph.com/rss/tag/federal-reserve",
    ]
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        headers = {"User-Agent": "Mozilla/5.0"}
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for url in urls:
                try:
                    async with session.get(url, headers=headers) as r:
                        if r.status != 200:
                            continue
                        root = ET.fromstring(await r.text())
                        for item in root.findall('.//item')[:8]:
                            title = (item.findtext('title') or '').lower()
                            if any(kw in title for kw in keywords):
                                events.append({
                                    "title": item.findtext('title', '')[:130],
                                    "impact": "🔴"
                                })
                except Exception as e:
                    logger.warning(f"macro RSS {url}: {e}")
    except Exception as e:
        logger.warning(f"fetch_macro_events: {e}")

    result = events[:5]
    _macro_cache[cache_key] = {'val': result, 'ts': now}
    return result

async def fetch_hack_news() -> list:
    """Новости о взломах и эксплойтах. Кеш 30 минут."""
    cache_key = "hack_news"
    now = datetime.now().timestamp()
    if cache_key in _macro_cache and now - _macro_cache[cache_key].get('ts', 0) < 1800:
        return _macro_cache[cache_key]['val']

    hacks = []
    keywords = ['hack', 'exploit', 'stolen', 'drained', 'breach']
    urls = [
        "https://rekt.news/rss/",
        "https://cointelegraph.com/rss/tag/hacks",
    ]
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        headers = {"User-Agent": "Mozilla/5.0"}
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for url in urls:
                try:
                    async with session.get(url, headers=headers) as r:
                        if r.status != 200:
                            continue
                        root = ET.fromstring(await r.text())
                        for item in root.findall('.//item')[:5]:
                            title = item.findtext('title', '')
                            if any(kw in title.lower() for kw in keywords):
                                hacks.append({"title": title[:130]})
                except Exception as e:
                    logger.warning(f"hack RSS {url}: {e}")
    except Exception as e:
        logger.warning(f"fetch_hack_news: {e}")

    result = hacks[:4]
    _macro_cache[cache_key] = {'val': result, 'ts': now}
    return result

# ================== VOLUME PROFILE ==================
def calculate_volume_profile(df, num_bins=120):
    """Векторизованный Volume Profile на numpy. Быстрее iterrows в 15-20x."""
    price_min = df['low'].min()
    price_max = df['high'].max()
    if price_min == price_max:
        return np.array([price_min]), np.array([df['volume'].sum()])

    bins    = np.linspace(price_min, price_max, num_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    vp      = np.zeros(num_bins)

    lows    = df['low'].values
    highs   = df['high'].values
    volumes = df['volume'].values

    lo_idx = np.searchsorted(bins, lows,  side='left')  - 1
    hi_idx = np.searchsorted(bins, highs, side='right') - 1
    lo_idx = np.clip(lo_idx, 0, num_bins - 1)
    hi_idx = np.clip(hi_idx, 0, num_bins - 1)

    for i in range(len(volumes)):
        lo, hi = lo_idx[i], hi_idx[i]
        if lo == hi:
            vp[lo] += volumes[i]
        else:
            vp[lo:hi+1] += volumes[i] / (hi - lo + 1)

    return centers, vp

def find_hvn(vp, centers, price, dist_limit=25):
    threshold = np.percentile(vp, 70)
    nodes = []
    for i in range(1, len(vp) - 1):
        if vp[i] > threshold and vp[i] > vp[i-1] and vp[i] > vp[i+1]:
            dist = abs(centers[i] - price) / price * 100
            if dist < dist_limit:
                nodes.append({
                    "price":        round(float(centers[i]), 6),
                    "strength":     round(float(vp[i]), 2),
                    "distance_pct": round(float(dist), 2),
                    "is_above":     centers[i] > price,
                    "type":         "local"
                })
    nodes.sort(key=lambda x: -x['strength'])
    return nodes[:12]

def merge_hvn_levels(local_nodes, daily_nodes):
    for n in daily_nodes:
        n['type'] = 'daily'
    merged = []
    for n in sorted(local_nodes + daily_nodes, key=lambda x: -x['strength']):
        if not any(
            abs(n['price'] - m['price']) / max(n['price'], 0.0001) * 100 < 0.5
            for m in merged
        ):
            merged.append(n)
    return sorted(merged, key=lambda x: x['price'])

# ================== ТЕХНИЧЕСКИЙ АНАЛИЗ ==================
def detect_candle_pattern(df):
    c, p = df.iloc[-1], df.iloc[-2]
    body = abs(c['close'] - c['open'])
    rng  = c['high'] - c['low']
    if rng == 0:
        return "Дожи"
    uw = c['high'] - max(c['close'], c['open'])
    lw = min(c['close'], c['open']) - c['low']
    if lw > body * 2 and uw < body * 0.5:
        return "📌 Бычий пин-бар"
    if uw > body * 2 and lw < body * 0.5:
        return "📌 Медвежий пин-бар"
    if (c['close'] > c['open'] and p['close'] < p['open']
            and c['close'] > p['open'] and c['open'] < p['close']):
        return "🟢 Бычье поглощение"
    if (c['close'] < c['open'] and p['close'] > p['open']
            and c['close'] < p['open'] and c['open'] > p['close']):
        return "🔴 Медвежье поглощение"
    if body < rng * 0.1:
        return "〰️ Дожи"
    return "Обычная свеча"

def calculate_delta(df):
    r = df.tail(5)
    if 'taker_buy_base' not in r.columns:
        return "N/A"
    bv, tv = r['taker_buy_base'].sum(), r['volume'].sum()
    if tv == 0:
        return "N/A"
    bp = bv / tv * 100
    e  = "🟢" if bp > 55 else ("🔴" if bp < 45 else "⚪")
    return f"{e} {bp:.0f}% покупок / {100-bp:.0f}% продаж"

# ================== РЕЖИМ РЫНКА ==================
def detect_market_regime(df: pd.DataFrame, atr: float) -> dict:
    """
    Определяет режим рынка: trending_up / trending_down / ranging / volatile.
    Использует EMA slope, ATR ratio и расстояние цены от EMA20.
    """
    if len(df) < 60:
        return {"regime": "unknown", "label": "❓ Режим неизвестен",
                "trend_score_mult": 1.0, "hvn_score_mult": 1.0}

    close = df['close'].values
    price = close[-1]

    ema20_arr = ta.ema(pd.Series(close), length=20).values
    ema50_arr = ta.ema(pd.Series(close), length=50).values

    ema20_valid = ema20_arr[~np.isnan(ema20_arr)]
    ema50_valid = ema50_arr[~np.isnan(ema50_arr)]
    if len(ema20_valid) < 10 or len(ema50_valid) < 10:
        return {"regime": "unknown", "label": "❓ Режим неизвестен",
                "trend_score_mult": 1.0, "hvn_score_mult": 1.0}

    ema50_slope = (ema50_valid[-1] - ema50_valid[-10]) / ema50_valid[-10] * 100

    atr_series = ta.atr(df['high'], df['low'], df['close'], length=14).dropna()
    atr_mean   = float(atr_series.iloc[-50:].mean()) if len(atr_series) >= 20 else float(atr_series.mean())
    atr_now    = float(atr_series.iloc[-1]) if len(atr_series) > 0 else atr
    vol_ratio  = atr_now / atr_mean if atr_mean > 0 else 1.0

    ema20_dist = abs(price - ema20_valid[-1]) / price * 100

    is_volatile      = vol_ratio > 1.5
    is_trending_up   = ema50_slope >  0.3 and price > ema20_valid[-1] > ema50_valid[-1]
    is_trending_down = ema50_slope < -0.3 and price < ema20_valid[-1] < ema50_valid[-1]
    is_ranging       = abs(ema50_slope) < 0.15 and ema20_dist < 1.5

    if is_volatile and not (is_trending_up or is_trending_down):
        regime, label, trend_mult, hvn_mult = "volatile",      "⚡ Высокая волатильность", 0.8, 1.3
    elif is_trending_up:
        regime, label, trend_mult, hvn_mult = "trending_up",   "📈 Тренд вверх",           1.4, 0.8
    elif is_trending_down:
        regime, label, trend_mult, hvn_mult = "trending_down", "📉 Тренд вниз",             1.4, 0.8
    elif is_ranging:
        regime, label, trend_mult, hvn_mult = "ranging",       "↔️ Флэт/боковик",           0.6, 1.5
    else:
        regime, label, trend_mult, hvn_mult = "mixed",         "🔀 Смешанный",              1.0, 1.0

    return {
        "regime": regime,
        "label":  label,
        "ema50_slope":       round(ema50_slope, 3),
        "vol_ratio":         round(vol_ratio, 2),
        "trend_score_mult":  trend_mult,
        "hvn_score_mult":    hvn_mult,
    }

def find_sr_levels(df, price):
    r = df.tail(200)
    highs, lows = [], []
    for i in range(2, len(r) - 2):
        h, l = r.iloc[i]['high'], r.iloc[i]['low']
        if h > r.iloc[i-1]['high'] and h > r.iloc[i+1]['high']:
            highs.append(float(round(h, 6)))
        if l < r.iloc[i-1]['low'] and l < r.iloc[i+1]['low']:
            lows.append(float(round(l, 6)))
    return (
        sorted([l for l in lows if l < price], reverse=True)[:3],
        sorted([h for h in highs if h > price])[:3]
    )

def get_trend(df, label):
    if df is None or len(df) < 50:
        return "UNKNOWN", f"Нет данных {label}"
    close = df['close']
    e20   = ta.ema(close, length=20).iloc[-1]
    e50   = ta.ema(close, length=50).iloc[-1]
    cur   = close.iloc[-1]
    if cur > e20 > e50:
        return "UPTREND",   f"🟢 {label} аптренд"
    if cur < e20 < e50:
        return "DOWNTREND", f"🔴 {label} даунтренд"
    return "SIDEWAYS", f"⚪ {label} боковик"

# ================== RSI ДИВЕРГЕНЦИЯ ==================
def _find_extrema(arr: np.ndarray, order: int = 3):
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
    if len(df) < 40 or 'rsi' not in df.columns:
        return ""

    recent   = df.tail(50).copy()
    prices   = recent['low'].values
    highs    = recent['high'].values
    rsi_vals = recent['rsi'].values

    valid = ~(np.isnan(prices) | np.isnan(rsi_vals))
    if valid.sum() < 30:
        return ""

    prices   = prices[valid]
    highs    = highs[valid]
    rsi_vals = rsi_vals[valid]

    _, minima = _find_extrema(prices, order=3)
    maxima, _ = _find_extrema(highs,  order=3)

    if len(minima) >= 2:
        i1, i2 = minima[-2], minima[-1]
        if prices[i2] < prices[i1] * 0.998 and rsi_vals[i2] > rsi_vals[i1] + 2:
            strength = round(rsi_vals[i2] - rsi_vals[i1], 1)
            return f"🔄 Бычья дивергенция RSI (+{strength} пунктов)"

    if len(maxima) >= 2:
        i1, i2 = maxima[-2], maxima[-1]
        if highs[i2] > highs[i1] * 1.002 and rsi_vals[i2] < rsi_vals[i1] - 2:
            strength = round(rsi_vals[i1] - rsi_vals[i2], 1)
            return f"🔄 Медвежья дивергенция RSI (-{strength} пунктов)"

    return ""

# ================== ПРОБОЙ HVN С ОБЪЁМОМ ==================
def detect_hvn_breakout(df: pd.DataFrame, hv_nodes: list, price: float) -> str:
    if len(df) < 20 or not hv_nodes:
        return ""

    avg_vol      = df['volume'].tail(20).mean()
    last_vol     = df['volume'].iloc[-1]
    last_candle  = df.iloc[-1]
    prev_candle  = df.iloc[-2]

    if avg_vol == 0:
        return ""

    vol_ratio = last_vol / avg_vol

    for node in hv_nodes[:5]:
        np_ = node['price']
        if prev_candle['close'] < np_ < last_candle['close'] and vol_ratio >= 1.8:
            return f"💥 Пробой HVN {np_} вверх (объём x{vol_ratio:.1f})"
        if prev_candle['close'] > np_ > last_candle['close'] and vol_ratio >= 1.8:
            return f"💥 Пробой HVN {np_} вниз (объём x{vol_ratio:.1f})"

    return ""

# ================== ВНЕШНИЕ ДАННЫЕ ==================
async def fetch_btc_dominance() -> str:
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
                    dom  = round(data['data']['market_cap_percentage'].get('btc', 0), 1)
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
    """RSS CoinDesk + Cointelegraph. Кеш 15 мин."""
    coin      = symbol.replace("/USDT", "").replace("/", "").upper()
    cache_key = f"news_{coin}"
    now       = datetime.now().timestamp()
    if cache_key in _cache and now - _cache[cache_key]['ts'] < 900:
        return _cache[cache_key]['val']

    feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
    ]
    headlines = []
    timeout   = aiohttp.ClientTimeout(total=8)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for feed_url in feeds:
                if len(headlines) >= 3:
                    break
                try:
                    async with session.get(feed_url, headers={"User-Agent": "Mozilla/5.0"}) as r:
                        if r.status != 200:
                            continue
                        root  = ET.fromstring(await r.text())
                        items = root.findall('.//item')
                        for item in items:
                            title = item.findtext('title', '')
                            if coin.lower() in title.lower() or ('bitcoin' in title.lower() and coin == 'BTC'):
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
    """Динамика OI за 24 часа через историю Binance Futures."""
    try:
        ticker  = symbol.replace("/", "")
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                "https://fapi.binance.com/futures/data/openInterestHist",
                params={"symbol": ticker, "period": "1h", "limit": 24}
            ) as r:
                if r.status != 200:
                    return ""
                data = await r.json()
                if not data:
                    return ""

                oi_usdt      = [float(d['sumOpenInterestValue']) for d in data]
                curr_oi      = oi_usdt[-1]
                oi_change_24h = round((curr_oi - oi_usdt[0]) / max(oi_usdt[0], 1) * 100, 1)

                if oi_change_24h > 10:
                    return f"📈 OI +{oi_change_24h}% за 24ч (накопление позиций)"
                elif oi_change_24h < -10:
                    return f"📉 OI {oi_change_24h}% за 24ч (ликвидации/закрытия)"
                else:
                    return f"➡️ OI {oi_change_24h:+.1f}% за 24ч (стабильно)"
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

SL_PCT      = {"15m": 0.06, "1h": 0.09, "4h": 0.14, "1d": 0.22}
TF_MAX_RISK = {"15m": 0.08, "1h": 0.12, "4h": 0.18, "1d": 0.25}

def calculate_sl_tp(signal, price, atr, hv_nodes, supports=None, resistances=None, tf="15m"):
    if signal not in ("🟩 LONG", "🟥 SHORT"):
        return {}
    is_long      = signal == "🟩 LONG"
    below        = [n for n in hv_nodes if not n['is_above']]
    above        = [n for n in hv_nodes if n['is_above']]
    hvn_above_p  = [n['price'] for n in above]
    hvn_below_p  = [n['price'] for n in below]
    res = list(resistances or [])
    sup = list(supports or [])

    sl_pct = SL_PCT.get(tf, 0.05)

    if is_long:
        sl   = round(price * (1 - sl_pct), 6)
        risk = price - sl
        tp1  = _snap_to_level(price + risk * 1.5, hvn_above_p + res)
        tp2  = round(max(_snap_to_level(price + risk * 2.5, hvn_above_p + res, 1.0), tp1 + price * 0.005), 6)
        far  = [n for n in above if n['price'] > tp2]
        tp3  = round(max(
            max(far, key=lambda x: x['strength'])['price'] if far else price + risk * 4.0,
            price + risk * 3.5, tp2 + price * 0.01), 6)
    else:
        sl   = round(price * (1 + sl_pct), 6)
        risk = sl - price
        tp1  = _snap_to_level(price - risk * 1.5, hvn_below_p + sup)
        tp2  = round(min(_snap_to_level(price - risk * 2.5, hvn_below_p + sup, 1.0), tp1 - price * 0.005), 6)
        far  = [n for n in below if n['price'] < tp2]
        tp3  = round(min(
            min(far, key=lambda x: x['strength'])['price'] if far else price - risk * 4.0,
            price - risk * 3.5, tp2 - price * 0.01), 6)
        tp3  = round(max(tp3, price * 0.01), 6)

    rr       = round(abs(tp2 - price) / max(abs(price - sl), 0.0001), 2)
    risk_pct = round(abs(price - sl) / price * 100, 2)

    warns = []
    if rr < 1.5:      warns.append("⚠️ R/R ниже 1.5")
    if risk_pct > 5:  warns.append(f"⚠️ Риск {risk_pct}% — уменьши размер позиции")

    return {
        "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "risk_pct":  risk_pct,
        "rr_ratio":  rr,
        "rr_warn":   " | ".join(warns),
    }

# ================== ВОЛАТИЛЬНОСТЬ МОНЕТЫ ==================
def classify_coin_volatility(atr_pct: float, tf: str) -> str:
    """
    Классифицирует монету по волатильности на основе ATR%.
    Пороги подобраны под реальный рынок:
      - 15m ATR > 1.5% → высокая (мемкоины, низкие капы)
      - 1h  ATR > 2.5% → высокая
      - 4h  ATR > 4.5% → высокая
      - 1d  ATR > 8%   → высокая
    Возвращает "high" / "normal".
    """
    thresholds = {"15m": 1.5, "1h": 2.5, "4h": 4.5, "1d": 8.0}
    limit = thresholds.get(tf, 1.5)
    return "high" if atr_pct >= limit else "normal"

# ================== СКОРИНГ → СИГНАЛ ==================
def compute_score_and_signal(
        rsi: float, price: float, atr: float,
        ema20: float, ema50: float,
        top_hvn, vp_mean: float,
        delta_str: str, trend_l: str, trend_h: str,
        candle: str, rsi_div: str, hvn_break: str,
        regime: dict, mode_cfg: dict,
        atr_pct: float = 1.0, tf: str = "15m"
) -> tuple:
    """
    Многоуровневый скоринг: качество > количество.

    Принципы:
    - Сигнал только при совпадении минимум 2 независимых факторов
    - Для волатильных монет (ATR% высокий) — особо жёсткий фильтр
    - WATCH убран совсем: либо сигнал, либо нет
    - Нормализованный скор 0–100 отражает реальную силу, не накрутку

    Возвращает (signal, reason, score, detail).
    """
    rsi_long  = mode_cfg["rsi_long"]
    rsi_short = mode_cfg["rsi_short"]
    hvn_mult  = mode_cfg["hvn_mult"]
    is_hard   = mode_cfg.get("label", "") == "🔴 HARD"

    trend_mult = regime.get("trend_score_mult", 1.0)
    hvn_w      = regime.get("hvn_score_mult",   1.0)
    market     = regime.get("regime", "mixed")

    coin_vol   = classify_coin_volatility(atr_pct, tf)
    is_hv_coin = coin_vol == "high"

    try:
        buy_pct = float(delta_str.split('%')[0].split()[-1])
    except:
        buy_pct = 50.0

    long_score  = 0.0
    short_score = 0.0
    long_reasons  = []
    short_reasons = []

    # ── 1. HVN / Volume Profile (макс ~30 баллов) ──────────────────────────
    # Для волатильных монет требуем более сильную полку (mult +0.5)
    hvn_threshold = hvn_mult + (0.5 if is_hv_coin else 0.0)
    if top_hvn:
        strength_ratio = top_hvn['strength'] / max(vp_mean, 0.0001)
        if strength_ratio > hvn_threshold:
            # Для волатильных монет HVN чуть менее надёжна (они легко пробивают)
            hv_penalty = 0.75 if is_hv_coin else 1.0
            hvn_pts    = min(28, strength_ratio * 9) * hvn_w * hv_penalty
            if top_hvn['is_above']:
                short_score += hvn_pts
                short_reasons.append(f"Сильная HVN сверху {top_hvn['price']} ({strength_ratio:.1f}x)")
            else:
                long_score += hvn_pts
                long_reasons.append(f"Сильная HVN снизу {top_hvn['price']} ({strength_ratio:.1f}x)")

    # ── 2. Тренд: ТОЛЬКО при совпадении TF + HTF (макс ~20 баллов) ─────────
    # Одиночный тренд без HTF = 0 баллов. Слишком дёшево давать баллы за тренд.
    if trend_l == "UPTREND" and trend_h == "UPTREND":
        # Боковик на рынке снижает ценность тренда
        pts = 20 * trend_mult
        long_score += pts
        long_reasons.append("Двойной аптренд TF+HTF")
    elif trend_l == "DOWNTREND" and trend_h == "DOWNTREND":
        pts = 20 * trend_mult
        short_score += pts
        short_reasons.append("Двойной даунтренд TF+HTF")
    # Одиночный тренд — 0 баллов, только как tiebreaker в HARD

    # ── 3. RSI — только экстремальные зоны (макс ~15 баллов) ──────────────
    # Убираем "слабые" RSI 35-45 — слишком много ложных
    # Для волатильных монет RSI гуляет сильнее → пороги жёстче
    rsi_long_thr  = 30 if is_hv_coin else rsi_long
    rsi_short_thr = 72 if is_hv_coin else rsi_short

    if rsi < rsi_long_thr:
        # Только глубокая перепроданность
        pts = 15 if rsi < 25 else 10
        long_score += pts
        long_reasons.append(f"RSI {rsi} — глубокая перепроданность")
    elif rsi > rsi_short_thr:
        pts = 15 if rsi > 75 else 10
        short_score += pts
        short_reasons.append(f"RSI {rsi} — сильная перекупленность")
    # RSI 35–65 = 0 баллов. Нейтральная зона не должна давать очки.

    # ── 4. Дивергенция RSI — сильный разворотный сигнал (макс ~22 балла) ──
    if rsi_div:
        # Для волатильных монет дивергенция менее надёжна
        div_pts = 16 if is_hv_coin else 22
        if "Бычья" in rsi_div:
            long_score  += div_pts
            long_reasons.append(rsi_div)
        elif "Медвежья" in rsi_div:
            short_score += div_pts
            short_reasons.append(rsi_div)

    # ── 5. Пробой HVN с объёмом — самый сильный сигнал (макс ~25 баллов) ──
    if hvn_break:
        # Для волатильных монет пробои часто ложные — снижаем вес
        brk_pts = 18 if is_hv_coin else 25
        if "вверх" in hvn_break:
            long_score  += brk_pts
            long_reasons.append(hvn_break)
        elif "вниз" in hvn_break:
            short_score += brk_pts
            short_reasons.append(hvn_break)

    # ── 6. Свечной паттерн (макс ~10 баллов) ──────────────────────────────
    if "Бычье поглощение" in candle or ("пин-бар" in candle and "Бычий" in candle):
        long_score  += 10
        long_reasons.append(candle)
    elif "Медвежье поглощение" in candle or ("пин-бар" in candle and "Медвежий" in candle):
        short_score += 10
        short_reasons.append(candle)

    # ── 7. Дельта объёма — только при сильном перекосе (макс ~9 баллов) ───
    delta_long_thr  = 68 if is_hv_coin else 62
    delta_short_thr = 32 if is_hv_coin else 38
    if buy_pct > delta_long_thr:
        long_score  += 9
        long_reasons.append(f"Дельта бычья ({buy_pct:.0f}%)")
    elif buy_pct < delta_short_thr:
        short_score += 9
        short_reasons.append(f"Дельта медвежья ({buy_pct:.0f}%)")

    # ── 8. Боковик — штрафуем одиночные тренд-сигналы ────────────────────
    # Если рынок в флэте, тренды ненадёжны — уже учтено через trend_mult=0.6
    # Дополнительно: в боковике без HVN и дивергенции — нет сигнала

    # ══ ФИНАЛЬНЫЙ РАСЧЁТ ══════════════════════════════════════════════════
    # Теоретический максимум: HVN(28) + тренд(25) + RSI(18) + пробой(25) + div(22) + свеча(10) + дельта(9) ≈ 137
    # Реальный "отличный сигнал": 3–4 фактора = 60–90 баллов → нормализуем к этому
    max_possible = 88.0

    diff             = long_score - short_score
    abs_winner       = max(long_score, short_score)
    normalized_score = min(100, int(abs_winner / max_possible * 100))

    # Количество сработавших независимых факторов
    long_factors  = len(long_reasons)
    short_factors = len(short_reasons)

    # Минимальные требования к сигналу:
    # - обычный: минимум 2 фактора, разрыв ≥ 20, скор ≥ 65
    # - hard:    минимум 2 фактора, разрыв ≥ 10, скор ≥ 50
    # - волатильная монета: пороги ещё жёстче (+20%)
    hv_boost    = 1.20 if is_hv_coin else 1.0
    min_diff    = int((10 if is_hard else 20) * hv_boost)
    min_score   = int((50 if is_hard else 65) * hv_boost)
    min_factors = 2  # всегда минимум 2 независимых фактора

    long_ok  = (long_score  > short_score and diff >= min_diff
                and normalized_score >= min_score and long_factors  >= min_factors)
    short_ok = (short_score > long_score  and -diff >= min_diff
                and normalized_score >= min_score and short_factors >= min_factors)

    if long_ok:
        signal = "🟩 LONG"
        reason = " + ".join(long_reasons[:3])
    elif short_ok:
        signal = "🟥 SHORT"
        reason = " + ".join(short_reasons[:3])
    elif is_hard and normalized_score >= int(40 * hv_boost):
        # HARD: по тренду, если хоть что-то есть
        if trend_l == "UPTREND":
            signal, reason = "🟩 LONG",  f"HARD: аптренд (скор {normalized_score})"
        elif trend_l == "DOWNTREND":
            signal, reason = "🟥 SHORT", f"HARD: даунтренд (скор {normalized_score})"
        else:
            signal = "🟩 LONG" if buy_pct > 52 else "🟥 SHORT"
            reason = f"HARD: дельта {buy_pct:.0f}%"
    else:
        signal = "НЕТ СИГНАЛА"
        reason = (
            f"Скоринг {normalized_score}/100"
            + (f" | {long_factors} факт. лонг / {short_factors} факт. шорт" if long_factors or short_factors else "")
            + (f" | ⚡Волатильная монета" if is_hv_coin else "")
        )

    detail = {
        "long_score":    round(long_score, 1),
        "short_score":   round(short_score, 1),
        "long_reasons":  long_reasons,
        "short_reasons": short_reasons,
        "coin_vol":      coin_vol,
    }
    return signal, reason, normalized_score, detail

# ================== АНАЛИЗ ==================
async def analyze_symbol(symbol, tf="15m", mode_cfg=None):
    if mode_cfg is None:
        mode_cfg = TRADE_MODES["mid"]

    try:
        results = await asyncio.gather(
            fetch_ohlcv(symbol, tf),
            fetch_higher_tf(symbol, tf),
            fetch_daily_vp(symbol),
            fetch_btc_dominance(),
            fetch_crypto_news(symbol),
            fetch_weekly_trend(symbol),
            fetch_macro_events(),
            fetch_hack_news(),
            return_exceptions=True
        )

        df, source, fr, oi = results[0] if not isinstance(results[0], Exception) else (None, None, None, None)
        df_htf, htf_label  = results[1] if not isinstance(results[1], Exception) else (None, None)
        df_daily           = results[2] if not isinstance(results[2], Exception) else None
        btc_dom            = results[3] if not isinstance(results[3], Exception) else ""
        news               = results[4] if not isinstance(results[4], Exception) else ""
        weekly_trend       = results[5] if not isinstance(results[5], Exception) else ""
        macro_ev           = results[6] if not isinstance(results[6], Exception) else []
        hack_ev            = results[7] if not isinstance(results[7], Exception) else []

    except Exception as e:
        logger.error(f"analyze_symbol error {symbol}: {e}")
        return None

    if df is None or len(df) < 52:
        return None

    # Работаем только с закрытыми свечами
    df_closed     = df.iloc[:-1].copy()
    price         = df_closed['close'].iloc[-1]
    current_price = df['close'].iloc[-1]

    # Санити-чек: если цены расходятся более чем на 20% — данные с разных бирж/символов
    if price > 0 and abs(current_price - price) / price > 0.20:
        logger.warning(f"Price sanity fail {symbol}: closed={price} current={current_price} — skip")
        return None

    df_closed['rsi'] = ta.rsi(df_closed['close'], length=14)
    df_closed['atr'] = ta.atr(df_closed['high'], df_closed['low'], df_closed['close'], length=14)
    rsi     = round(df_closed['rsi'].iloc[-1], 1)
    atr     = round(df_closed['atr'].iloc[-1], 6)
    atr_pct = round(atr / price * 100, 2)
    ema20   = round(ta.ema(df_closed['close'], length=20).iloc[-1], 6)
    ema50   = round(ta.ema(df_closed['close'], length=50).iloc[-1], 6)
    df = df_closed  # дальше работаем с закрытыми

    if price > ema20 > ema50:
        ema_trend = "📈 Восходящий"
    elif price < ema20 < ema50:
        ema_trend = "📉 Нисходящий"
    else:
        ema_trend = "↔️ Боковик"

    centers_l, vp_l = calculate_volume_profile(df)
    poc         = round(float(centers_l[np.argmax(vp_l)]), 6)
    local_nodes = find_hvn(vp_l, centers_l, price)

    daily_nodes = []
    if df_daily is not None and len(df_daily) > 50:
        centers_d, vp_d = calculate_volume_profile(df_daily, num_bins=150)
        daily_nodes = find_hvn(vp_d, centers_d, price, dist_limit=30)

    all_nodes  = merge_hvn_levels(local_nodes, daily_nodes)
    hvn_above  = [n for n in all_nodes if n['is_above']]
    hvn_below  = [n for n in all_nodes if not n['is_above']]

    supports, resistances = find_sr_levels(df, price)
    candle    = detect_candle_pattern(df)
    delta     = calculate_delta(df)
    rsi_div   = detect_rsi_divergence(df)
    hvn_break = detect_hvn_breakout(df, local_nodes, price)
    oi_trend  = await fetch_liq_levels(symbol, price)
    trend_l, trend_l_txt = get_trend(df, tf)
    trend_h, trend_h_txt = (
        get_trend(df_htf, htf_label) if df_htf is not None
        else ("UNKNOWN", "Нет данных HTF")
    )

    vp_mean     = float(np.mean(vp_l))
    strong_above = [n for n in hvn_above if n['distance_pct'] < 12]
    top_hvn      = strong_above[0] if strong_above else None

    regime = detect_market_regime(df, atr)

    signal, reason, score, score_detail = compute_score_and_signal(
        rsi=rsi, price=price, atr=atr,
        ema20=ema20, ema50=ema50,
        top_hvn=top_hvn, vp_mean=vp_mean,
        delta_str=delta, trend_l=trend_l, trend_h=trend_h,
        candle=candle, rsi_div=rsi_div, hvn_break=hvn_break,
        regime=regime, mode_cfg=mode_cfg,
        atr_pct=atr_pct, tf=tf
    )

    htf_conflict = ""
    if signal == "🟩 LONG"  and trend_h == "DOWNTREND":
        htf_conflict = f"⚠️ LONG против тренда {htf_label}!"
    if signal == "🟥 SHORT" and trend_h == "UPTREND":
        htf_conflict = f"⚠️ SHORT против тренда {htf_label}!"

    sl_tp = calculate_sl_tp(signal, price, atr, all_nodes, supports, resistances, tf)

    return {
        "symbol": symbol, "tf": tf,
        "price":         round(price, 6),
        "current_price": round(current_price, 6),
        "signal": signal, "reason": reason, "score": score,
        "rsi": rsi, "atr": atr, "atr_pct": atr_pct,
        "ema_trend": ema_trend, "poc": poc,
        "hvn_above": hvn_above, "hvn_below": hvn_below,
        "supports": supports, "resistances": resistances,
        "candle_pattern": candle, "delta": delta,
        "trend_local": trend_l_txt, "trend_higher": trend_h_txt,
        "htf_conflict": htf_conflict, "sl_tp": sl_tp, "source": source,
        "funding_rate":  round(fr, 4) if fr is not None else None,
        "open_interest": int(oi) if oi is not None else None,
        "rsi_divergence": rsi_div,
        "hvn_breakout":   hvn_break,
        "oi_trend":       oi_trend,
        "btc_dominance":  btc_dom,
        "news":           news,
        "weekly_trend":   weekly_trend,
        "time":           datetime.now().strftime("%H:%M"),
        "btc_trend_text": "",
        "mode_label":       mode_cfg["label"],
        "mode_personality": mode_cfg["personality"],
        "regime":       regime,
        "score_detail": score_detail,
        "coin_vol": score_detail.get("coin_vol", "normal"),
        "macro_events": macro_ev,
        "hack_news":    hack_ev,
    }

# ================== GROQ AI ==================
async def ask_ai(data: dict) -> str:
    if not groq_client:
        return "AI отключён (нет GROQ_API_KEY)"

    sl_tp    = data.get("sl_tp", {})
    tf       = data['tf']
    score    = data.get('score', 0)
    signal   = data['signal']
    price    = data['price']

    hvn_a    = [n['price'] for n in data['hvn_above'][:3]]
    hvn_b    = [n['price'] for n in data['hvn_below'][:3]]
    poc      = data['poc']
    sup      = data.get('supports', [])[:2]
    res      = data.get('resistances', [])[:2]
    conflict = data.get('htf_conflict', '')

    tf_ctx = {
        "15m": "скальпинг, сделка живёт 1-4 часа",
        "1h":  "интрадей, сделка живёт 4-24 часа",
        "4h":  "свинг, сделка живёт 2-7 дней",
        "1d":  "позиция, сделка живёт 1-4 недели"
    }.get(tf, "интрадей")

    rr       = float(sl_tp.get('rr_ratio', 0)) if sl_tp else 0
    risk_pct = float(sl_tp.get('risk_pct', 0)) if sl_tp else 0

    best_limit = None
    if hvn_b and abs(hvn_b[0] - price) / price < 0.05:
        best_limit = hvn_b[0]
    elif abs(poc - price) / price < 0.05:
        best_limit = poc
    elif sup:
        best_limit = sup[0]

    # Макро контекст для AI
    macro_ctx = ""
    if data.get('macro_events'):
        macro_lines = [ev.get('title', '')[:80] for ev in data['macro_events'][:2]]
        macro_ctx = "\nМакро события: " + " | ".join(macro_lines)
    if data.get('hack_news'):
        hack_lines = [h.get('title', '')[:60] for h in data['hack_news'][:1]]
        macro_ctx += "\nХаки/риски: " + " | ".join(hack_lines)

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
Неделя: {data.get('weekly_trend', 'нет данных')}{macro_ctx}

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
                    await asyncio.sleep(min(int(m2.group(1)) + 2 if m2 else 12, 30))
                elif attempt == 0:
                    await asyncio.sleep(3)
    return "⏳ AI: лимит запросов"

# ================== ЛИМИТНЫЕ ОРДЕРА ==================
def calculate_limit_orders(result: dict) -> dict:
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
        break_dn = levels_below[0]
        entry_dn = round(break_dn * 0.998, 6)
        dist_pct = round((price - break_dn) / price * 100, 2)
        sl_dn    = round(break_dn + atr * 0.8, 6)
        tp_dn    = round(entry_dn - (sl_dn - entry_dn) * 2.0, 6)
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
    if not limit_data:
        return ""
    orders = limit_data.get('orders', [])
    if not orders:
        return ""
    esc   = html_escape
    lines = ["\n📋 <b>ЛИМИТНЫЕ ОРДЕРА:</b>"]
    for o in orders[:1]:
        lines.append(
            f"\n{esc(o['type'])}\n"
            f"├ Вход: <b>{o['entry']}</b> ({esc(o['reason'])})\n"
            f"├ СЛ: {o['sl']}\n"
            f"└ ТП: {o['tp1']}"
        )
    return "\n".join(lines)

# ================== ФОРМАТИРОВАНИЕ ==================
def format_message(result, ai_text, is_scanner=False, limit_data=None):
    esc  = html_escape
    sl_tp = result.get("sl_tp", {})
    p     = result['price']

    def pct(t):
        return round(abs(t - p) / p * 100, 2)

    trade_block = ""
    if sl_tp and result['signal'] in ("🟩 LONG", "🟥 SHORT"):
        rr_warn = f"\n{esc(sl_tp['rr_warn'])}" if sl_tp.get('rr_warn') else ""
        trade_block = (
            f"\n📐 <b>ПЛАН СДЕЛКИ</b>\n"
            f"├ 🛑 СЛ: <b>{sl_tp['sl']}</b> (-{pct(sl_tp['sl'])}%)\n"
            f"├ 🎯 ТП1: <b>{sl_tp['tp1']}</b> (+{pct(sl_tp['tp1'])}%) → <i>БУ</i>\n"
            f"├ 🎯 ТП2: <b>{sl_tp['tp2']}</b> (+{pct(sl_tp['tp2'])}%) → <i>СЛ на ТП1</i>\n"
            f"└ 🏆 ТП3: <b>{sl_tp['tp3']}</b> (+{pct(sl_tp['tp3'])}%) → <i>финал</i>\n"
            f"R/R: 1:{sl_tp['rr_ratio']} | Риск: {sl_tp['risk_pct']}%{rr_warn}\n"
        )

    score     = result.get('score', 0)
    score_bar = "🟩" * (score // 20) + "⬜" * (5 - score // 20)

    fr_str = ""
    if result.get('funding_rate') is not None:
        fr     = result['funding_rate']
        fe     = "🔴" if fr > 0.05 else ("🟢" if fr < -0.05 else "⚪")
        fr_str = f"\nФандинг: {fe} {fr}%"
    oi_str   = f" | OI: {result['open_interest']:,}" if result.get('open_interest') else ""
    conflict = f"\n{esc(result['htf_conflict'])}" if result.get('htf_conflict') else ""
    rsi_div  = f"\n{esc(result['rsi_divergence'])}" if result.get('rsi_divergence') else ""
    hvn_brk  = f"\n{esc(result['hvn_breakout'])}"  if result.get('hvn_breakout') else ""
    oi_trend = f"\n{esc(result['oi_trend'])}"       if result.get('oi_trend') else ""
    btc_dom  = f"\n{esc(result['btc_dominance'])}"  if result.get('btc_dominance') else ""

    def fmt_nodes(nodes):
        out = [f"{'🌍' if n.get('type') == 'daily' else '📍'}{n['price']}" for n in nodes[:3]]
        return " ".join(out) if out else "—"

    header = "🔔 <b>АВТОСИГНАЛ</b>\n" if is_scanner else ""

    parts = [
        f"{header}📊 <b>{esc(result['symbol'])}</b> {result['tf']} • {result['time']} • {esc(result['mode_label'])}\n",
        f"<i>{esc(result['source'])}</i>\n\n",
        f"Цена свечи: <b>{p}</b>",
    ]
    if result.get('current_price', p) != p:
        parts.append(f" | Сейчас: <b>{result.get('current_price', p)}</b>")
    parts += [
        f"\nСигнал: <b>{esc(result['signal'])}</b>\n",
        f"Причина: {esc(result['reason'])}\n",
        f"Скоринг: {score_bar} <b>{score}/100</b>\n\n",
        f"RSI: {result['rsi']} | ATR: {result['atr']} ({result['atr_pct']}%){' ⚡Волат.' if result.get('coin_vol') == 'high' else ''}\n",
        f"EMA: {esc(result['ema_trend'])} | Свеча: {esc(result['candle_pattern'])}\n",
        f"Дельта: {esc(result['delta'])}{fr_str}{oi_str}\n\n",
        f"🌡️ Режим: {esc(result.get('regime', {}).get('label', ''))}\n",
        f"📈 Тренд {result['tf']}: {esc(result['trend_local'])}\n",
        f"📊 Тренд HTF: {esc(result['trend_higher'])}\n",
    ]
    if result.get('weekly_trend'):
        parts.append(f"{esc(result['weekly_trend'])}\n")

    # Макро события — одна секция, без дублирования
    if result.get('macro_events'):
        macro_lines = [
            f"🔴 {esc(ev.get('title', '')[:100])}"
            for ev in result['macro_events'][:2]
        ]
        if macro_lines:
            parts.append(f"\n<b>🔴 Макро:</b>\n" + "\n".join(macro_lines) + "\n")

    # Хак-новости
    if result.get('hack_news'):
        hack_lines = [
            f"💀 {esc(h.get('title', '')[:80])}"
            for h in result['hack_news'][:2]
        ]
        if hack_lines:
            parts.append("\n".join(hack_lines) + "\n")

    parts += [
        f"{conflict}{rsi_div}{hvn_brk}{oi_trend}{btc_dom}\n\n",
        f"POC: {result['poc']}\n",
        f"HVN↑: {fmt_nodes(result['hvn_above'])}\n",
        f"HVN↓: {fmt_nodes(result['hvn_below'])}\n",
        f"Сопр: {result['resistances'][:2]} | Подд: {result['supports'][:2]}\n",
        f"{trade_block}\n",
        f"{format_limit_orders(limit_data)}\n",
        f"🧠 <b>AI:</b>\n{esc(ai_text[:1200])}",
    ]
    return "".join(parts)

# ================== MEXC API ==================
async def mexc_request(method: str, path: str, params: dict = None) -> dict | None:
    if not MEXC_API_KEY or not MEXC_SECRET_KEY:
        return None
    params = params or {}
    ts = str(int(time.time() * 1000))
    params['timestamp'] = ts
    query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    sig   = hmac.new(MEXC_SECRET_KEY.encode(), query.encode(), hashlib.sha256).hexdigest()
    params['signature'] = sig
    url     = f"https://contract.mexc.com{path}"
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
    data = await mexc_request("GET", "/api/v1/private/position/open_positions")
    if not data or data.get('code') != 200:
        return []
    return data.get('data', [])

async def fetch_mexc_position(symbol: str) -> dict | None:
    mexc_sym  = symbol.replace("/", "_")
    positions = await fetch_mexc_positions()
    for p in positions:
        if p.get('symbol') == mexc_sym:
            return p
    return None

async def get_mexc_sl_tp(symbol: str) -> dict | None:
    mexc_sym = symbol.replace("/", "_")
    data = await mexc_request("GET", "/api/v1/private/stoporder/list/orders",
                               {"symbol": mexc_sym, "states": "1"})
    if not data or data.get('code') != 200:
        return None
    orders = data.get('data', {}).get('resultList', [])
    result = {}
    for o in orders:
        price = float(o.get('triggerPrice', 0))
        otype = o.get('orderType')
        if otype == 3 and price > 0:
            result['sl'] = price
        elif otype == 1 and price > 0:
            result['tp'] = price
    return result if result else None

async def sync_mexc_trades(app=None):
    """Синхронизирует позиции MEXC с ботом."""
    if not MEXC_API_KEY:
        return
    trades = load_trades()
    if not trades:
        return
    mexc_positions = await fetch_mexc_positions()
    mexc_symbols   = {p.get('symbol') for p in mexc_positions}

    for key, t in list(trades.items()):
        mexc_sym = t['symbol'].replace("/", "_")
        if mexc_sym not in mexc_symbols:
            logger.info(f"MEXC: позиция {mexc_sym} закрыта на бирже")
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

# ================== ПАМП/ДАМП ДЕТЕКТОР ==================
pump_scanner_active  = {}
_pump_scanner_running = False

def _load_pump_active() -> dict:
    state = load_scanner_state()
    raw   = state.get("pump_active_chats", {})
    return {int(k): v for k, v in raw.items()}

def _save_pump_active(active: dict):
    state = load_scanner_state()
    state["pump_active_chats"] = {str(k): v for k, v in active.items()}
    save_scanner_state(state)

pump_scanner_active = _load_pump_active()

def detect_pump_dump(df: pd.DataFrame, funding_rate: float = None) -> dict | None:
    """
    Памп/Дамп детектор с корректными уровнями входа.

    ПАМП: свеча +6.5%+ с объёмом x2 и RSI скачком +10
      → Лонг на откате к ближайшей HVN НИЖЕ цены (всегда < price_now)
      → Шорт от ближайшей HVN ВЫШЕ цены (всегда > price_now)

    ДАМП: памп +10% за 5 свечей + RSI>75 + объём угасает + фандинг
      → Шорт от ближайшей HVN ВЫШЕ (сопротивление)
      → СЛ строго выше хая свечи (= выше входа)
    """
    if len(df) < 30:
        return None

    closed    = df.iloc[:-1]
    last      = closed.iloc[-1]
    prev      = closed.iloc[-2]

    price_now  = float(last['close'])
    price_prev = float(prev['close'])
    vol_last   = float(last['volume'])
    avg_vol    = float(closed['volume'].tail(20).mean())
    if avg_vol == 0:
        return None

    candle_high = float(last['high'])
    vol_ratio   = vol_last / avg_vol
    candle_chg  = (price_now - price_prev) / price_prev * 100

    rsi_series = ta.rsi(closed['close'], length=14)
    rsi_now    = float(rsi_series.iloc[-1]) if rsi_series is not None else 50
    rsi_prev5  = float(rsi_series.iloc[-6]) if len(rsi_series) > 6 else rsi_now
    rsi_jump   = rsi_now - rsi_prev5

    price_5ago = float(closed.iloc[-6]['close']) if len(closed) > 6 else price_prev
    total_pump = (price_now - price_5ago) / price_5ago * 100

    vol_prev2  = float(closed.iloc[-2]['volume'])
    vol_prev3  = float(closed.iloc[-3]['volume'])
    vol_fading = vol_last < vol_prev2 * 0.8 and vol_prev2 < vol_prev3 * 0.9

    buy_pct = 50.0
    if 'taker_buy_base' in closed.columns and vol_last > 0:
        buy_pct = float(last['taker_buy_base']) / vol_last * 100

    # Volume Profile по последним 100 свечам
    centers, vp = calculate_volume_profile(closed.tail(100), num_bins=80)
    poc_price   = round(float(centers[vp.argmax()]), 6)
    vp_mean     = float(vp.mean())

    # HVN строго ниже текущей цены (для лонга на откате)
    hvn_below_prices = sorted([
        round(float(centers[i]), 6)
        for i in range(1, len(vp) - 1)
        if vp[i] > 1.5 * vp_mean and vp[i] > vp[i-1] and vp[i] > vp[i+1]
        and centers[i] < price_now
    ], reverse=True)[:3]  # ближайшая первая

    # HVN строго выше текущей цены (для шорта от сопротивления)
    hvn_above_prices = sorted([
        round(float(centers[i]), 6)
        for i in range(1, len(vp) - 1)
        if vp[i] > 1.5 * vp_mean and vp[i] > vp[i-1] and vp[i] > vp[i+1]
        and centers[i] > price_now
    ])[:3]  # ближайшая первая

    # ── Уровни ЛОНГ на откате (всегда ниже price_now) ──────────────────
    # Откат к ближайшей HVN снизу или -3% если HVN нет
    if hvn_below_prices:
        pullback_entry = hvn_below_prices[0]
    elif poc_price < price_now:
        pullback_entry = poc_price
    else:
        pullback_entry = round(price_now * 0.97, 6)

    pullback_pct = round((price_now - pullback_entry) / price_now * 100, 1)
    # СЛ лонга: ниже уровня на 1.5% (строго меньше входа)
    long_sl  = round(pullback_entry * 0.985, 6)
    long_risk = pullback_entry - long_sl
    # ТП лонга: R/R 1:2 и 1:3 от уровня откате, не от текущей цены
    long_tp1 = round(pullback_entry + long_risk * 2.0, 6)
    long_tp2 = round(pullback_entry + long_risk * 3.5, 6)

    # ── Уровни ШОРТ от сопротивления (всегда выше price_now) ──────────
    if hvn_above_prices:
        short_entry = hvn_above_prices[0]
    else:
        short_entry = round(price_now * 1.02, 6)

    # СЛ шорта: выше хая свечи (строго больше входа)
    short_sl   = round(max(candle_high, short_entry) * 1.003, 6)
    short_risk = short_sl - short_entry
    # ТП шорта: цели вниз — ближайшие HVN снизу
    short_tp1  = hvn_below_prices[0] if hvn_below_prices else round(price_now * 0.95, 6)
    short_tp2  = hvn_below_prices[1] if len(hvn_below_prices) > 1 else round(price_now * 0.90, 6)

    # ── ПАМП ─────────────────────────────────────────────────────────────
    if candle_chg >= 6.5 and vol_ratio >= 2.0 and rsi_jump >= 10:
        strength = "🔥🔥🔥" if candle_chg >= 12 else ("🔥🔥" if candle_chg >= 9 else "🔥")
        return {
            "type": "PUMP", "signal": f"⚡ ПАМП {strength}",
            "price":        round(price_now, 6),
            "candle_chg":   round(candle_chg, 2),
            "vol_ratio":    round(vol_ratio, 1),
            "rsi":          round(rsi_now, 1),
            "rsi_jump":     round(rsi_jump, 1),
            "buy_pct":      round(buy_pct, 1),
            "total_pump":   round(total_pump, 2),
            "poc":          poc_price,
            # Лонг на откате
            "pullback_entry": pullback_entry,
            "pullback_pct":   pullback_pct,
            "long_sl":        long_sl,
            "long_tp1":       long_tp1,
            "long_tp2":       long_tp2,
            # Шорт от сопротивления
            "short_entry": short_entry,
            "short_sl":    short_sl,
            "short_tp1":   short_tp1,
            "short_tp2":   short_tp2,
            "reason": f"Свеча +{candle_chg:.1f}% | Объём x{vol_ratio:.1f} | RSI {rsi_now:.0f} (+{rsi_jump:.0f})",
        }

    # ── ДАМП ─────────────────────────────────────────────────────────────
    dump_score = sum([
        total_pump >= 10.0,
        rsi_now >= 75,
        vol_fading,
        (funding_rate or 0) >= 0.05
    ])
    if dump_score >= 3:
        confidence = "🔴🔴🔴" if dump_score == 4 else "🔴🔴"
        return {
            "type": "DUMP", "signal": f"🔻 ДАМП вероятен {confidence}",
            "price":      round(price_now, 6),
            "candle_chg": round(candle_chg, 2),
            "vol_ratio":  round(vol_ratio, 1),
            "rsi":        round(rsi_now, 1),
            "total_pump": round(total_pump, 2),
            "funding":    funding_rate,
            "poc":        poc_price,
            "short_entry": short_entry,
            "short_sl":    short_sl,
            "short_tp1":   short_tp1,
            "short_tp2":   short_tp2,
            "reason": (
                f"Памп +{total_pump:.1f}% за 5св | RSI {rsi_now:.0f} | "
                f"Объём {'↘️угасает' if vol_fading else '→держится'} | "
                f"Фандинг {funding_rate or 0:.3f}%"
            ),
        }
    return None

def format_pump_message(symbol: str, tf: str, det: dict) -> str:
    esc     = html_escape
    is_pump = det['type'] == 'PUMP'
    price   = det['price']

    icon   = "⚡" if is_pump else "🔻"
    header = f"{icon} <b>АНОМАЛЬНЫЙ СИГНАЛ</b> | <b>{esc(symbol)}</b> {tf}\n"
    stats  = (
        f"Цена сейчас: <b>{price}</b> | {esc(det['signal'])}\n"
        f"📊 {esc(det['reason'])}\n"
    )

    if is_pump:
        emoji = "🟢" if det['buy_pct'] > 60 else "⚪"

        # Расстояния от текущей цены для понимания
        entry_l = det['pullback_entry']
        entry_s = det['short_entry']
        dist_l  = round((price - entry_l) / price * 100, 1)   # всегда > 0
        dist_s  = round((entry_s - price) / price * 100, 1)   # всегда > 0

        # R/R для лонга
        rr_l = round((det['long_tp1'] - entry_l) / max(entry_l - det['long_sl'], 0.0001), 1)

        levels = (
            f"\n📐 <b>УРОВНИ:</b>\n"
            f"POC (магнит): <b>{det['poc']}</b>\n\n"
            f"<b>📥 ЛОНГ на откате:</b>\n"
            f"├ Вход лимит: <b>{entry_l}</b> (-{dist_l}% ждём)\n"
            f"├ СЛ: {det['long_sl']} (-{round((entry_l - det['long_sl'])/entry_l*100,1)}% от входа)\n"
            f"├ ТП1: {det['long_tp1']} (R/R 1:{rr_l})\n"
            f"└ ТП2: {det['long_tp2']}\n\n"
            f"<b>📤 ШОРТ от сопротивления:</b>\n"
            f"├ Вход лимит: <b>{entry_s}</b> (+{dist_s}% ждём)\n"
            f"├ СЛ: {det['short_sl']} (+{round((det['short_sl'] - entry_s)/entry_s*100,1)}% от входа)\n"
            f"├ ТП1: {det['short_tp1']}\n"
            f"└ ТП2: {det['short_tp2']}\n\n"
            f"Дельта: {emoji} {det['buy_pct']}% покупок | Суммарно: +{det['total_pump']}%"
        )
    else:
        entry_s = det['short_entry']
        dist_s  = round((entry_s - price) / price * 100, 1)

        levels = (
            f"\n📐 <b>УРОВНИ:</b>\n"
            f"POC (магнит): <b>{det['poc']}</b>\n\n"
            f"<b>📤 ШОРТ:</b>\n"
            f"├ Вход лимит: <b>{entry_s}</b> (+{dist_s}% от цены)\n"
            f"├ СЛ: {det['short_sl']} (выше максимума)\n"
            f"├ ТП1: {det['short_tp1']}\n"
            f"└ ТП2: {det.get('short_tp2', '—')}\n"
        )

    return header + stats + levels + "\n⚠️ <i>Не входит в статистику</i>"

# ================== МОНИТОРИНГ СДЕЛОК ==================
async def fetch_current_price(symbol: str) -> float | None:
    """Текущая цена через lightweight ticker — быстро и точно."""
    ticker = symbol.replace("/", "")
    timeout = aiohttp.ClientTimeout(total=8)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Binance Futures
            async with session.get(
                "https://fapi.binance.com/fapi/v1/ticker/price",
                params={"symbol": ticker}
            ) as r:
                if r.status == 200:
                    d = await r.json()
                    return float(d["price"])
    except:
        pass
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Binance Spot fallback
            async with session.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={"symbol": ticker}
            ) as r:
                if r.status == 200:
                    d = await r.json()
                    return float(d["price"])
    except:
        pass
    return None

async def check_trades(app):
    trades = load_trades()
    if not trades:
        return

    for key, t in list(trades.items()):
        try:
            monitor_tf = t.get('tf', '15m')

            # Текущая цена — быстрый ticker, без задержки
            current_price = await fetch_current_price(t['symbol'])
            if current_price is None:
                logger.warning(f"check_trades {key}: no price from ticker")
                continue

            # Свечи нужны для диапазона high/low закрытых периодов
            df_monitor, _, _, _ = await fetch_ohlcv(t['symbol'], monitor_tf)
            if df_monitor is None or len(df_monitor) < 3:
                continue

            # Binance последней возвращает ТЕКУЩУЮ незакрытую свечу (iloc[-1])
            # Закрытые: iloc[-2] и iloc[-3]
            closed_last = df_monitor.iloc[-2]
            closed_prev = df_monitor.iloc[-3]

            # Объединяем диапазон двух закрытых свечей + текущую цену
            period_high = max(float(closed_last['high']), float(closed_prev['high']), current_price)
            period_low  = min(float(closed_last['low']),  float(closed_prev['low']),  current_price)

            price   = current_price
            is_long = "LONG" in t['signal']
            closed  = False
            msgs    = []

            check_up   = period_high
            check_down = period_low

            # ── ТП1 ──────────────────────────────────────────────────────
            if not t['tp1_hit']:
                tp1_reached = (is_long and check_up >= t['tp1']) or \
                              (not is_long and check_down <= t['tp1'])
                if tp1_reached:
                    t['tp1_hit']     = True
                    t['sl']          = t['entry']
                    t['sl_moved_be'] = True
                    msgs.append(
                        "\U0001f3af <b>ТП1 достигнут!</b>\n"
                        + f"{t['symbol']} {t['tf']} | Цена: {round(price, 6)}\n"
                        + f"СЛ перенесён в безубыток \u2192 <b>{t['entry']}</b>\n"
                        + f"Следующая цель ТП2: {t['tp2']}"
                    )

            # ── ТП2 (сразу после ТП1 — цена могла пройти оба за одну свечу) ──
            if t['tp1_hit'] and not t['tp2_hit']:
                tp2_reached = (is_long and check_up >= t['tp2']) or \
                              (not is_long and check_down <= t['tp2'])
                if tp2_reached:
                    t['tp2_hit']      = True
                    sl_before_tp2     = t['entry']
                    t['sl']           = t['tp1']
                    t['sl_moved_tp1'] = True
                    record_trade_result(
                        t['symbol'], t['tf'], t['signal'], t['entry'],
                        sl_before_tp2, t['tp1'], t['tp2'], t['tp3'],
                        t['tp2'], "tp2", 0, mode=t.get('mode', 'mid')
                    )
                    msgs.append(
                        "\U0001f3af\U0001f3af <b>ТП2 достигнут!</b>\n"
                        + f"{t['symbol']} {t['tf']} | Цена: {round(price, 6)}\n"
                        + f"СЛ перенесён на ТП1 \u2192 <b>{t['tp1']}</b>\n"
                        + f"Финальная цель ТП3: {t['tp3']}"
                    )

            # ── ТП3 ──────────────────────────────────────────────────────
            if t['tp1_hit'] and t['tp2_hit'] and not closed:
                tp3_reached = (is_long and check_up >= t['tp3']) or \
                              (not is_long and check_down <= t['tp3'])
                if tp3_reached:
                    record_trade_result(
                        t['symbol'], t['tf'], t['signal'], t['entry'],
                        t['sl'], t['tp1'], t['tp2'], t['tp3'],
                        t['tp3'], "tp3", 0, mode=t.get('mode', 'mid')
                    )
                    msgs.append(
                        "\U0001f3c6 <b>ТП3 достигнут! Полная цель!</b>\n"
                        + f"{t['symbol']} {t['tf']} | Цена: {round(price, 6)}\n"
                        + "Сделка успешно закрыта \U0001f389"
                    )
                    close_trade(key)
                    trades.pop(key, None)
                    closed = True

            # ── СЛ (только если сделка ещё не закрыта по ТП) ────────────
            if not closed and key in trades:
                sl_reached = (is_long and check_down <= t['sl']) or \
                             (not is_long and check_up >= t['sl'])
                if sl_reached:
                    tag    = "безубыток" if t.get('sl_moved_be') else "стоп-лосс"
                    exit_p = round(t['sl'], 6)
                    pnl    = round(
                        (exit_p - t['entry']) / t['entry'] * 100 * (1 if is_long else -1), 2
                    )
                    record_trade_result(
                        t['symbol'], t['tf'], t['signal'], t['entry'],
                        t['sl'], t['tp1'], t['tp2'], t['tp3'],
                        exit_p, tag, 0, mode=t.get('mode', 'mid')
                    )
                    pnl_str = f"+{pnl}%" if pnl > 0 else f"{pnl}%"
                    msgs.append(
                        "\U0001f6d1 <b>Закрыто по " + tag + "</b>\n"
                        + f"{t['symbol']} {t['tf']} | Цена: {exit_p}\n"
                        + f"P&L: <b>{pnl_str}</b>"
                    )
                    close_trade(key)
                    trades.pop(key, None)
                    closed = True

            # ── Сохраняем если сделка всё ещё открыта ────────────────────
            if not closed and key in trades:
                save_trades(trades)

            # ── Уведомления ──────────────────────────────────────────────
            chat_id = t.get('chat_id')
            if msgs:
                if chat_id:
                    for m in msgs:
                        try:
                            await app.bot.send_message(chat_id=chat_id, text=m, parse_mode='HTML')
                        except Exception as send_err:
                            logger.error(f"notify {key}: {send_err}")
                else:
                    logger.warning(f"check_trades {key}: no chat_id, msgs={msgs}")

        except Exception as ex:
            logger.error(f"check_trades {key}: {ex}", exc_info=True)



# ================== АВТОСКАНЕР ==================
scanner_threshold    = 80   # минимальный скор для сигнала
MAX_SIGNALS_PER_RUN  = 10   # мягкий лимит на прогон (защита от шторма)
_scanner_running     = False

def _load_scanner_active() -> dict:
    state = load_scanner_state()
    raw   = state.get("active_chats", {})
    return {int(k): v for k, v in raw.items()}

def _save_scanner_active(active: dict):
    state = load_scanner_state()
    state["active_chats"] = {str(k): v for k, v in active.items()}
    save_scanner_state(state)

scanner_active = _load_scanner_active()

async def run_scanner(app):
    global _scanner_running
    if _scanner_running:
        logger.info("Scanner: already running, skip")
        return
    _scanner_running = True
    try:
        active_chats = [cid for cid, active in scanner_active.items() if active]
        if not active_chats:
            return

        state    = load_scanner_state()
        last_run = state.get("last_run", 0)
        now      = datetime.now().timestamp()
        if (now - last_run) < 900:
            return

        state["last_run"] = now
        save_scanner_state(state)

        logger.info("Scanner: starting scan...")
        symbols = await fetch_binance_futures_symbols(limit=150)
        if not symbols:
            logger.warning("Scanner: no symbols fetched")
            return

        mode_cfg      = TRADE_MODES["mid"]
        scan_tfs      = ["15m", "1h"]
        sent_this_run = []
        MAX_SIGNALS_PER_RUN = 5  # максимум сигналов за одну итерацию

        for tf_scan in scan_tfs:
            logger.info(f"Scanner: {len(symbols)} symbols on {tf_scan}")
            for i in range(0, len(symbols), 5):
                batch   = symbols[i:i+5]
                results = await asyncio.gather(
                    *[analyze_symbol(s, tf_scan, mode_cfg) for s in batch],
                    return_exceptions=True
                )
                for result in results:
                    if isinstance(result, Exception) or result is None:
                        continue
                    if result['signal'] not in ("🟩 LONG", "🟥 SHORT"):
                        continue

                    # ── Фильтр 1: минимальный скор ──────────────────────────
                    if result['score'] < scanner_threshold:
                        continue

                    # ── Фильтр 2: HTF конфликт — блокируем полностью ────────
                    # Контртрендовые сделки слишком рискованны для автосканера
                    if result.get('htf_conflict'):
                        continue

                    # ── Фильтр 3: минимум 2 фактора ─────────────────────────
                    detail = result.get('score_detail', {})
                    winning_reasons = (
                        detail.get('long_reasons', []) if 'LONG' in result['signal']
                        else detail.get('short_reasons', [])
                    )
                    if len(winning_reasons) < 2:
                        continue

                    # ── Фильтр 4: волатильные монеты — скор 92+ ─────────────
                    if result.get('coin_vol') == 'high' and result['score'] < 92:
                        continue

                    # ── Фильтр 5: боковик — запрещаем тренд-сигналы ─────────
                    regime = result.get('regime', {}).get('regime', 'mixed')
                    if regime == 'ranging':
                        # В боковике только пробои HVN и дивергенции
                        has_breakout = bool(result.get('hvn_breakout'))
                        has_div      = bool(result.get('rsi_divergence'))
                        if not has_breakout and not has_div:
                            continue

                    # ── Лимит на прогон ─────────────────────────────────────
                    if len(sent_this_run) >= MAX_SIGNALS_PER_RUN:
                        logger.info(f"Scanner: hit MAX_SIGNALS_PER_RUN={MAX_SIGNALS_PER_RUN}, stopping")
                        break

                    symbol  = result['symbol']
                    sig_key = f"sent_{symbol}_{tf_scan}"
                    if (now - state.get(sig_key, 0)) < 21600:  # 6 часов антиспам
                        continue
                    state[sig_key] = now
                    save_scanner_state(state)
                    sent_this_run.append(f"{symbol}/{tf_scan}")

                    result['btc_trend_text'] = ""
                    ai_text    = await ask_ai(result) if groq_client else "AI отключён"
                    limit_data = calculate_limit_orders(result)
                    msg        = format_message(result, ai_text, is_scanner=True, limit_data=limit_data)
                    TG_LIMIT   = 4000
                    ai_split   = msg.find("🧠 <b>AI:</b>")

                    # Открываем сделку ОДИН РАЗ на символ (не на каждый чат)
                    # chat_id берём первый активный — мониторинг общий
                    trade_key = None
                    if result.get('sl_tp') and result['signal'] in ("🟩 LONG", "🟥 SHORT"):
                        # Не открываем если уже есть открытая сделка по этому символу+tf
                        existing = load_trades()
                        trade_key_candidate = f"{symbol.replace('/','')}{tf_scan}"
                        if trade_key_candidate not in existing:
                            trade_key = open_trade(symbol, tf_scan, result, active_chats[0], mode="scanner")
                        else:
                            logger.info(f"Scanner: skip trade {trade_key_candidate} — already open")

                    for chat_id in active_chats:
                        try:
                            if ai_split > 0 and len(msg) > TG_LIMIT:
                                await app.bot.send_message(
                                    chat_id=chat_id,
                                    text=msg[:ai_split].rstrip()[:TG_LIMIT],
                                    parse_mode='HTML'
                                )
                                await app.bot.send_message(
                                    chat_id=chat_id,
                                    text=msg[ai_split:][:TG_LIMIT],
                                    parse_mode='HTML'
                                )
                            else:
                                await app.bot.send_message(
                                    chat_id=chat_id,
                                    text=msg[:TG_LIMIT],
                                    parse_mode='HTML'
                                )
                            if trade_key:
                                await app.bot.send_message(
                                    chat_id=chat_id,
                                    text=f"📌 Сделка добавлена в мониторинг\n/close {trade_key}"
                                )
                        except Exception as ex:
                            logger.error(f"Scanner send {chat_id}: {ex}")
                await asyncio.sleep(1)

        logger.info(f"Scanner done. Signals sent: {sent_this_run}")

    except Exception as e:
        logger.error(f"run_scanner error: {e}", exc_info=True)
    finally:
        _scanner_running = False

async def run_pump_scanner(app):
    global _pump_scanner_running
    if _pump_scanner_running:
        return
    _pump_scanner_running = True
    try:
        active_chats = [cid for cid, v in pump_scanner_active.items() if v]
        if not active_chats:
            return

        state   = load_scanner_state()
        now     = datetime.now().timestamp()
        symbols = await fetch_binance_futures_symbols(limit=150)
        if not symbols:
            return

        for i in range(0, len(symbols), 10):
            batch       = symbols[i:i+10]
            raw_results = await asyncio.gather(*[fetch_ohlcv(s, "15m") for s in batch], return_exceptions=True)

            for j, raw in enumerate(raw_results):
                if isinstance(raw, Exception) or raw is None:
                    continue
                df, source, fr, oi = raw
                if df is None or len(df) < 30:
                    continue

                symbol = batch[j]
                det    = detect_pump_dump(df, fr)
                if det is None:
                    continue

                spam_key = f"pump_{symbol}_{det['type']}"
                if (now - state.get(spam_key, 0)) < 1800:
                    continue

                state[spam_key] = now
                save_scanner_state(state)

                msg = format_pump_message(symbol, "15m", det)
                for chat_id in active_chats:
                    try:
                        await app.bot.send_message(
                            chat_id=chat_id, text=msg[:4000], parse_mode='HTML'
                        )
                    except Exception as ex:
                        logger.error(f"pump_scanner send {chat_id}: {ex}")

            await asyncio.sleep(0.5)

    except Exception as e:
        logger.error(f"run_pump_scanner: {e}", exc_info=True)
    finally:
        _pump_scanner_running = False

# ================== СТАТИСТИКА ==================
def _get_pg_conn():
    if not DATABASE_URL:
        return None
    try:
        import psycopg2
        return psycopg2.connect(DATABASE_URL, sslmode="require")
    except Exception as e:
        logger.warning(f"PG connect failed: {e}")
        return None

def _ensure_pg_table():
    conn = _get_pg_conn()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bot_trades (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20),
                    tf VARCHAR(5),
                    signal VARCHAR(20),
                    entry FLOAT,
                    exit_price FLOAT,
                    pnl_pct FLOAT,
                    exit_reason VARCHAR(20),
                    score INT,
                    won BOOLEAN,
                    mode VARCHAR(20),
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
        conn.commit()
    except Exception as e:
        logger.warning(f"PG ensure table: {e}")
    finally:
        conn.close()

def load_stats() -> dict:
    conn = _get_pg_conn()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT symbol,tf,signal,entry,exit_price,pnl_pct,"
                    "exit_reason,score,won,mode,created_at::text "
                    "FROM bot_trades ORDER BY id"
                )
                rows = cur.fetchall()
            conn.close()
            trades = [
                {
                    "symbol": r[0], "tf": r[1], "signal": r[2],
                    "entry":  r[3], "exit": r[4], "pnl_pct": r[5],
                    "exit_reason": r[6], "score": r[7], "won": r[8],
                    "mode": r[9], "time": r[10]
                }
                for r in rows
            ]
            return {"trades": trades}
        except Exception as e:
            logger.warning(f"PG load: {e}")
    try:
        return json.loads(STATS_FILE.read_text()) if STATS_FILE.exists() else {}
    except:
        return {}

def save_stats(stats: dict):
    STATS_FILE.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

def save_trade_to_pg(trade: dict):
    conn = _get_pg_conn()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO bot_trades
                    (symbol,tf,signal,entry,exit_price,pnl_pct,exit_reason,score,won,mode)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                trade["symbol"], trade["tf"],    trade["signal"],
                trade["entry"],  trade["exit"],  trade["pnl_pct"],
                trade["exit_reason"], trade["score"], trade["won"], trade["mode"]
            ))
        conn.commit()
    except Exception as e:
        logger.warning(f"PG save trade: {e}")
    finally:
        conn.close()

def record_trade_result(symbol: str, tf: str, signal: str, entry: float,
                        sl: float, tp1: float, tp2: float, tp3: float,
                        exit_price: float, exit_reason: str, score: int,
                        mode: str = "mid"):
    if mode == "hard":
        logger.info(f"record_trade_result: skip hard mode {symbol}")
        return
    stats   = load_stats()
    is_long = "LONG" in signal
    pnl_pct = round(
        ((exit_price - entry) / entry * 100) if is_long
        else ((entry - exit_price) / entry * 100), 2
    )
    trade = {
        "symbol": symbol, "tf": tf, "signal": signal,
        "entry":  entry,  "exit": exit_price,
        "pnl_pct": pnl_pct, "exit_reason": exit_reason,
        "score": score, "won": pnl_pct > 0,
        "mode": mode,
        "time": datetime.now().isoformat()
    }
    save_trade_to_pg(trade)
    if "trades" not in stats:
        stats["trades"] = []
    stats["trades"].append(trade)
    save_stats(stats)

def _calc_stats_block(trades: list) -> str:
    if not trades:
        return "нет данных"
    total    = len(trades)
    wins     = sum(1 for t in trades if t['won'])
    losses   = total - wins
    winrate  = round(wins / total * 100, 1)
    avg_win  = round(sum(t['pnl_pct'] for t in trades if t['won'])      / max(wins,   1), 2)
    avg_loss = round(sum(t['pnl_pct'] for t in trades if not t['won'])  / max(losses, 1), 2)
    gross_profit = sum(t['pnl_pct'] for t in trades if t['pnl_pct'] > 0)
    gross_loss   = abs(sum(t['pnl_pct'] for t in trades if t['pnl_pct'] < 0))
    pf           = round(gross_profit / max(gross_loss, 0.001), 2)
    expectancy   = round(winrate / 100 * avg_win + (1 - winrate / 100) * avg_loss, 2)
    pf_e         = "✅" if pf >= 1.5 else ("⚠️" if pf >= 1.0 else "❌")
    wr_bar       = "🟩" * (int(winrate) // 20) + "⬜" * (5 - int(winrate) // 20)
    return (
        f"Сделок: {total} | Винрейт: {wr_bar} {winrate}%\n"
        f"Avg win: +{avg_win}% | Avg loss: {avg_loss}%\n"
        f"Профит-фактор: {pf_e} {pf} | Expectancy: {expectancy:+.2f}%"
    )

# ================== КОМАНДЫ ==================
MODES = {"low", "mid", "hard"}
TFS   = set(TF_MAP.keys())

def parse_args(text):
    parts  = text.lower().strip().lstrip("/").split()
    coin   = parts[0].upper().replace("USDT", "").replace("/", "")
    symbol = f"{coin}/USDT"
    tf     = next((p for p in parts[1:] if p in TFS),   DEFAULT_TF)
    mode   = next((p for p in parts[1:] if p in MODES), "mid")
    return symbol, tf, mode

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ <b>Signal Volume Bot v4</b>\n\n"
        "<b>Анализ монеты:</b>\n"
        "/btc — BTC 15m mid\n"
        "/eth 4h — ETH 4h mid\n"
        "/sol hard — SOL агрессив\n"
        "/btc 1h low — BTC 1h консерватив\n\n"
        "<b>Таймфреймы:</b> 15m 1h 4h 1d\n"
        "<b>Режимы:</b> 🟢low 🟡mid 🔴hard\n\n"
        "<b>Сканер авто-сигналов:</b>\n"
        "/scan on — включить\n"
        "/scan off — выключить\n"
        "/scan top — топ сигналы сейчас\n"
        "/scan debug — почему молчит\n"
        "/scan status — статус\n\n"
        "<b>Памп/Дамп детектор:</b>\n"
        "/pump on — включить\n"
        "/pump off — выключить\n\n"
        "<b>Сделки:</b>\n"
        "/trades — открытые позиции\n"
        "/close BTCUSDT15m — закрыть\n"
        "/stats — статистика бота\n\n"
        "📊 Бэктест:\n"
        "/backtest BTCUSDT 15m 30 — симуляция за 30 дней\n\n"
        "🌍 = дневная полка | 📍 = локальная HVN",
        parse_mode='HTML'
    )

async def cmd_pump(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    cmd     = context.args[0].lower() if context.args else "status"

    if cmd == "on":
        pump_scanner_active[chat_id] = True
        _save_pump_active(pump_scanner_active)
        await update.message.reply_text(
            "⚡ <b>Памп/Дамп детектор включён</b>\n\n"
            "Бот мониторит 150 монет каждые 3 минуты.\n"
            "Уведомления при:\n"
            "• Свеча 15m +6.5%+ с объёмом x2\n"
            "• Памп угасает (RSI >75 + объём падает)\n\n"
            "<i>Сигналы не входят в статистику</i>",
            parse_mode='HTML'
        )
    elif cmd == "off":
        pump_scanner_active[chat_id] = False
        _save_pump_active(pump_scanner_active)
        await update.message.reply_text("⏹ Памп/Дамп детектор выключен")
    else:
        active = pump_scanner_active.get(chat_id, False)
        await update.message.reply_text(
            f"⚡ Памп детектор: {'🟢 активен' if active else '🔴 выключен'}\n"
            f"/pump on — включить\n"
            f"/pump off — выключить"
        )

async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    cmd     = context.args[0].lower() if context.args else "status"

    if cmd == "on":
        scanner_active[chat_id] = True
        _save_scanner_active(scanner_active)
        state = load_scanner_state()
        state["last_run"] = 0
        save_scanner_state(state)
        await update.message.reply_text(
            "🔍 <b>Сканер включён</b>\n"
            f"Порог скоринга: {scanner_threshold}/100\n"
            "Первый скан запустится через ~1 минуту\n"
            "Далее каждые 15 минут",
            parse_mode='HTML'
        )

    elif cmd == "off":
        scanner_active[chat_id] = False
        _save_scanner_active(scanner_active)
        await update.message.reply_text("⏹ Сканер выключен")

    elif cmd == "top":
        msg      = await update.message.reply_text("🔍 Сканирую топ-100 монет (15m + 1h)\nПодожди ~2 минуты...")
        symbols  = await fetch_binance_futures_symbols(limit=100)
        mode_cfg = TRADE_MODES["mid"]
        results  = []

        for tf_top in ["15m", "1h"]:
            for i in range(0, len(symbols), 5):
                batch = symbols[i:i+5]
                batch_results = await asyncio.gather(
                    *[analyze_symbol(s, tf_top, mode_cfg) for s in batch],
                    return_exceptions=True
                )
                for r in batch_results:
                    if isinstance(r, Exception) or r is None:
                        continue
                    if r['signal'] in ("🟩 LONG", "🟥 SHORT"):
                        results.append(r)
                await asyncio.sleep(0.5)

        if not results:
            await msg.edit_text(
                "😶 Нет сигналов в топ-100 прямо сейчас\n\n"
                f"Текущий порог сканера: {scanner_threshold}/100\n"
                "Используй /scan debug для диагностики"
            )
            return

        results.sort(key=lambda x: -x['score'])
        lines = [f"📊 <b>Топ сигналы ({len(results)} найдено):</b>\n"]
        for r in results[:30]:
            bar    = "🟩" * (r['score'] // 20) + "⬜" * (5 - r['score'] // 20)
            thresh = "✅" if r['score'] >= scanner_threshold else "⚠️"
            lines.append(
                f"{thresh} {r['signal']} <b>{r['symbol']}</b> {r['tf']}\n"
                f"  {bar} {r['score']}/100 | RSI:{r['rsi']} | {r['reason']}"
            )
        await msg.edit_text("\n".join(lines), parse_mode='HTML')

    elif cmd == "debug":
        msg      = await update.message.reply_text("🔧 Диагностика: анализирую топ-30...")
        symbols  = await fetch_binance_futures_symbols(limit=30)
        mode_cfg = TRADE_MODES["mid"]
        passed, blocked_score, blocked_conflict, no_signal = [], [], [], []

        for i in range(0, len(symbols), 5):
            batch = symbols[i:i+5]
            batch_results = await asyncio.gather(
                *[analyze_symbol(s, "15m", mode_cfg) for s in batch],
                return_exceptions=True
            )
            for r in batch_results:
                if isinstance(r, Exception) or r is None:
                    continue
                sig      = r['signal']
                sc       = r['score']
                conflict = bool(r.get('htf_conflict'))
                if sig not in ("🟩 LONG", "🟥 SHORT"):
                    no_signal.append(f"{r['symbol']}({sc})")
                elif sc < scanner_threshold:
                    blocked_score.append(f"{r['symbol']} {sig}({sc})")
                elif conflict and sc < scanner_threshold + 15:
                    blocked_conflict.append(f"{r['symbol']} {sig}({sc})")
                else:
                    passed.append(f"{r['symbol']} {sig}({sc})")
            await asyncio.sleep(0.5)

        all_scored = passed + blocked_score + blocked_conflict
        scores_all = []
        for item in all_scored:
            try:
                scores_all.append(int(item.split('(')[-1].rstrip(')')))
            except:
                pass

        avg_score = round(sum(scores_all) / len(scores_all), 1) if scores_all else 0
        max_score = max(scores_all) if scores_all else 0

        text = (
            f"🔧 <b>Диагностика сканера</b>\n"
            f"Порог: {scanner_threshold}/100\n"
            f"Средний скоринг топ-30: <b>{avg_score}</b> | Макс: <b>{max_score}</b>\n\n"
            f"✅ Прошли бы ({len(passed)}): {', '.join(passed[:5]) or 'нет'}\n"
            f"⛔ Низкий скор ({len(blocked_score)}): {', '.join(blocked_score[:5]) or 'нет'}\n"
            f"⚠️ HTF конфликт ({len(blocked_conflict)}): {', '.join(blocked_conflict[:3]) or 'нет'}\n"
            f"➖ Нет сигнала ({len(no_signal)}): {', '.join(no_signal[:5]) or 'нет'}\n\n"
            f"💡 Если средний скор < 50 — рынок в боковике, сигналов мало"
        )
        await msg.edit_text(text, parse_mode='HTML')

    else:  # status
        active   = scanner_active.get(chat_id, False)
        state    = load_scanner_state()
        last     = state.get("last_run", 0)
        last_str = datetime.fromtimestamp(last).strftime("%H:%M:%S") if last else "никогда"
        now      = datetime.now().timestamp()
        blocked  = sum(1 for k, v in state.items() if k.startswith("sent_") and now - v < 7200)
        await update.message.reply_text(
            f"📡 Сканер: {'🟢 активен' if active else '🔴 выключен'}\n"
            f"Последний скан: {last_str}\n"
            f"Порог скоринга: {scanner_threshold}/100\n"
            f"Антиспам заблокировал: {blocked} монет\n\n"
            f"Команды:\n"
            f"/scan on — включить\n"
            f"/scan off — выключить\n"
            f"/scan top — топ сигналы прямо сейчас\n"
            f"/scan debug — почему молчит сканер"
        )

async def cmd_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    trades = load_trades()
    if not trades:
        await update.message.reply_text("📭 Нет открытых сделок")
        return
    for k, t in trades.items():
        is_long   = "LONG" in t['signal']
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
    if not MEXC_API_KEY:
        await update.message.reply_text(
            "❌ MEXC API не настроен\n\n"
            "Добавь в Railway Variables:\n"
            "MEXC_API_KEY = твой_ключ\n"
            "MEXC_SECRET_KEY = твой_секрет"
        )
        return

    msg       = await update.message.reply_text("🔄 Загружаю позиции с MEXC...")
    positions = await fetch_mexc_positions()

    if not positions:
        await msg.edit_text("📭 Нет открытых позиций на MEXC")
        return

    bot_trades = load_trades()
    lines      = [f"📊 <b>Позиции MEXC ({len(positions)}):</b>\n"]

    for p in positions:
        sym    = p.get('symbol', '').replace('_', '/')
        side   = "🟩 LONG" if p.get('positionType') == 1 else "🟥 SHORT"
        vol    = p.get('vol', 0)
        entry  = p.get('openAvgPrice', 0)
        upnl   = round(float(p.get('unrealisedPnl', 0)), 2)
        liq    = p.get('liquidatePrice', 0)
        in_bot = "📌 в боте" if sym.replace('/', '') + '15m' in bot_trades else "⚠️ нет в боте"

        lines.append(
            f"{'✅' if upnl >= 0 else '❌'} {side} <b>{sym}</b> {in_bot}\n"
            f"Вход: {entry} | Объём: {vol}\n"
            f"P&L: {upnl:+.2f} USDT | Ликвидация: {liq}\n"
        )

    await msg.edit_text("\n".join(lines), parse_mode='HTML')

async def cmd_close(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "Использование:\n"
            "/close КЛЮЧ — закрыть одну\n"
            "/close all — закрыть все"
        )
        return

    trades = load_trades()
    key    = context.args[0]

    if key.lower() == "all":
        if not trades:
            await update.message.reply_text("📭 Нет открытых сделок")
            return
        count = len(trades)
        save_trades({})
        await update.message.reply_text(f"✅ Закрыто всех сделок: {count}")
        return

    if key in trades:
        close_trade(key)
        await update.message.reply_text(f"✅ Сделка {key} закрыта вручную")
    else:
        keys = list(trades.keys())
        if keys:
            await update.message.reply_text("❌ Сделка не найдена\nОткрытые: " + ", ".join(keys))
        else:
            await update.message.reply_text("📭 Нет открытых сделок")

async def callback_close(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    key    = query.data.replace("close_", "", 1)
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

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats  = load_stats()
    trades = stats.get("trades", [])

    if not trades:
        await update.message.reply_text(
            "📭 Нет записанных сделок\n"
            "Статистика копится при закрытии сделок по ТП/СЛ."
        )
        return

    hard_trades    = [t for t in trades if t.get('mode') == 'hard']
    scanner_trades = [t for t in trades if t.get('mode') == 'scanner']
    clean_trades   = [t for t in trades if t.get('mode') not in ('hard', 'scanner')]

    best     = max(trades, key=lambda x: x['pnl_pct'])
    worst    = min(trades, key=lambda x: x['pnl_pct'])
    sym_wins = Counter(t['symbol'] for t in trades if t['won'])
    top_sym  = sym_wins.most_common(3)

    recent_lines = []
    for t in reversed(trades[-5:]):
        mode_tag = f"[{t.get('mode', '?')}]"
        emoji    = "✅" if t['won'] else "❌"
        recent_lines.append(f"{emoji} {t['symbol']} {t['signal']} {t['pnl_pct']:+.2f}% {mode_tag}")

    first_date = trades[0]['time'][:10]

    text = (
        f"📈 <b>СТАТИСТИКА БОТА</b>\n"
        f"<i>С {first_date} | Всего: {len(trades)} сделок</i>\n\n"
        f"<b>🟡 MID/LOW ({len(clean_trades)} сделок):</b>\n"
        f"{_calc_stats_block(clean_trades)}\n\n"
        f"<b>🔴 HARD режим ({len(hard_trades)} сделок):</b>\n"
        f"{_calc_stats_block(hard_trades)}\n\n"
        f"<b>🔍 Сканер ({len(scanner_trades)} сделок):</b>\n"
        f"{_calc_stats_block(scanner_trades)}\n\n"
        f"<b>📊 Всего:</b>\n"
        f"{_calc_stats_block(trades)}\n\n"
        f"🏆 Лучшая: {best['symbol']} {best['pnl_pct']:+.2f}%\n"
        f"💀 Худшая: {worst['symbol']} {worst['pnl_pct']:+.2f}%\n\n"
        f"<b>Топ монеты:</b> " + " | ".join(f"{s}({c})" for s, c in top_sym) +
        f"\n\n<b>Последние 5:</b>\n" + "\n".join(recent_lines)
    )
    await update.message.reply_text(text[:4000], parse_mode='HTML')

async def cmd_dbstats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Статистика напрямую из PostgreSQL."""
    conn = _get_pg_conn()
    if not conn:
        await update.message.reply_text(
            "❌ PostgreSQL не подключён\nПроверь DATABASE_URL в Railway Variables"
        )
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*)                              as total,
                    COALESCE(SUM(CASE WHEN won THEN 1 ELSE 0 END), 0) as wins,
                    ROUND(COALESCE(AVG(pnl_pct),0)::numeric, 2)       as avg_pnl,
                    ROUND(COALESCE(MAX(pnl_pct),0)::numeric, 2)       as best,
                    ROUND(COALESCE(MIN(pnl_pct),0)::numeric, 2)       as worst,
                    COALESCE(MIN(created_at)::text, 'нет данных')      as since
                FROM bot_trades
            """)
            row = cur.fetchone()
            total, wins, avg_pnl, best, worst, since = row

            # Приводим к нужным типам (psycopg2 может вернуть Decimal)
            total    = int(total or 0)
            wins     = int(wins or 0)
            avg_pnl  = float(avg_pnl or 0)
            best     = float(best or 0)
            worst    = float(worst or 0)
            losses   = total - wins
            winrate  = round(wins / total * 100, 1) if total > 0 else 0.0
            since_s  = str(since)[:10] if since else "нет данных"

            cur.execute("""
                SELECT mode,
                    COUNT(*) as cnt,
                    COALESCE(SUM(CASE WHEN won THEN 1 ELSE 0 END), 0) as w,
                    ROUND(COALESCE(AVG(pnl_pct),0)::numeric, 2)       as avg
                FROM bot_trades
                GROUP BY mode
                ORDER BY cnt DESC
            """)
            modes = cur.fetchall()

        conn.close()

        if total == 0:
            await update.message.reply_text(
                "🗄️ <b>PostgreSQL подключён ✅</b>\n\n"
                "Таблица пустая — сделки ещё не записаны.\n"
                "Они появятся после закрытия первой сделки по ТП/СЛ.",
                parse_mode='HTML'
            )
            return

        wr_bar     = "🟩" * (int(winrate) // 20) + "⬜" * (5 - int(winrate) // 20)
        pf_raw     = sum(float(m[3] or 0) * int(m[1] or 0) for m in modes if float(m[3] or 0) > 0)
        modes_text = "\n".join(
            "  {}: {} сд, WR {}%, avg {:+.2f}%".format(
                m[0],
                int(m[1] or 0),
                round(int(m[2] or 0) / max(int(m[1] or 1), 1) * 100, 1),
                float(m[3] or 0)
            )
            for m in modes
        )

        text = (
            "🗄️ <b>POSTGRESQL СТАТИСТИКА</b>\n"
            + f"<i>С {since_s} | Всего: {total} сделок</i>\n\n"
            + f"Винрейт: {wr_bar} <b>{winrate}%</b>\n"
            + f"✅ Прибыльных: {wins} | ❌ Убыточных: {losses}\n"
            + f"Средний P&L: <b>{avg_pnl:+.2f}%</b>\n"
            + f"🏆 Лучшая: {best:+.2f}% | 💀 Худшая: {worst:+.2f}%\n\n"
            + f"<b>По режимам:</b>\n{modes_text}"
        )
        await update.message.reply_text(text, parse_mode='HTML')

    except Exception as e:
        try:
            conn.close()
        except:
            pass
        await update.message.reply_text(
            f"❌ Ошибка: {html_escape(str(e))}",
            parse_mode='HTML'
        )

async def handle_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if not text.startswith('/'):
        return
    symbol, tf, mode_key = parse_args(text)
    mode_cfg = TRADE_MODES[mode_key]
    chat_id  = update.effective_chat.id

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

        limit_data = None
        if result['signal'] in ("⚠️ WATCH", "НЕТ СИГНАЛА") or result.get('score', 0) < 55:
            limit_data = calculate_limit_orders(result)

        main_text = format_message(result, ai_text, limit_data=limit_data)

        TG_LIMIT = 4000
        ai_split = main_text.find("🧠 <b>AI:</b>")
        if ai_split > 0 and len(main_text) > TG_LIMIT:
            await msg.edit_text(main_text[:ai_split].rstrip()[:TG_LIMIT], parse_mode='HTML')
            await update.message.reply_text(main_text[ai_split:][:TG_LIMIT], parse_mode='HTML')
        else:
            await msg.edit_text(main_text[:TG_LIMIT], parse_mode='HTML')

        sl_tp    = result.get('sl_tp', {})
        risk_pct = sl_tp.get('risk_pct', 99)
        score    = result.get('score', 0)

        vyvod_match = re.search(r'РЕШЕНИЕ[: ]+(\S+)', ai_text, re.IGNORECASE)
        ai_verdict  = vyvod_match.group(1).lower().strip('.,!') if vyvod_match else ""
        ai_skip     = "пропустить" in ai_verdict
        is_hard     = mode_key == "hard"

        if result['signal'] in ("🟩 LONG", "🟥 SHORT") and sl_tp:
            if is_hard:
                await update.message.reply_text(
                    "ℹ️ <b>HARD режим</b> — сделка не добавляется в мониторинг и статистику",
                    parse_mode='HTML'
                )
            else:
                skip_reasons = []
                if score < 35:    skip_reasons.append(f"скоринг {score}/100 слишком низкий")
                if risk_pct > 15: skip_reasons.append(f"риск {risk_pct}% слишком высокий")
                if ai_skip:       skip_reasons.append("AI рекомендует пропустить")

                if skip_reasons:
                    await update.message.reply_text(
                        f"⚠️ <b>Сделка НЕ добавлена в мониторинг:</b>\n"
                        + "\n".join(f"• {r}" for r in skip_reasons),
                        parse_mode='HTML'
                    )
                else:
                    trade_key = open_trade(symbol, tf, result, chat_id, mode=mode_key)
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

async def _pump_scanner_job(context):
    await run_pump_scanner(context.application)

async def post_init(app):
    _ensure_pg_table()
    db_status = "PostgreSQL ✅" if DATABASE_URL and not DATABASE_URL.startswith("$") else "JSON only ⚠️ (DATABASE_URL не настроен)"
    logger.info(f"Stats backend: {db_status}")
    # Сразу логируем если DATABASE_URL выглядит как нераскрытый template
    if DATABASE_URL and DATABASE_URL.startswith("$"):
        logger.error(f"DATABASE_URL не раскрылся: {DATABASE_URL[:30]} — проверь Railway Variables!")
    app.job_queue.run_repeating(_check_trades_job, interval=300, first=15)
    app.job_queue.run_repeating(_scanner_job,      interval=60,  first=30)
    app.job_queue.run_repeating(_pump_scanner_job, interval=180, first=60)
    logger.info("JobQueue started: check_trades 5min, scanner 1min, pump_scanner 3min")



async def cmd_resetdb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args or args[0].lower() != "confirm":
        text = (
            "⚠️ <b>Сброс базы статистики</b>\n\n"
            "Удалит ВСЕ записанные сделки из базы.\n"
            "Открытые сделки НЕ затрагиваются.\n\n"
            "Для подтверждения:\n"
            "<code>/resetdb confirm</code>"
        )
        await update.message.reply_text(text, parse_mode='HTML')
        return

    deleted_pg   = 0
    deleted_json = 0

    conn = _get_pg_conn()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM bot_trades")
                deleted_pg = cur.fetchone()[0]
                cur.execute("DELETE FROM bot_trades")
            conn.commit()
            conn.close()
        except Exception as e:
            if conn:
                conn.close()
            await update.message.reply_text(
                "❌ Ошибка PostgreSQL: " + html_escape(str(e)),
                parse_mode='HTML'
            )
            return

    if STATS_FILE.exists():
        stats = load_stats()
        deleted_json = len(stats.get("trades", []))
        save_stats({"trades": []})

    state = load_scanner_state()
    clean_state = {k: v for k, v in state.items() if not k.startswith("sent_")}
    save_scanner_state(clean_state)

    result_text = (
        "✅ <b>База сброшена</b>\n\n"
        + f"PostgreSQL: удалено {deleted_pg} записей\n"
        + f"JSON fallback: удалено {deleted_json} записей\n"
        + "Антиспам кэш сканера: очищен\n\n"
        + "<i>Открытые сделки сохранены</i>"
    )
    await update.message.reply_text(result_text, parse_mode='HTML')


# ================== БЭКТЕСТ ==================
async def run_backtest(symbol: str, tf: str, days: int) -> dict:
    """
    Прогоняет скоринг по историческим свечам.
    Для каждой свечи симулирует сигнал и проверяет дошла ли цена до TP/SL
    в следующих N свечах.

    Возвращает словарь с полной статистикой и factor attribution.
    """
    ticker = symbol.replace("/", "")
    cfg    = TF_MAP.get(tf, TF_MAP["15m"])

    # Свечей на день: 15m=96, 1h=24, 4h=6, 1d=1
    candles_per_day = {"15m": 96, "1h": 24, "4h": 6, "1d": 1}
    n_candles       = days * candles_per_day.get(tf, 96)
    # +50 для индикаторов + look-forward window
    fetch_limit     = min(n_candles + 150, 1500)

    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        df_raw, source = await _fetch_klines(ticker, cfg[0], cfg[1], fetch_limit, session)

    if df_raw is None or len(df_raw) < 100:
        return {"error": f"Нет данных для {symbol} {tf}"}

    # Индикаторы на полном датасете
    df_raw = df_raw.copy()
    df_raw['rsi'] = ta.rsi(df_raw['close'], length=14)
    df_raw['atr'] = ta.atr(df_raw['high'], df_raw['low'], df_raw['close'], length=14)
    df_raw['ema20'] = ta.ema(df_raw['close'], length=20)
    df_raw['ema50'] = ta.ema(df_raw['close'], length=50)

    # Look-forward: сколько свечей проверяем после сигнала
    look_fwd = {"15m": 16, "1h": 12, "4h": 8, "1d": 5}.get(tf, 16)

    mode_cfg = TRADE_MODES["mid"]
    trades   = []

    # Стартуем с индекса 60 (нужны данные для VP и индикаторов)
    start_idx = 60
    end_idx   = len(df_raw) - look_fwd - 1  # оставляем look_fwd свечей для проверки

    for i in range(start_idx, end_idx):
        # Данные до текущей свечи (закрытые)
        df_slice = df_raw.iloc[:i+1].copy()
        row      = df_slice.iloc[-1]

        rsi   = row['rsi']
        atr   = row['atr']
        ema20 = row['ema20']
        ema50 = row['ema50']
        price = float(row['close'])

        if pd.isna(rsi) or pd.isna(atr) or pd.isna(ema20) or pd.isna(ema50):
            continue

        atr_pct = round(atr / price * 100, 2)

        # Volume Profile на последних 100 свечах
        vp_slice        = df_slice.tail(100)
        centers, vp     = calculate_volume_profile(vp_slice, num_bins=60)
        vp_mean         = float(np.mean(vp))
        local_nodes     = find_hvn(vp, centers, price, dist_limit=20)
        supports, resis = find_sr_levels(df_slice, price)

        strong_above = [n for n in local_nodes if n['is_above'] and n['distance_pct'] < 12]
        top_hvn      = strong_above[0] if strong_above else None

        # Тренд TF
        if price > ema20 > ema50:
            trend_l = "UPTREND"
        elif price < ema20 < ema50:
            trend_l = "DOWNTREND"
        else:
            trend_l = "SIDEWAYS"

        # HTF тренд — используем простую EMA на срезе (нет отдельного запроса в бэктесте)
        # Берём каждую 4-ю свечу как приближение HTF
        htf_slice = df_raw.iloc[:i+1:4].copy() if tf == "15m" else df_raw.iloc[:i+1:2].copy()
        if len(htf_slice) >= 50:
            htf_e20 = ta.ema(htf_slice['close'], length=20).iloc[-1]
            htf_e50 = ta.ema(htf_slice['close'], length=50).iloc[-1]
            htf_p   = float(htf_slice['close'].iloc[-1])
            if htf_p > htf_e20 > htf_e50:
                trend_h = "UPTREND"
            elif htf_p < htf_e20 < htf_e50:
                trend_h = "DOWNTREND"
            else:
                trend_h = "SIDEWAYS"
        else:
            trend_h = trend_l

        # RSI дивергенция
        rsi_div  = detect_rsi_divergence(df_slice)
        # Пробой HVN
        hvn_brk  = detect_hvn_breakout(df_slice, local_nodes, price)
        # Свеча
        candle   = detect_candle_pattern(df_slice)
        # Дельта
        delta    = calculate_delta(df_slice)
        # Режим рынка
        regime   = detect_market_regime(df_slice, atr)

        signal, reason, score, detail = compute_score_and_signal(
            rsi=round(rsi, 1), price=price, atr=atr,
            ema20=ema20, ema50=ema50,
            top_hvn=top_hvn, vp_mean=vp_mean,
            delta_str=delta, trend_l=trend_l, trend_h=trend_h,
            candle=candle, rsi_div=rsi_div, hvn_break=hvn_brk,
            regime=regime, mode_cfg=mode_cfg,
            atr_pct=atr_pct, tf=tf
        )

        if signal not in ("🟩 LONG", "🟥 SHORT"):
            continue
        if score < 65:  # в бэктесте чуть мягче чтобы набрать статистику
            continue

        # SL/TP
        sl_tp = calculate_sl_tp(signal, price, atr, local_nodes, supports, resis, tf)
        if not sl_tp:
            continue

        sl, tp1, tp2, tp3 = sl_tp['sl'], sl_tp['tp1'], sl_tp['tp2'], sl_tp['tp3']
        is_long = "LONG" in signal

        # Симуляция на look_fwd следующих свечах
        future = df_raw.iloc[i+1 : i+1+look_fwd]
        result = "open"  # не достигнуто ни SL ни TP за окно

        tp1_hit = tp2_hit = tp3_hit = False
        exit_price = None
        exit_candle_idx = None

        for j, (_, fc) in enumerate(future.iterrows()):
            fh, fl = float(fc['high']), float(fc['low'])

            # Проверяем TP каскадно
            if not tp1_hit:
                if (is_long and fh >= tp1) or (not is_long and fl <= tp1):
                    tp1_hit = True

            if tp1_hit and not tp2_hit:
                if (is_long and fh >= tp2) or (not is_long and fl <= tp2):
                    tp2_hit = True

            if tp2_hit and not tp3_hit:
                if (is_long and fh >= tp3) or (not is_long and fl <= tp3):
                    tp3_hit  = True
                    result   = "tp3"
                    exit_price = tp3
                    exit_candle_idx = j
                    break

            # SL — только после установки текущего SL
            cur_sl = sl
            if tp1_hit:
                cur_sl = price  # БУ
            if tp2_hit:
                cur_sl = tp1

            sl_hit = (is_long and fl <= cur_sl) or (not is_long and fh >= cur_sl)
            if sl_hit and not tp3_hit:
                if tp2_hit:
                    result = "tp2"
                    exit_price = tp2
                elif tp1_hit:
                    result = "be"
                    exit_price = price
                else:
                    result = "sl"
                    exit_price = cur_sl
                exit_candle_idx = j
                break

        if result == "open":
            if tp2_hit:
                result, exit_price = "tp2", tp2
            elif tp1_hit:
                result, exit_price = "tp1_partial", tp1
            else:
                result, exit_price = "expired", price  # не дошло никуда

        # P&L
        if exit_price is not None:
            pnl = (exit_price - price) / price * 100 * (1 if is_long else -1)
        else:
            pnl = 0.0

        won = pnl > 0

        # Factor attribution — ключевые метаданные
        htf_aligned     = trend_l == trend_h and trend_l != "SIDEWAYS"
        has_divergence  = bool(rsi_div)
        has_hvn_break   = bool(hvn_brk)
        market_regime   = regime.get('regime', 'mixed')
        coin_vol        = classify_coin_volatility(atr_pct, tf)
        score_bucket    = f"{(score // 10) * 10}-{(score // 10) * 10 + 9}"

        trades.append({
            "idx":           i,
            "timestamp":     str(row['timestamp']),
            "signal":        signal,
            "score":         score,
            "score_bucket":  score_bucket,
            "reason":        reason,
            "result":        result,
            "pnl":           round(pnl, 2),
            "won":           won,
            "rsi":           round(rsi, 1),
            "atr_pct":       atr_pct,
            "exit_candle":   exit_candle_idx,
            # Factor attribution
            "htf_aligned":   htf_aligned,
            "has_divergence": has_divergence,
            "has_hvn_break": has_hvn_break,
            "regime":        market_regime,
            "coin_vol":      coin_vol,
            "trend_l":       trend_l,
            "trend_h":       trend_h,
        })

    return {"symbol": symbol, "tf": tf, "days": days, "trades": trades, "source": source}


def _bt_factor_stats(trades: list, key: str, val) -> str:
    """Статистика по конкретному фактору."""
    subset = [t for t in trades if t.get(key) == val]
    if len(subset) < 3:
        return ""
    wins = sum(1 for t in subset if t['won'])
    wr   = round(wins / len(subset) * 100, 1)
    avg  = round(sum(t['pnl'] for t in subset) / len(subset), 2)
    return f"{val}: {len(subset)} сд, WR {wr}%, avg {avg:+.2f}%"


async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /backtest BTCUSDT 15m 30
    /backtest ETHUSDT 1h 14
    """
    args = context.args
    if not args:
        await update.message.reply_text(
            "\U0001f4ca <b>Бэктест</b>\n\n"
            "Использование:\n"
            "<code>/backtest BTCUSDT 15m 30</code>\n"
            "<code>/backtest ETHUSDT 1h 14</code>\n"
            "<code>/backtest SOLUSDT 4h 60</code>\n\n"
            "Прогоняет скоринг по историческим данным\n"
            "и показывает реальную эффективность каждого фактора.",
            parse_mode='HTML'
        )
        return

    # Парсинг аргументов
    raw_sym = args[0].upper().replace("USDT", "").replace("/", "")
    symbol  = f"{raw_sym}/USDT"
    tf      = args[1].lower() if len(args) > 1 and args[1].lower() in TF_MAP else "15m"
    days    = int(args[2]) if len(args) > 2 and args[2].isdigit() else 14
    days    = min(days, 60)

    msg = await update.message.reply_text(
        f"\u23f3 Запускаю бэктест {symbol} {tf} за {days} дней...\n"
        f"Это займёт 20–60 секунд.",
        parse_mode='HTML'
    )

    try:
        bt = await run_backtest(symbol, tf, days)
    except Exception as e:
        await msg.edit_text(f"\u274c Ошибка бэктеста: {html_escape(str(e))}", parse_mode='HTML')
        return

    if "error" in bt:
        await msg.edit_text(f"\u274c {bt['error']}")
        return

    trades = bt['trades']
    if not trades:
        await msg.edit_text(
            f"\U0001f914 За {days} дней не нашлось сигналов с порогом 65+\n"
            f"Попробуй другой инструмент или таймфрейм."
        )
        return

    # ── Общая статистика ──────────────────────────────────────────────────
    total  = len(trades)
    wins   = sum(1 for t in trades if t['won'])
    losses = total - wins
    wr     = round(wins / total * 100, 1)

    pnls        = [t['pnl'] for t in trades]
    avg_win     = round(sum(p for p in pnls if p > 0) / max(wins,   1), 2)
    avg_loss    = round(sum(p for p in pnls if p < 0) / max(losses, 1), 2)
    total_pnl   = round(sum(pnls), 2)
    gross_p     = sum(p for p in pnls if p > 0)
    gross_l     = abs(sum(p for p in pnls if p < 0))
    pf          = round(gross_p / max(gross_l, 0.001), 2)
    expectancy  = round(wr/100 * avg_win + (1 - wr/100) * avg_loss, 2)
    max_dd_seq  = 0
    cur_dd      = 0
    for t in trades:
        cur_dd = cur_dd + 1 if not t['won'] else 0
        max_dd_seq = max(max_dd_seq, cur_dd)

    wr_bar  = "\U0001f7e9" * (int(wr) // 20) + "\u2b1c" * (5 - int(wr) // 20)
    pf_e    = "\u2705" if pf >= 1.5 else ("\u26a0\ufe0f" if pf >= 1.0 else "\u274c")
    exp_e   = "\u2705" if expectancy > 0 else "\u274c"

    # Результаты по типу
    by_result = {}
    for t in trades:
        by_result[t['result']] = by_result.get(t['result'], 0) + 1
    result_str = " | ".join(f"{k}:{v}" for k, v in sorted(by_result.items()))

    # ── Factor Attribution ────────────────────────────────────────────────
    # HTF alignment
    htf_yes = [t for t in trades if t['htf_aligned']]
    htf_no  = [t for t in trades if not t['htf_aligned']]
    htf_wr_yes = round(sum(1 for t in htf_yes if t['won']) / max(len(htf_yes), 1) * 100, 1)
    htf_wr_no  = round(sum(1 for t in htf_no  if t['won']) / max(len(htf_no),  1) * 100, 1)

    # Дивергенция
    div_yes = [t for t in trades if t['has_divergence']]
    div_no  = [t for t in trades if not t['has_divergence']]
    div_wr_yes = round(sum(1 for t in div_yes if t['won']) / max(len(div_yes), 1) * 100, 1)
    div_wr_no  = round(sum(1 for t in div_no  if t['won']) / max(len(div_no),  1) * 100, 1)

    # HVN пробой
    brk_yes = [t for t in trades if t['has_hvn_break']]
    brk_no  = [t for t in trades if not t['has_hvn_break']]
    brk_wr_yes = round(sum(1 for t in brk_yes if t['won']) / max(len(brk_yes), 1) * 100, 1)
    brk_wr_no  = round(sum(1 for t in brk_no  if t['won']) / max(len(brk_no),  1) * 100, 1)

    # Режим рынка
    regimes = set(t['regime'] for t in trades)
    regime_lines = []
    for reg in regimes:
        rt = [t for t in trades if t['regime'] == reg]
        rwr = round(sum(1 for t in rt if t['won']) / len(rt) * 100, 1)
        ravg = round(sum(t['pnl'] for t in rt) / len(rt), 2)
        regime_lines.append(f"  {reg}: {len(rt)} сд, WR {rwr}%, avg {ravg:+.2f}%")

    # По бакету скора
    buckets = sorted(set(t['score_bucket'] for t in trades))
    bucket_lines = []
    for b in buckets:
        bt2 = [t for t in trades if t['score_bucket'] == b]
        bwr  = round(sum(1 for t in bt2 if t['won']) / len(bt2) * 100, 1)
        bavg = round(sum(t['pnl'] for t in bt2) / len(bt2), 2)
        bucket_lines.append(f"  {b}: {len(bt2)} сд, WR {bwr}%, avg {bavg:+.2f}%")

    # Лучший / худший сигнал
    best  = max(trades, key=lambda t: t['pnl'])
    worst = min(trades, key=lambda t: t['pnl'])

    # ── Формируем сообщение ───────────────────────────────────────────────
    text = (
        f"\U0001f4ca <b>БЭКТЕСТ: {symbol} {tf} | {days} дней</b>\n"
        f"<i>Источник: {bt.get('source','?')} | Сигналов: {total}</i>\n\n"

        f"<b>\U0001f4c8 ОБЩИЙ РЕЗУЛЬТАТ:</b>\n"
        f"Винрейт: {wr_bar} <b>{wr}%</b> ({wins}W / {losses}L)\n"
        f"Avg win: +{avg_win}% | Avg loss: {avg_loss}%\n"
        f"Профит-фактор: {pf_e} <b>{pf}</b>\n"
        f"Expectancy: {exp_e} <b>{expectancy:+.2f}%</b> за сделку\n"
        f"Суммарный PnL: <b>{total_pnl:+.2f}%</b>\n"
        f"Макс серия SL подряд: <b>{max_dd_seq}</b>\n"
        f"Исходы: {result_str}\n\n"

        f"<b>\U0001f9ec FACTOR ATTRIBUTION:</b>\n"
        f"HTF aligned:\n"
        f"  \u2705 да ({len(htf_yes)} сд): WR {htf_wr_yes}%\n"
        f"  \u274c нет ({len(htf_no)} сд): WR {htf_wr_no}%\n\n"
        f"RSI дивергенция:\n"
        f"  \u2705 есть ({len(div_yes)} сд): WR {div_wr_yes}%\n"
        f"  \u274c нет ({len(div_no)} сд): WR {div_wr_no}%\n\n"
        f"Пробой HVN:\n"
        f"  \u2705 есть ({len(brk_yes)} сд): WR {brk_wr_yes}%\n"
        f"  \u274c нет ({len(brk_no)} сд): WR {brk_wr_no}%\n\n"

        f"<b>Режим рынка:</b>\n" + "\n".join(regime_lines) + "\n\n"
        f"<b>По скору:</b>\n" + "\n".join(bucket_lines) + "\n\n"

        f"<b>\U0001f3c6 Лучший:</b> {best['timestamp'][:10]} "
        f"{best['signal']} {best['pnl']:+.2f}% [{best['reason'][:40]}]\n"
        f"<b>\U0001f480 Худший:</b> {worst['timestamp'][:10]} "
        f"{worst['signal']} {worst['pnl']:+.2f}% [{worst['reason'][:40]}]\n\n"
    )

    # Вывод рекомендаций
    recs = []
    if htf_wr_yes > htf_wr_no + 15:
        recs.append("\u2714\ufe0f HTF alignment сильно повышает WR — оставь как жёсткий фильтр")
    if div_wr_yes > div_wr_no + 10:
        recs.append("\u2714\ufe0f Дивергенция реально работает — можно повысить её вес в скоринге")
    elif div_wr_yes < div_wr_no:
        recs.append("\u26a0\ufe0f Дивергенция не улучшает WR — возможно её вес стоит снизить")
    if brk_wr_yes > brk_wr_no + 10:
        recs.append("\u2714\ufe0f Пробой HVN — сильный сигнал, можно повысить вес")
    if max_dd_seq >= 5:
        recs.append(f"\u26a0\ufe0f Серия {max_dd_seq} SL подряд — kill switch нужен")
    if expectancy < 0:
        recs.append("\u274c Отрицательный expectancy — скоринг нужно пересматривать")
    elif expectancy > 0.5:
        recs.append(f"\u2705 Expectancy {expectancy:+.2f}% — есть реальный edge")

    if recs:
        text += "<b>\U0001f4a1 Выводы:</b>\n" + "\n".join(recs)

    # Telegram лимит — шлём двумя сообщениями если длинно
    TG_LIMIT = 4000
    if len(text) > TG_LIMIT:
        await msg.edit_text(text[:TG_LIMIT], parse_mode='HTML')
        await update.message.reply_text(text[TG_LIMIT:TG_LIMIT*2], parse_mode='HTML')
    else:
        await msg.edit_text(text, parse_mode='HTML')


def main():
    app = (Application.builder()
           .token(TELEGRAM_TOKEN)
           .post_init(post_init)
           .build())

    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("pump",    cmd_pump))
    app.add_handler(CommandHandler("scan",    cmd_scan))
    app.add_handler(CommandHandler("trades",  cmd_trades))
    app.add_handler(CommandHandler("mexc",    cmd_mexc))
    app.add_handler(CommandHandler("close",   cmd_close))
    app.add_handler(CommandHandler("stats",   cmd_stats))
    app.add_handler(CommandHandler("dbstats", cmd_dbstats))
    app.add_handler(CommandHandler("resetdb",   cmd_resetdb))
    app.add_handler(CommandHandler("backtest",  cmd_backtest))
    app.add_handler(CallbackQueryHandler(callback_close, pattern="^close_"))
    app.add_handler(MessageHandler(filters.COMMAND, handle_command))

    print("🚀 Signal Volume Bot v4 запущен")
    app.run_polling()

if __name__ == '__main__':
    main()
