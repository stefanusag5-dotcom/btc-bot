import logging
import asyncio
import aiohttp
import json
import os
import re
from html import escape as html_escape
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import pandas_ta as ta
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
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

async def fetch_binance_futures_symbols() -> list:
    """Топ символы с Binance Futures по объёму"""
    try:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get("https://fapi.binance.com/fapi/v1/ticker/24hr") as r:
                if r.status == 200:
                    data = await r.json()
                    # Только USDT пары, сортируем по объёму
                    pairs = [d for d in data if d['symbol'].endswith('USDT')]
                    pairs.sort(key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)
                    return [p['symbol'].replace('USDT', '/USDT') for p in pairs[:80]]
    except Exception as e:
        logger.error(f"fetch_symbols: {e}")
    return []

async def fetch_higher_tf(symbol, tf):
    higher = {"15m": "1h", "1h": "4h", "4h": "1d", "1d": "1d"}
    htf = higher.get(tf, "1h")
    df, _, _, _ = await fetch_ohlcv(symbol, htf)
    return df, htf

async def fetch_daily_vp(symbol):
    ticker = symbol.replace("/", "")
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        df, _ = await _fetch_klines(ticker, "1d", "D", 1000, session)
    return df

# ================== VOLUME PROFILE ==================
def calculate_volume_profile(df, num_bins=120):
    price_min, price_max = df['low'].min(), df['high'].max()
    if price_min == price_max:
        return np.array([price_min]), np.array([df['volume'].sum()])
    bins = np.linspace(price_min, price_max, num_bins + 1)
    vp = np.zeros(num_bins)
    for _, row in df.iterrows():
        lo = max(0, np.searchsorted(bins, row['low']) - 1)
        hi = min(num_bins - 1, np.searchsorted(bins, row['high']) - 1)
        if lo == hi: vp[lo] += row['volume']
        else: vp[lo:hi+1] += row['volume'] / (hi - lo + 1)
    return (bins[:-1] + bins[1:]) / 2, vp

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

# ================== RSI ДИВЕРГЕНЦИЯ ==================
def detect_rsi_divergence(df: pd.DataFrame) -> str:
    """
    Ищет дивергенцию RSI за последние 30 свечей.
    Бычья: цена делает новый лоу, RSI нет -> разворот вверх.
    Медвежья: цена делает новый хай, RSI нет -> разворот вниз.
    """
    if len(df) < 30 or 'rsi' not in df.columns:
        return ""
    recent = df.tail(30).copy()
    prices = recent['close'].values
    rsi_vals = recent['rsi'].values

    # Убираем NaN
    valid = ~(np.isnan(prices) | np.isnan(rsi_vals))
    if valid.sum() < 20:
        return ""
    prices = prices[valid]
    rsi_vals = rsi_vals[valid]

    # Ищем последние два значимых лоу (для бычьей дивергенции)
    # Простая версия: сравниваем первую и вторую половины
    mid = len(prices) // 2
    p1_low, p2_low = prices[:mid].min(), prices[mid:].min()
    r1_low = rsi_vals[prices[:mid].argmin()]
    r2_low = rsi_vals[mid + prices[mid:].argmin()]

    p1_high, p2_high = prices[:mid].max(), prices[mid:].max()
    r1_high = rsi_vals[prices[:mid].argmax()]
    r2_high = rsi_vals[mid + prices[mid:].argmax()]

    # Бычья дивергенция: цена ниже, RSI выше
    if p2_low < p1_low * 0.998 and r2_low > r1_low + 3:
        return "🔄 Бычья дивергенция RSI (цена↓ RSI↑)"

    # Медвежья дивергенция: цена выше, RSI ниже
    if p2_high > p1_high * 1.002 and r2_high < r1_high - 3:
        return "🔄 Медвежья дивергенция RSI (цена↑ RSI↓)"

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

def calculate_sl_tp(signal, price, atr, hv_nodes, supports=None, resistances=None):
    if signal not in ("🟩 LONG", "🟥 SHORT"): return {}
    is_long = signal == "🟩 LONG"
    below = [n for n in hv_nodes if not n['is_above']]
    above = [n for n in hv_nodes if n['is_above']]
    hvn_above_p = [n['price'] for n in above]
    hvn_below_p = [n['price'] for n in below]
    res = list(resistances or [])
    sup = list(supports or [])

    if is_long:
        sl = (min(below, key=lambda x: x['distance_pct'])['price'] - atr*0.3 if below else price - atr*1.5)
        sl = min(sl, price - atr*1.2)
        near_sup = [s for s in sup if s < price and s > sl - atr]
        if near_sup: sl = min(sl, min(near_sup) - atr*0.2)
        sl = round(sl, 6)
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
        sl = round(sl, 6)
        risk = sl - price
        tp1 = _snap_to_level(min(price - risk*1.5, price - atr*0.5), hvn_below_p + sup)
        tp2 = round(min(_snap_to_level(price - risk*2.5, hvn_below_p + sup, 1.0), tp1 - atr*0.5), 6)
        far = [n for n in below if n['price'] < tp2]
        tp3 = round(min(
            min(far, key=lambda x: x['strength'])['price'] if far else price - risk*4.0,
            price - risk*3.5, tp2 - atr), 6)

    rr = round(abs(tp2 - price) / max(abs(price - sl), 0.0001), 2)
    return {
        "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "risk_pct": round(abs(price - sl) / price * 100, 2),
        "rr_ratio": rr,
        "rr_warn": "⚠️ R/R ниже 1.5 — сделка под вопросом" if rr < 1.5 else "",
    }

# ================== СКОРИНГ ==================
def score_signal(rsi, signal, top_hvn, vp_mean, delta_str, trend_l, trend_h, candle, atr_pct):
    score = 0
    if signal in ("🟩 LONG", "🟥 SHORT"): score += 20
    if top_hvn and top_hvn['strength'] > 2.0*vp_mean: score += 25
    elif top_hvn: score += 10
    try:
        buy_pct = float(delta_str.split('%')[0].split()[-1])
    except: buy_pct = 50
    if signal == "🟩 LONG":
        if rsi < 30: score += 20
        elif rsi < 38: score += 10
        if trend_l == "UPTREND": score += 10
        if trend_h == "UPTREND": score += 15
        if buy_pct > 60: score += 10
    elif signal == "🟥 SHORT":
        if rsi > 70: score += 20
        elif rsi > 62: score += 10
        if trend_l == "DOWNTREND": score += 10
        if trend_h == "DOWNTREND": score += 15
        if buy_pct < 40: score += 10
    if "поглощение" in candle or "пин-бар" in candle: score += 10
    return min(score, 100)

# ================== АНАЛИЗ ==================
async def analyze_symbol(symbol, tf="15m", mode_cfg=None):
    if mode_cfg is None: mode_cfg = TRADE_MODES["mid"]
    main_task   = asyncio.create_task(fetch_ohlcv(symbol, tf))
    higher_task = asyncio.create_task(fetch_higher_tf(symbol, tf))
    daily_task  = asyncio.create_task(fetch_daily_vp(symbol))
    dom_task    = asyncio.create_task(fetch_btc_dominance())
    news_task   = asyncio.create_task(fetch_crypto_news(symbol))
    (df, source, fr, oi), (df_htf, htf_label), df_daily, btc_dom, news = await asyncio.gather(
        main_task, higher_task, daily_task, dom_task, news_task)
    if df is None or len(df) < 100: return None

    price = df['close'].iloc[-1]
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    rsi  = round(df['rsi'].iloc[-1], 1)
    atr  = round(df['atr'].iloc[-1], 6)
    atr_pct = round(atr / price * 100, 2)
    ema20 = round(ta.ema(df['close'], length=20).iloc[-1], 6)
    ema50 = round(ta.ema(df['close'], length=50).iloc[-1], 6)

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

    rsi_long  = mode_cfg["rsi_long"]
    rsi_short = mode_cfg["rsi_short"]
    hvn_mult  = mode_cfg["hvn_mult"]
    strong_above = [n for n in hvn_above if n['distance_pct'] < 12]
    top_hvn = strong_above[0] if strong_above else None
    vp_mean = float(np.mean(vp_l))

    is_hard = mode_cfg.get("label", "") == "🔴 HARD"

    signal, reason = "НЕТ СИГНАЛА", "Нейтральная структура"
    if top_hvn and top_hvn['strength'] > hvn_mult*vp_mean:
        if rsi > rsi_short:   signal, reason = "🟥 SHORT", f"Полка сверху {top_hvn['price']} + RSI {rsi}"
        elif rsi < rsi_long:  signal, reason = "🟩 LONG",  f"Полка сверху {top_hvn['price']} + RSI {rsi}"
        else:
            if is_hard:
                # В HARD режиме WATCH -> сигнал по тренду EMA
                if trend_l == "UPTREND":
                    signal, reason = "🟩 LONG",  f"HARD: полка {top_hvn['price']} + EMA аптренд"
                elif trend_l == "DOWNTREND":
                    signal, reason = "🟥 SHORT", f"HARD: полка {top_hvn['price']} + EMA даунтренд"
                else:
                    # Боковик — смотрим на дельту
                    try:
                        buy_pct = float(delta.split('%')[0].split()[-1])
                    except: buy_pct = 50
                    if buy_pct > 52:
                        signal, reason = "🟩 LONG",  f"HARD: полка {top_hvn['price']} + дельта бычья"
                    else:
                        signal, reason = "🟥 SHORT", f"HARD: полка {top_hvn['price']} + дельта медвежья"
            else:
                signal, reason = "⚠️ WATCH", f"Полка сверху {top_hvn['price']}, RSI нейтрален"
    elif rsi < rsi_long:  signal, reason = "🟩 LONG",  f"Перепроданность RSI ({rsi})"
    elif rsi > rsi_short: signal, reason = "🟥 SHORT", f"Перекупленность RSI ({rsi})"

    # В HARD режиме — если всё ещё нет сигнала, даём по тренду EMA
    if is_hard and signal == "НЕТ СИГНАЛА":
        if trend_l == "UPTREND":
            signal, reason = "🟩 LONG",  f"HARD: EMA аптренд, RSI {rsi}"
        elif trend_l == "DOWNTREND":
            signal, reason = "🟥 SHORT", f"HARD: EMA даунтренд, RSI {rsi}"
        else:
            try:
                buy_pct = float(delta.split('%')[0].split()[-1])
            except: buy_pct = 50
            if buy_pct > 52:
                signal, reason = "🟩 LONG",  f"HARD: дельта бычья ({buy_pct:.0f}% покупок)"
            else:
                signal, reason = "🟥 SHORT", f"HARD: дельта медвежья ({buy_pct:.0f}% покупок)"

    htf_conflict = ""
    if signal == "🟩 LONG"  and trend_h == "DOWNTREND": htf_conflict = f"⚠️ LONG против тренда {htf_label}!"
    if signal == "🟥 SHORT" and trend_h == "UPTREND":   htf_conflict = f"⚠️ SHORT против тренда {htf_label}!"

    sl_tp = calculate_sl_tp(signal, price, atr, all_nodes, supports, resistances)
    score = score_signal(rsi, signal, top_hvn, vp_mean, delta, trend_l, trend_h, candle, atr_pct)

    return {
        "symbol": symbol, "tf": tf, "price": round(price, 6),
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
        "time": datetime.now().strftime("%H:%M"),
        "btc_trend_text": "",
        "mode_label": mode_cfg["label"],
        "mode_personality": mode_cfg["personality"],
    }

# ================== GROQ AI ==================
async def ask_ai(data: dict) -> str:
    if not groq_client: return "AI отключён (нет GROQ_API_KEY)"
    sl_tp = data.get("sl_tp", {})
    tf = data['tf']
    tf_ctx = {"15m":"Скальп/интрадей, часы.","1h":"Интрадей, 1-2 дня.","4h":"Свинг, дни.","1d":"Позиция, недели."}.get(tf,"Интрадей.")
    hvn_a = ", ".join(f"{n['price']}({'D' if n.get('type')=='daily' else 'L'})" for n in data['hvn_above'][:3])
    hvn_b = ", ".join(f"{n['price']}({'D' if n.get('type')=='daily' else 'L'})" for n in data['hvn_below'][:3])
    conflict = f"\nКОНФЛИКТ: {data['htf_conflict']}" if data.get('htf_conflict') else ""

    system = f"""{data['mode_personality']} {tf_ctx}
ПРАВИЛА: отвечай ТОЛЬКО на основе переданных цифр. ЗАПРЕЩЕНО: новости, геополитика, макро, ФРС. ТП/СЛ уже рассчитаны — не меняй."""

    user = f"""Данные {data['symbol']} / {tf}:
Цена={data['price']} Сигнал={data['signal']} Скоринг={data['score']}/100
RSI={data['rsi']} ATR={data['atr_pct']}% Свеча={data['candle_pattern']}
Тренд {tf}:{data['trend_local']} HTF:{data['trend_higher']}{conflict}
Дельта={data['delta']} Фандинг={data.get('funding_rate','N/A')}% BTC={data.get('btc_trend_text','N/A')}
{data.get('btc_dominance','')}
RSI дивергенция: {data.get('rsi_divergence','нет')}
Пробой HVN: {data.get('hvn_breakout','нет')}
OI: {data.get('oi_trend','')}
POC={data['poc']} HVN↑={hvn_a} HVN↓={hvn_b}
СЛ={sl_tp.get('sl','?')} ТП1={sl_tp.get('tp1','?')} ТП2={sl_tp.get('tp2','?')} ТП3={sl_tp.get('tp3','?')} R/R=1:{sl_tp.get('rr_ratio','?')}
{data.get('news','')}

Ответ СТРОГО (без повтора данных):
✅ ВЫВОД: войти / пропустить / ждать
💪 СИЛА: слабый / средний / сильный
⚠️ РИСК: [одно предложение — только технический]
💡 СОВЕТ: [одно действие с учётом {tf}]"""

    for model in GROQ_MODELS:
        for attempt in range(2):
            try:
                r = await asyncio.get_event_loop().run_in_executor(None,
                    lambda m=model: groq_client.chat.completions.create(
                        model=m,
                        messages=[{"role":"system","content":system},{"role":"user","content":user}],
                        max_tokens=250, temperature=0.3))
                logger.info(f"Groq OK: {model}")
                return r.choices[0].message.content.strip()
            except Exception as ex:
                err = str(ex)
                logger.error(f"Groq {model} attempt {attempt+1}: {err}")
                if "404" in err or "not found" in err.lower() or "decommissioned" in err.lower(): break
                if "429" in err or "rate" in err.lower():
                    m2 = re.search(r'retryDelay[^0-9]+(\d+)', err)
                    await asyncio.sleep(min(int(m2.group(1))+2 if m2 else 12, 30))
                elif attempt == 0: await asyncio.sleep(3)
    return "⏳ AI: лимит запросов"

# ================== ФОРМАТИРОВАНИЕ ==================
def format_message(result, ai_text, is_scanner=False):
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
    news_txt  = f"\n\n{e(result['news'])}"          if result.get('news') else ""

    def fmt_nodes(nodes):
        out = [f"{'🌍' if n.get('type')=='daily' else '📍'}{n['price']}" for n in nodes[:3]]
        return " ".join(out) if out else "—"

    header = "🔔 <b>АВТОСИГНАЛ</b>\n" if is_scanner else ""
    return (
        f"{header}📊 <b>{e(result['symbol'])}</b> {result['tf']} • {result['time']} • {e(result['mode_label'])}\n"
        f"<i>{e(result['source'])}</i>\n\n"
        f"Цена: <b>{p}</b>\n"
        f"Сигнал: <b>{e(result['signal'])}</b>\n"
        f"Причина: {e(result['reason'])}\n"
        f"Скоринг: {score_bar} <b>{score}/100</b>\n\n"
        f"RSI: {result['rsi']} | ATR: {result['atr']} ({result['atr_pct']}%)\n"
        f"EMA: {e(result['ema_trend'])} | Свеча: {e(result['candle_pattern'])}\n"
        f"Дельта: {e(result['delta'])}"
        f"{fr_str}{oi_str}\n\n"
        f"📈 Тренд {result['tf']}: {e(result['trend_local'])}\n"
        f"📊 Тренд HTF: {e(result['trend_higher'])}"
        f"{conflict}{rsi_div}{hvn_break}{oi_trend}{btc_dom}\n\n"
        f"POC: {result['poc']}\n"
        f"HVN↑: {fmt_nodes(result['hvn_above'])}\n"
        f"HVN↓: {fmt_nodes(result['hvn_below'])}\n"
        f"Сопр: {result['resistances'][:2]} | Подд: {result['supports'][:2]}\n"
        f"{trade_block}\n"
        f"{news_txt}\n\n"
        f"🧠 <b>AI:</b>\n{e(ai_text)}"
    )

# ================== МОНИТОРИНГ СДЕЛОК ==================
async def check_trades(app):
    while True:
        await asyncio.sleep(300)
        trades = load_trades()
        if not trades: continue
        for key, t in list(trades.items()):
            try:
                df, _, _, _ = await fetch_ohlcv(t['symbol'], t['tf'])
                if df is None: continue
                price = float(df['close'].iloc[-1])
                is_long = "LONG" in t['signal']
                msgs = []

                if not t['tp1_hit']:
                    if (is_long and price >= t['tp1']) or (not is_long and price <= t['tp1']):
                        t['tp1_hit'] = True
                        t['sl'] = t['entry']
                        t['sl_moved_be'] = True
                        msgs.append(f"🎯 <b>ТП1 достигнут!</b> {t['symbol']} {t['tf']}\nСЛ → безубыток <b>{t['entry']}</b>\nЖдём ТП2: {t['tp2']}")
                elif not t['tp2_hit']:
                    if (is_long and price >= t['tp2']) or (not is_long and price <= t['tp2']):
                        t['tp2_hit'] = True
                        t['sl'] = t['tp1']
                        t['sl_moved_tp1'] = True
                        record_trade_result(t['symbol'],t['tf'],t['signal'],t['entry'],t['sl'],t['tp1'],t['tp2'],t['tp3'],t['tp2'],"tp2",0)
                        msgs.append(f"🎯🎯 <b>ТП2 достигнут!</b> {t['symbol']} {t['tf']}\nСЛ → ТП1 <b>{t['tp1']}</b>\nФинал ТП3: {t['tp3']}")
                else:
                    if (is_long and price >= t['tp3']) or (not is_long and price <= t['tp3']):
                        msgs.append(f"🏆 <b>ТП3 достигнут!</b> {t['symbol']} {t['tf']}\nСделка полностью закрыта!")
                        record_trade_result(t['symbol'],t['tf'],t['signal'],t['entry'],t['sl'],t['tp1'],t['tp2'],t['tp3'],t['tp3'],"tp3",0)
                        close_trade(key); trades.pop(key, None)

                if key in trades:
                    if (is_long and price <= t['sl']) or (not is_long and price >= t['sl']):
                        tag = "безубыток" if t['sl_moved_be'] else "стоп-лосс"
                        msgs.append(f"🛑 Закрыто по <b>{tag}</b>\n{t['symbol']} {t['tf']} | Цена: {round(price,6)}")
                        record_trade_result(t['symbol'],t['tf'],t['signal'],t['entry'],t['sl'],t['tp1'],t['tp2'],t['tp3'],round(price,6),tag,0)
                        close_trade(key); trades.pop(key, None)

                for m in msgs:
                    if t.get('chat_id'):
                        await app.bot.send_message(chat_id=t['chat_id'], text=m, parse_mode='HTML')
                if key in trades:
                    save_trades(trades)
            except Exception as ex:
                logger.error(f"check_trades {key}: {ex}")

# ================== АВТОСКАНЕР ==================
scanner_active = {}   # chat_id -> True/False
scanner_threshold = 75

async def run_scanner(app):
    """Сканирует топ-80 Binance Futures каждые 15 минут"""
    while True:
        await asyncio.sleep(60)  # проверяем каждую минуту нужно ли запускать
        active_chats = [cid for cid, active in scanner_active.items() if active]
        if not active_chats:
            continue

        # Запускаем раз в 15 минут
        state = load_scanner_state()
        last_run = state.get("last_run", 0)
        if (datetime.now().timestamp() - last_run) < 900:  # 15 минут
            continue

        state["last_run"] = datetime.now().timestamp()
        save_scanner_state(state)

        logger.info("Scanner: starting scan...")
        symbols = await fetch_binance_futures_symbols()
        if not symbols:
            logger.warning("Scanner: no symbols fetched")
            continue

        mode_cfg = TRADE_MODES["mid"]
        sent_this_run = []

        # Анализируем по 5 штук параллельно чтобы не перегружать
        for i in range(0, min(len(symbols), 80), 5):
            batch = symbols[i:i+5]
            tasks = [asyncio.create_task(analyze_symbol(s, "15m", mode_cfg)) for s in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception) or result is None: continue
                if result['signal'] not in ("🟩 LONG", "🟥 SHORT"): continue
                if result['score'] < scanner_threshold: continue
                if result.get('htf_conflict'): continue  # пропускаем конфликты

                symbol = result['symbol']
                # Антиспам: одна монета не чаще раза в 2 часа
                last_sent = state.get(f"sent_{symbol}", 0)
                if (datetime.now().timestamp() - last_sent) < 7200: continue

                state[f"sent_{symbol}"] = datetime.now().timestamp()
                save_scanner_state(state)
                sent_this_run.append(symbol)

                result['btc_trend_text'] = ""
                ai_text = await ask_ai(result) if groq_client else "AI отключён"
                msg = format_message(result, ai_text, is_scanner=True)

                for chat_id in active_chats:
                    try:
                        await app.bot.send_message(chat_id=chat_id, text=msg, parse_mode='HTML')
                        # Авто-открытие сделки
                        if result['sl_tp']:
                            trade_key = open_trade(symbol, "15m", result, chat_id)
                            await app.bot.send_message(
                                chat_id=chat_id,
                                text=f"📌 Сделка добавлена в мониторинг\n/close {trade_key}"
                            )
                    except Exception as ex:
                        logger.error(f"Scanner send to {chat_id}: {ex}")

            await asyncio.sleep(2)  # пауза между батчами

        logger.info(f"Scanner: done. Sent signals: {sent_this_run}")

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
        await update.message.reply_text(
            "🔍 <b>Сканер включён</b>\n"
            f"Порог скоринга: {scanner_threshold}/100\n"
            "Сканирую топ-80 Binance Futures каждые 15 минут\n"
            "Сигналы только при скоринге ≥75 и без конфликта HTF",
            parse_mode='HTML'
        )
    elif cmd == "off":
        scanner_active[chat_id] = False
        await update.message.reply_text("⏹ Сканер выключен")
    else:
        active = scanner_active.get(chat_id, False)
        state = load_scanner_state()
        last = state.get("last_run", 0)
        last_str = datetime.fromtimestamp(last).strftime("%H:%M:%S") if last else "никогда"
        await update.message.reply_text(
            f"📡 Сканер: {'🟢 активен' if active else '🔴 выключен'}\n"
            f"Последний скан: {last_str}\n"
            f"Порог: {scanner_threshold}/100",
        )

async def cmd_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    trades = load_trades()
    if not trades:
        await update.message.reply_text("📭 Нет открытых сделок")
        return
    lines = ["📋 <b>Открытые сделки:</b>\n"]
    for k, t in trades.items():
        lines.append(
            f"<b>{t['symbol']}</b> {t['tf']} {t['signal']}\n"
            f"Вход: {t['entry']} | СЛ: {t['sl']}\n"
            f"ТП1: {t['tp1']} {'✅' if t['tp1_hit'] else '⏳'} | ТП2: {t['tp2']} {'✅' if t['tp2_hit'] else '⏳'}\n"
            f"Ключ: <code>{k}</code>\n"
        )
    await update.message.reply_text("\n".join(lines), parse_mode='HTML')

async def cmd_close(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Использование: /close КЛЮЧ\nКлюч смотри в /trades")
        return
    key = context.args[0]
    trades = load_trades()
    if key in trades:
        close_trade(key)
        await update.message.reply_text(f"✅ Сделка {key} закрыта")
    else:
        await update.message.reply_text(f"❌ Не найдена. Открытые: {list(trades.keys())}")

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
        btc_task    = asyncio.create_task(fetch_ohlcv("BTC/USDT", "1h"))
        result_task = asyncio.create_task(analyze_symbol(symbol, tf, mode_cfg))
        (df_btc, _, _, _), result = await asyncio.gather(btc_task, result_task)

        if result is None:
            await msg.edit_text(f"❌ Нет данных для <b>{symbol}</b>", parse_mode='HTML')
            return

        _, btc_txt = get_trend(df_btc, "BTC")
        result["btc_trend_text"] = btc_txt

        ai_text = await ask_ai(result) if groq_client else "AI отключён"
        await msg.edit_text(format_message(result, ai_text), parse_mode='HTML')

        # Открываем мониторинг только при реальном сигнале
        if result['signal'] in ("🟩 LONG", "🟥 SHORT") and result['sl_tp']:
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
async def post_init(app):
    asyncio.create_task(check_trades(app))
    asyncio.create_task(run_scanner(app))
    logger.info("Background tasks started")

def main():
    app = (Application.builder()
           .token(TELEGRAM_TOKEN)
           .post_init(post_init)
           .build())

    app.add_handler(CommandHandler("start",     cmd_start))
    app.add_handler(CommandHandler("scan",      cmd_scan))
    app.add_handler(CommandHandler("trades",    cmd_trades))
    app.add_handler(CommandHandler("close",     cmd_close))
    app.add_handler(CommandHandler("backtest",  cmd_backtest))
    app.add_handler(CommandHandler("stats",     cmd_stats))
    app.add_handler(MessageHandler(filters.COMMAND, handle_command))

    print("🚀 Signal Volume Bot v4 запущен")
    app.run_polling()

if __name__ == '__main__':
    main()


# ================== БЭКТЕСТ ==================
STATS_FILE = Path("bot_stats.json")

def load_stats() -> dict:
    try:
        return json.loads(STATS_FILE.read_text()) if STATS_FILE.exists() else {}
    except: return {}

def save_stats(stats: dict):
    STATS_FILE.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

def record_trade_result(symbol: str, tf: str, signal: str, entry: float,
                         sl: float, tp1: float, tp2: float, tp3: float,
                         exit_price: float, exit_reason: str, score: int):
    """Записывает результат сделки в статистику"""
    stats = load_stats()
    is_long = "LONG" in signal
    if is_long:
        pnl_pct = (exit_price - entry) / entry * 100
    else:
        pnl_pct = (entry - exit_price) / entry * 100
    won = pnl_pct > 0

    trade = {
        "symbol": symbol, "tf": tf, "signal": signal,
        "entry": entry, "exit": exit_price,
        "pnl_pct": round(pnl_pct, 2),
        "exit_reason": exit_reason,
        "score": score, "won": won,
        "time": datetime.now().isoformat()
    }
    if "trades" not in stats:
        stats["trades"] = []
    stats["trades"].append(trade)
    save_stats(stats)


def _backtest_on_df(df: pd.DataFrame, mode_cfg: dict) -> list:
    """
    Прогоняет сигналы на историческом DataFrame.
    Для каждого сигнала симулирует выход по ТП1/ТП2/СЛ на следующих свечах.
    Возвращает список результатов.
    """
    results = []
    rsi_long  = mode_cfg["rsi_long"]
    rsi_short = mode_cfg["rsi_short"]
    hvn_mult  = mode_cfg["hvn_mult"]

    # Нужно минимум 150 свечей истории + 50 для симуляции выхода
    if len(df) < 200:
        return results

    df = df.copy()
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['ema20'] = ta.ema(df['close'], length=20)
    df['ema50'] = ta.ema(df['close'], length=50)

    # Шагаем по свечам с шагом 5 (не каждую — слишком много сигналов)
    for i in range(150, len(df) - 30, 5):
        window = df.iloc[:i]
        candle_i = df.iloc[i]

        rsi_val = window['rsi'].iloc[-1]
        atr_val = window['atr'].iloc[-1]
        if pd.isna(rsi_val) or pd.isna(atr_val): continue

        price = float(window['close'].iloc[-1])
        ema20 = float(window['ema20'].iloc[-1])
        ema50 = float(window['ema50'].iloc[-1])

        # Volume Profile на окне
        centers, vp = calculate_volume_profile(window)
        vp_mean = float(np.mean(vp))
        hvn = find_hvn(vp, centers, price, dist_limit=20)
        hvn_above = [n for n in hvn if n['is_above']]
        hvn_below = [n for n in hvn if not n['is_above']]
        strong_above = [n for n in hvn_above if n['distance_pct'] < 12]
        top_hvn = strong_above[0] if strong_above else None

        # EMA тренд
        if price > ema20 > ema50: trend = "UPTREND"
        elif price < ema20 < ema50: trend = "DOWNTREND"
        else: trend = "SIDEWAYS"

        # Сигнал
        signal = None
        if top_hvn and top_hvn['strength'] > hvn_mult * vp_mean:
            if rsi_val > rsi_short:  signal = "SHORT"
            elif rsi_val < rsi_long: signal = "LONG"
        elif rsi_val < rsi_long: signal = "LONG"
        elif rsi_val > rsi_short: signal = "SHORT"

        if signal is None: continue

        # ТП/СЛ
        all_nodes = hvn_above + hvn_below
        sl_tp = calculate_sl_tp(
            "🟩 LONG" if signal == "LONG" else "🟥 SHORT",
            price, float(atr_val), all_nodes
        )
        if not sl_tp: continue

        sl  = sl_tp['sl']
        tp1 = sl_tp['tp1']
        tp2 = sl_tp['tp2']

        # Симулируем выход на следующих 30 свечах
        future = df.iloc[i+1:i+31]
        exit_price = None
        exit_reason = "timeout"

        for _, fut_candle in future.iterrows():
            h, l = float(fut_candle['high']), float(fut_candle['low'])
            if signal == "LONG":
                if l <= sl:
                    exit_price, exit_reason = sl, "sl"
                    break
                if h >= tp2:
                    exit_price, exit_reason = tp2, "tp2"
                    break
                if h >= tp1:
                    exit_price, exit_reason = tp1, "tp1"
                    break
            else:
                if h >= sl:
                    exit_price, exit_reason = sl, "sl"
                    break
                if l <= tp2:
                    exit_price, exit_reason = tp2, "tp2"
                    break
                if l <= tp1:
                    exit_price, exit_reason = tp1, "tp1"
                    break

        if exit_price is None:
            exit_price = float(future['close'].iloc[-1])
            exit_reason = "timeout"

        if signal == "LONG":
            pnl = (exit_price - price) / price * 100
        else:
            pnl = (price - exit_price) / price * 100

        results.append({
            "signal": signal,
            "entry": round(price, 6),
            "exit": round(exit_price, 6),
            "pnl_pct": round(pnl, 2),
            "exit_reason": exit_reason,
            "rsi": round(rsi_val, 1),
            "score": 50,  # упрощённо для бэктеста
            "won": pnl > 0,
        })

    return results


async def run_backtest(symbol: str, tf: str, mode_cfg: dict) -> dict:
    """Загружает данные и прогоняет бэктест"""
    df, source, _, _ = await fetch_ohlcv(symbol, tf)
    if df is None or len(df) < 200:
        return {"error": "Недостаточно данных"}

    results = await asyncio.get_event_loop().run_in_executor(
        None, _backtest_on_df, df, mode_cfg
    )

    if not results:
        return {"error": "Нет сигналов на данном периоде"}

    total   = len(results)
    wins    = sum(1 for r in results if r['won'])
    losses  = total - wins
    winrate = round(wins / total * 100, 1)
    avg_pnl = round(sum(r['pnl_pct'] for r in results) / total, 2)
    avg_win = round(sum(r['pnl_pct'] for r in results if r['won']) / max(wins,1), 2)
    avg_loss= round(sum(r['pnl_pct'] for r in results if not r['won']) / max(losses,1), 2)
    best    = max(results, key=lambda x: x['pnl_pct'])
    worst   = min(results, key=lambda x: x['pnl_pct'])

    # По причинам выхода
    tp2_hits = sum(1 for r in results if r['exit_reason'] == 'tp2')
    tp1_hits = sum(1 for r in results if r['exit_reason'] == 'tp1')
    sl_hits  = sum(1 for r in results if r['exit_reason'] == 'sl')
    timeouts = sum(1 for r in results if r['exit_reason'] == 'timeout')

    # Профит-фактор
    gross_profit = sum(r['pnl_pct'] for r in results if r['pnl_pct'] > 0)
    gross_loss   = abs(sum(r['pnl_pct'] for r in results if r['pnl_pct'] < 0))
    profit_factor = round(gross_profit / max(gross_loss, 0.001), 2)

    return {
        "symbol": symbol, "tf": tf, "source": source,
        "total": total, "wins": wins, "losses": losses,
        "winrate": winrate, "avg_pnl": avg_pnl,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "best": best, "worst": worst,
        "tp2_hits": tp2_hits, "tp1_hits": tp1_hits,
        "sl_hits": sl_hits, "timeouts": timeouts,
        "candles": len(df),
    }


async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /backtest btc 1h mid
    /backtest eth 4h hard
    """
    args = context.args
    if not args:
        await update.message.reply_text(
            "Использование:\n/backtest BTC 1h mid\n/backtest ETH 4h hard\n\n"
            "Таймфреймы: 15m 1h 4h 1d\nРежимы: low mid hard"
        )
        return

    coin = args[0].upper().replace("USDT","").replace("/","")
    symbol = f"{coin}/USDT"
    tf   = args[1].lower() if len(args) > 1 and args[1].lower() in TF_MAP else "1h"
    mode = args[2].lower() if len(args) > 2 and args[2].lower() in TRADE_MODES else "mid"
    mode_cfg = TRADE_MODES[mode]

    msg = await update.message.reply_text(
        f"⏳ Запускаю бэктест {symbol} {tf} [{mode_cfg['label']}]...\n"
        f"Это займёт 20-40 секунд"
    )

    bt = await run_backtest(symbol, tf, mode_cfg)

    if "error" in bt:
        await msg.edit_text(f"❌ {bt['error']}")
        return

    e = html_escape
    wr_bar = "🟩"*(int(bt['winrate'])//20) + "⬜"*(5 - int(bt['winrate'])//20)
    pf_emoji = "✅" if bt['profit_factor'] >= 1.5 else ("⚠️" if bt['profit_factor'] >= 1.0 else "❌")

    verdict = ""
    if bt['winrate'] >= 55 and bt['profit_factor'] >= 1.5:
        verdict = "\n\n✅ <b>ВЫВОД: Стратегия рабочая</b> — можно торговать"
    elif bt['winrate'] >= 50 and bt['profit_factor'] >= 1.2:
        verdict = "\n\n⚠️ <b>ВЫВОД: Стратегия средняя</b> — нужна осторожность"
    else:
        verdict = "\n\n❌ <b>ВЫВОД: Стратегия убыточная</b> — не торговать"

    text = (
        f"📊 <b>БЭКТЕСТ: {e(symbol)} {tf} {e(mode_cfg['label'])}</b>\n"
        f"<i>Источник: {e(bt['source'])} | Свечей: {bt['candles']}</i>\n\n"
        f"<b>Общая статистика:</b>\n"
        f"Сигналов: <b>{bt['total']}</b> (✅{bt['wins']} / ❌{bt['losses']})\n"
        f"Винрейт: {wr_bar} <b>{bt['winrate']}%</b>\n"
        f"Средний P&L: <b>{bt['avg_pnl']:+.2f}%</b>\n\n"
        f"<b>Детали:</b>\n"
        f"Средний выигрыш: <b>+{bt['avg_win']}%</b>\n"
        f"Средний проигрыш: <b>{bt['avg_loss']}%</b>\n"
        f"Профит-фактор: {pf_emoji} <b>{bt['profit_factor']}</b>\n\n"
        f"<b>Выходы:</b>\n"
        f"🏆 ТП2: {bt['tp2_hits']} | 🎯 ТП1: {bt['tp1_hits']}\n"
        f"🛑 СЛ: {bt['sl_hits']} | ⏰ Таймаут: {bt['timeouts']}\n\n"
        f"<b>Лучшая сделка:</b> +{bt['best']['pnl_pct']}% ({bt['best']['signal']})\n"
        f"<b>Худшая сделка:</b> {bt['worst']['pnl_pct']}% ({bt['worst']['signal']})"
        f"{verdict}"
    )
    await msg.edit_text(text, parse_mode='HTML')


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Статистика реальных сделок бота"""
    stats = load_stats()
    trades = stats.get("trades", [])

    if not trades:
        await update.message.reply_text(
            "📭 Нет записанных сделок\n\n"
            "Статистика накапливается автоматически по мере торговли.\n"
            "Для исторического теста используй /backtest"
        )
        return

    total  = len(trades)
    wins   = sum(1 for t in trades if t['won'])
    losses = total - wins
    winrate = round(wins/total*100, 1)
    avg_pnl = round(sum(t['pnl_pct'] for t in trades)/total, 2)
    avg_win = round(sum(t['pnl_pct'] for t in trades if t['won'])/max(wins,1), 2)
    avg_loss= round(sum(t['pnl_pct'] for t in trades if not t['won'])/max(losses,1), 2)

    gross_profit = sum(t['pnl_pct'] for t in trades if t['pnl_pct'] > 0)
    gross_loss   = abs(sum(t['pnl_pct'] for t in trades if t['pnl_pct'] < 0))
    pf = round(gross_profit / max(gross_loss, 0.001), 2)

    best  = max(trades, key=lambda x: x['pnl_pct'])
    worst = min(trades, key=lambda x: x['pnl_pct'])

    # По символам
    from collections import Counter
    sym_wins = Counter(t['symbol'] for t in trades if t['won'])
    top_sym = sym_wins.most_common(3)

    # Последние 5 сделок
    recent = trades[-5:]
    recent_lines = []
    for t in reversed(recent):
        emoji = "✅" if t['won'] else "❌"
        recent_lines.append(f"{emoji} {t['symbol']} {t['signal']} {t['pnl_pct']:+.2f}%")

    wr_bar = "🟩"*(int(winrate)//20) + "⬜"*(5-int(winrate)//20)
    pf_emoji = "✅" if pf >= 1.5 else ("⚠️" if pf >= 1.0 else "❌")

    first_date = trades[0]['time'][:10] if trades else "—"

    text = (
        f"📈 <b>СТАТИСТИКА БОТА</b>\n"
        f"<i>С {first_date} | Всего сделок: {total}</i>\n\n"
        f"Винрейт: {wr_bar} <b>{winrate}%</b>\n"
        f"Профит-фактор: {pf_emoji} <b>{pf}</b>\n"
        f"Средний P&L: <b>{avg_pnl:+.2f}%</b>\n\n"
        f"✅ Прибыльных: <b>{wins}</b> (avg +{avg_win}%)\n"
        f"❌ Убыточных: <b>{losses}</b> (avg {avg_loss}%)\n\n"
        f"🏆 Лучшая: {best['symbol']} {best['pnl_pct']:+.2f}%\n"
        f"💀 Худшая: {worst['symbol']} {worst['pnl_pct']:+.2f}%\n\n"
        f"<b>Топ монеты:</b>\n"
        + "\n".join(f"  {s}: {c} побед" for s,c in top_sym) +
        f"\n\n<b>Последние сделки:</b>\n"
        + "\n".join(recent_lines)
    )
    await update.message.reply_text(text, parse_mode='HTML')
