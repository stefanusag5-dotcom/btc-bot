import logging
import asyncio
import aiohttp
import json
import os
from html import escape as html_escape
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import pandas_ta as ta
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
from google import genai

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRADES_FILE = Path("open_trades.json")

# Маппинг таймфреймов: аргумент -> (binance_interval, bybit_interval, label, свечей)
TF_MAP = {
    "15m": ("15m", "15",  "15m",  1000),
    "1h":  ("1h",  "60",  "1h",   1000),
    "4h":  ("4h",  "240", "4h",   500),
    "1d":  ("1d",  "D",   "1d",   365),
}
DEFAULT_TF = "15m"

# ================== РЕЖИМЫ ==================
TRADE_MODES = {
    "low": {
        "label": "🟢 LOW",
        "rsi_long": 32, "rsi_short": 72, "hvn_mult": 2.5,
        "personality": "Ты консервативный трейдер. Торгуешь только очень чёткие сигналы. "
                       "Если сигнал слабый или средний — говори ПРОПУСТИТЬ. "
                       "Всегда рекомендуй дождаться подтверждения закрытием свечи."
    },
    "mid": {
        "label": "🟡 MID",
        "rsi_long": 38, "rsi_short": 67, "hvn_mult": 2.0,
        "personality": "Ты сбалансированный интрадей трейдер на выбранном таймфрейме. "
                       "Торгуй средние и сильные сигналы. При слабом — жди подтверждения."
    },
    "hard": {
        "label": "🔴 HARD",
        "rsi_long": 45, "rsi_short": 58, "hvn_mult": 1.5,
        "personality": "Ты агрессивный скальпер волатильных альткоинов. "
                       "НИКОГДА не говори 'дождитесь подтверждения' или 'высокий риск'. "
                       "Волатильность — возможность. Давай конкретный вход прямо сейчас."
    },
}

# ================== ХРАНЕНИЕ СДЕЛОК ==================
def load_trades() -> dict:
    if TRADES_FILE.exists():
        try:
            return json.loads(TRADES_FILE.read_text())
        except:
            pass
    return {}

def save_trades(trades: dict):
    TRADES_FILE.write_text(json.dumps(trades, indent=2, ensure_ascii=False))

def open_trade(symbol: str, tf: str, data: dict):
    trades = load_trades()
    key = f"{symbol}_{tf}"
    trades[key] = {
        "symbol": symbol, "tf": tf,
        "entry": data["price"], "signal": data["signal"],
        "sl": data["sl_tp"]["sl"], "tp1": data["sl_tp"]["tp1"],
        "tp2": data["sl_tp"]["tp2"], "tp3": data["sl_tp"]["tp3"],
        "sl_moved_be": False, "sl_moved_tp1": False,
        "tp1_hit": False, "tp2_hit": False,
        "opened_at": datetime.now().isoformat(),
        "chat_id": data.get("chat_id"),
    }
    save_trades(trades)

def close_trade(key: str):
    trades = load_trades()
    trades.pop(key, None)
    save_trades(trades)

# ================== DATA FETCHING ==================
async def _fetch_klines(ticker: str, interval_bn: str, interval_bb: str,
                         limit: int, session: aiohttp.ClientSession):
    """Пробует Binance Futures -> Binance Spot -> Bybit"""
    # Binance Futures
    try:
        url = "https://fapi.binance.com/fapi/v1/klines"
        async with session.get(url, params={"symbol": ticker, "interval": interval_bn, "limit": limit}) as r:
            if r.status == 200:
                data = await r.json()
                if data:
                    return _parse_binance(data), "Binance Futures"
    except Exception as e:
        logger.warning(f"BF {ticker}: {e}")

    # Binance Spot
    try:
        url = "https://api.binance.com/api/v3/klines"
        async with session.get(url, params={"symbol": ticker, "interval": interval_bn, "limit": limit}) as r:
            if r.status == 200:
                data = await r.json()
                if data:
                    return _parse_binance(data), "Binance Spot"
    except Exception as e:
        logger.warning(f"BS {ticker}: {e}")

    # Bybit
    try:
        url = "https://api.bybit.com/v5/market/kline"
        async with session.get(url, params={"category": "linear", "symbol": ticker,
                                             "interval": interval_bb, "limit": limit}) as r:
            if r.status == 200:
                data = await r.json()
                if data.get("retCode") == 0:
                    raw = list(reversed(data["result"]["list"]))
                    return _parse_bybit(raw), "Bybit"
    except Exception as e:
        logger.warning(f"Bybit {ticker}: {e}")

    return None, None

def _parse_binance(data) -> pd.DataFrame:
    df = pd.DataFrame(data, columns=[
        'ts','open','high','low','close','volume',
        'ct','qv','trades','tbb','tbq','ignore'])
    df = df[['ts','open','high','low','close','volume','tbb']].copy()
    for c in ['open','high','low','close','volume','tbb']:
        df[c] = pd.to_numeric(df[c])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.rename(columns={'ts':'timestamp','tbb':'taker_buy_base'}, inplace=True)
    return df

def _parse_bybit(data) -> pd.DataFrame:
    df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume','turnover'])
    df = df[['ts','open','high','low','close','volume']].copy()
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c])
    df['ts'] = pd.to_datetime(pd.to_numeric(df['ts']), unit='ms')
    df.rename(columns={'ts':'timestamp'}, inplace=True)
    df['taker_buy_base'] = df['volume'] / 2
    return df

async def fetch_ohlcv(symbol: str, tf: str = "15m"):
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

async def fetch_higher_tf(symbol: str, tf: str):
    """Загружает старший таймфрейм для фильтрации сигнала"""
    higher = {"15m": "1h", "1h": "4h", "4h": "1d", "1d": "1d"}
    htf = higher.get(tf, "1h")
    df, _, _, _ = await fetch_ohlcv(symbol, htf)
    return df, htf

async def fetch_daily_vp(symbol: str):
    """1000 дневных свечей для глубокого Volume Profile"""
    ticker = symbol.replace("/", "")
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        df, _ = await _fetch_klines(ticker, "1d", "D", 1000, session)
    return df

# ================== VOLUME PROFILE ==================
def calculate_volume_profile(df: pd.DataFrame, num_bins: int = 120):
    price_min, price_max = df['low'].min(), df['high'].max()
    if price_min == price_max:
        return np.array([price_min]), np.array([df['volume'].sum()])
    bins = np.linspace(price_min, price_max, num_bins + 1)
    vp = np.zeros(num_bins)
    for _, row in df.iterrows():
        lo = max(0, np.searchsorted(bins, row['low']) - 1)
        hi = min(num_bins - 1, np.searchsorted(bins, row['high']) - 1)
        if lo == hi:
            vp[lo] += row['volume']
        else:
            vp[lo:hi+1] += row['volume'] / (hi - lo + 1)
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

def merge_hvn_levels(local_nodes: list, daily_nodes: list) -> list:
    """Помечает полки как локальные или глобальные (дневные)"""
    for n in daily_nodes:
        n['type'] = 'daily'
    all_nodes = local_nodes + daily_nodes
    # убираем дубли ближе 0.5%
    merged = []
    for n in sorted(all_nodes, key=lambda x: -x['strength']):
        if not any(abs(n['price'] - m['price']) / n['price'] * 100 < 0.5 for m in merged):
            merged.append(n)
    return sorted(merged, key=lambda x: x['price'])

# ================== ТЕХНИЧЕСКИЙ АНАЛИЗ ==================
def detect_candle_pattern(df: pd.DataFrame) -> str:
    c, p = df.iloc[-1], df.iloc[-2]
    body = abs(c['close'] - c['open'])
    rng = c['high'] - c['low']
    if rng == 0: return "Дожи"
    uw = c['high'] - max(c['close'], c['open'])
    lw = min(c['close'], c['open']) - c['low']
    if lw > body * 2 and uw < body * 0.5: return "📌 Бычий пин-бар"
    if uw > body * 2 and lw < body * 0.5: return "📌 Медвежий пин-бар"
    if (c['close'] > c['open'] and p['close'] < p['open'] and
            c['close'] > p['open'] and c['open'] < p['close']): return "🟢 Бычье поглощение"
    if (c['close'] < c['open'] and p['close'] > p['open'] and
            c['close'] < p['open'] and c['open'] > p['close']): return "🔴 Медвежье поглощение"
    if body < rng * 0.1: return "〰️ Дожи"
    return "Обычная свеча"

def calculate_delta(df: pd.DataFrame) -> str:
    r = df.tail(5)
    if 'taker_buy_base' not in r.columns: return "N/A"
    bv = r['taker_buy_base'].sum()
    tv = r['volume'].sum()
    if tv == 0: return "N/A"
    bp = bv / tv * 100
    e = "🟢" if bp > 55 else ("🔴" if bp < 45 else "⚪")
    return f"{e} {bp:.0f}% покупок / {100-bp:.0f}% продаж"

def find_sr_levels(df: pd.DataFrame, price: float):
    r = df.tail(200)
    highs, lows = [], []
    for i in range(2, len(r)-2):
        h = r.iloc[i]['high']
        l = r.iloc[i]['low']
        if h > r.iloc[i-1]['high'] and h > r.iloc[i+1]['high']: highs.append(h)
        if l < r.iloc[i-1]['low'] and l < r.iloc[i+1]['low']: lows.append(l)
    res = sorted([h for h in highs if h > price])[:3]
    sup = sorted([l for l in lows if l < price], reverse=True)[:3]
    return [round(s, 6) for s in sup], [round(r, 6) for r in res]

def get_trend(df: pd.DataFrame, label: str) -> tuple:
    if df is None or len(df) < 50: return "UNKNOWN", f"Нет данных {label}"
    close = df['close']
    e20 = ta.ema(close, length=20).iloc[-1]
    e50 = ta.ema(close, length=50).iloc[-1]
    cur = close.iloc[-1]
    if cur > e20 > e50: return "UPTREND", f"🟢 {label} аптренд"
    if cur < e20 < e50: return "DOWNTREND", f"🔴 {label} даунтренд"
    return "SIDEWAYS", f"⚪ {label} боковик"

# ================== РАСЧЁТ ТП/СЛ ==================
def _snap_to_level(target: float, levels: list, tolerance_pct: float = 0.8) -> float:
    """Подтягивает ТП к ближайшему уровню если он в пределах tolerance_pct%"""
    best = target
    best_dist = float('inf')
    for lvl in levels:
        dist = abs(lvl - target) / target * 100
        if dist < tolerance_pct and dist < best_dist:
            best = lvl
            best_dist = dist
    return round(best, 6)

def calculate_sl_tp(signal: str, price: float, atr: float,
                    hv_nodes: list, supports: list = None, resistances: list = None) -> dict:
    if signal not in ("🟩 LONG", "🟥 SHORT"): return {}
    is_long = signal == "🟩 LONG"
    below = [n for n in hv_nodes if not n['is_above']]
    above = [n for n in hv_nodes if n['is_above']]

    # Все уровни для снэппинга ТП
    hvn_prices_above = [n['price'] for n in above]
    hvn_prices_below = [n['price'] for n in below]
    res_levels = list(resistances or [])
    sup_levels = list(supports or [])

    if is_long:
        # СЛ: за ближайший HVN снизу или ATR*1.5, но не ближе ATR*1.2
        sl = (min(below, key=lambda x: x['distance_pct'])['price'] - atr * 0.3
              if below else price - atr * 1.5)
        sl = min(sl, price - atr * 1.2)
        # Подтягиваем СЛ к ближайшей поддержке снизу (не выше цены)
        near_sup = [s for s in sup_levels if s < price and s > sl - atr]
        if near_sup:
            sl = min(sl, min(near_sup) - atr * 0.2)
        sl = round(sl, 6)
        risk = price - sl

        # Минимальный фильтр: ТП1 должен быть дальше 0.5 ATR от цены
        min_tp1 = price + atr * 0.5
        tp1_raw = price + risk * 1.5
        tp1 = max(tp1_raw, min_tp1)
        # Снэппинг ТП1 к ближайшему сопротивлению или HVN выше
        tp1 = _snap_to_level(tp1, hvn_prices_above + res_levels, tolerance_pct=0.8)
        tp1 = round(tp1, 6)

        tp2_raw = price + risk * 2.5
        tp2 = _snap_to_level(tp2_raw, hvn_prices_above + res_levels, tolerance_pct=1.0)
        tp2 = round(max(tp2, tp1 + atr * 0.5), 6)  # ТП2 всегда дальше ТП1

        # ТП3: сильнейший дальний HVN или 4R
        far_above = [n for n in above if n['price'] > tp2]
        tp3_raw = (max(far_above, key=lambda x: x['strength'])['price']
                   if far_above else price + risk * 4.0)
        tp3 = round(max(tp3_raw, price + risk * 3.5, tp2 + atr), 6)

    else:  # SHORT
        sl = (max(above, key=lambda x: x['distance_pct'])['price'] + atr * 0.3
              if above else price + atr * 1.5)
        sl = max(sl, price + atr * 1.2)
        # Подтягиваем СЛ к ближайшему сопротивлению сверху
        near_res = [r for r in res_levels if r > price and r < sl + atr]
        if near_res:
            sl = max(sl, max(near_res) + atr * 0.2)
        sl = round(sl, 6)
        risk = sl - price

        min_tp1 = price - atr * 0.5
        tp1_raw = price - risk * 1.5
        tp1 = min(tp1_raw, min_tp1)
        tp1 = _snap_to_level(tp1, hvn_prices_below + sup_levels, tolerance_pct=0.8)
        tp1 = round(tp1, 6)

        tp2_raw = price - risk * 2.5
        tp2 = _snap_to_level(tp2_raw, hvn_prices_below + sup_levels, tolerance_pct=1.0)
        tp2 = round(min(tp2, tp1 - atr * 0.5), 6)

        far_below = [n for n in below if n['price'] < tp2]
        tp3_raw = (min(far_below, key=lambda x: x['strength'])['price']
                   if far_below else price - risk * 4.0)
        tp3 = round(min(tp3_raw, price - risk * 3.5, tp2 - atr), 6)

    # Финальный RR считаем по ТП2 (основная цель)
    rr = round(abs(tp2 - price) / max(abs(price - sl), 0.0001), 2)

    # Предупреждение если RR плохой
    rr_warn = ""
    if rr < 1.5:
        rr_warn = "⚠️ R/R ниже 1.5 — сделка под вопросом"

    return {
        "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "risk_pct": round(abs(price - sl) / price * 100, 2),
        "rr_ratio": rr,
        "rr_warn": rr_warn,
    }

# ================== СКОРИНГ СИГНАЛА ==================
def score_signal(rsi, signal, top_hvn, vp_mean, delta_str,
                 trend_local, trend_higher, candle_pattern, atr_pct) -> int:
    score = 0
    if signal in ("🟩 LONG", "🟥 SHORT"):
        score += 20
    if top_hvn and top_hvn['strength'] > 2.0 * vp_mean:
        score += 25
    elif top_hvn:
        score += 10
    if signal == "🟩 LONG":
        if rsi < 30: score += 20
        elif rsi < 38: score += 10
        if trend_local == "UPTREND": score += 10
        if trend_higher == "UPTREND": score += 15
        if "покупок" in delta_str and float(delta_str.split('%')[0].split()[-1]) > 60: score += 10
    elif signal == "🟥 SHORT":
        if rsi > 70: score += 20
        elif rsi > 62: score += 10
        if trend_local == "DOWNTREND": score += 10
        if trend_higher == "DOWNTREND": score += 15
        if "покупок" in delta_str and float(delta_str.split('%')[0].split()[-1]) < 40: score += 10
    if "поглощение" in candle_pattern or "пин-бар" in candle_pattern: score += 10
    return min(score, 100)

# ================== АНАЛИЗ ==================
async def analyze_symbol(symbol: str, tf: str = "15m", mode_cfg: dict = None) -> dict | None:
    if mode_cfg is None: mode_cfg = TRADE_MODES["mid"]

    # Грузим данные параллельно: основной TF, старший TF, дневной VP
    main_task   = asyncio.create_task(fetch_ohlcv(symbol, tf))
    higher_task = asyncio.create_task(fetch_higher_tf(symbol, tf))
    daily_task  = asyncio.create_task(fetch_daily_vp(symbol))

    (df, source, fr, oi), (df_htf, htf_label), df_daily = await asyncio.gather(
        main_task, higher_task, daily_task)

    if df is None or len(df) < 100: return None

    price = df['close'].iloc[-1]

    # Индикаторы
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    rsi = round(df['rsi'].iloc[-1], 1)
    atr = round(df['atr'].iloc[-1], 6)
    atr_pct = round(atr / price * 100, 2)
    ema20 = round(ta.ema(df['close'], length=20).iloc[-1], 6)
    ema50 = round(ta.ema(df['close'], length=50).iloc[-1], 6)
    ema200s = ta.ema(df['close'], length=200)
    ema200 = round(ema200s.iloc[-1], 6) if ema200s is not None and len(ema200s.dropna()) > 0 else None

    if price > ema20 > ema50:   ema_trend = "📈 Восходящий"
    elif price < ema20 < ema50: ema_trend = "📉 Нисходящий"
    else:                        ema_trend = "↔️ Боковик"

    # Volume Profile: локальный (текущий TF) + глобальный (дневной)
    centers_l, vp_l = calculate_volume_profile(df)
    poc_idx = np.argmax(vp_l)
    poc = round(float(centers_l[poc_idx]), 6)
    local_nodes = find_hvn(vp_l, centers_l, price, dist_limit=20)

    daily_nodes = []
    if df_daily is not None and len(df_daily) > 50:
        centers_d, vp_d = calculate_volume_profile(df_daily, num_bins=150)
        daily_nodes = find_hvn(vp_d, centers_d, price, dist_limit=30)
        for n in daily_nodes: n['type'] = 'daily'

    all_nodes = merge_hvn_levels(local_nodes, daily_nodes)
    hvn_above = [n for n in all_nodes if n['is_above']]
    hvn_below = [n for n in all_nodes if not n['is_above']]

    # Уровни, паттерн, дельта
    supports, resistances = find_sr_levels(df, price)
    supports    = [float(round(x, 6)) for x in supports]
    resistances = [float(round(x, 6)) for x in resistances]
    candle = detect_candle_pattern(df)
    delta = calculate_delta(df)

    # Тренды
    trend_l, trend_l_txt = get_trend(df, tf)
    trend_h, trend_h_txt = get_trend(df_htf, htf_label) if df_htf is not None else ("UNKNOWN", "Нет данных HTF")

    # Сигнал с порогами режима
    rsi_long  = mode_cfg["rsi_long"]
    rsi_short = mode_cfg["rsi_short"]
    hvn_mult  = mode_cfg["hvn_mult"]
    strong_above = [n for n in hvn_above if n['distance_pct'] < 12]
    top_hvn = strong_above[0] if strong_above else None
    vp_mean = np.mean(vp_l)

    signal, reason = "НЕТ СИГНАЛА", "Нейтральная структура"

    if top_hvn and top_hvn['strength'] > hvn_mult * vp_mean:
        if rsi > rsi_short:
            signal, reason = "🟥 SHORT", f"Полка тепла сверху {top_hvn['price']} + RSI {rsi}"
        elif rsi < rsi_long:
            signal, reason = "🟩 LONG", f"Полка тепла сверху {top_hvn['price']} + RSI {rsi}"
        else:
            signal, reason = "⚠️ WATCH", f"Полка тепла сверху {top_hvn['price']}, RSI нейтрален"
    elif rsi < rsi_long:
        signal, reason = "🟩 LONG", f"Перепроданность RSI ({rsi})"
    elif rsi > rsi_short:
        signal, reason = "🟥 SHORT", f"Перекупленность RSI ({rsi})"

    # Фильтр по старшему ТФ
    htf_conflict = ""
    if signal == "🟩 LONG" and trend_h == "DOWNTREND":
        htf_conflict = f"⚠️ LONG против тренда {htf_label}!"
    elif signal == "🟥 SHORT" and trend_h == "UPTREND":
        htf_conflict = f"⚠️ SHORT против тренда {htf_label}!"

    sl_tp = calculate_sl_tp(signal, price, atr, all_nodes, supports, resistances)

    score = score_signal(rsi, signal, top_hvn, vp_mean, delta,
                          trend_l, trend_h, candle, atr_pct)

    return {
        "symbol": symbol, "tf": tf, "price": round(price, 6),
        "signal": signal, "reason": reason, "score": score,
        "rsi": rsi, "atr": atr, "atr_pct": atr_pct,
        "ema20": ema20, "ema50": ema50, "ema200": ema200, "ema_trend": ema_trend,
        "poc": poc,
        "hvn_above": hvn_above, "hvn_below": hvn_below,
        "supports": supports, "resistances": resistances,
        "candle_pattern": candle, "delta": delta,
        "trend_local": trend_l_txt, "trend_higher": trend_h_txt,
        "htf_conflict": htf_conflict,
        "sl_tp": sl_tp, "source": source,
        "funding_rate": round(fr, 4) if fr is not None else None,
        "open_interest": int(oi) if oi is not None else None,
        "time": datetime.now().strftime("%H:%M"),
        "btc_trend_text": "",
        "mode_label": mode_cfg["label"],
        "mode_personality": mode_cfg["personality"],
    }

# ================== GEMINI ==================
async def ask_gemini(data: dict) -> str:
    if not gemini_client: return "AI отключён"
    sl_tp = data.get("sl_tp", {})
    tf = data['tf']
    # Адаптируем personality под таймфрейм
    tf_context = {
        "15m": "Это скальпинг/интрадей. Сделка живёт часы.",
        "1h":  "Это интрадей. Сделка живёт 1-2 дня.",
        "4h":  "Это свинг-трейдинг. Сделка живёт несколько дней.",
        "1d":  "Это позиционная торговля. Сделка живёт недели.",
    }.get(tf, "Интрадей.")

    hvn_a = ", ".join(f"{n['price']}({'D' if n.get('type')=='daily' else 'L'})" for n in data['hvn_above'][:3])
    hvn_b = ", ".join(f"{n['price']}({'D' if n.get('type')=='daily' else 'L'})" for n in data['hvn_below'][:3])
    conflict_line = f"\nКОНФЛИКТ: {data['htf_conflict']}" if data.get('htf_conflict') else ""

    prompt = f"""{data['mode_personality']}
{tf_context}

Данные по {data['symbol']} на таймфрейме {tf}:
Цена={data['price']} | Сигнал={data['signal']} | Скоринг={data['score']}/100
RSI={data['rsi']} | ATR={data['atr_pct']}% | Свеча={data['candle_pattern']}
Тренд {tf}: {data['trend_local']} | Тренд HTF: {data['trend_higher']}{conflict_line}
Дельта={data['delta']} | Фандинг={data.get('funding_rate','N/A')}% | BTC={data.get('btc_trend_text','N/A')}
POC={data['poc']} | HVN выше={hvn_a} | HVN ниже={hvn_b}
СЛ={sl_tp.get('sl','?')} ТП1={sl_tp.get('tp1','?')} ТП2={sl_tp.get('tp2','?')} ТП3={sl_tp.get('tp3','?')} R/R=1:{sl_tp.get('rr_ratio','?')}

СТРОГИЕ ПРАВИЛА:
- Отвечай ТОЛЬКО на основе цифр выше
- ЗАПРЕЩЕНО упоминать новости, геополитику, макроэкономику, события, инфляцию, ФРС, войны, выборы — даже если кажется что они важны
- ЗАПРЕЩЕНО использовать знания из обучающих данных о рыночных событиях
- Если фундаментал противоречит технике — игнорируй фундаментал, ты технический трейдер
- ТП и СЛ уже рассчитаны — не пересчитывай и не меняй их цифры

Дай короткий вывод СТРОГО в формате (без повторения данных выше):
✅ ВЫВОД: войти / пропустить / ждать
💪 СИЛА: слабый / средний / сильный
⚠️ РИСК: [технический риск одним предложением — только про уровни/индикаторы]
💡 СОВЕТ: [одно конкретное действие с учётом таймфрейма {tf}]"""

    MODELS = [
        "gemini-2.0-flash-lite",   # бесплатный, щедрый лимит
        "gemini-2.0-flash",        # фолбэк
    ]
    loop = asyncio.get_event_loop()

    for model in MODELS:
        for attempt in range(2):
            try:
                r = await loop.run_in_executor(None,
                    lambda m=model: gemini_client.models.generate_content(
                        model=m, contents=prompt))
                logger.info(f"Gemini OK: model={model}")
                return r.text.strip()
            except Exception as ex:
                err = str(ex)
                logger.error(f"Gemini model={model} attempt={attempt+1}: {err}")
                is_rate   = "429" in err or "quota" in err.lower()
                is_unavail = "404" in err or "not found" in err.lower()
                if is_unavail:
                    break  # сразу следующая модель
                if is_rate:
                    # Берём время из ответа API если есть, иначе 15s
                    import re
                    m2 = re.search(r'retry.*?(\d+)s', err, re.IGNORECASE)
                    wait = int(m2.group(1)) + 2 if m2 else 15
                    logger.info(f"Rate limit, waiting {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    await asyncio.sleep(3)

    return "⏳ Gemini: лимит исчерпан, попробуй позже"

# ================== ФОРМАТИРОВАНИЕ ==================
def format_message(result: dict, gemini_text: str) -> str:
    e = html_escape
    sl_tp = result.get("sl_tp", {})
    p = result['price']

    def pct(t): return round(abs(t - p) / p * 100, 2)

    trade_block = ""
    if sl_tp:
        rr_warn = sl_tp.get("rr_warn", "")
        rr_warn_line = f"\n{e(rr_warn)}" if rr_warn else ""
        trade_block = (
            f"\n📐 <b>ПЛАН СДЕЛКИ</b>\n"
            f"├ 🛑 СЛ: <b>{sl_tp['sl']}</b> (-{pct(sl_tp['sl'])}%)\n"
            f"├ 🎯 ТП1: <b>{sl_tp['tp1']}</b> (+{pct(sl_tp['tp1'])}%) → <i>БУ</i>\n"
            f"├ 🎯 ТП2: <b>{sl_tp['tp2']}</b> (+{pct(sl_tp['tp2'])}%) → <i>СЛ на ТП1</i>\n"
            f"└ 🏆 ТП3: <b>{sl_tp['tp3']}</b> (+{pct(sl_tp['tp3'])}%) → <i>финал</i>\n"
            f"R/R: 1:{sl_tp['rr_ratio']} | Риск: {sl_tp['risk_pct']}%{rr_warn_line}\n"
        )

    score = result.get('score', 0)
    score_bar = "🟩" * (score // 20) + "⬜" * (5 - score // 20)
    fr_str = ""
    if result.get('funding_rate') is not None:
        fr = result['funding_rate']
        fe = "🔴" if fr > 0.05 else ("🟢" if fr < -0.05 else "⚪")
        fr_str = f"\nФандинг: {fe} {fr}%"
    oi_str = f" | OI: {result['open_interest']:,}" if result.get('open_interest') else ""

    conflict = f"\n{e(result['htf_conflict'])}" if result.get('htf_conflict') else ""

    # Пометить глобальные полки
    def fmt_nodes(nodes):
        out = []
        for n in nodes[:3]:
            mark = "🌍" if n.get('type') == 'daily' else "📍"
            out.append(f"{mark}{n['price']}")
        return " ".join(out) if out else "—"

    return (
        f"📊 <b>{e(result['symbol'])}</b> {result['tf']} • {result['time']} • {e(result['mode_label'])}\n"
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
        f"{conflict}\n\n"
        f"POC: {result['poc']}\n"
        f"HVN↑: {fmt_nodes(result['hvn_above'])}\n"
        f"HVN↓: {fmt_nodes(result['hvn_below'])}\n"
        f"Сопр: {result['resistances'][:2]} | Подд: {result['supports'][:2]}\n"
        f"{trade_block}\n"
        f"🧠 <b>Gemini:</b>\n{e(gemini_text)}"
    )

# ================== МОНИТОРИНГ СДЕЛОК ==================
async def check_trades(app):
    """Фоновая задача: каждые 5 минут проверяет открытые сделки"""
    while True:
        await asyncio.sleep(300)
        trades = load_trades()
        if not trades:
            continue
        for key, t in list(trades.items()):
            try:
                df, _, _, _ = await fetch_ohlcv(t['symbol'], t['tf'])
                if df is None: continue
                price = df['close'].iloc[-1]
                is_long = "LONG" in t['signal']
                msgs = []

                if not t['tp1_hit']:
                    hit = price >= t['tp1'] if is_long else price <= t['tp1']
                    if hit:
                        t['tp1_hit'] = True
                        t['sl'] = t['entry']  # безубыток
                        t['sl_moved_be'] = True
                        msgs.append(
                            f"🎯 <b>ТП1 достигнут!</b> {t['symbol']} {t['tf']}\n"
                            f"СЛ перенесён в <b>безубыток ({t['entry']})</b>\n"
                            f"Ждём ТП2: {t['tp2']}"
                        )

                elif not t['tp2_hit']:
                    hit = price >= t['tp2'] if is_long else price <= t['tp2']
                    if hit:
                        t['tp2_hit'] = True
                        t['sl'] = t['tp1']
                        t['sl_moved_tp1'] = True
                        msgs.append(
                            f"🎯🎯 <b>ТП2 достигнут!</b> {t['symbol']} {t['tf']}\n"
                            f"СЛ перенесён на <b>ТП1 ({t['tp1']})</b>\n"
                            f"Финальная цель ТП3: {t['tp3']}"
                        )

                else:
                    hit = price >= t['tp3'] if is_long else price <= t['tp3']
                    if hit:
                        msgs.append(
                            f"🏆 <b>ТП3 достигнут! Сделка закрыта.</b>\n"
                            f"{t['symbol']} {t['tf']} — полная цель выполнена!"
                        )
                        close_trade(key)
                        trades.pop(key, None)

                # Проверка выбития по СЛ
                if key in trades:
                    sl_hit = price <= t['sl'] if is_long else price >= t['sl']
                    if sl_hit:
                        tag = "безубыток" if t['sl_moved_be'] else "стоп-лосс"
                        msgs.append(
                            f"🛑 <b>Сделка закрыта по {tag}</b>\n"
                            f"{t['symbol']} {t['tf']} | Цена: {round(price,6)}"
                        )
                        close_trade(key)
                        trades.pop(key, None)

                for msg in msgs:
                    chat_id = t.get('chat_id')
                    if chat_id:
                        await app.bot.send_message(chat_id=chat_id, text=msg, parse_mode='HTML')

                if key in trades:
                    save_trades(trades)
            except Exception as ex:
                logger.error(f"check_trades {key}: {ex}")

# ================== КОМАНДЫ ==================
MODES = {"low", "mid", "hard"}
TFS   = set(TF_MAP.keys())

def parse_args(text: str):
    """
    /btc          -> BTC/USDT, 15m, mid
    /btc 4h       -> BTC/USDT, 4h,  mid
    /btc 4h hard  -> BTC/USDT, 4h,  hard
    /siren hard   -> SIREN/USDT, 15m, hard
    """
    parts = text.lower().strip().lstrip("/").split()
    coin = parts[0].upper().replace("USDT","").replace("/","")
    symbol = f"{coin}/USDT"
    tf   = next((p for p in parts[1:] if p in TFS), DEFAULT_TF)
    mode = next((p for p in parts[1:] if p in MODES), "mid")
    return symbol, tf, mode

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ <b>Signal Volume Bot</b>\n\n"
        "<b>Синтаксис:</b>\n"
        "/btc — анализ BTC 15m mid\n"
        "/eth 4h — ETH на 4h\n"
        "/siren hard — агрессивный режим\n"
        "/btc 1h low — 1h консерватив\n\n"
        "<b>Таймфреймы:</b> 15m 1h 4h 1d\n"
        "<b>Режимы:</b> 🟢low 🟡mid 🔴hard\n\n"
        "<b>Ведение сделок:</b>\n"
        "После сигнала бот автоматически следит за ценой и пишет когда ТП1/ТП2/ТП3 достигнуты или СЛ выбит\n\n"
        "🌍 = глубокая дневная полка | 📍 = локальная полка",
        parse_mode='HTML'
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
            f"ТП1: {t['tp1']} {'✅' if t['tp1_hit'] else '⏳'} | "
            f"ТП2: {t['tp2']} {'✅' if t['tp2_hit'] else '⏳'}\n"
            f"Открыта: {t['opened_at'][:16]}\n"
        )
    await update.message.reply_text("\n".join(lines), parse_mode='HTML')

async def cmd_close(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text("Использование: /close BTCUSDT_4h")
        return
    key = args[0]
    trades = load_trades()
    if key in trades:
        close_trade(key)
        await update.message.reply_text(f"✅ Сделка {key} закрыта вручную")
    else:
        await update.message.reply_text(f"❌ Сделка {key} не найдена\nОткрытые: {list(trades.keys())}")

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
            await msg.edit_text(
                f"❌ Нет данных для <b>{symbol}</b>\n"
                f"Проверь тикер или попробуй позже.",
                parse_mode='HTML'
            )
            return

        _, btc_txt = get_trend(df_btc, "BTC")
        result["btc_trend_text"] = btc_txt

        gemini_text = await ask_gemini(result) if gemini_client else "AI отключён"
        response = format_message(result, gemini_text)
        await msg.edit_text(response, parse_mode='HTML')

        # Открываем сделку для мониторинга
        if result['signal'] in ("🟩 LONG", "🟥 SHORT") and result['sl_tp']:
            result['chat_id'] = chat_id
            open_trade(symbol, tf, result)
            await update.message.reply_text(
                f"📌 Сделка добавлена в мониторинг\n"
                f"Бот уведомит при достижении ТП/СЛ\n"
                f"Просмотр: /trades | Закрыть: /close {symbol.replace('/','')+'_'+tf}",
            )

    except Exception as ex:
        logger.error(f"handle_command: {ex}", exc_info=True)
        await msg.edit_text(f"❌ Ошибка: {html_escape(str(ex))}", parse_mode='HTML')

# ================== MAIN ==================
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("trades", cmd_trades))
    app.add_handler(CommandHandler("close",  cmd_close))
    app.add_handler(MessageHandler(filters.COMMAND, handle_command))

    # Запускаем мониторинг сделок в фоне
    loop = asyncio.get_event_loop()
    loop.create_task(check_trades(app))

    print("🚀 Signal Volume Bot v3 запущен")
    app.run_polling()

if __name__ == '__main__':
    main()
