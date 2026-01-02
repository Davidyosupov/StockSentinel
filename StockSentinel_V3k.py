# -*- coding: utf-8 -*-
"""
StockSentinel V3k: The Prism Multi-Profile Engine
==========================================
• 5 פרופילים: Yellow, Green, Pink, Black (Bottom), Blue (Bayesian)
• ערכי סינון מעודכנים מ-stock_screener_extended
• יצירת PDF עם גרפים לכל צבע
• Rate limiting חכם למניעת חסימות מ-Finviz
"""

from __future__ import annotations

import logging
import re
import math
import os
import time
import random
from datetime import datetime, timedelta
from io import StringIO
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import black, darkgrey, white, red, green, HexColor
from reportlab.lib.units import inch

# ============================ הגדרות ולוגינג ============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AdvancedScreener")

INTERMEDIATE_DIR = "intermediate"
CHARTS_DIR = "charts_temp"
OUTPUTS_DIR = "outputs"

for d in [INTERMEDIATE_DIR, CHARTS_DIR, OUTPUTS_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# ============================ קונפיגורציה ============================
CONFIG = {
    "BATCH_SIZE": 500,  # עיבוד בקבוצות
    "SAVE_EVERY_N": 100,  # שמירת ביניים כל N מניות
    "FINVIZ_DELAY_MIN": 0.8,  # השהיה מינימלית בין בקשות ל-Finviz
    "FINVIZ_DELAY_MAX": 1.5,  # השהיה מקסימלית
    "FINVIZ_BATCH_DELAY": 10,  # השהיה בין קבוצות גרפים
    "FINVIZ_BATCH_SIZE": 20,  # גרפים בכל קבוצה
    "MAX_CHARTS_PER_COLOR": 50,  # מקסימום גרפים לכל צבע ב-PDF
    "TICKER_DELAY": 0.05,  # השהיה קצרה בין טיקרים
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}


# ============================ עזרי חישוב ============================
def safe_float(x, default=np.nan) -> float:
    try:
        if x is None or (hasattr(pd, "isna") and pd.isna(x)):
            return float(default)
        return float(x)
    except:
        return float(default)


def safe_int(x, default=0) -> int:
    try:
        if x is None or (hasattr(pd, "isna") and pd.isna(x)):
            return int(default)
        return int(float(x))
    except:
        return int(default)


def is_num(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def call_with_timeout(func, timeout: float, *args, **kwargs):
    """מריץ פונקציה עם timeout קשיח"""
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(func, *args, **kwargs)
        try:
            return fut.result(timeout=timeout)
        except FuturesTimeout:
            fut.cancel()
            raise RuntimeError(f"timeout {timeout}s")
        except Exception as e:
            fut.cancel()
            raise e


def format_date(timestamp):
    """המרה של UNIX timestamp לתאריך קריא"""
    if timestamp:
        try:
            return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        except:
            return "N/A"
    return "N/A"


def format_number(num, decimals=2):
    """עיצוב מספרים לקריאות"""
    if not is_num(num):
        return "N/A"
    if abs(num) >= 1e9:
        return f"${num / 1e9:.{decimals}f}B"
    elif abs(num) >= 1e6:
        return f"${num / 1e6:.{decimals}f}M"
    elif abs(num) >= 1e3:
        return f"${num / 1e3:.{decimals}f}K"
    return f"${num:.{decimals}f}"


# ============================ אינדיקטורים ============================
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def dema(series: pd.Series, period: int) -> pd.Series:
    e1 = ema(series, period)
    e2 = ema(e1, period)
    return 2 * e1 - e2


def vwma(df: pd.DataFrame, period: int) -> pd.Series:
    pv = df['Close'] * df['Volume']
    return pv.rolling(period).sum() / df['Volume'].rolling(period).sum()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


# ============================ חישוב Bayesian Momentum (Blue) ============================
def calculate_bayesian_momentum(df: pd.DataFrame, length: int = 60, gap_length: int = 20, gap: int = 10) -> tuple:
    """מימוש מדויק של אינדיקטור Bayesian Momentum מ-Pine Script"""
    if len(df) < length + gap + 10:
        return 0.0, False

    source = (df['High'] + df['Low'] + df['Close']) / 3
    fast_len = max(1, length - gap_length)

    # אינדיקטורים איטיים
    e_slow = ema(source, length)
    s_slow = sma(source, length)
    d_slow = dema(source, length)
    v_slow = vwma(df, length)

    # אינדיקטורים מהירים
    e_fast = ema(source, fast_len)
    s_fast = sma(source, fast_len)
    d_fast = dema(source, fast_len)
    v_fast = vwma(df, fast_len)

    def sig_logic(src_series, ref_series, g):
        res = []
        for i in range(len(src_series)):
            if i < g + 10:
                res.append(0)
                continue
            val = 0
            if src_series.iloc[i] >= ref_series.iloc[i - g]:
                val = 1.0
            elif src_series.iloc[i] >= ref_series.iloc[i - g + 1]:
                val = 0.9
            elif src_series.iloc[i] >= ref_series.iloc[i - g + 2]:
                val = 0.8
            elif src_series.iloc[i] >= ref_series.iloc[i - g + 3]:
                val = 0.7
            elif src_series.iloc[i] >= ref_series.iloc[i - g + 4]:
                val = 0.6
            elif src_series.iloc[i] >= ref_series.iloc[i - g + 5]:
                val = 0.5
            elif src_series.iloc[i] >= ref_series.iloc[i - g + 6]:
                val = 0.4
            elif src_series.iloc[i] >= ref_series.iloc[i - g + 7]:
                val = 0.3
            elif src_series.iloc[i] >= ref_series.iloc[i - g + 8]:
                val = 0.2
            elif src_series.iloc[i] >= ref_series.iloc[i - g + 9]:
                val = 0.1
            res.append(val)
        return ema(pd.Series(res), 4)

    # חישוב טרנדים
    t_e_f = sig_logic(source, e_fast, gap)
    t_s_f = sig_logic(source, s_fast, gap)
    t_d_f = sig_logic(source, d_fast, gap)
    t_v_f = sig_logic(source, v_fast, gap)

    t_e_s = sig_logic(source, e_slow, gap)
    t_s_s = sig_logic(source, s_slow, gap)
    t_d_s = sig_logic(source, d_slow, gap)
    t_v_s = sig_logic(source, v_slow, gap)

    likelihood_up = (t_e_f + t_s_f + t_d_f + t_v_f) / 4
    prior_up = (t_e_s + t_s_s + t_d_s + t_v_s) / 4

    l_up = likelihood_up.iloc[-1]
    p_up = prior_up.iloc[-1]
    l_dn = 1 - l_up
    p_dn = 1 - p_up

    denom = (p_up * l_up + p_dn * l_dn)
    posterior_up = (p_up * l_up / denom) if denom != 0 else 0

    # בדיקת Crossover
    if len(prior_up) < 2 or len(likelihood_up) < 2:
        return posterior_up, False

    prev_denom = (prior_up.iloc[-2] * likelihood_up.iloc[-2] + (1 - prior_up.iloc[-2]) * (1 - likelihood_up.iloc[-2]))
    prev_post = (prior_up.iloc[-2] * likelihood_up.iloc[-2] / prev_denom) if prev_denom != 0 else 0

    long_signal = posterior_up > 0.5 and prev_post <= 0.5
    return posterior_up, long_signal


# ============================ איסוף נתונים ============================
def get_all_tickers() -> List[str]:
    """מושך רשימת טיקרים מ-NASDAQ ו-NYSE"""
    try:
        nasdaq_url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
        nyse_url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

        nasdaq = pd.read_csv(StringIO(requests.get(nasdaq_url, timeout=15).text), sep="|")
        nyse = pd.read_csv(StringIO(requests.get(nyse_url, timeout=15).text), sep="|")

        tickers = list(set(
            nasdaq.Symbol.dropna().astype(str).tolist() +
            nyse["ACT Symbol"].dropna().astype(str).tolist()
        ))
        tickers = [t.replace(".", "-") for t in tickers if re.fullmatch(r"[A-Z0-9.-]+", t)]

        logger.info(f"Fetched {len(tickers)} valid tickers")
        return tickers
    except Exception as exc:
        logger.error(f"Ticker fetch failed: {exc}")
        return []


def fetch_history(ticker: str, start: datetime, end: datetime, timeout_s: float = 12) -> Optional[pd.DataFrame]:
    """שליפת היסטוריה עם timeout"""
    try:
        tkr = yf.Ticker(ticker)
        df = call_with_timeout(tkr.history, timeout_s, start=start, end=end)
        if df is not None and not df.empty:
            return df
    except:
        pass
    return None


def get_stock_info(ticker: str) -> dict:
    """שליפת מידע בסיסי על המניה"""
    try:
        tkr = yf.Ticker(ticker)
        info = tkr.info
        return {
            "marketCap": safe_int(info.get("marketCap", 0)),
            "averageVolume": safe_int(info.get("averageVolume", 0)),
            "beta": safe_float(info.get("beta", 0)),
            "shortPercentOfFloat": safe_float(info.get("shortPercentOfFloat", 0)),
            "floatShares": safe_int(info.get("floatShares", 0)),
            "sharesOutstanding": safe_int(info.get("sharesOutstanding", 0)),
            "heldPercentInsiders": safe_float(info.get("heldPercentInsiders", 0)),
            "dividendYield": safe_float(info.get("dividendYield", 0)),
            "exDividendDate": info.get("exDividendDate"),
            "forwardPE": safe_float(info.get("forwardPE", 0)),
            "shortRatio": safe_float(info.get("shortRatio", 0)),
            "longName": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
        }
    except:
        return {}


# ============================ בניית מדדים ============================
def build_metrics(ticker: str) -> Optional[dict]:
    """בונה את כל המדדים הנדרשים לסינון"""
    try:
        end = datetime.now()
        start_1y = end - timedelta(days=365)
        start_5y = end - timedelta(days=5 * 365)

        # שליפת היסטוריה
        df_1y = fetch_history(ticker, start_1y, end, timeout_s=12)
        if df_1y is None or len(df_1y) < 200:
            return None

        df_5y = fetch_history(ticker, start_5y, end, timeout_s=12)
        if df_5y is None or len(df_5y) < 600:
            return None

        # מחיר נוכחי
        price = safe_float(df_1y["Close"].iloc[-1])
        if not is_num(price) or price <= 0:
            return None

        # חישוב אינדיקטורים בסיסיים
        rsi14_val = safe_float(rsi(df_1y["Close"], 14).iloc[-1])
        rsi2_val = safe_float(rsi(df_1y["Close"], 2).iloc[-1])

        ma50 = sma(df_1y["Close"], 50)
        ma150 = sma(df_1y["Close"], 150)
        ma200 = sma(df_1y["Close"], 200)

        ma50_val = safe_float(ma50.iloc[-1])
        ma150_val = safe_float(ma150.iloc[-1])
        ma200_val = safe_float(ma200.iloc[-1])

        # שיפוע MA200
        ma200_prev20 = safe_float(ma200.shift(20).iloc[-1])
        ma200_slope = (ma200_val - ma200_prev20) if is_num(ma200_val) and is_num(ma200_prev20) else np.nan

        # בדיקת עליה של MA200 ב-200 ימים
        ma200_5y = sma(df_5y["Close"], 200)
        ma200_rising_200d = False
        if len(ma200_5y) >= 201:
            ma200_rising_200d = bool(ma200_5y.iloc[-1] > ma200_5y.iloc[-201])

        # ATR
        atr14_val = safe_float(atr(df_1y, 14).iloc[-1])
        atr_pct = (atr14_val / price) * 100 if is_num(atr14_val) and price > 0 else np.nan

        # שפלים שונים
        low_30d = safe_float(df_1y["Low"].tail(30).min())
        low_60d = safe_float(df_1y["Low"].tail(60).min())
        low_90d = safe_float(df_1y["Low"].tail(90).min())
        low_3m = low_90d
        low_6m = safe_float(df_1y["Low"].tail(180).min())
        low_12m = safe_float(df_1y["Low"].min())

        low_3_6_12_min = np.nanmin([v for v in [low_3m, low_6m, low_12m] if is_num(v)])

        # שפל היסטורי 5 שנים
        all_time_low_5y = safe_float(df_5y["Low"].min())

        # שינוי 5 שנים
        chg_5y = np.nan
        if len(df_5y) > 1:
            first = safe_float(df_5y["Close"].iloc[0])
            last = safe_float(df_5y["Close"].iloc[-1])
            if is_num(first) and is_num(last) and first > 0:
                chg_5y = (last - first) / first

        # 52 week high/low
        high_52w = safe_float(df_1y["High"].max())
        low_52w = safe_float(df_1y["Low"].min())

        pct_above_52w_low = ((price / low_52w - 1) * 100) if is_num(low_52w) and low_52w > 0 else np.nan
        pct_below_52w_high = ((high_52w / price - 1) * 100) if is_num(high_52w) and price > 0 else np.nan

        # מידע על החברה
        info = get_stock_info(ticker)
        market_cap = info.get("marketCap", 0)
        avg_volume = info.get("averageVolume", 0)
        avg_dollar_vol = price * avg_volume if is_num(price) else 0

        # Free float
        free_float = info.get("floatShares")
        if free_float is None or free_float == 0:
            shares_out = info.get("sharesOutstanding")
            held_ins = info.get("heldPercentInsiders")
            if shares_out and held_ins:
                free_float = int(shares_out * (1 - held_ins))

        # Bayesian Momentum (Blue Profile)
        posterior_up, blue_signal = calculate_bayesian_momentum(df_1y)

        # Volume עשרה ימים אחרונים
        vol_avg_10d = df_1y['Volume'].tail(10).mean()

        # HV 30 יום
        hv_30d = safe_float(df_1y['Close'].pct_change().tail(30).std() * math.sqrt(252) * 100)

        return {
            "ticker": ticker,
            "price": price,
            "rsi14": rsi14_val,
            "rsi2": rsi2_val,
            "ma50": ma50_val,
            "ma150": ma150_val,
            "ma200": ma200_val,
            "ma200_slope": ma200_slope,
            "ma200_rising_200d": ma200_rising_200d,
            "atr_pct": atr_pct,
            "low_30d": low_30d,
            "low_60d": low_60d,
            "low_90d": low_90d,
            "low_3_6_12_min": low_3_6_12_min,
            "all_time_low_5y": all_time_low_5y,
            "chg_5y": chg_5y,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "pct_above_52w_low": pct_above_52w_low,
            "pct_below_52w_high": pct_below_52w_high,
            "market_cap": market_cap,
            "avg_volume": avg_volume,
            "avg_dollar_vol": avg_dollar_vol,
            "free_float": free_float,
            "beta": info.get("beta", 0),
            "short_float": info.get("shortPercentOfFloat", 0),
            "short_ratio": info.get("shortRatio", 0),
            "vol_avg_10d": vol_avg_10d,
            "hv_30d": hv_30d,
            "blue_posterior": posterior_up,
            "blue_signal": blue_signal,
            # נתונים נוספים ל-PDF
            "company_name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "dividend_yield": info.get("dividendYield", 0),
            "ex_dividend_date": info.get("exDividendDate"),
            "forward_pe": info.get("forwardPE", 0),
        }
    except Exception as e:
        logger.debug(f"Error building metrics for {ticker}: {e}")
        return None


# ============================ קריטריוני סינון - מעודכנים! ============================
def passes_yellow(m: dict) -> bool:
    """פרופיל צהוב - ללא שינוי מהמקור"""
    price = m["price"]
    if not (0.5 <= price <= 10):
        return False
    if m["market_cap"] < 50_000_000:
        return False
    if m["avg_volume"] < 100_000:
        return False
    if not (is_num(m["low_3_6_12_min"]) and price <= m["low_3_6_12_min"] * 1.10):
        return False
    if not (is_num(m["all_time_low_5y"]) and price > m["all_time_low_5y"] * 1.20):
        return False
    if not (m["ma200_rising_200d"] or (is_num(m["chg_5y"]) and m["chg_5y"] > 0.20)):
        return False
    return True


def passes_green(m: dict) -> bool:
    """פרופיל ירוק - מעודכן מ-stock_screener_extended
    שינויים: RSI 35-60 (במקום 25-45)
    """
    price = m["price"]
    if not (0.5 <= price <= 100):
        return False
    if m["market_cap"] < 200_000_000:
        return False
    if m["avg_dollar_vol"] < 3_000_000:
        return False
    if not (is_num(m["low_90d"]) and price <= m["low_90d"] * 1.20):
        return False
    if not (is_num(m["pct_above_52w_low"]) and 15 <= m["pct_above_52w_low"] <= 70):
        return False
    if not (is_num(m["pct_below_52w_high"]) and m["pct_below_52w_high"] <= 40):
        return False
    if not (is_num(m["ma200"]) and price > m["ma200"] and is_num(m["ma200_slope"]) and m["ma200_slope"] > 0):
        return False
    if not (is_num(m["rsi14"]) and 25 <= m["rsi14"] <= 50):
        return False
    if not (is_num(m["atr_pct"]) and 2 <= m["atr_pct"] <= 15):
        return False
    return True


def passes_pink(m: dict) -> bool:
    """פרופיל ורוד - מעודכן מ-stock_screener_extended
    שינויים:
    - pct_below_52w_high <= 35 (במקום 40)
    - RSI 40-55 OR RSI2 <= 10 (במקום רק RSI 40-55)
    - ATR 3-12 (במקום 2-15)
    """
    price = m["price"]
    if not (0.5 <= price <= 40):
        return False
    if m["market_cap"] < 300_000_000:
        return False
    if not (is_num(m["free_float"]) and m["free_float"] >= 20_000_000):
        return False
    if not (is_num(m["avg_dollar_vol"]) and m["avg_dollar_vol"] >= 5_000_000):
        return False
    if not (is_num(m["low_60d"]) and price <= m["low_60d"] * 1.15):
        return False
    if not (is_num(m["ma50"]) and is_num(m["ma150"]) and is_num(m["ma200"]) and
            price > m["ma50"] > m["ma150"] > m["ma200"]):
        return False
    if not (is_num(m["ma200_slope"]) and m["ma200_slope"] > 0):
        return False
    if not (is_num(m["pct_above_52w_low"]) and 20 <= m["pct_above_52w_low"] <= 60):
        return False
    if not (is_num(m["pct_below_52w_high"]) and m["pct_below_52w_high"] <= 35):
        return False
    rsi_ok = (is_num(m["rsi14"]) and 40 <= m["rsi14"] <= 55) or (is_num(m["rsi2"]) and m["rsi2"] <= 10)
    if not rsi_ok:
        return False
    if not (is_num(m["atr_pct"]) and 3 <= m["atr_pct"] <= 12):
        return False
    return True


def passes_black(m: dict) -> bool:
    """פרופיל שחור - Bottom Reversal"""
    if not (2 <= m["price"] <= 50):
        return False
    if not (is_num(m["ma200"]) and m["price"] < m["ma200"] * 0.7):
        return False
    if not (m["rsi14"] < 25):
        return False
    if m["market_cap"] < 50_000_000:
        return False
    # תנודתיות גבוהה
    if not (is_num(m["atr_pct"]) and m["atr_pct"] >= 4):
        return False
    return True


def passes_blue(m: dict) -> bool:
    """פרופיל כחול - Bayesian Momentum + Safety"""
    if not m.get("blue_signal", False):
        return False
    # Safety filters
    if m["market_cap"] < 300_000_000:
        return False
    if m["price"] < 2:
        return False
    return True


# ============================ שמירת תוצאות ============================
def to_df(rows: List[dict]) -> pd.DataFrame:
    """ממיר רשימת מדדים ל-DataFrame"""
    if not rows:
        return pd.DataFrame()

    records = []
    for m in rows:
        records.append({
            "Ticker": m["ticker"],
            "Price": round(m["price"], 2),
            "RSI14": round(m["rsi14"], 2) if is_num(m["rsi14"]) else None,
            "RSI2": round(m["rsi2"], 2) if is_num(m["rsi2"]) else None,
            "ATR%": round(m["atr_pct"], 2) if is_num(m["atr_pct"]) else None,
            "MA50": round(m["ma50"], 2) if is_num(m["ma50"]) else None,
            "MA200": round(m["ma200"], 2) if is_num(m["ma200"]) else None,
            "MarketCap": m["market_cap"],
            "%Above52wLow": round(m["pct_above_52w_low"], 2) if is_num(m["pct_above_52w_low"]) else None,
            "%Below52wHigh": round(m["pct_below_52w_high"], 2) if is_num(m["pct_below_52w_high"]) else None,
            "Blue_Posterior": round(m.get("blue_posterior", 0), 3),
            "HV30d": round(m["hv_30d"], 2) if is_num(m["hv_30d"]) else None
        })

    return pd.DataFrame(records)


def save_intermediate_live(y: List[dict], g: List[dict], p: List[dict], blk: List[dict], blu: List[dict]):
    """שומר קבצי ביניים - מתעדכן באופן שוטף"""
    try:
        if y:
            df_y = to_df(y)
            df_y.to_csv(os.path.join(INTERMEDIATE_DIR, "yellow_live.csv"), index=False)

        if g:
            df_g = to_df(g)
            df_g.to_csv(os.path.join(INTERMEDIATE_DIR, "green_live.csv"), index=False)

        if p:
            df_p = to_df(p)
            df_p.to_csv(os.path.join(INTERMEDIATE_DIR, "pink_live.csv"), index=False)

        if blk:
            df_blk = to_df(blk)
            df_blk.to_csv(os.path.join(INTERMEDIATE_DIR, "black_live.csv"), index=False)

        if blu:
            df_blu = to_df(blu)
            df_blu.to_csv(os.path.join(INTERMEDIATE_DIR, "blue_live.csv"), index=False)

    except Exception as e:
        logger.error(f"Error saving intermediate files: {e}")


def save_final_results(y: List[dict], g: List[dict], p: List[dict], blk: List[dict], blu: List[dict]):
    """שומר קבצים סופיים"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV files
    df_y = to_df(y)
    df_g = to_df(g)
    df_p = to_df(p)
    df_blk = to_df(blk)
    df_blu = to_df(blu)

    df_y.to_csv(os.path.join(OUTPUTS_DIR, f"yellow_results_{timestamp}.csv"), index=False)
    df_g.to_csv(os.path.join(OUTPUTS_DIR, f"green_results_{timestamp}.csv"), index=False)
    df_p.to_csv(os.path.join(OUTPUTS_DIR, f"pink_results_{timestamp}.csv"), index=False)
    df_blk.to_csv(os.path.join(OUTPUTS_DIR, f"black_results_{timestamp}.csv"), index=False)
    df_blu.to_csv(os.path.join(OUTPUTS_DIR, f"blue_results_{timestamp}.csv"), index=False)

    # Excel file
    excel_path = os.path.join(OUTPUTS_DIR, f"screener_results_{timestamp}.xlsx")
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        profiles = [
            ("Yellow", df_y, "#FFF59D", "Yellow - Oversold Near Low"),
            ("Green", df_g, "#C8E6C9", "Green - RSI 35-60 Trend"),
            ("Pink", df_p, "#F8BBD0", "Pink - Strict Momentum"),
            ("Black", df_blk, "#424242", "Black - Bottom Reversal"),
            ("Blue", df_blu, "#BBDEFB", "Blue - Bayesian Momentum")
        ]

        for name, df, color, title in profiles:
            if df.empty:
                df = pd.DataFrame({"Status": ["No results found"]})

            df.to_excel(writer, sheet_name=name, index=False, startrow=1)

            workbook = writer.book
            worksheet = writer.sheets[name]

            # Format header
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': color,
                'border': 1,
                'align': 'center'
            })

            title_format = workbook.add_format({
                'bold': True,
                'font_size': 14,
                'bg_color': color,
                'border': 1,
                'align': 'center'
            })

            # Write title
            worksheet.merge_range(0, 0, 0, len(df.columns) - 1, title, title_format)

            # Format columns
            for col_num, col_name in enumerate(df.columns):
                worksheet.write(1, col_num, col_name, header_format)
                worksheet.set_column(col_num, col_num, 15)

    logger.info(f"Final results saved with timestamp {timestamp}")
    logger.info(
        f"Yellow: {len(df_y)}, Green: {len(df_g)}, Pink: {len(df_p)}, Black: {len(df_blk)}, Blue: {len(df_blu)}")

    return timestamp


# ============================ PDF Generation ============================
class FinvizChartDownloader:
    """מנהל הורדת גרפים מ-Finviz עם rate limiting"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.last_request_time = 0
        self.request_count = 0

    def _wait_for_rate_limit(self):
        """המתנה חכמה בין בקשות"""
        elapsed = time.time() - self.last_request_time
        delay = random.uniform(CONFIG["FINVIZ_DELAY_MIN"], CONFIG["FINVIZ_DELAY_MAX"])

        # הגדלת השהיה כל 20 בקשות
        if self.request_count > 0 and self.request_count % CONFIG["FINVIZ_BATCH_SIZE"] == 0:
            delay += CONFIG["FINVIZ_BATCH_DELAY"]
            logger.info(f"Rate limit pause: waiting {delay:.1f}s after {self.request_count} charts")

        if elapsed < delay:
            time.sleep(delay - elapsed)

        self.last_request_time = time.time()
        self.request_count += 1

    def download_chart(self, ticker: str) -> Optional[str]:
        """הורדת גרף יחיד"""
        self._wait_for_rate_limit()

        url = f"https://charts.finviz.com/chart.ashx?t={ticker}&ty=c&ta=1&p=d&s=l"

        try:
            response = self.session.get(url, timeout=15)

            if response.status_code == 200:
                image_path = os.path.join(CHARTS_DIR, f"chart_{ticker}.png")
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                return image_path
            elif response.status_code == 429:
                # Rate limited - wait longer
                logger.warning(f"Rate limited by Finviz, waiting 60s...")
                time.sleep(60)
                return self.download_chart(ticker)  # Retry
            else:
                logger.warning(f"Failed to download chart for {ticker}: HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"Error downloading chart for {ticker}: {e}")

        return None


def calculate_volatility_from_metrics(m: dict) -> str:
    """חישוב תנודתיות היסטורית מהמדדים"""
    hv = m.get("hv_30d")
    if is_num(hv):
        return f"{hv:.2f}%"
    return "N/A"


def generate_pdf_for_color(color_name: str, stocks: List[dict], timestamp: str,
                           downloader: FinvizChartDownloader) -> Optional[str]:
    """יצירת PDF עם גרפים ונתונים לצבע מסוים"""

    if not stocks:
        logger.info(f"No stocks for {color_name}, skipping PDF generation")
        return None

    # הגבלת מספר הגרפים
    stocks_to_process = stocks[:CONFIG["MAX_CHARTS_PER_COLOR"]]

    # הגדרת צבעים לכל פרופיל
    color_map = {
        "Yellow": {"bg": HexColor("#FFF59D"), "accent": HexColor("#F9A825")},
        "Green": {"bg": HexColor("#C8E6C9"), "accent": HexColor("#2E7D32")},
        "Pink": {"bg": HexColor("#F8BBD0"), "accent": HexColor("#C2185B")},
        "Black": {"bg": HexColor("#616161"), "accent": HexColor("#212121")},
        "Blue": {"bg": HexColor("#BBDEFB"), "accent": HexColor("#1565C0")},
    }

    colors = color_map.get(color_name, {"bg": HexColor("#EEEEEE"), "accent": black})

    pdf_filename = os.path.join(OUTPUTS_DIR, f"{color_name}_Charts_{timestamp}.pdf")
    width, height = A4
    c = canvas.Canvas(pdf_filename, pagesize=A4)

    logger.info(f"Generating {color_name} PDF with {len(stocks_to_process)} stocks...")

    for i, m in enumerate(stocks_to_process):
        ticker = m["ticker"]
        logger.info(f"  [{i + 1}/{len(stocks_to_process)}] Processing chart for {ticker}")

        # כותרת
        c.setFillColor(colors["accent"])
        c.setFont("Helvetica-Bold", 24)
        c.drawCentredString(width / 2, height - 40, f"{ticker}")

        c.setFillColor(darkgrey)
        c.setFont("Helvetica", 12)
        company_sector = f"{m.get('company_name', ticker)[:40]} | {m.get('sector', 'N/A')}"
        c.drawCentredString(width / 2, height - 60, company_sector)

        # צבע פרופיל
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(colors["accent"])
        c.drawCentredString(width / 2, height - 78, f"Profile: {color_name}")

        # גרף
        image_path = downloader.download_chart(ticker)
        if image_path and os.path.exists(image_path):
            try:
                c.drawImage(image_path, 20, height - 430, width=550, height=330)
                os.remove(image_path)  # ניקוי
            except Exception as e:
                logger.error(f"Error drawing chart for {ticker}: {e}")
        else:
            # גרף לא זמין - הצגת הודעה
            c.setFont("Helvetica", 14)
            c.setFillColor(darkgrey)
            c.drawCentredString(width / 2, height - 280, "Chart not available")

        # טבלת נתונים
        y_start = height - 460
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(colors["accent"])
        c.drawString(40, y_start, "Stock Data:")

        c.setFont("Helvetica", 10)

        # הגדרת עמודות
        col_1 = 40
        col_2 = 210
        col_3 = 380
        row_gap = 20
        y_pos = y_start - 28

        # עיצוב הנתונים
        def draw_data_row(label1, val1, label2, val2, label3, val3, y):
            c.setFillColor(darkgrey)
            c.drawString(col_1, y, f"{label1}:")
            c.setFillColor(black)
            c.drawString(col_1 + 65, y, str(val1))

            c.setFillColor(darkgrey)
            c.drawString(col_2, y, f"{label2}:")
            c.setFillColor(black)
            c.drawString(col_2 + 65, y, str(val2))

            c.setFillColor(darkgrey)
            c.drawString(col_3, y, f"{label3}:")
            c.setFillColor(black)
            c.drawString(col_3 + 65, y, str(val3))

        # שורה 1: מחיר, Beta, Market Cap
        draw_data_row(
            "Price", f"${m['price']:.2f}",
            "Beta", f"{m.get('beta', 'N/A')}",
            "Market Cap", format_number(m['market_cap']),
            y_pos
        )
        y_pos -= row_gap

        # שורה 2: RSI, ATR, HV
        draw_data_row(
            "RSI(14)", f"{m['rsi14']:.1f}" if is_num(m['rsi14']) else "N/A",
            "RSI(2)", f"{m['rsi2']:.1f}" if is_num(m['rsi2']) else "N/A",
            "ATR%", f"{m['atr_pct']:.2f}%" if is_num(m['atr_pct']) else "N/A",
            y_pos
        )
        y_pos -= row_gap

        # שורה 3: תנודתיות ושורטים
        draw_data_row(
            "30d HV", calculate_volatility_from_metrics(m),
            "Avg Vol", f"{m.get('avg_volume', 0):,}",
            "Short Ratio", f"{m.get('short_ratio', 'N/A')}",
            y_pos
        )
        y_pos -= row_gap

        # שורה 4: Short Float, Dividend
        div_yield = m.get('dividend_yield', 0)
        if is_num(div_yield) and div_yield > 0:
            div_str = f"{div_yield * 100:.2f}%" if div_yield < 1 else f"{div_yield:.2f}%"
        else:
            div_str = "0.00%"

        ex_div = format_date(m.get('ex_dividend_date'))

        draw_data_row(
            "Short Float", f"{m.get('short_float', 0) * 100:.2f}%" if is_num(m.get('short_float')) else "N/A",
            "Ex-Div Date", ex_div,
            "Div Yield", div_str,
            y_pos
        )
        y_pos -= row_gap

        # שורה 5: 52W Range, Forward P/E
        draw_data_row(
            "52W High", f"${m['high_52w']:.2f}" if is_num(m['high_52w']) else "N/A",
            "52W Low", f"${m['low_52w']:.2f}" if is_num(m['low_52w']) else "N/A",
            "Fwd P/E", f"{m.get('forward_pe', 'N/A')}",
            y_pos
        )
        y_pos -= row_gap

        # שורה 6: מיקום יחסית ל-52W
        draw_data_row(
            "%Above Low", f"{m['pct_above_52w_low']:.1f}%" if is_num(m['pct_above_52w_low']) else "N/A",
            "%Below High", f"{m['pct_below_52w_high']:.1f}%" if is_num(m['pct_below_52w_high']) else "N/A",
            "MA200 Slope", f"{m['ma200_slope']:.3f}" if is_num(m['ma200_slope']) else "N/A",
            y_pos
        )

        # פרטים נוספים לפרופיל Blue
        if color_name == "Blue" and is_num(m.get('blue_posterior')):
            y_pos -= row_gap
            c.setFillColor(colors["accent"])
            c.setFont("Helvetica-Bold", 10)
            c.drawString(col_1, y_pos, f"Bayesian Posterior: {m['blue_posterior']:.3f}")

        # Footer
        c.setFont("Helvetica", 8)
        c.setFillColor(darkgrey)
        c.drawCentredString(width / 2, 30,
                            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Page {i + 1}/{len(stocks_to_process)}")

        c.showPage()

    c.save()
    logger.info(f"PDF saved: {pdf_filename}")
    return pdf_filename


def cleanup_charts_dir():
    """ניקוי תיקיית הגרפים הזמנית"""
    try:
        for f in os.listdir(CHARTS_DIR):
            os.remove(os.path.join(CHARTS_DIR, f))
    except:
        pass


# ============================ Main ============================
def main():
    logger.info("=" * 70)
    logger.info("=== StockSentinel V3k: The Prism Multi-Profile Engine ===")
    logger.info("=" * 70)
    logger.info("Features:")
    logger.info("  • 5 Profiles: Yellow, Green, Pink, Black, Blue")
    logger.info("  • Full ticker processing")
    logger.info("  • PDF generation with Finviz charts")
    logger.info("  • Smart rate limiting to avoid blocks")
    logger.info("=" * 70)

    # קבלת רשימת טיקרים
    tickers = get_all_tickers()
    if not tickers:
        logger.error("Failed to fetch tickers")
        return

    logger.info(f"Total tickers to process: {len(tickers)}")

    # תוצאות
    results_yellow = []
    results_green = []
    results_pink = []
    results_black = []
    results_blue = []

    # סטטיסטיקות
    processed = 0
    errors = 0
    start_time = datetime.now()

    #  עיבוד כל הטיקרים - לא רק 1000
    total_tickers = len(tickers)

    for idx, ticker in enumerate(tickers, 1):
        logger.info(f"[{idx}/{total_tickers}] Processing: {ticker}")

        try:
            m = build_metrics(ticker)
            processed += 1

            if m:
                # בדיקת כל הפרופילים
                if passes_yellow(m):
                    results_yellow.append(m)
                    logger.info(f"  ✓ YELLOW: {ticker}")

                if passes_green(m):
                    results_green.append(m)
                    logger.info(f"  ✓ GREEN: {ticker}")

                if passes_pink(m):
                    results_pink.append(m)
                    logger.info(f"  ✓ PINK: {ticker}")

                if passes_black(m):
                    results_black.append(m)
                    logger.info(f"  ✓ BLACK: {ticker}")

                if passes_blue(m):
                    results_blue.append(m)
                    logger.info(f"  ✓ BLUE: {ticker}")

            # עדכון intermediate כל N מניות
            if processed % CONFIG["SAVE_EVERY_N"] == 0:
                save_intermediate_live(results_yellow, results_green, results_pink, results_black, results_blue)

            # סטטיסטיקות כל 200 מניות
            if processed % 200 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = processed / elapsed if elapsed > 0 else 0
                eta_seconds = (total_tickers - idx) / rate if rate > 0 else 0
                eta_minutes = eta_seconds / 60

                logger.info(f"Progress: {idx}/{total_tickers} ({100 * idx / total_tickers:.1f}%) | "
                            f"Rate: {rate:.1f} tickers/sec | "
                            f"ETA: {eta_minutes:.1f} min | "
                            f"Results: Y={len(results_yellow)}, G={len(results_green)}, "
                            f"P={len(results_pink)}, B={len(results_black)}, Bl={len(results_blue)}")

            # השהיה קצרה למניעת עומס
            time.sleep(CONFIG["TICKER_DELAY"])

        except Exception as e:
            errors += 1
            logger.debug(f"Error processing {ticker}: {e}")
            continue

    # שמירת תוצאות סופיות
    logger.info("=" * 70)
    logger.info("Scan complete! Saving final results...")
    timestamp = save_final_results(results_yellow, results_green, results_pink, results_black, results_blue)

    # יצירת PDFs עם גרפים
    logger.info("=" * 70)
    logger.info("Generating PDF reports with charts...")

    downloader = FinvizChartDownloader()

    pdf_files = []
    color_results = [
        ("Yellow", results_yellow),
        ("Green", results_green),
        ("Pink", results_pink),
        ("Black", results_black),
        ("Blue", results_blue),
    ]

    for color_name, stocks in color_results:
        if stocks:
            pdf_path = generate_pdf_for_color(color_name, stocks, timestamp, downloader)
            if pdf_path:
                pdf_files.append(pdf_path)

    # ניקוי
    cleanup_charts_dir()

    # סיכום
    total_time = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 70)
    logger.info("SCAN SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total processed: {processed}/{total_tickers}")
    logger.info(f"Errors: {errors}")
    logger.info(f"Total time: {total_time / 60:.1f} minutes")
    logger.info(f"Average rate: {processed / total_time:.1f} tickers/sec")
    logger.info("")
    logger.info("RESULTS BY PROFILE:")
    logger.info(f"  Yellow (Near Low, Oversold):      {len(results_yellow)}")
    logger.info(f"  Green (Common Up Going Trend):          {len(results_green)}")
    logger.info(f"  Pink (Strict Momentum):           {len(results_pink)}")
    logger.info(f"  Black (Bottom Reversal):          {len(results_black)}")
    logger.info(f"  Blue (Bayesian Momentum):         {len(results_blue)}")
    logger.info("")
    logger.info(f"PDF Reports generated: {len(pdf_files)}")
    for pdf in pdf_files:
        logger.info(f"  • {pdf}")
    logger.info("=" * 70)

    if not any([results_yellow, results_green, results_pink, results_black, results_blue]):
        logger.info("No opportunities found matching any criteria")
    else:
        logger.info("✓ All results saved successfully!")


if __name__ == "__main__":
    main()