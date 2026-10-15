"""
NAS100 Daily Bias Engine
-------------------------
Pulls real market data via yfinance (free, no API key) and computes a
weighted, disciplined daily bias (BUY / SELL / NEUTRAL) with a confidence
score. Every number shown either comes from live/historical price data or
is explicitly labeled as unavailable -- nothing is invented.

This is a decision-support tool, not financial advice, and it does not
place trades. Treat its output as one input among many.
"""

from __future__ import annotations
import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

TICKERS = {
    "nas100": "NQ=F",       # E-mini Nasdaq-100 futures (primary instrument)
    "sp500": "ES=F",        # E-mini S&P 500 futures
    "dow": "YM=F",          # E-mini Dow futures
    "vix": "^VIX",
    "dxy": "DX-Y.NYB",
    "us10y": "^TNX",        # quoted x10, e.g. 47.8 = 4.78%
    "gold": "GC=F",
    "oil": "CL=F",
    "nvda": "NVDA",
    "smh": "SMH",           # semiconductor ETF, sector proxy
}

# EDIT THIS MANUALLY each week, or wire up a real economic-calendar API
# (Finnhub / FMP / Polygon all have free tiers for this specific endpoint).
# Format: "YYYY-MM-DD": ["event description", ...]
MANUAL_HIGH_IMPACT_EVENTS: dict[str, list[str]] = {
    # "2026-09-04": ["US Nonfarm Payrolls (8:30 AM ET)"],
    # "2026-09-17": ["FOMC Rate Decision (2:00 PM ET)"],
}

WEIGHTS = {
    "trend": 0.20,          # EMA stack / higher-timeframe structure
    "momentum": 0.15,       # RSI regime + recent price action
    "gap_context": 0.10,    # overnight gap vs prior range
    "cross_market": 0.20,   # ES/VIX/10Y/DXY/semis confirmation
    "historical": 0.25,     # backtested similar-day outcome stats
    "volatility_context": 0.10,  # ATR regime / VIX level sanity check
}

# ---------------------------------------------------------------------------
# DATA CONTAINERS
# ---------------------------------------------------------------------------

@dataclass
class InstrumentSnapshot:
    ticker: str
    last: Optional[float] = None
    prev_close: Optional[float] = None
    change_pct: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    as_of: Optional[str] = None
    ok: bool = False
    error: Optional[str] = None


@dataclass
class BiasResult:
    bias: str
    confidence: float
    bull_score: float
    bear_score: float
    unscored_weight: float
    component_scores: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# DATA FETCHING
# ---------------------------------------------------------------------------

def fetch_snapshot(ticker: str) -> InstrumentSnapshot:
    """Fetch a live/delayed snapshot for one ticker. Never raises -- returns
    ok=False with an error message on failure so the dashboard degrades
    gracefully instead of crashing."""
    snap = InstrumentSnapshot(ticker=ticker)
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d", interval="1d")
        if hist.empty:
            snap.error = "No historical data returned"
            return snap

        intraday = t.history(period="1d", interval="5m")
        if not intraday.empty:
            snap.last = float(intraday["Close"].dropna().iloc[-1])
            snap.day_high = float(intraday["High"].max())
            snap.day_low = float(intraday["Low"].min())
            snap.as_of = str(intraday.index[-1])
        else:
            snap.last = float(hist["Close"].iloc[-1])
            snap.day_high = float(hist["High"].iloc[-1])
            snap.day_low = float(hist["Low"].iloc[-1])
            snap.as_of = str(hist.index[-1])

        snap.prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else None
        if snap.prev_close:
            snap.change_pct = (snap.last - snap.prev_close) / snap.prev_close * 100
        snap.ok = True
    except Exception as e:  # noqa: BLE001
        snap.error = str(e)
    return snap


def fetch_daily_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Daily OHLCV history for technicals + historical pattern matching."""
    try:
        df = yf.Ticker(ticker).history(period=period, interval="1d")
        return df.dropna()
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# TECHNICALS
# ---------------------------------------------------------------------------

def compute_technicals(df: pd.DataFrame) -> dict:
    """EMA stack, RSI(14), ATR(14) from real daily history. Returns {} if
    there isn't enough data -- caller must treat that as 'unavailable', not
    zero."""
    if df.empty or len(df) < 210:
        return {}

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).iloc[-1]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean().iloc[-1]

    last_close = close.iloc[-1]
    trend = "up" if last_close > ema20 > ema50 > ema200 else (
        "down" if last_close < ema20 < ema50 < ema200 else "mixed"
    )

    return {
        "ema20": float(ema20), "ema50": float(ema50), "ema200": float(ema200),
        "rsi14": float(rsi) if pd.notna(rsi) else None,
        "atr14": float(atr14) if pd.notna(atr14) else None,
        "trend": trend,
        "last_close": float(last_close),
    }


# ---------------------------------------------------------------------------
# HISTORICAL PATTERN MATCHING (genuinely backtested, not invented)
# ---------------------------------------------------------------------------

def historical_pattern_match(df: pd.DataFrame, gap_pct: float, atr_pct: float,
                              tolerance_gap: float = 0.5,
                              tolerance_atr: float = 0.5) -> dict:
    """
    Finds days in `df` whose overnight gap (%) and ATR-as-%-of-price were
    similar to today's, then reports what actually happened the NEXT day.
    This is a real backtest over available history -- if the sample is too
    small, that is reported honestly instead of padding the numbers.
    """
    if df.empty or len(df) < 60:
        return {"available": False, "reason": "Not enough daily history fetched."}

    d = df.copy()
    d["prev_close"] = d["Close"].shift(1)
    d["gap_pct"] = (d["Open"] - d["prev_close"]) / d["prev_close"] * 100
    d["range_pct"] = (d["High"] - d["Low"]) / d["Close"] * 100
    d["next_ret_pct"] = d["Close"].shift(-1).sub(d["Close"]).div(d["Close"]) * 100
    d = d.dropna(subset=["gap_pct", "range_pct", "next_ret_pct"])

    matches = d[
        (d["gap_pct"].sub(gap_pct).abs() <= tolerance_gap) &
        (d["range_pct"].sub(atr_pct).abs() <= tolerance_atr)
    ]

    n = len(matches)
    if n < 8:
        return {
            "available": False,
            "reason": f"Only {n} historically similar sessions found "
                      f"(need >= 8 for a usable sample). "
                      f"Insufficient historical data to calculate reliable probability.",
            "sample_size": n,
        }

    bullish = int((matches["next_ret_pct"] > 0.05).sum())
    bearish = int((matches["next_ret_pct"] < -0.05).sum())
    neutral = n - bullish - bearish
    avg_ret = float(matches["next_ret_pct"].mean())
    mfe = float(matches["next_ret_pct"].max())
    mae = float(matches["next_ret_pct"].min())

    return {
        "available": True,
        "sample_size": n,
        "bullish": bullish,
        "bearish": bearish,
        "neutral": neutral,
        "bullish_prob": bullish / n * 100,
        "bearish_prob": bearish / n * 100,
        "avg_return_pct": avg_ret,
        "max_favorable_pct": mfe,
        "max_adverse_pct": mae,
    }


# ---------------------------------------------------------------------------
# BIAS ENGINE
# ---------------------------------------------------------------------------

def score_component(value: Optional[float], bullish_if_positive: bool = True) -> Optional[float]:
    """Maps a signed signal to a 0-100 bullishness score. Returns None (not
    0) when the underlying value is unavailable, so it can be excluded from
    the weighted average rather than silently dragging it toward 50."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    score = 50 + (value if bullish_if_positive else -value) * 10
    return float(np.clip(score, 0, 100))


def build_daily_bias(nas_snap: InstrumentSnapshot,
                      es_snap: InstrumentSnapshot,
                      vix_snap: InstrumentSnapshot,
                      us10y_snap: InstrumentSnapshot,
                      dxy_snap: InstrumentSnapshot,
                      smh_snap: InstrumentSnapshot,
                      technicals: dict,
                      hist_pattern: dict,
                      today_events: list[str]) -> BiasResult:

    components: dict[str, Optional[float]] = {}
    notes: list[str] = []

    # --- Trend (EMA stack) ---
    if technicals:
        trend = technicals["trend"]
        components["trend"] = {"up": 75, "down": 25, "mixed": 50}[trend]
        notes.append(f"EMA stack (20/50/200) indicates a '{trend}' higher-timeframe trend.")
    else:
        components["trend"] = None
        notes.append("Trend: insufficient daily history to compute EMA stack.")

    # --- Momentum (RSI) ---
    rsi = technicals.get("rsi14") if technicals else None
    if rsi is not None:
        if rsi > 70:
            components["momentum"] = 35  # overbought -> mean-reversion risk, not purely bullish
            notes.append(f"RSI(14) = {rsi:.1f} — overbought, raises mean-reversion risk.")
        elif rsi < 30:
            components["momentum"] = 65  # oversold -> mean-reversion bounce risk to shorts
            notes.append(f"RSI(14) = {rsi:.1f} — oversold, raises bounce risk for shorts.")
        else:
            components["momentum"] = float(np.clip(rsi, 0, 100))
            notes.append(f"RSI(14) = {rsi:.1f} — neutral momentum regime.")
    else:
        components["momentum"] = None
        notes.append("Momentum: RSI unavailable.")

    # --- Gap context ---
    if nas_snap.ok and nas_snap.prev_close:
        gap_pct = (nas_snap.last - nas_snap.prev_close) / nas_snap.prev_close * 100
        components["gap_context"] = score_component(gap_pct, bullish_if_positive=True)
        notes.append(f"Gap vs prior close: {gap_pct:+.2f}%.")
    else:
        components["gap_context"] = None
        gap_pct = None
        notes.append("Gap context: live NQ snapshot unavailable.")

    # --- Cross-market confirmation ---
    confirms = []
    if es_snap.ok and es_snap.change_pct is not None:
        confirms.append(1 if es_snap.change_pct > 0 else -1)
    if vix_snap.ok and vix_snap.change_pct is not None:
        confirms.append(-1 if vix_snap.change_pct > 0 else 1)  # VIX up = bearish for equities
    if us10y_snap.ok and us10y_snap.change_pct is not None:
        confirms.append(-1 if us10y_snap.change_pct > 0 else 1)  # rising yields = bearish for NQ
    if dxy_snap.ok and dxy_snap.change_pct is not None:
        confirms.append(-1 if dxy_snap.change_pct > 0 else 1)  # stronger dollar = mild headwind
    if smh_snap.ok and smh_snap.change_pct is not None:
        confirms.append(1 if smh_snap.change_pct > 0 else -1)

    if confirms:
        avg_confirm = sum(confirms) / len(confirms)
        components["cross_market"] = float(np.clip(50 + avg_confirm * 25, 0, 100))
        agree = sum(1 for c in confirms if c == (1 if avg_confirm > 0 else -1))
        notes.append(f"Cross-market: {agree}/{len(confirms)} signals agree on direction "
                      f"({'bullish' if avg_confirm > 0 else 'bearish' if avg_confirm < 0 else 'mixed'}).")
    else:
        components["cross_market"] = None
        notes.append("Cross-market: no confirming instruments available.")

    # --- Historical pattern matching ---
    if hist_pattern.get("available"):
        components["historical"] = hist_pattern["bullish_prob"]
        notes.append(
            f"Historical match: {hist_pattern['sample_size']} similar sessions -> "
            f"{hist_pattern['bullish_prob']:.0f}% bullish / {hist_pattern['bearish_prob']:.0f}% bearish "
            f"next day, avg return {hist_pattern['avg_return_pct']:+.2f}%."
        )
    else:
        components["historical"] = None
        notes.append(f"Historical match: {hist_pattern.get('reason', 'unavailable')}")

    # --- Volatility context (sanity check, not directional) ---
    if vix_snap.ok and vix_snap.last is not None:
        if vix_snap.last > 25:
            components["volatility_context"] = 40
            notes.append(f"VIX = {vix_snap.last:.1f} — elevated, widen expected ranges / lower confidence.")
        elif vix_snap.last < 13:
            components["volatility_context"] = 55
            notes.append(f"VIX = {vix_snap.last:.1f} — low, complacent regime.")
        else:
            components["volatility_context"] = 50
            notes.append(f"VIX = {vix_snap.last:.1f} — normal regime.")
    else:
        components["volatility_context"] = None

    # --- Event risk flag (manual calendar) ---
    if today_events:
        notes.append("Scheduled high-impact events today: " + "; ".join(today_events) +
                      " — expect a volatility catalyst around release time.")

    # --- Weighted aggregation, excluding unscored components ---
    used_weight = 0.0
    weighted_sum = 0.0
    for key, weight in WEIGHTS.items():
        val = components.get(key)
        if val is not None:
            weighted_sum += val * weight
            used_weight += weight

    if used_weight == 0:
        return BiasResult(
            bias="NEUTRAL", confidence=0, bull_score=0, bear_score=0,
            unscored_weight=1.0, component_scores=components,
            notes=notes + ["No components could be scored -- data unavailable."],
        )

    bull_score = weighted_sum / used_weight  # 0-100
    bear_score = 100 - bull_score
    unscored_weight = 1.0 - used_weight

    # Confidence is discounted by how much of the framework had no data
    raw_confidence = abs(bull_score - 50) * 2  # 0-100
    confidence = raw_confidence * used_weight  # penalize missing data

    if unscored_weight > 0.5 or confidence < 30:
        bias = "NEUTRAL"
    elif bull_score >= 60:
        bias = "BUY"
    elif bull_score <= 40:
        bias = "SELL"
    else:
        bias = "NEUTRAL"

    return BiasResult(
        bias=bias,
        confidence=round(confidence, 1),
        bull_score=round(bull_score, 1),
        bear_score=round(bear_score, 1),
        unscored_weight=round(unscored_weight, 2),
        component_scores=components,
        notes=notes,
    )
