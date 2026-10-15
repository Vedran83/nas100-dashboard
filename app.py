"""
NAS100 Daily Bias Dashboard — colorful, split-screen UI
---------------------------------------------------------
Left side: the bias, big and color-coded.
Right side: the reasons behind it, in plain language.

Deploy free on Streamlit Community Cloud -- see README.md. Every visit
re-pulls live data and recomputes the bias, so there's nothing to
schedule; just open the URL each morning.

DISCLAIMER: Educational decision-support tool. Not financial advice.
Does not place trades and does not guarantee outcomes.
"""

import datetime as dt
import pytz
import streamlit as st

from engine import (
    TICKERS, MANUAL_HIGH_IMPACT_EVENTS,
    fetch_snapshot, fetch_daily_history, compute_technicals,
    historical_pattern_match, build_daily_bias,
)

st.set_page_config(page_title="NAS100 Daily Bias", page_icon="📊", layout="wide")

NY_TZ = pytz.timezone("America/New_York")

# ---------------------------------------------------------------------------
# STYLE
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #0f1116 0%, #14171f 100%); }
    #MainMenu, footer, header {visibility: hidden;}

    .topbar {
        display: flex; justify-content: space-between; align-items: center;
        padding: 6px 0 18px 0;
    }
    .topbar h1 { font-size: 1.6rem; margin: 0; color: #f0f2f6; }
    .topbar .sub { color: #8b93a7; font-size: 0.85rem; }

    .ticker-strip {
        display: flex; gap: 10px; overflow-x: auto; padding-bottom: 14px;
    }
    .ticker-chip {
        background: #1b1f2a; border: 1px solid #2a2f3d; border-radius: 10px;
        padding: 8px 14px; min-width: 92px; text-align: center; flex-shrink: 0;
    }
    .ticker-chip .name { font-size: 0.7rem; color: #8b93a7; letter-spacing: 0.03em; }
    .ticker-chip .val { font-size: 0.95rem; font-weight: 700; color: #f0f2f6; }
    .ticker-chip .chg-up { color: #35d488; font-size: 0.8rem; font-weight: 600; }
    .ticker-chip .chg-down { color: #ff5c5c; font-size: 0.8rem; font-weight: 600; }
    .ticker-chip .chg-flat { color: #8b93a7; font-size: 0.8rem; font-weight: 600; }

    .bias-card {
        border-radius: 20px; padding: 34px 28px; text-align: center;
        box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    }
    .bias-card.buy { background: linear-gradient(160deg, #0d3d2a 0%, #0a2a1e 100%); border: 1px solid #1e8a5f; }
    .bias-card.sell { background: linear-gradient(160deg, #4a1414 0%, #2e0d0d 100%); border: 1px solid #b23a3a; }
    .bias-card.neutral { background: linear-gradient(160deg, #4a3a0d 0%, #2e250a 100%); border: 1px solid #c99a2e; }

    .bias-label { font-size: 0.9rem; letter-spacing: 0.15em; color: #b8bfcf; margin-bottom: 6px; }
    .bias-word { font-size: 3.4rem; font-weight: 800; letter-spacing: 0.02em; margin: 0; }
    .bias-word.buy { color: #35d488; }
    .bias-word.sell { color: #ff5c5c; }
    .bias-word.neutral { color: #f0b429; }

    .confidence-wrap { margin-top: 22px; }
    .confidence-track { background: #10131a; border-radius: 8px; height: 14px; overflow: hidden; }
    .confidence-fill.buy { background: linear-gradient(90deg, #1e8a5f, #35d488); height: 100%; }
    .confidence-fill.sell { background: linear-gradient(90deg, #b23a3a, #ff5c5c); height: 100%; }
    .confidence-fill.neutral { background: linear-gradient(90deg, #c99a2e, #f0b429); height: 100%; }
    .confidence-num { font-size: 1.6rem; font-weight: 700; color: #f0f2f6; margin-top: 8px; }
    .confidence-caption { font-size: 0.78rem; color: #8b93a7; }

    .score-row { display: flex; justify-content: space-between; margin-top: 26px; gap: 10px; }
    .score-box { flex: 1; background: rgba(255,255,255,0.04); border-radius: 12px; padding: 10px; }
    .score-box .lbl { font-size: 0.7rem; color: #8b93a7; }
    .score-box .num { font-size: 1.3rem; font-weight: 700; }
    .score-box.bull .num { color: #35d488; }
    .score-box.bear .num { color: #ff5c5c; }

    .reasons-title { color: #f0f2f6; font-size: 1.1rem; font-weight: 700; margin-bottom: 4px; }
    .reasons-sub { color: #8b93a7; font-size: 0.82rem; margin-bottom: 16px; }

    .reason-card {
        background: #1b1f2a; border-left: 4px solid #3a4258; border-radius: 8px;
        padding: 12px 16px; margin-bottom: 10px; color: #dfe3ec; font-size: 0.92rem;
        line-height: 1.4;
    }
    .reason-card.pos { border-left-color: #35d488; }
    .reason-card.neg { border-left-color: #ff5c5c; }
    .reason-card.warn { border-left-color: #f0b429; }
    .reason-card.neutral { border-left-color: #6c7488; }

    .missing-note {
        background: rgba(240,180,41,0.08); border: 1px dashed #f0b429; border-radius: 10px;
        padding: 12px 16px; color: #f0d090; font-size: 0.85rem; margin-top: 6px;
    }

    .section-title { color: #f0f2f6; font-size: 1rem; font-weight: 700; margin: 22px 0 10px 0; }
    .footer-note { color: #6c7488; font-size: 0.78rem; text-align: center; padding-top: 24px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_all():
    snaps = {name: fetch_snapshot(ticker) for name, ticker in TICKERS.items()}
    nas_daily = fetch_daily_history(TICKERS["nas100"], period="2y")
    technicals = compute_technicals(nas_daily)
    return snaps, nas_daily, technicals


now_ny = dt.datetime.now(NY_TZ)

top_l, top_r = st.columns([3, 1])
with top_l:
    st.markdown(f"""
    <div class="topbar">
        <div>
            <h1>📊 NAS100 Daily Bias</h1>
            <div class="sub">{now_ny.strftime('%A, %B %d, %Y — %I:%M %p %Z')} · live via Yahoo Finance</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with top_r:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with st.spinner("Pulling live market data..."):
    snaps, nas_daily, technicals = load_all()

nas = snaps["nas100"]

# ---------------------------------------------------------------------------
# TICKER STRIP (compact cross-market context, NAS100 stays the star)
# ---------------------------------------------------------------------------
chips = ""
for name, ticker in TICKERS.items():
    snap = snaps[name]
    if snap.ok and snap.last is not None:
        pct = snap.change_pct
        cls = "chg-up" if (pct or 0) > 0 else "chg-down" if (pct or 0) < 0 else "chg-flat"
        pct_str = f"{pct:+.2f}%" if pct is not None else "—"
        chips += f"""<div class="ticker-chip">
            <div class="name">{name.upper()}</div>
            <div class="val">{snap.last:,.1f}</div>
            <div class="{cls}">{pct_str}</div>
        </div>"""
    else:
        chips += f"""<div class="ticker-chip">
            <div class="name">{name.upper()}</div>
            <div class="val">N/A</div>
        </div>"""
st.markdown(f'<div class="ticker-strip">{chips}</div>', unsafe_allow_html=True)

if not nas.ok:
    st.error(f"Could not load NAS100 futures data: {nas.error}")
    st.stop()

# ---------------------------------------------------------------------------
# BUILD THE BIAS
# ---------------------------------------------------------------------------
hist_pattern = {"available": False, "reason": "Not enough data to backtest yet."}
if nas.prev_close and technicals.get("atr14"):
    gap_pct = (nas.last - nas.prev_close) / nas.prev_close * 100
    atr_pct = technicals["atr14"] / nas.last * 100
    hist_pattern = historical_pattern_match(nas_daily, gap_pct, atr_pct)

today_str = now_ny.strftime("%Y-%m-%d")
today_events = MANUAL_HIGH_IMPACT_EVENTS.get(today_str, [])

result = build_daily_bias(
    nas_snap=nas,
    es_snap=snaps["sp500"],
    vix_snap=snaps["vix"],
    us10y_snap=snaps["us10y"],
    dxy_snap=snaps["dxy"],
    smh_snap=snaps["smh"],
    technicals=technicals,
    hist_pattern=hist_pattern,
    today_events=today_events,
)

css_cls = {"BUY": "buy", "SELL": "sell", "NEUTRAL": "neutral"}[result.bias]
emoji = {"BUY": "🟢", "SELL": "🔴", "NEUTRAL": "🟡"}[result.bias]

# ---------------------------------------------------------------------------
# TWO-COLUMN LAYOUT: BIAS (left)  |  REASONS (right)
# ---------------------------------------------------------------------------
left, right = st.columns([1, 1.3], gap="large")

with left:
    st.markdown(f"""
    <div class="bias-card {css_cls}">
        <div class="bias-label">TODAY'S NAS100 BIAS</div>
        <div class="bias-word {css_cls}">{emoji} {result.bias}</div>
        <div class="confidence-wrap">
            <div class="confidence-track">
                <div class="confidence-fill {css_cls}" style="width:{max(result.confidence,3):.0f}%;"></div>
            </div>
            <div class="confidence-num">{result.confidence:.0f}% confidence</div>
            <div class="confidence-caption">Confidence is discounted automatically when data is missing</div>
        </div>
        <div class="score-row">
            <div class="score-box bull"><div class="lbl">BULL SCORE</div><div class="num">{result.bull_score:.0f}</div></div>
            <div class="score-box bear"><div class="lbl">BEAR SCORE</div><div class="num">{result.bear_score:.0f}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">NQ Snapshot</div>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    m1.metric("Last", f"{nas.last:,.2f}", f"{nas.change_pct:+.2f}%" if nas.change_pct else None)
    m2.metric("Prev Close", f"{nas.prev_close:,.2f}" if nas.prev_close else "N/A")
    m3, m4 = st.columns(2)
    m3.metric("Session High", f"{nas.day_high:,.2f}" if nas.day_high else "N/A")
    m4.metric("Session Low", f"{nas.day_low:,.2f}" if nas.day_low else "N/A")

    if today_events:
        st.markdown('<div class="section-title">⚠️ Events Today</div>', unsafe_allow_html=True)
        for e in today_events:
            st.markdown(f'<div class="reason-card warn">{e}</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="reasons-title">Why this bias?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="reasons-sub">Every reason below traces back to real data pulled this run — '
        'nothing is invented.</div>',
        unsafe_allow_html=True,
    )

    def classify_note(note: str) -> str:
        lower = note.lower()
        if "unavailable" in lower or "insufficient" in lower or "no components" in lower:
            return "neutral"
        if "elevated" in lower or "overbought" in lower or "oversold" in lower or "event" in lower:
            return "warn"
        if any(w in lower for w in ["bullish", "up", "bounce", "'up'"]):
            return "pos"
        if any(w in lower for w in ["bearish", "down", "'down'", "risk"]):
            return "neg"
        return "neutral"

    for n in result.notes:
        st.markdown(f'<div class="reason-card {classify_note(n)}">{n}</div>', unsafe_allow_html=True)

    if result.unscored_weight > 0:
        st.markdown(f"""
        <div class="missing-note">
            ⚠️ {result.unscored_weight*100:.0f}% of the scoring framework had no data available this run
            and was left out — not guessed. That's why confidence is capped at {result.confidence:.0f}%
            rather than shown as artificially high.
        </div>
        """, unsafe_allow_html=True)

    if hist_pattern.get("available"):
        st.markdown('<div class="section-title">📈 Historical Pattern Match</div>', unsafe_allow_html=True)
        h1, h2, h3 = st.columns(3)
        h1.metric("Similar sessions", hist_pattern["sample_size"])
        h2.metric("Bullish next-day", f"{hist_pattern['bullish_prob']:.0f}%")
        h3.metric("Bearish next-day", f"{hist_pattern['bearish_prob']:.0f}%")
        st.caption(
            f"Avg next-day return {hist_pattern['avg_return_pct']:+.2f}% "
            f"(range {hist_pattern['max_adverse_pct']:+.2f}% to {hist_pattern['max_favorable_pct']:+.2f}%) "
            f"— real backtest over ~2 years of NQ=F daily data, matched on gap % and volatility."
        )

    if technicals:
        st.markdown('<div class="section-title">📐 Technicals</div>', unsafe_allow_html=True)
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("EMA20", f"{technicals['ema20']:,.0f}")
        t2.metric("EMA50", f"{technicals['ema50']:,.0f}")
        t3.metric("EMA200", f"{technicals['ema200']:,.0f}")
        t4.metric("RSI14", f"{technicals['rsi14']:.0f}" if technicals.get("rsi14") else "N/A")

st.markdown("""
<div class="footer-note">
⚠️ Educational decision-support tool only — not financial advice, and it does not place trades.
Data via Yahoo Finance (yfinance), typically delayed ~15 minutes. Always confirm with your own
analysis and risk management before trading.
</div>
""", unsafe_allow_html=True)
