# NAS100 Daily Bias Dashboard

A hosted, browser-based dashboard that pulls live NAS100 (NQ futures) and
cross-market data, computes technicals and a genuine backtested historical
pattern-match, and produces a transparent BUY / SELL / NEUTRAL bias with a
confidence score. You open a URL each morning — no scheduler needed, since
it re-fetches data on every page load.

**This is a decision-support tool, not financial advice.** It does not
place trades and does not guarantee outcomes.

## What it does

- Pulls live/delayed data for NQ=F, ES=F, YM=F, VIX, DXY, US10Y yield,
  Gold, Oil, NVDA, SMH via Yahoo Finance (`yfinance` — free, no API key).
- Computes EMA(20/50/200), RSI(14), ATR(14) from 2 years of real daily bars.
- Runs an actual backtest: finds historical NQ sessions with a similar
  overnight gap % and volatility (ATR% of price), and reports what really
  happened the next day (sample size, bull/bear split, avg return). If the
  sample is too small, it says so instead of making up a number.
- Cross-checks ES / VIX / 10Y yield / DXY / semiconductors for directional
  agreement.
- Combines everything into a weighted score. **Any component it can't
  compute (missing data) is excluded from the average, not filled with a
  guess** — and confidence is discounted the more of the framework is
  missing. If most of the framework is unscored, it forces NEUTRAL rather
  than a confident-looking BUY/SELL.

## What it deliberately does NOT automate

- **Economic calendar / news events** — there's no reliable free,
  no-API-key source for this. Edit `MANUAL_HIGH_IMPACT_EVENTS` in
  `engine.py` weekly (takes 30 seconds), or upgrade later with a paid-tier
  key from Finnhub, Financial Modeling Prep, or Polygon.io (all have a
  `/calendar/economic` style endpoint) and wire it into `engine.py`.
- **Order execution** — this tool never places trades. If you eventually
  want that, it would need to be connected separately to a broker API
  (e.g. Alpaca, Interactive Brokers) with its own explicit safeguards —
  a meaningfully bigger and riskier project than a bias dashboard.

## Deploy it for free (Streamlit Community Cloud)

1. Create a free GitHub account if you don't have one, and a new
   **public** repo, e.g. `nas100-dashboard`.
2. Upload these three files to it: `app.py`, `engine.py`,
   `requirements.txt` (and this `README.md` if you like).
3. Go to **share.streamlit.io**, sign in with GitHub, click
   **"New app"**, pick your repo, branch `main`, and set the main file
   path to `app.py`.
4. Click **Deploy**. In a minute or two you'll get a permanent URL like
   `https://your-app-name.streamlit.app` — bookmark it and open it each
   morning. Every visit re-pulls fresh data automatically.

Streamlit Community Cloud is free for public repos and this app is light
enough to comfortably fit their free tier.

## Run it locally instead (optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Editing the weighting / logic

All the scoring logic lives in `engine.py` (`WEIGHTS` dict and
`build_daily_bias()`). Nothing is hidden in `app.py` — that file is purely
display. Adjust weights, add tickers to `TICKERS`, or tighten/loosen the
historical pattern-match tolerance (`tolerance_gap`, `tolerance_atr` in
`historical_pattern_match()`) as you like.
