"""
Sensex Signal Dashboard
=======================
Strategy: EMA20/50 + VWAP + RSI + Swing Levels on live 5-minute data
Symbol:   ^BSESN (BSE Sensex via Yahoo Finance)

Run locally:  streamlit run app.py
Deploy:       Push to GitHub → share.streamlit.io
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Sensex Signal Dashboard",
    page_icon="📊",
    layout="centered",
)

# ──────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0b0f1a; }
[data-testid="stHeader"]           { background: #0b0f1a; }
html, body, [class*="css"]         { font-family: 'Inter', sans-serif; color: #e2e8f0; }

[data-testid="metric-container"] {
    background: #111827 !important;
    border: 1px solid #1f2d45 !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
}
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: .7rem !important; letter-spacing: .08em; text-transform: uppercase; }

.sig-box { padding: 24px; border-radius: 14px; text-align: center; margin: 8px 0 20px; }
.sig-call    { background: rgba(34,197,94,.12);  border: 2px solid #22c55e; }
.sig-put     { background: rgba(239,68,68,.12);  border: 2px solid #ef4444; }
.sig-neutral { background: rgba(245,158,11,.10); border: 2px solid #f59e0b; }
.sig-label { font-size: .68rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: #64748b; margin-bottom: 6px; }
.sig-value { font-size: 2rem; font-weight: 700; }
.sig-call    .sig-value { color: #22c55e; }
.sig-put     .sig-value { color: #ef4444; }
.sig-neutral .sig-value { color: #f59e0b; }
.sig-sub { font-size: .78rem; color: #64748b; margin-top: 6px; }

.cond-box {
    background: #111827; border: 1px solid #1f2d45;
    border-radius: 10px; padding: 14px 18px;
    font-size: .8rem; color: #94a3b8; line-height: 2.1;
    margin-top: 6px;
}
.cond-box code { color: #cbd5e1; background: #1e293b; padding: 1px 6px; border-radius: 4px; }

.sec-label {
    font-size: .68rem; font-weight: 700; letter-spacing: .14em;
    text-transform: uppercase; color: #475569;
    border-bottom: 1px solid #1f2d45;
    padding-bottom: 8px; margin: 20px 0 14px;
}

.legend {
    background: #111827; border: 1px solid #1f2d45;
    border-radius: 12px; padding: 18px 22px;
    font-size: .78rem; color: #94a3b8; line-height: 1.9;
}
.legend strong { color: #e2e8f0; }

.ts { font-size: .7rem; color: #475569; text-align: right; margin-top: -8px; }

.signal-row {
    background: #111827; border: 1px solid #1f2d45;
    border-radius: 10px; padding: 12px 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: .78rem; margin-bottom: 6px;
    display: flex; justify-content: space-between; align-items: center;
}
hr { border-color: #1f2d45 !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
SYMBOL       = "^BSESN"   # BSE Sensex on Yahoo Finance
PERIOD       = "5d"        # Enough for intraday + VWAP calculation
INTERVAL     = "5m"        # 5-minute candles
AUTO_REFRESH = 60          # seconds


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance multi-level columns e.g. ('Close', '^BSESN') → 'Close'."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ──────────────────────────────────────────────
# INDICATORS
# ──────────────────────────────────────────────
def add_ema(df: pd.DataFrame) -> pd.DataFrame:
    """EMA20 and EMA50 on Close price."""
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    VWAP = cumulative(Typical Price × Volume) / cumulative(Volume)
    Typical Price = (High + Low + Close) / 3
    """
    df["TP"]      = (df["High"] + df["Low"] + df["Close"]) / 3
    df["CumVol"]  = df["Volume"].cumsum()
    df["CumTPVol"]= (df["TP"] * df["Volume"]).cumsum()
    df["VWAP"]    = df["CumTPVol"] / df["CumVol"]
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Standard RSI using numpy for speed."""
    delta    = df["Close"].diff()
    gain     = np.where(delta > 0, delta, 0)
    loss     = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=df.index).rolling(period).mean()
    avg_loss = pd.Series(loss, index=df.index).rolling(period).mean()
    rs       = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_swing_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Previous candle's High and Low as swing reference levels."""
    df["Prev_High"] = df["High"].shift(1)
    df["Prev_Low"]  = df["Low"].shift(1)
    return df


def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    CALL BUY — all 4 conditions must be true:
      1. Close > VWAP        (price above fair value)
      2. EMA20 > EMA50       (short-term trend above long-term)
      3. Close > Prev_High   (breakout above previous candle high)
      4. RSI > 55            (momentum is bullish, not overbought)

    PUT BUY — all 4 conditions must be true:
      1. Close < VWAP        (price below fair value)
      2. EMA20 < EMA50       (short-term trend below long-term)
      3. Close < Prev_Low    (breakdown below previous candle low)
      4. RSI < 45            (momentum is bearish, not oversold)
    """
    df["CALL_BUY"] = (
        (df["Close"] > df["VWAP"])      &
        (df["EMA20"] > df["EMA50"])     &
        (df["Close"] > df["Prev_High"]) &
        (df["RSI"]   > 55)
    )
    df["PUT_BUY"] = (
        (df["Close"] < df["VWAP"])     &
        (df["EMA20"] < df["EMA50"])    &
        (df["Close"] < df["Prev_Low"]) &
        (df["RSI"]   < 45)
    )
    return df


# ──────────────────────────────────────────────
# DATA FETCH + PROCESSING
# ──────────────────────────────────────────────
@st.cache_data(ttl=AUTO_REFRESH)
def fetch_and_analyze() -> dict:
    """
    Download live 5m Sensex data, compute all indicators,
    generate signals, and return everything the UI needs.
    """
    df = yf.download(
        SYMBOL,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No data returned for {SYMBOL}. Markets may be closed.")

    df = flatten_columns(df)
    df = df[~df.index.duplicated(keep="last")]   # fix NSE/BSE duplicate timestamps

    # Add all indicators
    df = add_ema(df)
    df = add_vwap(df)
    df = add_rsi(df)
    df = add_swing_levels(df)
    df = add_signals(df)

    # Latest bar
    latest = df.iloc[-1]

    close     = safe_float(latest["Close"])
    vwap      = safe_float(latest["VWAP"])
    ema20     = safe_float(latest["EMA20"])
    ema50     = safe_float(latest["EMA50"])
    rsi       = safe_float(latest["RSI"])
    prev_high = safe_float(latest["Prev_High"])
    prev_low  = safe_float(latest["Prev_Low"])
    call      = bool(latest["CALL_BUY"])
    put       = bool(latest["PUT_BUY"])

    # All historical signals for the signal log
    signal_rows = df[df["CALL_BUY"] | df["PUT_BUY"]].copy()

    return {
        "close":      close,
        "vwap":       vwap,
        "ema20":      ema20,
        "ema50":      ema50,
        "rsi":        rsi,
        "prev_high":  prev_high,
        "prev_low":   prev_low,
        "call":       call,
        "put":        put,
        "signal_rows": signal_rows,
        "df":         df,
    }


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────
# Header
col_title, col_btn = st.columns([4, 1])
with col_title:
    st.markdown("## 📊 Sensex Signal Dashboard")
    st.markdown("<div style='color:#64748b;font-size:.8rem;margin-top:-12px;margin-bottom:16px;'>EMA · VWAP · RSI · Swing Levels · 5-minute candles · ^BSESN</div>", unsafe_allow_html=True)
with col_btn:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if st.button("⟳ Refresh", use_container_width=True):
        st.cache_data.clear()

# Fetch
with st.spinner("Fetching live Sensex data…"):
    try:
        data  = fetch_and_analyze()
        error = None
    except Exception as e:
        error = str(e)

if error:
    st.error(f"⚠️ {error}")
    st.stop()

# Timestamp
ist = pytz.timezone("Asia/Kolkata")
now = datetime.now(ist).strftime("%d %b %Y  %I:%M:%S %p IST")

# ── PRICE ROW ────────────────────────────────
m1, m2, m3 = st.columns(3)
m1.metric("Sensex (^BSESN)",  f"₹ {data['close']:,.2f}"  if data["close"] else "—")
m2.metric("VWAP",             f"₹ {data['vwap']:,.2f}"   if data["vwap"]  else "—")
m3.metric("RSI (14)",         f"{data['rsi']:.2f}"        if data["rsi"]   else "—")

st.markdown(f"<div class='ts'>Last updated: {now}</div>", unsafe_allow_html=True)
st.markdown("---")

# ── SIGNAL BANNER ────────────────────────────
if data["call"]:
    final, cls, val, sub = "CALL", "sig-call", "🟢 CALL BUY", "All 4 bullish conditions confirmed on current 5m candle"
elif data["put"]:
    final, cls, val, sub = "PUT",  "sig-put",  "🔴 PUT BUY",  "All 4 bearish conditions confirmed on current 5m candle"
else:
    final, cls, val, sub = "WAIT", "sig-neutral", "🟡 NO TRADE", "Not all conditions aligned — wait for a clean setup"

st.markdown(f"""
<div class="sig-box {cls}">
  <div class="sig-label">Current Signal — 5 Min</div>
  <div class="sig-value">{val}</div>
  <div class="sig-sub">{sub}</div>
</div>
""", unsafe_allow_html=True)

# ── CONDITION BREAKDOWN ──────────────────────
st.markdown("<div class='sec-label'>Condition Breakdown — Current Candle</div>", unsafe_allow_html=True)

def ck(cond): return "✅" if cond else "❌"

c  = data["close"] or 0
v  = data["vwap"]  or 0
e20= data["ema20"] or 0
e50= data["ema50"] or 0
r  = data["rsi"]   or 0
ph = data["prev_high"] or 0
pl = data["prev_low"]  or 0

col_call, col_put = st.columns(2)

with col_call:
    st.markdown("**🟢 CALL conditions**")
    st.markdown(f"""
    <div class="cond-box">
      {ck(c > v)}  Close <b>{'above' if c > v else 'below'}</b> VWAP
        &nbsp;<code>{c:,.0f} vs {v:,.0f}</code><br>
      {ck(e20 > e50)}  EMA20 <b>{'above' if e20 > e50 else 'below'}</b> EMA50
        &nbsp;<code>{e20:,.0f} vs {e50:,.0f}</code><br>
      {ck(c > ph)}  Close <b>{'above' if c > ph else 'below'}</b> Prev High
        &nbsp;<code>{c:,.0f} vs {ph:,.0f}</code><br>
      {ck(r > 55)}  RSI <b>{'above' if r > 55 else 'below'}</b> 55
        &nbsp;<code>{r:.2f}</code>
    </div>
    """, unsafe_allow_html=True)

with col_put:
    st.markdown("**🔴 PUT conditions**")
    st.markdown(f"""
    <div class="cond-box">
      {ck(c < v)}  Close <b>{'below' if c < v else 'above'}</b> VWAP
        &nbsp;<code>{c:,.0f} vs {v:,.0f}</code><br>
      {ck(e20 < e50)}  EMA20 <b>{'below' if e20 < e50 else 'above'}</b> EMA50
        &nbsp;<code>{e20:,.0f} vs {e50:,.0f}</code><br>
      {ck(c < pl)}  Close <b>{'below' if c < pl else 'above'}</b> Prev Low
        &nbsp;<code>{c:,.0f} vs {pl:,.0f}</code><br>
      {ck(r < 45)}  RSI <b>{'below' if r < 45 else 'above'}</b> 45
        &nbsp;<code>{r:.2f}</code>
    </div>
    """, unsafe_allow_html=True)

# ── EMA METRICS ──────────────────────────────
st.markdown("<div class='sec-label'>EMA Levels</div>", unsafe_allow_html=True)
ea, eb, ec = st.columns(3)
ea.metric("EMA 20",     f"{data['ema20']:,.2f}" if data["ema20"] else "—")
eb.metric("EMA 50",     f"{data['ema50']:,.2f}" if data["ema50"] else "—")
ec.metric("EMA Spread", f"{(data['ema20'] or 0) - (data['ema50'] or 0):+.2f}")

# ── SIGNAL LOG ───────────────────────────────
st.markdown("<div class='sec-label'>Signal Log — All Signals (Last 5 Days)</div>", unsafe_allow_html=True)

sig_df = data["signal_rows"]
if sig_df.empty:
    st.info("No CALL or PUT signals triggered in the last 5 days.")
else:
    display = sig_df[["Close", "VWAP", "EMA20", "EMA50", "RSI", "CALL_BUY", "PUT_BUY"]].copy()
    display.index = display.index.strftime("%d %b  %I:%M %p")
    display.columns = ["Close", "VWAP", "EMA20", "EMA50", "RSI", "CALL", "PUT"]
    display["Signal"] = display.apply(
        lambda r: "🟢 CALL" if r["CALL"] else "🔴 PUT", axis=1
    )
    display = display.drop(columns=["CALL", "PUT"])
    for col in ["Close", "VWAP", "EMA20", "EMA50"]:
        display[col] = display[col].map(lambda x: f"{x:,.2f}")
    display["RSI"] = display["RSI"].map(lambda x: f"{x:.2f}")
    st.dataframe(display, use_container_width=True)

# ── CHARTS ───────────────────────────────────
st.markdown("<div class='sec-label'>RSI Chart (5-Min)</div>", unsafe_allow_html=True)
rsi_chart = data["df"][["RSI"]].dropna().tail(100)
st.line_chart(rsi_chart, color=["#3b82f6"])

st.markdown("<div class='sec-label'>EMA vs Price (5-Min, Last 100 Candles)</div>", unsafe_allow_html=True)
price_chart = data["df"][["Close", "EMA20", "EMA50"]].dropna().tail(100)
price_chart.columns = ["Close", "EMA 20", "EMA 50"]
st.line_chart(price_chart, color=["#e2e8f0", "#22c55e", "#f59e0b"])

st.markdown("---")

# ── LEGEND ───────────────────────────────────
st.markdown("""
<div class="legend">
  <strong>How signals are calculated (5-minute candles)</strong><br><br>

  🟢 <strong>CALL BUY</strong> — All 4 must be true:<br>
  &nbsp;&nbsp;① Close &gt; VWAP &nbsp;(price above fair value)<br>
  &nbsp;&nbsp;② EMA20 &gt; EMA50 &nbsp;(short-term trend above long-term)<br>
  &nbsp;&nbsp;③ Close &gt; Previous candle High &nbsp;(bullish breakout)<br>
  &nbsp;&nbsp;④ RSI &gt; 55 &nbsp;(bullish momentum confirmed)<br><br>

  🔴 <strong>PUT BUY</strong> — All 4 must be true:<br>
  &nbsp;&nbsp;① Close &lt; VWAP &nbsp;(price below fair value)<br>
  &nbsp;&nbsp;② EMA20 &lt; EMA50 &nbsp;(short-term trend below long-term)<br>
  &nbsp;&nbsp;③ Close &lt; Previous candle Low &nbsp;(bearish breakdown)<br>
  &nbsp;&nbsp;④ RSI &lt; 45 &nbsp;(bearish momentum confirmed)<br><br>

  🟡 <strong>NO TRADE</strong> — Conditions are not fully aligned. Wait for a cleaner setup.<br><br>

  <span style='color:#475569'>Data: Yahoo Finance ^BSESN · Interval: 5m · Period: 5 days · Refreshes every 60s</span>
</div>
""", unsafe_allow_html=True)
