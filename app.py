import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time
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

.ts { font-size: .7rem; color: #475569; text-align: right; margin-top: -8px; }

hr { border-color: #1f2d45 !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
SYMBOL       = "^BSESN"   # BSE Sensex on Yahoo Finance
PERIOD       = "1d"       # Today only
INTERVAL     = "5m"       # 5-minute candles
AUTO_REFRESH = 60         # seconds

india_tz = pytz.timezone("Asia/Kolkata")

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def add_ema(df: pd.DataFrame) -> pd.DataFrame:
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    df["TP"]      = (df["High"] + df["Low"] + df["Close"]) / 3
    df["CumVol"]  = df["Volume"].cumsum()
    df["CumTPVol"]= (df["TP"] * df["Volume"]).cumsum()
    df["VWAP"]    = df["CumTPVol"] / df["CumVol"]
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta    = df["Close"].diff()
    gain     = np.where(delta > 0, delta, 0)
    loss     = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=df.index).rolling(period).mean()
    avg_loss = pd.Series(loss, index=df.index).rolling(period).mean()
    rs       = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def market_status(now_dt: datetime) -> str:
    open_t  = time(9, 15)
    close_t = time(15, 30)
    if open_t <= now_dt.time() <= close_t:
        return "Market Open"
    return "Market Closed"


# ──────────────────────────────────────────────
# DATA FETCH + PROCESSING
# ──────────────────────────────────────────────
@st.cache_data(ttl=AUTO_REFRESH)
def fetch_and_analyze() -> dict:
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
    df = df[~df.index.duplicated(keep="last")]

    df = add_ema(df)
    df = add_vwap(df)
    df = add_rsi(df)

    latest = df.iloc[-1]

    close = safe_float(latest["Close"])
    vwap  = safe_float(latest["VWAP"])
    ema20 = safe_float(latest["EMA20"])
    ema50 = safe_float(latest["EMA50"])
    rsi   = safe_float(latest["RSI"])

    # Signal logic (simplified)
    call = (
        close is not None and vwap is not None and ema20 is not None and ema50 is not None and rsi is not None
        and close > vwap
        and ema20 > ema50
        and rsi > 55
    )
    put = (
        close is not None and vwap is not None and ema20 is not None and ema50 is not None and rsi is not None
        and close < vwap
        and ema20 < ema50
        and rsi < 45
    )

    return {
        "close": close,
        "vwap": vwap,
        "ema20": ema20,
        "ema50": ema50,
        "rsi": rsi,
        "call": call,
        "put": put,
        "df": df,
    }

# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────
col_title, col_btn = st.columns([4, 1])
with col_title:
    st.markdown("## 📊 Sensex Signal Dashboard")
    st.markdown(
        "<div style='color:#64748b;font-size:.8rem;margin-top:-12px;margin-bottom:16px;'>"
        "Sensex · 5-minute candles · EMA · VWAP · RSI · Trading-style signals"
        "</div>",
        unsafe_allow_html=True,
    )
with col_btn:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if st.button("⟳ Refresh", use_container_width=True):
        st.cache_data.clear()

with st.spinner("Fetching live Sensex data…"):
    try:
        data  = fetch_and_analyze()
        error = None
    except Exception as e:
        error = str(e)

if error:
    st.error(f"⚠️ {error}")
    st.stop()

now_dt = datetime.now(india_tz)
now_str = now_dt.strftime("%d %b %Y  %I:%M:%S %p IST")
status  = market_status(now_dt)

# ── PRICE + STATUS ROW ───────────────────────
m1, m2, m3 = st.columns(3)
m1.metric("Sensex (^BSESN)", f"₹ {data['close']:,.2f}" if data["close"] else "—")
m2.metric("VWAP",            f"₹ {data['vwap']:,.2f}"  if data["vwap"]  else "—")
m3.metric("RSI (14)",        f"{data['rsi']:.2f}"      if data["rsi"]   else "—")

st.markdown(f"<div class='ts'>Status: {status} · Last updated: {now_str}</div>", unsafe_allow_html=True)
st.markdown("---")

# ── SIGNAL BANNER ────────────────────────────
if data["call"]:
    cls = "sig-call"
    val = "🟢 CALL SIGNAL"
    sub = "Trend looks bullish — conditions aligned for a CALL setup."
elif data["put"]:
    cls = "sig-put"
    val = "🔴 PUT SIGNAL"
    sub = "Trend looks bearish — conditions aligned for a PUT setup."
else:
    cls = "sig-neutral"
    val = "🟡 NO TRADE"
    sub = "Conditions are not fully aligned — better to wait."

st.markdown(f"""
<div class="sig-box {cls}">
  <div class="sig-label">Current Signal — 5 Min</div>
  <div class="sig-value">{val}</div>
  <div class="sig-sub">{sub}</div>
</div>
""", unsafe_allow_html=True)

# ── CONDITION SUMMARY ────────────────────────
st.markdown("<div class='sec-label'>Condition Summary</div>", unsafe_allow_html=True)

def ck(cond): return "✅" if cond else "❌"

c   = data["close"] or 0
v   = data["vwap"]  or 0
e20 = data["ema20"] or 0
e50 = data["ema50"] or 0
r   = data["rsi"]   or 0

st.markdown(f"""
<div class="cond-box">
  {ck(c > v)} Close {'above' if c > v else 'below'} VWAP
    &nbsp;<code>{c:,.0f} vs {v:,.0f}</code><br>
  {ck(e20 > e50)} EMA20 {'above' if e20 > e50 else 'below'} EMA50
    &nbsp;<code>{e20:,.0f} vs {e50:,.0f}</code><br>
  {ck(r > 55)} RSI {'above' if r > 55 else 'below'} 55
    &nbsp;<code>{r:.2f}</code>
</div>
""", unsafe_allow_html=True)

# ── PRICE + EMA CHART ────────────────────────
st.markdown("<div class='sec-label'>Price vs EMA (5-Min, Today)</div>", unsafe_allow_html=True)
price_chart = data["df"][["Close", "EMA20", "EMA50"]].dropna()
price_chart.columns = ["Close", "EMA 20", "EMA 50"]
st.line_chart(price_chart, color=["#e2e8f0", "#22c55e", "#f59e0b"])
