# sensex
# 📊 Sensex Signal Dashboard

Live 5-minute trading signal dashboard for the **BSE Sensex (^BSESN)**.

Strategy: **EMA + VWAP + RSI + Swing Levels** — all 4 conditions must align for a signal.

## 🚀 Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy free on Streamlit Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select repo, set main file to `app.py` → Deploy

## 📊 Signal Logic
| Signal | Conditions (all 4 must be true) |
|--------|----------------------------------|
| 🟢 CALL | Close > VWAP · EMA20 > EMA50 · Close > Prev High · RSI > 55 |
| 🔴 PUT  | Close < VWAP · EMA20 < EMA50 · Close < Prev Low · RSI < 45 |
| 🟡 NO TRADE | Not all conditions met |

> ⚠️ For educational purposes only. Not financial advice.
