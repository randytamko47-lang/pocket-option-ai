import yfinance as yf
import pandas as pd
import numpy as np
import requests
import streamlit as st
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier

# -----------------------
# TELEGRAM CONFIG
# -----------------------
BOT_TOKEN = st.secrets["BOT_TOKEN"]
CHAT_ID = st.secrets["CHAT_ID"]

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": msg})

# -----------------------
# STREAMLIT UI
# -----------------------
st.set_page_config(page_title="Pocket Option AI", layout="centered")
st.title("📊 Pocket Option AI Signals")

SYMBOL = st.selectbox("Select Asset", ["EURUSD=X", "GBPUSD=X", "USDJPY=X"])
CONFIDENCE_THRESHOLD = st.slider("Min Confidence (%)", 55, 75, 60) / 100

# -----------------------
# DATA + MODEL
# -----------------------
data = yf.download(SYMBOL, interval="1h", period="180d")
data.dropna(inplace=True)

data["ema_fast"] = EMAIndicator(data["Close"], 10).ema_indicator()
data["ema_slow"] = EMAIndicator(data["Close"], 30).ema_indicator()
data["rsi"] = RSIIndicator(data["Close"], 14).rsi()
data.dropna(inplace=True)

data["target"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)

X = data[["ema_fast", "ema_slow", "rsi"]]
y = data["target"]

model = RandomForestClassifier(n_estimators=200, max_depth=6)
model.fit(X[:-1], y[:-1])

latest = X.iloc[-1:].values
prob = model.predict_proba(latest)[0]

direction = "CALL 📈" if prob[1] > 0.5 else "PUT 📉"
confidence = max(prob)
trend = "Bullish" if data["ema_fast"].iloc[-1] > data["ema_slow"].iloc[-1] else "Bearish"

# -----------------------
# DISPLAY
# -----------------------
st.metric("Signal", direction)
st.metric("Confidence", f"{round(confidence*100,2)}%")
st.metric("Trend", trend)

# -----------------------
# ALERT BUTTON
# -----------------------
if confidence >= CONFIDENCE_THRESHOLD:
    if st.button("📨 Send Telegram Alert"):
        send_telegram(f"""
📊 Pocket Option AI Signal

Asset: {SYMBOL}
Direction: {direction}
Confidence: {round(confidence*100,2)}%
Trend: {trend}
Timeframe: 1H

⏰ Trade next candle only
""")
