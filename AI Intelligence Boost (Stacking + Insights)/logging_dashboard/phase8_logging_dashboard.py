import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import joblib
import time
from datetime import datetime
import plotly.express as px
import threading

# ====== LOAD MODELS ======
model = joblib.load("AI_Trading_Stacked_Model.pkl")
scaler = joblib.load("AI_Trading_Scaler.pkl")

# ====== MT5 INITIALIZE ======
if not mt5.initialize():
    print("❌ MT5 initialization failed!")
    quit()
print("✅ MT5 Connected")

# ====== SETTINGS ======
symbol = "XAUUSD_"
lot = 0.01
interval = 5  # seconds
log_file = "live_trading_log.csv"

# ====== LOG FILE INIT ======
columns = ["time", "speed", "prediction", "confidence", "action", "price"]
if not pd.io.common.file_exists(log_file):
    pd.DataFrame(columns=columns).to_csv(log_file, index=False)

# ====== FUNCTION TO GET MARKET DATA ======
def get_live_speed(symbol, window=5):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, window)
    if rates is None or len(rates) < 2:
        return 0.0
    closes = np.array([r['close'] for r in rates])
    speed = abs(closes[-1] - closes[0]) / closes[0]
    return speed

# ====== LIVE LOOP ======
def live_agent():
    num_features = 15  # same as training phase
    while True:
        speed = get_live_speed(symbol)

        # Mock features for now (only first = speed)
        input_data = np.zeros((1, num_features))
        input_data[0, 0] = speed  

        scaled = scaler.transform(pd.DataFrame(input_data, columns=scaler.feature_names_in_))

        pred = model.predict(scaled)[0]
        prob = np.max(model.predict_proba(scaled))
        action = "BUY" if pred == 1 else "SELL"
        price = mt5.symbol_info_tick(symbol).bid

        log_entry = pd.DataFrame([[datetime.now(), speed, pred, prob, action, price]], columns=columns)
        log_entry.to_csv(log_file, mode='a', header=False, index=False)

        print(f"{datetime.now().strftime('%H:%M:%S')} | {action} | Speed:{speed:.4f} | Conf:{prob:.2f}")
        time.sleep(interval)

# ====== DASHBOARD THREAD ======
def dashboard():
    while True:
        try:
            df = pd.read_csv(log_file)
            if len(df) > 5:
                fig = px.line(df, x="time", y="speed", color="action", title="Live Market Speed & Predictions")
                fig.show()
                time.sleep(30)  # Update every 30s
        except Exception as e:
            print("Dashboard error:", e)
            time.sleep(10)

# ====== RUN BOTH THREADS ======
threading.Thread(target=live_agent, daemon=True).start()
threading.Thread(target=dashboard, daemon=True).start()

# Keep main thread alive
while True:
    time.sleep(1)
