"""
Phase 7 — Live Market Prediction & Trading Agent
Run on a machine with MT5 terminal logged in (demo recommended).
"""

import time
from datetime import datetime
import math
import joblib
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

# ----------- CONFIG -----------
SYMBOL = "XAUUSD_"          # adjust to your broker's symbol name
TIMEFRAME = mt5.TIMEFRAME_M1  # we'll use 1-minute candles for features
MODEL_FILE = "AI_Trading_Stacked_Model.pkl"
SCALER_FILE = "AI_Trading_Scaler.pkl"
PAPER_MODE = True           # True => don't send real orders, just print (recommended to test)
REFRESH_SECONDS = 5         # how often to run loop
SPEED_PAUSE_THRESH = 0.12   # pause trading when avg speed > this (tune for your asset)
SPEED_WINDOW = 10           # seconds (or ticks) moving average window for speed
COOLDOWN_SECONDS = 5        # wait this long after placing a trade before next trade
MAX_OPEN_TRADES = 4         # risk control
LOT = 0.01                  # sample volume
SL_POINTS = 20              # stop-loss in points (adjust to instrument decimals)
TP_POINTS = 30              # take-profit in points

# --------- END CONFIG ----------

# Load model & scaler
try:
    model = joblib.load(MODEL_FILE)
    print(f"✅ Loaded model: {MODEL_FILE}")
except Exception as e:
    raise SystemExit(f"Model load error: {e}")

try:
    scaler = joblib.load(SCALER_FILE)
    print(f"✅ Loaded scaler: {SCALER_FILE}")
except Exception as e:
    raise SystemExit(f"Scaler load error: {e}")

# Initialize MT5
if not mt5.initialize():
    raise SystemExit(f"MT5 initialize() failed, error: {mt5.last_error()}")
print("✅ MT5 initialized")

# helper: get last N minute candles as pandas DataFrame
def get_candles(symbol, timeframe=TIMEFRAME, n=200):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# feature engineering (must match training pipeline)
def make_features(df):
    df = df.copy()
    # basic indicators (same names as training)
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_30'] = df['close'].rolling(window=30).mean()
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA_30'] = df['close'].ewm(span=30, adjust=False).mean()

    # RSI 14
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    rs = ma_up / (ma_down + 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ATR 14
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()

    # Returns
    df['Returns'] = df['close'].pct_change()

    # Additional features you might have used (safe to compute if missing)
    df['Body_Size'] = (df['close'] - df['open']).abs()
    df['Upper_Shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['Lower_Shadow'] = df[['close','open']].min(axis=1) - df['low']
    df['Candle_Ratio'] = df['Body_Size'] / (df['high'] - df['low'] + 1e-9)
    df['Momentum_5'] = df['close'].diff(5)
    df['Volatility_5'] = df['close'].rolling(5).std()
    df['SMA_Cross'] = (df['SMA_10'] > df['SMA_30']).astype(int)

    # time features
    df['Hour'] = df['time'].dt.hour
    df['DayOfWeek'] = df['time'].dt.dayofweek

    df = df.dropna()
    return df

# function to align features with model expected columns
def prepare_input_row(features_df):
    # take last row
    row = features_df.iloc[-1].copy()
    
    # keep only numeric values
    numeric = row[row.apply(lambda x: isinstance(x, (int, float)))]
    
    # convert into a single-row DataFrame
    X = pd.DataFrame([numeric])
    
    # try aligning columns with model input
    try:
        cols = list(model.feature_names_in_)
        X = X.reindex(columns=cols, fill_value=0.0)
    except Exception:
        pass

    return X

# Function to compute "speed" like earlier (price change per second) using ticks
speed_history = []
def update_speed(symbol, window=SPEED_WINDOW):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    now = tick.time
    price = tick.bid
    speed_history.append((now, price))
    # keep only last 'window' entries
    while len(speed_history) > window:
        speed_history.pop(0)
    if len(speed_history) < 2:
        return 0.0
    # compute per-second speed average
    speeds = []
    for i in range(1, len(speed_history)):
        t0, p0 = speed_history[i-1]
        t1, p1 = speed_history[i]
        dt = max(1, t1 - t0)
        speeds.append(abs(p1 - p0) / dt)
    return sum(speeds) / len(speeds)

# trading helpers
last_trade_time = 0
open_positions = []  # store ticket numbers to monitor; for demo we will not query positions heavily

def get_open_positions_count(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return 0
    return len(positions)

def place_order(action, volume=LOT):
    """
    Place market order. action = "BUY" or "SELL"
    """
    if PAPER_MODE:
        print(f"{datetime.now().strftime('%H:%M:%S')} [PAPER] Would place {action} {volume} {SYMBOL}")
        return None

    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        print("No tick for price, abort order")
        return None

    price = tick.ask if action == "BUY" else tick.bid
    deviation = 50
    order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL

    # set SL/TP in absolute price (points assumed like instrument price ticks)
    point = mt5.symbol_info(SYMBOL).point
    sl = price - SL_POINTS * point if action == "BUY" else price + SL_POINTS * point
    tp = price + TP_POINTS * point if action == "BUY" else price - TP_POINTS * point

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": float(volume),
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": deviation,
        "magic": 234000,
        "comment": "AI_auto_trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result is None:
        print("order_send returned None")
        return None
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed: retcode={result.retcode}, comment={result.comment}")
        return None
    print(f"{datetime.now().strftime('%H:%M:%S')} ✅ {action} placed at {price} (ticket {result.order})")
    return result.order

# Controller loop
print("🔁 Live agent started — PRESS CTRL+C to stop")
try:
    while True:
        # 1) update speedometer and pause trading if market too fast
        avg_speed = update_speed(SYMBOL)
        if avg_speed is None:
            print("No ticks available, waiting...")
            time.sleep(REFRESH_SECONDS)
            continue

        trading_allowed = True
        if avg_speed > SPEED_PAUSE_THRESH:
            trading_allowed = False

        # 2) fetch recent candles and create features
        candles = get_candles(SYMBOL, timeframe=TIMEFRAME, n=300)
        if candles is None or len(candles) < 60:
            print("Not enough candles, waiting...")
            time.sleep(REFRESH_SECONDS)
            continue

        features = make_features(candles)

        # 3) prepare model input and scale
        X_row = prepare_input_row(features)
        # ensure same ordering and missing columns handled
        X_row = X_row.fillna(0.0)
        try:
            X_scaled = scaler.transform(X_row)
        except Exception as e:
            # fallback: if scaler expects different cols, align numeric then scale
            X_scaled = scaler.transform(X_row.reindex(columns=scaler.feature_names_in_, fill_value=0.0))

        # 4) predict
        pred = model.predict(X_scaled)[0]         # 1 or 0
        pred_proba = None
        try:
            pred_proba = model.predict_proba(X_scaled)[0].max()
        except Exception:
            pred_proba = None

        # 5) basic trading strategy (only act when trading_allowed and cooldown and limits ok)
        now_ts = time.time()
        open_cnt = get_open_positions_count(SYMBOL)
        can_trade = trading_allowed and (now_ts - last_trade_time > COOLDOWN_SECONDS) and (open_cnt < MAX_OPEN_TRADES)

        action = None
        if can_trade:
            if pred == 1:
                action = "BUY"
            else:
                action = "SELL"

            # optional further checks: spread, liquidity
            sym = mt5.symbol_info(SYMBOL)
            if sym is None:
                print("No symbol info, skipping order")
                action = None
            else:
                spread = (sym.ask - sym.bid)
                # skip if spread too wide
                max_spread = sym.point * 50  # 50 points threshold (adjust)
                if spread > max_spread:
                    print(f"Spread {spread} > max_spread {max_spread}, skipping")
                    action = None

        # 6) execute or simulate
        if action:
            ticket = place_order(action, volume=LOT)
            last_trade_time = time.time()
            if ticket:
                open_positions.append(ticket)

        # 7) logging
        status = "ALLOWED" if trading_allowed else "PAUSED"
        proba_str = f"{pred_proba:.2f}" if pred_proba is not None else "N/A"
        print(f"{datetime.now().strftime('%H:%M:%S')} | Speed: {avg_speed:.4f} | {status} | Pred: {pred} | P:{proba_str} | Open:{open_cnt}")

        time.sleep(REFRESH_SECONDS)

except KeyboardInterrupt:
    print("\n⛔ Stopped by user")
finally:
    mt5.shutdown()
    print("MT5 shutdown complete")
