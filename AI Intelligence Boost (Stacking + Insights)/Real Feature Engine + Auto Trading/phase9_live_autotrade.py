"""
phase9_live_autotrade.py
Phase 9: Real Feature Engine + Auto Trading (Paper mode by default)

Make sure the following files exist in the same folder:
- AI_Trading_Stacked_Model.pkl
- AI_Trading_Scaler.pkl

Dependencies:
pip install pandas numpy MetaTrader5 joblib ta
"""

import time
from datetime import datetime
import math
import joblib
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import os
import warnings

warnings.filterwarnings("ignore")

# ------------------ CONFIG ------------------
SYMBOL = "XAUUSD_"                      # change to your broker symbol
TIMEFRAME = mt5.TIMEFRAME_M1
MODEL_FILE = "AI_Trading_Stacked_Model.pkl"
SCALER_FILE = "AI_Trading_Scaler.pkl"

PAPER_MODE = True                       # True = simulate; False = place real orders
REFRESH_SECONDS = 5                     # main loop sleep
SPEED_WINDOW = 10                       # ticks to average speed
SPEED_PAUSE_THRESH = 0.12               # pause trading when speed above this
MIN_CONFIDENCE = 0.60                   # require model probability >= this to trade
COOLDOWN_SECONDS = 5                    # wait after placing a trade
MAX_OPEN_TRADES = 3
LOT = 0.01                              # base lot size
SL_POINTS = 20                          # SL in points (instrument points)
TP_POINTS = 30                          # TP in points
MAX_DRAWDOWN_PCT = 0.2                  # stop trading if simulated drawdown exceeds 20%
INITIAL_EQUITY = 10000.0                # used only in PAPER_MODE for PnL sim

LOG_FILE = "phase9_trading_log.csv"
# --------------------------------------------

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

# initialize MT5
if not mt5.initialize():
    raise SystemExit(f"MT5 initialize() failed, error: {mt5.last_error()}")
print("✅ MT5 initialized")

# helper: get last n candles
def get_candles(symbol, timeframe=TIMEFRAME, n=300):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# feature engineering - must match Phase3/5/6 features
def make_features(df):
    df = df.copy()
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

    # ATR14
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()

    # Returns
    df['Returns'] = df['close'].pct_change()

    # Additional features used earlier
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

# speed updater (ticks)
speed_history = []
def update_speed(symbol, window=SPEED_WINDOW):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    now = tick.time
    price = tick.bid
    speed_history.append((now, price))
    while len(speed_history) > window:
        speed_history.pop(0)
    if len(speed_history) < 2:
        return 0.0
    speeds = []
    for i in range(1, len(speed_history)):
        t0, p0 = speed_history[i-1]
        t1, p1 = speed_history[i]
        dt = max(1, t1 - t0)
        speeds.append(abs(p1 - p0) / dt)
    return sum(speeds) / len(speeds)

# Logging init
if not os.path.exists(LOG_FILE):
    df_init = pd.DataFrame(columns=[
        "timestamp","price","prediction","prob","action","speed",
        "open_positions","equity","real_trade_ticket","notes"
    ])
    df_init.to_csv(LOG_FILE, index=False)

# PAPER mode simulated positions and equity
sim_positions = []  # list of dicts: {id, action, entry_price, volume, sl, tp, open_time}
sim_equity = INITIAL_EQUITY
sim_max_equity = INITIAL_EQUITY
sim_drawdown = 0.0

last_trade_time = 0

def get_open_positions_count(symbol):
    if PAPER_MODE:
        return len(sim_positions)
    pos = mt5.positions_get(symbol=symbol)
    if pos is None:
        return 0
    return len(pos)

# place order (paper or real)
def place_order(action, volume=LOT):
    global sim_positions, sim_equity, last_trade_time
    now_price = mt5.symbol_info_tick(SYMBOL)
    if now_price is None:
        print("No tick - cannot place order")
        return None
    price = now_price.ask if action == "BUY" else now_price.bid
    point = mt5.symbol_info(SYMBOL).point

    if PAPER_MODE:
        # simulate entry
        entry = {
            "id": len(sim_positions)+1,
            "action": action,
            "entry_price": price,
            "volume": volume,
            "sl": price - SL_POINTS*point if action == "BUY" else price + SL_POINTS*point,
            "tp": price + TP_POINTS*point if action == "BUY" else price - TP_POINTS*point,
            "open_time": datetime.now()
        }
        sim_positions.append(entry)
        last_trade_time = time.time()
        print(f"{datetime.now().strftime('%H:%M:%S')} [PAPER] Open {action} at {price} vol={volume}")
        return entry["id"]
    else:
        # real order
        order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
        deviation = 50
        sl = price - SL_POINTS*point if action == "BUY" else price + SL_POINTS*point
        tp = price + TP_POINTS*point if action == "BUY" else price - TP_POINTS*point
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
            "comment": "AutoAI",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        last_trade_time = time.time()
        if result is None:
            print("Order send returned None")
            return None
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print("Order failed:", result.retcode, result.comment)
            return None
        print(f"{datetime.now().strftime('%H:%M:%S')} ✅ Placed {action} at {price}, ticket: {result.order}")
        return result.order

# function to update simulated positions PnL and close if TP/SL hit or opposite signal
def update_sim_positions():
    global sim_positions, sim_equity, sim_max_equity, sim_drawdown
    if len(sim_positions) == 0:
        return
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return
    price = tick.bid
    closed = []
    for pos in list(sim_positions):
        if pos["action"] == "BUY":
            pnl = (price - pos["entry_price"]) * pos["volume"]
        else:
            pnl = (pos["entry_price"] - price) * pos["volume"]
        # check TP/SL
        if (pos["action"] == "BUY" and (price <= pos["sl"] or price >= pos["tp"])) or \
           (pos["action"] == "SELL" and (price >= pos["sl"] or price <= pos["tp"])):
            # close pos and realize PnL
            sim_equity += pnl
            sim_positions.remove(pos)
            closed.append((pos, pnl, price))
    # track drawdown
    if sim_equity > sim_max_equity:
        sim_max_equity = sim_equity
    drawdown = (sim_max_equity - sim_equity) / sim_max_equity
    sim_drawdown = drawdown

# close positions by id (paper)
def close_sim_position_by_id(pid):
    global sim_positions, sim_equity
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return False
    price = tick.bid
    for pos in list(sim_positions):
        if pos["id"] == pid:
            if pos["action"] == "BUY":
                pnl = (price - pos["entry_price"]) * pos["volume"]
            else:
                pnl = (pos["entry_price"] - price) * pos["volume"]
            sim_equity += pnl
            sim_positions.remove(pos)
            print(f"{datetime.now().strftime('%H:%M:%S')} [PAPER] Closed pos {pid} price={price} pnl={pnl:.2f}")
            return True
    return False

# helper to build model input row using model.feature_names_in_ if available
def build_model_input_row(feature_df):
    # take last row as Series
    s = feature_df.iloc[-1].copy()
    # convert to DataFrame of numeric columns
    numeric = s[s.apply(lambda x: isinstance(x, (int, float, np.integer, np.floating)))]
    row_df = pd.DataFrame([numeric])
    # align to model expected features
    try:
        cols = list(model.feature_names_in_)
        row_df = row_df.reindex(columns=cols, fill_value=0.0)
    except Exception:
        # fallback: keep numeric columns only (assumes scaler was fit similarly)
        pass
    return row_df

# MAIN LOOP
print("🔁 Auto-trade agent started (PAPER_MODE=%s). Press Ctrl+C to stop." % PAPER_MODE)
try:
    while True:
        # update speed and simulated positions
        avg_speed = update_speed(SYMBOL)
        update_sim_positions()

        # safety: stop trading if drawdown too big
        if PAPER_MODE and sim_drawdown > MAX_DRAWDOWN_PCT:
            print("⚠️ Simulated drawdown > limit. Pausing trading.")
            time.sleep(REFRESH_SECONDS)
            continue

        # check enough candles
        candles = get_candles(SYMBOL, n=300)
        if candles is None or len(candles) < 60:
            print("Not enough candles, waiting...")
            time.sleep(REFRESH_SECONDS)
            continue

        features = make_features(candles)
        if features is None or features.shape[0] == 0:
            print("No features (after dropna), waiting...")
            time.sleep(REFRESH_SECONDS)
            continue

        # prepare input, scale
        input_df = build_model_input_row(features)
        # ensure all numeric columns exist, replace NaN
        input_df = input_df.fillna(0.0)
        # transform using scaler
        try:
            X_scaled = scaler.transform(input_df)  # matches fit columns
        except Exception:
            # fallback: try aligning columns with scaler.feature_names_in_
            try:
                X_aligned = input_df.reindex(columns=scaler.feature_names_in_, fill_value=0.0)
                X_scaled = scaler.transform(X_aligned)
            except Exception as e:
                print("Scaler transform error:", e)
                time.sleep(REFRESH_SECONDS)
                continue

        # prediction
        pred = model.predict(X_scaled)[0]
        prob = None
        try:
            prob = model.predict_proba(X_scaled)[0].max()
        except Exception:
            prob = None

        # trading logic + checks
        now_ts = time.time()
        open_cnt = get_open_positions_count(SYMBOL)
        can_trade = (avg_speed is not None) and (avg_speed <= SPEED_PAUSE_THRESH) \
                    and (now_ts - last_trade_time > COOLDOWN_SECONDS) and (open_cnt < MAX_OPEN_TRADES)

        action = None
        if can_trade and prob is not None and prob >= MIN_CONFIDENCE:
            action = "BUY" if pred == 1 else "SELL"
            # check spread
            sym = mt5.symbol_info(SYMBOL)
            if sym is None:
                print("Symbol info missing, skip trade")
                action = None
            else:
                spread = sym.ask - sym.bid
                max_spread = sym.point * 100
                if spread > max_spread:
                    print(f"Spread {spread:.5f} > max {max_spread:.5f} -> skip")
                    action = None

        # place (or simulate) order
        ticket = None
        if action:
            ticket = place_order(action, volume=LOT)

        # log
        tick = mt5.symbol_info_tick(SYMBOL)
        price_now = tick.bid if tick is not None else np.nan
        log_row = {
            "timestamp": datetime.now(),
            "price": price_now,
            "prediction": int(pred),
            "prob": float(prob) if prob is not None else None,
            "action": action if action is not None else "HOLD",
            "speed": float(avg_speed) if avg_speed is not None else None,
            "open_positions": get_open_positions_count(SYMBOL),
            "equity": sim_equity if PAPER_MODE else None,
            "real_trade_ticket": ticket,
            "notes": ""
        }
        pd.DataFrame([log_row]).to_csv(LOG_FILE, mode='a', header=False, index=False)

        # print summary
        proba_str = f"{prob:.2f}" if prob is not None else "N/A"
        status = "ALLOWED" if (avg_speed is not None and avg_speed <= SPEED_PAUSE_THRESH) else "PAUSED"
        print(f"{datetime.now().strftime('%H:%M:%S')} | {status} | Pred:{pred} | P:{proba_str} | Act:{log_row['action']} | Open:{log_row['open_positions']} | Eq:{sim_equity:.2f}")

        time.sleep(REFRESH_SECONDS)

except KeyboardInterrupt:
    print("\n⛔ Stopped by user")
finally:
    mt5.shutdown()
    print("MT5 shutdown complete")
