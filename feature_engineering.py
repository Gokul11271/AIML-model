# ===========================
# PHASE 3: FEATURE ENGINEERING
# ===========================

import pandas as pd
import numpy as np

# --------------------------
# 1️⃣ Load historical data
# --------------------------
file_path = "XAUUSD_History.csv"
data = pd.read_csv(file_path)

print("✅ Data Loaded:", data.shape)
print(data.head())

# Ensure time column is datetime
data['time'] = pd.to_datetime(data['time'])

# --------------------------
# 2️⃣ Calculate Technical Indicators
# --------------------------

# Simple Moving Average (SMA)
data['SMA_10'] = data['close'].rolling(window=10).mean()
data['SMA_30'] = data['close'].rolling(window=30).mean()

# Exponential Moving Average (EMA)
data['EMA_10'] = data['close'].ewm(span=10, adjust=False).mean()
data['EMA_30'] = data['close'].ewm(span=30, adjust=False).mean()

# Relative Strength Index (RSI)
def compute_RSI(series, period=14):
    delta = series.diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI_14'] = compute_RSI(data['close'], 14)

# MACD (12,26,9)
ema12 = data['close'].ewm(span=12, adjust=False).mean()
ema26 = data['close'].ewm(span=26, adjust=False).mean()
data['MACD'] = ema12 - ema26
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

# Average True Range (ATR)
high_low = data['high'] - data['low']
high_close = np.abs(data['high'] - data['close'].shift())
low_close = np.abs(data['low'] - data['close'].shift())
ranges = pd.concat([high_low, high_close, low_close], axis=1)
true_range = np.max(ranges, axis=1)
data['ATR_14'] = true_range.rolling(window=14).mean()

# Percentage Change (returns)
data['Returns'] = data['close'].pct_change()

# --------------------------
# 3️⃣ Create Target Column
# --------------------------
data['Target'] = np.where(data['close'].shift(-1) > data['close'], 1, 0)

# --------------------------
# 4️⃣ Clean the Data
# --------------------------
data = data.dropna().reset_index(drop=True)

# --------------------------
# 5️⃣ Save the processed file
# --------------------------
output_file = "XAUUSD_Features.csv"
data.to_csv(output_file, index=False)
print(f"💾 Feature dataset saved to {output_file}")
print("✅ Feature Engineering Completed Successfully!")
print(data.head(10))
