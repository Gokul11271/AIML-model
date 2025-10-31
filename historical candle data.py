import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


# Initialize connection
symbol = "XAUUSD_"   # change if your broker uses XAUUSD or XAUUSD.
if not mt5.initialize():
    print("❌ MT5 initialization failed:", mt5.last_error())
    quit()
else:
    print("✅ Connected to MT5")

# Fetch 10,000 candles of 1-minute data
rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 10000)

if rates is None or len(rates) == 0:
    print("❌ No data received. Check symbol or MT5 connection.")
    mt5.shutdown()
    quit()

# Convert to DataFrame
data = pd.DataFrame(rates)
data['time'] = pd.to_datetime(data['time'], unit='s')

# Keep relevant columns
data = data[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

print("✅ Data fetched:", len(data), "rows")
print(data.head())

# Save for later use
data.to_csv("XAUUSD_History.csv", index=False)
print("💾 Data saved to XAUUSD_History.csv")

mt5.shutdown()
