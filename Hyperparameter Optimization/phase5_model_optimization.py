import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ===========================
# 1️⃣ Load Dataset
# ===========================
df = pd.read_csv("XAUUSD_Features.csv")
print("✅ Data loaded:", df.shape)

# ===========================
# 2️⃣ Feature Engineering
# ===========================
df["Body_Size"] = abs(df["close"] - df["open"])
df["Upper_Shadow"] = df["high"] - df[["close", "open"]].max(axis=1)
df["Lower_Shadow"] = df[["close", "open"]].min(axis=1) - df["low"]
df["Candle_Ratio"] = df["Body_Size"] / (df["high"] - df["low"] + 1e-6)
df["Momentum_5"] = df["close"].diff(5)
df["Volatility_5"] = df["close"].rolling(5).std()
df["SMA_Cross"] = np.where(df["SMA_10"] > df["SMA_30"], 1, 0)
df["Hour"] = pd.to_datetime(df["time"]).dt.hour
df["DayOfWeek"] = pd.to_datetime(df["time"]).dt.dayofweek

# Drop NaN values after rolling operations
df.dropna(inplace=True)

# ===========================
# 3️⃣ Split Data
# ===========================
X = df.drop(columns=["time", "Target"])
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print("🔹 Training data:", X_train.shape)
print("🔹 Testing data:", X_test.shape)

# ===========================
# 4️⃣ Scale Features
# ===========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===========================
# 5️⃣ Grid Search Optimization
# ===========================
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [6, 8, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

model = RandomForestClassifier(random_state=42, n_jobs=-1)

grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", verbose=1, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print("\n🏆 Best Parameters:", grid_search.best_params_)

# ===========================
# 6️⃣ Evaluate Optimized Model
# ===========================
y_pred = best_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Optimized Accuracy: {acc:.4f}")
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# ===========================
# 7️⃣ Save Model & Scaler
# ===========================
joblib.dump(best_model, "Trade_AI_Model_Optimized.pkl")
joblib.dump(scaler, "Scaler_Optimized.pkl")

print("\n💾 Optimized model and scaler saved successfully!")
