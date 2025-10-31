import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ============================
# 1️⃣ Load Processed Dataset
# ============================
df = pd.read_csv("XAUUSD_Features.csv")
print("✅ Data loaded successfully:", df.shape)

# Separate features and target
X = df.drop(columns=["Target"])
y = df["Target"]

# Drop non-numeric columns (e.g. Timestamp, Symbol)
X = X.select_dtypes(include=["float64", "int64"])
print("🔍 Numeric features selected:", X.columns.tolist())

# ============================
# 2️⃣ Split Train/Test Data
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
print("🔹 Training data:", X_train.shape)
print("🔹 Testing data:", X_test.shape)

# ============================
# 3️⃣ Feature Scaling
# ============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================
# 4️⃣ Train Random Forest
# ============================
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)
print("✅ Model training complete!")

# ============================
# 5️⃣ Evaluate Model
# ============================
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n🎯 Accuracy: {accuracy:.4f}")
print("\n📊 Confusion Matrix:\n", cm)
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# ============================
# 6️⃣ Save Model & Scaler
# ============================
joblib.dump(model, "Trade_AI_Model.pkl")
joblib.dump(scaler, "Scaler.pkl")

print("\n💾 Model and Scaler saved successfully!")
