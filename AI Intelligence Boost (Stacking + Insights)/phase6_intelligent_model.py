import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# ===============================
# 1️⃣ Load Optimized Data
# ===============================
df = pd.read_csv("XAUUSD_Features.csv")
print("✅ Data loaded:", df.shape)

# Select only numeric columns (avoid 'time' or other strings)
X = df.select_dtypes(include=['float64', 'int64'])
y = df["Target"]

# Split train/test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print("🔹 Training data:", X_train.shape)
print("🔹 Testing data:", X_test.shape)

# ===============================
# 2️⃣ Scaling
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 3️⃣ Build Stacked Ensemble
# ===============================
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42))
]

stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=GradientBoostingClassifier(),
    n_jobs=-1
)

print("🤖 Training stacked ensemble model...")
stack_model.fit(X_train_scaled, y_train)
print("✅ Stacking model training complete!")

# ===============================
# 4️⃣ Evaluation
# ===============================
y_pred = stack_model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n🎯 Ensemble Accuracy: {acc:.4f}")
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", cm)

# ===============================
# 5️⃣ Visualization
# ===============================
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix (Accuracy={acc:.2f})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ===============================
# 6️⃣ Feature Importance (RandomForest)
# ===============================
rf = base_models[0][1]
rf.fit(X_train_scaled, y_train)
importance = pd.Series(rf.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x=importance[:10], y=importance.index[:10], palette='viridis')
plt.title("Top 10 Important Features")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# ===============================
# 7️⃣ Save Models
# ===============================
joblib.dump(stack_model, "AI_Trading_Stacked_Model.pkl")
joblib.dump(scaler, "AI_Trading_Scaler.pkl")

print("\n💾 Ensemble model and scaler saved successfully!")
