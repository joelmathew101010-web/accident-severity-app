import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

df = pd.read_csv("Road_Accident_Data.csv")

# ---------- Feature Engineering ----------
df['Accident Date'] = pd.to_datetime(df['Accident Date'], format='%d/%m/%Y', errors='coerce')
df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour

cat_cols = ['Month', 'Day_of_Week', 'Junction_Control', 'Junction_Detail',
            'Light_Conditions', 'Road_Surface_Conditions', 'Road_Type',
            'Urban_or_Rural_Area', 'Weather_Conditions', 'Vehicle_Type',
            'Carriageway_Hazards']

num_cols = ['Year', 'Hour', 'Number_of_Vehicles', 'Number_of_Casualties', 'Speed_limit']

for c in cat_cols:
    df[c] = df[c].fillna('Unknown')
df['Hour'] = df['Hour'].fillna(df['Hour'].median())

target = 'Accident_Severity'

encoders = {}
X = pd.DataFrame(index=df.index)
for c in cat_cols:
    le = LabelEncoder()
    X[c] = le.fit_transform(df[c].astype(str))
    encoders[c] = le
for c in num_cols:
    X[c] = df[c]

y_le = LabelEncoder()
y = y_le.fit_transform(df[target])
encoders['__target__'] = y_le

feature_order = cat_cols + num_cols
X = X[feature_order]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

results = {}

# Baseline: Logistic Regression
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
results['LogisticRegression'] = {
    'accuracy': accuracy_score(y_test, pred_lr),
    'f1_macro': f1_score(y_test, pred_lr, average='macro')
}

# Random Forest
rf = RandomForestClassifier(
    n_estimators=120, max_depth=12, min_samples_leaf=10, class_weight='balanced',
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
results['RandomForest'] = {
    'accuracy': accuracy_score(y_test, pred_rf),
    'f1_macro': f1_score(y_test, pred_rf, average='macro')
}

print("=== Model Comparison ===")
for name, r in results.items():
    print(f"{name}: accuracy={r['accuracy']:.4f}  f1_macro={r['f1_macro']:.4f}")

print("\n=== RandomForest classification report ===")
print(classification_report(y_test, pred_rf, target_names=y_le.classes_))

print("\n=== Feature importances (RandomForest) ===")
importances = pd.Series(rf.feature_importances_, index=feature_order).sort_values(ascending=False)
print(importances)

# Save best model (RandomForest) + encoders + feature order
joblib.dump(rf, 'accident_severity_model.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(feature_order, 'feature_order.pkl')
print("\nSaved model, encoders, feature_order.")
