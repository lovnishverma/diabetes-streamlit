import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from joblib import dump

# Load dataset
data = pd.read_csv("dia.csv")

# Features & Target
X = data.drop(columns=["Outcome"])
y = data["Outcome"]

# Replace zero values in certain columns (except Pregnancies)
cols_with_zero = ["Glucose", "BloodPressure",
                  "SkinThickness", "Insulin", "BMI"]
X[cols_with_zero] = X[cols_with_zero].replace(0, np.nan)

# Save median values used for imputation
medians = X.median().to_dict()

# Fill missing with median
X = X.fillna(medians)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Balance dataset with SMOTE
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train_scaled, y_train)

# Train Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_bal, y_train_bal)

# Evaluation @ Default Threshold 0.5
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(" Default Threshold Accuracy:", acc)
print("\nClassification Report (Threshold=0.5):\n",
      classification_report(y_test, y_pred))

# Threshold Tuning
print("\n Threshold Tuning Results")
y_proba = model.predict_proba(X_test_scaled)[:, 1]

for thresh in [0.3, 0.4, 0.5, 0.6]:
    y_pred_thresh = (y_proba >= thresh).astype(int)
    acc_thresh = accuracy_score(y_test, y_pred_thresh)
    print(f"\nThreshold = {thresh}")
    print("Accuracy:", acc_thresh)
    print(classification_report(y_test, y_pred_thresh, digits=3))

# Save model, scaler & medians
os.makedirs("models", exist_ok=True)
dump(model, "models/diabetes.sav")
dump(scaler, "models/scaler.sav")
dump(medians, "models/medians.sav")

print("Final Model, Scaler, and Medians saved in 'models/' folder.")
