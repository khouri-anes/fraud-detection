# =========================================================
# CREDIT CARD FRAUD DETECTION – NEW TRANSACTION PREDICTION
# 3-CLASS DECISION (NORMAL / COULD BE FRAUD / FRAUD)
# =========================================================

import pandas as pd
import numpy as np
import joblib
import sys


# ---------------------------------------------------------
# 1. LOAD SAVED PIPELINE
# ---------------------------------------------------------

try:
    bundle = joblib.load("fraud_pipelinesmote_final.pkl")
except FileNotFoundError:
    print("❌ Error: 'fraud_pipelinesmote_final.pkl' not found.")
    sys.exit(1)

model = bundle["model"]
scaler = bundle["scaler"]
iforest = bundle["iforest"]
features = bundle["features"]
HIGH_THRESH = bundle["threshold"]   # Optimized (≈ 0.88)
LOW_THRESH = 0.50                   # Medium-risk threshold

print(f"✅ Loaded pipeline")
print(f"   LOW threshold  = {LOW_THRESH}")
print(f"   HIGH threshold = {HIGH_THRESH:.3f}")


# ---------------------------------------------------------
# 2. PREPROCESS NEW TRANSACTION
# ---------------------------------------------------------

def preprocess_new_tx(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Apply EXACT same preprocessing as training pipeline
    """

    df = tx.copy()

    # ---- Amount scaling
    df["Amount"] = scaler.transform(df[["Amount"]])

    # ---- Time features (if Time exists)
    if "Time" in df.columns:
        df["Hour"] = (df["Time"] / 3600) % 24
        df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
        df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
        df.drop(columns=["Time", "Hour"], inplace=True)

    # ---- Interaction features
    df["V14_V10"] = df["V14"] * df["V10"]
    df["V12_V4"] = df["V12"] * df["V4"]

    # ---- Isolation Forest score (NO Class column!)
    df["iforest_score"] = iforest.decision_function(df)

    # ---- Keep only trained features (order matters)
    df = df[features]

    return df


# ---------------------------------------------------------
# 3. PREDICTION FUNCTION (3 CLASSES)
# ---------------------------------------------------------

def predict(tx_proc: pd.DataFrame):
    prob = model.predict_proba(tx_proc)[0][1]

    if prob >= HIGH_THRESH:
        label = "🚨 FRAUD"
        action = "BLOCK TRANSACTION"
    elif prob >= LOW_THRESH:
        label = "⚠️ COULD BE FRAUD"
        action = "REVIEW / OTP / USER CONFIRMATION"
    else:
        label = "✅ NORMAL"
        action = "APPROVE"

    print(
        f"Fraud probability: {prob:.4f} | "
        f"Prediction: {label} | "
        f"Action: {action}"
    )


# ---------------------------------------------------------
# 4. SIMULATE NEW TRANSACTIONS
# ---------------------------------------------------------

print("\n📥 Loading original dataset for simulation...")
full_df = pd.read_csv("creditcard.csv")

def create_normal_transaction():
    base = (
        full_df[full_df["Class"] == 0]
        .sample(1)
        .drop(columns=["Class"])
    )
    noise = np.random.normal(0, 0.1, size=base.shape)
    return pd.DataFrame(base.values + noise, columns=base.columns)

def create_fraud_transaction():
    base = (
        full_df[full_df["Class"] == 1]
        .sample(1)
        .drop(columns=["Class"])
    )
    noise = np.random.normal(0, 0.3, size=base.shape)
    amplified = base.values * np.random.uniform(1.2, 1.8)
    return pd.DataFrame(amplified + noise, columns=base.columns)


# ---------------------------------------------------------
# 5. TEST PREDICTIONS
# ---------------------------------------------------------

print("\n🟢 Testing NEW NORMAL transactions")
for _ in range(3):
    tx = create_normal_transaction()
    tx_proc = preprocess_new_tx(tx)
    predict(tx_proc)

print("\n🚨 Testing NEW FRAUD transactions")
for _ in range(3):
    tx = create_fraud_transaction()
    tx_proc = preprocess_new_tx(tx)
    predict(tx_proc)
