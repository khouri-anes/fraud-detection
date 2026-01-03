
# CREDIT CARD FRAUD DETECTION – NEW TRANSACTION PREDICTION


import pandas as pd
import numpy as np
import joblib


# 1. Load Saved Pipeline

try:
    bundle = joblib.load("fraud_pipelinesmote_final.pkl")
except FileNotFoundError:
    print("Error: 'fraud_pipelinesmote_final.pkl' not found.")
    exit()

model = bundle["model"]
scaler = bundle["scaler"]
iforest = bundle["iforest"]
features = bundle["features"]
threshold = bundle["threshold"]
# threshold = 0.5


print(f"✅ Loaded pipeline. Using threshold: {threshold:.3f}")


# 2. Helper Functions


def preprocess_new_tx(tx):
    """Apply same preprocessing as training pipeline."""
    df = tx.copy()
    # Scale Amount
    df["Amount"] = scaler.transform(df[["Amount"]])
    # Time features (simulate Time if present)
    if "Time" in df.columns:
        df["Hour"] = (df["Time"] / 3600) % 24
        df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
        df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
        df.drop(columns=["Time", "Hour"], inplace=True)
    # Interaction features
    df["V14_V10"] = df["V14"] * df["V10"]
    df["V12_V4"] = df["V12"] * df["V4"]
    # Isolation Forest score
    df["iforest_score"] = iforest.decision_function(df)
    # Keep only features used during training
    df = df[features]
    return df

def predict(tx):
    """Predict fraud for a preprocessed transaction DataFrame."""
    prob = model.predict_proba(tx)[0][1]
    pred = int(prob >= threshold)
    status = "🚨 FRAUD DETECTED" if pred else "✅ NORMAL"
    print(f"Fraud probability: {prob:.4f} | Prediction: {status}")


# 3. Simulate New Transactions


# Load original dataset for sampling
full_df = pd.read_csv("creditcard.csv")

# Create new transactions
def create_normal_transaction():
    base = full_df[full_df['Class'] == 0].sample(
        1, random_state=np.random.randint(0, 10000)
    ).drop(columns=["Class"])
    noise = np.random.normal(0, 0.1, size=base.shape)
    return pd.DataFrame(base.values + noise, columns=base.columns)

def create_fraud_transaction():
    base = full_df[full_df['Class'] == 1].sample(
        1, random_state=np.random.randint(0, 10000)
    ).drop(columns=["Class"])
    noise = np.random.normal(0, 0.3, size=base.shape)
    amplified = base.values * np.random.uniform(1.2, 1.8)
    return pd.DataFrame(amplified + noise, columns=base.columns)


# 4. Test Predictions

print("\n🟢 Testing New NORMAL Transactions")
for _ in range(3):
    tx = create_normal_transaction()
    tx_proc = preprocess_new_tx(tx)
    predict(tx_proc)

print("\n🚨 Testing New FRAUD Transactions")
for _ in range(3):
    tx = create_fraud_transaction()
    tx_proc = preprocess_new_tx(tx)
    predict(tx_proc)
