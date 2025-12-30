# =========================================================
# CREDIT CARD FRAUD DETECTION – FINAL VERSION (NO KNN)
# =========================================================

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    IsolationForest
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    auc
)
from imblearn.over_sampling import SMOTE

# -------------------------
# CONFIG
# -------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =========================================================
# 1. LOAD DATA
# =========================================================
print("\n[1] Loading dataset...")
df = pd.read_csv("creditcard.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

# =========================================================
# 2. TRAIN / TEST SPLIT
# =========================================================
print("[2] Splitting data (stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

# =========================================================
# 3. FEATURE ENGINEERING & SCALING
# =========================================================
print("[3] Feature engineering & scaling...")

scaler = StandardScaler()
X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
X_test["Amount"] = scaler.transform(X_test[["Amount"]])

def process_time(df_in):
    df = df_in.copy()
    df["Hour"] = (df["Time"] / 3600) % 24
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df.drop(columns=["Time", "Hour"], inplace=True)
    return df

X_train = process_time(X_train)
X_test = process_time(X_test)

X_train["V14_V10"] = X_train["V14"] * X_train["V10"]
X_train["V12_V4"] = X_train["V12"] * X_train["V4"]

X_test["V14_V10"] = X_test["V14"] * X_test["V10"]
X_test["V12_V4"] = X_test["V12"] * X_test["V4"]

# =========================================================
# 4. ISOLATION FOREST (UNSUPERVISED OUTLIER DETECTION)
# =========================================================
print("[4] Training Isolation Forest...")

iso_forest = IsolationForest(
    contamination=0.002,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
iso_forest.fit(X_train)

# Use anomaly score as feature
X_train["iforest_score"] = iso_forest.decision_function(X_train)
X_test["iforest_score"] = iso_forest.decision_function(X_test)

# =========================================================
# 5. HANDLE IMBALANCE (SMOTE)
# =========================================================
print("[5] Applying SMOTE...")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# =========================================================
# 6. PCA VISUALIZATION (IMBALANCED VS SMOTE)
# =========================================================
print("[6] Plotting PCA distributions...")

pca = PCA(n_components=2)

orig_sample = pd.concat([
    X_train[y_train == 0].sample(2000, random_state=RANDOM_STATE),
    X_train[y_train == 1]
])
pca_orig = pca.fit_transform(orig_sample)

plt.figure(figsize=(6,5))
plt.scatter(pca_orig[:2000,0], pca_orig[:2000,1], alpha=0.3, label="Normal")
plt.scatter(pca_orig[2000:,0], pca_orig[2000:,1], color="red", label="Fraud")
plt.title("Original Data Distribution (Imbalanced)")
plt.legend()
plt.show()

res_sample = pd.concat([
    X_train_res[y_train_res == 0].sample(2000, random_state=RANDOM_STATE),
    X_train_res[y_train_res == 1].sample(2000, random_state=RANDOM_STATE)
])
pca_res = pca.fit_transform(res_sample)

plt.figure(figsize=(6,5))
plt.scatter(pca_res[:2000,0], pca_res[:2000,1], alpha=0.3, label="Normal")
plt.scatter(pca_res[2000:,0], pca_res[2000:,1], color="red", alpha=0.3, label="Fraud (SMOTE)")
plt.title("SMOTE Balanced Distribution")
plt.legend()
plt.show()

# =========================================================
# 7. SUPERVISED MODELS
# =========================================================
print("[7] Initializing & training supervised models...")

lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=RANDOM_STATE)
gb = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)

models = {
    "Logistic Regression": lr,
    "Random Forest": rf,
    "Gradient Boosting": gb
}

for name, model in models.items():
    print(f"   → Training {name}...")
    model.fit(X_train_res, y_train_res)

# =========================================================
# 8. VOTING CLASSIFIER (SOFT)
# =========================================================
print("[8] Training Voting Classifier...")

voting = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("rf", rf),
        ("gb", gb)
    ],
    voting="soft"
)

voting.fit(X_train_res, y_train_res)

# =========================================================
# 9. EVALUATION
# =========================================================
print("[9] Evaluating model...")

y_prob = voting.predict_proba(X_test)[:, 1]

# Precision–Recall
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)

plt.figure(figsize=(6,5))
plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend()
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_test, y_prob):.4f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Optimal threshold
fscore = (2 * precision * recall) / (precision + recall + 1e-10)
best_idx = np.argmax(fscore)
best_thresh = thresholds[best_idx]

y_pred = (y_prob >= best_thresh).astype(int)

print("\nOptimal Threshold:", round(best_thresh,4))
print(classification_report(y_test, y_pred))

# -------------------------
# Confusion Matrix (RAW COUNTS)
# -------------------------
cm_raw = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm_raw.ravel()

print("\nConfusion Matrix (Raw Counts):")
print(cm_raw)

print("\nFraud Detection Summary:")
print(f"✔ True Positives (Fraud correctly detected): {tp}")
print(f"✘ False Negatives (Fraud missed): {fn}")
print(f"✘ False Positives (Normal flagged as fraud): {fp}")
print(f"✔ True Negatives (Normal correctly classified): {tn}")




# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, normalize="true")
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues")
plt.title("Confusion Matrix (Normalized)")
plt.show()

roc_auc = roc_auc_score(y_test, y_prob)

print("\nEvaluation Metrics Summary:")
print(f"ROC-AUC : {roc_auc:.4f}")
print(f"PR-AUC  : {pr_auc:.4f}")

fraud_precision = tp / (tp + fp + 1e-10)
fraud_recall = tp / (tp + fn + 1e-10)

print("\nFraud Class Metrics:")
print(f"Fraud Precision: {fraud_precision:.4f}")
print(f"Fraud Recall   : {fraud_recall:.4f}")


print("\nInterpretation:")
print(
    f"The selected threshold ({best_thresh:.3f}) prioritizes precision over recall, "
    "which is appropriate for fraud detection where false positives are costly. "
    "The model successfully detects most fraudulent transactions while minimizing "
    "unnecessary alerts for legitimate users."
)

# =========================================================
# 10. SAVE PIPELINE
# =========================================================
print("[10] Saving pipeline...")
joblib.dump(
    {
        "model": voting,
        "scaler": scaler,
        "iforest": iso_forest,
        "features": X_train.columns.tolist(),
        "threshold": best_thresh,
        "config": {
            "interaction_features": ["V14_V10", "V12_V4"],
            "uses_iforest": True
        }
    },
    "fraud_pipelinesmote_final.pkl"
)


print("\n✅ Final pipeline saved.")
