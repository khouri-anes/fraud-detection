# =========================================================
# CREDIT CARD FRAUD DETECTION - STRICTLY ALIGNED VERSION
# =========================================================

# -------------------------
# 1. Imports
# -------------------------
import numpy as np
import pandas as pd

# Visualization (optional – NOT ASKED)
# import matplotlib.pyplot as plt
# import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Feature selection NOT ASKED
# from sklearn.feature_selection import VarianceThreshold

# Models
from sklearn.linear_model import LogisticRegression

# from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier

# Metrics
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    auc
)

# Imbalanced learning
from imblearn.over_sampling import SMOTE

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# 2. Load Dataset

df = pd.read_csv("creditcard.csv")

print("Class distribution:")
print(df["Class"].value_counts(normalize=True))


# 3. Feature Engineering (ASKED)


# Scale Amount
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
df.drop(columns=["Amount"], inplace=True)

# Time transformation (cyclical)
df["Hour"] = (df["Time"] / 3600) % 24
df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
df.drop(columns=["Time", "Hour"], inplace=True)

# PCA interactions
df["V14_V10"] = df["V14"] * df["V10"]
df["V12_V4"] = df["V12"] * df["V4"]


# 4. Train / Test Split

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)


# 5. Handle Imbalance (ASKED)

smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(pd.Series(y_train_res).value_counts())


# 6. Feature Selection (NOT ASKED → COMMENTED)

# selector = VarianceThreshold(threshold=0.01)
# X_train_res = selector.fit_transform(X_train_res)
# X_test_sel = selector.transform(X_test)

# Using raw features (no feature selection)
X_test_sel = X_test


# 7. Models (STRICTLY ASKED ONES)

print("🤖 Defining models...")

# Logistic Regression (INTERPRETABILITY)
lr = LogisticRegression(
    class_weight={0: 1, 1: 10},
    max_iter=1000,
    random_state=RANDOM_STATE
)

# Random Forest (PERFORMANCE)
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight={0: 1, 1: 10},
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Gradient Boosting (PERFORMANCE)
gb = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.05,
    random_state=RANDOM_STATE
)

models = {
    "Logistic Regression": lr,
    "Random Forest": rf,
    "Gradient Boosting": gb
}

print("✅ Models defined")

# 8. Train Models
print("🏋 Training models...")
for model in models.values():
    model.fit(X_train_res, y_train_res)
print("✅ All models trained")

# 9. Evaluation Function

def evaluate_model(model, X_test, y_test, name="Model"):
    print(f"\n📊 Evaluating {name}...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{name}")
    print("=" * 40)
    print(classification_report(y_test, y_pred))

    roc = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)

    print(f"ROC-AUC : {roc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}")


# 10. Evaluate Individual Models

for name, model in models.items():
    evaluate_model(model, X_test_sel, y_test, name)


# 11. Voting Classifier
voting = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("rf", rf),
        ("gb", gb)
    ],
    voting="soft"
)

voting.fit(X_train_res, y_train_res)
evaluate_model(voting, X_test_sel, y_test, "Voting Classifier")


# 12. Stacking Classifier

stacking = StackingClassifier(
    estimators=[
        ("rf", rf),
        ("gb", gb)
    ],
    final_estimator=LogisticRegression(
        class_weight={0: 1, 1: 10},
        max_iter=1000
    ),
    passthrough=True
)

stacking.fit(X_train_res, y_train_res)
evaluate_model(stacking, X_test_sel, y_test, "Stacking Classifier")

# 13. OUTLIER DETECTION

# Using One-Class SVM trained only on NORMAL transactions

ocsvm = OneClassSVM(
    kernel="rbf",
    nu=0.002,
    gamma="scale"
)

X_train_normal = X_train[y_train == 0]
ocsvm.fit(X_train_normal)

anomaly_pred = ocsvm.predict(X_test)
# -1 = anomaly (fraud candidate), +1 = normal

print("\n✅ One-Class SVM finished (outlier detection)")


# END OF PROJECT (STRICTLY ALIGNED)

