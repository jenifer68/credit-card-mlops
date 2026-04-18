
# ============================================================# 
# Student: FSB32-Minh Ha
# Exercise: Final Project 
# TRAIN CHALLENGERS – BASELINE vs XGBOOST
# ============================================================

import os
import time
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

# ============================================================
# CONFIG
# ============================================================
TARGET = "target"
DATA_DIR = "dataset"
OUTPUT_DIR = "model_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
train = pd.read_csv(f"{DATA_DIR}/card_train_fe.csv")
valid = pd.read_csv(f"{DATA_DIR}/card_valid_fe.csv")

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]

X_valid = valid.drop(columns=[TARGET])
y_valid = valid[TARGET]

print("Train shape:", train.shape, "Target rate:", y_train.mean())
print("Valid shape:", valid.shape, "Target rate:", y_valid.mean())

# ============================================================
# METRICS (BANKING STANDARD)
# ============================================================
def ks_stat(y_true, y_score):
    df = pd.DataFrame({"y": y_true, "score": y_score}).sort_values(
        "score", ascending=False
    )
    df["cum_pos"] = df["y"].cumsum() / df["y"].sum()
    df["cum_neg"] = (1 - df["y"]).cumsum() / (1 - df["y"]).sum()
    return np.max(np.abs(df["cum_pos"] - df["cum_neg"]))


def lift_at_k(y_true, y_score, k=0.1):
    df = pd.DataFrame({"y": y_true, "score": y_score}).sort_values(
        "score", ascending=False
    )
    top_k = int(len(df) * k)
    return df.iloc[:top_k]["y"].mean() / df["y"].mean()


# ============================================================
# MODEL 1 – LOGISTIC REGRESSION (BASELINE)
# ============================================================
start = time.time()

scaler = StandardScaler()
Xtr_s = scaler.fit_transform(X_train)
Xva_s = scaler.transform(X_valid)

logit = LogisticRegression(
    max_iter=500,
    solver="lbfgs"
)
logit.fit(Xtr_s, y_train)

pred_logit = logit.predict_proba(Xva_s)[:, 1]

auc_logit = roc_auc_score(y_valid, pred_logit)
ks_logit = ks_stat(y_valid, pred_logit)
lift_logit = lift_at_k(y_valid, pred_logit)

time_logit = time.time() - start

print("\n===== LOGISTIC REGRESSION (BASELINE) =====")
print(f"AUC        : {auc_logit:.4f}")
print(f"KS         : {ks_logit:.4f}")
print(f"Lift@10%   : {lift_logit:.2f}")
print(f"Train time : {time_logit:.2f}s")

# ============================================================
# MODEL 2 – XGBOOST (CHALLENGER)
# ============================================================
start = time.time()

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=42
)

xgb.fit(X_train, y_train)

pred_xgb = xgb.predict_proba(X_valid)[:, 1]

auc_xgb = roc_auc_score(y_valid, pred_xgb)
ks_xgb = ks_stat(y_valid, pred_xgb)
lift_xgb = lift_at_k(y_valid, pred_xgb)

time_xgb = time.time() - start

print("\n===== XGBOOST (CHALLENGER) =====")
print(f"AUC        : {auc_xgb:.4f}")
print(f"KS         : {ks_xgb:.4f}")
print(f"Lift@10%   : {lift_xgb:.2f}")
print(f"Train time : {time_xgb:.2f}s")

# ============================================================
# COMPARISON TABLE
# ============================================================
results = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Role": "Baseline",
        "AUC": auc_logit,
        "KS": ks_logit,
        "Lift@10%": lift_logit,
        "Train_time_sec": time_logit
    },
    {
        "Model": "XGBoost",
        "Role": "Challenger",
        "AUC": auc_xgb,
        "KS": ks_xgb,
        "Lift@10%": lift_xgb,
        "Train_time_sec": time_xgb
    }
])

results = results.sort_values("AUC", ascending=False)
results.to_csv(f"{OUTPUT_DIR}/challenger_results.csv", index=False)

print("\n========== CHALLENGER COMPARISON ==========")
print(results)

print("\n✅ Challenger pipeline completed")
print(f"📂 Output saved to: {OUTPUT_DIR}/challenger_results.csv")
