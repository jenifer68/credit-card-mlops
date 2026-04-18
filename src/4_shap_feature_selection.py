# ============================================================
# Student: FSB32-Minh Ha
# Exercise: Final Project 
# ============================================================

import os
import pandas as pd
import numpy as np
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================
DATA_DIR = "dataset"
OUTPUT_DIR = "model_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

train = pd.read_csv(f"{DATA_DIR }/card_train_fe.csv")
valid = pd.read_csv(f"{DATA_DIR }/card_valid_fe.csv")

TARGET = "target"

X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
X_valid, y_valid = valid.drop(columns=[TARGET]), valid[TARGET]

print("Train shape:", X_train.shape)
print("Valid shape:", X_valid.shape)

# =========================
# TRAIN LIGHTGBM (RE-TRAIN FOR SHAP)
# =========================
lgb_train = lgb.Dataset(X_train, y_train)
lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 42
}

model = lgb.train(
    params,
    lgb_train,
    num_boost_round=800,
    valid_sets=[lgb_valid],
    callbacks=[lgb.early_stopping(100)]
)

# =========================
# SHAP EXPLAINABILITY
# =========================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_valid)

# =========================
# GLOBAL FEATURE IMPORTANCE (SHAP)
# =========================
shap_abs_mean = np.abs(shap_values).mean(axis=0)

shap_importance = pd.DataFrame({
    "feature": X_valid.columns,
    "mean_abs_shap": shap_abs_mean
}).sort_values("mean_abs_shap", ascending=False)

print("\n=== TOP FEATURES BY SHAP ===")
print(shap_importance.head(33))

# =========================
# SAVE FEATURE IMPORTANCE
# =========================
shap_importance.to_csv(f"{OUTPUT_DIR}/shap_feature_importance.csv", index=False)

# =========================
# PLOTS
# =========================
shap.summary_plot(shap_values, X_valid, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_importance_bar.png")
plt.close()

shap.summary_plot(shap_values, X_valid, show=False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/shap_summary_beeswarm.png")
plt.close()

print("\nSHAP plots saved:")
print("- shap_importance_bar.png")
print("- shap_summary_beeswarm.png")
