# ============================================================
# Student: FSB32-Minh Ha
# Exercise: Final Project 
# FULL ML PIPELINE – CHAMPION MODEL (LIGHTGBM)
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score, roc_curve
import lightgbm as lgb
import shap

# ============================================================
# CONFIG
# ============================================================
TARGET = "target"
DATASET_DIR = "dataset"   
OUTPUT_DIR = "model_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


plt.style.use("seaborn-v0_8")

# ============================================================
# LOAD DATA
# ============================================================
train = pd.read_csv(f"{DATASET_DIR}/card_train_fe.csv")
valid = pd.read_csv(f"{DATASET_DIR}/card_valid_fe.csv")
test  = pd.read_csv(f"{DATASET_DIR}/card_test_fe.csv")

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]

X_valid = valid.drop(columns=[TARGET])
y_valid = valid[TARGET]

X_test  = test.drop(columns=[TARGET])
y_test  = test[TARGET]

print(f"Train: {train.shape} | Target rate: {y_train.mean():.4f}")
print(f"Valid: {valid.shape} | Target rate: {y_valid.mean():.4f}")
print(f"Test : {test.shape}  | Target rate: {y_test.mean():.4f}")

# ============================================================
# METRIC FUNCTIONS (BANKING STANDARD)
# ============================================================
def ks_stat(y_true, y_pred):
    df = pd.DataFrame({"y": y_true, "score": y_pred}).sort_values("score", ascending=False)
    df["cum_pos"] = df["y"].cumsum() / df["y"].sum()
    df["cum_neg"] = (1 - df["y"]).cumsum() / (1 - df["y"]).sum()
    return np.max(np.abs(df["cum_pos"] - df["cum_neg"]))

def lift_at_k(y_true, y_pred, k=0.1):
    df = pd.DataFrame({"y": y_true, "score": y_pred}).sort_values("score", ascending=False)
    top_k = int(len(df) * k)
    return df.iloc[:top_k]["y"].mean() / df["y"].mean()

def decile_table_banking(y_true, y_pred):
    df = pd.DataFrame({"y": y_true, "score": y_pred})
    df["decile"] = pd.qcut(df["score"], 10, labels=False, duplicates="drop")

    overall_rate = df["y"].mean()

    dec = (
        df.groupby("decile", observed=False)
          .agg(
              total=("y", "count"),
              positives=("y", "sum")
          )
          .sort_index(ascending=False)
          .reset_index()
    )

    dec["negatives"] = dec["total"] - dec["positives"]
    dec["response_rate"] = dec["positives"] / dec["total"]
    dec["lift"] = dec["response_rate"] / overall_rate

    dec["cum_total"] = dec["total"].cumsum()
    dec["cum_positives"] = dec["positives"].cumsum()
    dec["cum_negatives"] = dec["negatives"].cumsum()

    dec["cum_total_pct"] = dec["cum_total"] / dec["total"].sum()
    dec["cum_positives_pct"] = dec["cum_positives"] / dec["positives"].sum()
    dec["cum_negatives_pct"] = dec["cum_negatives"] / dec["negatives"].sum()

    return dec

# ============================================================
# TRAIN LIGHTGBM (CHAMPION)
# ============================================================
lgb_train = lgb.Dataset(X_train, y_train)
lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42
}

model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_valid],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(100)]
)

model.save_model(f"{OUTPUT_DIR}/lightgbm_model.txt")

pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)

auc = roc_auc_score(y_valid, pred_valid)
ks  = ks_stat(y_valid, pred_valid)
lift10 = lift_at_k(y_valid, pred_valid, 0.1)

valid_scored = valid.copy()
valid_scored["propensity_score"] = pred_valid

valid_scored.to_csv(
    f"{OUTPUT_DIR}/card_valid_scored.csv",
    index=False
)


print("\n===== LIGHTGBM (CHAMPION) =====")
print(f"AUC      : {auc:.4f}")
print(f"KS       : {ks:.4f}")
print(f"Lift@10% : {lift10:.2f}")

# ============================================================
# MODEL SUMMARY TABLE
# ============================================================
summary = pd.DataFrame([{
    "Model": "LightGBM",
    "AUC": auc,
    "KS": ks,
    "Lift@10%": lift10
}])
summary.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)

# ============================================================
# ROC CURVE
# ============================================================
fpr, tpr, _ = roc_curve(y_valid, pred_valid)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0,1],[0,1],'k--')
plt.legend()
plt.title("ROC Curve – LightGBM")
plt.savefig(f"{OUTPUT_DIR}/roc_curve.png")
plt.close()

# ============================================================
# KS CURVE
# ============================================================
df_ks = pd.DataFrame({"y": y_valid, "score": pred_valid}).sort_values("score", ascending=False)
df_ks["cum_pos"] = df_ks["y"].cumsum() / df_ks["y"].sum()
df_ks["cum_neg"] = (1 - df_ks["y"]).cumsum() / (1 - df_ks["y"]).sum()

plt.figure()
plt.plot(df_ks["cum_pos"], label="Cum Positive")
plt.plot(df_ks["cum_neg"], label="Cum Negative")
plt.legend()
plt.title("KS Curve – LightGBM")
plt.savefig(f"{OUTPUT_DIR}/ks_curve.png")
plt.close()

# ============================================================
# DECILE + LIFT + GAIN
# ============================================================
decile = decile_table_banking(y_valid, pred_valid)
decile.to_csv(f"{OUTPUT_DIR}/decile_table.csv", index=False)

# Lift Curve
plt.figure()
plt.plot(decile.index + 1, decile["lift"], marker="o")
plt.axhline(1, linestyle="--", color="red")
plt.title("Lift Curve – LightGBM")
plt.xlabel("Decile")
plt.ylabel("Lift")
plt.savefig(f"{OUTPUT_DIR}/lift_curve.png")
plt.close()

# Cumulative Gain
plt.figure()
plt.plot(decile["cum_total_pct"], decile["cum_positives_pct"], marker="o")
plt.plot([0,1],[0,1],'k--')
plt.title("Cumulative Gain – LightGBM")
plt.xlabel("% Population")
plt.ylabel("% Positives Captured")
plt.savefig(f"{OUTPUT_DIR}/cumulative_gain.png")
plt.close()

# ============================================================
# SCORE DISTRIBUTION
# ============================================================
plt.figure()
sns.kdeplot(pred_valid[y_valid==1], label="Target = 1")
sns.kdeplot(pred_valid[y_valid==0], label="Target = 0")
plt.legend()
plt.title("Score Distribution – LightGBM")
plt.savefig(f"{OUTPUT_DIR}/score_distribution.png")
plt.close()

# ============================================================
# CORRELATION HEATMAP
# ============================================================
plt.figure(figsize=(10,8))
sns.heatmap(X_train.corr(), cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png")
plt.close()

# ============================================================
# SHAP
# ============================================================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_valid)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

shap_df = pd.DataFrame({
    "feature": X_valid.columns,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

shap_df.to_csv(f"{OUTPUT_DIR}/shap_feature_importance.csv", index=False)

shap.summary_plot(shap_values, X_valid, plot_type="bar", show=False)
plt.savefig(f"{OUTPUT_DIR}/shap_importance_bar.png")
plt.close()

shap.summary_plot(shap_values, X_valid, show=False)
plt.savefig(f"{OUTPUT_DIR}/shap_beeswarm.png")
plt.close()

print("\n✅ FULL CHAMPION PIPELINE COMPLETED")
print(f"📂 Outputs saved to: {OUTPUT_DIR}")
