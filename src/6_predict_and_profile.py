# ============================================================
# Student: FSB32-Minh Ha
# Exercise: Final Project 
# ============================================================


import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb

# ============================================================
# CONFIG
# ============================================================
TARGET = "target"
DATA_DIR = "dataset"
TOP_RATE = 0.10   # Top 10%
OUTPUT_DIR = "model_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


print("📂 Output directory:", OUTPUT_DIR)

# ============================================================
# LOAD DATA
# ============================================================
# Feature-engineered data (for prediction)
card_valid_fe = pd.read_csv(f"{DATA_DIR}/card_valid_fe.csv")

# Raw data (for business profiling)
card_valid_raw = pd.read_csv(f"{DATA_DIR}/card_valid.csv")

print("FE shape :", card_valid_fe.shape)
print("RAW shape:", card_valid_raw.shape)

# ============================================================
# LOAD MODEL
# ============================================================
model = lgb.Booster(model_file="model_outputs/lightgbm_model.txt")
print("✅ LightGBM model loaded")

# ============================================================
# PREDICT PROPENSITY SCORE
# ============================================================
X_valid = card_valid_fe.drop(columns=[TARGET])
pred_valid = model.predict(X_valid)

card_valid_raw["propensity_score"] = pred_valid

print("Avg score:", pred_valid.mean())
print("Max score:", pred_valid.max())

# ============================================================
# SELECT TOP X% CUSTOMERS
# ============================================================
top_df = (
    card_valid_raw
    .sort_values("propensity_score", ascending=False)
    .head(int(len(card_valid_raw) * TOP_RATE))
)

print(f"🎯 Top {int(TOP_RATE*100)}% customers:", top_df.shape)

# Save top customers list
top_df.to_csv(f"{OUTPUT_DIR}/top_propensity_customers.csv", index=False)

# ============================================================
# ===== CUSTOMER PROFILE VISUALIZATION =====
# ============================================================

# ------------------------------------------------------------
# 1️⃣ GENDER DISTRIBUTION
# ------------------------------------------------------------
if top_df["gender"].dtype != object:
    top_df["gender_label"] = top_df["gender"].map({0: "Female", 1: "Male"})
else:
    top_df["gender_label"] = top_df["gender"]

plt.figure(figsize=(5,4))
sns.countplot(data=top_df, x="gender_label")
plt.title("Gender Distribution – Top 10% Propensity")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/profile_gender.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# 2️⃣ AGE DISTRIBUTION
# ------------------------------------------------------------
plt.figure(figsize=(6,4))
sns.histplot(top_df["age"], bins=20, kde=True)
plt.title("Age Distribution – Top 10% Propensity")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/profile_age.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# 3️⃣ LOAN BEHAVIOR (Proxy)
# ------------------------------------------------------------
top_df["num_active_loans"] = (
    (top_df["max_loan_dsbr_amt"] > 0).astype(int) +
    (top_df["avg_loan_lmt"] > 0).astype(int)
)

plt.figure(figsize=(6,4))
sns.countplot(data=top_df, x="num_active_loans")
plt.title("Active Loan Count – Top 10% Propensity")
plt.xlabel("Number of Active Loans")
plt.ylabel("Customers")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/profile_active_loans.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# 4️⃣ CASA DISTRIBUTION
# ------------------------------------------------------------
plt.figure(figsize=(6,4))
sns.histplot(top_df["avg_casa_this_m"], bins=30)
plt.title("CASA Balance – Top 10% Propensity")
plt.xlabel("Avg CASA (This Month)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/profile_casa.png", dpi=150)
plt.close()

# ============================================================
# PROFILE SUMMARY TABLE (FOR PPT)
# ============================================================
summary = pd.DataFrame({
    "Metric": [
        "Avg Age",
        "Male Ratio",
        "Avg CASA",
        "Avg Loan Limit",
        "Avg Propensity Score"
    ],
    "Top 10% Customers": [
        top_df["age"].mean(),
        (top_df["gender_label"] == "Male").mean(),
        top_df["avg_casa_this_m"].mean(),
        top_df["avg_loan_lmt"].mean(),
        top_df["propensity_score"].mean()
    ],
    "Overall Population": [
        card_valid_raw["age"].mean(),
        (card_valid_raw["gender"].map({0:"Female",1:"Male"}) == "Male").mean(),
        card_valid_raw["avg_casa_this_m"].mean(),
        card_valid_raw["avg_loan_lmt"].mean(),
        card_valid_raw["propensity_score"].mean()
    ]
})

summary.to_csv(f"{OUTPUT_DIR}/profile_summary_table.csv", index=False)

print("\n✅ CUSTOMER PROFILING COMPLETED")
print("📊 Outputs generated:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if "profile_" in f or "top_propensity" in f:
        print(" -", f)
