"""
Phase 1 — Hybrid Synthetic Data Pipeline
=========================================
Generates synthetic disaster records using CTGAN, balances class distribution
using SMOTE, validates quality with KS-test, and merges with historical data.

Usage:
    python src/synthetic_data.py

Output:
    data/processed/hybrid_dataset.csv   ← use this for model training
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ── SDV / CTGAN ────────────────────────────────────────────────────────────
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# ── Imbalanced-learn ─────────────────────────────────────────────────────────
from imblearn.over_sampling import SMOTE

# ── Your existing pipeline ────────────────────────────────────────────────────
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.preprocess import load_and_clean_data
from src.feature_engineering import add_severity_score, add_severity_level


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
RAW_DATA_PATH        = "data/raw/emdat.xlsx"
OUTPUT_PATH          = "data/processed/hybrid_dataset.csv"
SYNTHETIC_RATIO      = 0.5      # synthetic rows = 50 % of historical count
CTGAN_EPOCHS         = 300      # increase to 500+ for better quality (slower)
RANDOM_STATE         = 42
KS_ALPHA             = 0.05     # columns with p < this get a warning

NUMERIC_COLS = [
    "Total Deaths",
    "No. Injured",
    "No. Affected",
    "No. Homeless",
    "Total Affected",
    "Total Damage ('000 US$)",
]

FEATURE_COLS = [
    "Start Year",
    "Total Deaths",
    "No. Injured",
    "No. Affected",
    "No. Homeless",
    "Total Affected",
    "Total Damage ('000 US$)",
    "log_deaths",
    "log_injured",
    "log_affected",
    "log_damage",
]

TARGET_COL = "severity_level"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load & engineer features on historical data
# ─────────────────────────────────────────────────────────────────────────────
def load_historical(path: str) -> pd.DataFrame:
    print("📂 Loading historical data …")
    df = load_and_clean_data(path)
    df = add_severity_score(df)
    df = add_severity_level(df)
    print(f"   Historical records : {len(df):,}")
    print(f"   Class distribution :\n{df[TARGET_COL].value_counts().sort_index().rename({0:'Low',1:'Medium',2:'High'})}\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Train CTGAN and generate synthetic rows
# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic(df_hist: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    print(f"🤖 Training CTGAN on {len(df_hist):,} rows ({CTGAN_EPOCHS} epochs) …")

    # CTGAN works on the raw numeric columns + target (before log transforms)
    train_cols = NUMERIC_COLS + ["Start Year", TARGET_COL]
    df_train = df_hist[train_cols].copy()

    # Build SDV metadata (all numeric)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_train)
    # Override types that SDV sometimes misdetects
    for col in NUMERIC_COLS:
        metadata.update_column(column_name=col, sdtype="numerical")
    metadata.update_column(column_name="Start Year",   sdtype="numerical")
    metadata.update_column(column_name=TARGET_COL,     sdtype="categorical")

    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=CTGAN_EPOCHS,
        verbose=True,
    )
    synthesizer.fit(df_train)

    print(f"\n✨ Sampling {n_rows:,} synthetic rows …")
    df_syn = synthesizer.sample(num_rows=n_rows)

    # Clip negatives (CTGAN can sometimes produce them for skewed columns)
    for col in NUMERIC_COLS:
        df_syn[col] = df_syn[col].clip(lower=0)

    df_syn["Start Year"] = df_syn["Start Year"].round().astype(int).clip(1900, 2024)

    # Re-derive log features so synthetic data matches historical pipeline exactly
    df_syn["log_deaths"]   = np.log1p(df_syn["Total Deaths"])
    df_syn["log_injured"]  = np.log1p(df_syn["No. Injured"])
    df_syn["log_affected"] = np.log1p(df_syn["Total Affected"])
    df_syn["log_damage"]   = np.log1p(df_syn["Total Damage ('000 US$)"])

    df_syn["source"] = "synthetic"
    print(f"   Synthetic class distribution :\n{df_syn[TARGET_COL].value_counts().sort_index().rename({0:'Low',1:'Medium',2:'High'})}\n")
    return df_syn


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Validate quality with KS-test (per numeric column)
# ─────────────────────────────────────────────────────────────────────────────
def validate_quality(df_hist: pd.DataFrame, df_syn: pd.DataFrame) -> None:
    print("🔍 KS-test: comparing real vs synthetic distributions …")
    print(f"   {'Column':<35} {'KS stat':>8}  {'p-value':>8}  {'Status':>8}")
    print("   " + "─" * 68)
    all_pass = True
    for col in NUMERIC_COLS:
        ks_stat, p_val = stats.ks_2samp(df_hist[col].dropna(), df_syn[col].dropna())
        status = "✅ OK" if p_val >= KS_ALPHA else "⚠️  WARN"
        if p_val < KS_ALPHA:
            all_pass = False
        print(f"   {col:<35} {ks_stat:>8.4f}  {p_val:>8.4f}  {status}")
    print()
    if all_pass:
        print("   All columns pass KS-test — synthetic data looks realistic.\n")
    else:
        print("   Some columns flagged. Consider increasing CTGAN_EPOCHS or checking skew.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Merge historical + synthetic into a hybrid dataset
# ─────────────────────────────────────────────────────────────────────────────
def merge_datasets(df_hist: pd.DataFrame, df_syn: pd.DataFrame) -> pd.DataFrame:
    df_hist = df_hist.copy()
    df_hist["source"] = "historical"

    keep_cols = FEATURE_COLS + [TARGET_COL, "source"]
    df_merged = pd.concat(
        [df_hist[keep_cols], df_syn[keep_cols]],
        ignore_index=True
    )
    print(f"🔗 Merged dataset : {len(df_merged):,} rows")
    print(f"   Historical : {(df_merged['source']=='historical').sum():,}")
    print(f"   Synthetic  : {(df_merged['source']=='synthetic').sum():,}\n")
    return df_merged


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — SMOTE to balance class distribution
# ─────────────────────────────────────────────────────────────────────────────
def apply_smote(df: pd.DataFrame) -> pd.DataFrame:
    print("⚖️  Applying SMOTE to balance severity classes …")
    print(f"   Before SMOTE :\n{df[TARGET_COL].value_counts().sort_index().rename({0:'Low',1:'Medium',2:'High'})}\n")

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X, y)

    df_balanced = pd.DataFrame(X_res, columns=FEATURE_COLS)
    df_balanced[TARGET_COL] = y_res
    df_balanced["source"] = "smote_augmented"

    print(f"   After SMOTE :\n{pd.Series(y_res).value_counts().sort_index().rename({0:'Low',1:'Medium',2:'High'})}\n")
    return df_balanced


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Save
# ─────────────────────────────────────────────────────────────────────────────
def save(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"💾 Saved hybrid dataset → {path}")
    print(f"   Total rows : {len(df):,}  |  Columns : {list(df.columns)}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def build_hybrid_dataset(
    raw_path: str = RAW_DATA_PATH,
    output_path: str = OUTPUT_PATH,
    synthetic_ratio: float = SYNTHETIC_RATIO,
    apply_smote_balancing: bool = True,
) -> pd.DataFrame:

    # 1. Historical
    df_hist = load_historical(raw_path)

    # 2. Synthetic
    n_synthetic = int(len(df_hist) * synthetic_ratio)
    df_syn = generate_synthetic(df_hist, n_rows=n_synthetic)

    # 3. Validate
    validate_quality(df_hist, df_syn)

    # 4. Merge
    df_hybrid = merge_datasets(df_hist, df_syn)

    # 5. Balance with SMOTE
    if apply_smote_balancing:
        df_hybrid = apply_smote(df_hybrid)

    # 6. Save
    save(df_hybrid, output_path)

    return df_hybrid


if __name__ == "__main__":
    build_hybrid_dataset()
