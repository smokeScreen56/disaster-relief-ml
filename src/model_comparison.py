"""
Phase 2 — Multi-Model Training & Comparison
=============================================
Trains RandomForest (baseline), XGBoost, and LightGBM on the hybrid dataset,
compares them on F1-macro, Accuracy, and ROC-AUC, prints a summary table,
and saves the best model as models/priority_model.pkl.

Usage:
    python src/model_comparison.py

Dependencies:
    pip install xgboost lightgbm scikit-learn pandas numpy tabulate
"""

import numpy as np
import pandas as pd
import joblib
import time
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize

import xgboost as xgb
import lightgbm as lgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
HYBRID_DATA_PATH  = "data/processed/hybrid_dataset.csv"
HIST_DATA_PATH    = "data/processed/disaster_features.csv"   # fallback
MODEL_SAVE_PATH   = "models/priority_model.pkl"
REPORT_SAVE_PATH  = "models/comparison_report.csv"
PLOT_SAVE_PATH    = "models/comparison_plot.png"
RANDOM_STATE      = 42
TEST_SIZE         = 0.2
CV_FOLDS          = 5
CLASSES           = [0, 1, 2]
CLASS_NAMES       = ["Low", "Medium", "High"]

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
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────
def get_models():
    return {
        "RandomForest (baseline)": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> tuple:
    path = HYBRID_DATA_PATH if Path(HYBRID_DATA_PATH).exists() else HIST_DATA_PATH
    print(f"📂 Loading data from: {path}")
    df = pd.read_csv(path)

    # Only keep columns that actually exist (hybrid has 'source', historical may not)
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features].values
    y = df[TARGET_COL].values

    print(f"   Rows: {len(df):,}  |  Features: {len(available_features)}")
    print(f"   Class distribution: { {CLASS_NAMES[i]: int((y==i).sum()) for i in CLASSES} }\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return X_train, X_test, y_train, y_test, available_features


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, X_train, X_test, y_train, y_test) -> dict:
    # Hold-out metrics
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred, average="macro")

    # ROC-AUC (OvR, macro)
    y_bin   = label_binarize(y_test, classes=CLASSES)
    roc_auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")

    # Cross-validated F1 (more reliable on smaller datasets)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_f1 = cross_val_score(model, X_train, y_train,
                             scoring="f1_macro", cv=cv, n_jobs=-1).mean()

    return {
        "Accuracy":   round(acc, 4),
        "F1-macro":   round(f1, 4),
        "ROC-AUC":    round(roc_auc, 4),
        f"CV F1 ({CV_FOLDS}-fold)": round(cv_f1, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_all(X_train, X_test, y_train, y_test) -> tuple:
    models    = get_models()
    results   = {}
    trained   = {}

    print("=" * 60)
    for name, model in models.items():
        print(f"\n🏋️  Training: {name}")
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"   Done in {elapsed:.1f}s")

        metrics = evaluate(model, X_train, X_test, y_train, y_test)
        results[name] = metrics
        trained[name] = model

        for k, v in metrics.items():
            print(f"   {k:<22}: {v:.4f}")

    print("=" * 60)
    return trained, results


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table + winner
# ─────────────────────────────────────────────────────────────────────────────
def summarise(results: dict) -> str:
    df = pd.DataFrame(results).T
    df.index.name = "Model"

    # Rank by F1-macro (primary) then ROC-AUC (tiebreak)
    df["Score"] = df["F1-macro"] * 0.6 + df["ROC-AUC"] * 0.4
    df = df.sort_values("Score", ascending=False)
    winner = df.index[0]

    print("\n📊 COMPARISON SUMMARY")
    print("─" * 74)
    try:
        from tabulate import tabulate
        print(tabulate(df.drop(columns="Score"), headers="keys", tablefmt="github", floatfmt=".4f"))
    except ImportError:
        print(df.drop(columns="Score").to_string())
    print("─" * 74)
    print(f"\n🏆 Best model : {winner}")
    print(f"   F1-macro   : {df.loc[winner, 'F1-macro']:.4f}")
    print(f"   ROC-AUC    : {df.loc[winner, 'ROC-AUC']:.4f}\n")
    return winner, df


# ─────────────────────────────────────────────────────────────────────────────
# Per-model classification report
# ─────────────────────────────────────────────────────────────────────────────
def print_reports(trained: dict, X_test, y_test) -> None:
    print("\n📋 PER-CLASS CLASSIFICATION REPORTS")
    for name, model in trained.items():
        y_pred = model.predict(X_test)
        print(f"\n── {name} ──")
        print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))


# ─────────────────────────────────────────────────────────────────────────────
# Save best model
# ─────────────────────────────────────────────────────────────────────────────
def save_best(trained: dict, winner: str, feature_names: list) -> None:
    Path(MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    best_model = trained[winner]

    # Store feature names on model object for decision_engine.py compatibility
    best_model.feature_names_in_ = np.array(feature_names)

    joblib.dump(best_model, MODEL_SAVE_PATH)
    print(f"💾 Saved best model ({winner}) → {MODEL_SAVE_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_comparison(df_results: pd.DataFrame) -> None:
    metrics = ["Accuracy", "F1-macro", "ROC-AUC"]
    df_plot = df_results[metrics].copy()

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(df_plot))
    width = 0.25
    colors = ["#3B8BD4", "#1D9E75", "#D85A30"]

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + i * width, df_plot[metric], width, label=metric,
                      color=color, alpha=0.88)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(df_plot.index, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison — Disaster priority classification")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    Path(PLOT_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_SAVE_PATH, dpi=150)
    print(f"📈 Comparison chart saved → {PLOT_SAVE_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Save CSV report
# ─────────────────────────────────────────────────────────────────────────────
def save_report(df_results: pd.DataFrame) -> None:
    df_results.to_csv(REPORT_SAVE_PATH)
    print(f"📄 Report saved → {REPORT_SAVE_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run_comparison():
    print("\n🚀 Phase 2 — Multi-model training & comparison\n")

    X_train, X_test, y_train, y_test, feature_names = load_data()
    trained, results = train_all(X_train, X_test, y_train, y_test)
    winner, df_results = summarise(results)

    print_reports(trained, X_test, y_test)
    save_best(trained, winner, feature_names)
    plot_comparison(df_results)
    save_report(df_results)

    print("\n✅ Phase 2 complete.")
    print(f"   Next step: update decision_engine.py to load from {MODEL_SAVE_PATH}")
    print(f"   The best model is: {winner}\n")


if __name__ == "__main__":
    run_comparison()
