import numpy as np
import pandas as pd
import joblib
import time
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize

import xgboost as xgb
import lightgbm as lgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HYBRID_DATA_PATH  = "data/processed/hybrid_dataset.csv"
HIST_DATA_PATH    = "data/processed/disaster_features.csv"
MODEL_SAVE_PATH   = "models/priority_model.pkl"
THRESHOLD_PATH    = "models/severity_thresholds.pkl"
REPORT_SAVE_PATH  = "models/comparison_report.csv"
PLOT_SAVE_PATH    = "models/comparison_plot.png"
RANDOM_STATE      = 42
TEST_SIZE         = 0.2
CV_FOLDS          = 5
CLASSES           = [0, 1, 2]
CLASS_NAMES       = ["Low", "Medium", "High"]

FEATURE_COLS = ["log_deaths", "log_injured", "log_affected", "log_damage"]
TARGET_COL   = "severity_level"


def compute_score(df):
    return (
        0.4  * df["log_deaths"]
        + 0.25 * df["log_affected"]
        + 0.2  * df["log_injured"]
        + 0.15 * df["log_damage"]
    )


def get_thresholds_from_real_data():
    """Compute thresholds from the original real data, not the augmented set."""
    df_real = pd.read_csv(HIST_DATA_PATH)
    scores  = compute_score(df_real)
    low_t   = float(scores.quantile(0.33))
    high_t  = float(scores.quantile(0.66))
    print(f"   Thresholds from real data: Low≤{low_t:.4f}, Med≤{high_t:.4f}, High>{high_t:.4f}")
    return low_t, high_t


def apply_labels(df, low_t, high_t):
    """Assign severity labels using fixed thresholds derived from real data."""
    df = df.copy()
    scores = compute_score(df)
    df[TARGET_COL] = scores.apply(
        lambda s: 0 if s <= low_t else (1 if s <= high_t else 2)
    )
    return df


def get_models():
    return {
        "RandomForest (baseline)": RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        ),
    }


def load_data():
    path = HYBRID_DATA_PATH if Path(HYBRID_DATA_PATH).exists() else HIST_DATA_PATH
    print(f"📂 Loading: {path}")
    df = pd.read_csv(path)

    # Get thresholds from clean real data, then apply to whole dataset
    low_t, high_t = get_thresholds_from_real_data()
    df = apply_labels(df, low_t, high_t)

    # Save thresholds for use at inference time
    Path(THRESHOLD_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"low_t": low_t, "high_t": high_t}, THRESHOLD_PATH)
    print(f"💾 Thresholds saved → {THRESHOLD_PATH}")

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    print(f"   Rows: {len(df):,}  |  Classes: { {CLASS_NAMES[i]: int((y==i).sum()) for i in CLASSES} }\n")

    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def evaluate(model, X_train, X_test, y_train, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred, average="macro")
    y_bin   = label_binarize(y_test, classes=CLASSES)
    roc_auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
    cv      = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_f1   = cross_val_score(model, X_train, y_train, scoring="f1_macro", cv=cv, n_jobs=-1).mean()
    return {
        "Accuracy":              round(acc, 4),
        "F1-macro":              round(f1, 4),
        "ROC-AUC":               round(roc_auc, 4),
        f"CV F1 ({CV_FOLDS}-fold)": round(cv_f1, 4),
    }


def train_all(X_train, X_test, y_train, y_test):
    models, results, trained = get_models(), {}, {}
    print("=" * 60)
    for name, model in models.items():
        print(f"\n🏋️  Training: {name}")
        t0 = time.time()
        model.fit(X_train, y_train)
        print(f"   Done in {time.time()-t0:.1f}s")
        metrics = evaluate(model, X_train, X_test, y_train, y_test)
        results[name] = metrics
        trained[name] = model
        for k, v in metrics.items():
            print(f"   {k:<22}: {v:.4f}")
    print("=" * 60)
    return trained, results


def summarise(results):
    df = pd.DataFrame(results).T
    df.index.name = "Model"
    df["Score"] = df["F1-macro"] * 0.6 + df["ROC-AUC"] * 0.4
    df = df.sort_values("Score", ascending=False)
    winner = df.index[0]
    print(f"\n🏆 Best model : {winner}")
    print(f"   F1-macro   : {df.loc[winner, 'F1-macro']:.4f}")
    print(f"   ROC-AUC    : {df.loc[winner, 'ROC-AUC']:.4f}\n")
    return winner, df


def save_best(trained, winner, feature_names):
    Path(MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    best = trained[winner]
    try:
        best.feature_names_in_ = np.array(feature_names)
    except AttributeError:
        pass
    joblib.dump({"model": best, "feature_names": feature_names}, MODEL_SAVE_PATH)
    print(f"💾 Model saved → {MODEL_SAVE_PATH}")


def plot_comparison(df_results):
    metrics = ["Accuracy", "F1-macro", "ROC-AUC"]
    df_plot = df_results[metrics].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_plot))
    width = 0.25
    colors = ["#3B8BD4", "#1D9E75", "#D85A30"]
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + i * width, df_plot[metric], width, label=metric, color=color, alpha=0.88)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h+0.005, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(df_plot.index, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — Disaster Priority Classification\n(Fixed Thresholds from Real Data)", fontsize=12)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, dpi=150)
    print(f"📈 Chart saved → {PLOT_SAVE_PATH}")


def run_comparison():
    print("\n🚀 DRDS Model Training\n")
    X_train, X_test, y_train, y_test = load_data()
    trained, results = train_all(X_train, X_test, y_train, y_test)
    winner, df_results = summarise(results)

    for name, model in trained.items():
        y_pred = model.predict(X_test)
        print(f"\n── {name} ──")
        print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    save_best(trained, winner, FEATURE_COLS)
    plot_comparison(df_results)
    df_results.to_csv(REPORT_SAVE_PATH)
    print(f"\n✅ Done. Best model: {winner}")


if __name__ == "__main__":
    run_comparison()