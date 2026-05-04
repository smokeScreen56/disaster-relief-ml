import numpy as np
import pandas as pd
import joblib
from src.utils import load_model

MODEL_PATH     = "models/priority_model.pkl"
THRESHOLD_PATH = "models/severity_thresholds.pkl"

FEATURE_NAMES = ["log_deaths", "log_injured", "log_affected", "log_damage"]
PRIORITY_MAP  = {0: "Low", 1: "Medium", 2: "High"}


def _get_feature_importances(model, feature_names):
    try:
        importances = model.feature_importances_
    except AttributeError:
        importances = np.ones(len(feature_names)) / len(feature_names)
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
    )


def predict_with_explanation(feature_dict: dict) -> dict:
    model, saved_names = load_model(MODEL_PATH)
    names = saved_names if saved_names else FEATURE_NAMES

    X = pd.DataFrame([feature_dict], columns=names)

    proba      = model.predict_proba(X)[0]
    prediction = int(np.argmax(proba))
    confidence = min(round(float(np.max(proba)), 2), 0.95)

    importance_df = _get_feature_importances(model, names)
    top_features  = importance_df.head(3)

    explanation = (
        f"The model classified this disaster as **{PRIORITY_MAP[prediction]} priority** "
        f"primarily due to the following contributing factors:"
    )

    reasons = [
        f"{row.feature} (importance: {row.importance:.2f})"
        for _, row in top_features.iterrows()
    ]

    return {
        "priority":      PRIORITY_MAP[prediction],
        "confidence":    confidence,
        "reasons":       reasons,
        "explanation":   explanation,
        "probabilities": {
            PRIORITY_MAP[i]: round(float(p), 3)
            for i, p in enumerate(proba)
        },
    }


if __name__ == "__main__":
    print("\n🧪 Testing all three severity levels:\n")
    tests = [
        ("LOW  — 10 deaths, 0 injured, 0 affected, $0 damage",
         {"log_deaths": np.log1p(10), "log_injured": np.log1p(0),
          "log_affected": np.log1p(0), "log_damage": np.log1p(0)}),
        ("MED  — 14 deaths, 0 injured, 0 affected, $600k damage",
         {"log_deaths": np.log1p(14), "log_injured": np.log1p(0),
          "log_affected": np.log1p(0), "log_damage": np.log1p(600000)}),
        ("HIGH — 420 deaths, 1300 injured, 80000 affected, $900k damage",
         {"log_deaths": np.log1p(420), "log_injured": np.log1p(1300),
          "log_affected": np.log1p(80000), "log_damage": np.log1p(900000)}),
    ]
    for label, features in tests:
        result = predict_with_explanation(features)
        print(f"{label}")
        print(f"  → Predicted: {result['priority']} (confidence: {result['confidence']})")
        print(f"  → Proba: {result['probabilities']}\n")