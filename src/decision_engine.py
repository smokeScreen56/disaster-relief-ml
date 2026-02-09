import numpy as np
import pandas as pd
from src.utils import load_model

MODEL_PATH = "models/priority_model.pkl"

FEATURE_NAMES = [
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
    "severity_score",
]

PRIORITY_MAP = {0: "Low", 1: "Medium", 2: "High"}


# -------------------------
# Prediction + Explanation
# -------------------------
def predict_with_explanation(feature_dict):
    model = load_model(MODEL_PATH)

    X = pd.DataFrame([feature_dict], columns=FEATURE_NAMES)

    proba = model.predict_proba(X)[0]
    prediction = int(np.argmax(proba))
    confidence = min(round(float(np.max(proba)), 2), 0.95)

    
    # Feature importance (global but valid)
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    top_features = importance_df.head(3)

    explanation = (
        f"The model classified this disaster as **{PRIORITY_MAP[prediction]} priority** "
        f"primarily due to the following contributing factors:"
    )

    reasons = [
        f"{row.feature} (importance: {row.importance:.2f})"
        for _, row in top_features.iterrows()
    ]

    return {
        "priority": PRIORITY_MAP[prediction],
        "confidence": confidence,
        "reasons": reasons,
        "explanation": explanation,
    }


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":

    input_features = {
        "Start Year": 2018,
        "Total Deaths": 420,
        "No. Injured": 1300,
        "No. Affected": 80000,
        "No. Homeless": 12000,
        "Total Affected": 93000,
        "Total Damage ('000 US$)": 900000,
        "log_deaths": np.log1p(420),
        "log_injured": np.log1p(1300),
        "log_affected": np.log1p(93000),
        "log_damage": np.log1p(900000),
        "severity_score": 8.9,
    }

    output = predict_with_explanation(input_features)

    print("\n🧠 ML Decision Support Output:\n")
    for k, v in output.items():
        print(f"{k}: {v}")
