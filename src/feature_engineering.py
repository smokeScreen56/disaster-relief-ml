import pandas as pd
import numpy as np


def add_severity_score(df: pd.DataFrame) -> pd.DataFrame:

    cols = [
    "Total Deaths",
    "No. Injured",
    "No. Affected",
    "No. Homeless",
    "Total Affected",
    "Total Damage ('000 US$)"
    ]

    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0

    df["log_deaths"] = np.log1p(df["Total Deaths"])
    df["log_injured"] = np.log1p(df["No. Injured"])
    df["log_affected"] = np.log1p(df["Total Affected"])
    df["log_damage"] = np.log1p(df["Total Damage ('000 US$)"])

    df["severity_score"] = (
        0.4 * df["log_deaths"]
        + 0.25 * df["log_affected"]
        + 0.2 * df["log_injured"]
        + 0.15 * df["log_damage"]
    )

    return df


def add_severity_level(df: pd.DataFrame) -> pd.DataFrame:

    low_threshold = df["severity_score"].quantile(0.33)
    high_threshold = df["severity_score"].quantile(0.66)

    def classify(score):
        if score <= low_threshold:
            return 0  # Low
        elif score <= high_threshold:
            return 1  # Medium
        else:
            return 2  # High

    df["severity_level"] = df["severity_score"].apply(classify)

    return df
