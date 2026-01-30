import pandas as pd

def load_and_clean_data(path):
    df = pd.read_excel(path)

    cols = [
        "Disaster Type",
        "Country",
        "Start Year",
        "Total Deaths",
        "No. Injured",
        "No. Affected",
        "No. Homeless",
        "Total Affected",
        "Total Damage ('000 US$)"
    ]

    df = df[cols]

    numeric_cols = [
        "Total Deaths",
        "No. Injured",
        "No. Affected",
        "No. Homeless",
        "Total Affected",
        "Total Damage ('000 US$)"
    ]

    # Fill missing numeric values with 0 (industry-standard for disaster data)
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Drop rows missing essential categorical info
    df = df.dropna(subset=["Disaster Type", "Country", "Start Year"])

    return df
