import pandas as pd
from preprocess import load_and_clean_data
from feature_engineering import add_severity_score, add_severity_level

RAW_DATA_PATH = "data/raw/emdat.xlsx"
PROCESSED_DATA_PATH = "data/processed/disaster_features.csv"

def main():
    df = load_and_clean_data(RAW_DATA_PATH)

    df = add_severity_score(df)
    df = add_severity_level(df)

    # Drop non-ML categorical columns
    df_model = df.drop(columns=["Disaster Type", "Country"])

    df_model.to_csv(PROCESSED_DATA_PATH, index=False)
    print("Processed data created successfully.")

if __name__ == "__main__":
    main()
