def create_priority_score(df):
    df["priority_score"] = (
        df["Total Deaths"] * 0.4 +
        df["Total Affected"] * 0.4 +
        df["Total Damage ('000 US$)"] * 0.2
    )
    return df
