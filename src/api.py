from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from src.decision_engine import predict_with_explanation

app = FastAPI(title="Disaster Relief Decision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DisasterInput(BaseModel):
    deaths: int
    injured: int
    affected: int
    homeless: int
    damage_usd: float
    area_affected: float

@app.post("/predict")
def predict(data: DisasterInput):

    # Map frontend input → ML features
    feature_dict = {
    "Start Year": 2024,

    "Total Deaths": data.deaths,
    "No. Injured": data.injured,
    "No. Affected": data.affected,
    "No. Homeless": data.homeless,
    "Total Affected": data.affected,

    "Total Damage ('000 US$)": data.damage_usd,

    "log_deaths": np.log1p(data.deaths),
    "log_injured": np.log1p(data.injured),
    "log_affected": np.log1p(data.affected),
    "log_damage": np.log1p(data.damage_usd),
    }

    return predict_with_explanation(feature_dict)
