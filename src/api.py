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
    severity_score: float
    total_affected: int
    infrastructure_damage: float
    area_affected: float


@app.post("/predict")
def predict(data: DisasterInput):

    # Map frontend input → ML features
    feature_dict = {
        "Start Year": 2024,
        "Total Deaths": int(data.severity_score * 50),
        "No. Injured": int(data.total_affected * 0.05),
        "No. Affected": data.total_affected,
        "No. Homeless": int(data.total_affected * 0.1),
        "Total Affected": data.total_affected,
        "Total Damage ('000 US$)": int(data.infrastructure_damage * 100000),
        "log_deaths": np.log1p(int(data.severity_score * 50)),
        "log_injured": np.log1p(int(data.total_affected * 0.05)),
        "log_affected": np.log1p(data.total_affected),
        "log_damage": np.log1p(int(data.infrastructure_damage * 100000)),
        "severity_score": data.severity_score,
    }

    return predict_with_explanation(feature_dict)
