from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

from src.decision_engine import predict_with_explanation
from src.llm_engine import get_llm_explanation

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
    damage_usd: float


@app.post("/predict")
def predict(data: DisasterInput):

    feature_dict = {
        "log_deaths":   np.log1p(data.deaths),
        "log_injured":  np.log1p(data.injured),
        "log_affected": np.log1p(data.affected),
        "log_damage":   np.log1p(data.damage_usd),
    }

    # ML prediction
    ml_result = predict_with_explanation(feature_dict)

    # LLM explanation (Gemini → Groq → static fallback)
    llm_text = get_llm_explanation(ml_result["priority"], feature_dict)

    return {
        **ml_result,
        "llm_explanation": llm_text,
    }