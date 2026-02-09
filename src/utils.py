import joblib
from pathlib import Path

def load_model(model_path: str):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)
