import joblib
from pathlib import Path

def load_model(model_path: str):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    data = joblib.load(model_path)
    # Support both old format (raw model) and new format (dict with model + feature_names)
    if isinstance(data, dict) and "model" in data:
        return data["model"], data.get("feature_names", None)
    return data, None