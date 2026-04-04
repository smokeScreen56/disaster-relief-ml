# Disaster Relief Decision System

ML + LLM pipeline that classifies disaster severity (Low / Medium / High) and generates actionable response recommendations.

## Stack

- **Data** — EM-DAT historical records + CTGAN synthetic augmentation + SMOTE balancing
- **Models** — RandomForest, XGBoost, LightGBM (auto-selects best by F1-macro)
- **LLM** — Gemini 1.5 Flash → Groq/Llama 3 → static fallback
- **API** — FastAPI `/predict` endpoint
- **Frontend** — Single-file terminal-style UI (`index.html`)

## Setup

```bash
git clone https://github.com/smokeScreen56/disaster-relief-ml.git
cd disaster-relief-ml
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` in the project root:
```
GEMINI_API_KEY=your-key   # https://aistudio.google.com/app/apikey
GROQ_API_KEY=your-key     # https://console.groq.com/keys
```

## Usage

```bash
python src/prepare_data.py       # 1. prepare historical data
python src/synthetic_data.py     # 2. generate hybrid dataset
python src/model_comparison.py   # 3. train & compare models
uvicorn src.api:app --reload     # 4. start API
# open index.html in browser     # 5. launch UI
```

## API

`POST /predict`

```json
{ "deaths": 420, "injured": 1300, "affected": 80000,
  "homeless": 12000, "damage_usd": 900000, "area_affected": 2400 }
```

Returns `priority`, `confidence`, `probabilities`, `reasons`, and `llm_explanation`.

## Data Source

EM-DAT International Disaster Database — [emdat.be](https://www.emdat.be)

## License

MIT
