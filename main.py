import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent
STAT_COLS = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]

app = FastAPI(title="Pokémon Battle Predictor")

# Load everything ONCE, when the server starts
model = joblib.load(BASE_DIR / "model.joblib")
with open(BASE_DIR / "feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)

pokemon = pd.read_csv(BASE_DIR / "pokemon.csv").dropna(subset=["Name"])
pokemon["key"] = pokemon["Name"].str.lower()
pokemon = pokemon.drop_duplicates(subset="key").set_index("key")


class BattleRequest(BaseModel):
    first: str
    second: str


def get_stats(name: str) -> dict:
    key = name.strip().lower()
    if key not in pokemon.index:
        raise HTTPException(status_code=404, detail=f"Unknown Pokémon: {name}")
    row = pokemon.loc[key]
    return {col: int(row[col]) for col in STAT_COLS}


def build_features(first: dict, second: dict) -> pd.DataFrame:
    row = {}
    for col in STAT_COLS:
        row["first_" + col] = first[col]
        row["second_" + col] = second[col]
        row["diff_" + col] = first[col] - second[col]
    return pd.DataFrame([row])[FEATURE_COLUMNS]


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/pokemon")
def list_pokemon():
    return sorted(pokemon["Name"].tolist())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(battle: BattleRequest):
    first = get_stats(battle.first)
    second = get_stats(battle.second)
    X = build_features(first, second)

    p_first = float(model.predict_proba(X)[0][1])
    winner = battle.first if p_first >= 0.5 else battle.second

    return {
        "first": battle.first,
        "second": battle.second,
        "winner": winner,
        "probability_first_wins": round(p_first, 3),
    }