# Duelcast: Pokémon Battle Predictor

A machine learning project that predicts which of two Pokémon wins a battle. A random forest trained on 50,000 battles is served through a FastAPI backend, packaged with Docker, and used by a small web frontend where you pick two Pokémon and see the predicted winner with a probability.


## Results

All scores are measured on 10,000 battles the model never saw during training.

| Model | Test accuracy |
|---|---|
| Always guess the likelier side (baseline) | 52.5% |
| Decision tree, depth 3 | 86.3% |
| Random forest (100 trees) with stat-difference features | 95.6% |

Confusion matrix for the final model:

| | Predicted: second wins | Predicted: first wins |
|---|---|---|
| **Second actually won** | 4,993 | 254 |
| **First actually won** | 188 | 4,565 |

That is 442 mistakes in 10,000 battles. Precision and recall are about 0.95 to 0.96 for both outcomes, so the model does not favor one side.

## How it works

1. **Data.** Two tables from Kaggle: `pokemon.csv` (stats per Pokémon) and `combats.csv` (which two Pokémon fought and who won, by ID).
2. **Training table.** The two tables are merged so each battle row holds both fighters' HP, Attack, Defense, Sp. Atk, Sp. Def and Speed. The label is `first_wins` (1 or 0). ID columns and the winner column are removed from the inputs to avoid data leakage.
3. **Features.** Six difference columns (for example `diff_Speed = first_Speed - second_Speed`) are added, taking the inputs from 12 to 18 features. A tree can only ask about one column at a time, so a difference lets it ask "who is faster?" in a single split.
4. **Model.** A random forest with 100 trees. Settings were compared with 5-fold cross-validation on the training data, and the test set was used once, at the end.
5. **Serving.** The trained model is saved with joblib. The API loads it once at startup, looks up both Pokémon by name, builds the same 18 features in the same column order as training, and returns the prediction.

## API

| Method | Path | What it does |
|---|---|---|
| GET | `/` | Serves the web frontend |
| GET | `/pokemon` | Returns the sorted list of Pokémon names |
| GET | `/health` | Returns `{"status": "ok"}` |
| POST | `/predict` | Takes two names, returns the predicted winner |

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"first": "Charizard", "second": "Blastoise"}'
```

The response has this shape (values vary by matchup):

```json
{
  "first": "Charizard",
  "second": "Blastoise",
  "winner": "...",
  "probability_first_wins": 0.0
}
```

Names are not case sensitive. An unknown name returns a 404 with a clear message, and a malformed request returns a 422. FastAPI's interactive docs are available at `/docs`.

## Run it locally

You need Python 3.11.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Then open <http://127.0.0.1:8000/> for the frontend, or <http://127.0.0.1:8000/docs> for the API docs.

## Run it with Docker

```bash
docker build -t pokemon-predictor .
docker run -p 8000:8000 pokemon-predictor
```

Then open <http://127.0.0.1:8000/>.

## Tests

```bash
pip install pytest httpx
python -m pytest
```

The tests cover the health check, a valid prediction, case-insensitive names, unknown names (404), missing fields (422), consistency when the fighters are swapped, and the frontend and name-list routes.

## Project structure

```
.
├── main.py                  FastAPI app
├── model.joblib             Trained random forest
├── feature_columns.json     Feature names in training order
├── pokemon.csv              Pokémon stats (used by the API for lookups)
├── static/index.html        Web frontend
├── explore.ipynb            Data exploration and model training
├── test_api.py              API tests
├── Dockerfile
├── requirements.txt         Pinned runtime dependencies
└── requirements-dev.txt     Full development environment
```

## Limitations

- The model only sees six battle stats per Pokémon. Types, abilities and moves are not used, so matchups like water against fire are invisible to it.
- The probability reflects how strongly the forest's trees agree. It is not a guarantee of the outcome.
- Results describe this dataset, not real competitive play.
- The model is not perfectly symmetric: swapping the first and second fighter can shift the probability slightly.

## What I learned

- Splitting data into training and test sets, and why the test set must stay untouched until the end.
- Spotting overfitting by comparing training and test accuracy at different tree depths.
- Choosing settings with cross-validation instead of peeking at the test set.
- Feature engineering: adding stat differences and switching from one tree to a forest took test accuracy from 86.3% to 95.6%.
- Avoiding data leakage by removing the winner column from the inputs.
- Building the inputs at prediction time exactly as they were built at training time, including column order.

## Next steps

- Add Pokémon types as features and measure whether accuracy improves.
- Look at the model's mistakes in more detail, since most seem to occur when the two Pokémon are close in speed.

## Credits

- Data: the "Pokemon- Weedle's Cave" dataset by T7 - Hackathon Challenge on Kaggle. Check the dataset page for its license terms before reusing the data.
- Pokémon is a trademark of its respective owners. The creatures and sprites on the web page are original pixel art.