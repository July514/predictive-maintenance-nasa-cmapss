# ml/src/evaluation/results.py
# ─────────────────────────────────────────────────────────────────────────────
# Zentrale Ablage der Test-Ergebnisse.
#   model_results.json   – Metriken pro Modell (Quelle für 06, 08 und die Website)
#   predictions_test.csv – Vorhersage pro Test-Engine und Modell (Quelle für 07)
# Jedes Modell-Notebook schreibt hier seine Werte, statt sie in andere
# Notebooks zu kopieren. So gibt es für jede Zahl genau eine Quelle.
# ─────────────────────────────────────────────────────────────────────────────

import json

import numpy as np
import pandas as pd

from src.config import PREDICTIONS_CSV, RESULTS_JSON

MODEL_ORDER = ["Random Forest", "XGBoost", "SVR (RBF)", "BiLSTM + Attention"]


def load_results() -> dict:
    """Alle gespeicherten Metriken laden (leeres Dict, falls noch nichts da ist)."""
    if not RESULTS_JSON.exists():
        return {}
    with open(RESULTS_JSON, encoding="utf-8") as f:
        return json.load(f)


def save_results(results: dict) -> None:
    """Metriken eines Modells (Ausgabe von evaluate_all) eintragen oder ersetzen."""
    all_results = load_results()
    all_results[results["Model"]] = results
    ordered = {m: all_results[m] for m in MODEL_ORDER if m in all_results}
    ordered |= {m: v for m, v in all_results.items() if m not in ordered}
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2)
        f.write("\n")
    print(f"Metriken gespeichert: {RESULTS_JSON.name} → {results}")


def load_predictions() -> pd.DataFrame:
    """Vorhersagen pro Test-Engine laden (Index = Engine, Spalten = RUL_true + Modelle)."""
    if not PREDICTIONS_CSV.exists():
        raise FileNotFoundError(
            f"{PREDICTIONS_CSV} fehlt – zuerst die Notebooks 03, 04, 05 und 08 ausführen."
        )
    return pd.read_csv(PREDICTIONS_CSV, index_col="Engine")


def save_predictions(model_name: str, engine_ids, y_true, y_pred) -> None:
    """Test-Vorhersagen eines Modells als Spalte in predictions_test.csv schreiben."""
    new = pd.DataFrame(
        {"RUL_true": np.asarray(y_true, dtype=float)},
        index=pd.Index(np.asarray(engine_ids, dtype=int), name="Engine"),
    )
    if PREDICTIONS_CSV.exists():
        df = load_predictions()
        if not df.index.equals(new.index) or not np.allclose(df["RUL_true"], new["RUL_true"]):
            raise ValueError("Engine-IDs oder wahre RUL passen nicht zu predictions_test.csv")
    else:
        df = new
    df[model_name] = np.asarray(y_pred, dtype=float)
    cols = ["RUL_true"] + [m for m in MODEL_ORDER if m in df.columns]
    cols += [c for c in df.columns if c not in cols]
    df[cols].to_csv(PREDICTIONS_CSV)
    print(f"Vorhersagen gespeichert: {PREDICTIONS_CSV.name} (Spalte '{model_name}')")
