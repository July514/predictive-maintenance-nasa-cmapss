# ml/src/evaluation/metrics.py
# ─────────────────────────────────────────────────────────────────────────────
# Evaluierungsmetriken für RUL-Vorhersage.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA asymmetrische Score-Funktion (CMAPSS-Standard).

    d = pred - true
      d < 0: zu pessimistisch (sagt 'bald kaputt', hält noch)  → mild bestraft
      d > 0: zu optimistisch (sagt 'noch OK', fällt früher aus) → hart bestraft

    Niedrigerer Score = besser.
    """
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(scores))


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray,
                 model_name: str = "Model") -> dict:
    """Alle Metriken auf einmal berechnen."""
    return {
        "Model":       model_name,
        "RMSE":        round(rmse(y_true, y_pred), 2),
        "MAE":         round(mae(y_true, y_pred), 2),
        "R²":          round(r2_score(y_true, y_pred), 3),
        "NASA Score":  round(nasa_score(y_true, y_pred), 1),
    }


def compare_models(results: list[dict]) -> pd.DataFrame:
    """Mehrere Modell-Ergebnisse in eine Vergleichstabelle (sortiert nach RMSE)."""
    return pd.DataFrame(results).set_index("Model").sort_values("RMSE")
