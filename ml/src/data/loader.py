# ml/src/data/loader.py
# ─────────────────────────────────────────────────────────────────────────────
# Lädt die in R aufbereiteten CSVs aus data/processed/.
# Datenaufbereitung (RUL-Berechnung, Rolling Stats, Normalisierung) wurde
# vollständig in R durchgeführt (statistik/analysis/02_preprocessing.qmd).
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from src.config import DATA_PROC, FINAL_SENSORS, ROLLING_COLS, NORM_SENSOR_COLS, NORM_ROLLING_COLS


def load_train() -> pd.DataFrame:
    """
    Aufbereitete Trainingsdaten laden (aus R exportiert).

    Enthält: 24.720 Zeilen, 78 Spalten
      - 12 rohe Sensoren
      - 12 Rolling Mean (window=5) + 12 Rolling Std (window=5)
      - 36 normalisierte Versionen
      - RUL (gecappt bei 125)
      - NAs in ersten 4 Zyklen jeder Engine (Rolling-Fenster) → werden gedroppt
    """
    df = pd.read_csv(DATA_PROC / "train.csv")

    # Rolling-NAs entfernen (erste 4 Zeilen pro Engine, window=5)
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"Train geladen: {after} Zeilen ({before - after} Rolling-NAs entfernt)")
    return df


def load_test() -> pd.DataFrame:
    """
    Aufbereitete Testdaten laden (letzte Beobachtung pro Engine + wahre RUL).

    Enthält: 100 Zeilen (eine pro Engine), keine NAs.
    RUL_true kommt aus RUL_FD003.txt (via R eingelesen).
    """
    df = pd.read_csv(DATA_PROC / "test.csv")
    print(f"Test geladen: {len(df)} Zeilen (eine pro Engine)")
    return df


def get_feature_cols(feature_set: str = "raw+rolling") -> list[str]:
    """
    Feature-Spalten für ein gegebenes Feature-Set zurückgeben.

    Parameters
    ----------
    feature_set : "raw" | "raw+rolling" | "norm+rolling"
        raw         – nur rohe Sensoren (12 Features)
        raw+rolling – rohe + Rolling Mean/Std, nicht normalisiert (36 Features)
        norm+rolling – normalisierte Sensoren + normalisierte Rolling (36 Features)

    Returns
    -------
    Liste von Spaltennamen
    """
    sets = {
        "raw":          FINAL_SENSORS,
        "raw+rolling":  FINAL_SENSORS + ROLLING_COLS,
        "norm+rolling": NORM_SENSOR_COLS + NORM_ROLLING_COLS,
    }
    if feature_set not in sets:
        raise ValueError(f"feature_set muss eines von {list(sets.keys())} sein")
    return sets[feature_set]
