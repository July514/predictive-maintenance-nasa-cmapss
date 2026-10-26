# ml/src/config.py
# ─────────────────────────────────────────────────────────────────────────────
# Zentrale Konfiguration: Pfade, Konstanten, Hyperparameter.
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path

# ── Pfade ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]   # predictive-maintenance/
DATA_PROC = ROOT / "data" / "processed"           # predictive-maintenance/data/processed
REPORTS   = ROOT / "ml" / "reports" / "figures"   # predictive-maintenance/ml/reports/figures
RAW_DIR   = ROOT / "data" / "raw" / "CMaps"         # NASA-Rohdaten (für das BiLSTM)

# ── Ergebnis-Ablage (gemeinsame Quelle für Notebooks 06/07 und die Website) ──
RESULTS_JSON    = ROOT / "ml" / "reports" / "model_results.json"    # Metriken pro Modell
PREDICTIONS_CSV = ROOT / "ml" / "reports" / "predictions_test.csv"  # Vorhersage pro Test-Engine
LSTM_MODEL_PATH = ROOT / "ml" / "reports" / "lstm_best.keras"       # Checkpoint des Originallaufs

SEED = 42

# ── Datensatz ─────────────────────────────────────────────────────────────────
SUBSET  = "FD003"
RUL_CAP = 125    # Piecewise Linear RUL Cap (NASA-Konvention)

# ── Finale Sensoren nach VIF-Check (aus R/EDA Phase 1) ───────────────────────
# Rausgeflogen wegen SD ≈ 0:       sensor_1, 5, 16, 18, 19
# Rausgeflogen wegen VIF > 10:     sensor_7, 9, 12, 13
# Verbleiben 12 finale Sensoren:
FINAL_SENSORS = [
    "sensor_2",  "sensor_3",  "sensor_4",  "sensor_6",
    "sensor_8",  "sensor_10", "sensor_11", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

# ── Feature-Spalten in train.csv / test.csv (aus R exportiert) ───────────────
# Rohe Sensoren + Rolling Mean/Std (window=5) + normalisierte Versionen
ROLLING_COLS      = [f"{s}_roll_mean" for s in FINAL_SENSORS] + \
                    [f"{s}_roll_sd"   for s in FINAL_SENSORS]
NORM_SENSOR_COLS  = [f"{s}_norm"          for s in FINAL_SENSORS]
NORM_ROLLING_COLS = [f"{s}_roll_mean_norm" for s in FINAL_SENSORS] + \
                    [f"{s}_roll_sd_norm"   for s in FINAL_SENSORS]

# Feature-Sets die im Notebook zur Auswahl stehen
FEATURE_SETS = {
    # Nur rohe Sensoren (Baseline ohne Rolling)
    "raw":          FINAL_SENSORS,
    # Rohe + Rolling (nicht normalisiert) – das ist was wir nutzen
    "raw+rolling":  FINAL_SENSORS + ROLLING_COLS,
    # Normalisierte Version (für SVR wichtig)
    "norm+rolling": NORM_SENSOR_COLS + NORM_ROLLING_COLS,
}

# ── Modell-Hyperparameter (Defaults) ─────────────────────────────────────────
RF_PARAMS = {
    "n_estimators": 250,
    "max_depth":    None,
    "random_state": 42,
    "n_jobs":       -1,
}

XGB_PARAMS = {
    "n_estimators":          1000,
    "learning_rate":         0.04,
    "max_depth":             6,
    "min_child_weight":      5,
    "subsample":             0.8,
    "colsample_bytree":      0.8,
    "reg_alpha":             0.1,
    "reg_lambda":            1.0,
    "tree_method":           "hist",
    "early_stopping_rounds": 50,
    "random_state":          42,
    "n_jobs":                -1,
}

SVR_PARAMS = {
    "kernel":  "rbf",
    "C":       100,
    "gamma":   "scale",
    "epsilon": 5,
}

# ── BiLSTM + Attention (Originallauf, Juni 2026) ─────────────────────────────
LSTM_PARAMS = {
    "window_size":   50,     # Zyklen pro Input-Sequenz
    "lstm_units_1":  256,    # BiLSTM-Schicht 1 (Einheiten je Richtung)
    "lstm_units_2":  128,    # BiLSTM-Schicht 2
    "dropout":       0.3,
    "dense_units":   32,
    "learning_rate": 1e-3,
    "epochs":        80,
    "batch_size":    64,
}
