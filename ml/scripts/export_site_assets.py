"""Export der ML-Ergebnisse für die AnvilLabs-Projektseite.

Liest ausschließlich Dateien, die die Notebooks bereits geschrieben haben, und erzeugt
daraus eine einzige JSON-Datei für die Website (Ergebnistabelle, Trainingsverlauf,
Kontext-Ablation und Engine-Explorer). Es wird nichts neu trainiert.

Voraussetzungen (in dieser Reihenfolge ausgeführt):
    03_baseline_rf, 04_xgboost, 05_svr  → model_results.json, predictions_test.csv
    08_lstm_bilstm_attention            → + BiLSTM, lstm_attention_test.csv
    09_context_ablation                 → context_ablation.csv

Aufruf (aus ml/):
    uv run python scripts/export_site_assets.py --site ../../anvil-labs
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

from src.config import FINAL_SENSORS, LSTM_PARAMS, RAW_DIR, RUL_CAP  # noqa: E402

REPORTS = ML_DIR / "reports"

MODELS = {  # Schlüssel in der JSON-Datei → Spaltenname in den Notebook-Ergebnissen
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "svr": "SVR (RBF)",
    "bilstm": "BiLSTM + Attention",
}

# Sensoren für den Explorer: Ps30, T30, htBleed und T50 sind in RF und XGBoost die vier
# wichtigsten Merkmale (Rolling Mean). BPR ist der einzige Sensor, der bei einem Teil der
# Engines steigt und bei einem anderen fällt (Spur der zwei Ausfallmodi). T24 ergänzt die
# Temperaturkette Verdichter → Turbine. Alle sechs gehören zu den 12 Modell-Sensoren.
EXPLORER_SENSORS = {
    "sensor_11": "Ps30 · statischer Druck HPC-Austritt",
    "sensor_4": "T50 · Temperatur LPT-Austritt",
    "sensor_3": "T30 · Temperatur HPC-Austritt",
    "sensor_17": "htBleed · Bleed-Enthalpie",
    "sensor_15": "BPR · Nebenstromverhältnis",
    "sensor_2": "T24 · Temperatur LPC-Austritt",
}

# Beschreibung nach Saxena et al. (2008), "Damage Propagation Modeling", Tabelle 2
# (liegt als PDF in data/raw/CMaps/). Einheiten wie im Paper: °R = Rankine, psia = psi absolut.
SENSOR_INFO = {
    "sensor_11": {"symbol": "Ps30", "unit": "psia", "decimals": 2, "station": "Station 30, nach dem Hochdruckverdichter (HPC)",
                  "desc": "Statischer Druck am Austritt des Hochdruckverdichters."},
    "sensor_4": {"symbol": "T50", "unit": "°R", "decimals": 1, "station": "Station 50, nach der Niederdruckturbine (LPT)",
                 "desc": "Totaltemperatur am Austritt der Niederdruckturbine, also die Abgastemperatur."},
    "sensor_3": {"symbol": "T30", "unit": "°R", "decimals": 1, "station": "Station 30, nach dem Hochdruckverdichter (HPC)",
                 "desc": "Totaltemperatur am Austritt des Hochdruckverdichters, vor der Brennkammer."},
    "sensor_17": {"symbol": "htBleed", "unit": "", "decimals": 0, "station": "an der Zapfluft-Entnahme des Verdichters",
                  "desc": "Bleed-Enthalpie: Energie der Luft, die für Kühlung und Kabine abgezapft wird. "
                          "Im Datensatz nur ganzzahlig, daher die Treppenstufen."},
    "sensor_15": {"symbol": "BPR", "unit": "", "decimals": 3, "station": "an der Aufteilung hinter dem Fan",
                  "desc": "Nebenstromverhältnis: Luftmasse, die am Kern vorbeiströmt, geteilt durch die Luftmasse durch den Kern."},
    "sensor_2": {"symbol": "T24", "unit": "°R", "decimals": 2, "station": "Station 24, nach dem Niederdruckverdichter (LPC)",
                 "desc": "Totaltemperatur am Austritt des Niederdruckverdichters."},
}


def training_reference(train: pd.DataFrame) -> dict:
    """Wie verhält sich jeder Explorer-Sensor in den Trainingsdaten (alle Engines laufen bis zum Ausfall)?
    Anfang = Mittel der ersten 30 Zyklen je Engine, Ende = Mittel der letzten 10 Zyklen, jeweils Median über Engines."""
    life = train.groupby("unit_number")["time_in_cycles"].max()
    ref = {}
    for s in EXPLORER_SENSORS:
        g = train.groupby("unit_number")[s]
        start, end = g.apply(lambda x: x.iloc[:30].mean()), g.apply(lambda x: x.iloc[-10:].mean())
        noise = g.apply(lambda x: x.iloc[:30].std()).median()
        delta = end - start
        r = {"start": round(float(start.median()), 4), "end": round(float(end.median()), 4),
             "delta": round(float(delta.median()), 4), "snr": round(float(delta.median() / noise), 1),
             "rising": int((delta > 0).sum()), "engines": int(len(delta))}
        if (delta > 0).mean() < 0.9:      # Sensor steigt nicht bei allen Engines → zwei Gruppen ausweisen
            up, down = delta > 0, delta <= 0
            r["groups"] = [
                {"label": "steigend", "n": int(up.sum()), "end": round(float(end[up].median()), 4), "life": int(life[up].median())},
                {"label": "fallend", "n": int(down.sum()), "end": round(float(end[down].median()), 4), "life": int(life[down].median())},
            ]
        ref[s] = r
    return ref

COLS = (["unit_number", "time_in_cycles"] + [f"os_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)])


def _require(path: Path, hint: str) -> Path:
    if not path.exists():
        raise SystemExit(f"Fehlt: {path} – zuerst {hint} ausführen.")
    return path


def _r(x, nd: int):
    """Rundet Skalare und Arrays für eine kompakte JSON-Datei."""
    return np.round(np.asarray(x, dtype=float), nd).tolist()


def load_raw(split: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / f"{split}_FD003.txt", sep=r"\s+", header=None, names=COLS)


def build() -> dict:
    results = json.loads(_require(REPORTS / "model_results.json", "die Notebooks 03–05 und 08").read_text())
    preds = pd.read_csv(_require(REPORTS / "predictions_test.csv", "die Notebooks 03–05 und 08"), index_col="Engine")
    attn = pd.read_csv(_require(REPORTS / "lstm_attention_test.csv", "Notebook 08"), index_col="Engine")
    log = pd.read_csv(_require(REPORTS / "lstm_training_log_original.csv", "Notebook 08"))
    abl = pd.read_csv(_require(REPORTS / "context_ablation.csv", "Notebook 09"))

    if not attn.index.equals(preds.index):
        raise SystemExit("Engine-IDs in lstm_attention_test.csv und predictions_test.csv passen nicht zusammen")

    missing = [name for name in MODELS.values() if name not in results or name not in preds.columns]
    if missing:
        raise SystemExit(f"Ergebnisse fehlen für: {missing}")

    # Konsistenz: Metriken müssen aus genau diesen Vorhersagen stammen
    for name in MODELS.values():
        rmse = float(np.sqrt(np.mean((preds[name] - preds["RUL_true"]) ** 2)))
        if abs(rmse - results[name]["RMSE"]) > 0.006:
            raise SystemExit(f"{name}: RMSE aus predictions_test.csv ({rmse:.3f}) ≠ model_results.json "
                             f"({results[name]['RMSE']})")

    train, test = load_raw("train"), load_raw("test")
    lo, hi = train[FINAL_SENSORS].min(), train[FINAL_SENSORS].max()       # Min-Max wie in Notebook 08
    test_norm = (test[FINAL_SENSORS] - lo) / (hi - lo)
    test_norm.insert(0, "unit_number", test["unit_number"])

    engines = []
    for eid, grp in test_norm.groupby("unit_number"):
        row = preds.loc[eid]
        engines.append({
            "id": int(eid),
            "rul": int(row["RUL_true"]),
            "n": int(len(grp)),
            "pred": {k: round(float(row[v]), 1) for k, v in MODELS.items()},
            "attn": _r(attn.loc[eid].values, 4),
            "s": {s: _r(grp[s].values, 3) for s in EXPLORER_SENSORS},
        })

    window = LSTM_PARAMS["window_size"]
    return {
        "meta": {
            "dataset": "NASA CMAPSS FD003",
            "exported": date.today().isoformat(),
            "rul_cap": RUL_CAP,
            "window": window,
            "test_rul_max": int(preds["RUL_true"].max()),
            "padded_engines": [e["id"] for e in engines if e["n"] < window],
            "models": MODELS,
            "sensors": EXPLORER_SENSORS,
            "sensor_info": {s: SENSOR_INFO[s] | {"lo": float(lo[s]), "hi": float(hi[s])} for s in EXPLORER_SENSORS},
            "training_reference": training_reference(train),
            "n_train_engines": int(train["unit_number"].nunique()),
            "sources": {
                "raw": "NASA CMAPSS FD003: train_FD003.txt, test_FD003.txt, RUL_FD003.txt",
                "predictions": "ml/reports/predictions_test.csv (Notebooks 03, 04, 05, 08)",
                "attention": "ml/reports/lstm_attention_test.csv (Notebook 08)",
                "ablation": "ml/reports/context_ablation.csv (Notebook 09)",
                "training_log": "ml/reports/lstm_training_log_original.csv",
                "script": "ml/scripts/export_site_assets.py",
            },
            "note": "Metriken und Vorhersagen aus dem Neulauf (Oktober 2026, macOS); für BiLSTM und SVR "
                    "identisch mit dem Originallauf (Juni 2026), für Random Forest und XGBoost mit "
                    "kleinen Plattformabweichungen. BiLSTM: Checkpoint lstm_best.keras, bester von "
                    "mehreren Trainingsläufen.",
        },
        "metrics": {
            k: {"rmse": results[v]["RMSE"], "mae": results[v]["MAE"],
                "r2": results[v]["R²"], "nasa": results[v]["NASA Score"]}
            for k, v in MODELS.items()
        },
        "training": {
            "epoch": log["epoch"].astype(int).tolist(),
            "loss": _r(log["loss"], 2), "val_loss": _r(log["val_loss"], 2),
            "mae": _r(log["mae"], 3), "val_mae": _r(log["val_mae"], 3),
            "lr": log["learning_rate"].tolist(),
            "best_epoch": int(log.loc[log["val_loss"].idxmin(), "epoch"]),
        },
        "ablation": [
            {"variant": r["Variante"], "window": int(r["Fenster"]), "n_features": int(r["n Features"]),
             "cv_rmse": round(float(r["CV-RMSE"]), 2), "cv_sd": round(float(r["CV-RMSE SD"]), 2),
             "test_rmse": round(float(r["Test-RMSE"]), 2), "test_sd": round(float(r["Test-RMSE SD"]), 2),
             "nasa": round(float(r["NASA Score"]), 1)}
            for _, r in abl.iterrows()
        ],
        "attention_mean": _r(attn.values.mean(axis=0), 5),
        "attention_groups": [
            {"label": label, "n": int(mask.sum()), "mean": _r(attn.values[mask.values].mean(axis=0), 5)}
            for label, mask in [("RUL ≤ 50 · nahe am Ausfall", preds["RUL_true"] <= 50),
                                ("RUL > 100 · gesund", preds["RUL_true"] > 100)]
        ],
        "engines": engines,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", type=Path, required=True, help="Wurzelordner der AnvilLabs-Website")
    args = ap.parse_args()

    data = build()
    out = args.site / "assets" / "data" / "predictive-maintenance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    print(f"Geschrieben: {out} ({out.stat().st_size / 1024:.0f} KB, {len(data['engines'])} Engines)")
    for k, m in data["metrics"].items():
        print(f"  {MODELS[k]:20s} RMSE {m['rmse']:6.2f}  MAE {m['mae']:6.2f}  R² {m['r2']:.3f}  NASA {m['nasa']:7.1f}")


if __name__ == "__main__":
    main()
