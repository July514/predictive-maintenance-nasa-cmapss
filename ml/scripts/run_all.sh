#!/usr/bin/env bash
# Führt alle ML-Notebooks headless in der richtigen Reihenfolge aus und exportiert
# danach die Daten für die AnvilLabs-Projektseite.
#
#   cd ml && bash scripts/run_all.sh              (Website-Pfad: SITE=... bash scripts/run_all.sh)
#
# Reihenfolge: Die Modell-Notebooks (03, 04, 05, 08) schreiben model_results.json und
# predictions_test.csv; 06, 07 und 09 lesen diese Dateien. 08 lädt standardmäßig den
# Checkpoint des Originallaufs (RETRAIN = False) und trainiert nicht neu.
set -euo pipefail
cd "$(dirname "$0")/.."            # → ml/

SITE="${SITE:-../../anvil-labs}"

# Eigene Umgebung für macOS/Linux: Die vorhandene .venv stammt von Windows und würde
# von uv sonst als inkompatibel gelöscht. Python 3.13 wie beim Originallauf.
export UV_PROJECT_ENVIRONMENT=".venv-mac"
uv sync --python 3.13

# Fortschritt pro Zelle; dauert insgesamt ca. 30–60 min (Grid Searches in 03, 04, 05)
uv run python scripts/run_notebooks.py "$@" \
    notebooks/03_baseline_rf.ipynb notebooks/04_xgboost.ipynb notebooks/05_svr.ipynb \
    notebooks/08_lstm_bilstm_attention.ipynb notebooks/06_comparison.ipynb \
    notebooks/07_example_prediction.ipynb notebooks/09_context_ablation.ipynb

if [ -d "${SITE}" ]; then
  uv run python scripts/export_site_assets.py --site "${SITE}"
else
  echo "Website-Ordner ${SITE} nicht gefunden – Export übersprungen."
fi
