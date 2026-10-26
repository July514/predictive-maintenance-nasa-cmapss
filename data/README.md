# data/

## Struktur

```
data/
├── raw/
│   └── CMaps/           ← NASA-CMAPSS-Rohdaten (nicht eingecheckt)
│       ├── train_FD003.txt
│       ├── test_FD003.txt
│       └── RUL_FD003.txt
│
└── processed/           ← von R erzeugt, in Python und Shiny wiederverwendet (eingecheckt)
    ├── train.csv
    ├── test.csv
    └── feature_metadata.json
```

## Rohdaten herunterladen

**Option A – NASA (empfohlen):**
1. [NASA PCoE Data Set Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) → „Turbofan Engine Degradation Simulation“ oder direkt das [ZIP-Archiv](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip) laden (Mirror: [PHM Society](https://data.phmsociety.org/nasa/)).
2. `train_FD003.txt`, `test_FD003.txt` und `RUL_FD003.txt` nach `data/raw/CMaps/` entpacken.

**Option B – Kaggle:**
```bash
kaggle datasets download -d behrad3d/nasa-cmaps -p data/raw/ --unzip
mv data/raw/CMAPSSData/* data/raw/CMaps/
```

## Processed-Daten erzeugen

`data/processed/` entsteht in `statistik/analysis/02_preprocessing.qmd`: Rohdaten laden, RUL bei 125 kappen, konstante Sensoren entfernen, die 12 Sensoren aus dem VIF-Check auswählen, Rolling Mean/SD (Fenster 5) berechnen und normalisieren. Die Roh-Loader dafür stehen in `statistik/shiny-app/R/data_loader.R`.

```bash
cd statistik
quarto render analysis/02_preprocessing.qmd
```

Die Dateien sind bewusst eingecheckt: Die ML-Notebooks 03–07 und die Shiny-App lesen sie direkt, und alle ML-Ergebnisse beziehen sich exakt auf diesen Stand.

**Reproduzierbarkeit (geprüft am 25.10.2026, R 4.5.3, macOS/Apple Silicon):** Ein Neulauf ergibt dieselben Zeilen, Spalten und fehlenden Werte (train 24.720 × 78, test 100 × 78). Die Zahlen weichen in den Rolling-Mean-Spalten und den daraus normalisierten Spalten um höchstens 2.4·10⁻¹¹ ab. Ursache ist die Summation in `zoo::rollmean`: R summiert `cumsum` in `long double`, das unter Windows/x86-64 80 Bit hat und auf Apple Silicon 64 Bit. Die eingecheckten Werte liegen entsprechend näher am exakten Fenstermittel. `feature_metadata.json` unterscheidet sich nur in `created` und `r_version`.

## In Python laden

Aus `ml/` heraus (die Notebooks hängen `ml/` an `sys.path` an):

```python
from src.data.loader import load_train, load_test
train = load_train()
test = load_test()
```
