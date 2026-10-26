# Shiny Dashboard – NASA CMAPSS FD003

Interaktives Dashboard für die Predictive-Maintenance-Analyse.

## Voraussetzungen

Die folgenden Daten müssen vorhanden sein:
- `predictive-maintenance/data/processed/train.csv`
- `predictive-maintenance/data/processed/test.csv`

Diese werden vom R-Preprocessing erzeugt (`statistik/analysis/02_preprocessing.qmd`).

## Pakete

Falls noch nicht installiert, in der R-Konsole:

```r
install.packages(c(
  "shiny", "bs4Dash", "tidyverse", "plotly", "DT",
  "survival", "survminer", "flexsurv", "broom"
))
```

Bei renv:

```r
renv::install(c("shiny", "bs4Dash", "tidyverse", "plotly", "DT",
                "survival", "survminer", "flexsurv", "broom"))
renv::snapshot()
```

## App starten

Aus dem `statistik/shiny-app/` Ordner:

```r
shiny::runApp()
```

Oder direkt aus RStudio: `app.R` öffnen → "Run App" klicken.

## Aufbau

| Tab | Inhalt |
|---|---|
| **Übersicht** | Datensatz-Steckbrief, Lebensdauer-Histogramm, Datenvorschau |
| **Deskriptiv** | Engine-Verläufe, Engine-Vergleich, Sensor-Statistiken, Korrelationsmatrix |
| **Inferenz** | t-Test/Mann-Whitney (Early vs Late), Log-Rank-Test, Shapiro-Wilk |
| **Survival** | Kaplan-Meier, Weibull AFT, Cox PH (alles interaktiv) |
| **RUL-Vorhersage** | Live-Prognose für Test-Engines, Vergleich mit wahrer RUL |

## Datei-Struktur

```
shiny-app/
├── app.R                          # Hauptdatei
├── R/
│   ├── helpers.R                  # col_N, Farben, Theme
│   ├── data_loader.R              # Lädt processed CSVs
│   └── survival_helpers.R         # KM, Weibull, Cox + Predictions
└── modules/
    ├── mod_overview.R             # Tab 1
    ├── mod_descriptive.R          # Tab 2
    ├── mod_inference.R            # Tab 3
    ├── mod_survival.R             # Tab 4
    └── mod_prediction.R           # Tab 5
```

## Methodische Konsistenz

Die Survival-Modelle nutzen denselben Ansatz wie die qmd-Notebooks:
**Statische Covariaten = Mittelwert der letzten 20 Zyklen vor dem Ausfall.**

Vergleichswerte aus der Cox-Analyse (statistik/analysis/05_survival_cox.qmd):
- Cox PH statisch: C-Index = 0.756
- Cox PH zeitabhängig: C-Index = 0.986 (In-Sample)
