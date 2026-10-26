# Statistik-Teil: Vollständige Erklärung
## Predictive Maintenance – NASA CMAPSS FD003
> Konzeptuelle Erklärung des Statistik-Teils: Datensatz, EDA, Preprocessing, Kaplan-Meier, Weibull AFT, Cox PH und Brücke zum ML-Teil.  
> Analysen: `analysis/01`–`05` (Quarto, gerendert als HTML), Dashboard: `shiny-app/`, Setup: Abschnitt 12.

---

## Inhaltsverzeichnis

1. [Die zentrale Story](#1-die-zentrale-story)
2. [Der Datensatz](#2-der-datensatz)
3. [Was ist Survival Analysis und warum hier?](#3-was-ist-survival-analysis-und-warum-hier)
4. [EDA: Was die Daten erzählen](#4-eda-was-die-daten-erzählen)
5. [Kaplan-Meier: Die zwei Fault Modes sichtbar machen](#5-kaplan-meier-die-zwei-fault-modes-sichtbar-machen)
6. [Weibull AFT: Sensor-Effekte quantifizieren](#6-weibull-aft-sensor-effekte-quantifizieren)
7. [Cox PH: Zeitabhängige Risikomodellierung](#7-cox-ph-zeitabhängige-risikomodellierung)
8. [Die drei Methoden im Vergleich](#8-die-drei-methoden-im-vergleich)
9. [Brücke zum ML-Teil](#9-brücke-zum-ml-teil)
10. [Methodische Einschränkungen (Ehrlichkeit)](#10-methodische-einschränkungen-ehrlichkeit)
11. [Kernergebnisse auf einen Blick](#11-kernergebnisse-auf-einen-blick)
12. [Setup und Stolpersteine](#12-setup-und-stolpersteine)

---

## 1. Die zentrale Story

Das gesamte Statistik-Projekt folgt einer einzigen durchgehenden Geschichte:

**Ausgangsfrage:** Wann fällt ein Turbofan-Triebwerk aus — und warum?

**Antwort in drei Schritten:**

1. **KM:** Es gibt zwei klar verschiedene Populationen (Fault Modes). Die Lebensdauern sind bimodal, und eine Gruppierung nach dem Verlauf des Bypass-Verhältnisses trennt die Überlebenskurven deutlich (Log-Rank p = 3.4·10⁻¹¹, Abschnitt 5).
2. **Weibull AFT:** Sensorwerte beeinflussen die Überlebenszeit messbar. sensor_11 allein verlängert die erwartete Lebensdauer um Faktor 6.9.
3. **Cox PH:** Der aktuelle Sensorwert zum jeweiligen Zeitpunkt ist fast perfekt prädiktiv (C = 0.986). Aber das Modell hat Grenzen — es setzt lineare, zeitkonstante Effekte voraus.

Diese Grenzen motivieren den ML-Teil: Random Forest und XGBoost kennen keine solchen Annahmen.

---

## 2. Der Datensatz

**NASA CMAPSS FD003** — Turbofan Engine Degradation Simulation Data Set.

- 100 Trainings-Engines, je mit vollständiger Laufzeit bis zum Ausfall
- 21 Sensoren + 3 Operational Settings pro Zyklus
- Eine einzige Betriebsbedingung (Operating Condition)
- **Zwei Ausfallmodi:** HPC-Degradation (High Pressure Compressor) und Fan-Degradation — in den Rohdaten nicht gelabelt

**Besonderheit des Trainings vs. Test:**

| Datei | Beobachtet bis | Zensiert? |
|---|---|---|
| `train_FD003.txt` | Echter Ausfall | Nein — alle Ausfälle beobachtet |
| `test_FD003.txt` | Irgendwann vorher | Ja — rechts-zensiert |
| `RUL_FD003.txt` | — | Wahre RUL für Evaluation |

Zensierung bedeutet: wir wissen nur, dass die Engine mindestens so lange gehalten hat — nicht wann sie wirklich ausgefallen wäre. Survival Analysis kann mit diesen unvollständigen Informationen arbeiten; klassische Regression nicht.

**RUL-Berechnung:** Remaining Useful Life = max_cycle − current_cycle, gecappt bei 125 (NASA-Konvention). Der Cap verhindert, dass Modelle weit in der Zukunft präzise sein müssen, wo noch keine Degradation sichtbar ist.

---

## 3. Was ist Survival Analysis und warum hier?

### Das Problem mit klassischer Regression

Eine Engine hat einen Ausfall-Zeitpunkt. Wenn man diesen direkt regressieren will, braucht man für alle Einheiten einen Zielwert — aber Test-Engines laufen noch. Ihre RUL ist unbekannt. Klassische Regression würde diese zensierten Einheiten wegwerfen oder falsch behandeln.

### Was Survival Analysis anders macht

Survival Analysis modelliert die **Überlebensfunktion S(t)** — die Wahrscheinlichkeit, einen Zeitpunkt t zu überstehen:

$$S(t) = P(T > t)$$

Der Clou: Zensierte Beobachtungen gehen nicht verloren. Sie tragen die Information "überlebt mindestens bis t" bei, ohne den Ausfall-Zeitpunkt zu kennen. Jede Information wird genutzt.

### Warum passt das zu Turbofan-Engines?

- Die Frage "wann fällt eine Maschine aus?" ist eine klassische Survival-Frage
- Die Lebensdauer ist nicht normalverteilt (bimodal, rechtschief, physikalisch motivierter Verschleißcharakter)
- Sensorverläufe liefern zeitabhängige Covariaten — genau das was Cox PH nutzt
- Der Vergleich zweier Ausfallmechanismen ist ein natürlicher Log-Rank-Test

---

## 4. EDA: Was die Daten erzählen

### Die bimodale Lebensdauerverteilung

Das wichtigste EDA-Ergebnis: Die Lebensdauern sind **bimodal verteilt**. Zwei Cluster:

- Kleine Engines: ~150–295 Zyklen (79 Engines)
- Lange Engines: ~300–525 Zyklen (21 Engines)

Das ist kein Zufall — das sind die zwei Fault Modes. HPC-Degradation führt schneller zum Ausfall als Fan-Degradation. Die Rohdaten enthalten kein Label dafür, aber die Bimodalität macht sie sichtbar.

### Sensor-Degradationsverläufe

Drei charakteristische Muster:
- **Steigende Sensoren (z.B. sensor_11, sensor_2):** Steigen monoton an, konvergieren bei Ausfall auf einen gemeinsamen Threshold — unabhängig von der Lebensdauer der Engine.
- **Fallende Sensoren (z.B. sensor_7):** Fallen monoton, gleiche Konvergenz.
- **Konstante Sensoren (sensor_1, 5, 16, 18, 19):** Keine Varianz, werden gelöscht.

**Degradations-Konvergenz:** Alle Engines enden auf ähnlichen Sensorwerten, unabhängig davon wie lang sie gelebt haben. Das hat methodische Konsequenzen für Weibull AFT (kein sinnvoller Profil-Vergleich möglich).

### Sensor-Auswahl VIF-Check Variance Inflation Factor)

Von 21 Sensoren verbleiben nach VIF-Check 12 finale Sensoren. Iterativer Prozess: immer den Sensor mit dem höchsten VIF entfernen, neu berechnen. Paarweise Korrelation allein reicht nicht — VIF prüft ob ein Sensor durch eine Kombination aller anderen vorhergesagt werden kann.

Wichtigste Ausschlüsse: sensor_12 (VIF 52.7, r = 0.99 mit sensor_7), sensor_9 (VIF 14.9, r = 0.95 mit sensor_14).

### Weibull-Fit der Lebensdauern

Shape β = 2.93 > 1 → **Verschleißausfall** bestätigt. Die Ausfallrate steigt mit dem Alter — physikalisch erwartet für mechanische Systeme. Das rechtfertigt Weibull-basierte Survival-Methoden.

---

## 5. Kaplan-Meier: Die zwei Fault Modes sichtbar machen

### Was KM macht

KM schätzt S(t) non-parametrisch — ohne Verteilungsannahme. Bei jedem beobachteten Ausfall-Zeitpunkt wird die Überlebenswahrscheinlichkeit aktualisiert:

$$S(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)$$

$d_i$ = Ausfälle zum Zeitpunkt $t_i$, $n_i$ = Einheiten noch am Leben.

### Unsere Analyse

**Gesamtpopulation:** Median-Lebensdauer 220.5 Zyklen [KI: 204–234]. Der unregelmäßige Verlauf (kein glattes Abfallen) deutet schon auf Heterogenität hin.

**Subgruppen:** Trennung bei 300 Zyklen (aus EDA-Bimodalität abgeleitet):
- Kurze Gruppe (79 Engines): Median 201 Zyklen [194–220]
- Lange Gruppe (21 Engines): Median 347 Zyklen [324–459]

Die KM-Kurven trennen sich früh und klar. Konfidenzintervalle überlappen nicht.

### Log-Rank-Test

Formaler Nachweis dass die zwei Gruppen statistisch verschieden sind:
- H₀: Beide Gruppen haben dieselbe Überlebensfunktion
- Ergebnis: Chi² = 64.62, p = 8.88e-16 — H₀ hochsignifikant verworfen

Das Verhältnis Beobachtet/Erwartet zeigt das erwartete Muster: kurze Gruppe fällt früher aus als unter H₀ erwartet (Obs/Exp = 1.69), lange Gruppe später (Obs/Exp = 0.39).

**Einschränkung:** Der Schwellwert 300 Zyklen wurde post-hoc aus der EDA abgeleitet und teilt die Engines nach ihrer Lebensdauer selbst; die beiden Gruppen überlappen nicht (152–295 bzw. 301–525 Zyklen). Der Unterschied ist damit garantiert: Der Test bestätigt nur die Trennung und ist kein unabhängiger Nachweis.

**Gruppierung unabhängig von der Lebensdauer:** Das Bypass-Verhältnis (BPR, `sensor_15`; Mittel der ersten 30 gegen die letzten 10 Zyklen) steigt bei 56 Engines und fällt bei 44. Die Median-Lebensdauer liegt bei 193 bzw. 286 Zyklen, Log-Rank χ² = 43.9, p = 3.4·10⁻¹¹ (nachgerechnet im Oktober 2026, nicht Teil der Quarto-Analysen). Die Aussage „zwei Populationen“ hält damit auch ohne zirkuläre Gruppierung.

### Kumulative Hazardfunktion

Beide Gruppen zeigen konvexen H(t)-Anstieg → steigende Ausfallrate mit dem Alter → Verschleißcharakter. Das ist konsistent mit β = 2.93 aus der EDA und motiviert das Weibull-Modell.

---

## 6. Weibull AFT: Sensor-Effekte quantifizieren

### Das Modell

Accelerated Failure Time bedeutet: Covariaten (Sensorwerte) beschleunigen oder verlangsamen die Zeit bis zum Ausfall. Das Modell auf der log-Zeit-Skala:

$$\log(T) = \mu + \gamma_1 x_1 + \gamma_2 x_2 + \cdots + \sigma \varepsilon$$

Das ist eine lineare Regression auf log(Lebensdauer). Der Fehlerterm folgt einer Extremwertverteilung — was auf der Zeitskala Weibull ergibt.

### Warum AFT statt nur Weibull-Fit?

Der Baseline-Fit (ohne Covariaten) beschreibt die Lebensdauerverteilung. Das AFT-Modell mit Covariaten beantwortet: **wie verändern Sensorwerte die Überlebenszeit?**

Die Antwort kommt als **Time Ratio (TR = e^γ):**
- TR = 0.8: 20% kürzere erwartete Lebensdauer pro Einheit Sensoranstieg
- TR = 1.5: 50% längere erwartete Lebensdauer

### Unsere Ergebnisse

**Sensor-Auswahl:** sensor_11, sensor_17, sensor_3, sensor_2 — Top-4 nach Korrelation mit Lifetime. sensor_4 ausgeschlossen trotz hoher Korrelation wegen r = 0.85 mit sensor_11.

**Features:** Mittelwert der letzten 20 Zyklen pro Sensor — AFT braucht einen Wert pro Engine, keine Zeitreihe. Letzte 20 Zyklen sind ein Kompromiss zwischen Aktualität und Rauschunterdrückung.

**Modellvergleich:**
- Baseline AIC = 1177.08 → Covariaten AIC = 1104.84
- ΔAIC = 72.24 → Covariaten-Modell klar besser

**Time Ratios:**

| Sensor | TR | Bedeutung |
|---|---|---|
| sensor_11 | 6.91 | Höherer Wert → Engine lebt länger (sensor_11 steigt bei Degradation, hoher Startwert = weniger Degradation) |
| sensor_17 | 1.41 | Moderater positiver Effekt |
| sensor_3 | 1.07 | Schwacher positiver Effekt, aber signifikant |
| sensor_2 | 0.36 | Höherer Wert → Engine lebt kürzer (sensor_2 steigt bei Degradation) |

### Warum kein Profil-Vergleich?

Alle Engines konvergieren am Ende auf ähnliche Sensorwerte (Degradations-Konvergenz). Der IQR der Sensorfeatures in den letzten 20 Zyklen ist so klein (sd_last sensor_11 = 0.059 vs. sd_first = 0.154), dass ein Profil-Vergleich Extrapolation außerhalb des Trainingsbereichs erfordern würde. Das Modell wäre dann numerisch instabil. Bewusst weggelassen — die Time Ratios sind aussagekräftiger.

---

## 7. Cox PH: Zeitabhängige Risikomodellierung

### Das Modell

Cox modelliert den **Hazard** (momentanes Ausfallrisiko), nicht die Zeit:

$$h(t \mid x) = h_0(t) \cdot \exp(\beta_1 x_1 + \beta_2 x_2 + \cdots)$$

- $h_0(t)$: Baseline-Hazard — wie sieht das Risiko ohne Covariaten aus. Cox schätzt das nicht-parametrisch.
- $\exp(\beta_i x_i)$: Covariate-Effekt — multiplikative Risikomodifikation.

Der Output ist die **Hazard Ratio (HR = e^β):**
- HR > 1: höheres Ausfallrisiko
- HR < 1: niedrigeres Ausfallrisiko

### Drei Modell-Varianten

**Modell A (statisch):** Mittelwert letzte 20 Zyklen als Feature — direkter Vergleich zu Weibull. Concordance C = 0.756.

**Modell B (zeitabhängig):** Sensorwert zum jeweiligen Zeitpunkt, Counting-Process-Format. Jede Engine hat ~247 Zeilen statt einer. Concordance C = 0.986.

**Modell C (stratifiziert):** Modell B + separate Baseline-Hazard für jede Fault-Gruppe. Adressiert potenzielle PH-Verletzungen.

### Der Concordance-Sprung: 0.756 → 0.986

Das ist das wichtigste Einzelergebnis des gesamten Statistik-Teils. Die vollständige Zeitreihe der Sensoren enthält fast die gesamte Prognose-Information. Statische End-of-Life-Features verpassen den Degradationsverlauf.

Was C = 0.986 bedeutet: In fast allen zufälligen Paaren von Engines sagt das Modell korrekt vorher, welche zuerst ausfällt.

### PH-Annahme (Proportional Hazards)

Die PH-Annahme verlangt: Die Hazard Ratio zwischen zwei Einheiten ist zu jedem Zeitpunkt dieselbe. Die Kurven dürfen sich nicht kreuzen.

Test via Schoenfeld-Residuen (`cox.zph()`):
- Statisches Modell: sensor_2 knapp verletzt (p = 0.012), global aber okay (p = 0.128)
- Zeitabhängiges Modell: sensor_11 leicht verletzt (p = 0.037), global okay (p = 0.197)
- Nach Stratifizierung: global p = 0.165

Gesamtbefund: **PH hält global** für alle drei Modelle. Die leichten Einzelverletzungen sind physikalisch plausibel — sensor_11 steigt erst spät in der Degradation stark an, sein Effekt ist also nicht wirklich zeitkonstant.

---

## 8. Die drei Methoden im Vergleich

| Aspekt | Kaplan-Meier | Weibull AFT | Cox PH |
|---|---|---|---|
| **Typ** | Non-parametrisch | Parametrisch | Semi-parametrisch |
| **Fragestellung** | Wie ist Überlebenszeit verteilt? | Wie verändern Covariaten die Zeit? | Wie verändern Covariaten das Risiko? |
| **Output** | S(t)-Kurve | Time Ratios | Hazard Ratios + C-Index |
| **Covariaten** | Nur Gruppen | Statische Features | Statisch oder zeitabhängig |
| **Verteilungsannahme** | Keine | Weibull | Keine für Baseline |
| **Unser C-Index** | — | — | 0.756 (statisch) / 0.986 (zeitabhängig) |
| **Stärke** | Intuitiv, robust | Interpretierbare Effektgröße | Flexible Zeitreihen-Einbindung |
| **Schwäche** | Keine Covariaten-Kontrolle | Verteilungsannahme, statische Features | PH-Annahme, lineare Effekte |

Die drei Methoden sind keine Konkurrenten — sie beantworten verschiedene Fragen und bauen aufeinander auf:

KM → zeigt Heterogenität → motiviert Gruppen-Strategie in Cox  
Weibull → quantifiziert Sensor-Effekte → zeigt welche Sensoren relevant sind  
Cox → nutzt vollständige Zeitreihe → liefert Baseline für ML-Vergleich

---

## 9. Brücke zum ML-Teil

### Was Statistik leistet

- Identifiziert relevante Sensoren mit interpretierbaren Effektgrößen (Time Ratios, Hazard Ratios)
- Weist zwei Subpopulationen statistisch nach (Log-Rank-Test)
- Liefert C-Index-Baseline (0.756) für fairen ML-Vergleich
- Erklärt physikalisch warum Engines ausfallen

### Was Statistik nicht kann

- **Nicht-lineare Effekte:** Cox modelliert exp(β·x). sensor_7 hat einen klaren nicht-linearen Verlauf (lange flach, dann schnell), der im Pearson-Korrelationskoeffizient (r = -0.36) und in linearen Modellen untergeht.
- **Sensor-Interaktionen:** Wechselwirkungen zwischen Sensoren werden ignoriert. In der Realität hängen Degradationsprozesse von Kombinations-Effekten ab.
- **PH-Annahme:** Cox setzt zeitkonstante Effekte voraus. sensor_11 verletzt das leicht — ein Zeichen dass sich der Einfluss über den Lebenszyklus verändert.

### Die Fragen sind verschieden — nicht besser/schlechter

| Statistik fragt | ML fragt |
|---|---|
| Warum fallen Engines aus? | Wann genau fällt diese Engine aus? |
| Welche Sensoren sind signifikant? | Welche Features sind am wichtigsten? |
| Wie ist die Überlebenszeit verteilt? | Was ist die beste RUL-Punktprognose? |
| Gilt die PH-Annahme? | Wie minimiere ich RMSE / NASA Score? |

Beide Antworten haben ihren Platz. Im regulatorischen Kontext (Pharma, Luftfahrt) sind Statistik-Modelle mit Konfidenzintervallen und Annahmen-Tests oft Pflicht. Im Betrieb zählt oft die Punktprognose.

---

## 10. Methodische Einschränkungen (Ehrlichkeit)

Diese Punkte gehören zu einer ehrlichen Einordnung der Ergebnisse:

**Kein echter Zensierungsvorteil im Training:** Da alle 100 Training-Engines bis zum Ausfall laufen, ist KM im Training nicht methodisch notwendig. Der Mehrwert liegt im formalen Rahmen und in der Vorbereitung für Cox PH.

**Explorativer Schwellwert:** Die 300-Zyklen-Grenze wurde post-hoc aus der Bimodalität abgeleitet. Ohne externe Fault-Mode-Labels ist das statistisch eine zirkuläre Begründung. Eine nicht-zirkuläre Alternative ist die Gruppierung nach dem BPR-Verlauf (Abschnitt 5).

**Bimodalität und unimodale Modelle:** Weibull ist unimodal — die zwei Fault Modes werden nur näherungsweise abgebildet. Ein Mixture-Weibull wäre präziser, übersteigt aber den Kursrahmen.

**Kleine Stichprobe:** n = 100 Engines für 4 Prädiktoren ist vertretbar (n/p = 25), aber Konfidenzintervalle sind breit. Schätzungen sind entsprechend unsicher.

**In-Sample Concordance:** Der C-Index von 0.986 wurde auf den Trainingsdaten berechnet, nicht kreuzvalidiert. Das überschätzt die echte Prognosequalität. Für fairen Vergleich mit ML müsste man Leave-One-Out-CV auf Engine-Ebene rechnen.

**Degradations-Konvergenz:** Alle Engines enden auf ähnlichen Sensorwerten, was Profil-Vergleiche und damit die praktische Anwendbarkeit von Weibull AFT einschränkt.

---

## 11. Kernergebnisse auf einen Blick

### EDA

- 100 Training-Engines, bimodale Lebensdauerverteilung (zwei Fault Modes)
- 12 finale Sensoren nach VIF-Check (von 21)
- Weibull-Shape β = 2.93 → Verschleißausfall bestätigt

### Kaplan-Meier

- Median Gesamtpopulation: 220.5 Zyklen [204–234]
- Gruppe kurz (n=79): Median 201 Zyklen [194–220]
- Gruppe lang (n=21): Median 347 Zyklen [324–459]
- Log-Rank-Test (Gruppen über die Lebensdauer definiert, daher zirkulär): Chi² = 64.62, p = 8.88e-16
- Gruppierung nach BPR-Verlauf (56 steigend / 44 fallend): Median 193 bzw. 286 Zyklen, Log-Rank p = 3.4·10⁻¹¹

### Weibull AFT

- Baseline: β = 2.929, η = 276.8 Zyklen, AIC = 1177.08
- Covariaten-Modell: AIC = 1104.84 (ΔAIC = 72.24)
- Stärkster Prädiktor: sensor_11 (TR = 6.91)
- Einziger Risikofaktor: sensor_2 (TR = 0.36)

### Cox PH

| Modell | C-Index | PH global p |
|---|---|---|
| Statisch | 0.756 | 0.128 |
| Zeitabhängig | 0.986 | 0.197 |
| Stratifiziert | 0.978 | 0.165 |

PH-Annahme hält global in allen drei Modellen. Baseline für ML-Vergleich: C = 0.756.

---

## 12. Setup und Stolpersteine

| Komponente | Version | Hinweis |
|---|---|---|
| R | 4.5.1 | Paketstände in `renv.lock` |
| renv | 1.2.3 | projekteigene Bibliothek in `statistik/` |
| Quarto CLI | – | zum Rendern der Analysen |
| RTools | 4.5 | nur unter Windows, für Pakete aus dem Quellcode |

**Schnellstart:** R im Repo-Root starten. Das `.Rprofile` dort wechselt nach `statistik/` und aktiviert renv.

```r
renv::restore()                                        # Pakete aus renv.lock
library(tidyverse); library(survival); library(shiny)  # Smoke-Test
shiny::runApp("shiny-app")                             # Dashboard
```

`getwd()` und `renv::project()` müssen danach beide auf `…/statistik` zeigen. `quarto render analysis/` rendert die Analysen; sie brauchen die Rohdaten in `data/raw/CMaps/` (siehe [../data/README.md](../data/README.md)).

**Bekannte Stolpersteine**

- **R-Version:** R 4.5.1 statt 4.6.0. Für 4.6.0 gab es im Mai 2026 noch nicht für alle Pakete fertige Windows-Binaries; die Installation brach mit „failed to download“ ab.
- **Symlinks unter Windows:** renv verlinkt Pakete standardmäßig aus dem Cache, Windows blockiert Symlinks ohne Admin-Rechte. Abhilfe: `options(renv.config.symlinks.enabled = FALSE)` vor der Installation.
- **Falscher Startordner:** VSCode mit `predictive-maintenance/` als Root öffnen, sonst greift das `.Rprofile` nicht und renv landet im falschen Ordner. Eine Einstellung `r.workingDirectory` gibt es in der R-Extension nicht.
- **Einzelne Pakete scheitern als Binary:** `install.packages("Rcpp", type = "source")` bzw. `renv::install(c("lifecycle", "R6"), rebuild = TRUE)`.
- **CRAN-Mirror mit SSL-Fehler:** `options(repos = c(CRAN = "https://cloud.r-project.org"))`.
- **27 statt 26 Spalten:** Jede Zeile der NASA-Dateien endet mit einem Leerzeichen, `read_table` sieht deshalb eine leere 27. Spalte. Die Loader in `shiny-app/R/data_loader.R` benennen sie `dummy` und überspringen sie.
- **Plots in VSCode:** Plots mit `ggsave()` speichern oder `httpgd` installieren (`r.plot.useHttpgd = true`).

---

*ML-Teil: [../ml/README.md](../ml/README.md)*
