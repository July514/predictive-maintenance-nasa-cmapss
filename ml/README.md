# ML-Teil: Vollständige Erklärung
## Predictive Maintenance – NASA CMAPSS FD003
> Konzeptuelle Erklärung des ML-Teils: Feature Engineering, Cross-Validation, Random Forest, SVR, BiLSTM + Attention, Vergleich und Reproduzierbarkeit.  
> Notebooks: `notebooks/03`–`09`, kompletter Lauf mit `bash scripts/run_all.sh` (aus `ml/`).

---

## Inhaltsverzeichnis

1. [Die zentrale Story](#1-die-zentrale-story)
2. [Vom Statistik- zum ML-Teil: Was sich ändert](#2-vom-statistik--zum-ml-teil-was-sich-ändert)
3. [Daten: Zwei Formate für zwei Modellfamilien](#3-daten-zwei-formate-für-zwei-modellfamilien)
4. [Cross-Validation: Warum GroupKFold?](#4-cross-validation-warum-groupkfold)
5. [Random Forest](#5-random-forest)
6. [Support Vector Regression](#6-support-vector-regression)
7. [BiLSTM + Attention](#7-bilstm--attention)
8. [Die Ergebnisse – und warum BiLSTM gewonnen hat](#8-die-ergebnisse--und-warum-bilstm-gewonnen-hat)
9. [NASA Score vs. RMSE: Zwei verschiedene Wahrheiten](#9-nasa-score-vs-rmse-zwei-verschiedene-wahrheiten)
10. [Methodische Einschränkungen (Ehrlichkeit)](#10-methodische-einschränkungen-ehrlichkeit)
11. [Kernergebnisse auf einen Blick](#11-kernergebnisse-auf-einen-blick)
12. [Notebooks und Reproduzierbarkeit](#12-notebooks-und-reproduzierbarkeit)

---

## 1. Die zentrale Story

Der ML-Teil beantwortet eine andere Frage als der Statistik-Teil — nicht warum, sondern **wann genau**:

**Ausgangsfrage:** Wie viele Zyklen hat diese konkrete Engine noch?

**Drei methodisch unterschiedliche Antworten:**

1. **Random Forest:** Ensemble aus 200 Entscheidungsbäumen, trainiert auf 36 tabellarischen Features (Sensoren + Rolling Stats aus R). Robuster Baseline-Ansatz, RMSE 20.72.
2. **SVR (RBF):** Kernel-basierte Regression auf denselben tabellarischen Features — methodisch komplett anderer Ansatz, mathematisch unabhängiger Vergleichspunkt zu RF. RMSE 21.12, aber bester NASA Score unter den tabellarischen Modellen (1809).
3. **BiLSTM + Attention:** Deep-Learning-Modell, das die letzten 50 Zyklen als Sequenz verarbeitet — nicht als einzelne Zeile wie RF und SVR. RMSE **13.33**, NASA Score **265** — in **allen** Metriken das beste Modell, mit deutlichem Abstand (bester von mehreren Trainingsläufen, siehe Abschnitt 10).

Das zentrale Ergebnis: **BiLSTM schlägt RF beim RMSE um 36 %.** Der Grund ist kein "Deep Learning ist generell besser", sondern ein methodischer: RF und SVR sehen pro Vorhersage eine Momentaufnahme — den Sensorzustand zum Zeitpunkt t. BiLSTM sieht den *Verlauf* der letzten 50 Zyklen und kann daraus lernen, wie schnell eine Engine degradiert, nicht nur wie sie aktuell aussieht. Das ist der fundamentale Unterschied, und er erklärt den gesamten Sprung.

Ein viertes Modell (XGBoost) wurde in einer frühen Phase ebenfalls evaluiert (RMSE 20.84, NASA Score 1941 — zwischen RF und SVR, ohne eine eigene methodische Perspektive zu liefern) und aus der finalen Modellauswahl entfernt. Begründung: Mit RF und SVR sind bereits Baum-Ensemble und Kernel-Methode als tabellarische Ansätze vertreten — ein zweites Baum-Ensemble fügt wenig methodische Vielfalt hinzu. BiLSTM bringt dagegen eine *grundlegend andere* Perspektive (sequenziell statt tabellarisch) und ist damit der lehrreichere dritte Baustein.

---

## 2. Vom Statistik- zum ML-Teil: Was sich ändert

### Andere Frage, anderes Werkzeug

Statistik fragte: Welche Sensoren beeinflussen die Lebensdauer — und wie stark? Die Antwort waren Time Ratios und Hazard Ratios mit Konfidenzintervallen.

ML fragt: Gegeben diese Sensorwerte (bzw. diesen Sensorverlauf) — wie viele Zyklen bleiben noch? Die Antwort ist eine Zahl in Zyklen, bewertet mit RMSE und NASA Score.

Das ist keine Hierarchie (ML besser als Statistik), sondern eine Verschiebung der Fragestellung. Statistik erklärt das System. ML prognostiziert konkrete Maschinen.

### Was Statistik für ML vorbereitet hat

Der Statistik-Teil hat zwei direkte Beiträge zum ML-Teil geleistet:

**Feature-Selektion:** Die 12 finalen Sensoren nach VIF-Check aus der EDA sind die Basis für *alle* drei Modelle — auch für BiLSTM, das diese 12 Sensoren als Rohwerte in der Sequenz verwendet. Ohne diesen Schritt würden redundante Sensoren (sensor_12 mit VIF 52.7, sensor_9 mit VIF 14.9) die Modelle verzerren.

**Preprocessing für die tabellarischen Modelle:** Rolling Mean und Rolling Std (window=5) wurden vollständig in R berechnet und in `data/processed/train.csv` exportiert. RF und SVR lesen diese fertigen Features — kein doppeltes Preprocessing. BiLSTM dagegen verwendet einen eigenen Datenpfad (siehe Abschnitt 3).

### Was ML anders macht

- Keine Verteilungsannahmen (kein Weibull, keine PH-Bedingung)
- Keine Linearitätsannahme (Bäume und neuronale Netze lernen beliebige Nicht-Linearitäten)
- Keine Zensierungs-Behandlung nötig (ML nutzt nur die Trainings-Engines, die alle bis zum Ausfall laufen)
- Dafür: keine Konfidenzintervalle, keine kausale Interpretation, keine testbaren Hypothesen

---

## 3. Daten: Zwei Formate für zwei Modellfamilien

### Tabellarisch: RF und SVR

RF und SVR lesen direkt `data/processed/train.csv` (24.320 Zeilen nach Rolling-NA-Entfernung) und `test.csv` (100 Zeilen, eine pro Engine). Pro Zeile stecken darin:

- 12 rohe Sensoren (finale Auswahl nach VIF)
- 12 Rolling Mean (window=5) — glättet kurzfristiges Rauschen
- 12 Rolling Std (window=5) — misst lokale Volatilität als Degradationssignal
- Normalisierte Versionen aller obigen (für SVR relevant)
- RUL (gecappt bei 125)

Insgesamt 36 Features. Jede Zeile ist für RF und SVR eine unabhängige Beobachtung — eine Momentaufnahme des Zustands zum Zeitpunkt t.

**Warum window=5 für die Rolling Stats reicht:** Größere Fenster (window=30, window=50) würden stabilere Schätzer liefern, aber bei n=100 Engines gehen die ersten 29 bzw. 49 Zyklen jeder Engine als NAs verloren. Bei mittleren Lebensdauern von ~247 Zyklen ist das ein relevanter Datenverlust. Für RF und SVR rechtfertigt der Mehrwert das nicht.

> **Nachtrag (Oktober 2026, Notebook 09):** Diese Abwägung war zu vorsichtig. Mit `min_periods` lassen sich auch 50-Zyklen-Fenster ohne Datenverlust berechnen (am Anfang jeder Engine über die bisher verfügbaren Zyklen). Mit 50-Zyklen-Mittelwert, -Streuung und -Steigung plus `time_in_cycles` erreicht der Random Forest Test-RMSE 14.59 statt 21.15 und schließt rund 83 % des Abstands zum BiLSTM. Den Ausschlag gibt der Trend (Steigung), nicht die stärkere Glättung.

### Sequenziell: BiLSTM

BiLSTM bekommt einen komplett anderen Input: Tensoren der Form `(batch, window=50, features=12)`. Pro Vorhersage ist das kein einzelner Zeitpunkt, sondern die **letzten 50 Zyklen** der 12 finalen Sensoren — normalisiert, aber *ohne* Rolling Mean/Std.

**Das ist eine bewusste und wichtige Entscheidung:** Die Rolling-Features aus dem R-Preprocessing wären für BiLSTM kontraproduktiv. Ein gleitender Mittelwert über 5 Zyklen *ist* bereits eine geglättete, vorverarbeitete Repräsentation der Zeitreihe — würde man diese Werte als Input für ein Modell verwenden, das selbst dafür gebaut ist, zeitliche Muster zu lernen, würde man dem Modell genau die Rohinformation entziehen, aus der es lernen soll. Kurz: **Rolling Features würden die temporalen Muster zerstören, die das LSTM selbst extrahieren soll.** BiLSTM bekommt deshalb die normalisierten Rohsensoren als Sequenz und lernt Glättung, Trend und Geschwindigkeit selbst — als Teil des Trainings, nicht als vorgegebenes Feature.

**Zwei "Window"-Begriffe, die nicht verwechselt werden dürfen:**

| Begriff | Bedeutung | Betrifft |
|---|---|---|
| Rolling-Window = 5 | Glättungsfenster im R-Preprocessing | RF, SVR (Feature Engineering) |
| Sequence-Window = 30 | Anzahl Zeitschritte, die BiLSTM pro Vorhersage sieht | BiLSTM (Modell-Input) |

Beide heißen "Window", lösen aber unterschiedliche Probleme auf unterschiedlichen Ebenen der Pipeline.

### Gemeinsame Basis trotz unterschiedlicher Formate

Trotz der unterschiedlichen Datenformate teilen sich alle drei Modelle: dieselben 12 finalen Sensoren (VIF-Check aus der Statistik), dieselbe Min-Max-Normalisierung (Parameter nur aus Training, vgl. Data-Leakage-Diskussion im Statistik-Teil), und dieselbe RUL-Definition (gecappt bei 125 nach NASA-Konvention).

---

## 4. Cross-Validation: Warum GroupKFold?

### Das Problem mit Standard-CV

Der Trainingsdatensatz hat 24.320 Zeilen (bzw. für BiLSTM: tausende Sequenzfenster) — aber das sind keine unabhängigen Beobachtungen. Es sind ~243 Zeilen pro Engine, und diese Zeilen sind stark autokorreliert: Zyklus 150 einer Engine hängt direkt von Zyklus 149 ab.

Standard-CV macht zufällige Splits. Zyklus 150 von Engine 5 landet in Validation, Zyklus 151 von Engine 5 im Training. Das Modell hat diese Engine quasi schon gesehen — die CV-Scores sind zu optimistisch. Das ist Information Leakage auf Zeilen-Ebene.

### Was GroupKFold macht

GroupKFold garantiert: Jede Engine landet komplett entweder in Train oder in Validation. Bei 5 Folds trainiert das Modell auf 80 Engines und validiert auf 20 Engines, die es nie gesehen hat. Das entspricht dem echten Einsatzszenario — neue, unbekannte Maschinen bewerten.

```
Standard-CV:  Engine 5, Zyklus 1–100  → Train
              Engine 5, Zyklus 101–200 → Validation  ← Leakage!

GroupKFold:   Engine 1–80  → Train
              Engine 81–100 → Validation             ← sauber
```

### Konsequenz für BiLSTM: Validation-Split fürs Early Stopping

BiLSTM braucht — wie jedes neuronale Netz mit Early Stopping — einen Validation-Split *während* des Trainings, um zu erkennen, wann das Modell anfängt zu überanpassen. Dieser Split erfolgt ebenfalls über GroupKFold auf Engine-Ebene: Ein Teil der Trainings-Engines wird komplett zurückgehalten und nur zur Überwachung des Validation-RMSE verwendet (Early Stopping mit `patience=15`). Anders als beim klassischen zweistufigen Boosting-Verfahren muss hier *nicht* in einem zweiten Schritt nachtrainiert werden — neuronale Netze werden ohnehin pro Fold von Grund auf neu trainiert, das Early-Stopping-Kriterium bestimmt nur, bei welcher Epoche das jeweilige Training gestoppt wird.

---

## 5. Random Forest

### Wie RF funktioniert

Random Forest baut eine große Anzahl von Entscheidungsbäumen, jeder auf einer zufälligen Untermenge der Daten (Bootstrapping) und einer zufälligen Untermenge der Features. Die finale Vorhersage ist der Durchschnitt aller Bäume.

Ein einzelner Entscheidungsbaum fragt zum Beispiel:

```
sensor_11_roll_mean > 47.3?
  Ja → sensor_2_roll_mean > 542.1?
         Ja → RUL = 18 Zyklen
         Nein → RUL = 34 Zyklen
  Nein → sensor_17 > 394.2?
           ...
```

Der Baum teilt den Datensatz an jedem Knoten so auf, dass der Vorhersagefehler (MSE) maximal reduziert wird. Jeder Baum sieht andere Datenpunkte und andere Features — das ist die Quelle der Varianz-Reduktion.

**Warum Ensemble?** Ein einzelner tiefer Baum überanpasst stark — er lernt Rauschen auswendig. Viele verschiedene Bäume mit zufälligen Untermengen, dann gemittelt, reduzieren die Varianz bei moderater Zunahme der Verzerrung. Das nennt man Bagging.

### Feature Importance

RF liefert eine direkt messbare Feature Importance: wie viel reduziert dieses Feature den MSE über alle Bäume und alle Splits, gemittelt. Das ist kein kausaler Koeffizient wie in Cox — es ist ein Maß für Prognose-Beitrag.

Erwartetes Ergebnis: Die Rolling-Mean-Features der Sensoren mit hoher Korrelation zu RUL (sensor_11, sensor_4, sensor_17) sollten dominieren. Rolling Std sollte ebenfalls wichtig sein — Volatilität steigt in der Degradationsphase.

### Hyperparameter

| Parameter | Gewählt | Bedeutung |
|---|---|---|
| `n_estimators` | 200 | Anzahl Bäume — mehr ist stabiler, aber diminishing returns (Neulauf Okt. 2026 auf macOS: Grid wählt 500) |
| `max_depth` | None | Bäume wachsen bis Blätter rein sind — Grid Search findet Optimum |
| `min_samples_leaf` | 1–10 | Mindestgröße eines Blattes — verhindert Überanpassung |
| `max_features` | sqrt, 0.5 | Features pro Split — Quelle der Dekorrelation zwischen Bäumen |

### Stärken und Schwächen

**Stärken:**
- Keine Feature-Skalierung nötig (Bäume sind invariant gegenüber monotonen Transformationen)
- Robust gegenüber Ausreißern
- Feature Importance direkt verfügbar
- Wenige kritische Hyperparameter, funktioniert auch bei kleinen Datensätzen robust

**Schwächen:**
- Kann keine Extrapolation (außerhalb des Trainingsbereichs werden nur Trainingswerte vorhergesagt)
- Kein Unsicherheitsmaß für einzelne Vorhersagen
- Sieht pro Vorhersage nur eine Momentaufnahme — keinen Verlauf

---

## 6. Support Vector Regression

### Wie SVR funktioniert

SVR sucht eine Regressionsfunktion f(x), die möglichst alle Datenpunkte innerhalb eines Toleranzbands der Breite ±ε trifft:

```
Ziel: |y - f(x)| ≤ ε  für möglichst viele Punkte
Gleichzeitig: f(x) möglichst "flach" (kleine Gewichte)
```

Punkte innerhalb des ε-Bands kosten nichts. Punkte außerhalb werden mit dem Abstand zum Band bestraft (ε-insensitive Loss). Der Parameter C kontrolliert den Trade-off: großes C → wenige Verletzungen toleriert → komplexeres Modell; kleines C → mehr Toleranz → glatteres Modell.

### Der RBF-Kernel

Mit dem RBF-Kernel (Radial Basis Function) kann SVR nicht-lineare Entscheidungsgrenzen lernen. Der Trick: Die Daten werden implizit in einen hochdimensionalen Raum transformiert, in dem eine lineare Regression möglich ist. Man berechnet diesen Raum nie explizit — nur die Ähnlichkeit zwischen Punkten:

$$K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)$$

γ kontrolliert die "Reichweite": kleines γ → jeder Punkt beeinflusst viele andere (globales Modell); großes γ → nur nahe Punkte beeinflussen sich (lokales Modell).

### Warum Standardisierung Pflicht ist

SVR berechnet Abstände zwischen Datenpunkten. Wenn sensor_2 Werte um 540 hat und sensor_10 Werte um 1.2, dann dominiert sensor_2 den Abstand rein durch seine Skala — obwohl beide gleich informativ sein könnten. Standardisierung (Mittelwert 0, SD 1) gibt allen Features die gleiche Einheit.

**Kritisch:** Der StandardScaler muss innerhalb der Cross-Validation gefittet werden — auf den Trainingsdaten jedes Folds, nicht auf allen Daten. Sonst fließen Mittelwert und SD der Validation-Daten ins Training: Information Leakage. Die `sklearn Pipeline` macht das automatisch korrekt.

### Stärken und Schwächen

**Stärken:**
- Kernel-Trick erlaubt nicht-lineare Modelle ohne explizite Feature-Transformation
- Robust gegenüber Ausreißern (dank ε-Band)
- Gut motivierter theoretischer Rahmen (Maximale Margin)
- Methodisch komplett verschieden von Baummodellen → guter Vergleichspunkt zu RF

**Schwächen:**
- O(n²)-Komplexität bei der Trainingszeit
- Keine direkte Feature Importance
- Skalierung zwingend nötig
- Empfindlich gegenüber Hyperparameter-Wahl (C, γ, ε)
- Wie RF: sieht pro Vorhersage nur eine Momentaufnahme, keinen Verlauf

---

## 7. BiLSTM + Attention

### Warum überhaupt ein sequenzielles Modell?

RF und SVR behandeln jede Zeile unabhängig — sie sehen den Sensorzustand zum Zeitpunkt t, sonst nichts. Aus der Statistik-EDA wissen wir aber: Engines fallen nicht aus, weil sie ein bestimmtes Alter erreichen, sondern weil sie einen kritischen Degradationszustand erreichen — und der Weg dahin (wie schnell, mit welcher Beschleunigung) trägt Information, die eine einzelne Momentaufnahme nicht enthält. BiLSTM ist das einzige der drei Modelle, das diesen *Verlauf* direkt sieht.

### Was ist ein LSTM?

Ein normales (Feed-Forward-)Netz verarbeitet jeden Input unabhängig — es hat kein Gedächtnis. Ein LSTM (Long Short-Term Memory) dagegen besitzt einen internen Zustand (Cell State), der über Zeitschritte hinweg weitergegeben wird. Bei jedem neuen Zeitschritt entscheidet das LSTM über drei "Gates": Was wird aus dem Gedächtnis gelöscht (Forget Gate), was wird neu aufgenommen (Input Gate), und was wird als Output weitergegeben (Output Gate). Dadurch kann ein LSTM Muster über viele Zeitschritte hinweg "merken" — zum Beispiel: "sensor_11 ist seit 10 Zyklen kontinuierlich gefallen".

### Bidirektional: vorwärts *und* rückwärts

"Bi" in BiLSTM steht für bidirektional: Die Sequenz wird zweimal durchlaufen — einmal vorwärts (Zyklus 1 → 30) und einmal rückwärts (Zyklus 30 → 1). Beide Richtungen haben eigene LSTM-Zellen, deren Outputs am Ende konkateniert werden. Bei 128 Units pro Richtung ergibt das 256 Dimensionen pro Zeitschritt.

**Warum das hilft:** Innerhalb des 30-Zyklen-Fensters können Informationen aus früheren *und* späteren Zeitpunkten kombiniert werden. Ein Muster, das in Zyklus 15 beginnt, wird robuster erkannt, wenn das Modell auch weiß, wie es sich bis Zyklus 30 entwickelt.

### Attention: welche Zeitschritte zählen?

Ein LSTM allein komprimiert die gesamte Sequenz am Ende in einen einzigen Zustandsvektor — bei 50 Zeitschritten kann dabei früh im Fenster liegende Information verwässert werden. Der Attention-Mechanismus löst das: Er berechnet für jeden der 50 Zeitschritte ein Gewicht (zwischen 0 und 1) und bildet eine gewichtete Summe über alle Zeitschritt-Repräsentationen. Praktisch bedeutet das: Zyklen kurz vor dem aktuellen Zeitpunkt (potenziell näher am Ausfall) können stärker gewichtet werden als frühe, stabile Phasen — das Modell lernt selbst, welche Abschnitte des Fensters am informativsten sind.

### Die Architektur im Detail

```
Input  (batch, 50, 12)
  │
BiLSTM(256)        # 256 Units vorwärts + 256 rückwärts, Output: (batch, 50, 512)
  │
Dropout(0.3)       # 30% der Verbindungen zufällig deaktiviert (nur im Training)
  │
BiLSTM(128)        # 128 + 128 Units, verdichtet die Sequenz weiter, Output: (batch, 50, 256)
  │
Dropout(0.3)
  │
Attention          # Dense(1, tanh) → Softmax über die 50 Zeitschritte → gewichtete Summe → (batch, 256)
  │
Dense(32)          # nichtlineare Kombination (ReLU)
  │
Output (1)         # lineare Aktivierung — Regression, keine Klassifikation
```

Insgesamt 1.215.810 trainierbare Parameter.

**Schicht für Schicht:**

- **BiLSTM(256):** Extrahiert grobe zeitliche Muster aus dem Rohsignal. Output bleibt eine Sequenz (50 Zeitschritte), weil die nächste Schicht ebenfalls eine Sequenz braucht.
- **Dropout(0.3):** Regularisierung. Während des Trainings werden zufällig 30% der Verbindungen abgeschaltet — bei jedem Batch anders. Das Netz kann sich nicht auf einzelne Neuronen verlassen und lernt robustere, redundantere Repräsentationen. Bei Inference ist Dropout deaktiviert.
- **BiLSTM(128):** Kleiner als die erste Schicht — verdichtet die extrahierten Muster weiter. Typisches Deep-Learning-Muster: erst breit (viele Merkmale extrahieren), dann enger (auf das Relevante fokussieren).
- **Attention:** Übergang von "Sequenz verarbeiten" zu "eine Aussage treffen" — fasst die 50 gewichteten Zeitschritt-Repräsentationen zu einem einzigen Vektor zusammen.
- **Dense(32):** Normales Fully-Connected-Layer mit ReLU-Aktivierung, lernt nichtlineare Kombinationen der von Attention extrahierten Features.
- **Output:** Ein Neuron, lineare Aktivierung (kein Sigmoid/Softmax) — gibt die vorhergesagte RUL direkt als Zahl aus.

### Training

| Parameter | Gewählt | Bedeutung |
|---|---|---|
| `window` | 50 | Sequenzlänge — größerer Kontext als 30, bestes Validierungsergebnis; kostet die ersten 49 Zyklen jeder Engine als Targets |
| Batch / Epochen | 64 / max. 80 | im Originallauf Early Stopping nach 61 Epochen, beste Gewichte aus Epoche 46 |
| `ReduceLROnPlateau` | Faktor 0.5, patience=7 | Lernrate halbiert sich bei Stagnation (im Originallauf nach Epoche 12, 53 und 60) |
| Optimizer | Adam, lr=0.001 | adaptive Lernrate, pro Gewicht automatisch skaliert |
| Early Stopping | patience=15 | Training stoppt, wenn sich Val-RMSE 15 Epochen nicht verbessert |
| Validierung | GroupKFold | Engine-Trennung, siehe Abschnitt 4 |

**Zur Lernrate kurz:** Adam mit lr=0.001 bestimmt, wie groß die Schritte sind, mit denen die rund 1,2 Mio. Gewichte bei jedem Update angepasst werden. Zu groß: das Training oszilliert oder divergiert. Zu klein: das Training konvergiert kaum innerhalb sinnvoller Zeit. lr=0.001 ist bei BiLSTM-Architekturen ein etablierter, vorsichtiger Startwert — Adam passt die effektive Schrittgröße pro Gewicht zusätzlich automatisch an, basierend auf den bisherigen Gradienten.

### Eine technische Besonderheit: `Lambda`-Layer und Serialisierung

Das Attention-Pooling summiert die gewichteten Zeitschritt-Vektoren mit einem `Lambda`-Layer (`tf.reduce_sum(x, axis=1)`). Dieser Layer steckt auch im gespeicherten Checkpoint des Originallaufs (`ml/reports/lstm_best.keras`). Keras speichert eine Lambda-Funktion als Python-Bytecode. Ein komplettes `load_model` müsste diesen Bytecode deserialisieren: Das ist an die Python-Version gebunden (hier 3.13), braucht `safe_mode=False`, und die Funktion findet `tf` danach nicht im eigenen Namensraum.

Notebook 08 lädt den Checkpoint deshalb anders: `build_model()` baut die Architektur im Code neu auf (die Layer-Namen entsprechen exakt dem Checkpoint), und `model.load_weights("lstm_best.keras")` lädt nur die Gewichte. Ein Abgleich des Validierungs-Loss mit dem gespeicherten Trainingslog (Epoche 46, `val_loss` 138.05) bestätigt, dass die richtigen Gewichte geladen wurden.

Für künftige Trainingsläufe wäre eine eigene, mit `@keras.saving.register_keras_serializable()` registrierte Layer-Klasse robuster. Hier bleibt der Lambda-Layer, damit der Checkpoint die Werte des Originallaufs reproduziert.

### Stärken und Schwächen

**Stärken:**
- Sieht den tatsächlichen Verlauf, nicht nur eine Momentaufnahme — dadurch deutlich präzisere Vorhersagen (siehe Abschnitt 8)
- Attention macht teilweise sichtbar, welche Zeitschritte das Modell als relevant gewichtet
- Lernt Glättung und Trend-Extraktion selbst, statt auf vorgegebene Rolling-Features angewiesen zu sein

**Schwächen:**
- Deutlich mehr Parameter (rund 1,2 Mio.) bei n=100 Engines — Overfitting-Risiko, das durch Dropout, Early Stopping und GroupKFold adressiert, aber nicht eliminiert wird
- "Black Box": Trotz Attention keine Konfidenzintervalle, keine kausale Interpretation wie bei Time Ratios/Hazard Ratios
- Längere Trainingszeit pro Fold als RF/SVR
- Serialisierung/Deployment empfindlicher (siehe Custom-Layer-Hinweis oben)

---

## 8. Die Ergebnisse – und warum BiLSTM gewonnen hat

### Die Zahlen

| Modell | RMSE | MAE | R² | NASA Score |
|---|---|---|---|---|
| Random Forest | 20.72 | 15.40 | 0.749 | 2051 |
| SVR (RBF) | 21.12 | 15.73 | 0.740 | 1809 |
| **BiLSTM + Attention** | **13.33** | **10.05** | **0.896** | **265** |

BiLSTM gewinnt in **allen vier Metriken**, mit großem Abstand: RMSE −36 % gegenüber RF, NASA Score um den Faktor 6.8 niedriger als bei SVR (dem bisher besten Modell in dieser Metrik). Einschränkung: 13.33 ist der beste von mehreren Trainingsläufen (siehe Abschnitt 10).

### Warum ist BiLSTM so viel besser?

Der Grund ist nicht "Deep Learning ist mächtiger" — es ist eine Frage dessen, *welche Information jedes Modell überhaupt sehen kann*:

- RF und SVR bekommen 36 Zahlen: den aktuellen Sensorstatus plus geglättete Mittelwerte/Volatilität der letzten 5 Zyklen. Das ist eine Momentaufnahme mit etwas Kontext.
- BiLSTM bekommt 50 × 12 = 600 Zahlen: den vollständigen Rohverlauf der letzten 50 Zyklen. Das Modell sieht direkt, *wie schnell* sich jeder Sensor verändert, ob sich die Veränderung beschleunigt, und wie die 12 Sensoren *zueinander* in der Zeit verlaufen.

Degradation ist ein Prozess, kein Zustand. Ein Modell, das den Prozess sieht, hat einen strukturellen Vorteil gegenüber einem Modell, das nur den aktuellen Zustand (plus ein bisschen Glättung) sieht — unabhängig davon, wie ausgefeilt das tabellarische Modell sonst ist.

### Warum RF knapp besser ist als SVR

Ein sekundärer, aber lehrreicher Befund: RF schlägt SVR beim RMSE leicht (20.72 vs. 21.12), aber SVR gewinnt beim NASA Score unter den tabellarischen Modellen (1809 vs. 2051). Bei n=100 Engines ist der Datensatz für ML-Verhältnisse klein — RF profitiert davon, dass es einfach über viele Bäume mittelt, was bei kleinen Stichproben robust funktioniert. SVR mit RBF-Kernel reagiert empfindlicher auf die Hyperparameter-Wahl (C, γ, ε) und ist bei kleinen Datensätzen weniger stabil zu tunen. Warum SVR trotzdem beim NASA Score gewinnt, erklärt Abschnitt 9.

---

## 9. NASA Score vs. RMSE: Zwei verschiedene Wahrheiten

### RMSE ist symmetrisch

RMSE gewichtet alle Fehler gleich:

$$RMSE = \sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$$

Wenn ein Modell bei 50 Engines um 20 Zyklen zu hoch vorhersagt und bei 50 Engines um 20 Zyklen zu niedrig, ist der RMSE derselbe wie bei einem Modell, das immer genau trifft mit ±20 Abweichung. Die Richtung des Fehlers spielt keine Rolle.

In der Realität ist die Richtung aber entscheidend:
- **Zu optimistisch** (Vorhersage > Wahre RUL): Die Engine wird zu spät gewartet. Im schlimmsten Fall ungeplanter Ausfall, Produktionsstopp, Schaden.
- **Zu pessimistisch** (Vorhersage < Wahre RUL): Die Engine wird zu früh gewartet. Unnötige Kosten, aber keine Sicherheitsgefahr.

### NASA Score ist asymmetrisch

Der NASA Score bildet diese Asymmetrie ab:

```
d = Vorhersage - Wahre RUL
d < 0 (zu pessimistisch): Strafe = exp(-d/13) - 1  ← mild
d > 0 (zu optimistisch):  Strafe = exp(d/10)  - 1  ← hart
```

```
d = +30 (zu optimistisch):  Strafe = exp(30/10) - 1 ≈ 19.1
d = -30 (zu pessimistisch): Strafe = exp(30/13) - 1 ≈ 9.1
```

Doppelte Strafe für dieselbe absolute Abweichung in die gefährliche Richtung.

### Was das für unsere drei Modelle bedeutet

Bei BiLSTM ist die Frage "welche Metrik zählt" akademisch — es gewinnt ohnehin beide, weil es keinen Fehler über +32.5 Zyklen macht. Für RF vs. SVR lag zunächst die Vermutung nahe, SVR sage systematisch leicht pessimistisch vorher. **Eine Nachprüfung (Oktober 2026) widerlegt das:** SVR liegt im Mittel *optimistischer* als RF (mittlerer Fehler +6.7 vs. +5.5 Zyklen, 63 % der Vorhersagen zu optimistisch). Der bessere NASA Score kommt allein von einem kleineren größten Fehler (+58.8 statt +65.2 Zyklen). Bei den tabellarischen Modellen machen die fünf schlechtesten Engines 65–75 % des NASA Scores aus. Die Rangfolge RF vs. SVR beim NASA Score hängt damit an einzelnen Ausreißern und ist nicht belastbar. Die eigentliche Lektion bleibt: Der RMSE bewertet alle Fehler, der NASA Score vor allem die wenigen großen optimistischen. Ein Modell für die Wartung muss vor allem grobe Fehleinschätzungen nach oben vermeiden.

### Welche Metrik zählt?

Das hängt vom Kontext ab. Für einen Proof-of-Concept ist RMSE die intuitivere Metrik — Zyklen sind direkt verständlich. Für eine reale Deployment-Entscheidung ist NASA Score oder eine ähnliche asymmetrische Metrik die ehrlichere Bewertung. In unserem Fall ist die Aussage einfach: **BiLSTM gewinnt unabhängig davon, welche Metrik man priorisiert.**

---

## 10. Methodische Einschränkungen (Ehrlichkeit)

**Kein fairer Vergleich mit Cox zeitabhängig:** Der C-Index 0.986 von Cox PH wurde In-Sample berechnet. ML wurde auf einem echten Holdout-Testset (100 Test-Engines) evaluiert. Das sind methodisch verschiedene Designs — kein direkter Vergleich möglich.

**Kein C-Index für ML:** ML-Modelle geben keine Rangkorrelation aus. Für einen fairen Vergleich mit Cox statisch (C = 0.756) müsste man den C-Index für die ML-Vorhersagen auf dem Testset berechnen (Harrell's C auf den 100 Test-Engines). Das ist möglich, wurde aber nicht gemacht.

**Kleiner Datensatz für Deep Learning:** BiLSTM hat deutlich mehr Parameter als RF oder SVR, aber nur n=100 Trainings-Engines zur Verfügung. Dropout, Early Stopping und GroupKFold-Validierung mindern Overfitting, eliminieren es aber nicht vollständig. Mit deutlich mehr Trainings-Engines wäre ein noch größerer Vorsprung von BiLSTM plausibel — oder ein stabileres, weniger varianzanfälliges Ergebnis.

**Keine Uncertainty Quantification:** Keines der drei Modelle gibt Konfidenzintervalle für einzelne Vorhersagen aus. In der Praxis ist "RUL = 47 ± 12 Zyklen" informativer als "RUL = 47 Zyklen". RF könnte das via Quantile Regression Forests, BiLSTM via Monte-Carlo-Dropout oder Quantile-Loss — beides wurde nicht implementiert.

**Sequence-Window=50 und Zero-Padding:** Drei Test-Engines haben weniger als 50 beobachtete Zyklen (Engine 37: 47, 52: 38, 88: 41). Ihre Sequenzen werden am Anfang mit Nullen aufgefüllt. Im Min-Max-Raum entspricht 0 dem Trainingsminimum jedes Sensors, ist also kein neutraler Wert. Sauberer wäre Edge-Padding oder ein `Masking`-Layer. Hier bleibt das Zero-Padding, damit der Checkpoint die Werte des Originallaufs reproduziert.

**Bester von mehreren Läufen:** Neuronale Netze sind nicht deterministisch. Der RMSE schwankt zwischen Trainingsläufen um etwa ±1 Zyklus. 13.33 ist der beste von mehreren Läufen mit einem einzelnen Validierungssplit; ein Mittelwert über Seeds wäre konservativer.

**XGBoost wurde nicht wegen schlechter Leistung entfernt:** RMSE 20.84 und NASA Score 1941 lagen zwischen RF und SVR — kein schwaches Modell. Die Entscheidung, es aus der finalen Modellauswahl zu nehmen, war eine *methodische*: Mit BiLSTM als drittem Modell ist die Vielfalt der Ansätze (Baum-Ensemble, Kernel-Methode, sequenzielles Deep-Learning-Modell) größer als mit zwei Baum-basierten Modellen (RF, XGBoost) plus SVR.

**Reproduzierbarkeit über Plattformen:** Ein Neulauf aller Notebooks auf macOS (Apple Silicon, Oktober 2026) mit denselben Paketversionen (`uv.lock`) ergibt für BiLSTM (Checkpoint) und SVR exakt die Werte des Originallaufs (Juni 2026, Windows). Random Forest (RMSE 20.66 statt 20.72, NASA 2012 statt 2051) und XGBoost (20.45 statt 20.84, NASA 1801 statt 1941) weichen leicht ab. Ursache sind Fließkomma-Unterschiede zwischen Windows/x86 und macOS/ARM: Bei XGBoost unterscheidet sich schon die unveränderte Baseline (beste Iteration 112 statt 85), und die Grid Searches wählen bei praktisch gleichem CV-Wert andere Parameter. Die README nennt die Werte des Originallaufs (`ml/reports/model_results_original.json`); die Notebooks zeigen den Neulauf (Details in Abschnitt 12). Die Unterschiede zwischen RF, XGBoost und SVR sind damit kleiner als die Streuung zwischen zwei Rechnern. Eine Rangfolge unter den tabellarischen Modellen ist nicht belastbar.

**Kleine Stichprobe:** 100 Test-Engines für die finale Evaluation. Metriken wie RMSE sind auf so kleinen Stichproben noch relativ instabil — die Konfidenzintervalle der Metriken selbst wären breit.

**Simulationscharakter:** Wie im Statistik-Teil — CMAPSS ist eine physikalische Simulation. Die Methodik ist übertragbar, die absoluten Zahlenwerte nicht ohne Weiteres auf reale Triebwerksflotten.

---

## 11. Kernergebnisse auf einen Blick

### Modell-Ergebnisse

| Modell | RMSE | MAE | R² | NASA Score | Bestes bei |
|---|---|---|---|---|---|
| Random Forest | 20.72 | 15.40 | 0.749 | 2051 | — |
| SVR (RBF) | 21.12 | 15.73 | 0.740 | 1809 | — |
| **BiLSTM + Attention** | **13.33** | **10.05** | **0.896** | **265** | **alle Metriken** |

### Warum BiLSTM gewonnen hat

- Sieht den Sensorverlauf der letzten 50 Zyklen statt einer Momentaufnahme
- Degradation ist ein Prozess — ein Modell, das den Prozess sieht, hat einen strukturellen Vorteil
- RMSE −36 % gegenüber RF, NASA Score um den Faktor 6.8 niedriger als bei SVR
- Bidirektionale LSTMs + Attention lassen das Modell selbst lernen, welche Zeitschritte relevant sind

### Warum RF und SVR sich unterschiedlich schlagen

- RF: besserer RMSE — robustes Mitteln über viele Bäume, funktioniert gut bei n=100
- SVR: besserer NASA Score unter den tabellarischen Modellen, weil sein größter optimistischer Fehler kleiner ist (+58.8 statt +65.2 Zyklen); ein systematisch pessimistischer Bias liegt nicht vor (Abschnitt 9)
- Die Unterschiede zwischen RF, XGBoost und SVR sind kleiner als die Abweichung zwischen zwei Rechnern (Abschnitt 10), also gleichauf

### Feature Engineering

- Tabellarisch (RF, SVR): 36 Features (12 Sensoren + 12 Rolling Mean + 12 Rolling Std, window=5), aus R importiert
- Sequenziell (BiLSTM): 12 Rohsensoren als (window=50 × 12)-Tensor, *ohne* Rolling Stats — bewusst, damit das Modell zeitliche Muster selbst lernt
- GroupKFold garantiert für alle drei Modelle eine saubere CV ohne Engine-Level-Leakage

### Methodischer Vergleich Statistik vs. ML

| Aspekt | Statistik | ML |
|---|---|---|
| Fragestellung | Warum fallen Engines aus? | Wann genau fällt diese Engine aus? |
| Output | Time Ratios, Hazard Ratios, C-Index | RUL in Zyklen (RMSE, NASA Score) |
| Annahmen | Weibull-Form, PH-Bedingung, Linearität | Keine parametrischen Annahmen |
| Interpretierbarkeit | Hoch (KI, Signifikanztests) | Mittel (Feature Importance) bis niedrig (BiLSTM) |
| Punktprognose | Indirekt, ungenau | Direkt, optimiert — BiLSTM mit Abstand am genauesten |
| Konfidenzintervalle | Ja | Nein (keines der drei Modelle) |
| Beide komplementär? | **Ja — verschiedene Fragen, verschiedene Stärken** | |

---

## 12. Notebooks und Reproduzierbarkeit

| Notebook | Inhalt |
|---|---|
| `03_baseline_rf.ipynb` | Random Forest, GroupKFold, Grid Search |
| `04_xgboost.ipynb` | XGBoost, Early Stopping, stufenweises Tuning (evaluiert, nicht in der finalen Auswahl) |
| `05_svr.ipynb` | SVR (RBF) mit StandardScaler in einer Pipeline, voller Datensatz |
| `06_comparison.ipynb` | Vergleichstabelle und Abbildungen aller Modelle |
| `07_example_prediction.ipynb` | Vorhersagen aller Modelle für einzelne Engines |
| `08_lstm_bilstm_attention.ipynb` | BiLSTM + Attention auf 50-Zyklen-Sequenzen; lädt standardmäßig den Checkpoint |
| `09_context_ablation.ipynb` | Kontext-Ablation: Random Forest mit wachsendem Zeitfenster (Oktober 2026) |

Alle Notebooks laufen headless in der richtigen Reihenfolge mit `bash scripts/run_all.sh` (aus `ml/`). Die Modell-Notebooks 03, 04, 05 und 08 schreiben ihre Metriken nach `reports/model_results.json` und die Vorhersagen pro Test-Engine nach `reports/predictions_test.csv`; 06, 07 und 09 lesen diese Dateien.

### Originallauf und Neulauf

| Modell | Originallauf (Juni 2026, Windows) | Neulauf (Oktober 2026, macOS) | Status |
|---|---|---|---|
| BiLSTM + Attention | 13.33 · 10.05 · 0.896 · 265.4 | 13.33 · 10.05 · 0.896 · 265.4 | exakt reproduziert |
| SVR (RBF) | 21.12 · 15.73 · 0.740 · 1809.1 | 21.12 · 15.73 · 0.740 · 1809.1 | exakt reproduziert |
| Random Forest | 20.72 · 15.40 · 0.749 · 2051.3 | 20.66 · 15.36 · 0.751 · 2011.9 | Plattformabweichung |
| XGBoost | 20.84 · 15.35 · 0.746 · 1940.5 | 20.45 · 15.21 · 0.756 · 1801.4 | Plattformabweichung |

Spalten: RMSE · MAE · R² · NASA Score. Die README nennt die Werte des Originallaufs (`reports/model_results_original.json`); `reports/model_results.json` und die Notebook-Ausgaben enthalten den Neulauf.

- **BiLSTM-Checkpoint:** `reports/lstm_best.keras` stammt aus dem Lauf im Trainingslog `reports/lstm_training_log_original.csv` (Val-Loss beim Laden 137.86, im Log bei Epoche 46 138.05).
- **Random Forest und XGBoost:** Die Abweichung kommt von Fließkomma-Unterschieden zwischen Windows/x86 und macOS/ARM. Bei XGBoost weicht schon die unveränderte Baseline ab (beste Iteration 112 statt 85); die Grid Searches wählen dann bei fast gleichem CV-Wert andere Parameter.
- **macOS:** TensorFlow hängt, wenn pandas (über pyarrow) vorher importiert wird; Notebook 08 importiert TensorFlow deshalb zuerst. XGBoost braucht `libomp` (`brew install libomp`); `scripts/run_notebooks.py` nutzt sonst die von scikit-learn mitgelieferte Kopie.
- **Laufzeit:** Die Grid Search in 03 dauert rund 12 Minuten. `nbconvert --execute` zeigt dabei keinen Fortschritt; `scripts/run_notebooks.py` zeigt jede Zelle mit Laufzeit.

---

*Finale Modellauswahl: Random Forest, SVR (RBF), BiLSTM + Attention; XGBoost evaluiert*  
*Statistik-Teil: [../statistik/README.md](../statistik/README.md)*
