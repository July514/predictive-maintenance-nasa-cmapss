# statistik/shiny-app/R/data_loader.R
# ─────────────────────────────────────────────────────────────────────────────
# Lädt die in R aufbereiteten CSVs aus data/processed/.
# Konsistent mit analysis/02_preprocessing.qmd: Rolling Stats, Normalisierung,
# RUL-Cap bei 125 sind alle bereits in den CSVs enthalten.
# ─────────────────────────────────────────────────────────────────────────────

library(dplyr)
library(readr)

# Pfade relativ zur App (statistik/shiny-app/) → ../../data/processed/
DATA_PROC <- "../../data/processed"

#' Aufbereitete Trainingsdaten laden
#'
#' Enthält: 24.720 Zeilen, 78 Spalten
#'   - unit_number, time_in_cycles, os_1..3
#'   - 12 finale Sensoren (nach SD- und VIF-Filtering)
#'   - 12 Rolling Mean (window=5) + 12 Rolling SD
#'   - 36 normalisierte Versionen
#'   - RUL (gecappt bei 125)
#'   - NAs in ersten 4 Zyklen jeder Engine (Rolling-Fenster)
load_processed_train <- function() {
  path <- file.path(DATA_PROC, "train.csv")

  if (!file.exists(path)) {
    stop(sprintf("train.csv nicht gefunden unter: %s\n",
                 normalizePath(path, mustWork = FALSE)),
         "Bitte 02_preprocessing.qmd in R ausführen.")
  }

  df <- read_csv(path, show_col_types = FALSE)

  # Rolling-NAs entfernen (erste 4 Zeilen pro Engine bei window=5)
  before <- nrow(df)
  df     <- df %>% filter(!is.na(.data[[grep("_roll_mean$", names(df), value = TRUE)[1]]]))
  message(sprintf("Train: %d Zeilen geladen (%d Rolling-NAs entfernt)",
                  nrow(df), before - nrow(df)))

  df
}

#' Aufbereitete Testdaten laden
#'
#' Enthält 100 Zeilen (letzte Beobachtung pro Engine) + wahre RUL.
load_processed_test <- function() {
  path <- file.path(DATA_PROC, "test.csv")

  if (!file.exists(path)) {
    stop(sprintf("test.csv nicht gefunden unter: %s", normalizePath(path, mustWork = FALSE)))
  }

  df <- read_csv(path, show_col_types = FALSE)
  message(sprintf("Test: %d Engines geladen", nrow(df)))
  df
}


# ─────────────────────────────────────────────────────────────────────────────
# Rohdaten-Loader für die Quarto-Analysen (statistik/analysis/01–05)
# ─────────────────────────────────────────────────────────────────────────────
# Diese Funktionen wurden beim Umbau dieser Datei für die Shiny-App entfernt
# (Commit f2bb0ae, 31.05.2026), die Analysen 01–05 rufen sie aber weiter auf.
# Unverändert aus dem Stand davor (f2bb0ae^) wiederhergestellt.
# 02_preprocessing.qmd erzeugt damit data/processed/train.csv und test.csv.

# Spalten-Namen nach NASA-Dokumentation
CMAPSS_COLS <- c(
  "unit_number", "time_in_cycles",
  paste0("os_", 1:3),
  paste0("sensor_", 1:21)
)

#' Rohdaten einer CMAPSS-Datei laden
#'
#' @param file_path Pfad zur .txt-Datei (z.B. "train_FD003.txt")
#' @return data.frame mit benannten Spalten
load_cmapss_raw <- function(file_path) {
  col_names <- c("unit_number", "time_in_cycles",
                 paste0("os_", 1:3),
                 paste0("sensor_", 1:21),
                 "dummy")   # 27. leere Spalte benennen

  df <- read_table(
    file_path,
    col_names  = col_names,
    col_types  = cols(.default = col_double(), dummy = col_skip()),
    show_col_types = FALSE
  )

  return(df)
}

#' RUL für Trainingsdaten berechnen
#'
#' @param df Roher Trainings-Dataframe
#' @param rul_cap Maximale RUL (NASA-Konvention: 125)
#' @return df mit zusätzlicher Spalte `RUL`
add_rul <- function(df, rul_cap = 125) {
  df %>%
    group_by(unit_number) %>%
    mutate(
      max_cycle = max(time_in_cycles),
      RUL       = pmin(max_cycle - time_in_cycles, rul_cap)
    ) %>%
    ungroup() %>%
    select(-max_cycle)
}

#' Konstante Sensoren entfernen (nahezu keine Varianz über alle Engines)
#'
#' @param df Dataframe
#' @param threshold SD-Schwellwert unter dem ein Sensor als konstant gilt
#' @return df ohne konstante Sensor-Spalten
drop_constant_sensors <- function(df, threshold = 0.001) {
  # Bei FD003 typischerweise: sensor_1, 5, 6, 10, 16, 18, 19
  sensor_cols <- grep("^sensor_", names(df), value = TRUE)
  sds         <- sapply(df[sensor_cols], sd, na.rm = TRUE)
  keep        <- sensor_cols[sds > threshold]

  message(sprintf("Sensoren behalten: %d / %d", length(keep), length(sensor_cols)))
  message(sprintf("Gedroppt: %s", paste(sensor_cols[sds <= threshold], collapse = ", ")))

  df[, c(setdiff(names(df), sensor_cols), keep)]
}

#' Komplett aufbereitete Trainingsdaten laden
#'
#' @param base_path Pfad zum CMaps-Ordner
#' @return Aufbereiteter Dataframe
load_train_fd003 <- function(base_path = "../../data/raw/CMaps") {
  path <- file.path(base_path, "train_FD003.txt")
  df   <- load_cmapss_raw(path)
  df   <- add_rul(df)
  df   <- drop_constant_sensors(df)
  df
}

#' Test-Daten laden inkl. wahrer RUL aus separater Datei
#'
#' @param base_path Pfad zum CMaps-Ordner
#' @return Test-Dataframe mit `RUL_true`
load_test_fd003 <- function(base_path = "../../data/raw/CMaps") {
  test <- load_cmapss_raw(file.path(base_path, "test_FD003.txt"))
  test <- drop_constant_sensors(test)

  rul_true <- read_table(file.path(base_path, "RUL_FD003.txt"),
    col_names = c("RUL_true", "dummy"),
    col_types = cols(RUL_true = col_double(), dummy = col_skip()),
    show_col_types = FALSE)
  rul_true$unit_number <- seq_len(nrow(rul_true))

  # Nur letzte Zeile pro Engine (Zustand zum Evaluierungszeitpunkt)
  last_cycle <- test %>%
    group_by(unit_number) %>%
    slice_tail(n = 1) %>%
    ungroup()

  left_join(last_cycle, rul_true, by = "unit_number")
}
# Temporär für Rolling Stats – komplette Testverläufe
load_test_fd003_full <- function(base_path = "../../data/raw/CMaps") {
  test <- load_cmapss_raw(file.path(base_path, "test_FD003.txt"))
  test <- drop_constant_sensors(test)
  test
}
