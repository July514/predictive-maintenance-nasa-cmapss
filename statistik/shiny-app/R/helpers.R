# statistik/shiny-app/R/helpers.R
# ─────────────────────────────────────────────────────────────────────────────
# Kleine UI-Hilfsfunktionen die im Code überall genutzt werden.
# ─────────────────────────────────────────────────────────────────────────────

# Bootstrap column shortcuts (12-Grid-System)
col_2  <- function(...) column(width = 2,  ...)
col_3  <- function(...) column(width = 3,  ...)
col_4  <- function(...) column(width = 4,  ...)
col_5  <- function(...) column(width = 5,  ...)
col_6  <- function(...) column(width = 6,  ...)
col_7  <- function(...) column(width = 7,  ...)
col_8  <- function(...) column(width = 8,  ...)
col_9  <- function(...) column(width = 9,  ...)
col_10 <- function(...) column(width = 10, ...)
col_12 <- function(...) column(width = 12, ...)

# Konsistente Farben für das Dashboard (analog zu den qmd-Plots)
COLORS <- list(
  primary   = "#0d6efd",
  secondary = "#6c757d",
  success   = "#198754",
  warning   = "#fd7e14",
  danger    = "#dc3545",
  info      = "#0dcaf0",
  short     = "#0d6efd",   # kurze Lebensdauer (Hauptgruppe)
  long      = "#fd7e14"    # lange Lebensdauer (Ausreißer-Gruppe)
)

# Konsistentes ggplot-Theme
theme_pdm <- function() {
  theme_minimal(base_size = 12) +
    theme(
      plot.title    = element_text(face = "bold", size = 13),
      plot.subtitle = element_text(color = "grey40", size = 10),
      panel.grid.minor = element_blank(),
      legend.position = "bottom"
    )
}

# Hilfsfunktion: Formatierter p-Wert
format_pvalue <- function(p) {
  if (p < 0.001) "p < 0.001"
  else if (p < 0.01) sprintf("p = %.3f", p)
  else sprintf("p = %.3f", round(p, 3))
}
