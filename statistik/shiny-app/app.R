# statistik/shiny-app/app.R
# ─────────────────────────────────────────────────────────────────────────────
# Predictive Maintenance Dashboard – NASA CMAPSS FD003
# Hauptdatei: lädt Helper, Module und startet die App
# ─────────────────────────────────────────────────────────────────────────────

library(shiny)
library(bs4Dash)
library(tidyverse)
library(plotly)
library(DT)
library(survival)
library(survminer)
library(flexsurv)
library(broom)

# Dark-Mode CSS-Fixes
app_css <- "
  .dataTables_wrapper { color: #ffffff !important; }
  .dataTables_wrapper table.dataTable { color: #ffffff !important; }
  .dataTables_filter label, .dataTables_length label,
  .dataTables_info, .dataTables_paginate { color: #cccccc !important; }
  table.dataTable thead th { 
    color: #ffffff !important; 
    border-bottom: 1px solid #444 !important;
  }
  table.dataTable tbody tr { background-color: #2d3035 !important; }
  table.dataTable tbody tr:hover { background-color: #3d4045 !important; }
  .verbatimTextOutput, pre { color: #f8f8f2 !important; }
"

# ── Helper-Funktionen laden ───────────────────────────────────────────────────
source("R/helpers.R")
source("R/data_loader.R")
source("R/survival_helpers.R")


# ── Module laden ──────────────────────────────────────────────────────────────
source("modules/mod_overview.R")
source("modules/mod_descriptive.R")
source("modules/mod_inference.R")
source("modules/mod_survival.R")
source("modules/mod_prediction.R")

# ── Globale Daten (einmal beim App-Start laden) ───────────────────────────────
message("Lade aufbereitete CMAPSS FD003 Daten...")
train_data   <- load_processed_train()
test_data    <- load_processed_test()
lifetimes    <- get_engine_lifetimes(train_data)
surv_obj     <- make_surv_object(lifetimes)

# Sensor-Spalten (nur rohe Sensoren, ohne Rolling/Norm)
sensor_cols  <- grep("^sensor_\\d+$", names(train_data), value = TRUE)

message(sprintf("Daten geladen: %d Engines, %d rohe Sensoren, %d Features total",
                n_distinct(train_data$unit_number),
                length(sensor_cols),
                ncol(train_data)))

# ── UI ────────────────────────────────────────────────────────────────────────
ui <- bs4DashPage(
  title = "Predictive Maintenance – NASA CMAPSS FD003",
  dark = TRUE,

  header = bs4DashNavbar(
    tags$head(tags$style(HTML(app_css))),
    title = bs4DashBrand(
      title = "PdM Dashboard · NASA FD003",
      color = "primary"
    ),
    skin = "dark"
  ),

  sidebar = bs4DashSidebar(
    skin = "dark",
    bs4SidebarMenu(
      bs4SidebarMenuItem("Übersicht",       tabName = "overview",    icon = icon("info-circle")),
      bs4SidebarMenuItem("Deskriptiv",      tabName = "descriptive", icon = icon("chart-bar")),
      bs4SidebarMenuItem("Inferenz",        tabName = "inference",   icon = icon("flask")),
      bs4SidebarMenuItem("Survival",        tabName = "survival",    icon = icon("heartbeat")),
      bs4SidebarMenuItem("RUL-Vorhersage",  tabName = "prediction",  icon = icon("clock"))
    )
  ),

  body = bs4DashBody(
    bs4TabItems(
      bs4TabItem(tabName = "overview",    overviewUI("overview")),
      bs4TabItem(tabName = "descriptive", descriptiveUI("descriptive")),
      bs4TabItem(tabName = "inference",   inferenceUI("inference")),
      bs4TabItem(tabName = "survival",    survivalUI("survival")),
      bs4TabItem(tabName = "prediction",  predictionUI("prediction"))
    )
  ),

  footer = bs4DashFooter(
    left  = "DHBW Lörrach · Statistik-Projekt",
    right = "Mai 2026"
  )
)

# ── Server ────────────────────────────────────────────────────────────────────
server <- function(input, output, session) {

  app_data <- reactiveValues(
    train       = train_data,
    test        = test_data,
    lifetimes   = lifetimes,
    surv_obj    = surv_obj,
    sensor_cols = sensor_cols
  )

  overviewServer("overview",       app_data)
  descriptiveServer("descriptive", app_data)
  inferenceServer("inference",     app_data)
  survivalServer("survival",       app_data)
  predictionServer("prediction",   app_data)
}

# ── App starten ──────────────────────────────────────────────────────────────
shinyApp(ui = ui, server = server)
