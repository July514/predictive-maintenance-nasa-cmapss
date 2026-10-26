# statistik/shiny-app/modules/mod_overview.R
# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Übersicht – Datensatz-Steckbrief und Lebensdauer-Verteilung
# ─────────────────────────────────────────────────────────────────────────────

overviewUI <- function(id) {
  ns <- NS(id)

  tagList(
    # Hero-Box mit Projektbeschreibung
    fluidRow(
      col_12(
        bs4Callout(
          title = "NASA CMAPSS FD003 – Predictive Maintenance",
          status = "info",
          width  = 12,
          tags$p("Dieses Dashboard analysiert die Lebensdauer von 100 simulierten ",
                 "Turbofan-Triebwerken. Die NASA CMAPSS-Daten (Commercial Modular ",
                 "Aero-Propulsion System Simulation) sind ein Standard-Benchmark ",
                 "für Predictive Maintenance."),
          tags$p(tags$strong("Subset FD003: "),
                 "1 Operating Condition, 2 Fault Modes (HPC Degradation + Fan Degradation).")
        )
      )
    ),

    # Kennzahlen-Boxen
    fluidRow(
      bs4ValueBoxOutput(ns("n_engines"),  width = 3),
      bs4ValueBoxOutput(ns("n_cycles"),   width = 3),
      bs4ValueBoxOutput(ns("median_life"), width = 3),
      bs4ValueBoxOutput(ns("fault_modes"), width = 3)
    ),

    # Lebensdauer-Histogramm + Steckbrief
    fluidRow(
      col_8(
        bs4Card(
          title = "Verteilung der Engine-Lebensdauern",
          status = "primary",
          width = 12,
          solidHeader = TRUE,
          plotlyOutput(ns("lifetime_hist"), height = "400px"),
          tags$p(tags$em("Bimodale Verteilung: zwei klar getrennte Gruppen – ",
                         "Hinweis auf die zwei Fault Modes."),
                 style = "color: grey; margin-top: 10px;")
        )
      ),
      col_4(
        bs4Card(
          title = "Datensatz-Steckbrief",
          status = "info",
          width = 12,
          solidHeader = TRUE,
          tags$table(class = "table table-sm",
            tags$tr(tags$td(tags$strong("Engines (Training):")), tags$td(textOutput(ns("ds_n_train")))),
            tags$tr(tags$td(tags$strong("Engines (Test):")),     tags$td(textOutput(ns("ds_n_test")))),
            tags$tr(tags$td(tags$strong("Zyklen (Training):")),  tags$td(textOutput(ns("ds_cycles")))),
            tags$tr(tags$td(tags$strong("Sensoren (gesamt):")),  tags$td("21")),
            tags$tr(tags$td(tags$strong("Sensoren (nach Filter):")), tags$td(textOutput(ns("ds_n_sensors")))),
            tags$tr(tags$td(tags$strong("RUL-Cap:")),            tags$td("125 Zyklen")),
            tags$tr(tags$td(tags$strong("Operating Conditions:")), tags$td("1")),
            tags$tr(tags$td(tags$strong("Fault Modes:")),        tags$td("2 (HPC + Fan)"))
          )
        )
      )
    ),

    # Datenvorschau
    fluidRow(
      col_12(
        bs4Card(
          title = "Datenvorschau (erste 100 Zeilen)",
          status = "secondary",
          width = 12,
          collapsible = TRUE,
          collapsed = TRUE,
          DTOutput(ns("data_preview"))
        )
      )
    )
  )
}

overviewServer <- function(id, app_data) {
  moduleServer(id, function(input, output, session) {

    output$n_engines <- renderbs4ValueBox({
      bs4ValueBox(
        value = n_distinct(app_data$train$unit_number),
        subtitle = "Engines",
        icon = icon("cogs"),
        color = "primary"
      )
    })

    output$n_cycles <- renderbs4ValueBox({
      bs4ValueBox(
        value = format(nrow(app_data$train), big.mark = "."),
        subtitle = "Trainings-Zyklen",
        icon = icon("clock"),
        color = "info"
      )
    })

    output$median_life <- renderbs4ValueBox({
      bs4ValueBox(
        value = round(median(app_data$lifetimes$lifetime)),
        subtitle = "Median-Lebensdauer (Zyklen)",
        icon = icon("heartbeat"),
        color = "success"
      )
    })

    output$fault_modes <- renderbs4ValueBox({
      bs4ValueBox(
        value = "2",
        subtitle = "Fault Modes (HPC + Fan)",
        icon = icon("exclamation-triangle"),
        color = "warning"
      )
    })

    # Steckbrief-Texte
    output$ds_n_train   <- renderText(n_distinct(app_data$train$unit_number))
    output$ds_n_test    <- renderText(nrow(app_data$test))
    output$ds_cycles    <- renderText(format(nrow(app_data$train), big.mark = "."))
    output$ds_n_sensors <- renderText(length(app_data$sensor_cols))

    # Histogramm der Lebensdauern
    output$lifetime_hist <- renderPlotly({
      lt <- app_data$lifetimes
      lt$group <- ifelse(lt$lifetime > 300, "Lang (> 300 Zyklen)", "Kurz (≤ 300 Zyklen)")

      p <- ggplot(lt, aes(x = lifetime, fill = group)) +
        geom_histogram(binwidth = 20, alpha = 0.85, color = "white") +
        scale_fill_manual(values = c("Kurz (≤ 300 Zyklen)" = COLORS$short,
                                      "Lang (> 300 Zyklen)" = COLORS$long)) +
        labs(
          x = "Lebensdauer (Zyklen)",
          y = "Anzahl Engines",
          fill = NULL
        ) +
        theme_pdm()

      ggplotly(p) %>% config(displayModeBar = FALSE)
    })

    # Datenvorschau
    output$data_preview <- renderDT({
      app_data$train %>%
        select(unit_number, time_in_cycles, RUL,
               all_of(head(app_data$sensor_cols, 6))) %>%
        head(100) %>%
        datatable(
          options = list(pageLength = 10, scrollX = TRUE, dom = 'tip'),
          rownames = FALSE
        ) %>%
        formatRound(columns = head(app_data$sensor_cols, 6), digits = 2)
    })
  })
}
