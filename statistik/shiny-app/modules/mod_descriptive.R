# statistik/shiny-app/modules/mod_descriptive.R
# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Deskriptive Statistik
# - Engine-Verläufe (einzeln und Vergleich)
# - Sensor-Statistiken
# - Korrelationsmatrix mit RUL
# ─────────────────────────────────────────────────────────────────────────────

descriptiveUI <- function(id) {
  ns <- NS(id)

  tagList(
    # Sub-Tabs für die verschiedenen deskriptiven Analysen
    bs4TabCard(
      width = 12,
      title = "Deskriptive Statistik",
      status = "primary",
      solidHeader = TRUE,
      collapsible = FALSE,

      # ── Tab A: Sensor-Verläufe einzelner Engines ───────────────────────────
      tabPanel(
        title = "Engine-Verläufe",

        fluidRow(
          col_3(
            tags$h5("Auswahl"),
            selectInput(ns("engine_id"), "Engine:",
                        choices = 1:100, selected = 1),
            selectInput(ns("sensor_sel"), "Sensor:",
                        choices = NULL),
            checkboxInput(ns("show_rolling"), "Rolling Mean (window=5) anzeigen",
                          value = TRUE),
            hr(),
            bs4Callout(
              title = "Lesehilfe",
              status = "info",
              width = 12,
              tags$small(
                "Sensor-Werte über den Lebenszyklus einer Engine. ",
                "Die letzten Zyklen vor dem Ausfall zeigen meist klare ",
                "Degradationsmuster."
              )
            )
          ),
          col_9(
            plotlyOutput(ns("sensor_plot"), height = "450px"),
            hr(),
            tags$h5("Engine-Steckbrief"),
            uiOutput(ns("engine_facts"))
          )
        )
      ),

      # ── Tab B: Mehrere Engines vergleichen ─────────────────────────────────
      tabPanel(
        title = "Engine-Vergleich",

        fluidRow(
          col_3(
            tags$h5("Vergleich"),
            selectInput(ns("compare_sensor"), "Sensor:",
                        choices = NULL),
            sliderInput(ns("compare_n"), "Anzahl Engines (zufällig gewählt):",
                        min = 5, max = 30, value = 10, step = 5),
            actionButton(ns("compare_btn"), "Neu sampeln",
                         class = "btn-secondary w-100")
          ),
          col_9(
            plotlyOutput(ns("compare_plot"), height = "500px")
          )
        )
      ),

      # ── Tab C: Sensor-Statistiken ──────────────────────────────────────────
      tabPanel(
        title = "Sensor-Statistiken",

        fluidRow(
          col_12(
            tags$p("Zusammenfassende Statistiken aller 12 informativen Sensoren ",
                   "(nach VIF-Filter). Konstante Sensoren wurden bereits entfernt."),
            DTOutput(ns("sensor_stats_table"))
          )
        )
      ),

      # ── Tab D: Korrelationsmatrix ──────────────────────────────────────────
      tabPanel(
        title = "Korrelation mit RUL",

        fluidRow(
          col_4(
            tags$h5("Top-Prädiktoren"),
            tags$p("Sortiert nach absoluter Pearson-Korrelation mit RUL."),
            DTOutput(ns("corr_table"))
          ),
          col_8(
            plotlyOutput(ns("corr_heatmap"), height = "550px")
          )
        )
      )
    )
  )
}

descriptiveServer <- function(id, app_data) {
  moduleServer(id, function(input, output, session) {

    # Sensor-Auswahl füllen
    observe({
      updateSelectInput(session, "sensor_sel",
                        choices = app_data$sensor_cols,
                        selected = "sensor_11")
      updateSelectInput(session, "compare_sensor",
                        choices = app_data$sensor_cols,
                        selected = "sensor_11")
    })

    # ── Tab A: Engine-Verlauf ─────────────────────────────────────────────────
    output$sensor_plot <- renderPlotly({
      req(input$engine_id, input$sensor_sel)

      df <- app_data$train %>%
        filter(unit_number == as.integer(input$engine_id))

      sensor_col <- input$sensor_sel
      rolling_col <- paste0(sensor_col, "_roll_mean")

      p <- ggplot(df, aes(x = time_in_cycles)) +
        geom_line(aes(y = .data[[sensor_col]]),
                  color = COLORS$secondary, alpha = 0.5, linewidth = 0.5) +
        labs(
          title = sprintf("Engine %s – %s", input$engine_id, sensor_col),
          subtitle = sprintf("Lebensdauer: %d Zyklen",
                             max(df$time_in_cycles)),
          x = "Zyklus",
          y = sensor_col
        ) +
        theme_pdm()

      if (input$show_rolling && rolling_col %in% names(df)) {
        p <- p + geom_line(aes(y = .data[[rolling_col]]),
                           color = COLORS$primary, linewidth = 1.1)
      }

      ggplotly(p) %>% config(displayModeBar = FALSE)
    })

    output$engine_facts <- renderUI({
      req(input$engine_id)
      df <- app_data$train %>%
        filter(unit_number == as.integer(input$engine_id))

      lifetime <- max(df$time_in_cycles)
      group <- if (lifetime > 300) "Lang-Lebensdauer-Gruppe (> 300 Zyklen)" else "Kurz-Lebensdauer-Gruppe (≤ 300 Zyklen)"

      tags$div(
        class = "row",
        tags$div(class = "col-md-3", tags$strong("Lebensdauer: "), lifetime, " Zyklen"),
        tags$div(class = "col-md-3", tags$strong("Beobachtungen: "), nrow(df)),
        tags$div(class = "col-md-3", tags$strong("Min. RUL: "), min(df$RUL)),
        tags$div(class = "col-md-3", tags$strong("Gruppe: "), group)
      )
    })

    # ── Tab B: Engine-Vergleich ───────────────────────────────────────────────
    sampled_engines <- reactiveVal()

    observe({
      sampled_engines(sample(1:100, 10))
    })

    observeEvent(input$compare_btn, {
      sampled_engines(sample(1:100, input$compare_n))
    })

    output$compare_plot <- renderPlotly({
      req(input$compare_sensor)
      eng_ids <- sampled_engines()

      df <- app_data$train %>%
        filter(unit_number %in% eng_ids) %>%
        mutate(unit_number = factor(unit_number))

      p <- ggplot(df, aes(x = time_in_cycles,
                          y = .data[[input$compare_sensor]],
                          color = unit_number,
                          group = unit_number)) +
        geom_line(alpha = 0.7, linewidth = 0.6) +
        labs(
          title = sprintf("%s – Verlauf für %d Engines",
                          input$compare_sensor, length(eng_ids)),
          x = "Zyklus",
          y = input$compare_sensor,
          color = "Engine"
        ) +
        theme_pdm() +
        theme(legend.position = "right")

      ggplotly(p) %>% config(displayModeBar = FALSE)
    })

    # ── Tab C: Sensor-Statistiken ─────────────────────────────────────────────
    output$sensor_stats_table <- renderDT({
      stats <- app_data$train %>%
        select(all_of(app_data$sensor_cols)) %>%
        summarise(across(everything(), list(
          Mittelwert = ~ mean(.x, na.rm = TRUE),
          SD         = ~ sd(.x, na.rm = TRUE),
          Min        = ~ min(.x, na.rm = TRUE),
          Median     = ~ median(.x, na.rm = TRUE),
          Max        = ~ max(.x, na.rm = TRUE)
        ), .names = "{.col}__{.fn}")) %>%
        pivot_longer(everything(),
                     names_to = c("Sensor", "Statistik"),
                     names_sep = "__",
                     values_to = "Wert") %>%
        pivot_wider(names_from = Statistik, values_from = Wert)

      datatable(stats,
                options = list(pageLength = 12, dom = 'tip'),
                rownames = FALSE) %>%
        formatRound(columns = c("Mittelwert", "SD", "Min", "Median", "Max"),
                    digits = 3)
    })

    # ── Tab D: Korrelation ────────────────────────────────────────────────────
    corr_data <- reactive({
      app_data$train %>%
        select(all_of(app_data$sensor_cols), RUL) %>%
        cor(use = "complete.obs")
    })

    output$corr_table <- renderDT({
      corr_with_rul <- corr_data()[, "RUL"]
      corr_with_rul <- corr_with_rul[names(corr_with_rul) != "RUL"]

      df <- data.frame(
        Sensor      = names(corr_with_rul),
        Korrelation = round(corr_with_rul, 3),
        `|r|`       = round(abs(corr_with_rul), 3),
        check.names = FALSE
      ) %>% arrange(desc(`|r|`))

      datatable(df,
                options = list(pageLength = 12, dom = 't', order = list()),
                rownames = FALSE) %>%
        formatStyle("Korrelation",
                    backgroundColor = styleInterval(c(-0.5, 0, 0.5),
                                                    c("#ffcdd2", "#fff9c4",
                                                      "#fff9c4", "#c8e6c9")))
    })

    output$corr_heatmap <- renderPlotly({
      cm <- corr_data()

      plot_ly(
        x = colnames(cm),
        y = rownames(cm),
        z = cm,
        type = "heatmap",
        colorscale = "RdBu",
        zmid = 0,
        zmin = -1, zmax = 1,
        colorbar = list(title = "r")
      ) %>%
        layout(
          title = "Korrelationsmatrix: Sensoren × RUL",
          xaxis = list(tickangle = -45)
        )
    })
  })
}
