# statistik/shiny-app/modules/mod_prediction.R
# ─────────────────────────────────────────────────────────────────────────────
# Tab 5: RUL-Vorhersage
# Wählt eine Test-Engine aus, fittet Weibull + Cox auf Training,
# prognostiziert RUL und vergleicht mit dem wahren Wert.
# ─────────────────────────────────────────────────────────────────────────────

predictionUI <- function(id) {
  ns <- NS(id)

  tagList(
    bs4Callout(
      title = "RUL-Vorhersage live",
      status = "info",
      width = 12,
      tags$p("Wähle eine Test-Engine, fitte Weibull-AFT und Cox-PH auf den ",
             "Trainingsdaten und vergleiche die Vorhersagen mit der wahren ",
             "verbleibenden Lebensdauer aus RUL_FD003.txt.")
    ),

    fluidRow(
      col_4(
        bs4Card(
          title = "Engine-Auswahl",
          status = "primary",
          width = 12,
          solidHeader = TRUE,

          selectInput(ns("test_engine"), "Test-Engine:",
                      choices = 1:100, selected = 1),

          tags$h6("Modell-Konfiguration"),
          tags$p(tags$small("Sensoren für Vorhersage-Modelle:")),
          checkboxGroupInput(ns("pred_cov"),
                             label = NULL,
                             choices = NULL,
                             selected = NULL),

          actionButton(ns("predict_btn"), "RUL vorhersagen",
                       class = "btn-primary w-100", icon = icon("magic")),

          hr(),

          tags$h6("Engine-Steckbrief"),
          uiOutput(ns("engine_info"))
        )
      ),

      col_8(
        bs4Card(
          title = "Vorhersageergebnis",
          status = "success",
          width = 12,
          solidHeader = TRUE,

          uiOutput(ns("prediction_boxes")),

          hr(),

          plotlyOutput(ns("prediction_plot"), height = "400px")
        )
      )
    ),

    fluidRow(
      col_12(
        bs4Card(
          title = "Sensor-Profil der gewählten Test-Engine",
          status = "info",
          width = 12,
          collapsible = TRUE,

          selectInput(ns("profile_sensor"), "Sensor:", choices = NULL),
          plotlyOutput(ns("engine_profile"), height = "350px")
        )
      )
    )
  )
}

predictionServer <- function(id, app_data) {
  moduleServer(id, function(input, output, session) {

    # Sensor-Auswahl
    observe({
      defaults <- intersect(c("sensor_2", "sensor_11", "sensor_17", "sensor_3"),
                            app_data$sensor_cols)
      updateCheckboxGroupInput(session, "pred_cov",
                               choices  = app_data$sensor_cols,
                               selected = defaults)
      updateSelectInput(session, "profile_sensor",
                        choices = app_data$sensor_cols,
                        selected = "sensor_11")
    })

    # Engine-Info anzeigen
    output$engine_info <- renderUI({
      req(input$test_engine)
      eng_id <- as.integer(input$test_engine)
      test_row <- app_data$test %>% filter(unit_number == eng_id)

      if (nrow(test_row) == 0) return(tags$p("Engine nicht gefunden."))

      tagList(
        tags$table(class = "table table-sm",
          tags$tr(tags$td("Engine-ID:"),     tags$td(eng_id)),
          tags$tr(tags$td("Aktueller Zyklus:"), tags$td(test_row$time_in_cycles[1])),
          tags$tr(tags$td("Wahre RUL:"),
                  tags$td(tags$strong(sprintf("%d Zyklen", test_row$RUL_true[1]))))
        )
      )
    })

    # Modelle fitten (cached für gewählte Covariaten)
    models <- eventReactive(input$predict_btn, {
      req(length(input$pred_cov) > 0, input$test_engine)

      # Trainingsdaten für statisches Modell
      train_static <- build_snapshot_covariates(app_data$train,
                                                input$pred_cov)

      # Test-Engine: aktuelle Sensorwerte (eine Zeile aus test.csv)
      eng_id <- as.integer(input$test_engine)
      test_row <- app_data$test %>% filter(unit_number == eng_id)

      # Modelle
      weibull_m <- fit_weibull_aft_lt(train_static, input$pred_cov)
      cox_m     <- fit_cox_ph_lt(train_static, input$pred_cov)

      # Vorhersage für die Test-Engine
      new_data <- scale_newdata(test_row[, input$pred_cov, drop = FALSE],
                                train_static)
      current_age <- test_row$time_in_cycles[1]
      true_rul    <- test_row$RUL_true[1]

      list(
        weibull = weibull_m,
        cox     = cox_m,
        new_data = new_data,
        current_age = current_age,
        true_rul = true_rul,
        pred_weibull = predict_rul_weibull(weibull_m, new_data, current_age),
        pred_cox     = predict_rul_cox(cox_m, new_data, current_age)
      )
    })

    output$prediction_boxes <- renderUI({
      req(models())
      m <- models()

      fmt <- function(x) {
        if (is.na(x)) return("n.v.")
        if (x == 0)   return("≤ 0 (überschritten)")
        sprintf("%.1f Zyklen", x)
      }
      diff_str <- function(pred, true) {
        if (is.na(pred) || pred == 0) return("–")
        sprintf("%+.1f", pred - true)
      }

      fluidRow(
        col_12(
          bs4ValueBox(
            value    = fmt(m$pred_weibull),
            subtitle = sprintf("Weibull AFT (Δ %s)", diff_str(m$pred_weibull, m$true_rul)),
            icon     = icon("chart-line"),
            color    = "primary",
            width    = 12
          )
        ),
        col_12(
          bs4ValueBox(
            value    = fmt(m$pred_cox),
            subtitle = sprintf("Cox PH (Δ %s)", diff_str(m$pred_cox, m$true_rul)),
            icon     = icon("chart-area"),
            color    = "warning",
            width    = 12
          )
        ),
        col_12(
          bs4ValueBox(
            value    = fmt(m$true_rul),
            subtitle = "Wahre RUL (Referenz)",
            icon     = icon("bullseye"),
            color    = "success",
            width    = 12
          )
        )
      )
    })

    output$prediction_plot <- renderPlotly({
      req(models())
      m <- models()

      # Überlebenskurve aus Weibull-Modell für die Test-Engine
      t_seq <- seq(m$current_age, m$current_age + 300, by = 5)
      pred  <- summary(m$weibull, newdata = m$new_data, t = t_seq,
                       ci = TRUE, tidy = TRUE)

      p <- ggplot(pred, aes(x = time, y = est)) +
        geom_line(color = COLORS$primary, linewidth = 1.2) +
        geom_ribbon(aes(ymin = lcl, ymax = ucl),
                    alpha = 0.2, fill = COLORS$primary) +
        geom_vline(xintercept = m$current_age, linetype = "dashed",
                   color = "grey50") +
        geom_vline(xintercept = m$current_age + m$true_rul,
                   linetype = "solid", color = COLORS$success, linewidth = 1) +
        geom_vline(xintercept = m$current_age + m$pred_weibull,
                   linetype = "dotted", color = COLORS$primary, linewidth = 1) +
        annotate("text", x = m$current_age, y = 0.05,
                 label = "Jetzt", hjust = -0.1, color = "grey40") +
        annotate("text", x = m$current_age + m$true_rul, y = 0.95,
                 label = "Wahre Ausfallzeit", hjust = -0.1,
                 color = COLORS$success) +
        labs(title = sprintf("Weibull-Prognose – Engine %d", as.integer(input$test_engine)),
             subtitle = "Grün = wahre Ausfallzeit, blau-gestrichelt = Median-Prognose",
             x = "Zyklus (absolut)", y = "S(t)") +
        theme_pdm() +
        ylim(0, 1)

      ggplotly(p) %>% config(displayModeBar = FALSE)
    })

    output$engine_profile <- renderPlotly({
      req(input$test_engine, input$profile_sensor)

      eng_id <- as.integer(input$test_engine)
      test_row <- app_data$test %>% filter(unit_number == eng_id)
      if (nrow(test_row) == 0) return(plotly_empty())

      sensor_val <- test_row[[input$profile_sensor]]
      current_age <- test_row$time_in_cycles[1]
      true_failure <- current_age + test_row$RUL_true[1]

      # Vergleich mit ähnlichen Training-Engines (alle Trajektorien)
      train_sample_ids <- sample(unique(app_data$train$unit_number), 20)
      train_sample <- app_data$train %>%
        filter(unit_number %in% train_sample_ids)

      p <- ggplot() +
        geom_line(data = train_sample,
                  aes(x = time_in_cycles, y = .data[[input$profile_sensor]],
                      group = unit_number),
                  color = "grey70", alpha = 0.4, linewidth = 0.4) +
        geom_point(aes(x = current_age, y = sensor_val),
                   color = COLORS$danger, size = 4) +
        geom_vline(xintercept = true_failure,
                   linetype = "solid", color = COLORS$success, linewidth = 1) +
        labs(title = sprintf("%s – Test-Engine %d (rot) vs. 20 Training-Engines (grau)",
                             input$profile_sensor, eng_id),
             subtitle = sprintf("Aktueller Zyklus: %d  |  Wahre Ausfallzeit: %d",
                                current_age, true_failure),
             x = "Zyklus", y = input$profile_sensor) +
        theme_pdm()

      ggplotly(p) %>% config(displayModeBar = FALSE)
    })
  })
}
