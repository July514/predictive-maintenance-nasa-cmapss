# statistik/shiny-app/modules/mod_inference.R
# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: Inferenz-Statistik (Hypothesentests)
# - Welch t-Test / Mann-Whitney U für Sensorwerte Early vs Late
# - Log-Rank-Test für Subgruppenvergleich (Lebensdauer-Gruppen)
# - Shapiro-Wilk Normalverteilungstest
# ─────────────────────────────────────────────────────────────────────────────

inferenceUI <- function(id) {
  ns <- NS(id)

  tagList(
    bs4TabCard(
      width = 12,
      title = "Inferenz-Statistik",
      status = "primary",
      solidHeader = TRUE,

      # ── Tab A: Mittelwert-Vergleich Early vs Late ──────────────────────────
      tabPanel(
        title = "Sensor-Vergleich (Early vs Late)",

        fluidRow(
          col_3(
            tags$h5("Hypothese"),
            tags$p(tags$em("H₀: Sensorwerte sind in Early- und Late-Phase gleich."),
                   tags$br(),
                   tags$em("H₁: Sensorwerte unterscheiden sich.")),
            hr(),
            selectInput(ns("test_sensor"), "Sensor:", choices = NULL),
            selectInput(ns("test_type"), "Test:",
                        choices = c("Welch t-Test (parametrisch)" = "ttest",
                                    "Mann-Whitney U (non-parametrisch)" = "wilcox")),
            numericInput(ns("rul_cutoff"), "RUL-Grenze Early/Late:",
                         value = 30, min = 5, max = 125, step = 5),
            actionButton(ns("run_test"), "Test ausführen",
                         class = "btn-primary w-100")
          ),
          col_9(
            bs4Card(
              title = "Testergebnis",
              status = "success",
              width = 12,
              collapsible = FALSE,
              verbatimTextOutput(ns("test_result"))
            ),
            plotlyOutput(ns("boxplot"), height = "400px")
          )
        )
      ),

      # ── Tab B: Log-Rank-Test (Survival-Subgruppen) ─────────────────────────
      tabPanel(
        title = "Log-Rank-Test",

        fluidRow(
          col_3(
            tags$h5("Hypothese"),
            tags$p(tags$em("H₀: Überlebenskurven der Subgruppen sind gleich."),
                   tags$br(),
                   tags$em("H₁: Überlebenskurven unterscheiden sich.")),
            hr(),
            selectInput(ns("logrank_split"), "Gruppen-Definition:",
                        choices = c("Schwellwert 300 Zyklen (Fault Modes)" = "thresh_300",
                                    "Mediansplit" = "median_split")),
            actionButton(ns("run_logrank"), "Test ausführen",
                         class = "btn-primary w-100"),
            hr(),
            bs4Callout(
              title = "Hintergrund",
              status = "info",
              width = 12,
              tags$small(
                "Aus der EDA-Phase: Die Lebensdauerverteilung ist bimodal. ",
                "Der Schwellwert 300 trennt die zwei Fault-Modes (HPC vs Fan). ",
                "Der Log-Rank-Test ist die statistische Bestätigung dieser ",
                "visuellen Beobachtung."
              )
            )
          ),
          col_9(
            bs4Card(
              title = "Testergebnis",
              status = "success",
              width = 12,
              verbatimTextOutput(ns("logrank_result"))
            ),
            plotlyOutput(ns("logrank_plot"), height = "400px")
          )
        )
      ),

      # ── Tab C: Normalverteilungstest ───────────────────────────────────────
      tabPanel(
        title = "Verteilungstest",

        fluidRow(
          col_3(
            tags$h5("Shapiro-Wilk-Test"),
            tags$p(tags$em("H₀: Werte sind normalverteilt."),
                   tags$br(),
                   tags$em("H₁: Werte sind nicht normalverteilt.")),
            hr(),
            selectInput(ns("dist_var"), "Variable:",
                        choices = c("Lebensdauer (Engines)" = "lifetime",
                                    "RUL (alle Zyklen)" = "rul")),
            actionButton(ns("run_dist"), "Test ausführen",
                         class = "btn-primary w-100"),
            hr(),
            bs4Callout(
              title = "Anwendung",
              status = "warning",
              width = 12,
              tags$small(
                "Normalverteilungstests sind die Voraussetzung für parametrische ",
                "Verfahren. Aus der EDA wissen wir bereits: Die Lebensdauer ist ",
                "bimodal verteilt (zwei Fault Modes) – Normalverteilung wird ",
                "deshalb klar verworfen. Weibull ist die bessere Wahl."
              )
            )
          ),
          col_9(
            bs4Card(
              title = "Testergebnis",
              status = "success",
              width = 12,
              verbatimTextOutput(ns("dist_result"))
            ),
            fluidRow(
              col_6(plotlyOutput(ns("hist_plot"), height = "350px")),
              col_6(plotlyOutput(ns("qq_plot"),   height = "350px"))
            )
          )
        )
      )
    )
  )
}

inferenceServer <- function(id, app_data) {
  moduleServer(id, function(input, output, session) {

    observe({
      updateSelectInput(session, "test_sensor",
                        choices = app_data$sensor_cols,
                        selected = "sensor_11")
    })

    # ── Tab A: Sensor-Vergleich Early vs Late ─────────────────────────────────
    test_data <- eventReactive(input$run_test, {
      req(input$test_sensor)
      app_data$train %>%
        mutate(phase = if_else(RUL <= input$rul_cutoff, "Late", "Early"))
    })

    output$test_result <- renderPrint({
      df <- test_data()
      early <- df %>% filter(phase == "Early") %>% pull(input$test_sensor)
      late  <- df %>% filter(phase == "Late")  %>% pull(input$test_sensor)

      cat(sprintf("Sensor: %s\nRUL-Grenze: %d Zyklen\n",
                  input$test_sensor, input$rul_cutoff))
      cat(sprintf("n Early: %d  |  n Late: %d\n\n", length(early), length(late)))
      cat(sprintf("Early: M = %.3f, SD = %.3f\n", mean(early), sd(early)))
      cat(sprintf("Late:  M = %.3f, SD = %.3f\n\n", mean(late), sd(late)))

      if (input$test_type == "ttest") {
        result <- t.test(early, late)
      } else {
        result <- wilcox.test(early, late)
      }
      print(result)

      cat("\n─────────────────────────────\n")
      if (result$p.value < 0.05) {
        cat("→ H₀ wird verworfen (p < 0.05).\n")
        cat("  Die Sensorwerte unterscheiden sich statistisch signifikant.\n")
      } else {
        cat("→ H₀ wird nicht verworfen (p ≥ 0.05).\n")
        cat("  Kein signifikanter Unterschied nachweisbar.\n")
      }
    })

    output$boxplot <- renderPlotly({
      df <- test_data()

      p <- ggplot(df, aes(x = phase, y = .data[[input$test_sensor]], fill = phase)) +
        geom_boxplot(alpha = 0.75, outlier.size = 0.5) +
        scale_fill_manual(values = c("Early" = COLORS$primary,
                                      "Late"  = COLORS$danger)) +
        labs(
          title = sprintf("%s: Early vs Late Phase (Cutoff RUL = %d)",
                          input$test_sensor, input$rul_cutoff),
          x = "Phase", y = input$test_sensor
        ) +
        theme_pdm() +
        theme(legend.position = "none")

      ggplotly(p) %>% config(displayModeBar = FALSE)
    })

    # ── Tab B: Log-Rank-Test ──────────────────────────────────────────────────
    logrank_data <- eventReactive(input$run_logrank, {
      lt <- app_data$lifetimes
      if (input$logrank_split == "thresh_300") {
        lt$group <- ifelse(lt$lifetime > 300, "Lang (> 300)", "Kurz (≤ 300)")
      } else {
        med <- median(lt$lifetime)
        lt$group <- ifelse(lt$lifetime > med,
                           sprintf("Lang (> %g)", med),
                           sprintf("Kurz (≤ %g)", med))
      }
      lt
    })

    output$logrank_result <- renderPrint({
      lt <- logrank_data()
      result <- survdiff(Surv(lifetime, event) ~ group, data = lt)
      p_value <- 1 - pchisq(result$chisq, df = length(result$n) - 1)

      cat("Log-Rank-Test (survdiff)\n")
      cat("─────────────────────────\n")
      print(result)
      cat(sprintf("\np-Wert: %s\n", format_pvalue(p_value)))
      cat("\n─────────────────────────\n")
      if (p_value < 0.05) {
        cat("→ H₀ wird verworfen.\n")
        cat("  Die Überlebenskurven der Gruppen unterscheiden sich signifikant.\n")
        cat("  Statistischer Nachweis der zwei verschiedenen Fault Modes.\n")
      } else {
        cat("→ H₀ wird nicht verworfen.\n")
        cat("  Kein signifikanter Unterschied der Überlebenskurven.\n")
      }
    })

    output$logrank_plot <- renderPlotly({
      lt <- logrank_data()
      fit <- survfit(Surv(lifetime, event) ~ group, data = lt)

      # Manuell zu ggplot (ggsurvplot → plotly ist fummelig)
      surv_df <- data.frame(
        time     = fit$time,
        surv     = fit$surv,
        upper    = fit$upper,
        lower    = fit$lower,
        strata   = rep(names(fit$strata), fit$strata)
      )

      p <- ggplot(surv_df, aes(x = time, y = surv, color = strata, fill = strata)) +
        geom_step(linewidth = 1) +
        geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.15, color = NA) +
        scale_color_manual(values = c(COLORS$short, COLORS$long)) +
        scale_fill_manual(values  = c(COLORS$short, COLORS$long)) +
        labs(
          title = "Kaplan-Meier-Kurven nach Subgruppe",
          x = "Zyklen", y = "Überlebenswahrscheinlichkeit S(t)",
          color = NULL, fill = NULL
        ) +
        theme_pdm()

      ggplotly(p) %>% config(displayModeBar = FALSE)
    })

    # ── Tab C: Verteilungstest ────────────────────────────────────────────────
    dist_data <- eventReactive(input$run_dist, {
      if (input$dist_var == "lifetime") {
        list(values = app_data$lifetimes$lifetime, label = "Engine-Lebensdauer (Zyklen)")
      } else {
        list(values = sample(app_data$train$RUL, 5000), label = "RUL (Sample n=5000)")
      }
    })

    output$dist_result <- renderPrint({
      d <- dist_data()
      vals <- d$values
      n_for_test <- min(length(vals), 5000)  # Shapiro-Wilk Limit
      vals_test  <- if (length(vals) > 5000) sample(vals, 5000) else vals

      cat(sprintf("Variable: %s\n", d$label))
      cat(sprintf("n = %d\n\n", length(vals)))

      sw <- shapiro.test(vals_test)
      print(sw)

      cat("\nDeskriptive Statistik:\n")
      cat(sprintf("Mittelwert: %.2f\n", mean(vals)))
      cat(sprintf("Median:     %.2f\n", median(vals)))
      cat(sprintf("SD:         %.2f\n", sd(vals)))
      cat(sprintf("Schiefe:    %.3f\n", mean((vals - mean(vals))^3) / sd(vals)^3))

      cat("\n─────────────────────────\n")
      if (sw$p.value < 0.05) {
        cat("→ H₀ wird verworfen (p < 0.05).\n")
        cat("  Die Werte sind nicht normalverteilt.\n")
      } else {
        cat("→ H₀ wird nicht verworfen.\n")
        cat("  Normalverteilung kann nicht abgelehnt werden.\n")
      }
    })

    output$hist_plot <- renderPlotly({
      req(dist_data())
      d <- dist_data()
      df <- data.frame(value = d$values)

      p <- ggplot(df, aes(x = value)) +
        geom_histogram(bins = 30, fill = COLORS$primary, alpha = 0.8, color = "white") +
        labs(title = "Histogramm", x = d$label, y = "Häufigkeit") +
        theme_pdm()

      ggplotly(p) %>% config(displayModeBar = FALSE)
    })

    output$qq_plot <- renderPlotly({
      req(dist_data())
      d  <- dist_data()
      vals <- d$values

      # Theoretische vs. empirische Quantile manuell berechnen
      # (umgeht den stat_qq-Bug in ggplot2 4.0+)
      n     <- length(vals)
      probs <- ppoints(n)
      df_qq <- data.frame(
        theoretical = qnorm(probs),
        sample      = sort(vals)
      )

      p <- ggplot(df_qq, aes(x = theoretical, y = sample)) +
        geom_point(color = COLORS$primary, alpha = 0.5, size = 1) +
        geom_smooth(method = "lm", se = FALSE,
                    color = COLORS$danger, linewidth = 1) +
        labs(title = "Q-Q-Plot (vs. Normalverteilung)",
            x = "Theoretische Quantile",
            y = "Stichproben-Quantile") +
        theme_pdm()

      ggplotly(p) %>% config(displayModeBar = FALSE)
    })
  })
}
