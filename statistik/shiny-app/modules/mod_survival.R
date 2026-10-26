# statistik/shiny-app/modules/mod_survival.R
# ─────────────────────────────────────────────────────────────────────────────
# Tab 4: Survival Analysis
# Drei Sub-Tabs: Kaplan-Meier | Weibull AFT | Cox PH
# Konsistent mit den qmd-Notebooks: "Letzte 20 Zyklen"-Mittelwerte als Covariaten.
# ─────────────────────────────────────────────────────────────────────────────

survivalUI <- function(id) {
  ns <- NS(id)

  tagList(
    bs4TabCard(
      width = 12,
      title = "Survival Analysis",
      status = "primary",
      solidHeader = TRUE,

      # ── Tab 1: Kaplan-Meier ─────────────────────────────────────────────────
      tabPanel(
        title = "Kaplan-Meier",

        fluidRow(
          col_3(
            tags$h5("Subgruppen"),
            selectInput(ns("km_group"), "Gruppierung:",
                        choices = c("Keine (Gesamt)"        = "none",
                                    "Schwellwert 300"       = "thresh_300",
                                    "Mediansplit"           = "median_split")),
            actionButton(ns("km_run"), "Kurve aktualisieren",
                         class = "btn-primary w-100"),
            hr(),
            uiOutput(ns("km_summary"))
          ),
          col_9(
            plotlyOutput(ns("km_plot"), height = "450px"),
            hr(),
            verbatimTextOutput(ns("km_table"))
          )
        )
      ),

      # ── Tab 2: Weibull AFT ──────────────────────────────────────────────────
      tabPanel(
        title = "Weibull AFT",

        fluidRow(
          col_3(
            tags$h5("Modell-Konfiguration"),
            tags$p(tags$small("Covariaten: Mittelwert der letzten 20 Zyklen vor Ausfall.")),
            checkboxGroupInput(ns("weibull_cov"), "Sensoren:",
                               choices = NULL, selected = NULL),
            actionButton(ns("weibull_run"), "Modell fitten",
                         class = "btn-primary w-100"),
            hr(),
            uiOutput(ns("weibull_summary"))
          ),
          col_9(
            plotlyOutput(ns("weibull_baseline"), height = "350px"),
            hr(),
            tags$h5("Time Ratios (e^γ) – Effekt der Sensoren auf die Überlebenszeit"),
            tags$p(tags$small("TR > 1: verlängert Überlebenszeit  |  TR < 1: verkürzt")),
            DTOutput(ns("weibull_table"))
          )
        )
      ),

      # ── Tab 3: Cox PH ───────────────────────────────────────────────────────
      tabPanel(
        title = "Cox PH",

        fluidRow(
          col_3(
            tags$h5("Modell-Konfiguration"),
            tags$p(tags$small("Statisches Modell (Modell A aus 05_survival_cox.qmd).")),
            checkboxGroupInput(ns("cox_cov"), "Sensoren:",
                               choices = NULL, selected = NULL),
            actionButton(ns("cox_run"), "Modell fitten",
                         class = "btn-primary w-100"),
            hr(),
            uiOutput(ns("cox_summary"))
          ),
          col_9(
            plotlyOutput(ns("cox_forest"), height = "400px"),
            hr(),
            tags$h5("Hazard Ratios"),
            DTOutput(ns("cox_table")),
            hr(),
            tags$h5("Proportional-Hazards-Annahme (Schoenfeld-Test)"),
            tags$p(tags$small("p > 0.05 bedeutet: PH-Annahme wird nicht verworfen.")),
            DTOutput(ns("cox_ph_test"))
          )
        )
      )
    )
  )
}

survivalServer <- function(id, app_data) {
  moduleServer(id, function(input, output, session) {

    # Sensor-Auswahl füllen (sinnvolle Defaults)
    observe({
      defaults <- intersect(c("sensor_2", "sensor_11", "sensor_17", "sensor_3"),
                            app_data$sensor_cols)
      updateCheckboxGroupInput(session, "weibull_cov",
                               choices  = app_data$sensor_cols,
                               selected = defaults)
      updateCheckboxGroupInput(session, "cox_cov",
                               choices  = app_data$sensor_cols,
                               selected = defaults)
    })

    # ══════════════════════════════════════════════════════════════════════════
    # ── KAPLAN-MEIER ──────────────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════

    km_data <- eventReactive(input$km_run, {
      lt <- app_data$lifetimes

      if (input$km_group == "thresh_300") {
        lt$group <- ifelse(lt$lifetime > 300, "Lang (> 300)", "Kurz (≤ 300)")
      } else if (input$km_group == "median_split") {
        med <- median(lt$lifetime)
        lt$group <- ifelse(lt$lifetime > med, "Lang", "Kurz")
      }
      lt
    }, ignoreNULL = FALSE)

    output$km_plot <- renderPlotly({
      lt <- km_data()

      if (input$km_group == "none") {
        fit <- survfit(Surv(lifetime, event) ~ 1, data = lt)
        surv_df <- data.frame(
          time = fit$time, surv = fit$surv,
          upper = fit$upper, lower = fit$lower,
          strata = "Gesamt"
        )
        colors <- c("Gesamt" = COLORS$primary)
      } else {
        fit <- survfit(Surv(lifetime, event) ~ group, data = lt)
        surv_df <- data.frame(
          time = fit$time, surv = fit$surv,
          upper = fit$upper, lower = fit$lower,
          strata = rep(gsub("group=", "", names(fit$strata)), fit$strata)
        )
        colors <- c(COLORS$short, COLORS$long)
      }

      p <- ggplot(surv_df, aes(x = time, y = surv, color = strata, fill = strata)) +
        geom_step(linewidth = 1.1) +
        geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.15, color = NA) +
        scale_color_manual(values = colors) +
        scale_fill_manual(values  = colors) +
        labs(title = "Kaplan-Meier-Überlebenskurve",
             x = "Zyklen", y = "S(t)", color = NULL, fill = NULL) +
        theme_pdm() +
        ylim(0, 1)

      ggplotly(p) %>% config(displayModeBar = FALSE)
    })

    output$km_summary <- renderUI({
      lt <- km_data()
      fit <- survfit(Surv(lifetime, event) ~ 1, data = lt)
      med <- summary(fit)$table["median"]

      tagList(
        tags$h6("Kennzahlen"),
        tags$table(class = "table table-sm",
          tags$tr(tags$td("n Engines:"), tags$td(nrow(lt))),
          tags$tr(tags$td("Median (Gesamt):"), tags$td(paste(med, "Zyklen"))),
          tags$tr(tags$td("Min:"), tags$td(min(lt$lifetime))),
          tags$tr(tags$td("Max:"), tags$td(max(lt$lifetime)))
        )
      )
    })

    output$km_table <- renderPrint({
      lt <- km_data()
      if (input$km_group == "none") {
        fit <- survfit(Surv(lifetime, event) ~ 1, data = lt)
      } else {
        fit <- survfit(Surv(lifetime, event) ~ group, data = lt)
      }
      print(fit)
    })

    # ══════════════════════════════════════════════════════════════════════════
    # ── WEIBULL AFT ───────────────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════

    weibull_fit <- eventReactive(input$weibull_run, {
      req(length(input$weibull_cov) > 0)

      data_static <- build_static_covariates(app_data$train,
                                             input$weibull_cov,
                                             last_n = 20)
      fit_weibull_aft(data_static, input$weibull_cov)
    })

    output$weibull_baseline <- renderPlotly({
      req(weibull_fit())
      fit <- weibull_fit()

      # Baseline-Survival-Kurve
      t_seq <- seq(0, 500, by = 5)
      pred  <- summary(fit, t = t_seq, ci = TRUE, tidy = TRUE)

      p <- ggplot(pred, aes(x = time, y = est)) +
        geom_line(color = COLORS$primary, linewidth = 1.2) +
        geom_ribbon(aes(ymin = lcl, ymax = ucl),
                    alpha = 0.2, fill = COLORS$primary) +
        labs(title = "Weibull AFT – Überlebenswahrscheinlichkeit",
             subtitle = "Baseline-Kurve (Covariaten auf Mittelwert)",
             x = "Zyklen", y = "S(t)") +
        theme_pdm() +
        ylim(0, 1)

      ggplotly(p) %>% config(displayModeBar = FALSE)
    })

    output$weibull_table <- renderDT({
      req(weibull_fit())
      tab <- tidy_weibull(weibull_fit())

      datatable(tab,
                options = list(pageLength = 12, dom = 't'),
                rownames = FALSE) %>%
        formatStyle("Time Ratio",
                    backgroundColor = styleInterval(c(0.95, 1.05),
                                                    c("#ffcdd2", "#fff9c4", "#c8e6c9")))
    })

    output$weibull_summary <- renderUI({
      req(weibull_fit())
      fit <- weibull_fit()
      aic_val <- AIC(fit)

      tagList(
        tags$h6("Modell-Statistik"),
        tags$table(class = "table table-sm",
          tags$tr(tags$td("AIC:"), tags$td(round(aic_val, 2))),
          tags$tr(tags$td("LogLik:"), tags$td(round(logLik(fit), 2))),
          tags$tr(tags$td("Shape (β):"), tags$td(round(fit$res["shape", "est"], 3))),
          tags$tr(tags$td("Scale (η):"), tags$td(round(fit$res["scale", "est"], 1))),
          tags$tr(tags$td("Covariaten:"), tags$td(length(input$weibull_cov)))
        )
      )
    })

    # ══════════════════════════════════════════════════════════════════════════
    # ── COX PH ────────────────────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════

    cox_fit <- eventReactive(input$cox_run, {
      req(length(input$cox_cov) > 0)
      data_static <- build_static_covariates(app_data$train,
                                             input$cox_cov,
                                             last_n = 20)
      fit_cox_ph(data_static, input$cox_cov)
    })

    output$cox_forest <- renderPlotly({
      req(cox_fit())
      tab <- tidy_cox(cox_fit())

      p <- ggplot(tab, aes(x = `Hazard Ratio`, y = Sensor,
                            xmin = `CI 2.5%`, xmax = `CI 97.5%`)) +
        geom_vline(xintercept = 1, linetype = "dashed", color = "grey50") +
        geom_pointrange(color = COLORS$danger, size = 0.6) +
        scale_x_log10() +
        labs(title = "Hazard Ratios (Forest Plot)",
             subtitle = "Log-Skala; HR > 1 = höheres Ausfallrisiko",
             x = "Hazard Ratio (95%-KI)", y = NULL) +
        theme_pdm()

      ggplotly(p) %>% config(displayModeBar = FALSE)
    })

    output$cox_table <- renderDT({
      req(cox_fit())
      datatable(tidy_cox(cox_fit()),
                options = list(pageLength = 12, dom = 't'),
                rownames = FALSE) %>%
        formatStyle("p-Wert",
                    backgroundColor = styleInterval(c(0.01, 0.05),
                                                    c("#c8e6c9", "#fff9c4", "#ffcdd2")))
    })

    output$cox_ph_test <- renderDT({
      req(cox_fit())
      datatable(test_ph_assumption(cox_fit()),
                options = list(pageLength = 12, dom = 't'),
                rownames = FALSE) %>%
        formatStyle("p",
                    backgroundColor = styleInterval(c(0.05),
                                                    c("#ffcdd2", "#c8e6c9")))
    })

    output$cox_summary <- renderUI({
      req(cox_fit())
      fit <- cox_fit()
      c_index <- summary(fit)$concordance[1]

      tagList(
        tags$h6("Modell-Statistik"),
        tags$table(class = "table table-sm",
          tags$tr(tags$td("C-Index:"), tags$td(round(c_index, 3))),
          tags$tr(tags$td("n Events:"), tags$td(fit$nevent)),
          tags$tr(tags$td("LogLik:"), tags$td(round(fit$loglik[2], 2))),
          tags$tr(tags$td("Covariaten:"), tags$td(length(input$cox_cov)))
        ),
        tags$p(tags$small(tags$em("Vergleich aus der Cox-Analyse: C = 0.756 (statisch), ",
                                   "0.986 (zeitabhängig)")))
      )
    })
  })
}
