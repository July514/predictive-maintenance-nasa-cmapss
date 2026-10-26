# statistik/shiny-app/R/survival_helpers.R
# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen für Survival Analysis.
# Methodisch konsistent mit den qmd-Notebooks (03_kaplan_meier, 04_weibull,
# 05_cox_ph): "Letzte-20-Zyklen"-Mittelwerte für statische Covariaten.
# ─────────────────────────────────────────────────────────────────────────────

library(survival)
library(flexsurv)
library(broom)

#' Lebensdauer pro Engine berechnen
#'
#' @param train_df Aufbereiteter Trainings-Dataframe
#' @return Dataframe mit unit_number, lifetime, event (immer 1, da bis Ausfall)
get_engine_lifetimes <- function(train_df) {
  train_df %>%
    group_by(unit_number) %>%
    summarise(
      lifetime = max(time_in_cycles),
      event    = 1L,
      .groups  = "drop"
    )
}

#' Surv-Objekt erzeugen
make_surv_object <- function(lifetimes) {
  with(lifetimes, Surv(time = lifetime, event = event))
}

#' Statische Covariaten aus letzten 20 Zyklen pro Engine
#'
#' Konsistent mit 05_survival_cox.qmd (Modell A).
#' Pro Engine wird der Mittelwert der angegebenen Sensoren über die letzten
#' 20 Zyklen vor dem Ausfall berechnet.
#'
#' @param train_df Trainings-Dataframe
#' @param covariates Sensor-Spalten
#' @return Dataframe mit unit_number + Sensor-Mittelwerten + lifetime + event
build_static_covariates <- function(train_df, covariates, last_n = 20) {
  train_df %>%
    group_by(unit_number) %>%
    arrange(time_in_cycles, .by_group = TRUE) %>%
    slice_tail(n = last_n) %>%
    summarise(across(all_of(covariates), mean, na.rm = TRUE), .groups = "drop") %>%
    left_join(get_engine_lifetimes(train_df), by = "unit_number")
}

#' Mid-Life-Snapshot-Covariaten mit Linkstrunkierung (nur RUL-Vorhersage-Tab).
#' Covariaten werden z-standardisiert (numerische Stabilität für flexsurv);
#' Center/Scale werden als Attribute angehängt für die Test-Daten.
build_snapshot_covariates <- function(train_df, covariates, seed = 42) {
  set.seed(seed)
  lt <- get_engine_lifetimes(train_df)
  snap <- train_df %>%
    left_join(lt %>% select(unit_number, lifetime), by = "unit_number") %>%
    group_by(unit_number) %>%
    filter(time_in_cycles >= 30, time_in_cycles <= lifetime - 10) %>%
    slice_sample(n = 1) %>%
    ungroup() %>%
    transmute(unit_number,
              start = time_in_cycles, stop = lifetime, event = 1,
              across(all_of(covariates)))

  ctr <- sapply(snap[covariates], mean)
  scl <- sapply(snap[covariates], sd)
  scl[scl < 1e-9] <- 1                          # Schutz gegen konstante Sensoren
  snap[covariates] <- scale(snap[covariates], center = ctr, scale = scl)
  attr(snap, "cov_center") <- ctr
  attr(snap, "cov_scale")  <- scl
  snap
}

#' Test-Covariaten mit den Trainings-Parametern standardisieren
scale_newdata <- function(new_data, train_static) {
  ctr <- attr(train_static, "cov_center")
  scl <- attr(train_static, "cov_scale")
  cols <- names(new_data)
  as.data.frame(scale(new_data, center = ctr[cols], scale = scl[cols]))
}

#' Kaplan-Meier-Fit
fit_kaplan_meier <- function(lifetimes, group_var = NULL) {
  if (is.null(group_var) || group_var == "none") {
    survfit(Surv(lifetime, event) ~ 1, data = lifetimes)
  } else {
    formula <- as.formula(paste("Surv(lifetime, event) ~", group_var))
    survfit(formula, data = lifetimes)
  }
}

#' Log-Rank-Test
logrank_test <- function(lifetimes, group_var) {
  formula <- as.formula(paste("Surv(lifetime, event) ~", group_var))
  survdiff(formula, data = lifetimes)
}


#' Weibull AFT mit Covariaten
fit_weibull_aft <- function(data, covariates = NULL) {
  if (is.null(covariates) || length(covariates) == 0) {
    formula <- Surv(lifetime, event) ~ 1
  } else {
    formula <- as.formula(paste("Surv(lifetime, event) ~",
                                paste(covariates, collapse = " + ")))
  }
  flexsurvreg(formula, data = data, dist = "weibull")
}

#' Weibull AFT mit Linkstrunkierung — mit expliziten Startwerten,
#' sonst läuft optim unter Trunkierung in ein degeneriertes Optimum (Shape << 1)
fit_weibull_aft_lt <- function(data, covariates) {
  formula <- as.formula(paste("Surv(start, stop, event) ~",
                              paste(covariates, collapse = " + ")))
  base  <- flexsurvreg(Surv(stop, event) ~ 1, data = data, dist = "weibull")
  inits <- c(base$res["shape", "est"], base$res["scale", "est"],
             rep(0, length(covariates)))
  flexsurvreg(formula, data = data, dist = "weibull",
              inits = inits, control = list(maxit = 10000))
}

#' Cox PH mit Covariaten
fit_cox_ph <- function(data, covariates) {
  formula <- as.formula(paste("Surv(lifetime, event) ~",
                              paste(covariates, collapse = " + ")))
  coxph(formula, data = data)
}
#' Cox PH mit Linkstrunkierung — NUR für den RUL-Vorhersage-Tab
fit_cox_ph_lt <- function(data, covariates) {
  formula <- as.formula(paste("Surv(start, stop, event) ~",
                              paste(covariates, collapse = " + ")))
  coxph(formula, data = data)
}

#' Proportional-Hazards-Annahme testen
test_ph_assumption <- function(cox_fit) {
  ph_test <- cox.zph(cox_fit)
  df      <- as.data.frame(ph_test$table)
  df$Variable <- rownames(df)
  df %>%
    select(Variable, chisq, df, p) %>%
    mutate(across(c(chisq, p), ~ round(.x, 4)))
}

#' Cox-Output als tidy Hazard-Ratio-Tabelle
tidy_cox <- function(cox_fit) {
  broom::tidy(cox_fit, exponentiate = TRUE, conf.int = TRUE) %>%
    select(term, estimate, conf.low, conf.high, p.value) %>%
    rename(
      Sensor         = term,
      `Hazard Ratio` = estimate,
      `CI 2.5%`      = conf.low,
      `CI 97.5%`     = conf.high,
      `p-Wert`       = p.value
    ) %>%
    mutate(across(where(is.numeric), ~ round(.x, 4)))
}

#' Weibull-Output als tidy Time-Ratio-Tabelle
tidy_weibull <- function(weibull_fit) {
  tab <- weibull_fit$res.t  # Log-Skala
  tab <- as.data.frame(tab)
  tab$Parameter <- rownames(tab)

  # Nur Covariaten (nicht shape/scale)
  cov_rows <- !tab$Parameter %in% c("shape", "scale")

  result <- tab[cov_rows, ] %>%
    mutate(
      `Time Ratio` = round(exp(est), 4),
      `CI 2.5%`    = round(exp(`L95%`), 4),
      `CI 97.5%`   = round(exp(`U95%`), 4)
    ) %>%
    select(Parameter, `Time Ratio`, `CI 2.5%`, `CI 97.5%`)

  result
}

#' RUL-Vorhersage via Weibull AFT — bedingt auf Überleben bis current_age
predict_rul_weibull <- function(weibull_fit, new_data, current_age) {
  tryCatch({
    t_seq <- seq(current_age, current_age + 600, by = 1)
    s <- summary(weibull_fit, newdata = new_data, t = t_seq, tidy = TRUE)
    s_age <- s$est[1]
    if (is.na(s_age) || s_age <= 0) return(NA_real_)
    s_cond <- s$est / s_age
    idx <- which(s_cond <= 0.5)[1]
    if (is.na(idx)) return(NA_real_)
    round(s$time[idx] - current_age, 1)
  }, error = function(e) NA_real_)
}

#' RUL-Vorhersage via Cox PH — bedingt auf Überleben bis current_age
predict_rul_cox <- function(cox_fit, new_data, current_age) {
  tryCatch({
    sf <- survfit(cox_fit, newdata = new_data)
    s_age <- approx(sf$time, sf$surv, xout = current_age,
                    method = "constant", rule = 2)$y
    if (is.na(s_age) || s_age <= 0) return(NA_real_)
    idx <- which(sf$surv <= 0.5 * s_age & sf$time > current_age)[1]
    med <- if (is.na(idx)) max(sf$time) else sf$time[idx]
    round(max(med - current_age, 0), 1)
  }, error = function(e) NA_real_)
}
