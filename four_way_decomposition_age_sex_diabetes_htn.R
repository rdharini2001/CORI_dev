# ======================================================================
# FOUR-WAY DECOMPOSITION: AGE + SEX + DIABETES + HYPERTENSION
# No assessment-center or height adjustment.
# Clean held-out D2 + D4 scores only.
# ======================================================================

required_packages <- c("data.table", "survival", "ggplot2")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0) {
  stop("Install first: ", paste(missing_packages, collapse = ", "))
}

suppressPackageStartupMessages({
  library(data.table)
  library(survival)
  library(ggplot2)
})

ROOT <- "F:/CORI_Final/CORI_JACC_Final/CORI_Submit"
INPUT_CSV <- file.path(
  ROOT,
  "outputs_refactored",
  "scores",
  "mediation_heldout_D2_D4.csv"
)
OUTPUT_DIR <- file.path(
  ROOT,
  "outputs_refactored",
  "four_way_decomposition_age_sex_diabetes_htn"
)
TABLE_DIR <- file.path(OUTPUT_DIR, "tables")
FIGURE_DIR <- file.path(OUTPUT_DIR, "figures")
dir.create(TABLE_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIGURE_DIR, recursive = TRUE, showWarnings = FALSE)

N_BOOT <- 500L       # Change to 5000L for final manuscript results.
SEED <- 20260720L
TAU <- 3650 / 365.25
MSTAR_PRIMARY <- 0
MSTAR_SENSITIVITY <- c(-1, 0, 1)
set.seed(SEED)

cat("Input:", INPUT_CSV, "\n")
cat("Output:", OUTPUT_DIR, "\n")
cat("Adjustment: age + sex + diabetes + hypertension\n")
cat("Bootstrap repetitions:", N_BOOT, "\n")

as_binary <- function(x, name) {
  if (is.logical(x)) {
    out <- as.integer(x)
  } else if (is.numeric(x) || is.integer(x)) {
    out <- as.integer(x != 0)
  } else {
    value <- tolower(trimws(as.character(x)))
    out <- ifelse(
      value %in% c("1", "true", "yes", "y", "event", "case", "present"),
      1L,
      ifelse(
        value %in% c("0", "false", "no", "n", "control", "censored", "absent"),
        0L,
        NA_integer_
      )
    )
  }
  if (anyNA(out)) stop("Could not convert ", name, " to binary.")
  out
}

zscore <- function(x) {
  x <- as.numeric(x)
  s <- sd(x, na.rm = TRUE)
  if (!is.finite(s) || s <= 0) stop("Cannot standardize score.")
  (x - mean(x, na.rm = TRUE)) / s
}

bootstrap_p <- function(x) {
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  lo <- (sum(x <= 0) + 1) / (length(x) + 1)
  hi <- (sum(x >= 0) + 1) / (length(x) + 1)
  min(1, 2 * min(lo, hi))
}

km_survival_at <- function(fit, times) {
  if (is.null(fit) || !length(fit$time)) return(rep(1, length(times)))
  index <- findInterval(as.numeric(times), fit$time)
  out <- rep(1, length(index))
  keep <- index > 0
  out[keep] <- fit$surv[index[keep]]
  pmax(out, 1e-4)
}

add_ipcw <- function(data, tau = TAU) {
  out <- data
  out$Y_10y <- as.integer(out$Y_mace == 1 & out$time_years <= tau)
  out$ipcw <- 0

  for (a in sort(unique(out$A_cancer_clean))) {
    index <- which(out$A_cancer_clean == a)
    group <- out[index, , drop = FALSE]
    censor_event <- as.integer(group$Y_mace == 0 & group$time_years < tau)
    censor_fit <- if (sum(censor_event) == 0) NULL else {
      survfit(Surv(group$time_years, censor_event) ~ 1)
    }

    event_index <- which(group$Y_10y == 1)
    complete_index <- which(group$Y_10y == 0 & group$time_years >= tau)

    if (length(event_index)) {
      out$ipcw[index[event_index]] <- 1 / km_survival_at(
        censor_fit,
        pmin(group$time_years[event_index], tau)
      )
    }
    if (length(complete_index)) {
      out$ipcw[index[complete_index]] <- 1 / km_survival_at(
        censor_fit,
        rep(tau, length(complete_index))
      )
    }
  }

  positive <- out$ipcw[is.finite(out$ipcw) & out$ipcw > 0]
  if (!length(positive)) stop("No positive IPCW weights.")
  cap <- as.numeric(quantile(positive, 0.99, names = FALSE))
  out$ipcw <- ifelse(is.finite(out$ipcw), pmin(out$ipcw, cap), 0)
  out
}

bootstrap_sample <- function(data) {
  # Preserve cancer/never-cancer sample sizes without using center.
  strata <- interaction(data$A_cancer_clean, drop = TRUE)
  index <- unlist(
    lapply(split(seq_len(nrow(data)), strata), function(i) {
      sample(i, length(i), replace = TRUE)
    }),
    use.names = FALSE
  )
  out <- data[index, , drop = FALSE]
  rownames(out) <- NULL
  out
}

if (!file.exists(INPUT_CSV)) stop("Input not found: ", INPUT_CSV)
analysis <- fread(INPUT_CSV, data.table = FALSE, showProgress = FALSE)

required <- c(
  "eid", "A_cancer", "Y_mace", "time_years",
  "age", "female", "Diabetes", "HTN",
  "CORI_z", "MMACE_equal_z", "MMACE_full_z"
)
missing <- setdiff(required, names(analysis))
if (length(missing)) stop("Missing columns: ", paste(missing, collapse = ", "))

analysis$eid <- sub("\\.0$", "", as.character(analysis$eid))
analysis$A_cancer_clean <- as_binary(analysis$A_cancer, "A_cancer")
analysis$Y_mace <- as_binary(analysis$Y_mace, "Y_mace")
analysis$female <- as_binary(analysis$female, "female")
analysis$Diabetes <- as_binary(analysis$Diabetes, "Diabetes")
analysis$HTN <- as_binary(analysis$HTN, "HTN")
analysis$time_years <- as.numeric(analysis$time_years)
analysis$age <- as.numeric(analysis$age)
analysis$CORI_clean <- zscore(analysis$CORI_z)
analysis$MMACE_equal_clean <- zscore(analysis$MMACE_equal_z)
analysis$MMACE_full_clean <- zscore(analysis$MMACE_full_z)

if (anyDuplicated(analysis$eid)) stop("Duplicate participant EIDs in D2+D4 input.")

mediator_map <- c(
  "Clean CORI" = "CORI_clean",
  "Equal-complexity MMACE" = "MMACE_equal_clean",
  "Practical/full MMACE" = "MMACE_full_clean"
)

manifest <- data.frame(
  adjustment = "age + sex + diabetes + hypertension",
  center_adjusted = FALSE,
  height_adjusted = FALSE,
  N = nrow(analysis),
  events = sum(analysis$Y_mace),
  cancer_N = sum(analysis$A_cancer_clean == 1),
  cancer_events = sum(analysis$A_cancer_clean == 1 & analysis$Y_mace == 1),
  never_cancer_N = sum(analysis$A_cancer_clean == 0),
  never_cancer_events = sum(analysis$A_cancer_clean == 0 & analysis$Y_mace == 1),
  bootstraps = N_BOOT
)
fwrite(manifest, file.path(TABLE_DIR, "Table_00_fourway_analysis_manifest.csv"))
print(manifest)

fit_fourway_once <- function(data, mediator_column, mstar = 0) {
  needed <- c(
    "Y_mace", "time_years", "A_cancer_clean",
    "age", "female", "Diabetes", "HTN", mediator_column
  )
  use <- data[complete.cases(data[, needed, drop = FALSE]), needed, drop = FALSE]
  names(use)[names(use) == mediator_column] <- "M"

  mediator_model <- lm(
    M ~ A_cancer_clean + age + female + Diabetes + HTN,
    data = use
  )
  mediator_residual <- residuals(mediator_model)

  exposure_0 <- use
  exposure_1 <- use
  exposure_0$A_cancer_clean <- 0L
  exposure_1$A_cancer_clean <- 1L
  M0 <- as.numeric(predict(mediator_model, newdata = exposure_0)) + mediator_residual
  M1 <- as.numeric(predict(mediator_model, newdata = exposure_1)) + mediator_residual

  weighted <- add_ipcw(use, TAU)
  outcome_data <- weighted[weighted$ipcw > 0, , drop = FALSE]
  outcome_model <- glm(
    Y_10y ~ A_cancer_clean * M + age + female + Diabetes + HTN,
    data = outcome_data,
    family = quasibinomial(link = "logit"),
    weights = ipcw,
    control = glm.control(maxit = 100)
  )

  predict_outcome <- function(a, m) {
    newdata <- use
    newdata$A_cancer_clean <- a
    newdata$M <- m
    as.numeric(predict(outcome_model, newdata = newdata, type = "response"))
  }

  Mstar <- rep(mstar, nrow(use))
  Y1_Mstar <- predict_outcome(1, Mstar)
  Y0_Mstar <- predict_outcome(0, Mstar)
  Y1_M0 <- predict_outcome(1, M0)
  Y0_M0 <- predict_outcome(0, M0)
  Y1_M1 <- predict_outcome(1, M1)
  Y0_M1 <- predict_outcome(0, M1)

  CDE <- mean(Y1_Mstar - Y0_Mstar)
  INTref <- mean((Y1_M0 - Y0_M0) - (Y1_Mstar - Y0_Mstar))
  INTmed <- mean((Y1_M1 - Y0_M1) - (Y1_M0 - Y0_M0))
  PIE <- mean(Y0_M1 - Y0_M0)
  TE <- mean(Y1_M1 - Y0_M0)

  mediator_summary <- coef(summary(mediator_model))
  outcome_summary <- coef(summary(outcome_model))
  interaction_term <- if ("A_cancer_clean:M" %in% rownames(outcome_summary)) {
    "A_cancer_clean:M"
  } else {
    "M:A_cancer_clean"
  }

  c(
    CDE = CDE,
    INTref = INTref,
    INTmed = INTmed,
    PIE = PIE,
    TE = TE,
    interaction_total = INTref + INTmed,
    mediation_total = PIE + INTmed,
    decomposition_error = TE - CDE - INTref - INTmed - PIE,
    a_path = mediator_summary["A_cancer_clean", "Estimate"],
    a_path_p = mediator_summary["A_cancer_clean", "Pr(>|t|)"],
    b_path_when_A0 = outcome_summary["M", "Estimate"],
    b_path_when_A0_p = outcome_summary["M", "Pr(>|t|)"],
    A_by_M_interaction = outcome_summary[interaction_term, "Estimate"],
    A_by_M_interaction_p = outcome_summary[interaction_term, "Pr(>|t|)"],
    N = nrow(use),
    outcome_N = nrow(outcome_data),
    events = sum(outcome_data$Y_10y)
  )
}

bootstrap_fourway <- function(data, mediator_column, mediator_label, mstar, repetitions) {
  point <- fit_fourway_once(data, mediator_column, mstar)
  effects <- c(
    "CDE", "INTref", "INTmed", "PIE", "TE",
    "interaction_total", "mediation_total"
  )
  draws <- matrix(NA_real_, nrow = repetitions, ncol = length(effects))
  colnames(draws) <- effects

  for (b in seq_len(repetitions)) {
    fit <- tryCatch(
      fit_fourway_once(bootstrap_sample(data), mediator_column, mstar),
      error = function(e) NULL
    )
    if (!is.null(fit)) draws[b, ] <- fit[effects]
    if (b %% 100 == 0) cat(mediator_label, ":", b, "/", repetitions, "\n")
  }

  draws <- draws[complete.cases(draws), , drop = FALSE]
  if (nrow(draws) < 100) stop("Too few successful bootstrap fits for ", mediator_label)

  summary <- data.frame(
    mediator = mediator_label,
    mediator_column = mediator_column,
    method = "ipcw_logistic",
    adjustment = "age + sex + diabetes + hypertension",
    mstar = mstar,
    component = effects,
    estimate = as.numeric(point[effects]),
    ci_low = apply(draws, 2, quantile, 0.025, na.rm = TRUE),
    ci_high = apply(draws, 2, quantile, 0.975, na.rm = TRUE),
    p_value = apply(draws, 2, bootstrap_p),
    successful_bootstraps = nrow(draws),
    stringsAsFactors = FALSE
  )
  summary$estimate_pp <- 100 * summary$estimate
  summary$ci_low_pp <- 100 * summary$ci_low
  summary$ci_high_pp <- 100 * summary$ci_high
  summary$share_of_TE <- summary$estimate / point[["TE"]]

  paths <- data.frame(
    mediator = mediator_label,
    mediator_column = mediator_column,
    adjustment = "age + sex + diabetes + hypertension",
    mstar = mstar,
    N = point[["N"]],
    outcome_N = point[["outcome_N"]],
    events = point[["events"]],
    a_path = point[["a_path"]],
    a_path_p = point[["a_path_p"]],
    b_path_when_A0 = point[["b_path_when_A0"]],
    b_path_when_A0_p = point[["b_path_when_A0_p"]],
    A_by_M_interaction = point[["A_by_M_interaction"]],
    A_by_M_interaction_p = point[["A_by_M_interaction_p"]],
    decomposition_error = point[["decomposition_error"]]
  )

  list(summary = summary, paths = paths, draws = draws)
}

# Reference-value point-estimate sensitivity.
mstar_rows <- list()
k <- 1L
for (label in names(mediator_map)) {
  column <- mediator_map[[label]]
  for (mstar in MSTAR_SENSITIVITY) {
    point <- fit_fourway_once(analysis, column, mstar)
    for (component in c(
      "CDE", "INTref", "INTmed", "PIE", "TE",
      "interaction_total", "mediation_total"
    )) {
      mstar_rows[[k]] <- data.frame(
        mediator = label,
        mstar = mstar,
        component = component,
        estimate = point[[component]],
        estimate_pp = 100 * point[[component]]
      )
      k <- k + 1L
    }
  }
}
fwrite(
  rbindlist(mstar_rows),
  file.path(TABLE_DIR, "Table_01_fourway_mstar_sensitivity_point_estimates.csv")
)

summary_list <- list()
path_list <- list()
draw_list <- list()
k <- 1L
for (label in names(mediator_map)) {
  cat("\nRunning:", label, "\n")
  result <- bootstrap_fourway(
    analysis,
    mediator_map[[label]],
    label,
    MSTAR_PRIMARY,
    N_BOOT
  )
  summary_list[[k]] <- result$summary
  path_list[[k]] <- result$paths
  draw_list[[label]] <- result$draws
  k <- k + 1L
}

primary_summary <- rbindlist(summary_list, fill = TRUE)
primary_paths <- rbindlist(path_list, fill = TRUE)
fwrite(primary_summary, file.path(TABLE_DIR, "Table_02_fourway_primary_ipcw_logistic.csv"))
fwrite(primary_paths, file.path(TABLE_DIR, "Table_03_fourway_primary_path_diagnostics.csv"))

for (label in names(draw_list)) {
  safe <- gsub("[^A-Za-z0-9]+", "_", label)
  fwrite(
    as.data.frame(draw_list[[label]]),
    file.path(TABLE_DIR, paste0("bootstrap_draws_", safe, ".csv"))
  )
}

component_labels <- c(
  CDE = "Controlled direct effect: neither mediation nor interaction",
  INTref = "Reference interaction: interaction only",
  INTmed = "Mediated interaction: mediation and interaction",
  PIE = "Pure indirect effect: mediation only",
  TE = "Total effect",
  interaction_total = "Total contribution involving interaction",
  mediation_total = "Total contribution involving mediation"
)

cori_summary <- primary_summary[
  primary_summary$mediator == "Clean CORI",
  ,
  drop = FALSE
]
cori_summary$interpretation <- unname(component_labels[cori_summary$component])
fwrite(cori_summary, file.path(TABLE_DIR, "Table_04_CORI_fourway_manuscript_summary.csv"))
print(cori_summary)

plot_data <- primary_summary[
  primary_summary$component %in% c("CDE", "INTref", "INTmed", "PIE"),
  ,
  drop = FALSE
]
plot_data$component <- factor(
  plot_data$component,
  levels = c("PIE", "INTmed", "INTref", "CDE"),
  labels = c("Mediation only", "Mediation + interaction", "Interaction only", "Neither")
)

plot <- ggplot(
  plot_data,
  aes(x = estimate_pp, y = component, xmin = ci_low_pp, xmax = ci_high_pp)
) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  geom_errorbar(width = 0.16) +
  geom_point(size = 2.5) +
  facet_wrap(~ mediator, scales = "free_x") +
  labs(
    x = "Absolute 10-year MACE risk difference, percentage points",
    y = NULL,
    title = "Four-way decomposition of cancer-associated MACE risk",
    subtitle = "Adjusted for age, sex, diabetes, and hypertension; no center adjustment"
  ) +
  theme_classic(base_size = 11)

ggsave(
  file.path(FIGURE_DIR, "Figure_fourway_decomposition_forest.png"),
  plot,
  width = 10,
  height = 6,
  dpi = 400
)

writeLines(capture.output(sessionInfo()), file.path(OUTPUT_DIR, "sessionInfo.txt"))
cat("\nComplete. Main output:\n")
cat(file.path(TABLE_DIR, "Table_04_CORI_fourway_manuscript_summary.csv"), "\n")
