# ================================================================
# VALIDATING DiD FOR CHATGPT EFFECT – CORRECTED (base R + ggplot2 only)
# ================================================================

library(ggplot2)

# ---------- 1. Load and prepare data ----------
df <- read.csv("so_panel.csv", stringsAsFactors = FALSE)
df$date <- as.Date(df$date)

# True intervention: November 2022
true_start <- as.Date("2022-11-01")
df$post_true <- ifelse(df$date >= true_start, 1, 0)
df$did_true <- df$is_treatment * df$post_true

# ---------- 2. Standard DiD ----------
did_model <- lm(question_count ~ did_true + factor(tag) + factor(time_index), data = df)
cat("\n========== STANDARD DiD (question_count) ==========\n")
print(summary(did_model)$coefficients["did_true", ])

# ---------- 3. EVENT STUDY (parallel trends visual) ----------
df$time_rel <- df$time_index - min(df$time_index[df$post_true == 1])
df$rel_time_f <- factor(df$time_rel)

# Model with leads/lags (excluding -1 as reference)
event_model <- lm(question_count ~ is_treatment * rel_time_f + factor(tag) + factor(time_index),
                  data = df)

coef_event <- summary(event_model)$coefficients
interact_rows <- grep("is_treatment:rel_time_f", rownames(coef_event))
event_res <- data.frame(
  time = as.numeric(gsub("is_treatment:rel_time_f", "", rownames(coef_event)[interact_rows])),
  est = coef_event[interact_rows, "Estimate"],
  se = coef_event[interact_rows, "Std. Error"]
)
event_res$lower <- event_res$est - 1.96 * event_res$se
event_res$upper <- event_res$est + 1.96 * event_res$se
event_res <- rbind(event_res, data.frame(time = -1, est = 0, se = 0, lower = 0, upper = 0))
event_res <- event_res[order(event_res$time), ]

# Plot event study
p_event <- ggplot(event_res, aes(x = time, y = est)) +
  geom_point() + geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  geom_hline(yintercept = 0, linetype = "dotted") +
  labs(title = "Event study: Treatment effect over time",
       x = "Time relative to Nov 2022", y = "Coefficient (treatment vs control)") +
  theme_minimal()
print(p_event)

# ---------- 4. FORMAL PRE‑TREND TEST (using only pre‑intervention data) ----------
# Keep only dates before true intervention (Nov 2022)
df_pre <- df[df$date < true_start, ]

# Choose a fake intervention date *within* the pre‑period, e.g., November 2021
fake_start <- as.Date("2021-11-01")
df_pre$post_fake <- ifelse(df_pre$date >= fake_start, 1, 0)
df_pre$did_fake <- df_pre$is_treatment * df_pre$post_fake

# Run DiD on pre‑period data only
pre_test <- lm(question_count ~ did_fake + factor(tag) + factor(time_index), data = df_pre)

cat("\n========== FORMAL PRE‑TREND TEST (fake intervention Nov 2021, pre‑period only) ==========\n")
print(summary(pre_test)$coefficients["did_fake", ])

if (summary(pre_test)$coefficients["did_fake", "Pr(>|t|)"] < 0.05) {
  cat("❌ REJECT parallel trends: Significant effect of fake intervention in pre‑period.\n")
} else {
  cat("✓ Cannot reject parallel trends (pre‑period placebo non‑significant).\n")
}

# ---------- 5. MULTIPLE PLACEBO TESTS (full data, false intervention dates) ----------
placebo_dates <- seq(as.Date("2021-01-01"), as.Date("2022-10-01"), by = "month")
placebo_results <- data.frame(date = placebo_dates, estimate = NA, pval = NA)

for (i in seq_along(placebo_dates)) {
  df_temp <- df
  df_temp$post_placebo <- ifelse(df_temp$date >= placebo_dates[i], 1, 0)
  df_temp$did_placebo <- df_temp$is_treatment * df_temp$post_placebo
  m_placebo <- lm(question_count ~ did_placebo + factor(tag) + factor(time_index), data = df_temp)
  coef_pl <- summary(m_placebo)$coefficients["did_placebo", ]
  placebo_results$estimate[i] <- coef_pl["Estimate"]
  placebo_results$pval[i] <- coef_pl["Pr(>|t|)"]
}

p_placebo <- ggplot(placebo_results, aes(x = date, y = estimate)) +
  geom_point() + geom_line() +
  geom_hline(yintercept = 0, linetype = "dotted") +
  geom_vline(xintercept = true_start, color = "red", linewidth = 1) +
  labs(title = "Placebo DiD estimates for false intervention dates",
       x = "Placebo start date", y = "DiD coefficient", caption = "Red line = true Nov 2022") +
  theme_minimal()
print(p_placebo)

# ---------- 6. DiD WITH GROUP‑SPECIFIC LINEAR TRENDS (robustness) ----------
df$time_linear <- df$time_index
did_trend <- lm(question_count ~ did_true + is_treatment * time_linear + factor(tag) + factor(time_index),
                data = df)
cat("\n========== DiD with group‑specific linear trends ==========\n")
print(summary(did_trend)$coefficients["did_true", ])

# ---------- 7. SENSITIVITY: Different outcomes ----------
outcomes <- c("question_count", "answered_rate", "accepted_answer_rate", "avg_score")
sensitivity <- data.frame(outcome = outcomes, did_estimate = NA, did_pval = NA)

for (out in outcomes) {
  form <- as.formula(paste0(out, " ~ did_true + factor(tag) + factor(time_index)"))
  m_sens <- lm(form, data = df)
  coef_sens <- summary(m_sens)$coefficients["did_true", ]
  sensitivity$did_estimate[sensitivity$outcome == out] <- coef_sens["Estimate"]
  sensitivity$did_pval[sensitivity$outcome == out] <- coef_sens["Pr(>|t|)"]
}
cat("\n========== DiD estimates for different outcomes ==========\n")
print(sensitivity)

# ---------- 8. FINAL VERDICT ----------
cat("\n========== VALIDATION SUMMARY ==========\n")

# Check pre‑period placebo test
pre_pval <- summary(pre_test)$coefficients["did_fake", "Pr(>|t|)"]
if (pre_pval < 0.05) {
  cat("❌ PARALLEL TRENDS REJECTED (pre‑period placebo test p =", pre_pval, ").\n")
} else {
  cat("✓ Parallel trends NOT rejected by pre‑period placebo test (p =", pre_pval, ").\n")
}

# Check multiple placebo dates (any significant before true intervention)
sig_placebo <- any(placebo_results$date < true_start & placebo_results$pval < 0.05)
if (sig_placebo) {
  cat("❌ PLACEBO TESTS FAILED: At least one false intervention before Nov 2022 is significant.\n")
} else {
  cat("✓ Placebo tests passed (no significant false interventions).\n")
}

cat("\nFinal recommendation:\n")
if (pre_pval < 0.05 | sig_placebo) {
  cat("⚠️ The parallel trends assumption is violated. The standard DiD coefficient cannot be interpreted as causal.\n")
  cat("   Consider using DiD with group‑specific trends, synthetic control, or other methods.\n")
  cat("   The DiD with group‑specific trends (model 6) provides a robustness check but still assumes linearity.\n")
} else {
  cat("✅ Parallel trends assumption holds. The standard DiD coefficient can be interpreted as the causal effect of ChatGPT on the outcome.\n")
}





library(ggplot2)

# ---------- Prepare data ----------
# Ensure date is Date type
df$date <- as.Date(df$date)

# Define placebo intervention: November 2021
placebo_start <- as.Date("2021-11-01")
df$post_placebo <- ifelse(df$date >= placebo_start, 1, 0)
df$did_placebo <- df$is_treatment * df$post_placebo

# ---------- Run placebo DiD regression ----------
placebo_model <- lm(question_count ~ did_placebo + factor(tag) + factor(time_index), data = df)

cat("\n========== PLACEBO TEST (November 2021 as false intervention) ==========\n")
print(summary(placebo_model)$coefficients["did_placebo", ])

# ---------- Graph: average question_count over time ----------
# Aggregate data by date and treatment group
agg_data <- aggregate(question_count ~ date + is_treatment, data = df, FUN = mean)
agg_data$group <- ifelse(agg_data$is_treatment == 1, "Treatment (AI-substitutable)", "Control (AI-insulated)")

# Plot with vertical line at placebo date (Nov 2021)
p <- ggplot(agg_data, aes(x = date, y = question_count, color = group)) +
  geom_line(linewidth = 1) +
  geom_vline(xintercept = placebo_start, linetype = "dashed", color = "red", linewidth = 1) +
  labs(title = "Placebo test: False intervention date = November 2021",
       subtitle = "Vertical red line shows the fake intervention (should have no effect if parallel trends held)",
       x = "Date", y = "Average question count",
       color = "Group") +
  theme_minimal() +
  annotate("text", x = placebo_start, y = max(agg_data$question_count), 
           label = "Placebo: Nov 2021", angle = 90, vjust = -0.5, hjust = 1, color = "red")

print(p)