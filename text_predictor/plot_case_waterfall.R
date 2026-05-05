suppressPackageStartupMessages({
  library(xgboost)
  library(readr)
  library(ggplot2)
})

try(Sys.setlocale("LC_CTYPE", "en_US.UTF-8"), silent = TRUE)
try(Sys.setlocale("LC_CTYPE", "zh_CN.UTF-8"), silent = TRUE)

args <- commandArgs(trailingOnly = TRUE)
export_dir <- if (length(args) >= 1) args[[1]] else "."
case_index <- if (length(args) >= 2) as.integer(args[[2]]) else 1L

export_dir <- normalizePath(export_dir, winslash = "/", mustWork = TRUE)
model_path <- file.path(export_dir, "best_xgboost_model.json")
feature_path <- file.path(export_dir, "sample_transformed_features.csv")

if (!file.exists(model_path)) {
  stop("Missing model JSON: ", model_path)
}
if (!file.exists(feature_path)) {
  stop("Missing transformed feature CSV: ", feature_path)
}

model <- xgb.load(model_path)
feature_df <- read_csv(feature_path, show_col_types = FALSE)

if (case_index < 1 || case_index > nrow(feature_df)) {
  stop("case_index is out of range. Valid range: 1 to ", nrow(feature_df))
}

feature_label_map <- c(
  "脱位程度" = "Dislocation degree",
  "矫正视力" = "Corrected vision",
  "矫正球镜度数(D)" = "Spherical power (D)",
  "矫正柱镜度数(D)" = "Cylindrical power (D)",
  "IOLMaster-Cyl(D)" = "IOLMaster-Cyl (D)"
)

feature_matrix <- data.matrix(feature_df)
case_matrix <- matrix(feature_matrix[case_index, ], nrow = 1)
colnames(case_matrix) <- colnames(feature_matrix)

pred_prob <- as.numeric(predict(model, case_matrix))
pred_logit <- qlogis(pred_prob)

contrib <- as.numeric(predict(model, case_matrix, predcontrib = TRUE))
contrib_names <- colnames(predict(model, case_matrix, predcontrib = TRUE))

bias_idx <- grep("BIAS", contrib_names, ignore.case = TRUE)
if (length(bias_idx) != 1) {
  stop("Unable to locate BIAS term in predcontrib output.")
}

bias_value <- contrib[bias_idx]
feature_contrib <- contrib[-bias_idx]
feature_names <- contrib_names[-bias_idx]

feature_labels <- ifelse(
  feature_names %in% names(feature_label_map),
  feature_label_map[feature_names],
  feature_names
)

wf <- data.frame(
  feature = feature_names,
  feature_label = feature_labels,
  value = feature_contrib,
  abs_value = abs(feature_contrib),
  direction = ifelse(feature_contrib >= 0, "Increase surgery probability", "Decrease surgery probability"),
  stringsAsFactors = FALSE
)

wf <- wf[order(wf$abs_value, decreasing = TRUE), ]
wf$start <- c(bias_value, bias_value + cumsum(head(wf$value, -1)))
wf$end <- wf$start + wf$value
wf$ymin <- pmin(wf$start, wf$end)
wf$ymax <- pmax(wf$start, wf$end)
wf$x <- seq_len(nrow(wf))

baseline_df <- data.frame(
  x = 0.4,
  xend = nrow(wf) + 0.6,
  y = bias_value,
  yend = bias_value
)

plot_obj <- ggplot(wf) +
  geom_segment(
    data = baseline_df,
    aes(x = x, xend = xend, y = y, yend = yend),
    inherit.aes = FALSE,
    linetype = "dashed",
    color = "grey55"
  ) +
  geom_rect(
    aes(
      xmin = x - 0.38,
      xmax = x + 0.38,
      ymin = ymin,
      ymax = ymax,
      fill = direction
    ),
    color = "white",
    linewidth = 0.4
  ) +
  geom_text(
    aes(
      x = x,
      y = ymax + 0.03 * max(abs(c(bias_value, end, pred_logit)), 1),
      label = sprintf("%+.3f", value)
    ),
    size = 3.5
  ) +
  scale_x_continuous(
    breaks = wf$x,
    labels = wf$feature_label,
    expand = expansion(mult = c(0.02, 0.02))
  ) +
  scale_fill_manual(
    values = c(
      "Increase surgery probability" = "#D55E00",
      "Decrease surgery probability" = "#0072B2"
    )
  ) +
  labs(
    title = paste0("Case ", case_index, " SHAP Waterfall"),
    subtitle = sprintf(
      "Baseline log-odds = %.3f | Final log-odds = %.3f | Predicted surgery probability = %.3f",
      bias_value, pred_logit, pred_prob
    ),
    x = NULL,
    y = "Contribution on log-odds scale",
    fill = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 20, hjust = 1),
    panel.grid.minor = element_blank(),
    legend.position = "top"
  )

out_csv <- file.path(export_dir, sprintf("case_%03d_waterfall_data.csv", case_index))
out_png <- file.path(export_dir, sprintf("case_%03d_waterfall.png", case_index))

write_csv(
  data.frame(
    feature = wf$feature,
    feature_label = wf$feature_label,
    contribution_logodds = wf$value,
    contribution_direction = wf$direction,
    predicted_probability = pred_prob,
    baseline_logodds = bias_value,
    final_logodds = pred_logit
  ),
  out_csv
)

ggsave(
  filename = out_png,
  plot = plot_obj,
  width = 9,
  height = 5.4,
  dpi = 320
)

print(plot_obj)
