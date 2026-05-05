suppressPackageStartupMessages({
  library(xgboost)
  library(readr)
  library(ggplot2)
})

try(Sys.setlocale("LC_CTYPE", "en_US.UTF-8"), silent = TRUE)
try(Sys.setlocale("LC_CTYPE", "zh_CN.UTF-8"), silent = TRUE)

args <- commandArgs(trailingOnly = TRUE)
export_dir <- if (length(args) >= 1) args[[1]] else "."
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
feature_matrix <- data.matrix(feature_df)

shap_matrix <- predict(model, feature_matrix, predcontrib = TRUE)
shap_df <- as.data.frame(shap_matrix)

bias_name <- names(shap_df)[grepl("BIAS", names(shap_df), ignore.case = TRUE)]
if (length(bias_name) > 0) {
  shap_df[[bias_name[[1]]]] <- NULL
}

importance_df <- data.frame(
  feature = names(shap_df),
  mean_abs_shap = vapply(shap_df, function(x) mean(abs(x), na.rm = TRUE), numeric(1)),
  stringsAsFactors = FALSE
)

importance_df <- importance_df[order(importance_df$mean_abs_shap, decreasing = TRUE), ]
feature_label_map <- c(
  "脱位程度" = "Dislocation degree",
  "矫正视力" = "Corrected vision",
  "矫正球镜度数(D)" = "Spherical power (D)",
  "矫正柱镜度数(D)" = "Cylindrical power (D)",
  "IOLMaster-Cyl(D)" = "IOLMaster-Cyl (D)"
)
importance_df$feature_label <- unname(ifelse(
  importance_df$feature %in% names(feature_label_map),
  feature_label_map[importance_df$feature],
  importance_df$feature
))
importance_df$feature_label <- factor(
  importance_df$feature_label,
  levels = rev(importance_df$feature_label)
)

plot_obj <- ggplot(importance_df, aes(x = feature_label, y = mean_abs_shap)) +
  geom_col(width = 0.72, fill = "#2878B5") +
  coord_flip() +
  labs(
    title = "Text Predictor SHAP Importance",
    x = NULL,
    y = "Mean absolute SHAP value"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

write_csv(importance_df, file.path(export_dir, "shap_importance_from_r.csv"))
ggsave(
  filename = file.path(export_dir, "shap_importance_from_r.png"),
  plot = plot_obj,
  width = 8,
  height = 4.8,
  dpi = 320
)

print(plot_obj)
