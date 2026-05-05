suppressPackageStartupMessages({
  library(xgboost)
  library(readr)
  library(ggplot2)
  library(shapviz)
  library(cowplot)
})

try(Sys.setlocale("LC_CTYPE", "en_US.UTF-8"), silent = TRUE)
try(Sys.setlocale("LC_CTYPE", "zh_CN.UTF-8"), silent = TRUE)

args <- commandArgs(trailingOnly = TRUE)
export_dir <- if (length(args) >= 1) args[[1]] else "."
case_index <- if (length(args) >= 2) as.integer(args[[2]]) else 1L
font_family <- if (length(args) >= 3) args[[3]] else "serif"

export_dir <- normalizePath(export_dir, winslash = "/", mustWork = TRUE)
model_path <- file.path(export_dir, "best_xgboost_model.json")
feature_path <- file.path(export_dir, "sample_transformed_features.csv")
display_path <- file.path(export_dir, "sample_input_standardized.csv")

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

display_df <- if (file.exists(display_path)) {
  read_csv(display_path, show_col_types = FALSE)
} else {
  feature_df
}

model_features <- colnames(feature_df)
display_df <- display_df[, model_features, drop = FALSE]
if ("脱位程度" %in% colnames(display_df)) {
  dislocation_map <- c("轻" = "Mild", "中" = "Moderate", "重" = "Severe")
  raw_values <- as.character(display_df[["脱位程度"]])
  mapped <- dislocation_map[raw_values]
  raw_values[!is.na(mapped)] <- unname(mapped[!is.na(mapped)])
  display_df[["脱位程度"]] <- raw_values
}

name_map <- c(
  "脱位程度" = "EL Severity",
  "矫正视力" = "BCVA (Decimal)",
  "矫正球镜度数(D)" = "Spherical Power (D)",
  "矫正柱镜度数(D)" = "Cylindrical Power (D)",
  "IOLMaster-Cyl(D)" = "Corneal Astigmatism (D)"
)

patient_matrix <- data.matrix(feature_df)
patient_display <- as.data.frame(display_df)

colnames(patient_matrix)[match(names(name_map), colnames(patient_matrix))] <- name_map
colnames(patient_display)[match(names(name_map), colnames(patient_display))] <- name_map

plot_patient_waterfall <- function(model, patient_matrix, patient_display, patient_index = 1L) {
  shp <- shapviz(
    model,
    X_pred = patient_matrix,
    X = patient_display
  )

  p <- sv_waterfall(
    shp,
    row_id = patient_index,
    fill_colors = c("#BC3C29FF", "#0072B5FF")
  ) +
    labs(tag = NULL) +
    theme(
      axis.text = element_text(size = 10, face = "bold", family = font_family),
      axis.title = element_text(size = 12, face = "bold", family = font_family),
      text = element_text(family = font_family, face = "bold", size = 12),
      plot.title = element_text(family = font_family, face = "bold", size = 12),
      plot.subtitle = element_text(family = font_family, face = "bold", size = 10),
      axis.text.x = element_text(),
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA)
    )

  ggdraw(p) +
    draw_label(
      "Lower Risk",
      x = 0.05,
      y = 0.98,
      hjust = 0,
      vjust = 1,
      fontfamily = font_family,
      fontface = "bold",
      size = 12,
      color = "#0072B5FF"
    ) +
    draw_label(
      "Higher Risk",
      x = 0.95,
      y = 0.98,
      hjust = 1,
      vjust = 1,
      fontfamily = font_family,
      fontface = "bold",
      size = 12,
      color = "#BC3C29FF"
    )
}

pred_prob <- as.numeric(predict(model, patient_matrix[case_index, , drop = FALSE]))
plot_obj <- plot_patient_waterfall(
  model = model,
  patient_matrix = patient_matrix,
  patient_display = patient_display,
  patient_index = case_index
) +
  ggtitle(sprintf("Patient %d: XGBoost Waterfall Plot", case_index))

out_png <- file.path(export_dir, sprintf("case_%03d_waterfall_shapviz.png", case_index))
ggsave(
  filename = out_png,
  plot = plot_obj,
  width = 7,
  height = 5,
  units = "in",
  dpi = 300
)

write_csv(
  data.frame(
    case_index = case_index,
    predicted_surgery_probability = pred_prob
  ),
  file.path(export_dir, sprintf("case_%03d_waterfall_shapviz_meta.csv", case_index))
)

print(plot_obj)
