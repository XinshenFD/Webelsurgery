suppressPackageStartupMessages({
  library(xgboost)
  library(readr)
  library(ggplot2)
  library(shapviz)
  library(cowplot)
  library(jsonlite)
})

try(Sys.setlocale("LC_CTYPE", "en_US.UTF-8"), silent = TRUE)
try(Sys.setlocale("LC_CTYPE", "zh_CN.UTF-8"), silent = TRUE)

args <- commandArgs(trailingOnly = TRUE)
export_dir <- if (length(args) >= 1) args[[1]] else "."
raw_input_path <- if (length(args) >= 2) args[[2]] else file.path(export_dir, "sample_input_standardized.csv")
case_index <- if (length(args) >= 3) as.integer(args[[3]]) else 1L
font_family <- if (length(args) >= 4) args[[4]] else "serif"

export_dir <- normalizePath(export_dir, winslash = "/", mustWork = TRUE)
raw_input_path <- normalizePath(raw_input_path, winslash = "/", mustWork = TRUE)

model_path <- file.path(export_dir, "best_xgboost_model.json")
metadata_path <- file.path(export_dir, "export_metadata.json")

if (!file.exists(model_path)) {
  stop("Missing model JSON: ", model_path)
}
if (!file.exists(metadata_path)) {
  stop("Missing export metadata JSON: ", metadata_path)
}

read_input_table <- function(path) {
  ext <- tolower(tools::file_ext(path))
  if (ext %in% c("csv", "txt")) {
    return(read_csv(path, show_col_types = FALSE))
  }
  if (ext %in% c("xlsx", "xls")) {
    if (!requireNamespace("readxl", quietly = TRUE)) {
      stop("readxl is required to read Excel input files.")
    }
    return(readxl::read_excel(path))
  }
  stop("Unsupported input format: ", ext)
}

metadata <- fromJSON(metadata_path, simplifyVector = TRUE)
prep <- metadata$preprocessing
model <- xgb.load(model_path)
raw_df <- as.data.frame(read_input_table(raw_input_path))

apply_aliases <- function(df, aliases) {
  out <- df
  for (old_name in names(aliases)) {
    new_name <- aliases[[old_name]]
    if (old_name %in% names(out) && !(new_name %in% names(out))) {
      names(out)[names(out) == old_name] <- new_name
    }
  }
  out
}

transform_raw_input <- function(df, prep) {
  out <- apply_aliases(df, prep$column_aliases)

  feature_names <- unlist(prep$feature_names, use.names = FALSE)
  numeric_cols <- unlist(prep$numerical_features, use.names = FALSE)
  categorical_cols <- unlist(prep$categorical_features, use.names = FALSE)

  missing_cols <- setdiff(c(numeric_cols, categorical_cols), names(out))
  if (length(missing_cols) > 0) {
    stop("Input data is missing required columns: ", paste(missing_cols, collapse = ", "))
  }

  display_df <- out[, feature_names, drop = FALSE]

  for (col in numeric_cols) {
    values <- suppressWarnings(as.numeric(out[[col]]))
    fill_value <- prep$numerical_imputer_statistics[[col]]
    values[is.na(values)] <- fill_value
    mean_value <- prep$scaler[[col]]$mean
    scale_value <- prep$scaler[[col]]$scale
    out[[col]] <- (values - mean_value) / scale_value
    display_df[[col]] <- values
  }

  for (col in categorical_cols) {
    values <- as.character(out[[col]])
    values[is.na(values) | values == ""] <- prep$categorical_imputer_statistics[[col]]

    classes <- unlist(prep$label_encoders[[col]]$classes, use.names = FALSE)
    mapping <- prep$label_encoders[[col]]$mapping

    fallback <- classes[[1]]
    values[!(values %in% classes)] <- fallback

    encoded <- unname(as.numeric(mapping[values]))
    out[[col]] <- encoded
    display_df[[col]] <- values
  }

  list(
    transformed = out[, feature_names, drop = FALSE],
    display = display_df
  )
}

rename_for_plot <- function(df) {
  name_map <- c(
    "脱位程度" = "EL Severity",
    "矫正视力" = "BCVA (Decimal)",
    "矫正球镜度数(D)" = "Spherical Power (D)",
    "矫正柱镜度数(D)" = "Cylindrical Power (D)",
    "IOLMaster-Cyl(D)" = "Corneal Astigmatism (D)"
  )

  out <- df
  if ("脱位程度" %in% names(out)) {
    dislocation_map <- c("轻" = "Mild", "中" = "Moderate", "重" = "Severe")
    raw_values <- as.character(out[["脱位程度"]])
    mapped <- dislocation_map[raw_values]
    raw_values[!is.na(mapped)] <- unname(mapped[!is.na(mapped)])
    out[["脱位程度"]] <- raw_values
  }

  idx <- match(names(name_map), names(out))
  names(out)[idx[!is.na(idx)]] <- unname(name_map[!is.na(idx)])
  out
}

plot_patient_waterfall <- function(model, patient_matrix, patient_display, patient_index, font_family) {
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

transformed <- transform_raw_input(raw_df, prep)
patient_display <- rename_for_plot(as.data.frame(transformed$display))
patient_matrix_df <- rename_for_plot(as.data.frame(transformed$transformed))
patient_matrix <- data.matrix(patient_matrix_df)

if (case_index < 1 || case_index > nrow(patient_matrix)) {
  stop("case_index is out of range. Valid range: 1 to ", nrow(patient_matrix))
}

pred_prob <- as.numeric(predict(model, patient_matrix[case_index, , drop = FALSE]))
plot_obj <- plot_patient_waterfall(
  model = model,
  patient_matrix = patient_matrix,
  patient_display = patient_display,
  patient_index = case_index,
  font_family = font_family
) +
  ggtitle(sprintf("Patient %d: XGBoost Waterfall Plot", case_index))

out_png <- file.path(export_dir, sprintf("case_%03d_waterfall_from_raw.png", case_index))
out_csv <- file.path(export_dir, sprintf("case_%03d_transformed_from_raw.csv", case_index))
out_meta <- file.path(export_dir, sprintf("case_%03d_waterfall_from_raw_meta.csv", case_index))

write_csv(
  transformed$transformed[case_index, , drop = FALSE],
  out_csv
)
write_csv(
  data.frame(
    case_index = case_index,
    predicted_surgery_probability = pred_prob
  ),
  out_meta
)

ggsave(
  filename = out_png,
  plot = plot_obj,
  width = 7,
  height = 5,
  units = "in",
  dpi = 300
)

print(plot_obj)
