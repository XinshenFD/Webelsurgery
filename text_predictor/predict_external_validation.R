suppressPackageStartupMessages({
  library(xgboost)
  library(readr)
  library(readxl)
  library(writexl)
  library(jsonlite)
})

try(Sys.setlocale("LC_CTYPE", "en_US.UTF-8"), silent = TRUE)
try(Sys.setlocale("LC_CTYPE", "zh_CN.UTF-8"), silent = TRUE)

args <- commandArgs(trailingOnly = TRUE)
input_path <- if (length(args) >= 1) args[[1]] else "外部验证-text/data_text_external.xlsx"
export_dir <- if (length(args) >= 2) args[[2]] else "text_predictor/r_export"
output_dir <- if (length(args) >= 3) args[[3]] else "text_predictor/output/external_text_validation"
bcva_mode <- if (length(args) >= 4) args[[4]] else "raw"

input_path <- normalizePath(input_path, winslash = "/", mustWork = TRUE)
export_dir <- normalizePath(export_dir, winslash = "/", mustWork = TRUE)
output_dir <- normalizePath(output_dir, winslash = "/", mustWork = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

model_path <- file.path(export_dir, "best_xgboost_model.json")
metadata_path <- file.path(export_dir, "export_metadata.json")

if (!file.exists(model_path)) {
  stop("Missing model JSON: ", model_path)
}
if (!file.exists(metadata_path)) {
  stop("Missing export metadata JSON: ", metadata_path)
}
if (!(bcva_mode %in% c("raw", "logmar_to_decimal"))) {
  stop("bcva_mode must be one of: raw, logmar_to_decimal")
}

metadata <- fromJSON(metadata_path, simplifyVector = TRUE)
prep <- metadata$preprocessing
model <- xgb.load(model_path)

raw_df <- as.data.frame(read_excel(input_path))

external_aliases <- c(
  "球" = "矫正球镜度数(D)",
  "柱" = "矫正柱镜度数(D)",
  "BCVA" = "矫正视力",
  "角膜散光" = "IOLMaster-Cyl(D)"
)

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
  out <- apply_aliases(out, external_aliases)

  if ("矫正视力" %in% names(out) && bcva_mode == "logmar_to_decimal") {
    out[["矫正视力"]] <- 10^(-suppressWarnings(as.numeric(out[["矫正视力"]])))
  }

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

label_map_true <- c(
  "No-surgery" = 0,
  "Surgery" = 1,
  "No Surgery" = 0,
  "Non-surgery" = 0,
  "不手术" = 0,
  "手术" = 1
)

truth_column_candidates <- c(
  "True_label",
  "纳入（1 保守 2 手术）",
  "纳入",
  "真值",
  "label"
)

transformed <- transform_raw_input(raw_df, prep)
feature_matrix <- data.matrix(transformed$transformed)
pred_prob_surgery <- as.numeric(predict(model, feature_matrix))
pred_class_num <- ifelse(pred_prob_surgery >= 0.5, 1L, 0L)

true_label_raw <- if ("True_label" %in% names(raw_df)) as.character(raw_df$True_label) else rep(NA_character_, nrow(raw_df))
truth_col <- truth_column_candidates[truth_column_candidates %in% names(raw_df)]
truth_col <- if (length(truth_col) > 0) truth_col[[1]] else NA_character_
true_label_raw <- if (!is.na(truth_col)) as.character(raw_df[[truth_col]]) else rep(NA_character_, nrow(raw_df))
true_label_num <- unname(label_map_true[true_label_raw])
true_label_num[is.na(true_label_num)] <- NA_real_

result_df <- raw_df
result_df$p_no_surgery <- 1 - pred_prob_surgery
result_df$p_surgery <- pred_prob_surgery
result_df$pred_label_num <- pred_class_num
result_df$pred_label_cn <- ifelse(pred_class_num == 1, "手术", "不手术")
result_df$pred_label_en <- ifelse(pred_class_num == 1, "Surgery", "No-surgery")
result_df$true_label_num <- true_label_num
result_df$truth_source_column <- truth_col
result_df$true_label_cn <- ifelse(is.na(true_label_num), NA, ifelse(true_label_num == 1, "手术", "不手术"))
result_df$true_label_en <- ifelse(is.na(true_label_num), NA, ifelse(true_label_num == 1, "Surgery", "No-surgery"))
result_df$include_for_roc <- !is.na(true_label_num)
result_df$correct_prediction <- ifelse(is.na(true_label_num), NA, as.integer(pred_class_num == true_label_num))

result_df$y_true_no_surgery <- ifelse(is.na(true_label_num), NA, as.integer(true_label_num == 0))
result_df$y_true_surgery <- ifelse(is.na(true_label_num), NA, as.integer(true_label_num == 1))

result_df$矫正视力_model <- transformed$display$矫正视力
result_df$矫正球镜度数_model <- transformed$display[["矫正球镜度数(D)"]]
result_df$矫正柱镜度数_model <- transformed$display[["矫正柱镜度数(D)"]]
result_df$IOLMaster_Cyl_model <- transformed$display[["IOLMaster-Cyl(D)"]]
result_df$脱位程度_model <- transformed$display$脱位程度

csv_path <- file.path(output_dir, "external_text_predictions_for_roc.csv")
xlsx_path <- file.path(output_dir, "external_text_predictions_for_roc.xlsx")
summary_path <- file.path(output_dir, "external_text_predictions_summary.csv")

write_csv(result_df, csv_path)
write_xlsx(list(predictions = result_df), xlsx_path)

summary_df <- data.frame(
  metric = c(
    "input_rows",
    "roc_eligible_rows",
    "predicted_surgery_rows",
    "predicted_no_surgery_rows",
    "row_accuracy_on_labeled_rows",
    "bcva_mode"
  ),
  value = c(
    nrow(result_df),
    sum(result_df$include_for_roc, na.rm = TRUE),
    sum(result_df$pred_label_num == 1, na.rm = TRUE),
    sum(result_df$pred_label_num == 0, na.rm = TRUE),
    mean(result_df$correct_prediction[result_df$include_for_roc], na.rm = TRUE),
    bcva_mode
  )
)
write_csv(summary_df, summary_path)

cat("Saved CSV:  ", csv_path, "\n")
cat("Saved XLSX: ", xlsx_path, "\n")
cat("Saved summary: ", summary_path, "\n")
