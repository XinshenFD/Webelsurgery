# WebELSurgery / CEL-GRADE

This repository contains the sanitized modeling code for the CEL-GRADE project, an interpretable artificial intelligence framework for:

- image-based severity grading of congenital ectopia lentis (EL)
- clinical decision support for surgical recommendation

The public CEL-GRADE platform is available at:

- [https://fudanlensdisease.cn/webelsurgery/](https://fudanlensdisease.cn/webelsurgery/)

## Repository Structure

```text
Webelsurgery/
├── image_predictor/
│   ├── model.py
│   ├── predict.py
│   ├── explain.py
│   ├── validate_external_subset.py
│   ├── requirements.txt
│   └── weights/
├── text_predictor/
│   ├── predict.py
│   ├── predict_external_validation.R
│   ├── export_for_r.py
│   ├── plot_shap_importance.R
│   ├── plot_case_waterfall.R
│   ├── plot_case_waterfall_from_raw.R
│   ├── plot_case_waterfall_shapviz.R
│   ├── requirements.txt
│   ├── text_classification/
│   └── weights/
└── README.md
```

## Data Availability

Code is available in this repository. The underlying clinical data and images are not publicly available due to privacy and ethics restrictions.

## Required Private Files

The following files are required for actual inference:

- `image_predictor/weights/best_model.pth`
- `text_predictor/weights/best_xgboost_model.pkl`
- `text_predictor/weights/feature_engineer.pkl`

## Input Description

### `image_predictor`

- Input: RGB anterior segment image files
- Output: EL severity class probabilities and predicted label

### `text_predictor`

Required fields:

- `脱位程度`
- `矫正视力` (`BCVA`, decimal acuity)
- `矫正球镜度数(D)`
- `矫正柱镜度数(D)`
- `IOLMaster-Cyl(D)`

## Note

If the original BCVA is recorded in logMAR, it should be converted to decimal acuity before being fed into the final text model workflow.
