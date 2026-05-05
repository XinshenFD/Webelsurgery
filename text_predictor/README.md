# text_predictor

This module contains the clinical-variable-based surgical recommendation code.

### Main files

- `predict.py`: batch and single-case prediction
- `predict_external_validation.R`: external validation workflow in R
- `export_for_r.py`: export helper for R-side analysis
- `plot_shap_importance.R`: global SHAP importance plotting
- `plot_case_waterfall*.R`: waterfall plotting helpers
- `text_classification/`: configuration utilities
- `requirements.txt`: Python dependencies

### Private files required

- `weights/best_xgboost_model.pkl`
- `weights/feature_engineer.pkl`

Without these private artifacts, the released code cannot perform final prediction.

### Input requirement

The final model expects:

- EL severity
- BCVA in decimal acuity
- spherical power
- cylindrical power
- corneal astigmatism

If BCVA is originally recorded in logMAR, it should be converted to decimal acuity before final inference.
