# image_predictor

This module contains the image-based EL severity grading code.

### Main files

- `model.py`: ResNet50-based classifier definition
- `predict.py`: single-image and folder-level prediction
- `explain.py`: interpretation script used during the explainability workflow
- `validate_external_subset.py`: helper for subset-based external validation
- `requirements.txt`: Python dependencies

### Private file required

- `weights/best_model.pth`

Without this weight file, inference cannot be executed.
