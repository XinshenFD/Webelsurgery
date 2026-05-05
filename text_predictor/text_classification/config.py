#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stub config module required for unpickling feature_engineer.pkl and
best_xgboost_model.pkl, which were serialized with this class embedded.
Must be kept identical to the original text_classification/config.py.
"""

import os


class Config:
    TRAIN_DATA_FILE = 'new_data/文本-N.xlsx'
    VAL_DATA_FILE = 'new_data/文本-外.xlsx'

    DATA_FILES = [
        'new_data/train.xlsx',
    ]

    TARGET_COLUMN = '是否需要手术'
    PRESERVE_COLUMNS = ['姓名', '身份证号', '省份']
    CATEGORICAL_FEATURES = ['是否配合检查', '脱位程度']
    NUMERICAL_FEATURES = [
        '矫正视力',
        '矫正球镜度数(D)',
        '矫正柱镜度数(D)',
        'IOLMaster-Cyl(D)'
    ]
    FEATURE_COLUMNS = [
        '是否配合',
        '脱位程度',
        '矫正视力',
        '矫正球镜度数(D)',
        '矫正柱镜度数(D)',
        'IOLMaster-Cyl(D)',
        '年龄'
    ]

    FEATURE_WEIGHT_INFO = {
        'description': 'Feature importance ranking',
        'order': [
            '是否配合',
            '脱位程度',
            '矫正视力',
            '矫正球镜度数(D)',
            '矫正柱镜度数(D)',
            'IOLMaster-Cyl(D)',
            '年龄'
        ]
    }

    LABEL_MAPPING = {'不手术': 0, '手术': 1}
    LABEL_NAMES = {0: '不手术', 1: '手术'}
    LABEL_NAMES_EN = {0: 'Non-Surgery', 1: 'Surgery'}

    FEATURE_NAME_MAPPING = {
        '是否配合': 'Cooperation',
        '脱位程度': 'Dislocation Degree',
        '矫正视力': 'Corrected Vision',
        '矫正球镜度数(D)': 'Spherical Power(D)',
        '矫正柱镜度数(D)': 'Cylindrical Power(D)',
        'IOLMaster-Cyl(D)': 'IOLMaster-Cyl(D)',
        '年龄': 'Age'
    }

    FILL_STRATEGY = {
        'numerical': 'median',
        'categorical': 'most_frequent'
    }

    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    RANDOM_STATE = 42
    USE_SEPARATE_VAL_FILE = True

    SMOTE_CONFIG = {
        'enabled': True,
        'sampling_strategy': 'auto',
        'k_neighbors': 5,
        'random_state': 42
    }

    USE_CLASS_WEIGHT = True

    XGBOOST_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'min_child_weight': 3,
        'gamma': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.05,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 1
    }

    EARLY_STOPPING_ROUNDS = 30
    ENABLE_HYPERPARAMETER_TUNING = True
    TUNING_METHOD = 'random'

    RANDOM_SEARCH_PARAMS = {
        'n_iter': 50,
        'cv': 5,
        'scoring': 'f1',
        'n_jobs': -1,
        'verbose': 2,
        'random_state': 42
    }

    PARAM_GRID = {
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.05, 0.1],
        'reg_lambda': [0.5, 1.0, 1.5]
    }

    CV_FOLDS = 5
    STRATIFIED = True
    MODEL_SAVE_DIR = 'text_classification/models'
    RESULTS_SAVE_DIR = 'text_classification/results'
    BEST_MODEL_NAME = 'best_xgboost_model.pkl'
    FEATURE_IMPORTANCE_NAME = 'feature_importance.png'

    EVALUATION_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    PLOT_CONFIG = {
        'style': 'seaborn-v0_8-darkgrid',
        'figure_size': (12, 8),
        'dpi': 150,
        'font_size': 12,
        'title_size': 16
    }

    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    FEATURE_IMPORTANCE_THRESHOLD = 0.01
    SAVE_INTERMEDIATE_RESULTS = True

    @classmethod
    def get_absolute_paths(cls, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return {
            'data_files': [os.path.join(base_dir, f) for f in cls.DATA_FILES],
            'train_data_file': os.path.join(base_dir, cls.TRAIN_DATA_FILE),
            'val_data_file': os.path.join(base_dir, cls.VAL_DATA_FILE),
            'model_save_dir': os.path.join(base_dir, cls.MODEL_SAVE_DIR),
            'results_save_dir': os.path.join(base_dir, cls.RESULTS_SAVE_DIR)
        }

    @classmethod
    def validate_config(cls):
        errors = []
        if not 0 < cls.TEST_SIZE < 1:
            errors.append("TEST_SIZE must be between 0 and 1")
        if not 0 < cls.VAL_SIZE < 1:
            errors.append("VAL_SIZE must be between 0 and 1")
        if cls.TEST_SIZE + cls.VAL_SIZE >= 1:
            errors.append("TEST_SIZE + VAL_SIZE must be less than 1")
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
        return True


config = Config()
