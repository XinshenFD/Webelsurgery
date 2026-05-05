#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export the text_predictor XGBoost model and preprocessing metadata into
R-friendly files without requiring the Python xgboost package.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import types
import warnings
from pathlib import Path
from struct import unpack
from typing import Any

import numpy as np
import pandas as pd


COLUMN_ALIASES = {
    "是否配合检查": "是否配合",
    "球镜 (D)": "矫正球镜度数(D)",
    "柱镜 (D)": "矫正柱镜度数(D)",
    "角膜散光Cyl(D)": "IOLMaster-Cyl(D)",
    "答案_脱位程度": "脱位程度",
}

LABEL_MAP = {"0": "不手术", "1": "手术"}


def _install_xgboost_pickle_stubs() -> None:
    """Create minimal stub classes so pickle can load the saved artifacts."""
    xgb = types.ModuleType("xgboost")
    sk = types.ModuleType("xgboost.sklearn")
    core = types.ModuleType("xgboost.core")
    compat = types.ModuleType("xgboost.compat")

    class Capture:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __setstate__(self, state: Any) -> None:
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self._state = state

    class XGBClassifier(Capture):
        pass

    class Booster(Capture):
        pass

    class XGBoostLabelEncoder(Capture):
        pass

    sk.XGBClassifier = XGBClassifier
    core.Booster = Booster
    compat.XGBoostLabelEncoder = XGBoostLabelEncoder
    xgb.sklearn = sk
    xgb.core = core
    xgb.compat = compat

    sys.modules["xgboost"] = xgb
    sys.modules["xgboost.sklearn"] = sk
    sys.modules["xgboost.core"] = core
    sys.modules["xgboost.compat"] = compat


def _to_python(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


class UBJSONDecoder:
    """Minimal UBJSON decoder sufficient for XGBoost save_raw(raw_format='ubj')."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def decode(self) -> Any:
        value = self._parse_value()
        if self.pos != len(self.data):
            raise ValueError(f"Trailing UBJSON bytes at position {self.pos} of {len(self.data)}")
        return value

    def _read(self, n: int) -> bytes:
        out = self.data[self.pos:self.pos + n]
        if len(out) != n:
            raise ValueError("Unexpected end of UBJSON stream")
        self.pos += n
        return out

    def _peek(self) -> str | None:
        if self.pos >= len(self.data):
            return None
        return chr(self.data[self.pos])

    def _parse_length(self) -> int:
        marker = self._read(1).decode("ascii")
        value = self._parse_numeric(marker)
        if value < 0:
            raise ValueError("Negative UBJSON length is invalid")
        return int(value)

    def _parse_numeric(self, marker: str) -> int | float:
        if marker == "i":
            return unpack(">b", self._read(1))[0]
        if marker == "U":
            return self._read(1)[0]
        if marker == "I":
            return unpack(">h", self._read(2))[0]
        if marker == "l":
            return unpack(">i", self._read(4))[0]
        if marker == "L":
            return unpack(">q", self._read(8))[0]
        if marker == "d":
            return unpack(">f", self._read(4))[0]
        if marker == "D":
            return unpack(">d", self._read(8))[0]
        raise ValueError(f"Unsupported UBJSON numeric marker: {marker}")

    def _parse_string(self) -> str:
        length = self._parse_length()
        return self._read(length).decode("utf-8")

    def _parse_key(self) -> str:
        return self._parse_string()

    def _parse_array(self) -> list[Any]:
        value_marker = None
        count = None
        if self._peek() == "$":
            self.pos += 1
            value_marker = self._read(1).decode("ascii")
        if self._peek() == "#":
            self.pos += 1
            count = self._parse_length()

        items: list[Any] = []
        if count is not None:
            for _ in range(count):
                items.append(self._parse_value(value_marker))
            if self._peek() == "]":
                self.pos += 1
            return items

        while self._peek() != "]":
            items.append(self._parse_value(value_marker))
        self.pos += 1
        return items

    def _parse_object(self) -> dict[str, Any]:
        value_marker = None
        count = None
        if self._peek() == "$":
            self.pos += 1
            value_marker = self._read(1).decode("ascii")
        if self._peek() == "#":
            self.pos += 1
            count = self._parse_length()

        obj: dict[str, Any] = {}
        if count is not None:
            for _ in range(count):
                key = self._parse_key()
                obj[key] = self._parse_value(value_marker)
            if self._peek() == "}":
                self.pos += 1
            return obj

        while self._peek() != "}":
            key = self._parse_key()
            obj[key] = self._parse_value(value_marker)
        self.pos += 1
        return obj

    def _parse_value(self, forced_marker: str | None = None) -> Any:
        marker = forced_marker or self._read(1).decode("ascii")

        if marker == "Z":
            return None
        if marker == "T":
            return True
        if marker == "F":
            return False
        if marker == "N":
            return self._parse_value(forced_marker)
        if marker in {"i", "U", "I", "l", "L", "d", "D"}:
            return self._parse_numeric(marker)
        if marker == "S":
            return self._parse_string()
        if marker == "C":
            return self._read(1).decode("utf-8")
        if marker == "[":
            return self._parse_array()
        if marker == "{":
            return self._parse_object()
        raise ValueError(f"Unsupported UBJSON marker: {marker!r} at position {self.pos}")


def load_artifacts(model_path: Path, fe_path: Path, project_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _install_xgboost_pickle_stubs()
    sys.path.insert(0, str(project_dir))

    with model_path.open("rb") as f:
        model_data = pickle.load(f)
    with fe_path.open("rb") as f:
        fe_data = pickle.load(f)
    return model_data, fe_data


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.rename(columns=COLUMN_ALIASES, inplace=True)
    return out


def preprocess_for_export(df: pd.DataFrame, fe_data: dict[str, Any]) -> pd.DataFrame:
    out = standardize_columns(df)

    feature_names = list(fe_data["feature_names"])
    numerical_imputer = fe_data["numerical_imputer"]
    scaler = fe_data["scaler"]
    categorical_imputer = fe_data["categorical_imputer"]
    label_encoders = fe_data["label_encoders"]

    numeric_cols = list(numerical_imputer.feature_names_in_)
    categorical_cols = list(label_encoders.keys())
    required_cols = numeric_cols + categorical_cols
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out[numeric_cols] = numerical_imputer.transform(out[numeric_cols])
    out[numeric_cols] = scaler.transform(out[numeric_cols])

    out[categorical_cols] = categorical_imputer.transform(out[categorical_cols])
    for col in categorical_cols:
        encoder = label_encoders[col]
        known = set(encoder.classes_)
        out[col] = out[col].astype(str).apply(lambda v: v if v in known else encoder.classes_[0])
        out[col] = encoder.transform(out[col])

    return out[feature_names]


def build_preprocessing_metadata(fe_data: dict[str, Any]) -> dict[str, Any]:
    numeric_cols = list(fe_data["numerical_imputer"].feature_names_in_)
    categorical_cols = list(fe_data["label_encoders"].keys())

    numeric_fill = {
        col: float(value)
        for col, value in zip(numeric_cols, fe_data["numerical_imputer"].statistics_)
    }
    categorical_fill = {
        col: str(value)
        for col, value in zip(categorical_cols, fe_data["categorical_imputer"].statistics_)
    }
    scaler_meta = {
        col: {
            "mean": float(mean),
            "scale": float(scale),
            "variance": float(var),
        }
        for col, mean, scale, var in zip(
            numeric_cols,
            fe_data["scaler"].mean_,
            fe_data["scaler"].scale_,
            fe_data["scaler"].var_,
        )
    }
    label_meta = {
        col: {
            "classes": [str(v) for v in encoder.classes_],
            "mapping": {str(v): int(i) for i, v in enumerate(encoder.classes_)},
        }
        for col, encoder in fe_data["label_encoders"].items()
    }

    return {
        "feature_names": list(fe_data["feature_names"]),
        "original_features": list(fe_data.get("original_features", fe_data["feature_names"])),
        "column_aliases": COLUMN_ALIASES,
        "numerical_features": numeric_cols,
        "categorical_features": categorical_cols,
        "numerical_imputer_statistics": numeric_fill,
        "categorical_imputer_statistics": categorical_fill,
        "scaler": scaler_meta,
        "label_encoders": label_meta,
        "label_map": LABEL_MAP,
    }


def export_model_json(model_data: dict[str, Any], output_dir: Path) -> tuple[Path, Path, Path]:
    model = model_data["model"]
    raw_bytes = bytes(model._Booster.handle)

    ubj_path = output_dir / "best_xgboost_model.ubj"
    ubj_path.write_bytes(raw_bytes)

    snapshot_json = UBJSONDecoder(raw_bytes).decode()
    snapshot_path = output_dir / "best_xgboost_snapshot.json"
    snapshot_path.write_text(
        json.dumps(snapshot_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    model_json = snapshot_json["Model"] if isinstance(snapshot_json, dict) and "Model" in snapshot_json else snapshot_json
    json_path = output_dir / "best_xgboost_model.json"
    json_path.write_text(
        json.dumps(model_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ubj_path, snapshot_path, json_path


def export_metadata(model_data: dict[str, Any], fe_data: dict[str, Any], output_dir: Path) -> None:
    feature_importance = pd.DataFrame(
        {
            "feature": list(model_data["feature_names"]),
            "model_feature_importance": np.asarray(model_data["feature_importance"], dtype=float),
        }
    ).sort_values("model_feature_importance", ascending=False)
    feature_importance.to_csv(output_dir / "model_feature_importance.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "model_type": model_data.get("model_type"),
        "feature_names": list(model_data["feature_names"]),
        "feature_importance": _to_python(model_data["feature_importance"]),
        "preprocessing": build_preprocessing_metadata(fe_data),
        "label_map": LABEL_MAP,
        "note": "Use best_xgboost_model.json in R with xgboost::xgb.load().",
    }
    (output_dir / "export_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_input_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    raise ValueError(f"Unsupported input format: {path.suffix}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export text_predictor XGBoost model and preprocessing metadata for RStudio.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("text_predictor/weights/best_xgboost_model.pkl"),
        help="Path to best_xgboost_model.pkl",
    )
    parser.add_argument(
        "--feature-engineer",
        type=Path,
        default=Path("text_predictor/weights/feature_engineer.pkl"),
        help="Path to feature_engineer.pkl",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("示例数据/医生测试题目.xlsx"),
        help="Optional input file to export a transformed feature matrix example.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("text_predictor/r_export"),
        help="Directory for exported R-friendly artifacts.",
    )
    return parser


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = build_argument_parser()
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model.resolve()
    fe_path = args.feature_engineer.resolve()

    model_data, fe_data = load_artifacts(model_path, fe_path, project_dir)
    ubj_path, snapshot_path, json_path = export_model_json(model_data, output_dir)
    export_metadata(model_data, fe_data, output_dir)

    if args.input_file and args.input_file.exists():
        raw_df = load_input_table(args.input_file.resolve())
        standardized = standardize_columns(raw_df)
        standardized.to_csv(output_dir / "sample_input_standardized.csv", index=False, encoding="utf-8-sig")

        transformed = preprocess_for_export(raw_df, fe_data)
        transformed.to_csv(output_dir / "sample_transformed_features.csv", index=False, encoding="utf-8-sig")

    print(f"Exported UBJSON model: {ubj_path}")
    print(f"Exported snapshot:     {snapshot_path}")
    print(f"Exported JSON model:   {json_path}")
    print(f"Export directory:      {output_dir}")


if __name__ == "__main__":
    main()
