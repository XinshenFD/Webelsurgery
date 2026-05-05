#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手术决策文本预测工具

功能：
  1. 预测单条病例数据是否需要手术（手术 / 不手术）
  2. 批量预测 Excel/CSV 文件中的所有病例
  3. 批量预测结果导出为 Excel 文件

使用方法：
  python predict.py --file 病例数据.xlsx
  python predict.py --file 病例数据.xlsx --output 结果.xlsx
  python predict.py --single --dislocation 中 --vision 0.3 --sphere -3.0 --cylinder -1.5 --iol -1.0
"""

import argparse
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


# 允许的同义列名（统一重命名到标准列名）
COLUMN_ALIASES = {
    '是否配合检查': '是否配合',
}

# 预测结果标签
LABEL_MAP = {0: '不手术', 1: '手术'}


def get_default_weights_dir():
    """获取默认模型权重目录"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')


def load_artifacts(model_path, fe_path):
    """加载模型和特征工程器"""
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在 - {model_path}")
        print("请将 best_xgboost_model.pkl 放置到 weights/ 目录下")
        sys.exit(1)

    if not os.path.exists(fe_path):
        print(f"错误：特征工程文件不存在 - {fe_path}")
        print("请将 feature_engineer.pkl 放置到 weights/ 目录下")
        sys.exit(1)

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    with open(fe_path, 'rb') as f:
        fe_data = pickle.load(f)

    # 模型保存为 dict，取出实际模型对象
    if isinstance(model_data, dict):
        actual_model = model_data.get('model', model_data)
    else:
        # 可能是 SurgeryClassifier 包装类
        actual_model = model_data.model if hasattr(model_data, 'model') else model_data

    print(f"模型加载成功: {model_path}")
    print(f"特征工程器加载成功: {fe_path}")

    return actual_model, fe_data


def preprocess(df, fe_data):
    """
    使用已拟合的特征工程器对数据进行预处理。
    所有特征列、编码器、归一化器全部从 fe_data（pkl）中读取，不依赖硬编码配置。

    fe_data 是一个 dict，包含：
      - numerical_imputer: 数值型缺失值填充器（含 feature_names_in_）
      - scaler: 标准化器
      - categorical_imputer: 分类型缺失值填充器
      - label_encoders: dict，每个分类特征对应一个 LabelEncoder
      - feature_names: 训练时最终特征列顺序（必须严格对齐）
    """
    X = df.copy()

    # 重命名同义列名
    X.rename(columns=COLUMN_ALIASES, inplace=True)

    # 从 pkl 读取实际特征信息
    feature_names = fe_data.get('feature_names', [])
    numerical_imputer = fe_data.get('numerical_imputer')
    scaler = fe_data.get('scaler')
    categorical_imputer = fe_data.get('categorical_imputer')
    label_encoders = fe_data.get('label_encoders', {})

    # 从 imputer 中读取实际数值列名
    if numerical_imputer is not None and hasattr(numerical_imputer, 'feature_names_in_'):
        num_cols = list(numerical_imputer.feature_names_in_)
    else:
        num_cols = [c for c in feature_names if c not in label_encoders]

    # 分类列 = label_encoders 的键
    cat_cols = list(label_encoders.keys())

    # 检查必需列是否存在
    all_needed = num_cols + cat_cols
    missing_cols = [c for c in all_needed if c not in X.columns]
    if missing_cols:
        raise ValueError(
            f"输入数据缺少以下必需列: {missing_cols}\n"
            f"数值列: {num_cols}\n"
            f"分类列: {cat_cols}"
        )

    # 强制数值列转为 float
    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # 数值型：缺失值填充 + 标准化
    if numerical_imputer and num_cols:
        X[num_cols] = numerical_imputer.transform(X[num_cols])
    if scaler and num_cols:
        X[num_cols] = scaler.transform(X[num_cols])

    # 分类型：缺失值填充 + 标签编码
    if categorical_imputer and cat_cols:
        X[cat_cols] = categorical_imputer.transform(X[cat_cols])

    for col in cat_cols:
        le = label_encoders[col]
        known = set(le.classes_)
        X[col] = X[col].astype(str).apply(
            lambda v: v if v in known else le.classes_[0]
        )
        X[col] = le.transform(X[col])

    # 按训练时的特征顺序排列
    X = X[feature_names]

    return X.values


def predict_single_record(model, fe_data, dislocation, vision, sphere, cylinder, iol):
    """预测单条病例（仅使用模型实际用到的5个特征）"""
    row = {
        '脱位程度': str(dislocation),
        '矫正视力': vision,
        '矫正球镜度数(D)': sphere,
        '矫正柱镜度数(D)': cylinder,
        'IOLMaster-Cyl(D)': iol,
    }
    df = pd.DataFrame([row])
    X = preprocess(df, fe_data)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return int(pred), proba


def predict_file(model, fe_data, file_path, output_path=None):
    """批量预测文件中的所有病例"""
    if not os.path.exists(file_path):
        print(f"错误：数据文件不存在 - {file_path}")
        return

    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    elif ext == '.csv':
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    else:
        print(f"不支持的文件格式: {ext}（支持 .xlsx / .xls / .csv）")
        return

    print(f"\n加载数据: {file_path}（共 {len(df)} 条）")

    # 重命名同义列名
    df.rename(columns=COLUMN_ALIASES, inplace=True)

    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"错误：数据文件缺少以下必需列: {missing_cols}")
        print(f"必需列: {FEATURE_COLUMNS}")
        return

    try:
        X = preprocess(df, fe_data)
    except Exception as e:
        print(f"数据预处理失败: {e}")
        return

    print("预测中...")
    preds = model.predict(X)
    probas = model.predict_proba(X)

    df['预测结果'] = [LABEL_MAP[p] for p in preds]
    df['不手术概率'] = probas[:, 0].round(4)
    df['手术概率'] = probas[:, 1].round(4)

    # 统计
    print("\n" + "=" * 60)
    print("预测统计:")
    for label in ['不手术', '手术']:
        count = (df['预测结果'] == label).sum()
        print(f"  {label}: {count} 例 ({count / len(df) * 100:.1f}%)")

    # 保存
    if output_path is None:
        base = os.path.splitext(file_path)[0]
        output_path = base + '_预测结果.xlsx'

    df.to_excel(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='手术决策文本预测工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  批量预测 Excel 文件:
    python predict.py --file 病例数据.xlsx

  批量预测并指定输出路径:
    python predict.py --file 病例数据.xlsx --output 结果.xlsx

  批量预测 CSV 文件:
    python predict.py --file 病例数据.csv

  预测单条病例:
    python predict.py --single --cooperation 能配合 --dislocation 中 \\
                      --vision 0.3 --sphere -3.0 --cylinder -1.5 \\
                      --iol -1.0 --age 8

  指定模型路径:
    python predict.py --file data.xlsx --model weights/best_xgboost_model.pkl \\
                      --fe weights/feature_engineer.pkl
        """
    )

    parser.add_argument('--file', type=str, help='病例数据文件路径（.xlsx / .csv）')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径（默认: 原文件名_预测结果.xlsx）')
    parser.add_argument('--model', type=str, default=None, help='模型权重路径（默认: weights/best_xgboost_model.pkl）')
    parser.add_argument('--fe', type=str, default=None, help='特征工程器路径（默认: weights/feature_engineer.pkl）')

    # 单条预测参数
    parser.add_argument('--single', action='store_true', help='预测单条病例')
    parser.add_argument('--dislocation', type=str, help='脱位程度（如: 轻 / 中 / 重）')
    parser.add_argument('--vision', type=float, help='矫正视力（如: 0.3）')
    parser.add_argument('--sphere', type=float, help='矫正球镜度数(D)（如: -3.0）')
    parser.add_argument('--cylinder', type=float, help='矫正柱镜度数(D)（如: -1.5）')
    parser.add_argument('--iol', type=float, help='IOLMaster-Cyl(D)（如: -1.0）')

    args = parser.parse_args()

    if not args.file and not args.single:
        parser.print_help()
        print("\n错误：请指定 --file 或 --single 参数")
        sys.exit(1)

    # 路径配置
    weights_dir = get_default_weights_dir()
    model_path = args.model or os.path.join(weights_dir, 'best_xgboost_model.pkl')
    fe_path = args.fe or os.path.join(weights_dir, 'feature_engineer.pkl')

    # 加载模型
    model, fe_data = load_artifacts(model_path, fe_path)

    # 执行预测
    if args.single:
        required = ['dislocation', 'vision', 'sphere', 'cylinder', 'iol']
        missing = [f'--{r}' for r in required if getattr(args, r) is None]
        if missing:
            print(f"错误：单条预测需要提供以下参数: {missing}")
            sys.exit(1)

        pred_class, proba = predict_single_record(
            model, fe_data,
            dislocation=args.dislocation,
            vision=args.vision,
            sphere=args.sphere,
            cylinder=args.cylinder,
            iol=args.iol
        )

        print("\n" + "=" * 50)
        print("预测结果:")
        print(f"  建议: {LABEL_MAP[pred_class]}")
        print(f"\n各类别概率:")
        print(f"  不手术: {proba[0]:.4f}")
        print(f"  手术:   {proba[1]:.4f}")
        print("=" * 50)

    elif args.file:
        predict_file(model, fe_data, args.file, args.output)


if __name__ == '__main__':
    main()
