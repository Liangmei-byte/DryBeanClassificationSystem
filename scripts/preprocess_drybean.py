#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dry Bean Dirty Dataset: 数据清洗 + 特征工程脚本



输出目录默认：
DryBeanProcessed/
    cleaned_train.csv / cleaned_val.csv / cleaned_test.csv
    model_train.csv / model_val.csv / model_test.csv
    drybean_processed.npz
    preprocess_config.json
    clean_report.json

说明：
1. 标签清洗：去空格、转大写、把 0->O、3->E。
2. 数值清洗：处理 NaN、?、带单位字符串如 "0.8252 cm"。
3. 物理异常：面积、周长、轴长等非正数置为缺失，不直接删除测试集。
4. 派生特征修复：按数学关系重算 AspectRation、Eccentricity、EquivDiameter、
   Solidity、roundness、Compactness、ShapeFactor1/2/3。
5. 训练集拟合：缺失值中位数、异常值截断边界、标准化器都只在训练集上拟合，
   然后应用到验证集和测试集，避免数据泄漏。
6. 输出两套特征：
   - X_*：中位数填补 + IQR 截断后的未标准化特征，适合 LightGBM / XGBoost。
   - X_*_scaled：再经过 StandardScaler 标准化后的特征，适合 Softmax / KNN。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ========== 1. 基础配置 ==========

LABELS: List[str] = [
    "BARBUNYA",
    "BOMBAY",
    "CALI",
    "DERMASON",
    "HOROZ",
    "SEKER",
    "SIRA",
]

LABEL_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(LABELS)}
ID_TO_LABEL: Dict[int, str] = {idx: name for name, idx in LABEL_TO_ID.items()}

ORIGINAL_FEATURES: List[str] = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",       # 原始文件列名就是 AspectRation，疑似拼写错误，不强制改名
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]

TARGET_COL = "Class"

REQUIRED_COLUMNS = ORIGINAL_FEATURES + [TARGET_COL]

# 明确必须大于 0 的基础几何量
POSITIVE_COLUMNS = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "ConvexArea",
    "EquivDiameter",
]

# 理论上处于 (0, 1] 或 [0, 1] 附近的比例类变量
RATIO_COLUMNS_0_1 = [
    "Eccentricity",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor3",
    "ShapeFactor4",
]

# 需要从训练集生成、并保证 train / val / test 顺序一致的最终特征列
ENGINEERED_FEATURES: List[str] = [
    # 原始/修复后的 16 个特征
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",

    # 新增特征：尺度、形状、凸包缺陷、比例关系
    "log_Area",
    "log_Perimeter",
    "log_ConvexArea",
    "log_MajorAxisLength",
    "log_MinorAxisLength",
    "AxisLengthDiff",
    "AxisLengthSum",
    "AxisLengthProduct",
    "AxisRatio_Major_Minor",
    "Area_to_ConvexArea",
    "ConvexDefect",
    "ConvexDefectRatio",
    "Perimeter_to_sqrtArea",
    "Area_to_Perimeter2",
    "Major_to_sqrtArea",
    "Minor_to_sqrtArea",
    "EquivalentDiameter_to_Major",
    "EquivalentDiameter_to_Minor",
]


@dataclass
class PreprocessConfig:
    """
    记录训练集拟合得到的预处理参数。
    """
    labels: List[str]
    label_to_id: Dict[str, int]
    feature_columns: List[str]
    median_values: Dict[str, float]
    clip_lower: Dict[str, float]
    clip_upper: Dict[str, float]
    scaler_mean: Dict[str, float]
    scaler_scale: Dict[str, float]
    repair_derived_features: bool
    clip_outliers: bool
    iqr_factor: float


# ========== 2. 通用工具函数 ==========

def safe_divide(numerator, denominator):
    """
    安全除法：分母为 0 或无穷时返回 NaN。
    支持 pandas Series / numpy array。
    """
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    out = numerator / denominator
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def rel_diff(a, b):
    """
    相对误差，用于统计派生特征不一致。
    """
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-12)
    return np.abs(a - b) / denom


def parse_numeric_series(s: pd.Series) -> pd.Series:
    """
    把一列转换为数值型。

    可处理：
    - np.nan
    - "?"
    - "0.8252 cm"
    - 字符串数字
    - 前后空格

    如果字符串中包含合法数字，则提取第一个数字；否则记为 NaN。
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    text = s.astype("string").str.strip()
    text = text.replace(
        {
            "": pd.NA,
            "?": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
            "None": pd.NA,
            "null": pd.NA,
            "NULL": pd.NA,
        }
    )

    # 提取第一个数字，支持负数、小数和科学计数法
    extracted = text.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def standardize_label(x) -> Optional[str]:
    """
    标准化 Class 标签。

    污染示例：
    - "sira" -> "SIRA"
    - "SIRA " -> "SIRA"
    - "S3K3R" -> "SEKER"
    - "D3RMAS0N" -> "DERMASON"
    - "H0R0Z" -> "HOROZ"
    - "B0MBAY" -> "BOMBAY"
    """
    if pd.isna(x):
        return None

    label = str(x).strip().upper()
    label = label.replace("0", "O").replace("3", "E")

    # 可扩展的人工映射表。当前污染通过上面规则基本都能恢复。
    alias = {
        "BARBUNYA": "BARBUNYA",
        "BOMBAY": "BOMBAY",
        "CALI": "CALI",
        "DERMASON": "DERMASON",
        "HOROZ": "HOROZ",
        "SEKER": "SEKER",
        "SIRA": "SIRA",
    }
    label = alias.get(label, label)

    if label in LABEL_TO_ID:
        return label
    return None


def validate_columns(df: pd.DataFrame, split_name: str) -> None:
    """
    检查必要列是否存在。
    """
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{split_name} 缺少必要列: {missing_cols}")


def count_question_mark(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    return int((df[col].astype("string").str.strip() == "?").sum())


def count_contains_unit(df: pd.DataFrame, col: str, unit: str = "cm") -> int:
    if col not in df.columns:
        return 0
    return int(df[col].astype("string").str.contains(unit, case=False, na=False).sum())


def summarize_raw_pollution(df_raw: pd.DataFrame, df_numeric: pd.DataFrame, split_name: str) -> Dict:
    """
    生成每个 split 的清洗前污染统计。
    """
    report = {
        "split": split_name,
        "n_rows_raw": int(len(df_raw)),
        "n_cols_raw": int(df_raw.shape[1]),
        "missing_raw": {
            col: int(df_raw[col].isna().sum())
            for col in df_raw.columns
        },
        "question_mark_count": {
            col: count_question_mark(df_raw, col)
            for col in df_raw.columns
        },
        "compactness_with_cm_count": count_contains_unit(df_raw, "Compactness", "cm"),
    }

    # 标签污染统计
    if TARGET_COL in df_raw.columns:
        raw_label = df_raw[TARGET_COL]
        std_label = raw_label.map(standardize_label)
        raw_trim_upper = raw_label.astype("string").str.strip().str.upper()
        report["class_unique_raw_count"] = int(raw_label.nunique(dropna=False))
        report["class_unique_raw_values"] = sorted([str(x) for x in raw_label.dropna().unique()])
        report["class_invalid_after_standardize_count"] = int(std_label.isna().sum())
        report["class_changed_by_standardize_count"] = int(
            (raw_trim_upper.fillna("<NA>") != pd.Series(std_label, index=df_raw.index).astype("string").fillna("<NA>")).sum()
        )

    # 物理异常统计
    for col in POSITIVE_COLUMNS:
        if col in df_numeric.columns:
            report[f"{col}_non_positive_count"] = int((df_numeric[col] <= 0).sum(skipna=True))
            report[f"{col}_missing_after_numeric_parse"] = int(df_numeric[col].isna().sum())

    # 比例范围异常
    for col in RATIO_COLUMNS_0_1:
        if col in df_numeric.columns:
            report[f"{col}_outside_0_1_count"] = int(((df_numeric[col] < 0) | (df_numeric[col] > 1)).sum(skipna=True))
            report[f"{col}_missing_after_numeric_parse"] = int(df_numeric[col].isna().sum())

    # 派生特征一致性异常统计，清洗前统计，容忍浮点误差
    report["formula_inconsistency_before_repair"] = formula_inconsistency_counts(df_numeric)

    return report


def formula_inconsistency_counts(df: pd.DataFrame, tol: float = 1e-3) -> Dict[str, int]:
    """
    统计派生特征与公式不一致的数量。
    只统计公式两边都非缺失的行。
    """
    out = {}

    required = set(df.columns)

    if {"AspectRation", "MajorAxisLength", "MinorAxisLength"}.issubset(required):
        expected = safe_divide(df["MajorAxisLength"], df["MinorAxisLength"])
        mask = df["AspectRation"].notna() & expected.notna() & (rel_diff(df["AspectRation"], expected) > tol)
        out["AspectRation_vs_Major_div_Minor"] = int(mask.sum())

    if {"Eccentricity", "MajorAxisLength", "MinorAxisLength"}.issubset(required):
        ratio = safe_divide(df["MinorAxisLength"], df["MajorAxisLength"])
        expected = np.sqrt(1 - np.square(ratio))
        expected = pd.Series(expected, index=df.index).where((ratio > 0) & (ratio <= 1))
        mask = df["Eccentricity"].notna() & expected.notna() & (rel_diff(df["Eccentricity"], expected) > tol)
        out["Eccentricity_vs_axes"] = int(mask.sum())

    if {"EquivDiameter", "Area"}.issubset(required):
        expected = np.sqrt(4 * df["Area"] / np.pi)
        expected = pd.Series(expected, index=df.index).where(df["Area"] > 0)
        mask = df["EquivDiameter"].notna() & expected.notna() & (rel_diff(df["EquivDiameter"], expected) > tol)
        out["EquivDiameter_vs_Area"] = int(mask.sum())

    if {"Solidity", "Area", "ConvexArea"}.issubset(required):
        expected = safe_divide(df["Area"], df["ConvexArea"])
        mask = df["Solidity"].notna() & expected.notna() & (rel_diff(df["Solidity"], expected) > tol)
        out["Solidity_vs_Area_div_ConvexArea"] = int(mask.sum())

    if {"roundness", "Area", "Perimeter"}.issubset(required):
        expected = safe_divide(4 * np.pi * df["Area"], np.square(df["Perimeter"]))
        mask = df["roundness"].notna() & expected.notna() & (rel_diff(df["roundness"], expected) > tol)
        out["roundness_vs_4piArea_div_Perimeter2"] = int(mask.sum())

    if {"Compactness", "EquivDiameter", "MajorAxisLength"}.issubset(required):
        expected = safe_divide(df["EquivDiameter"], df["MajorAxisLength"])
        mask = df["Compactness"].notna() & expected.notna() & (rel_diff(df["Compactness"], expected) > tol)
        out["Compactness_vs_EquivDiameter_div_Major"] = int(mask.sum())

    if {"ShapeFactor1", "MajorAxisLength", "Area"}.issubset(required):
        expected = safe_divide(df["MajorAxisLength"], df["Area"])
        mask = df["ShapeFactor1"].notna() & expected.notna() & (rel_diff(df["ShapeFactor1"], expected) > tol)
        out["ShapeFactor1_vs_Major_div_Area"] = int(mask.sum())

    if {"ShapeFactor2", "MinorAxisLength", "Area"}.issubset(required):
        expected = safe_divide(df["MinorAxisLength"], df["Area"])
        mask = df["ShapeFactor2"].notna() & expected.notna() & (rel_diff(df["ShapeFactor2"], expected) > tol)
        out["ShapeFactor2_vs_Minor_div_Area"] = int(mask.sum())

    if {"ShapeFactor3", "Compactness"}.issubset(required):
        expected = np.square(df["Compactness"])
        mask = df["ShapeFactor3"].notna() & expected.notna() & (rel_diff(df["ShapeFactor3"], expected) > tol)
        out["ShapeFactor3_vs_Compactness2"] = int(mask.sum())

    return out


# ========== 3. 单个 split 清洗 ==========

def convert_feature_columns_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    把所有特征列转为数值。
    """
    out = df.copy()
    for col in ORIGINAL_FEATURES:
        out[col] = parse_numeric_series(out[col])
    return out


def mark_invalid_physical_values_as_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    对明显物理不可能或范围不合理的值置为 NaN。

    注意：这里只处理“单元格级别”的不可信值，不删除整行；
    这样验证集和测试集仍可保持样本规模，后续用训练集中位数填补。
    """
    out = df.copy()

    # 基础几何量必须为正
    for col in POSITIVE_COLUMNS:
        out.loc[out[col] <= 0, col] = np.nan

    # ConvexArea 理论上应 >= Area；如果 ConvexArea < Area，则 ConvexArea 不可信
    mask_bad_convex = (
        out["Area"].notna()
        & out["ConvexArea"].notna()
        & (out["ConvexArea"] < out["Area"])
    )
    out.loc[mask_bad_convex, "ConvexArea"] = np.nan

    # 长轴应 >= 短轴；如果相反，优先交换，而不是置空
    mask_swap_axes = (
        out["MajorAxisLength"].notna()
        & out["MinorAxisLength"].notna()
        & (out["MajorAxisLength"] < out["MinorAxisLength"])
    )
    major_tmp = out.loc[mask_swap_axes, "MajorAxisLength"].copy()
    out.loc[mask_swap_axes, "MajorAxisLength"] = out.loc[mask_swap_axes, "MinorAxisLength"]
    out.loc[mask_swap_axes, "MinorAxisLength"] = major_tmp

    # 比例类特征一般应在 [0, 1] 内，轻微浮点越界也当作不可信
    for col in RATIO_COLUMNS_0_1:
        out.loc[(out[col] < 0) | (out[col] > 1), col] = np.nan

    # AspectRation 理论上 >= 1
    out.loc[out["AspectRation"] < 1, "AspectRation"] = np.nan

    # ShapeFactor1/2 应为正
    for col in ["ShapeFactor1", "ShapeFactor2"]:
        out.loc[out[col] <= 0, col] = np.nan

    return out


def repair_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    用更基础的几何量重算可由公式得到的派生特征，修复派生特征不一致污染。

    可重算：
    - AspectRation = MajorAxisLength / MinorAxisLength
    - Eccentricity = sqrt(1 - (MinorAxisLength / MajorAxisLength)^2)
    - EquivDiameter = sqrt(4 * Area / pi)
    - Solidity = Area / ConvexArea
    - roundness = 4 * pi * Area / Perimeter^2
    - Compactness = EquivDiameter / MajorAxisLength
    - ShapeFactor1 = MajorAxisLength / Area
    - ShapeFactor2 = MinorAxisLength / Area
    - ShapeFactor3 = Compactness^2

    不重算：
    - Extent：缺少外接矩形面积，不能可靠重算。
    - ShapeFactor4：仅凭当前列不可靠重算，保留原值。
    """
    out = df.copy()

    area = out["Area"]
    perimeter = out["Perimeter"]
    major = out["MajorAxisLength"]
    minor = out["MinorAxisLength"]
    convex_area = out["ConvexArea"]

    # 只在基础量有效时重算，否则结果保持 NaN
    out["AspectRation"] = safe_divide(major, minor)

    ratio = safe_divide(minor, major)
    out["Eccentricity"] = np.sqrt(1 - np.square(ratio))
    out.loc[(ratio <= 0) | (ratio > 1), "Eccentricity"] = np.nan

    out["EquivDiameter"] = np.sqrt(4 * area / np.pi)
    out.loc[area <= 0, "EquivDiameter"] = np.nan

    out["Solidity"] = safe_divide(area, convex_area)

    out["roundness"] = safe_divide(4 * np.pi * area, np.square(perimeter))

    # Compactness 与原始样本吻合的公式：EquivDiameter / MajorAxisLength
    out["Compactness"] = safe_divide(out["EquivDiameter"], major)

    out["ShapeFactor1"] = safe_divide(major, area)
    out["ShapeFactor2"] = safe_divide(minor, area)
    out["ShapeFactor3"] = np.square(out["Compactness"])

    # 修复后再次把范围不合理的值置空
    out = mark_invalid_physical_values_as_nan(out)

    return out


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加特征工程列。
    """
    out = df.copy()

    # 对数特征：对长尾分布更友好，尤其有利于线性 Softmax
    for col in ["Area", "Perimeter", "ConvexArea", "MajorAxisLength", "MinorAxisLength"]:
        out[f"log_{col}"] = np.log1p(out[col].where(out[col] > 0))

    # 轴长度相关
    out["AxisLengthDiff"] = out["MajorAxisLength"] - out["MinorAxisLength"]
    out["AxisLengthSum"] = out["MajorAxisLength"] + out["MinorAxisLength"]
    out["AxisLengthProduct"] = out["MajorAxisLength"] * out["MinorAxisLength"]
    out["AxisRatio_Major_Minor"] = safe_divide(out["MajorAxisLength"], out["MinorAxisLength"])

    # 凸包缺陷
    out["Area_to_ConvexArea"] = safe_divide(out["Area"], out["ConvexArea"])
    out["ConvexDefect"] = out["ConvexArea"] - out["Area"]
    out["ConvexDefectRatio"] = safe_divide(out["ConvexArea"] - out["Area"], out["ConvexArea"])

    # 尺度归一化形状特征
    sqrt_area = np.sqrt(out["Area"])
    out["Perimeter_to_sqrtArea"] = safe_divide(out["Perimeter"], sqrt_area)
    out["Area_to_Perimeter2"] = safe_divide(out["Area"], np.square(out["Perimeter"]))
    out["Major_to_sqrtArea"] = safe_divide(out["MajorAxisLength"], sqrt_area)
    out["Minor_to_sqrtArea"] = safe_divide(out["MinorAxisLength"], sqrt_area)
    out["EquivalentDiameter_to_Major"] = safe_divide(out["EquivDiameter"], out["MajorAxisLength"])
    out["EquivalentDiameter_to_Minor"] = safe_divide(out["EquivDiameter"], out["MinorAxisLength"])

    # 统一替换无穷大
    out = out.replace([np.inf, -np.inf], np.nan)

    return out


def clean_one_split(
    df_raw: pd.DataFrame,
    split_name: str,
    repair_derived: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    清洗单个 split，返回：
    - 清洗并完成特征工程的 DataFrame
    - 该 split 的污染统计报告
    """
    validate_columns(df_raw, split_name)

    # 只保留需要列，避免额外索引列污染
    df_raw = df_raw[REQUIRED_COLUMNS].copy()

    # 特征转数值
    df_numeric = convert_feature_columns_to_numeric(df_raw)

    report = summarize_raw_pollution(df_raw, df_numeric, split_name)

    # 标签标准化
    df_numeric[TARGET_COL] = df_raw[TARGET_COL].map(standardize_label)

    n_before_label_drop = len(df_numeric)
    df_numeric = df_numeric.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    report["dropped_rows_invalid_label"] = int(n_before_label_drop - len(df_numeric))

    # 物理异常单元格置 NaN
    df_clean = mark_invalid_physical_values_as_nan(df_numeric)

    if repair_derived:
        df_clean = repair_derived_features(df_clean)

    # 再统计一次公式不一致
    report["formula_inconsistency_after_repair"] = formula_inconsistency_counts(df_clean)

    # 特征工程
    df_clean = add_engineered_features(df_clean)

    # 只保留目标列和最终特征列，顺序固定
    missing_features = [c for c in ENGINEERED_FEATURES if c not in df_clean.columns]
    if missing_features:
        raise RuntimeError(f"{split_name} 特征工程后缺少列: {missing_features}")

    df_clean = df_clean[ENGINEERED_FEATURES + [TARGET_COL]]

    report["n_rows_after_label_cleaning"] = int(len(df_clean))
    report["missing_after_cell_cleaning_and_feature_engineering"] = {
        col: int(df_clean[col].isna().sum())
        for col in ENGINEERED_FEATURES
    }

    return df_clean, report


# ========== 4. 训练集拟合：中位数、异常值截断、标准化 ==========

def fit_iqr_bounds(X_train: pd.DataFrame, iqr_factor: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """
    基于训练集拟合 IQR 异常值截断边界。
    这里用 3.0 * IQR，比常见 1.5 * IQR 更温和，避免过度裁剪真实类别差异。
    """
    q1 = X_train.quantile(0.25, numeric_only=True)
    q3 = X_train.quantile(0.75, numeric_only=True)
    iqr = q3 - q1

    lower = q1 - iqr_factor * iqr
    upper = q3 + iqr_factor * iqr

    # 如果某列 IQR=0，则改用 min/max，避免所有值都被裁到同一点
    zero_iqr_cols = iqr[iqr == 0].index
    if len(zero_iqr_cols) > 0:
        lower.loc[zero_iqr_cols] = X_train[zero_iqr_cols].min()
        upper.loc[zero_iqr_cols] = X_train[zero_iqr_cols].max()

    return lower, upper


def apply_numeric_preprocess(
    X: pd.DataFrame,
    medians: pd.Series,
    clip_lower: Optional[pd.Series] = None,
    clip_upper: Optional[pd.Series] = None,
    clip_outliers: bool = True,
) -> pd.DataFrame:
    """
    对任意 split 应用训练集拟合好的：
    - IQR 截断边界
    - 中位数缺失填补
    """
    out = X.copy()
    out = out.replace([np.inf, -np.inf], np.nan)

    if clip_outliers and clip_lower is not None and clip_upper is not None:
        out = out.clip(lower=clip_lower, upper=clip_upper, axis=1)

    out = out.fillna(medians)

    # 如果某列训练集中全缺失，median 仍为 NaN；兜底填 0
    out = out.fillna(0.0)

    return out.astype(float)


def fit_and_transform_all_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    clip_outliers: bool = True,
    iqr_factor: float = 3.0,
    repair_derived_features: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray], PreprocessConfig, Dict]:
    """
    用训练集拟合数值预处理参数，并转换 train/val/test。
    """
    X_train_raw = df_train[ENGINEERED_FEATURES].copy()
    X_val_raw = df_val[ENGINEERED_FEATURES].copy()
    X_test_raw = df_test[ENGINEERED_FEATURES].copy()

    y_train = df_train[TARGET_COL].map(LABEL_TO_ID).astype(int).to_numpy()
    y_val = df_val[TARGET_COL].map(LABEL_TO_ID).astype(int).to_numpy()
    y_test = df_test[TARGET_COL].map(LABEL_TO_ID).astype(int).to_numpy()

    medians = X_train_raw.median(numeric_only=True)
    clip_lower, clip_upper = fit_iqr_bounds(X_train_raw, iqr_factor=iqr_factor)

    X_train = apply_numeric_preprocess(
        X_train_raw, medians, clip_lower, clip_upper, clip_outliers=clip_outliers
    )
    X_val = apply_numeric_preprocess(
        X_val_raw, medians, clip_lower, clip_upper, clip_outliers=clip_outliers
    )
    X_test = apply_numeric_preprocess(
        X_test_raw, medians, clip_lower, clip_upper, clip_outliers=clip_outliers
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=ENGINEERED_FEATURES,
        index=X_train.index,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=ENGINEERED_FEATURES,
        index=X_val.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=ENGINEERED_FEATURES,
        index=X_test.index,
    )

    config = PreprocessConfig(
        labels=LABELS,
        label_to_id=LABEL_TO_ID,
        feature_columns=ENGINEERED_FEATURES,
        median_values={k: float(v) for k, v in medians.to_dict().items()},
        clip_lower={k: float(v) for k, v in clip_lower.to_dict().items()},
        clip_upper={k: float(v) for k, v in clip_upper.to_dict().items()},
        scaler_mean={k: float(v) for k, v in zip(ENGINEERED_FEATURES, scaler.mean_)},
        scaler_scale={k: float(v) for k, v in zip(ENGINEERED_FEATURES, scaler.scale_)},
        repair_derived_features=repair_derived_features,
        clip_outliers=clip_outliers,
        iqr_factor=iqr_factor,
    )

    processed_dfs = {
        "train": pd.concat([X_train, df_train[[TARGET_COL]].reset_index(drop=True)], axis=1),
        "val": pd.concat([X_val, df_val[[TARGET_COL]].reset_index(drop=True)], axis=1),
        "test": pd.concat([X_test, df_test[[TARGET_COL]].reset_index(drop=True)], axis=1),
        "train_scaled": pd.concat([X_train_scaled, df_train[[TARGET_COL]].reset_index(drop=True)], axis=1),
        "val_scaled": pd.concat([X_val_scaled, df_val[[TARGET_COL]].reset_index(drop=True)], axis=1),
        "test_scaled": pd.concat([X_test_scaled, df_test[[TARGET_COL]].reset_index(drop=True)], axis=1),
    }

    arrays = {
        # 未标准化，推荐用于 LightGBM / XGBoost
        "X_train": X_train.to_numpy(dtype=np.float32),
        "X_val": X_val.to_numpy(dtype=np.float32),
        "X_test": X_test.to_numpy(dtype=np.float32),

        # 标准化，推荐用于 Softmax / KNN
        "X_train_scaled": X_train_scaled.to_numpy(dtype=np.float32),
        "X_val_scaled": X_val_scaled.to_numpy(dtype=np.float32),
        "X_test_scaled": X_test_scaled.to_numpy(dtype=np.float32),

        "y_train": y_train.astype(np.int64),
        "y_val": y_val.astype(np.int64),
        "y_test": y_test.astype(np.int64),
        "feature_names": np.array(ENGINEERED_FEATURES),
        "label_names": np.array(LABELS),
    }

    preprocess_report = {
        "n_features": len(ENGINEERED_FEATURES),
        "feature_columns": ENGINEERED_FEATURES,
        "label_distribution_after_cleaning": {
            "train": df_train[TARGET_COL].value_counts().reindex(LABELS, fill_value=0).astype(int).to_dict(),
            "val": df_val[TARGET_COL].value_counts().reindex(LABELS, fill_value=0).astype(int).to_dict(),
            "test": df_test[TARGET_COL].value_counts().reindex(LABELS, fill_value=0).astype(int).to_dict(),
        },
        "missing_after_final_preprocess": {
            "train": int(np.isnan(arrays["X_train"]).sum()),
            "val": int(np.isnan(arrays["X_val"]).sum()),
            "test": int(np.isnan(arrays["X_test"]).sum()),
            "train_scaled": int(np.isnan(arrays["X_train_scaled"]).sum()),
            "val_scaled": int(np.isnan(arrays["X_val_scaled"]).sum()),
            "test_scaled": int(np.isnan(arrays["X_test_scaled"]).sum()),
        },
    }

    return processed_dfs, arrays, config, preprocess_report


# ========== 5. 文件读写主流程 ==========

def read_csv_safely(path: Path) -> pd.DataFrame:
    """
    读取 CSV。先用默认编码，不行再尝试 utf-8-sig / gbk。
    """
    encodings = ["utf-8", "utf-8-sig", "gbk"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
    raise last_err


def load_datasets(input_dir: Path) -> Dict[str, pd.DataFrame]:
    files = {
        "train": "Dry_Bean_Dataset_Dirty_train.csv",
        "val": "Dry_Bean_Dataset_Dirty_val.csv",
        "test": "Dry_Bean_Dataset_Dirty_test.csv",
    }

    data = {}
    for split, filename in files.items():
        path = input_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"找不到文件: {path}")
        data[split] = read_csv_safely(path)
    return data


def save_outputs(
    output_dir: Path,
    cleaned_dfs: Dict[str, pd.DataFrame],
    processed_dfs: Dict[str, pd.DataFrame],
    arrays: Dict[str, np.ndarray],
    config: PreprocessConfig,
    clean_report: Dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 保存“完成脏值修复 + 特征工程，但未训练集填补/截断/标准化”的数据
    for split in ["train", "val", "test"]:
        cleaned_dfs[split].to_csv(output_dir / f"cleaned_{split}.csv", index=False)

    # 2) 保存最终模型输入 CSV
    for name, df in processed_dfs.items():
        df.to_csv(output_dir / f"model_{name}.csv", index=False)

    # 3) 保存 npz，后续训练脚本可直接 np.load
    np.savez_compressed(output_dir / "drybean_processed.npz", **arrays)

    # 4) 保存配置和报告
    with open(output_dir / "preprocess_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)

    with open(output_dir / "clean_report.json", "w", encoding="utf-8") as f:
        json.dump(clean_report, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Dry Bean Dirty Dataset 数据清洗与特征工程")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="DryBeanDataset",
        help="包含 train/val/test 三个 CSV 的文件夹路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="DryBeanProcessed",
        help="清洗后数据输出目录",
    )
    parser.add_argument(
        "--no_repair_derived",
        action="store_true",
        help="不按公式重算派生特征。默认会重算，建议不要关闭。",
    )
    parser.add_argument(
        "--no_clip_outliers",
        action="store_true",
        help="不进行训练集 IQR 异常值截断。默认启用。",
    )
    parser.add_argument(
        "--iqr_factor",
        type=float,
        default=3.0,
        help="IQR 截断系数，默认 3.0，越大越宽松。",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    repair_derived = not args.no_repair_derived
    clip_outliers = not args.no_clip_outliers

    print(f"[1/5] 读取数据目录: {input_dir}")
    raw = load_datasets(input_dir)

    print("[2/5] 清洗标签、数值类型、物理异常和派生特征...")
    cleaned_dfs = {}
    reports = {}
    for split in ["train", "val", "test"]:
        cleaned_dfs[split], reports[split] = clean_one_split(
            raw[split],
            split_name=split,
            repair_derived=repair_derived,
        )
        print(
            f"  - {split}: raw={len(raw[split])}, "
            f"after_label_clean={len(cleaned_dfs[split])}"
        )

    print("[3/5] 使用训练集拟合中位数、IQR 截断边界和标准化器...")
    processed_dfs, arrays, config, preprocess_report = fit_and_transform_all_splits(
        cleaned_dfs["train"],
        cleaned_dfs["val"],
        cleaned_dfs["test"],
        clip_outliers=clip_outliers,
        iqr_factor=args.iqr_factor,
        repair_derived_features=repair_derived,
    )

    clean_report = {
        "raw_cleaning_report": reports,
        "final_preprocess_report": preprocess_report,
        "notes": [
            "所有缺失值填补、IQR 截断边界和标准化参数均只基于训练集拟合。",
            "LightGBM/XGBoost 建议使用 drybean_processed.npz 中的 X_train/X_val/X_test。",
            "Softmax/KNN 建议使用 drybean_processed.npz 中的 X_train_scaled/X_val_scaled/X_test_scaled。",
            "标签编码顺序固定为: BARBUNYA=0, BOMBAY=1, CALI=2, DERMASON=3, HOROZ=4, SEKER=5, SIRA=6。",
        ],
    }

    print(f"[4/5] 保存结果到: {output_dir}")
    save_outputs(
        output_dir=output_dir,
        cleaned_dfs=cleaned_dfs,
        processed_dfs=processed_dfs,
        arrays=arrays,
        config=config,
        clean_report=clean_report,
    )

    print("[5/5] 完成。输出文件：")
    for p in sorted(output_dir.iterdir()):
        print(f"  - {p}")


if __name__ == "__main__":
    main()
