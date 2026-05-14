# -*- coding: utf-8 -*-
"""
GMM 二次侧高维异常检测脚本

适用数据结构：
root_dir/
  N/*.csv
  F01/*.csv, F02/*.csv, ...
  M/*.csv, S/*.csv, U/*.csv
  正常数据分类分工况/   # 自动忽略

核心逻辑：
1. 只使用 N 文件夹中的部分设备训练 GMM；
2. 阈值来自 N_train 异常分数分位数：0.9500, 0.9750, 0.9900, 0.9975, 0.9999；
3. N 作为正常，F* 作为异常，分别输出不同阈值下的误报率、召回率、准确率、精确率等；
4. M/S/U 不参与主指标，只输出预测结果和分组异常比例；
5. 同时输出 heldout 指标与 all_N_vs_F 自查指标：
   - heldout：N_eval + 全部 F，较严谨；
   - all_N_vs_F：全部 N + 全部 F，适合交付前全量自查。

运行示例：
python gmm_secondary_anomaly.py --root_dir "D:/your_root" --output_dir "D:/gmm_output"
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =========================
# 1. 可按你的真实列名微调的配置
# =========================
THRESHOLD_QUANTILES = [0.9500, 0.9750, 0.9900, 0.9975, 0.9999]
VALID_CODES = {"N", "M", "S", "U"}
IGNORE_DIR_NAMES = {"正常数据分类分工况"}

# 如果你的二次侧泵压差定义是“泵出口压力 - 泵入口压力”，改成 "outlet_minus_inlet"。
# 如果沿用之前讨论的“泵入口压力 - 泵出口压力”，保持 inlet_minus_outlet。
PUMP_DP_SIGN = "inlet_minus_outlet"

# 管阻比如果没有现成列，脚本会用该公式临时构造。
# 可选："supply_return_over_pump" 或 "plate_hex_over_pump"
PIPE_RATIO_MODE = "supply_return_over_pump"

# 是否把入口/出口温度也放入 GMM。第一版建议 False。
INCLUDE_TEMPERATURE = False


# =========================
# 2. 列名识别与特征构造
# =========================
def _norm_name(s: object) -> str:
    s = str(s)
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"[\s_\-\.\[\]【】()]+", "", s)
    return s.lower()


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def find_col(
    df: pd.DataFrame,
    include: Sequence[str],
    exclude: Sequence[str] = (),
    prefer_number: Optional[str] = None,
) -> Optional[str]:
    """按包含/排除关键词寻找列名。prefer_number 用于优先找 1/2 号测点。"""
    include_n = [_norm_name(x) for x in include]
    exclude_n = [_norm_name(x) for x in exclude]
    candidates = []
    for col in df.columns:
        nc = _norm_name(col)
        if all(k in nc for k in include_n) and not any(k in nc for k in exclude_n):
            candidates.append(col)
    if not candidates:
        return None
    if prefer_number is not None:
        pn = _norm_name(prefer_number)
        numbered = [c for c in candidates if pn in _norm_name(c)]
        if numbered:
            # 更短的列名通常更接近目标列，避免误匹配到派生列
            return sorted(numbered, key=lambda x: len(_norm_name(x)))[0]
    return sorted(candidates, key=lambda x: len(_norm_name(x)))[0]


def find_existing_derived_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    """优先用完全匹配，其次用包含匹配。"""
    norm_map = {_norm_name(c): c for c in df.columns}
    for name in names:
        nn = _norm_name(name)
        if nn in norm_map:
            return norm_map[nn]
    for name in names:
        nn = _norm_name(name)
        hits = [c for c in df.columns if nn in _norm_name(c)]
        if hits:
            return sorted(hits, key=lambda x: len(_norm_name(x)))[0]
    return None


def mean_of_available(df: pd.DataFrame, cols: Sequence[Optional[str]]) -> Optional[pd.Series]:
    valid_cols = [c for c in cols if c is not None and c in df.columns]
    if not valid_cols:
        return None
    arr = pd.concat([_to_numeric(df[c]) for c in valid_cols], axis=1)
    return arr.mean(axis=1, skipna=True)


def get_time_col(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        nc = _norm_name(col)
        if any(k in nc for k in ["时间", "timestamp", "datetime", "date", "time"]):
            return col
    return None


def build_secondary_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    """
    构造二次侧特征。
    返回：features, column_mapping。
    """
    mapping: Dict[str, Optional[str]] = {}
    out: Dict[str, pd.Series] = {}

    # -------- 工况变量 --------
    speed_col = find_col(df, ["二次侧", "泵", "转速"], exclude=["一次侧"])
    valve_col = find_col(df, ["二次侧", "阀", "开度"], exclude=["一次侧"])
    control_col = find_col(df, ["控制", "压差"], exclude=["一次侧"])
    if control_col is None:
        control_col = find_col(df, ["目标", "压差"], exclude=["一次侧"])

    mapping["二次侧泵转速"] = speed_col
    mapping["二次侧阀开度"] = valve_col
    mapping["控制压差目标值"] = control_col

    if speed_col is not None:
        out["二次侧泵转速"] = _to_numeric(df[speed_col])
    if valve_col is not None:
        out["二次侧阀开度"] = _to_numeric(df[valve_col])
    if control_col is not None:
        out["控制压差目标值"] = _to_numeric(df[control_col])

    # -------- 原始压力测点，用于构造派生压差 --------
    sec_in_1 = find_col(df, ["二次侧", "入口", "压力"], exclude=["泵", "一次侧", "温度"], prefer_number="1")
    sec_in_2 = find_col(df, ["二次侧", "入口", "压力"], exclude=["泵", "一次侧", "温度"], prefer_number="2")
    sec_out_1 = find_col(df, ["二次侧", "出口", "压力"], exclude=["泵", "一次侧", "温度"], prefer_number="1")
    sec_out_2 = find_col(df, ["二次侧", "出口", "压力"], exclude=["泵", "一次侧", "温度"], prefer_number="2")
    pump_in_1 = find_col(df, ["二次侧", "泵", "入口", "压力"], exclude=["一次侧", "温度"], prefer_number="1")
    pump_in_2 = find_col(df, ["二次侧", "泵", "入口", "压力"], exclude=["一次侧", "温度"], prefer_number="2")
    pump_out = find_col(df, ["二次侧", "泵", "出口", "压力"], exclude=["一次侧", "温度"])

    mapping.update(
        {
            "二次侧入口压力1": sec_in_1,
            "二次侧入口压力2": sec_in_2,
            "二次侧出口压力1": sec_out_1,
            "二次侧出口压力2": sec_out_2,
            "二次侧泵入口压力1": pump_in_1,
            "二次侧泵入口压力2": pump_in_2,
            "二次侧泵出口压力": pump_out,
        }
    )

    sec_in = mean_of_available(df, [sec_in_1, sec_in_2])
    sec_out = mean_of_available(df, [sec_out_1, sec_out_2])
    pump_in = mean_of_available(df, [pump_in_1, pump_in_2])
    pump_out_s = _to_numeric(df[pump_out]) if pump_out is not None else None

    # -------- 优先读取已有派生特征；不存在时再计算 --------
    supply_return_col = find_existing_derived_col(df, ["二次侧供回水压差", "供回水压差"])
    plate_col = find_existing_derived_col(df, ["二次侧板换压差", "板换压差"])
    pump_dp_col = find_existing_derived_col(df, ["二次侧泵压差", "泵压差"])
    filter_col = find_existing_derived_col(df, ["二次侧过滤器压差", "过滤器压差"])
    pipe_ratio_col = find_existing_derived_col(df, ["二次侧管阻比", "管阻比"])

    mapping.update(
        {
            "二次侧供回水压差": supply_return_col,
            "二次侧板换压差": plate_col,
            "二次侧泵压差": pump_dp_col,
            "二次侧过滤器压差": filter_col,
            "二次侧管阻比": pipe_ratio_col,
        }
    )

    if supply_return_col is not None:
        supply_return = _to_numeric(df[supply_return_col])
    elif sec_in is not None and sec_out is not None:
        supply_return = sec_in - sec_out
    else:
        supply_return = None

    if plate_col is not None:
        plate_dp = _to_numeric(df[plate_col])
    elif sec_in is not None and pump_in is not None:
        plate_dp = sec_in - pump_in
    else:
        plate_dp = None

    if pump_dp_col is not None:
        pump_dp = _to_numeric(df[pump_dp_col])
    elif pump_in is not None and pump_out_s is not None:
        if PUMP_DP_SIGN == "outlet_minus_inlet":
            pump_dp = pump_out_s - pump_in
        else:
            pump_dp = pump_in - pump_out_s
    else:
        pump_dp = None

    if filter_col is not None:
        filter_dp = _to_numeric(df[filter_col])
    else:
        filter_dp = None

    if pipe_ratio_col is not None:
        pipe_ratio = _to_numeric(df[pipe_ratio_col])
    else:
        eps = 1e-9
        if PIPE_RATIO_MODE == "plate_hex_over_pump" and plate_dp is not None and pump_dp is not None:
            pipe_ratio = plate_dp / (pump_dp.replace(0, np.nan) + eps)
        elif supply_return is not None and pump_dp is not None:
            pipe_ratio = supply_return / (pump_dp.replace(0, np.nan) + eps)
        else:
            pipe_ratio = None

    derived = {
        "二次侧供回水压差": supply_return,
        "二次侧板换压差": plate_dp,
        "二次侧泵压差": pump_dp,
        "二次侧过滤器压差": filter_dp,
        "二次侧管阻比": pipe_ratio,
    }
    for name, s in derived.items():
        if s is not None:
            out[name] = s

    # -------- 可选温度 --------
    if INCLUDE_TEMPERATURE:
        temp_in_1 = find_col(df, ["二次侧", "入口", "温度"], exclude=["一次侧"], prefer_number="1")
        temp_in_2 = find_col(df, ["二次侧", "入口", "温度"], exclude=["一次侧"], prefer_number="2")
        temp_out_1 = find_col(df, ["二次侧", "出口", "温度"], exclude=["一次侧"], prefer_number="1")
        temp_out_2 = find_col(df, ["二次侧", "出口", "温度"], exclude=["一次侧"], prefer_number="2")
        temp_in = mean_of_available(df, [temp_in_1, temp_in_2])
        temp_out = mean_of_available(df, [temp_out_1, temp_out_2])
        if temp_in is not None:
            out["二次侧入口温度"] = temp_in
        if temp_out is not None:
            out["二次侧出口温度"] = temp_out
        mapping.update({"二次侧入口温度1": temp_in_1, "二次侧入口温度2": temp_in_2, "二次侧出口温度1": temp_out_1, "二次侧出口温度2": temp_out_2})

    features = pd.DataFrame(out, index=df.index)
    features = features.replace([np.inf, -np.inf], np.nan)
    return features, mapping


# =========================
# 3. 文件读取与划分
# =========================
def read_csv_flexible(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"无法读取 CSV：{path}；最后错误：{last_err}")


@dataclass(frozen=True)
class CsvItem:
    code: str
    device: str
    path: Path


def is_valid_code(name: str) -> bool:
    up = name.upper()
    if up in VALID_CODES:
        return True
    return bool(re.fullmatch(r"F\d+", up))


def collect_csv_items(root_dir: Path) -> List[CsvItem]:
    items: List[CsvItem] = []
    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name in IGNORE_DIR_NAMES:
            continue
        code = child.name.upper()
        if not is_valid_code(code):
            continue
        for csv_path in sorted(child.glob("*.csv")):
            items.append(CsvItem(code=code, device=csv_path.stem, path=csv_path))
    return items


def split_n_devices(items: List[CsvItem], train_frac: float, seed: int) -> Dict[Tuple[str, Path], str]:
    n_devices = sorted({it.device for it in items if it.code == "N"})
    rng = np.random.default_rng(seed)
    shuffled = n_devices.copy()
    rng.shuffle(shuffled)
    n_train = max(1, int(round(len(shuffled) * train_frac))) if shuffled else 0
    train_devices = set(shuffled[:n_train])
    split: Dict[Tuple[str, Path], str] = {}
    for it in items:
        if it.code == "N":
            split[(it.code, it.path)] = "train" if it.device in train_devices else "eval"
        else:
            split[(it.code, it.path)] = "eval"
    return split


def sample_rows(df: pd.DataFrame, max_rows: Optional[int], seed: int) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed)


def build_training_matrix(
    train_items: List[CsvItem],
    max_rows_per_device: int,
    max_total_rows: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    parts: List[pd.DataFrame] = []
    first_mapping: Dict[str, Optional[str]] = {}
    for i, it in enumerate(train_items):
        df = read_csv_flexible(it.path)
        feats, mapping = build_secondary_features(df)
        if not first_mapping:
            first_mapping = mapping
        feats = feats.dropna(axis=0, how="any")
        feats = sample_rows(feats, max_rows_per_device, seed + i)
        if not feats.empty:
            parts.append(feats)
    if not parts:
        raise RuntimeError("N_train 中没有可用于 GMM 训练的有效特征行。请检查列名识别和缺失值。")
    X = pd.concat(parts, axis=0, ignore_index=True)
    X = sample_rows(X, max_total_rows, seed)
    return X, first_mapping


# =========================
# 4. GMM 拟合、评分与指标
# =========================
def fit_gmm_auto(
    X_train: pd.DataFrame,
    n_components_candidates: Sequence[int],
    covariance_type: str,
    reg_covar: float,
    seed: int,
) -> Tuple[StandardScaler, GaussianMixture, int, pd.DataFrame]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train.values)

    rows = []
    best_gmm = None
    best_bic = math.inf
    best_k = None
    for k in n_components_candidates:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gmm = GaussianMixture(
                n_components=int(k),
                covariance_type=covariance_type,
                reg_covar=reg_covar,
                random_state=seed,
                n_init=3,
                max_iter=300,
            )
            gmm.fit(Xs)
        bic = gmm.bic(Xs)
        aic = gmm.aic(Xs)
        rows.append({"n_components": int(k), "bic": bic, "aic": aic, "converged": bool(gmm.converged_)})
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_k = int(k)

    assert best_gmm is not None and best_k is not None
    return scaler, best_gmm, best_k, pd.DataFrame(rows)


def calc_scores_gmm(feats: pd.DataFrame, feature_cols: Sequence[str], scaler: StandardScaler, gmm: GaussianMixture) -> np.ndarray:
    X = feats.loc[:, feature_cols].values
    Xs = scaler.transform(X)
    return -gmm.score_samples(Xs)


def safe_metrics(y_true: np.ndarray, score: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    pred = np.asarray(pred).astype(int)
    score = np.asarray(score).astype(float)

    if len(y_true) == 0:
        return {}
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=labels).ravel()

    out = {
        "n_valid": int(len(y_true)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan,
        "fpr": float(fp / (tn + fp)) if (tn + fp) > 0 else np.nan,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)) if len(np.unique(y_true)) > 1 else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)) if len(np.unique(y_true)) > 1 else np.nan,
    }
    if len(np.unique(y_true)) > 1:
        out["auc_roc"] = float(roc_auc_score(y_true, score))
        out["auc_pr"] = float(average_precision_score(y_true, score))
    else:
        out["auc_roc"] = np.nan
        out["auc_pr"] = np.nan
    return out


def max_consecutive_true(arr: np.ndarray) -> int:
    max_run = 0
    cur = 0
    for v in arr.astype(bool):
        if v:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return int(max_run)


def format_q(q: float) -> str:
    return f"q{int(round(q * 10000)):04d}"


def evaluate_and_write_outputs(
    items: List[CsvItem],
    split_map: Dict[Tuple[str, Path], str],
    feature_cols: Sequence[str],
    scaler: StandardScaler,
    gmm: GaussianMixture,
    thresholds: Dict[float, float],
    output_dir: Path,
) -> None:
    record_path = output_dir / "gmm_record_predictions.csv"
    device_rows: List[Dict[str, object]] = []
    code_rows: List[Dict[str, object]] = []

    # 用于总体指标。heldout 不含 N_train；all_N_vs_F 含全部 N。
    scores_by_scope = {"heldout": [], "all_N_vs_F": []}
    labels_by_scope = {"heldout": [], "all_N_vs_F": []}
    preds_by_scope = {scope: {q: [] for q in thresholds} for scope in scores_by_scope}

    per_code_accumulator: Dict[Tuple[str, float], Dict[str, int]] = {}

    wrote_header = False
    for item_idx, it in enumerate(items):
        split = split_map[(it.code, it.path)]
        df = read_csv_flexible(it.path)
        time_col = get_time_col(df)
        feats, _ = build_secondary_features(df)
        valid_mask = feats.loc[:, feature_cols].notna().all(axis=1)
        valid_idx = np.where(valid_mask.values)[0]
        scores = np.full(len(df), np.nan, dtype=float)
        if valid_mask.any():
            scores_valid = calc_scores_gmm(feats.loc[valid_mask, feature_cols], feature_cols, scaler, gmm)
            scores[valid_idx] = scores_valid
        else:
            scores_valid = np.array([], dtype=float)

        label = 1 if it.code.startswith("F") else 0 if it.code == "N" else -1
        base = pd.DataFrame(
            {
                "source_code": it.code,
                "device_name": it.device,
                "split": split,
                "row_index": np.arange(len(df), dtype=int),
                "is_valid": valid_mask.values,
                "label": label,
                "gmm_score": scores,
            }
        )
        if time_col is not None:
            base.insert(3, "time", df[time_col].astype(str).values)
        for q, th in thresholds.items():
            qname = format_q(q)
            base[f"gmm_pred_{qname}"] = np.where(np.isfinite(scores), scores > th, False)
            base[f"gmm_threshold_{qname}"] = th

        base.to_csv(record_path, mode="a", index=False, header=not wrote_header, encoding="utf-8-sig")
        wrote_header = True

        # 设备级汇总
        for q, th in thresholds.items():
            pred_valid = np.asarray(scores_valid > th, dtype=bool)
            device_rows.append(
                {
                    "source_code": it.code,
                    "device_name": it.device,
                    "split": split,
                    "threshold_quantile": q,
                    "threshold_name": format_q(q),
                    "threshold_value": th,
                    "total_records": int(len(df)),
                    "valid_records": int(valid_mask.sum()),
                    "abnormal_records": int(pred_valid.sum()),
                    "abnormal_rate_valid": float(pred_valid.mean()) if len(pred_valid) else np.nan,
                    "max_consecutive_abnormal": max_consecutive_true(pred_valid) if len(pred_valid) else 0,
                    "mean_score": float(np.nanmean(scores_valid)) if len(scores_valid) else np.nan,
                    "median_score": float(np.nanmedian(scores_valid)) if len(scores_valid) else np.nan,
                    "p95_score": float(np.nanquantile(scores_valid, 0.95)) if len(scores_valid) else np.nan,
                    "p99_score": float(np.nanquantile(scores_valid, 0.99)) if len(scores_valid) else np.nan,
                }
            )

        # 总体指标数据
        if label in (0, 1) and len(scores_valid):
            y_valid = np.full(len(scores_valid), label, dtype=int)
            # heldout：N_train 不参与，F 全部参与
            if not (it.code == "N" and split == "train"):
                scores_by_scope["heldout"].append(scores_valid)
                labels_by_scope["heldout"].append(y_valid)
                for q, th in thresholds.items():
                    preds_by_scope["heldout"][q].append(scores_valid > th)
            # all_N_vs_F：全部 N + 全部 F
            scores_by_scope["all_N_vs_F"].append(scores_valid)
            labels_by_scope["all_N_vs_F"].append(y_valid)
            for q, th in thresholds.items():
                preds_by_scope["all_N_vs_F"][q].append(scores_valid > th)

            # 分代码指标累计
            for q, th in thresholds.items():
                key = (it.code, q)
                d = per_code_accumulator.setdefault(key, {"valid": 0, "pred_abnormal": 0, "label": label})
                d["valid"] += int(len(scores_valid))
                d["pred_abnormal"] += int((scores_valid > th).sum())

        print(f"[{item_idx + 1}/{len(items)}] scored: {it.code}/{it.path.name}, valid={int(valid_mask.sum())}/{len(df)}")

    # 阈值总体指标
    metric_rows: List[Dict[str, object]] = []
    for scope in ["heldout", "all_N_vs_F"]:
        if not scores_by_scope[scope]:
            continue
        y = np.concatenate(labels_by_scope[scope]).astype(int)
        score = np.concatenate(scores_by_scope[scope]).astype(float)
        for q, th in thresholds.items():
            pred = np.concatenate(preds_by_scope[scope][q]).astype(int)
            row = {
                "model": "GMM",
                "eval_scope": scope,
                "threshold_quantile": q,
                "threshold_name": format_q(q),
                "threshold_value": th,
            }
            row.update(safe_metrics(y, score, pred))
            metric_rows.append(row)

    # 分代码汇总。N 是异常率/误报率；F 是召回率；M/S/U 是模型判异常比例。
    for (code, q), d in sorted(per_code_accumulator.items()):
        valid = d["valid"]
        pred_abn = d["pred_abnormal"]
        label = d["label"]
        code_rows.append(
            {
                "model": "GMM",
                "source_code": code,
                "threshold_quantile": q,
                "threshold_name": format_q(q),
                "valid_records": valid,
                "pred_abnormal_records": pred_abn,
                "pred_abnormal_rate": float(pred_abn / valid) if valid else np.nan,
                "meaning": "FPR_on_N" if code == "N" else "Recall_on_F" if code.startswith("F") else "Predicted_abnormal_rate_only",
                "label": label,
            }
        )

    pd.DataFrame(metric_rows).to_csv(output_dir / "gmm_metrics_by_threshold.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(code_rows).to_csv(output_dir / "gmm_metrics_by_code_threshold.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(device_rows).to_csv(output_dir / "gmm_device_summary_by_threshold.csv", index=False, encoding="utf-8-sig")


# =========================
# 5. 主函数
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GMM 二次侧异常检测")
    p.add_argument("--root_dir", type=str, required=True, help="根目录，下面包含 N、F01、F02、M、S、U 等文件夹")
    p.add_argument("--output_dir", type=str, required=True, help="输出目录")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n_train_frac", type=float, default=0.7, help="N 文件夹按设备划分训练集比例")
    p.add_argument("--max_train_rows_per_device", type=int, default=5000)
    p.add_argument("--max_train_rows_total", type=int, default=200000)
    p.add_argument("--n_components", type=str, default="1,2,3,4,5", help="GMM 成分数候选，如 '1,2,3,4,5'；脚本按 BIC 选择")
    p.add_argument("--covariance_type", type=str, default="full", choices=["full", "tied", "diag", "spherical"])
    p.add_argument("--reg_covar", type=float, default=1e-6)
    p.add_argument("--threshold_quantiles", type=str, default=",".join(str(x) for x in THRESHOLD_QUANTILES), help="逗号分隔，如 0.95,0.975,0.99,0.9975,0.9999")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = collect_csv_items(root_dir)
    if not items:
        raise RuntimeError(f"未在根目录中找到 N/F*/M/S/U 下的 csv：{root_dir}")
    n_items = [it for it in items if it.code == "N"]
    if not n_items:
        raise RuntimeError("未找到 N 文件夹 CSV，无法训练 GMM。")

    split_map = split_n_devices(items, train_frac=args.n_train_frac, seed=args.seed)
    train_items = [it for it in n_items if split_map[(it.code, it.path)] == "train"]

    print(f"发现 CSV 文件数：{len(items)}；N 训练设备文件数：{len(train_items)}；N 总文件数：{len(n_items)}")
    X_train, first_mapping = build_training_matrix(
        train_items,
        max_rows_per_device=args.max_train_rows_per_device,
        max_total_rows=args.max_train_rows_total,
        seed=args.seed,
    )
    feature_cols = list(X_train.columns)
    if len(feature_cols) < 2:
        raise RuntimeError(f"有效特征过少：{feature_cols}")
    print("GMM 使用特征：")
    for c in feature_cols:
        print(f"  - {c}")
    print(f"GMM 训练样本数：{len(X_train)}")

    n_components_candidates = [int(x.strip()) for x in args.n_components.split(",") if x.strip()]
    scaler, gmm, best_k, bic_table = fit_gmm_auto(
        X_train,
        n_components_candidates=n_components_candidates,
        covariance_type=args.covariance_type,
        reg_covar=args.reg_covar,
        seed=args.seed,
    )
    bic_table.to_csv(output_dir / "gmm_bic_table.csv", index=False, encoding="utf-8-sig")
    print(f"BIC 选择的 GMM 成分数：{best_k}")

    train_scores = calc_scores_gmm(X_train, feature_cols, scaler, gmm)
    quantiles = [float(x.strip()) for x in args.threshold_quantiles.split(",") if x.strip()]
    thresholds = {q: float(np.nanquantile(train_scores, q)) for q in quantiles}

    with open(output_dir / "gmm_config_and_columns.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "root_dir": str(root_dir),
                "feature_cols": feature_cols,
                "first_file_column_mapping": first_mapping,
                "thresholds": {format_q(q): th for q, th in thresholds.items()},
                "best_n_components": best_k,
                "pump_dp_sign": PUMP_DP_SIGN,
                "pipe_ratio_mode": PIPE_RATIO_MODE,
                "include_temperature": INCLUDE_TEMPERATURE,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    evaluate_and_write_outputs(
        items=items,
        split_map=split_map,
        feature_cols=feature_cols,
        scaler=scaler,
        gmm=gmm,
        thresholds=thresholds,
        output_dir=output_dir,
    )
    print(f"完成。输出目录：{output_dir}")


if __name__ == "__main__":
    main()
