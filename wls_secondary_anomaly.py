# -*- coding: utf-8 -*-
"""
WLS 二次侧泵压差—转速机理异常检测脚本

适用数据结构：
root_dir/
  N/*.csv
  F01/*.csv, F02/*.csv, ...
  M/*.csv, S/*.csv, U/*.csv
  正常数据分类分工况/   # 自动忽略

核心逻辑：
1. 只使用 N 文件夹中的部分设备训练 WLS；
2. 机理关系：二次侧泵压差 ~ 1 + (二次侧泵转速 / speed_scale)^2；
3. 先 OLS 估计残差方差随转速变化，再用 1/sigma^2 做加权最小二乘；
4. 异常分数 = |残差| / 当前转速区间的残差标准差；
5. 阈值来自 N_train 异常分数分位数：0.9500, 0.9750, 0.9900, 0.9975, 0.9999；
6. N 作为正常，F* 作为异常，输出不同阈值下的误报率、召回率、准确率、精确率等；
7. M/S/U 不参与主指标，只输出预测结果和分组异常比例；
8. 同时输出 heldout 指标与 all_N_vs_F 自查指标。

运行示例：
python wls_secondary_anomaly.py --root_dir "D:/your_root" --output_dir "D:/wls_output"
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
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
# 1. 可按真实列名和工程定义微调的配置
# =========================
THRESHOLD_QUANTILES = [0.9500, 0.9750, 0.9900, 0.9975, 0.9999]
VALID_CODES = {"N", "M", "S", "U"}
IGNORE_DIR_NAMES = {"正常数据分类分工况"}

# 如果你的二次侧泵压差定义是“泵出口压力 - 泵入口压力”，改成 "outlet_minus_inlet"。
# 如果沿用之前讨论的“泵入口压力 - 泵出口压力”，保持 inlet_minus_outlet。
PUMP_DP_SIGN = "inlet_minus_outlet"


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
            return sorted(numbered, key=lambda x: len(_norm_name(x)))[0]
    return sorted(candidates, key=lambda x: len(_norm_name(x)))[0]


def find_existing_derived_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
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


def build_wls_features(df: pd.DataFrame, speed_scale: float) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    """
    构造 WLS 所需变量：
    y = 二次侧泵压差
    x = (二次侧泵转速 / speed_scale)^2
    condition = 控制压差目标值，用于可选分工况建模
    """
    mapping: Dict[str, Optional[str]] = {}

    speed_col = find_col(df, ["二次侧", "泵", "转速"], exclude=["一次侧"])
    control_col = find_col(df, ["控制", "压差"], exclude=["一次侧"])
    if control_col is None:
        control_col = find_col(df, ["目标", "压差"], exclude=["一次侧"])

    sec_in_1 = find_col(df, ["二次侧", "入口", "压力"], exclude=["泵", "一次侧", "温度"], prefer_number="1")
    sec_in_2 = find_col(df, ["二次侧", "入口", "压力"], exclude=["泵", "一次侧", "温度"], prefer_number="2")
    sec_out_1 = find_col(df, ["二次侧", "出口", "压力"], exclude=["泵", "一次侧", "温度"], prefer_number="1")
    sec_out_2 = find_col(df, ["二次侧", "出口", "压力"], exclude=["泵", "一次侧", "温度"], prefer_number="2")
    pump_in_1 = find_col(df, ["二次侧", "泵", "入口", "压力"], exclude=["一次侧", "温度"], prefer_number="1")
    pump_in_2 = find_col(df, ["二次侧", "泵", "入口", "压力"], exclude=["一次侧", "温度"], prefer_number="2")
    pump_out = find_col(df, ["二次侧", "泵", "出口", "压力"], exclude=["一次侧", "温度"])
    pump_dp_col = find_existing_derived_col(df, ["二次侧泵压差", "泵压差"])

    mapping.update(
        {
            "二次侧泵转速": speed_col,
            "控制压差目标值": control_col,
            "二次侧泵入口压力1": pump_in_1,
            "二次侧泵入口压力2": pump_in_2,
            "二次侧泵出口压力": pump_out,
            "二次侧泵压差": pump_dp_col,
            "二次侧入口压力1": sec_in_1,
            "二次侧入口压力2": sec_in_2,
            "二次侧出口压力1": sec_out_1,
            "二次侧出口压力2": sec_out_2,
        }
    )

    if speed_col is None:
        speed = pd.Series(np.nan, index=df.index)
    else:
        speed = _to_numeric(df[speed_col])

    if control_col is None:
        control = pd.Series(np.nan, index=df.index)
    else:
        control = _to_numeric(df[control_col])

    if pump_dp_col is not None:
        pump_dp = _to_numeric(df[pump_dp_col])
    else:
        pump_in = mean_of_available(df, [pump_in_1, pump_in_2])
        pump_out_s = _to_numeric(df[pump_out]) if pump_out is not None else None
        if pump_in is not None and pump_out_s is not None:
            if PUMP_DP_SIGN == "outlet_minus_inlet":
                pump_dp = pump_out_s - pump_in
            else:
                pump_dp = pump_in - pump_out_s
        else:
            pump_dp = pd.Series(np.nan, index=df.index)

    speed_norm_sq = (speed / speed_scale) ** 2
    out = pd.DataFrame(
        {
            "pump_dp": pump_dp,
            "speed": speed,
            "speed_norm_sq": speed_norm_sq,
            "control_dp_target": control,
        },
        index=df.index,
    )
    out = out.replace([np.inf, -np.inf], np.nan)
    return out, mapping


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
    speed_scale: float,
) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    parts: List[pd.DataFrame] = []
    first_mapping: Dict[str, Optional[str]] = {}
    for i, it in enumerate(train_items):
        df = read_csv_flexible(it.path)
        feats, mapping = build_wls_features(df, speed_scale=speed_scale)
        if not first_mapping:
            first_mapping = mapping
        feats = feats.dropna(axis=0, subset=["pump_dp", "speed_norm_sq"])
        feats = feats[np.isfinite(feats["pump_dp"]) & np.isfinite(feats["speed_norm_sq"])]
        feats = sample_rows(feats, max_rows_per_device, seed + i)
        if not feats.empty:
            parts.append(feats)
    if not parts:
        raise RuntimeError("N_train 中没有可用于 WLS 训练的有效行。请检查二次侧泵压差和转速列名。")
    X = pd.concat(parts, axis=0, ignore_index=True)
    X = sample_rows(X, max_total_rows, seed)
    return X, first_mapping


# =========================
# 4. WLS 模型
# =========================
def _design(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    return np.column_stack([np.ones_like(x), x])


def _safe_lstsq(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def _make_quantile_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.nanquantile(x, qs))
    if len(edges) < 3:
        xmin, xmax = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
            return np.array([-np.inf, np.inf])
        edges = np.linspace(xmin, xmax, min(n_bins, 3) + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _bin_index(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, x, side="right") - 1


@dataclass
class WLSSubModel:
    beta: np.ndarray
    sigma_edges: np.ndarray
    sigma_by_bin: np.ndarray
    global_sigma: float
    n_train: int
    r2: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return _design(x) @ self.beta

    def sigma(self, x: np.ndarray) -> np.ndarray:
        idx = _bin_index(np.asarray(x, dtype=float), self.sigma_edges)
        idx = np.clip(idx, 0, len(self.sigma_by_bin) - 1)
        sigma = self.sigma_by_bin[idx]
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, self.global_sigma)
        return sigma

    def score(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        yhat = self.predict(x)
        resid = np.asarray(y, dtype=float) - yhat
        sigma = self.sigma(x)
        score = np.abs(resid) / sigma
        return score, resid, yhat


def fit_wls_submodel(df: pd.DataFrame, n_bins: int, min_sigma: float) -> WLSSubModel:
    x = df["speed_norm_sq"].to_numpy(dtype=float)
    y = df["pump_dp"].to_numpy(dtype=float)
    X = _design(x)

    beta_ols = _safe_lstsq(X, y)
    resid_ols = y - X @ beta_ols
    global_sigma = float(np.nanstd(resid_ols, ddof=2))
    if not np.isfinite(global_sigma) or global_sigma < min_sigma:
        global_sigma = min_sigma

    edges = _make_quantile_bins(x, n_bins=n_bins)
    idx = _bin_index(x, edges)
    n_bin = len(edges) - 1
    sigma_by_bin = np.full(n_bin, global_sigma, dtype=float)
    for b in range(n_bin):
        r = resid_ols[idx == b]
        if len(r) >= 20:
            s = float(np.nanstd(r, ddof=2))
            if np.isfinite(s) and s >= min_sigma:
                sigma_by_bin[b] = s

    sig = sigma_by_bin[np.clip(idx, 0, n_bin - 1)]
    weights = 1.0 / np.maximum(sig, min_sigma) ** 2
    Xw = X * np.sqrt(weights[:, None])
    yw = y * np.sqrt(weights)
    beta_wls = _safe_lstsq(Xw, yw)

    resid = y - X @ beta_wls
    global_sigma2 = float(np.nanstd(resid, ddof=2))
    if not np.isfinite(global_sigma2) or global_sigma2 < min_sigma:
        global_sigma2 = global_sigma
    sigma_by_bin2 = np.full(n_bin, global_sigma2, dtype=float)
    for b in range(n_bin):
        r = resid[idx == b]
        if len(r) >= 20:
            s = float(np.nanstd(r, ddof=2))
            if np.isfinite(s) and s >= min_sigma:
                sigma_by_bin2[b] = s

    yhat = X @ beta_wls
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return WLSSubModel(beta=beta_wls, sigma_edges=edges, sigma_by_bin=sigma_by_bin2, global_sigma=global_sigma2, n_train=len(df), r2=float(r2))


def control_key(v: object, decimals: int = 3) -> str:
    try:
        fv = float(v)
        if np.isfinite(fv):
            return f"{round(fv, decimals):.{decimals}f}"
    except Exception:
        pass
    return "__MISSING__"


class WLSModel:
    def __init__(self, min_group_rows: int = 1000, n_bins: int = 10, min_sigma: float = 1e-6, use_condition_models: bool = True):
        self.min_group_rows = min_group_rows
        self.n_bins = n_bins
        self.min_sigma = min_sigma
        self.use_condition_models = use_condition_models
        self.global_model: Optional[WLSSubModel] = None
        self.group_models: Dict[str, WLSSubModel] = {}

    def fit(self, train_df: pd.DataFrame) -> None:
        train_df = train_df.dropna(axis=0, subset=["pump_dp", "speed_norm_sq"])
        self.global_model = fit_wls_submodel(train_df, n_bins=self.n_bins, min_sigma=self.min_sigma)
        self.group_models = {}
        if self.use_condition_models and "control_dp_target" in train_df.columns:
            tmp = train_df.copy()
            tmp["_ckey"] = tmp["control_dp_target"].apply(control_key)
            for key, g in tmp.groupby("_ckey"):
                if key == "__MISSING__":
                    continue
                if len(g) >= self.min_group_rows:
                    self.group_models[key] = fit_wls_submodel(g, n_bins=self.n_bins, min_sigma=self.min_sigma)

    def _choose_models(self, control_values: pd.Series) -> List[WLSSubModel]:
        if self.global_model is None:
            raise RuntimeError("WLSModel 尚未 fit。")
        models = []
        for v in control_values:
            key = control_key(v)
            models.append(self.group_models.get(key, self.global_model))
        return models

    def score_frame(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = df["speed_norm_sq"].to_numpy(dtype=float)
        y = df["pump_dp"].to_numpy(dtype=float)
        control = df.get("control_dp_target", pd.Series(np.nan, index=df.index))
        models = self._choose_models(control)
        score = np.empty(len(df), dtype=float)
        resid = np.empty(len(df), dtype=float)
        yhat = np.empty(len(df), dtype=float)
        sigma = np.empty(len(df), dtype=float)
        for i, m in enumerate(models):
            s, r, yh = m.score(np.array([x[i]]), np.array([y[i]]))
            score[i] = s[0]
            resid[i] = r[0]
            yhat[i] = yh[0]
            sigma[i] = m.sigma(np.array([x[i]]))[0]
        return score, resid, yhat, sigma

    def to_dict(self) -> Dict[str, object]:
        def sub_to_dict(m: WLSSubModel) -> Dict[str, object]:
            return {
                "beta_intercept": float(m.beta[0]),
                "beta_speed_norm_sq": float(m.beta[1]),
                "sigma_edges": [float(x) for x in m.sigma_edges],
                "sigma_by_bin": [float(x) for x in m.sigma_by_bin],
                "global_sigma": float(m.global_sigma),
                "n_train": int(m.n_train),
                "r2": float(m.r2) if np.isfinite(m.r2) else None,
            }
        return {
            "min_group_rows": self.min_group_rows,
            "n_bins": self.n_bins,
            "min_sigma": self.min_sigma,
            "use_condition_models": self.use_condition_models,
            "global_model": sub_to_dict(self.global_model) if self.global_model is not None else None,
            "group_models": {k: sub_to_dict(v) for k, v in self.group_models.items()},
        }


# =========================
# 5. 指标与输出
# =========================
def safe_metrics(y_true: np.ndarray, score: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    pred = np.asarray(pred).astype(int)
    score = np.asarray(score).astype(float)

    if len(y_true) == 0:
        return {}
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
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
    model: WLSModel,
    thresholds: Dict[float, float],
    output_dir: Path,
    speed_scale: float,
) -> None:
    record_path = output_dir / "wls_record_predictions.csv"
    device_rows: List[Dict[str, object]] = []
    code_rows: List[Dict[str, object]] = []

    scores_by_scope = {"heldout": [], "all_N_vs_F": []}
    labels_by_scope = {"heldout": [], "all_N_vs_F": []}
    preds_by_scope = {scope: {q: [] for q in thresholds} for scope in scores_by_scope}
    per_code_accumulator: Dict[Tuple[str, float], Dict[str, int]] = {}

    wrote_header = False
    for item_idx, it in enumerate(items):
        split = split_map[(it.code, it.path)]
        df = read_csv_flexible(it.path)
        time_col = get_time_col(df)
        feats, _ = build_wls_features(df, speed_scale=speed_scale)
        valid_mask = feats[["pump_dp", "speed_norm_sq"]].notna().all(axis=1)
        valid_mask &= np.isfinite(feats["pump_dp"]) & np.isfinite(feats["speed_norm_sq"])
        valid_idx = np.where(valid_mask.values)[0]

        scores = np.full(len(df), np.nan, dtype=float)
        residuals = np.full(len(df), np.nan, dtype=float)
        yhat = np.full(len(df), np.nan, dtype=float)
        sigmas = np.full(len(df), np.nan, dtype=float)
        if valid_mask.any():
            scores_valid, resid_valid, yhat_valid, sigma_valid = model.score_frame(feats.loc[valid_mask].reset_index(drop=True))
            scores[valid_idx] = scores_valid
            residuals[valid_idx] = resid_valid
            yhat[valid_idx] = yhat_valid
            sigmas[valid_idx] = sigma_valid
        else:
            scores_valid = np.array([], dtype=float)
            resid_valid = np.array([], dtype=float)
            yhat_valid = np.array([], dtype=float)
            sigma_valid = np.array([], dtype=float)

        label = 1 if it.code.startswith("F") else 0 if it.code == "N" else -1
        base = pd.DataFrame(
            {
                "source_code": it.code,
                "device_name": it.device,
                "split": split,
                "row_index": np.arange(len(df), dtype=int),
                "is_valid": valid_mask.values,
                "label": label,
                "pump_dp": feats["pump_dp"].values,
                "speed": feats["speed"].values,
                "speed_norm_sq": feats["speed_norm_sq"].values,
                "control_dp_target": feats["control_dp_target"].values,
                "wls_yhat": yhat,
                "wls_residual": residuals,
                "wls_sigma": sigmas,
                "wls_score": scores,
            }
        )
        if time_col is not None:
            base.insert(3, "time", df[time_col].astype(str).values)
        for q, th in thresholds.items():
            qname = format_q(q)
            base[f"wls_pred_{qname}"] = np.where(np.isfinite(scores), scores > th, False)
            base[f"wls_threshold_{qname}"] = th

        base.to_csv(record_path, mode="a", index=False, header=not wrote_header, encoding="utf-8-sig")
        wrote_header = True

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
                    "mean_abs_residual": float(np.nanmean(np.abs(resid_valid))) if len(resid_valid) else np.nan,
                }
            )

        if label in (0, 1) and len(scores_valid):
            y_valid = np.full(len(scores_valid), label, dtype=int)
            if not (it.code == "N" and split == "train"):
                scores_by_scope["heldout"].append(scores_valid)
                labels_by_scope["heldout"].append(y_valid)
                for q, th in thresholds.items():
                    preds_by_scope["heldout"][q].append(scores_valid > th)
            scores_by_scope["all_N_vs_F"].append(scores_valid)
            labels_by_scope["all_N_vs_F"].append(y_valid)
            for q, th in thresholds.items():
                preds_by_scope["all_N_vs_F"][q].append(scores_valid > th)

            for q, th in thresholds.items():
                key = (it.code, q)
                d = per_code_accumulator.setdefault(key, {"valid": 0, "pred_abnormal": 0, "label": label})
                d["valid"] += int(len(scores_valid))
                d["pred_abnormal"] += int((scores_valid > th).sum())

        print(f"[{item_idx + 1}/{len(items)}] scored: {it.code}/{it.path.name}, valid={int(valid_mask.sum())}/{len(df)}")

    metric_rows: List[Dict[str, object]] = []
    for scope in ["heldout", "all_N_vs_F"]:
        if not scores_by_scope[scope]:
            continue
        y = np.concatenate(labels_by_scope[scope]).astype(int)
        score = np.concatenate(scores_by_scope[scope]).astype(float)
        for q, th in thresholds.items():
            pred = np.concatenate(preds_by_scope[scope][q]).astype(int)
            row = {
                "model": "WLS",
                "eval_scope": scope,
                "threshold_quantile": q,
                "threshold_name": format_q(q),
                "threshold_value": th,
            }
            row.update(safe_metrics(y, score, pred))
            metric_rows.append(row)

    for (code, q), d in sorted(per_code_accumulator.items()):
        valid = d["valid"]
        pred_abn = d["pred_abnormal"]
        label = d["label"]
        code_rows.append(
            {
                "model": "WLS",
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

    pd.DataFrame(metric_rows).to_csv(output_dir / "wls_metrics_by_threshold.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(code_rows).to_csv(output_dir / "wls_metrics_by_code_threshold.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(device_rows).to_csv(output_dir / "wls_device_summary_by_threshold.csv", index=False, encoding="utf-8-sig")


# =========================
# 6. 主函数
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WLS 二次侧泵压差—转速异常检测")
    p.add_argument("--root_dir", type=str, required=True, help="根目录，下面包含 N、F01、F02、M、S、U 等文件夹")
    p.add_argument("--output_dir", type=str, required=True, help="输出目录")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n_train_frac", type=float, default=0.7, help="N 文件夹按设备划分训练集比例")
    p.add_argument("--max_train_rows_per_device", type=int, default=5000)
    p.add_argument("--max_train_rows_total", type=int, default=200000)
    p.add_argument("--speed_scale", type=float, default=100.0, help="转速归一化尺度；若转速是 0-100，保持 100")
    p.add_argument("--min_group_rows", type=int, default=1000, help="按控制压差分组建 WLS 的最小训练行数；不足则回退全局模型")
    p.add_argument("--n_bins", type=int, default=10, help="估计异方差残差标准差的转速分箱数")
    p.add_argument("--min_sigma", type=float, default=1e-6)
    p.add_argument("--no_condition_models", action="store_true", help="不按控制压差分组，只拟合一个全局 WLS")
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
        raise RuntimeError("未找到 N 文件夹 CSV，无法训练 WLS。")

    split_map = split_n_devices(items, train_frac=args.n_train_frac, seed=args.seed)
    train_items = [it for it in n_items if split_map[(it.code, it.path)] == "train"]

    print(f"发现 CSV 文件数：{len(items)}；N 训练设备文件数：{len(train_items)}；N 总文件数：{len(n_items)}")
    train_df, first_mapping = build_training_matrix(
        train_items,
        max_rows_per_device=args.max_train_rows_per_device,
        max_total_rows=args.max_train_rows_total,
        seed=args.seed,
        speed_scale=args.speed_scale,
    )
    print(f"WLS 训练样本数：{len(train_df)}")

    model = WLSModel(
        min_group_rows=args.min_group_rows,
        n_bins=args.n_bins,
        min_sigma=args.min_sigma,
        use_condition_models=not args.no_condition_models,
    )
    model.fit(train_df)

    train_scores, _, _, _ = model.score_frame(train_df.reset_index(drop=True))
    quantiles = [float(x.strip()) for x in args.threshold_quantiles.split(",") if x.strip()]
    thresholds = {q: float(np.nanquantile(train_scores, q)) for q in quantiles}

    with open(output_dir / "wls_config_and_columns.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "root_dir": str(root_dir),
                "first_file_column_mapping": first_mapping,
                "thresholds": {format_q(q): th for q, th in thresholds.items()},
                "speed_scale": args.speed_scale,
                "pump_dp_sign": PUMP_DP_SIGN,
                "model": model.to_dict(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    evaluate_and_write_outputs(
        items=items,
        split_map=split_map,
        model=model,
        thresholds=thresholds,
        output_dir=output_dir,
        speed_scale=args.speed_scale,
    )
    print(f"完成。输出目录：{output_dir}")


if __name__ == "__main__":
    main()
