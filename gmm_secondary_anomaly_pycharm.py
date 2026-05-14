# -*- coding: utf-8 -*-
"""
GMM 二次侧高维异常检测脚本（含丰富可视化）

数据结构：
root_dir/
  N/*.csv
  F01/*.csv, F02/*.csv, F03/*.csv, F04/*.csv, F06/*.csv
  M/*.csv, S/*.csv, U/*.csv
  正常数据分类分工况/  # 自动忽略

二次侧异常定义：
- F03、F04、F06：二次侧异常，作为异常类；
- N：正常类；
- F01、F02：一次侧异常。由于一次侧与二次侧相互独立，本脚本提供开关：
    --f01_f02_mode as_normal  把 F01/F02 作为二次侧正常样本，相当于并入 N；
    --f01_f02_mode exclude    直接剔除 F01/F02，不参与训练和评价。
- M/S/U：默认只打分和输出统计，不参与主二分类指标。

GMM 说明：
- GMM 不剔除高转速区间；
- 只用正常类中的训练设备拟合 GMM；
- 阈值来自训练正常样本异常分数的分位数：0.9500, 0.9750, 0.9900, 0.9975, 0.9999；
- 异常分数 score = -log p(x)，越大越异常。

运行示例：
python gmm_secondary_anomaly_visual.py --root_dir "D:/data_root" --output_dir "D:/gmm_output" --f01_f02_mode exclude
python gmm_secondary_anomaly_visual.py --root_dir "D:/data_root" --output_dir "D:/gmm_output_as_normal" --f01_f02_mode as_normal
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
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
    roc_curve,
    precision_recall_curve,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False


# =========================
# 常量配置
# =========================
DEFAULT_THRESHOLD_QUANTILES = [0.9500, 0.9750, 0.9900, 0.9975, 0.9999]
IGNORE_DIR_NAMES = {"正常数据分类分工况"}
SECONDARY_ANOMALY_CODES = {"F03", "F04", "F06"}
PRIMARY_ANOMALY_CODES = {"F01", "F02"}
UNLABELED_CODES = {"M", "S", "U"}
ALWAYS_READ_CODES = {"N"} | SECONDARY_ANOMALY_CODES | PRIMARY_ANOMALY_CODES | UNLABELED_CODES

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 减少无关 warning 对长任务日志的干扰；真实错误仍会抛出。
warnings.filterwarnings("ignore", message="Glyph .* missing from font.*")
warnings.filterwarnings("ignore", message="A single label was found.*")
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true.*")


# =========================
# 0. PyCharm 直接运行配置区：你主要改这里
# =========================
# 说明：
# 1）在 PyCharm 里直接点运行时，会使用下面这些默认配置；
# 2）如果你在 Run/Debug Configurations 的 Parameters 里又填写了 --root_dir 等参数，命令行参数会覆盖这里；
# 3）F01/F02 是一次侧异常。由于一次侧和二次侧相互独立，二次侧实验有两种处理方式：
#    - "exclude"：剔除 F01/F02，不参与训练、阈值和评价；
#    - "as_normal"：把 F01/F02 当作二次侧正常样本，相当于并入 N。
DATA_ROOT_DIR = r"D:\请把这里改成你的数据根目录"
OUTPUT_DIR = r"D:\gmm_secondary_output"
F01_F02_MODE = "exclude"       # 可选："exclude" 或 "as_normal"
SKIP_M_S_U = False             # False=读取 M/S/U 并打分，但不参与主指标；True=跳过 M/S/U
PUMP_DP_SIGN_CONFIG = "inlet_minus_outlet"  # 可选："inlet_minus_outlet" 或 "outlet_minus_inlet"
INCLUDE_TEMPERATURE_CONFIG = False
THRESHOLD_QUANTILES_CONFIG = "0.9500,0.9750,0.9900,0.9975,0.9999"


# =========================
# 列名识别与特征构造
# =========================
def _norm_name(s: object) -> str:
    s = str(s)
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"[\s_\-\.\[\]【】()（）:：/\\]+", "", s)
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
    candidates: List[str] = []
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


def build_secondary_features(
    df: pd.DataFrame,
    pump_dp_sign: str = "inlet_minus_outlet",
    include_temperature: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    """构造二次侧特征，并返回列名映射。"""
    mapping: Dict[str, Optional[str]] = {}
    out: Dict[str, pd.Series] = {}

    # 工况变量
    speed_col = find_col(df, ["二次侧", "泵", "转速"], exclude=["一次侧"])
    valve_col = find_col(df, ["二次侧", "阀", "开度"], exclude=["一次侧"])
    control_col = find_col(df, ["控制", "压差"], exclude=["一次侧"])
    if control_col is None:
        control_col = find_col(df, ["目标", "压差"], exclude=["一次侧"])

    mapping.update({"二次侧泵转速": speed_col, "二次侧阀开度": valve_col, "控制压差目标值": control_col})
    if speed_col is not None:
        out["二次侧泵转速"] = _to_numeric(df[speed_col])
    if valve_col is not None:
        out["二次侧阀开度"] = _to_numeric(df[valve_col])
    if control_col is not None:
        out["控制压差目标值"] = _to_numeric(df[control_col])

    # 原始压力测点：排除“压差”，避免误把衍生列当成原始压力列
    sec_in_1 = find_col(df, ["二次侧", "入口", "压力"], exclude=["泵", "一次侧", "温度", "压差"], prefer_number="1")
    sec_in_2 = find_col(df, ["二次侧", "入口", "压力"], exclude=["泵", "一次侧", "温度", "压差"], prefer_number="2")
    sec_out_1 = find_col(df, ["二次侧", "出口", "压力"], exclude=["泵", "一次侧", "温度", "压差"], prefer_number="1")
    sec_out_2 = find_col(df, ["二次侧", "出口", "压力"], exclude=["泵", "一次侧", "温度", "压差"], prefer_number="2")
    pump_in_1 = find_col(df, ["二次侧", "泵", "入口", "压力"], exclude=["一次侧", "温度", "压差"], prefer_number="1")
    pump_in_2 = find_col(df, ["二次侧", "泵", "入口", "压力"], exclude=["一次侧", "温度", "压差"], prefer_number="2")
    pump_out = find_col(df, ["二次侧", "泵", "出口", "压力"], exclude=["一次侧", "温度", "压差"])

    mapping.update({
        "二次侧入口压力1": sec_in_1,
        "二次侧入口压力2": sec_in_2,
        "二次侧出口压力1": sec_out_1,
        "二次侧出口压力2": sec_out_2,
        "二次侧泵入口压力1": pump_in_1,
        "二次侧泵入口压力2": pump_in_2,
        "二次侧泵出口压力": pump_out,
    })

    sec_in = mean_of_available(df, [sec_in_1, sec_in_2])
    sec_out = mean_of_available(df, [sec_out_1, sec_out_2])
    pump_in = mean_of_available(df, [pump_in_1, pump_in_2])
    pump_out_s = _to_numeric(df[pump_out]) if pump_out is not None else None

    # 派生特征：优先使用已有列，不存在时按原始测点计算
    supply_return_col = find_existing_derived_col(df, ["二次侧供回水压差", "供回水压差"])
    plate_col = find_existing_derived_col(df, ["二次侧板换压差", "板换压差"])
    pump_dp_col = find_existing_derived_col(df, ["二次侧泵压差", "泵压差"])
    filter_col = find_existing_derived_col(df, ["二次侧过滤器压差", "过滤器压差"])
    pipe_ratio_col = find_existing_derived_col(df, ["二次侧管阻比", "管阻比"])

    mapping.update({
        "二次侧供回水压差": supply_return_col,
        "二次侧板换压差": plate_col,
        "二次侧泵压差": pump_dp_col,
        "二次侧过滤器压差": filter_col,
        "二次侧管阻比": pipe_ratio_col,
    })

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
        if pump_dp_sign == "outlet_minus_inlet":
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
    elif supply_return is not None and pump_dp is not None:
        pipe_ratio = supply_return / (pump_dp.abs() + 1e-6)
    else:
        pipe_ratio = None

    for name, s in [
        ("二次侧供回水压差", supply_return),
        ("二次侧板换压差", plate_dp),
        ("二次侧泵压差", pump_dp),
        ("二次侧过滤器压差", filter_dp),
        ("二次侧管阻比", pipe_ratio),
    ]:
        if s is not None:
            out[name] = s

    if include_temperature:
        tin = find_col(df, ["二次侧", "入口", "温度"], exclude=["一次侧"])
        tout = find_col(df, ["二次侧", "出口", "温度"], exclude=["一次侧"])
        mapping.update({"二次侧入口温度": tin, "二次侧出口温度": tout})
        if tin is not None:
            out["二次侧入口温度"] = _to_numeric(df[tin])
        if tout is not None:
            out["二次侧出口温度"] = _to_numeric(df[tout])

    return pd.DataFrame(out, index=df.index), mapping


# =========================
# 数据读取与标签处理
# =========================
def read_csv_auto(path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, nrows=max_rows)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"无法读取 CSV：{path}；最后错误：{last_err}")


def code_to_label(code: str, f01_f02_mode: str) -> Optional[int]:
    code = code.upper()
    if code == "N":
        return 0
    if code in SECONDARY_ANOMALY_CODES:
        return 1
    if code in PRIMARY_ANOMALY_CODES:
        if f01_f02_mode == "as_normal":
            return 0
        return None
    return None


def load_dataset(args: argparse.Namespace) -> Tuple[pd.DataFrame, List[Dict[str, Optional[str]]]]:
    root = Path(args.root_dir)
    rows: List[pd.DataFrame] = []
    mappings: List[Dict[str, Optional[str]]] = []

    if not root.exists():
        raise FileNotFoundError(f"root_dir 不存在：{root}")

    code_dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and p.name not in IGNORE_DIR_NAMES]
    for code_dir in code_dirs:
        code = code_dir.name.upper()
        if not (code == "N" or code in SECONDARY_ANOMALY_CODES or code in PRIMARY_ANOMALY_CODES or code in UNLABELED_CODES or code.startswith("F")):
            continue
        if code in PRIMARY_ANOMALY_CODES and args.f01_f02_mode == "exclude":
            print(f"[INFO] 跳过 {code}：f01_f02_mode=exclude")
            continue
        # 非二次侧 F，例如额外 Fxx，如果不是 F03/F04/F06/F01/F02，默认跳过，避免标签含义不清
        if code.startswith("F") and code not in SECONDARY_ANOMALY_CODES and code not in PRIMARY_ANOMALY_CODES:
            print(f"[WARN] 跳过未定义 F 文件夹 {code}：请确认是否属于二次侧异常")
            continue
        if code in UNLABELED_CODES and args.skip_unlabeled:
            continue

        csv_files = sorted(code_dir.glob("*.csv"))
        print(f"[INFO] 读取 {code}: {len(csv_files)} 个 CSV")
        for fp in csv_files:
            try:
                raw = read_csv_auto(fp, max_rows=args.max_rows_per_file)
                feats, mapping = build_secondary_features(raw, args.pump_dp_sign, args.include_temperature)
                if feats.empty:
                    print(f"[WARN] 未识别到二次侧特征，跳过：{fp}")
                    continue
                time_col = get_time_col(raw)
                meta = pd.DataFrame({
                    "source_code": code,
                    "device_name": fp.stem,
                    "file_path": str(fp),
                    "row_id": np.arange(len(raw), dtype=int),
                }, index=raw.index)
                if time_col is not None:
                    meta["timestamp"] = raw[time_col].astype(str)
                else:
                    meta["timestamp"] = ""
                meta["binary_label"] = code_to_label(code, args.f01_f02_mode)
                meta["label_name"] = np.where(meta["binary_label"].eq(0), "normal", np.where(meta["binary_label"].eq(1), "secondary_anomaly", "unlabeled"))
                meta["device_key"] = meta["source_code"].astype(str) + "/" + meta["device_name"].astype(str)
                rows.append(pd.concat([meta.reset_index(drop=True), feats.reset_index(drop=True)], axis=1))

                mapping_row = {"source_code": code, "device_name": fp.stem, "file_path": str(fp)}
                mapping_row.update(mapping)
                mappings.append(mapping_row)
            except Exception as e:
                print(f"[ERROR] 读取失败：{fp}；错误：{e}")

    if not rows:
        raise RuntimeError("没有读取到任何可用 CSV。请检查 root_dir 和列名。")
    data = pd.concat(rows, ignore_index=True)
    return data, mappings


def parse_quantiles(s: str) -> List[float]:
    vals = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    return vals


def choose_feature_columns(data: pd.DataFrame, args: argparse.Namespace) -> List[str]:
    candidate_cols = [
        "二次侧供回水压差",
        "二次侧板换压差",
        "二次侧泵压差",
        "二次侧过滤器压差",
        "二次侧管阻比",
        "二次侧泵转速",
        "二次侧阀开度",
        "控制压差目标值",
    ]
    if args.include_temperature:
        candidate_cols += ["二次侧入口温度", "二次侧出口温度"]
    candidate_cols = [c for c in candidate_cols if c in data.columns]

    normal = data[data["binary_label"].eq(0)]
    keep = []
    for c in candidate_cols:
        valid_rate = normal[c].notna().mean() if len(normal) else 0.0
        nunique = normal[c].nunique(dropna=True)
        if valid_rate >= args.min_feature_valid_rate and nunique >= 2:
            keep.append(c)
        else:
            print(f"[WARN] 特征 {c} 在正常样本中有效率 {valid_rate:.3f}，唯一值 {nunique}，不进入 GMM")
    if len(keep) < 2:
        raise RuntimeError(f"可用特征少于 2 个：{keep}。请检查列名识别或降低 --min_feature_valid_rate。")
    return keep


def split_normal_devices(data: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    rng = np.random.default_rng(args.random_state)
    normal_keys = np.array(sorted(data.loc[data["binary_label"].eq(0), "device_key"].dropna().unique()))
    if len(normal_keys) < 2:
        raise RuntimeError("正常设备数量少于 2，无法做设备级训练/留出划分。")
    rng.shuffle(normal_keys)
    n_train = max(1, int(round(len(normal_keys) * args.train_normal_frac)))
    n_train = min(n_train, len(normal_keys) - 1)
    train_keys = set(normal_keys[:n_train])
    split = pd.Series("unlabeled", index=data.index, dtype="object")
    split.loc[data["binary_label"].eq(1)] = "anomaly"
    split.loc[data["binary_label"].eq(0) & data["device_key"].isin(train_keys)] = "normal_train"
    split.loc[data["binary_label"].eq(0) & ~data["device_key"].isin(train_keys)] = "normal_heldout"
    return split


def sample_rows(df: pd.DataFrame, n: int, random_state: int, stratify_col: Optional[str] = None) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    if stratify_col is None or stratify_col not in df.columns:
        return df.sample(n=n, random_state=random_state)
    parts = []
    counts = df[stratify_col].fillna("NA").value_counts()
    for val, cnt in counts.items():
        take = max(1, int(round(n * cnt / len(df))))
        sub = df[df[stratify_col].fillna("NA").eq(val)]
        parts.append(sub.sample(n=min(take, len(sub)), random_state=random_state))
    out = pd.concat(parts, axis=0)
    if len(out) > n:
        out = out.sample(n=n, random_state=random_state)
    return out


# =========================
# GMM 建模与评分
# =========================
def fit_gmm_model(data: pd.DataFrame, feature_cols: List[str], args: argparse.Namespace):
    train_mask = data["split"].eq("normal_train")
    valid_mask = train_mask & data[feature_cols].notna().all(axis=1)
    train = data.loc[valid_mask, feature_cols].copy()
    if len(train) == 0:
        raise RuntimeError("GMM 训练集为空：请检查 N 文件夹和特征缺失情况。")
    if args.max_train_rows and len(train) > args.max_train_rows:
        train = train.sample(n=args.max_train_rows, random_state=args.random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train.values)

    bic_rows = []
    best_model = None
    best_bic = np.inf
    if args.n_components > 0:
        candidate_components = [args.n_components]
    else:
        candidate_components = list(range(args.n_components_min, args.n_components_max + 1))
    for k in candidate_components:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GaussianMixture(
                n_components=k,
                covariance_type=args.covariance_type,
                reg_covar=args.reg_covar,
                random_state=args.random_state,
                max_iter=args.max_iter,
                n_init=args.n_init,
            )
            model.fit(X_train)
            bic = model.bic(X_train)
            aic = model.aic(X_train)
        bic_rows.append({"n_components": k, "bic": bic, "aic": aic})
        if bic < best_bic:
            best_bic = bic
            best_model = model
    assert best_model is not None
    return scaler, best_model, pd.DataFrame(bic_rows), X_train


def score_gmm(data: pd.DataFrame, feature_cols: List[str], scaler: StandardScaler, model: GaussianMixture) -> pd.DataFrame:
    data = data.copy()
    data["valid_for_gmm"] = data[feature_cols].notna().all(axis=1)
    data["gmm_score"] = np.nan
    data["gmm_component"] = np.nan
    valid_idx = data.index[data["valid_for_gmm"]]
    if len(valid_idx):
        X = scaler.transform(data.loc[valid_idx, feature_cols].values)
        data.loc[valid_idx, "gmm_score"] = -model.score_samples(X)
        data.loc[valid_idx, "gmm_component"] = model.predict(X)
    return data


def compute_thresholds(data: pd.DataFrame, quantiles: List[float]) -> Dict[float, float]:
    train_scores = data.loc[data["split"].eq("normal_train") & data["gmm_score"].notna(), "gmm_score"]
    if len(train_scores) == 0:
        raise RuntimeError("无法计算阈值：normal_train 中没有有效 GMM score。")
    return {q: float(np.quantile(train_scores, q)) for q in quantiles}


# =========================
# 指标计算
# =========================
def safe_auc(y_true: np.ndarray, score: np.ndarray, kind: str) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        if kind == "roc":
            return float(roc_auc_score(y_true, score))
        return float(average_precision_score(y_true, score))
    except Exception:
        return np.nan


def metrics_for_df(df: pd.DataFrame, threshold: float, score_col: str = "gmm_score") -> Dict[str, float]:
    d = df[df["binary_label"].notna() & df[score_col].notna()].copy()
    if d.empty:
        return {k: np.nan for k in ["n", "tp", "fp", "tn", "fn", "fpr", "recall", "precision", "accuracy", "f1", "mcc", "balanced_accuracy", "auc_roc", "auc_pr"]}
    y_true = d["binary_label"].astype(int).values
    score = d[score_col].values
    y_pred = (score >= threshold).astype(int)
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    return {
        "n": int(len(d)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "fpr": float(fpr),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 or len(np.unique(y_true)) > 1 else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auc_roc": safe_auc(y_true, score, "roc"),
        "auc_pr": safe_auc(y_true, score, "pr"),
    }


def max_consecutive_ones(arr: Sequence[int]) -> int:
    best = cur = 0
    for v in arr:
        if int(v) == 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def build_metric_tables(data: pd.DataFrame, thresholds: Dict[float, float]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = data.copy()
    metric_rows = []
    code_rows = []
    device_rows = []

    scope_masks = {
        "heldout": records["split"].isin(["normal_heldout", "anomaly"]),
        "all_labeled": records["binary_label"].notna(),
    }

    for q, thr in thresholds.items():
        pred_col = f"gmm_pred_q{int(round(q * 10000)):04d}"
        records[pred_col] = np.where(records["gmm_score"].notna(), (records["gmm_score"] >= thr).astype(int), np.nan)
        for scope, mask in scope_masks.items():
            m = metrics_for_df(records.loc[mask], thr, score_col="gmm_score")
            metric_rows.append({"eval_scope": scope, "threshold_quantile": q, "threshold": thr, **m})

        for code, sub in records.groupby("source_code", dropna=False):
            valid = sub[sub["gmm_score"].notna()]
            pred = valid[pred_col].dropna().astype(int) if pred_col in valid else pd.Series(dtype=int)
            abnormal_count = int(pred.sum()) if len(pred) else 0
            total_valid = int(len(valid))
            row = {
                "threshold_quantile": q,
                "threshold": thr,
                "source_code": code,
                "total_records": int(len(sub)),
                "valid_records": total_valid,
                "pred_abnormal_count": abnormal_count,
                "pred_abnormal_rate": abnormal_count / total_valid if total_valid else np.nan,
                "label_meaning": "normal" if sub["binary_label"].eq(0).any() and not sub["binary_label"].eq(1).any() else ("secondary_anomaly" if sub["binary_label"].eq(1).any() else "unlabeled"),
            }
            if sub["binary_label"].notna().any():
                m_code = metrics_for_df(sub, thr, score_col="gmm_score")
                row.update({f"metric_{k}": v for k, v in m_code.items()})
            code_rows.append(row)

        for (code, dev), sub in records.groupby(["source_code", "device_name"], dropna=False):
            s = sub.sort_values("row_id")
            pred = s[pred_col].dropna().astype(int)
            valid_count = int(s["gmm_score"].notna().sum())
            abnormal_count = int(pred.sum()) if len(pred) else 0
            device_rows.append({
                "threshold_quantile": q,
                "threshold": thr,
                "source_code": code,
                "device_name": dev,
                "device_key": f"{code}/{dev}",
                "label_name": s["label_name"].iloc[0] if len(s) else "",
                "total_records": int(len(s)),
                "valid_records": valid_count,
                "pred_abnormal_count": abnormal_count,
                "pred_abnormal_rate": abnormal_count / valid_count if valid_count else np.nan,
                "max_consecutive_abnormal": max_consecutive_ones(pred.values) if len(pred) else 0,
                "mean_score": float(s["gmm_score"].mean(skipna=True)) if valid_count else np.nan,
                "median_score": float(s["gmm_score"].median(skipna=True)) if valid_count else np.nan,
                "p95_score": float(s["gmm_score"].quantile(0.95)) if valid_count else np.nan,
                "p99_score": float(s["gmm_score"].quantile(0.99)) if valid_count else np.nan,
            })

    return records, pd.DataFrame(metric_rows), pd.DataFrame(code_rows), pd.DataFrame(device_rows)


# =========================
# 可视化工具
# =========================
def count_text(df: pd.DataFrame, score_col: str = "gmm_score") -> str:
    total = len(df)
    valid = int(df[score_col].notna().sum()) if score_col in df.columns else total
    normal = int(df["binary_label"].eq(0).sum()) if "binary_label" in df.columns else 0
    anomaly = int(df["binary_label"].eq(1).sum()) if "binary_label" in df.columns else 0
    unlabeled = int(df["binary_label"].isna().sum()) if "binary_label" in df.columns else 0
    pct = lambda n: 100 * n / total if total else 0
    return f"全部 {total:,}；有效 {valid:,}；正常 {normal:,} ({pct(normal):.1f}%)；异常 {anomaly:,} ({pct(anomaly):.1f}%)；未标注 {unlabeled:,} ({pct(unlabeled):.1f}%)"


def add_count_box(ax, text: str) -> None:
    ax.text(
        0.01, 0.99, text,
        transform=ax.transAxes,
        va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
    )


def save_threshold_metric_plots(metrics_df: pd.DataFrame, fig_dir: Path) -> None:
    for scope, sub in metrics_df.groupby("eval_scope"):
        sub = sub.sort_values("threshold_quantile")
        # 主要指标随阈值变化
        fig, ax = plt.subplots(figsize=(10, 6))
        for metric in ["fpr", "recall", "precision", "accuracy", "f1", "mcc", "balanced_accuracy"]:
            if metric in sub.columns:
                ax.plot(sub["threshold_quantile"], sub[metric], marker="o", label=metric)
        ax.axhline(0.05, linestyle="--", linewidth=1, label="FPR=0.05")
        ax.axhline(0.50, linestyle=":", linewidth=1, label="Recall=0.50")
        ax.set_xlabel("阈值分位数（来自训练正常样本 score）")
        ax.set_ylabel("指标值")
        ax.set_title(f"GMM 不同阈值下指标变化 - {scope}")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / f"gmm_metrics_vs_threshold_{scope}.png", dpi=200)
        plt.close(fig)

        # FPR-Recall 权衡图
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(sub["fpr"], sub["recall"], s=80)
        for _, r in sub.iterrows():
            ax.annotate(f"{r['threshold_quantile']:.4f}", (r["fpr"], r["recall"]), textcoords="offset points", xytext=(5, 5), fontsize=9)
        ax.axvline(0.05, linestyle="--", linewidth=1, label="FPR=0.05")
        ax.axhline(0.50, linestyle=":", linewidth=1, label="Recall=0.50")
        ax.set_xlabel("误报率 FPR")
        ax.set_ylabel("召回率 Recall")
        ax.set_title(f"GMM 误报率-召回率权衡 - {scope}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"gmm_fpr_recall_tradeoff_{scope}.png", dpi=200)
        plt.close(fig)


def save_score_distribution_plots(data: pd.DataFrame, thresholds: Dict[float, float], fig_dir: Path) -> None:
    d = data[data["gmm_score"].notna()].copy()
    if d.empty:
        return
    # 合并 N vs F 的分布
    fig, ax = plt.subplots(figsize=(10, 6))
    normal = d[d["binary_label"].eq(0)]["gmm_score"]
    anomaly = d[d["binary_label"].eq(1)]["gmm_score"]
    ax.hist(normal, bins=80, alpha=0.55, density=True, label=f"正常 n={len(normal):,}")
    ax.hist(anomaly, bins=80, alpha=0.55, density=True, label=f"二次侧异常 n={len(anomaly):,}")
    for q, thr in thresholds.items():
        ax.axvline(thr, linestyle="--", linewidth=1, label=f"q={q:.4f}")
    ax.set_xlabel("GMM 异常分数 score=-log p(x)")
    ax.set_ylabel("密度")
    ax.set_title("GMM 异常分数分布：正常 vs 二次侧异常")
    add_count_box(ax, count_text(d))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "gmm_score_distribution_normal_vs_anomaly.png", dpi=200)
    plt.close(fig)

    # 按 source_code 的分布
    codes = sorted(d["source_code"].unique())
    for code in codes:
        sub = d[d["source_code"].eq(code)]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(sub["gmm_score"], bins=80, alpha=0.7, density=False)
        for q, thr in thresholds.items():
            ax.axvline(thr, linestyle="--", linewidth=1, label=f"q={q:.4f}")
        ax.set_xlabel("GMM 异常分数")
        ax.set_ylabel("记录数")
        ax.set_title(f"GMM 异常分数分布 - {code}")
        add_count_box(ax, count_text(sub))
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"gmm_score_distribution_{code}.png", dpi=200)
        plt.close(fig)


def save_roc_pr_plots(data: pd.DataFrame, fig_dir: Path) -> None:
    for scope, mask in {
        "heldout": data["split"].isin(["normal_heldout", "anomaly"]),
        "all_labeled": data["binary_label"].notna(),
    }.items():
        d = data.loc[mask & data["binary_label"].notna() & data["gmm_score"].notna()].copy()
        if d.empty or d["binary_label"].nunique() < 2:
            continue
        y = d["binary_label"].astype(int).values
        s = d["gmm_score"].values
        fpr, tpr, _ = roc_curve(y, s)
        auc_roc = roc_auc_score(y, s)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, label=f"AUC={auc_roc:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_xlabel("FPR")
        ax.set_ylabel("Recall / TPR")
        ax.set_title(f"GMM ROC 曲线 - {scope}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"gmm_roc_{scope}.png", dpi=200)
        plt.close(fig)

        precision, recall, _ = precision_recall_curve(y, s)
        auc_pr = average_precision_score(y, s)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(recall, precision, label=f"AP={auc_pr:.4f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"GMM PR 曲线 - {scope}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"gmm_pr_{scope}.png", dpi=200)
        plt.close(fig)


def save_device_plots(device_df: pd.DataFrame, fig_dir: Path, html_dir: Path, main_quantile: float) -> None:
    if device_df.empty:
        return
    sub = device_df[np.isclose(device_df["threshold_quantile"], main_quantile)].copy()
    if sub.empty:
        q = sorted(device_df["threshold_quantile"].unique())[-1]
        sub = device_df[device_df["threshold_quantile"].eq(q)].copy()
        main_quantile = q
    sub = sub.sort_values("pred_abnormal_rate", ascending=False)
    top = sub.head(30).copy()
    if not top.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(top["device_key"].astype(str)[::-1], top["pred_abnormal_rate"].values[::-1])
        ax.set_xlabel("设备异常率")
        ax.set_title(f"GMM Top 30 设备异常率（q={main_quantile:.4f}）")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"gmm_top30_device_abnormal_rate_q{int(main_quantile*10000):04d}.png", dpi=200)
        plt.close(fig)

    topc = sub.sort_values("max_consecutive_abnormal", ascending=False).head(30)
    if not topc.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(topc["device_key"].astype(str)[::-1], topc["max_consecutive_abnormal"].values[::-1])
        ax.axvline(10, linestyle="--", linewidth=1, label="连续误报 10 次参考线")
        ax.set_xlabel("最大连续判异常次数")
        ax.set_title(f"GMM Top 30 最大连续异常次数（q={main_quantile:.4f}）")
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"gmm_top30_device_max_consecutive_q{int(main_quantile*10000):04d}.png", dpi=200)
        plt.close(fig)

    # 按文件夹的设备异常率箱线图
    fig, ax = plt.subplots(figsize=(10, 6))
    groups = [g["pred_abnormal_rate"].dropna().values for _, g in sub.groupby("source_code")]
    labels = [str(k) for k, _ in sub.groupby("source_code")]
    if groups:
        ax.boxplot(groups, labels=labels, showfliers=False)
        ax.set_ylabel("设备异常率")
        ax.set_title(f"GMM 各文件夹设备异常率分布（q={main_quantile:.4f}）")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"gmm_device_abnormal_rate_box_by_code_q{int(main_quantile*10000):04d}.png", dpi=200)
    plt.close(fig)

    if PLOTLY_AVAILABLE and not sub.empty:
        top_html = sub.head(200).copy()
        fig = px.bar(
            top_html.sort_values("pred_abnormal_rate"),
            x="pred_abnormal_rate", y="device_key", color="source_code", orientation="h",
            hover_data=["valid_records", "pred_abnormal_count", "max_consecutive_abnormal", "mean_score", "p95_score"],
            title=f"GMM 设备异常率交互图 Top 200（q={main_quantile:.4f}）",
        )
        fig.write_html(html_dir / f"gmm_device_abnormal_rate_top200_q{int(main_quantile*10000):04d}.html")


def save_bic_plot(bic_df: pd.DataFrame, fig_dir: Path) -> None:
    if bic_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bic_df["n_components"], bic_df["bic"], marker="o", label="BIC")
    ax.plot(bic_df["n_components"], bic_df["aic"], marker="o", label="AIC")
    ax.set_xlabel("GMM 成分数")
    ax.set_ylabel("信息准则")
    ax.set_title("GMM 成分数选择：AIC/BIC")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "gmm_aic_bic_by_components.png", dpi=200)
    plt.close(fig)


def save_pca_scatter_plots(
    data: pd.DataFrame,
    feature_cols: List[str],
    scaler: StandardScaler,
    thresholds: Dict[float, float],
    args: argparse.Namespace,
    fig_dir: Path,
    html_dir: Path,
) -> None:
    dvalid = data[data["valid_for_gmm"]].copy()
    if len(dvalid) < 3:
        return
    plot_df = sample_rows(dvalid, args.plot_sample_size, args.random_state, stratify_col="source_code")
    X = scaler.transform(plot_df[feature_cols].values)
    pca = PCA(n_components=2, random_state=args.random_state)
    coords = pca.fit_transform(X)
    plot_df["PC1"] = coords[:, 0]
    plot_df["PC2"] = coords[:, 1]

    main_q = args.main_plot_quantile
    if main_q not in thresholds:
        main_q = sorted(thresholds.keys())[-1]
    thr = thresholds[main_q]
    plot_df["判定"] = np.where(plot_df["gmm_score"] >= thr, "判异常", "判正常")
    plot_df["真实类别"] = np.where(plot_df["binary_label"].eq(0), "正常", np.where(plot_df["binary_label"].eq(1), "二次侧异常", "未标注"))

    # 真实类别散点图
    fig, ax = plt.subplots(figsize=(10, 7))
    for label, sub in plot_df.groupby("真实类别"):
        ax.scatter(sub["PC1"], sub["PC2"], s=6, alpha=0.45, label=f"{label} plotted={len(sub):,}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("GMM PCA 二维散点：正常/异常/未标注分布")
    add_count_box(ax, count_text(dvalid) + f"\n实际绘制 {len(plot_df):,} 点")
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "gmm_pca_scatter_true_label.png", dpi=220)
    plt.close(fig)

    # 异常分数散点图
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(plot_df["PC1"], plot_df["PC2"], c=plot_df["gmm_score"], s=6, alpha=0.55)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("GMM 异常分数")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("GMM PCA 二维散点：异常分数颜色映射")
    add_count_box(ax, count_text(dvalid) + f"\n实际绘制 {len(plot_df):,} 点")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "gmm_pca_scatter_score.png", dpi=220)
    plt.close(fig)

    # 判定散点图
    fig, ax = plt.subplots(figsize=(10, 7))
    for label, sub in plot_df.groupby("判定"):
        ax.scatter(sub["PC1"], sub["PC2"], s=6, alpha=0.45, label=f"{label} plotted={len(sub):,}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"GMM PCA 二维散点：阈值 q={main_q:.4f} 下判定结果")
    add_count_box(ax, count_text(dvalid) + f"\n实际绘制 {len(plot_df):,} 点；阈值={thr:.4g}")
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_dir / f"gmm_pca_scatter_pred_q{int(main_q*10000):04d}.png", dpi=220)
    plt.close(fig)

    if PLOTLY_AVAILABLE:
        hover_cols = ["source_code", "device_name", "row_id", "真实类别", "判定", "gmm_score", "gmm_component"]
        fig = px.scatter(
            plot_df,
            x="PC1", y="PC2", color="真实类别",
            hover_data=[c for c in hover_cols if c in plot_df.columns],
            title=f"GMM PCA 交互散点：正常/异常/未标注；全量 {len(dvalid):,}，绘制 {len(plot_df):,}",
            opacity=0.55,
        )
        fig.write_html(html_dir / "gmm_pca_scatter_true_label.html")
        fig = px.scatter(
            plot_df,
            x="PC1", y="PC2", color="gmm_score",
            hover_data=[c for c in hover_cols if c in plot_df.columns],
            title=f"GMM PCA 交互散点：异常分数；全量 {len(dvalid):,}，绘制 {len(plot_df):,}",
            opacity=0.55,
        )
        fig.write_html(html_dir / "gmm_pca_scatter_score.html")


def save_component_table(data: pd.DataFrame, out_dir: Path) -> None:
    d = data[data["gmm_component"].notna()].copy()
    if d.empty:
        return
    tab = pd.pivot_table(
        d,
        index="gmm_component",
        columns="source_code",
        values="row_id",
        aggfunc="count",
        fill_value=0,
    )
    tab["total"] = tab.sum(axis=1)
    for c in tab.columns:
        if c != "total":
            tab[f"{c}_pct_in_component"] = tab[c] / tab["total"].replace(0, np.nan)
    tab.to_csv(out_dir / "gmm_component_code_distribution.csv", encoding="utf-8-sig")


def save_all_visualizations(
    data: pd.DataFrame,
    metrics_df: pd.DataFrame,
    device_df: pd.DataFrame,
    bic_df: pd.DataFrame,
    feature_cols: List[str],
    scaler: StandardScaler,
    thresholds: Dict[float, float],
    args: argparse.Namespace,
    fig_dir: Path,
    html_dir: Path,
) -> None:
    save_threshold_metric_plots(metrics_df, fig_dir)
    save_score_distribution_plots(data, thresholds, fig_dir)
    save_roc_pr_plots(data, fig_dir)
    save_device_plots(device_df, fig_dir, html_dir, args.main_plot_quantile)
    save_bic_plot(bic_df, fig_dir)
    save_pca_scatter_plots(data, feature_cols, scaler, thresholds, args, fig_dir, html_dir)


# =========================
# 主函数
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(description="GMM 二次侧异常检测（含丰富可视化）")
    parser.add_argument("--root_dir", default=DATA_ROOT_DIR, help="原始数据根目录。PyCharm 直接运行时默认使用代码顶部 DATA_ROOT_DIR。")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="输出目录。PyCharm 直接运行时默认使用代码顶部 OUTPUT_DIR。")
    parser.add_argument("--f01_f02_mode", choices=["as_normal", "exclude"], default=F01_F02_MODE,
                        help="F01/F02 为一次侧异常。as_normal=作为二次侧正常；exclude=剔除不考虑。默认使用代码顶部 F01_F02_MODE。")
    parser.add_argument("--skip_unlabeled", action="store_true", default=SKIP_M_S_U, help="是否跳过 M/S/U。默认读取并打分，但不参与主指标。默认使用代码顶部 SKIP_M_S_U。")
    parser.add_argument("--pump_dp_sign", choices=["inlet_minus_outlet", "outlet_minus_inlet"], default=PUMP_DP_SIGN_CONFIG,
                        help="没有现成二次侧泵压差列时，如何由泵入口/出口压力计算泵压差。默认使用代码顶部 PUMP_DP_SIGN_CONFIG。")
    parser.add_argument("--include_temperature", action="store_true", default=INCLUDE_TEMPERATURE_CONFIG, help="是否把二次侧入口/出口温度放入 GMM。默认使用代码顶部 INCLUDE_TEMPERATURE_CONFIG。")
    parser.add_argument("--threshold_quantiles", default=THRESHOLD_QUANTILES_CONFIG,
                        help="阈值分位数，逗号分隔，例如 0.95,0.975,0.99,0.9975,0.9999。默认使用代码顶部 THRESHOLD_QUANTILES_CONFIG。")
    parser.add_argument("--train_normal_frac", type=float, default=0.7, help="正常设备中用于训练 GMM 的比例。")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--max_rows_per_file", type=int, default=None, help="每个 CSV 最多读取多少行；默认读取全部。")
    parser.add_argument("--max_train_rows", type=int, default=300000, help="GMM 训练最多使用多少条正常记录，防止内存过大。")
    parser.add_argument("--min_feature_valid_rate", type=float, default=0.50, help="特征在正常样本中的最低有效率，低于该值不进入模型。")
    parser.add_argument("--n_components", type=int, default=0, help="GMM 成分数；0 表示用 BIC 在范围内自动选择。")
    parser.add_argument("--n_components_min", type=int, default=1)
    parser.add_argument("--n_components_max", type=int, default=8)
    parser.add_argument("--covariance_type", default="full", choices=["full", "tied", "diag", "spherical"])
    parser.add_argument("--reg_covar", type=float, default=1e-6)
    parser.add_argument("--max_iter", type=int, default=300)
    parser.add_argument("--n_init", type=int, default=2)
    parser.add_argument("--plot_sample_size", type=int, default=60000, help="散点图最多绘制多少个点；图中会标注全量点数量。")
    parser.add_argument("--main_plot_quantile", type=float, default=0.9975, help="设备图和判定散点图默认使用的阈值分位数。")
    args = parser.parse_args()

    # PyCharm 直接运行时，若忘记修改代码顶部路径，给出更明确的提示。
    if (not args.root_dir) or ("请把这里改成" in str(args.root_dir)):
        raise ValueError("请先在代码顶部修改 DATA_ROOT_DIR 为你的数据根目录，或者在 PyCharm Parameters 中传入 --root_dir。")
    if (not args.output_dir) or ("请把这里改成" in str(args.output_dir)):
        raise ValueError("请先在代码顶部修改 OUTPUT_DIR 为你的结果输出目录，或者在 PyCharm Parameters 中传入 --output_dir。")

    out_dir = Path(args.output_dir)
    tables_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    html_dir = out_dir / "html"
    for d in [tables_dir, fig_dir, html_dir]:
        d.mkdir(parents=True, exist_ok=True)

    quantiles = parse_quantiles(args.threshold_quantiles)

    print("[STEP] 读取数据与构造二次侧特征")
    data, mappings = load_dataset(args)
    pd.DataFrame(mappings).to_csv(tables_dir / "gmm_column_mapping_by_file.csv", index=False, encoding="utf-8-sig")

    print("[STEP] 选择 GMM 特征")
    feature_cols = choose_feature_columns(data, args)
    print("[INFO] GMM 使用特征：", feature_cols)

    print("[STEP] 设备级划分 normal_train / normal_heldout")
    data["split"] = split_normal_devices(data, args)
    split_summary = data.groupby(["source_code", "label_name", "split"]).size().reset_index(name="records")
    split_summary.to_csv(tables_dir / "gmm_split_summary.csv", index=False, encoding="utf-8-sig")

    print("[STEP] 拟合 GMM")
    scaler, model, bic_df, _ = fit_gmm_model(data, feature_cols, args)
    bic_df.to_csv(tables_dir / "gmm_bic_aic_table.csv", index=False, encoding="utf-8-sig")

    print("[STEP] 全量打分")
    scored = score_gmm(data, feature_cols, scaler, model)
    thresholds = compute_thresholds(scored, quantiles)

    print("[STEP] 计算不同阈值下指标与记录级预测")
    records, metrics_df, code_df, device_df = build_metric_tables(scored, thresholds)
    records.to_csv(tables_dir / "gmm_record_predictions.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(tables_dir / "gmm_metrics_by_threshold.csv", index=False, encoding="utf-8-sig")
    code_df.to_csv(tables_dir / "gmm_metrics_by_code_threshold.csv", index=False, encoding="utf-8-sig")
    device_df.to_csv(tables_dir / "gmm_device_summary_by_threshold.csv", index=False, encoding="utf-8-sig")
    save_component_table(records, tables_dir)

    print("[STEP] 生成可视化")
    save_all_visualizations(records, metrics_df, device_df, bic_df, feature_cols, scaler, thresholds, args, fig_dir, html_dir)

    config = {
        "model": "GMM",
        "root_dir": str(Path(args.root_dir)),
        "output_dir": str(out_dir),
        "f01_f02_mode": args.f01_f02_mode,
        "secondary_anomaly_codes": sorted(SECONDARY_ANOMALY_CODES),
        "primary_anomaly_codes": sorted(PRIMARY_ANOMALY_CODES),
        "thresholds": {str(q): v for q, v in thresholds.items()},
        "feature_cols": feature_cols,
        "chosen_n_components": int(model.n_components),
        "covariance_type": args.covariance_type,
        "note": "GMM 不剔除高转速区间；F03/F04/F06 作为二次侧异常；F01/F02 由 f01_f02_mode 控制。",
    }
    with open(tables_dir / "gmm_config_and_columns.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("[DONE] GMM 输出完成：")
    print(f"  表格：{tables_dir}")
    print(f"  图片：{fig_dir}")
    print(f"  HTML：{html_dir}")


if __name__ == "__main__":
    main()
