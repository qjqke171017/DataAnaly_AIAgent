from __future__ import annotations

import gc
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import HuberRegressor


# ==========================================================
# 0. 用户配置区
# ==========================================================
# 把下面 3 个路径改成你自己的根目录。每个组目录下仍然有 5 个按转速分箱的文件夹，
# 代码会递归读取 csv，但建模时会忽略这层分箱，把该控制组所有速度放在一起拟合。
GROUP_DIRS: Dict[str, str] = {
    "0.8": r"/path/to/泵控制_目标值0.8区间",
    "1.4": r"/path/to/泵控制_目标值1.4区间",
    "1.65": r"/path/to/泵控制_目标值1.65区间",
}

OUTPUT_DIR = r"/path/to/output_dir"

# 如果自动识别列名失败，在这里手动写真实列名。未写的字段会继续自动匹配。
COLUMN_NAME_OVERRIDES: Dict[str, str] = {
    # "speed_col": "二次侧泵转速",
    # "valve_open_col": "二次侧阀开度",
    # "sec_inlet1_col": "二次侧入口压力1",
    # "sec_inlet2_col": "二次侧入口压力2",
    # "sec_outlet1_col": "二次侧出口压力1",
    # "sec_outlet2_col": "二次侧出口压力2",
    # "pump_dp_col": "二次侧泵压差",
    # "pump_inlet1_col": "二次侧泵入口压力1",
    # "pump_inlet2_col": "二次侧泵入口压力2",
    # "pump_outlet_col": "二次侧泵出口压力",
    # "sec_sr_dp_col": "二次侧供回水压差",
    # "sec_hex_dp_col": "二次侧板换压差",
    # "sec_filter_dp_col": "二次侧过滤器压差",
    # "timestamp_col": "时间戳",
}

# 对泵转速非常接近 100 的样本，不参与拟合、阈值估计和异常判断。
EXCLUDE_SPEED_RANGE: Optional[Tuple[float, float]] = (95.0, 100.0)

# 阈值同时输出三套：99.5%、99.75%、99.9%。主结果默认使用 99.75%。
PRIMARY_THRESHOLD_QUANTILE = 0.9975
EXTRA_THRESHOLD_QUANTILES = [0.9950, 0.9990]

# 局部主带筛选参数：现在不按原始 5 个转速 bin 分模型，但仍会在 x 上临时做局部分位分组，
# 用来挑选稳健的种子正常样本。
LOCAL_SEED_BINS = 20
SEED_BAND_K = 2.5
INLIER_ZMAX = 3.5
HUBER_EPSILON = 1.35
HUBER_ALPHA = 0.0001
MAX_HUBER_ROUNDS = 6

# 数据量很大时，为了防止 Huber 拟合太慢，可以只在当前内点中抽样拟合，
# 但仍会对该组所有样本打分。None 表示拟合时使用全部当前内点。
FIT_MAX_ROWS_PER_GROUP_MODEL: Optional[int] = 2_000_000

# 图表输出控制
INTERACTIVE_MAX_POINTS = 60_000
STATIC_ANOMALY_OVERLAY_MAX = 15_000
TOP_SAMPLES_PER_GROUP_MODEL = 2_000
DEVICE_SCATTER_TOPN = 500
PLOT_SAMPLE_RANDOM_STATE = 42


# ==========================================================
# 1. 模型定义
# ==========================================================
@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    response_desc: str
    required_roles: Tuple[str, ...]
    response_builder: Callable[[pd.DataFrame, Dict[str, Optional[str]]], pd.Series]


def build_pump_dp(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> pd.Series:
    pump_dp_col = cols.get("pump_dp_col")
    inlet1_col = cols.get("pump_inlet1_col")
    inlet2_col = cols.get("pump_inlet2_col")
    outlet_col = cols.get("pump_outlet_col")

    if pump_dp_col and pump_dp_col in df.columns:
        return pd.to_numeric(df[pump_dp_col], errors="coerce")

    if outlet_col and inlet1_col and outlet_col in df.columns and inlet1_col in df.columns:
        outlet = pd.to_numeric(df[outlet_col], errors="coerce")
        inlet1 = pd.to_numeric(df[inlet1_col], errors="coerce")
        if inlet2_col and inlet2_col in df.columns:
            inlet2 = pd.to_numeric(df[inlet2_col], errors="coerce")
            inlet_mean = pd.concat([inlet1, inlet2], axis=1).mean(axis=1)
        else:
            inlet_mean = inlet1
        return outlet - inlet_mean

    return pd.Series(np.nan, index=df.index, dtype="float64")


def build_secsum(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> pd.Series:
    sr = cols.get("sec_sr_dp_col")
    hx = cols.get("sec_hex_dp_col")
    if not sr or not hx or sr not in df.columns or hx not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[sr], errors="coerce") + pd.to_numeric(df[hx], errors="coerce")


def build_hex_filter_sum(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> pd.Series:
    hx = cols.get("sec_hex_dp_col")
    flt = cols.get("sec_filter_dp_col")
    if not hx or not flt or hx not in df.columns or flt not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[hx], errors="coerce") + pd.to_numeric(df[flt], errors="coerce")


MODEL_SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="m1_pumpdp",
        display_name="模型1_二次侧泵压差_vs_二次侧泵转速平方",
        response_desc="二次侧泵压差",
        required_roles=("speed_col",),
        response_builder=build_pump_dp,
    ),
    ModelSpec(
        model_id="m2_secsum",
        display_name="模型2_二次侧供回水压差加二次侧板换压差_vs_二次侧泵转速平方",
        response_desc="二次侧供回水压差 + 二次侧板换压差",
        required_roles=("speed_col", "sec_sr_dp_col", "sec_hex_dp_col"),
        response_builder=build_secsum,
    ),
    ModelSpec(
        model_id="m3_hex_filter_sum",
        display_name="模型3_二次侧板换压差加二次侧过滤器压差_vs_二次侧泵转速平方",
        response_desc="二次侧板换压差 + 二次侧过滤器压差",
        required_roles=("speed_col", "sec_hex_dp_col", "sec_filter_dp_col"),
        response_builder=build_hex_filter_sum,
    ),
)


# ==========================================================
# 2. 列名和读数工具
# ==========================================================
COLUMN_CANDIDATES: Dict[str, List[str]] = {
    "speed_col": ["二次侧泵转速", "泵转速", "二次侧转速", "二次泵转速"],
    "valve_open_col": ["二次侧阀开度", "二次阀开度", "阀开度"],
    "sec_inlet1_col": ["二次侧入口压力1", "二次入口压力1"],
    "sec_inlet2_col": ["二次侧入口压力2", "二次入口压力2"],
    "sec_outlet1_col": ["二次侧出口压力1", "二次出口压力1"],
    "sec_outlet2_col": ["二次侧出口压力2", "二次出口压力2"],
    "pump_dp_col": ["二次侧泵压差", "二次泵压差"],
    "pump_inlet1_col": ["二次侧泵入口压力1", "二次泵入口压力1"],
    "pump_inlet2_col": ["二次侧泵入口压力2", "二次泵入口压力2"],
    "pump_outlet_col": ["二次侧泵出口压力", "二次泵出口压力"],
    "sec_sr_dp_col": ["二次侧供回水压差", "二次供回水压差"],
    "sec_hex_dp_col": ["二次侧板换压差", "二次板换压差"],
    "sec_filter_dp_col": ["二次侧过滤器压差", "二次过滤器压差", "二次侧过滤压差"],
    "timestamp_col": ["时间戳", "timestamp", "时间", "采样时间"],
}


DEVICE_REGEX = re.compile(r"^(?P<device>.+?)_二次侧")


def safe_read_csv(path: Path, usecols: Optional[List[str]] = None, encodings: Optional[List[str]] = None) -> pd.DataFrame:
    if encodings is None:
        encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, usecols=usecols, encoding=enc, low_memory=False)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"读取失败: {path}\n最后一个错误: {last_error}")


def find_csv_files(root_dir: str) -> List[Path]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"路径不存在: {root_dir}")
    return sorted([p for p in root.rglob("*.csv") if p.is_file()])


def choose_existing_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    col_set = set(columns)
    for c in candidates:
        if c in col_set:
            return c
    normalized = {c.strip().lower().replace(" ", ""): c for c in columns}
    for cand in candidates:
        key = cand.strip().lower().replace(" ", "")
        if key in normalized:
            return normalized[key]
    return None


def resolve_columns(example_file: Path, overrides: Dict[str, str]) -> Dict[str, Optional[str]]:
    head = safe_read_csv(example_file).head(3)
    cols = list(head.columns)
    resolved: Dict[str, Optional[str]] = {}
    for role, candidates in COLUMN_CANDIDATES.items():
        if role in overrides and overrides[role] in cols:
            resolved[role] = overrides[role]
        else:
            resolved[role] = choose_existing_column(cols, candidates)
    return resolved


def extract_device_name(file_stem: str) -> str:
    m = DEVICE_REGEX.match(file_stem)
    if m:
        return m.group("device")
    return file_stem


def sanitize_filename(text: str) -> str:
    return str(text).replace("/", "__").replace("\\", "__").replace(":", "_")


def quantile_label(q: float) -> str:
    s = f"{q:.4f}".rstrip("0").rstrip(".")
    return "q" + s.replace(".", "")


def robust_mad(x: np.ndarray, eps: float = 1e-9) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return max(1.4826 * mad, eps)


def qcut_with_fallback(x: pd.Series, q: int) -> pd.Series:
    try:
        return pd.qcut(x, q=q, duplicates="drop")
    except Exception:
        ranks = x.rank(method="first")
        return pd.qcut(ranks, q=q, duplicates="drop")


def _series_or_na(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if col and col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _pair_present(df: pd.DataFrame, col1: Optional[str], col2: Optional[str]) -> pd.Series:
    s1 = _series_or_na(df, col1)
    s2 = _series_or_na(df, col2)
    return s1.notna() | s2.notna()


# ==========================================================
# 3. 数据读取
# ==========================================================

def load_group_table(group_name: str, group_dir: str, resolved_cols: Dict[str, Optional[str]]) -> pd.DataFrame:
    csv_files = find_csv_files(group_dir)
    if not csv_files:
        raise FileNotFoundError(f"分组 {group_name} 下未找到 csv 文件: {group_dir}")

    usecols = [c for c in resolved_cols.values() if c]
    usecols = list(dict.fromkeys(usecols))
    group_root = Path(group_dir).resolve()
    frames: List[pd.DataFrame] = []

    for fp in csv_files:
        df = safe_read_csv(fp, usecols=usecols if usecols else None)
        rel = fp.resolve().relative_to(group_root)
        speed_bin_folder = rel.parts[0] if len(rel.parts) >= 2 else "bin_unknown"
        file_stem = fp.stem
        device_name = extract_device_name(file_stem)

        df["device_name"] = device_name
        df["raw_file_stem"] = file_stem
        df["source_speed_bin"] = speed_bin_folder
        df["source_relpath"] = rel.as_posix()
        df["row_in_file"] = np.arange(len(df), dtype=np.int32)
        df["control_group"] = group_name
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    numeric_roles = [
        "speed_col",
        "valve_open_col",
        "sec_inlet1_col",
        "sec_inlet2_col",
        "sec_outlet1_col",
        "sec_outlet2_col",
        "pump_dp_col",
        "pump_inlet1_col",
        "pump_inlet2_col",
        "pump_outlet_col",
        "sec_sr_dp_col",
        "sec_hex_dp_col",
        "sec_filter_dp_col",
    ]
    for role in numeric_roles:
        col = resolved_cols.get(role)
        if col and col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").astype("float32")

    tcol = resolved_cols.get("timestamp_col")
    if tcol and tcol in data.columns:
        data["timestamp"] = pd.to_datetime(data[tcol], errors="coerce")

    # 用 category 降内存
    for c in ["device_name", "raw_file_stem", "source_speed_bin", "source_relpath", "control_group"]:
        data[c] = data[c].astype("category")

    return data


# ==========================================================
# 4. 模型准备、种子样本、Huber
# ==========================================================

def prepare_model_table(group_df: pd.DataFrame, model_spec: ModelSpec, resolved_cols: Dict[str, Optional[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    speed_col = resolved_cols.get("speed_col")
    if not speed_col or speed_col not in group_df.columns:
        raise ValueError("未找到二次侧泵转速列。")

    df = group_df[[
        "device_name", "raw_file_stem", "source_speed_bin", "source_relpath", "row_in_file", "control_group"
    ]].copy()
    if "timestamp" in group_df.columns:
        df["timestamp"] = group_df["timestamp"]

    df["speed_raw"] = pd.to_numeric(group_df[speed_col], errors="coerce").astype("float32")
    df["response_y"] = model_spec.response_builder(group_df, resolved_cols).astype("float32")
    df["speed_sq"] = (df["speed_raw"] ** 2).astype("float32")
    df["response_formula"] = model_spec.response_desc

    # 二次侧基础完整性检查：
    # 入口压力1/2 有其一即可；出口压力1/2 有其一即可；泵入口压力1/2 有其一即可；
    # 另外二次侧泵出口压力、二次侧泵转速、二次侧阀开度必须非空。
    base_presence_items = {
        "二次侧入口压力(1或2)": _pair_present(group_df, resolved_cols.get("sec_inlet1_col"), resolved_cols.get("sec_inlet2_col")),
        "二次侧出口压力(1或2)": _pair_present(group_df, resolved_cols.get("sec_outlet1_col"), resolved_cols.get("sec_outlet2_col")),
        "二次侧泵入口压力(1或2)": _pair_present(group_df, resolved_cols.get("pump_inlet1_col"), resolved_cols.get("pump_inlet2_col")),
        "二次侧泵出口压力": _series_or_na(group_df, resolved_cols.get("pump_outlet_col")).notna(),
        "二次侧泵转速": _series_or_na(group_df, resolved_cols.get("speed_col")).notna(),
        "二次侧阀开度": _series_or_na(group_df, resolved_cols.get("valve_open_col")).notna(),
    }
    base_complete = pd.concat(base_presence_items.values(), axis=1).all(axis=1)

    # 模型专属响应完整性
    if model_spec.model_id == "m1_pumpdp":
        model_required_ok = df["response_y"].notna()
    elif model_spec.model_id == "m2_secsum":
        model_required_ok = (
            _series_or_na(group_df, resolved_cols.get("sec_sr_dp_col")).notna()
            & _series_or_na(group_df, resolved_cols.get("sec_hex_dp_col")).notna()
        )
    elif model_spec.model_id == "m3_hex_filter_sum":
        model_required_ok = (
            _series_or_na(group_df, resolved_cols.get("sec_hex_dp_col")).notna()
            & _series_or_na(group_df, resolved_cols.get("sec_filter_dp_col")).notna()
        )
    else:
        model_required_ok = df["response_y"].notna()

    df["missing_base_required"] = (~base_complete).to_numpy()
    df["missing_model_required"] = (~model_required_ok).to_numpy()
    df["missing_required"] = (~(base_complete & model_required_ok)).to_numpy()

    df["invalid_speed"] = (~np.isfinite(df["speed_raw"])) | (df["speed_raw"] < 0)
    df["invalid_response"] = ~np.isfinite(df["response_y"])

    if EXCLUDE_SPEED_RANGE is None:
        df["is_speed_excluded"] = False
    else:
        lo, hi = EXCLUDE_SPEED_RANGE
        df["is_speed_excluded"] = df["speed_raw"].between(lo, hi, inclusive="both").fillna(False)

    df["is_evaluable"] = (~df["missing_required"]) & (~df["invalid_speed"]) & (~df["invalid_response"]) & (~df["is_speed_excluded"])

    total_n = len(df)
    missing_summary_rows = []
    for label, present_mask in base_presence_items.items():
        n_missing = int((~present_mask).sum())
        missing_summary_rows.append({
            "raw_column": label,
            "missing_count": n_missing,
            "missing_rate": n_missing / total_n if total_n else np.nan,
            "check_type": "base_required",
        })

    model_specific_checks = {
        "m1_pumpdp": [("二次侧泵压差/由泵出口与泵入口构造", model_required_ok)],
        "m2_secsum": [
            ("二次侧供回水压差", _series_or_na(group_df, resolved_cols.get("sec_sr_dp_col")).notna()),
            ("二次侧板换压差", _series_or_na(group_df, resolved_cols.get("sec_hex_dp_col")).notna()),
        ],
        "m3_hex_filter_sum": [
            ("二次侧板换压差", _series_or_na(group_df, resolved_cols.get("sec_hex_dp_col")).notna()),
            ("二次侧过滤器压差", _series_or_na(group_df, resolved_cols.get("sec_filter_dp_col")).notna()),
        ],
    }
    for label, present_mask in model_specific_checks.get(model_spec.model_id, []):
        n_missing = int((~present_mask).sum())
        missing_summary_rows.append({
            "raw_column": label,
            "missing_count": n_missing,
            "missing_rate": n_missing / total_n if total_n else np.nan,
            "check_type": "model_required",
        })

    missing_summary_rows.append({
        "raw_column": "基础字段与模型字段联合缺失",
        "missing_count": int(df["missing_required"].sum()),
        "missing_rate": float(df["missing_required"].mean()) if total_n else np.nan,
        "check_type": "combined_required",
    })

    missing_summary = pd.DataFrame(missing_summary_rows)
    missing_summary["control_group"] = str(group_df["control_group"].iloc[0])
    missing_summary["model_id"] = model_spec.model_id
    missing_summary["model_name"] = model_spec.display_name
    missing_summary["total_rows"] = total_n
    missing_summary["evaluable_rows"] = int(df["is_evaluable"].sum())
    missing_summary["excluded_speed_rows"] = int(df["is_speed_excluded"].sum())

    return df, missing_summary



def select_seed_by_local_band(work: pd.DataFrame, x_col: str = "speed_sq", y_col: str = "response_y") -> pd.Series:
    if len(work) < 100:
        return pd.Series(np.ones(len(work), dtype=bool), index=work.index)

    real_bins = min(LOCAL_SEED_BINS, max(4, len(work) // 5000))
    bins = qcut_with_fallback(work[x_col], q=real_bins)
    tmp = pd.DataFrame({"x_bin": bins, y_col: work[y_col]})
    centers = tmp.groupby("x_bin", observed=False)[y_col].median().rename("local_median")
    scales = tmp.groupby("x_bin", observed=False)[y_col].apply(lambda s: robust_mad(s.to_numpy())).rename("local_sigma")
    tmp = tmp.join(centers, on="x_bin").join(scales, on="x_bin")
    seed_mask = (tmp[y_col] - tmp["local_median"]).abs() <= SEED_BAND_K * tmp["local_sigma"]
    return pd.Series(seed_mask.fillna(False).to_numpy(), index=work.index)



def sample_fit_subset(work: pd.DataFrame, inlier_mask: pd.Series, max_rows: Optional[int]) -> pd.DataFrame:
    inliers = work.loc[inlier_mask].copy()
    if max_rows is None or len(inliers) <= max_rows:
        return inliers

    tmp = inliers[["speed_sq"]].copy()
    tmp["x_bin"] = qcut_with_fallback(tmp["speed_sq"], q=20)
    sampled = (
        inliers.groupby(tmp["x_bin"], observed=False, group_keys=False)
        .apply(lambda g: g.sample(n=max(1, int(round(len(g) / len(inliers) * max_rows))), random_state=PLOT_SAMPLE_RANDOM_STATE))
    )
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=PLOT_SAMPLE_RANDOM_STATE)
    return sampled



def fit_huber_once(x: np.ndarray, y: np.ndarray) -> HuberRegressor:
    model = HuberRegressor(
        epsilon=HUBER_EPSILON,
        alpha=HUBER_ALPHA,
        fit_intercept=True,
        max_iter=500,
    )
    model.fit(x.reshape(-1, 1), y)
    return model



def iterative_huber_fit(work: pd.DataFrame) -> Tuple[HuberRegressor, pd.DataFrame, Dict[str, float]]:
    if len(work) < 20:
        raise ValueError("可评估样本过少，无法拟合。")

    seed_mask = select_seed_by_local_band(work)
    if seed_mask.mean() < 0.2:
        seed_mask = pd.Series(np.ones(len(work), dtype=bool), index=work.index)

    current_inlier = seed_mask.copy().astype(bool)
    diagnostics: Dict[str, float] = {
        "seed_count": int(seed_mask.sum()),
        "seed_ratio": float(seed_mask.mean()),
    }

    for round_idx in range(MAX_HUBER_ROUNDS):
        fit_df = sample_fit_subset(work, current_inlier, FIT_MAX_ROWS_PER_GROUP_MODEL)
        x_fit = fit_df["speed_sq"].to_numpy(dtype=float)
        y_fit = fit_df["response_y"].to_numpy(dtype=float)
        model = fit_huber_once(x_fit, y_fit)

        pred = model.predict(work["speed_sq"].to_numpy(dtype=float).reshape(-1, 1))
        resid = work["response_y"].to_numpy(dtype=float) - pred

        inlier_resid = resid[current_inlier.to_numpy()]
        center = float(np.median(inlier_resid)) if len(inlier_resid) else float(np.median(resid))
        sigma = float(robust_mad(inlier_resid if len(inlier_resid) else resid))
        score = np.abs(resid - center) / sigma
        new_inlier = pd.Series(score <= INLIER_ZMAX, index=work.index)

        work["y_pred"] = pred.astype("float32")
        work["residual"] = resid.astype("float32")
        work["robust_score"] = score.astype("float32")
        work["is_inlier"] = new_inlier.to_numpy()

        if new_inlier.equals(current_inlier):
            diagnostics["rounds"] = round_idx + 1
            diagnostics["final_sigma"] = sigma
            diagnostics["final_center"] = center
            diagnostics["fit_sample_count"] = len(fit_df)
            return model, work, diagnostics

        current_inlier = new_inlier

    diagnostics["rounds"] = MAX_HUBER_ROUNDS
    diagnostics["final_sigma"] = float(robust_mad(work.loc[current_inlier, "residual"].to_numpy(dtype=float)))
    diagnostics["final_center"] = float(np.median(work.loc[current_inlier, "residual"].to_numpy(dtype=float)))
    diagnostics["fit_sample_count"] = int(current_inlier.sum()) if FIT_MAX_ROWS_PER_GROUP_MODEL is None else min(int(current_inlier.sum()), FIT_MAX_ROWS_PER_GROUP_MODEL)
    return model, work, diagnostics


# ==========================================================
# 5. 汇总输出
# ==========================================================

def enrich_threshold_flags(scored: pd.DataFrame, quantiles: Sequence[float]) -> Tuple[pd.DataFrame, Dict[float, float]]:
    inlier_scores = scored.loc[scored["is_inlier"], "robust_score"].to_numpy(dtype=float)
    if len(inlier_scores) == 0:
        inlier_scores = scored["robust_score"].to_numpy(dtype=float)

    thresholds: Dict[float, float] = {}
    for q in sorted(set(quantiles)):
        thr = float(np.quantile(inlier_scores, q)) if len(inlier_scores) else np.nan
        thresholds[q] = thr
        scored[f"is_anomaly_{quantile_label(q)}"] = scored["robust_score"] > thr
    return scored, thresholds



def build_device_summary(base_df: pd.DataFrame, scored_df: pd.DataFrame, thresholds: Dict[float, float], model_spec: ModelSpec) -> pd.DataFrame:
    base = base_df.copy()
    scored = scored_df.copy()

    summary = base.groupby("device_name", observed=False).agg(
        total_samples=("device_name", "size"),
        missing_required_samples=("missing_required", "sum"),
        invalid_speed_samples=("invalid_speed", "sum"),
        invalid_response_samples=("invalid_response", "sum"),
        excluded_speed_samples=("is_speed_excluded", "sum"),
        evaluable_samples=("is_evaluable", "sum"),
        n_raw_files=("raw_file_stem", "nunique"),
        n_speed_bins=("source_speed_bin", "nunique"),
    ).reset_index()

    score_agg = scored.groupby("device_name", observed=False).agg(
        max_score=("robust_score", "max"),
        mean_score=("robust_score", "mean"),
        median_score=("robust_score", "median"),
        p95_score=("robust_score", lambda s: float(np.quantile(s, 0.95)) if len(s) else np.nan),
        max_abs_residual=("residual", lambda s: float(np.max(np.abs(s))) if len(s) else np.nan),
        mean_abs_residual=("residual", lambda s: float(np.mean(np.abs(s))) if len(s) else np.nan),
        median_abs_residual=("residual", lambda s: float(np.median(np.abs(s))) if len(s) else np.nan),
        p95_abs_residual=("residual", lambda s: float(np.quantile(np.abs(s), 0.95)) if len(s) else np.nan),
        mean_residual=("residual", "mean"),
        speed_min=("speed_raw", "min"),
        speed_max=("speed_raw", "max"),
    ).reset_index()

    summary = summary.merge(score_agg, on="device_name", how="left")

    for q, thr in thresholds.items():
        qlab = quantile_label(q)
        flag_col = f"is_anomaly_{qlab}"
        agg = scored.groupby("device_name", observed=False).agg(
            **{
                f"anomaly_count_{qlab}": (flag_col, "sum"),
            }
        ).reset_index()
        summary = summary.merge(agg, on="device_name", how="left")
        summary[f"anomaly_count_{qlab}"] = summary[f"anomaly_count_{qlab}"].fillna(0).astype(int)
        denom = summary["evaluable_samples"].replace(0, np.nan)
        summary[f"anomaly_rate_{qlab}"] = summary[f"anomaly_count_{qlab}"] / denom

    primary_label = quantile_label(PRIMARY_THRESHOLD_QUANTILE)
    summary["severe_count_2x_primary"] = (
        scored.assign(severe=scored["robust_score"] > 2 * thresholds[PRIMARY_THRESHOLD_QUANTILE])
        .groupby("device_name", observed=False)["severe"].sum()
        .reindex(summary["device_name"])
        .fillna(0)
        .astype(int)
        .to_numpy()
    )

    summary["control_group"] = str(base["control_group"].iloc[0])
    summary["model_id"] = model_spec.model_id
    summary["model_name"] = model_spec.display_name
    summary = summary.sort_values([f"anomaly_rate_{primary_label}", "max_score", "evaluable_samples"], ascending=[False, False, False]).reset_index(drop=True)
    return summary



def build_top_samples(scored: pd.DataFrame, device_summary: pd.DataFrame, thresholds: Dict[float, float], model_spec: ModelSpec) -> pd.DataFrame:
    primary_label = quantile_label(PRIMARY_THRESHOLD_QUANTILE)
    keep_cols = [
        "device_name", "raw_file_stem", "source_speed_bin", "source_relpath", "row_in_file",
        "control_group", "speed_raw", "speed_sq", "response_y", "y_pred", "residual", "robust_score",
        f"is_anomaly_{primary_label}",
    ]
    if "timestamp" in scored.columns:
        keep_cols.append("timestamp")
    top = scored.sort_values("robust_score", ascending=False).head(TOP_SAMPLES_PER_GROUP_MODEL)[keep_cols].copy()

    merge_cols = [
        "device_name", "total_samples", "evaluable_samples", f"anomaly_count_{primary_label}",
        f"anomaly_rate_{primary_label}", "max_score", "mean_score", "p95_score", "mean_abs_residual"
    ]
    top = top.merge(device_summary[merge_cols], on="device_name", how="left")
    top["primary_threshold"] = thresholds[PRIMARY_THRESHOLD_QUANTILE]
    top["model_id"] = model_spec.model_id
    top["model_name"] = model_spec.display_name
    return top



def build_model_summary(scored: pd.DataFrame, thresholds: Dict[float, float], diagnostics: Dict[str, float], model: HuberRegressor, model_spec: ModelSpec) -> pd.DataFrame:
    out = {
        "control_group": [str(scored["control_group"].iloc[0])],
        "model_id": [model_spec.model_id],
        "model_name": [model_spec.display_name],
        "response_desc": [model_spec.response_desc],
        "total_rows": [len(scored)],
        "evaluable_rows": [len(scored)],
        "seed_count": [diagnostics.get("seed_count")],
        "seed_ratio": [diagnostics.get("seed_ratio")],
        "final_inlier_count": [int(scored["is_inlier"].sum())],
        "final_inlier_ratio": [float(scored["is_inlier"].mean())],
        "huber_intercept": [float(model.intercept_)],
        "huber_slope": [float(model.coef_[0])],
        "residual_center": [diagnostics.get("final_center")],
        "residual_sigma": [diagnostics.get("final_sigma")],
        "rounds": [diagnostics.get("rounds")],
        "fit_sample_count": [diagnostics.get("fit_sample_count")],
        "score_mean": [float(scored["robust_score"].mean())],
        "score_median": [float(scored["robust_score"].median())],
        "score_p95": [float(np.quantile(scored["robust_score"], 0.95))],
    }
    for q, thr in thresholds.items():
        qlab = quantile_label(q)
        out[f"threshold_{qlab}"] = [thr]
        out[f"anomaly_count_{qlab}"] = [int(scored[f"is_anomaly_{qlab}"].sum())]
        out[f"anomaly_rate_{qlab}"] = [float(scored[f"is_anomaly_{qlab}"].mean())]
    return pd.DataFrame(out)


# ==========================================================
# 6. 绘图
# ==========================================================

def sample_for_plot(scored: pd.DataFrame, thresholds: Dict[float, float], max_points: int) -> pd.DataFrame:
    primary_label = quantile_label(PRIMARY_THRESHOLD_QUANTILE)
    anomaly_col = f"is_anomaly_{primary_label}"
    anomalies = scored.loc[scored[anomaly_col]].copy()
    normals = scored.loc[~scored[anomaly_col]].copy()

    if len(anomalies) >= max_points:
        return anomalies.sample(n=max_points, random_state=PLOT_SAMPLE_RANDOM_STATE)

    normal_n = max_points - len(anomalies)
    if len(normals) > normal_n:
        normals = normals.sample(n=normal_n, random_state=PLOT_SAMPLE_RANDOM_STATE)
    out = pd.concat([anomalies, normals], ignore_index=True)
    return out



def save_missing_bar(missing_summary: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.bar(missing_summary["raw_column"], missing_summary["missing_rate"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("缺失率")
    plt.title(f"{missing_summary['model_name'].iloc[0]} | 控制组 {missing_summary['control_group'].iloc[0]} | 关键字段缺失率")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()



def save_density_fit_png(scored: pd.DataFrame, model: HuberRegressor, model_spec: ModelSpec, out_png: Path) -> None:
    plt.figure(figsize=(10, 7))
    plt.hexbin(scored["speed_sq"], scored["response_y"], gridsize=120, mincnt=1, bins="log")
    x_min, x_max = float(scored["speed_sq"].min()), float(scored["speed_sq"].max())
    xs = np.linspace(x_min, x_max, 300)
    ys = model.predict(xs.reshape(-1, 1))
    plt.plot(xs, ys, linewidth=2.0)

    overlay = scored.nlargest(min(STATIC_ANOMALY_OVERLAY_MAX, len(scored)), "robust_score")
    plt.scatter(overlay["speed_sq"], overlay["response_y"], s=8, alpha=0.35)
    plt.xlabel("二次侧泵转速平方")
    plt.ylabel(model_spec.response_desc)
    plt.title(f"{model_spec.display_name} | 控制组 {scored['control_group'].iloc[0]} | 密度图 + Huber线")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()



def save_residual_density_png(scored: pd.DataFrame, thresholds: Dict[float, float], model_spec: ModelSpec, out_png: Path) -> None:
    primary_thr = thresholds[PRIMARY_THRESHOLD_QUANTILE]
    sigma = robust_mad(scored.loc[scored["is_inlier"], "residual"].to_numpy(dtype=float))
    center = float(np.median(scored.loc[scored["is_inlier"], "residual"].to_numpy(dtype=float)))
    resid_band = primary_thr * sigma

    plt.figure(figsize=(10, 7))
    plt.hexbin(scored["speed_raw"], scored["residual"], gridsize=120, mincnt=1, bins="log")
    plt.axhline(center + resid_band, linestyle="--", linewidth=1.3)
    plt.axhline(center - resid_band, linestyle="--", linewidth=1.3)
    overlay = scored.nlargest(min(STATIC_ANOMALY_OVERLAY_MAX, len(scored)), "robust_score")
    plt.scatter(overlay["speed_raw"], overlay["residual"], s=8, alpha=0.35)
    plt.xlabel("二次侧泵转速")
    plt.ylabel("残差")
    plt.title(f"{model_spec.display_name} | 控制组 {scored['control_group'].iloc[0]} | 残差-转速图")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()



def save_score_hist_png(scored: pd.DataFrame, thresholds: Dict[float, float], model_spec: ModelSpec, out_png: Path) -> None:
    plt.figure(figsize=(9, 6))
    plt.hist(scored["robust_score"], bins=120)
    for q, thr in thresholds.items():
        plt.axvline(thr, linestyle="--", linewidth=1.4, label=f"{q:.4f}")
    plt.xlabel("稳健异常分数")
    plt.ylabel("样本数")
    plt.title(f"{model_spec.display_name} | 控制组 {scored['control_group'].iloc[0]} | 分数分布")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()



def save_device_rate_hist_png(device_summary: pd.DataFrame, out_png: Path) -> None:
    primary_label = quantile_label(PRIMARY_THRESHOLD_QUANTILE)
    plt.figure(figsize=(9, 6))
    vals = device_summary[f"anomaly_rate_{primary_label}"].dropna().to_numpy()
    plt.hist(vals, bins=80)
    plt.xlabel("设备异常率")
    plt.ylabel("设备数")
    plt.title(f"{device_summary['model_name'].iloc[0]} | 控制组 {device_summary['control_group'].iloc[0]} | 设备异常率分布")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()



def _merge_hover_device_stats(sampled: pd.DataFrame, device_summary: pd.DataFrame) -> pd.DataFrame:
    primary_label = quantile_label(PRIMARY_THRESHOLD_QUANTILE)
    cols = [
        "device_name", "total_samples", "evaluable_samples", f"anomaly_count_{primary_label}",
        f"anomaly_rate_{primary_label}", "max_score", "mean_score", "p95_score", "mean_abs_residual"
    ]
    return sampled.merge(device_summary[cols], on="device_name", how="left")



def save_interactive_scatter(scored: pd.DataFrame, model: HuberRegressor, device_summary: pd.DataFrame, model_spec: ModelSpec, out_html: Path) -> None:
    sampled = sample_for_plot(scored, {}, INTERACTIVE_MAX_POINTS)
    sampled = _merge_hover_device_stats(sampled, device_summary)
    primary_label = quantile_label(PRIMARY_THRESHOLD_QUANTILE)
    sampled["status"] = np.where(sampled[f"is_anomaly_{primary_label}"], "异常", "正常")

    fig = px.scatter(
        sampled,
        x="speed_sq",
        y="response_y",
        color="status",
        hover_data={
            "device_name": True,
            "raw_file_stem": True,
            "source_speed_bin": True,
            "row_in_file": True,
            "speed_raw": ":.4f",
            "response_y": ":.6f",
            "y_pred": ":.6f",
            "residual": ":.6f",
            "robust_score": ":.4f",
            f"anomaly_rate_{primary_label}": ":.4%",
            f"anomaly_count_{primary_label}": True,
            "total_samples": True,
            "evaluable_samples": True,
            "max_score": ":.4f",
            "p95_score": ":.4f",
            "mean_abs_residual": ":.6f",
        },
        title=f"{model_spec.display_name} | 控制组 {scored['control_group'].iloc[0]} | 交互散点图",
        opacity=0.6,
    )
    x_min, x_max = float(sampled["speed_sq"].min()), float(sampled["speed_sq"].max())
    xs = np.linspace(x_min, x_max, 300)
    ys = model.predict(xs.reshape(-1, 1))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Huber拟合线"))
    fig.write_html(out_html)



def save_interactive_residual(scored: pd.DataFrame, device_summary: pd.DataFrame, model_spec: ModelSpec, out_html: Path) -> None:
    sampled = sample_for_plot(scored, {}, INTERACTIVE_MAX_POINTS)
    sampled = _merge_hover_device_stats(sampled, device_summary)
    primary_label = quantile_label(PRIMARY_THRESHOLD_QUANTILE)
    sampled["status"] = np.where(sampled[f"is_anomaly_{primary_label}"], "异常", "正常")

    fig = px.scatter(
        sampled,
        x="speed_raw",
        y="residual",
        color="status",
        hover_data={
            "device_name": True,
            "raw_file_stem": True,
            "source_speed_bin": True,
            "row_in_file": True,
            "speed_sq": ":.4f",
            "response_y": ":.6f",
            "y_pred": ":.6f",
            "residual": ":.6f",
            "robust_score": ":.4f",
            f"anomaly_rate_{primary_label}": ":.4%",
            f"anomaly_count_{primary_label}": True,
            "evaluable_samples": True,
            "max_score": ":.4f",
        },
        title=f"{model_spec.display_name} | 控制组 {scored['control_group'].iloc[0]} | 交互残差图",
        opacity=0.6,
    )
    fig.write_html(out_html)



def save_interactive_device_scatter(device_summary: pd.DataFrame, out_html: Path) -> None:
    primary_label = quantile_label(PRIMARY_THRESHOLD_QUANTILE)
    plot_df = device_summary.copy().head(min(DEVICE_SCATTER_TOPN, len(device_summary)))
    fig = px.scatter(
        plot_df,
        x="evaluable_samples",
        y=f"anomaly_rate_{primary_label}",
        size="max_score",
        hover_data={
            "device_name": True,
            "total_samples": True,
            "evaluable_samples": True,
            f"anomaly_count_{primary_label}": True,
            f"anomaly_rate_{primary_label}": ":.4%",
            "max_score": ":.4f",
            "p95_score": ":.4f",
            "mean_abs_residual": ":.6f",
            "n_raw_files": True,
            "n_speed_bins": True,
            "speed_min": ":.4f",
            "speed_max": ":.4f",
        },
        title=f"{plot_df['model_name'].iloc[0]} | 控制组 {plot_df['control_group'].iloc[0]} | 设备风险散点图",
    )
    fig.write_html(out_html)


# ==========================================================
# 7. 主流程
# ==========================================================

def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")



def process_one_group_one_model(
    group_name: str,
    group_df: pd.DataFrame,
    model_spec: ModelSpec,
    resolved_cols: Dict[str, Optional[str]],
    model_out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_df, missing_summary = prepare_model_table(group_df, model_spec, resolved_cols)
    eval_df = model_df.loc[model_df["is_evaluable"]].copy().reset_index(drop=True)
    if len(eval_df) < 20:
        raise ValueError(f"控制组 {group_name} | {model_spec.display_name} 可评估样本过少。")

    model, scored, diagnostics = iterative_huber_fit(eval_df)
    scored, thresholds = enrich_threshold_flags(scored, [PRIMARY_THRESHOLD_QUANTILE, *EXTRA_THRESHOLD_QUANTILES])

    device_summary = build_device_summary(model_df, scored, thresholds, model_spec)
    top_samples = build_top_samples(scored, device_summary, thresholds, model_spec)
    model_summary = build_model_summary(scored, thresholds, diagnostics, model, model_spec)

    group_slug = sanitize_filename(group_name)
    save_csv(missing_summary, model_out_dir / f"missing_summary_{group_slug}.csv")
    save_csv(device_summary, model_out_dir / f"device_summary_{group_slug}.csv")
    save_csv(top_samples, model_out_dir / f"top_samples_{group_slug}.csv")
    save_csv(model_summary, model_out_dir / f"model_summary_{group_slug}.csv")

    save_missing_bar(missing_summary, model_out_dir / f"missing_bar_{group_slug}.png")
    save_density_fit_png(scored, model, model_spec, model_out_dir / f"density_fit_{group_slug}.png")
    save_residual_density_png(scored, thresholds, model_spec, model_out_dir / f"residual_density_{group_slug}.png")
    save_score_hist_png(scored, thresholds, model_spec, model_out_dir / f"score_hist_{group_slug}.png")
    save_device_rate_hist_png(device_summary, model_out_dir / f"device_rate_hist_{group_slug}.png")

    save_interactive_scatter(scored, model, device_summary, model_spec, model_out_dir / f"interactive_scatter_{group_slug}.html")
    save_interactive_residual(scored, device_summary, model_spec, model_out_dir / f"interactive_residual_{group_slug}.html")
    save_interactive_device_scatter(device_summary, model_out_dir / f"interactive_device_risk_{group_slug}.html")

    # 返回总表
    return model_summary, device_summary, top_samples, missing_summary



def main() -> None:
    out_root = Path(OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    first_group = next(iter(GROUP_DIRS.values()))
    example_file = find_csv_files(first_group)[0]
    resolved_cols = resolve_columns(example_file, COLUMN_NAME_OVERRIDES)

    # 保存列映射，方便核对
    pd.DataFrame({"role": list(resolved_cols.keys()), "resolved_column": list(resolved_cols.values())}).to_csv(
        out_root / "resolved_columns.csv", index=False, encoding="utf-8-sig"
    )

    all_model_summaries: List[pd.DataFrame] = []
    all_device_summaries: List[pd.DataFrame] = []
    all_top_samples: List[pd.DataFrame] = []
    all_missing_summaries: List[pd.DataFrame] = []

    for group_name, group_dir in GROUP_DIRS.items():
        print(f"\n==== 读取控制组 {group_name} ====")
        group_df = load_group_table(group_name, group_dir, resolved_cols)
        print(f"控制组 {group_name} 原始行数: {len(group_df):,}")

        for model_spec in MODEL_SPECS:
            print(f"---- 开始 {model_spec.display_name} ----")
            model_out_dir = out_root / model_spec.model_id
            model_out_dir.mkdir(parents=True, exist_ok=True)
            model_summary, device_summary, top_samples, missing_summary = process_one_group_one_model(
                group_name=group_name,
                group_df=group_df,
                model_spec=model_spec,
                resolved_cols=resolved_cols,
                model_out_dir=model_out_dir,
            )
            all_model_summaries.append(model_summary)
            all_device_summaries.append(device_summary)
            all_top_samples.append(top_samples)
            all_missing_summaries.append(missing_summary)
            print(f"完成 {model_spec.display_name} | 控制组 {group_name}")
            gc.collect()

        del group_df
        gc.collect()

    save_csv(pd.concat(all_model_summaries, ignore_index=True), out_root / "model_summary_all.csv")
    save_csv(pd.concat(all_device_summaries, ignore_index=True), out_root / "device_summary_all.csv")
    save_csv(pd.concat(all_top_samples, ignore_index=True), out_root / "top_samples_all.csv")
    save_csv(pd.concat(all_missing_summaries, ignore_index=True), out_root / "missing_summary_all.csv")

    print("\n全部完成。输出目录:", out_root)


if __name__ == "__main__":
    main()
