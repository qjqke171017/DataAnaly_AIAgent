from __future__ import annotations

import gc
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.font_manager as fm
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

# 异方差修正控制：只对 1.4 和 1.65 组启用非等宽置信带；0.8 组保持原来的等宽逻辑。
# 如后续发现 0.8 也有轻微异方差，可把 "0.8" 加入集合。
HETEROSCEDASTIC_CONTROL_GROUPS = {"1.4", "1.65"}

# 局部残差尺度估计：用于让高转速区间的置信带自然变宽。
LOCAL_SCALE_BINS = 30
LOCAL_SCALE_MIN_SAMPLES_PER_BIN = 50
LOCAL_SCALE_FLOOR_RATIO = 0.20
LOCAL_SCALE_CEIL_RATIO = 8.0
ENFORCE_MONOTONE_WIDENING = True
USE_WEIGHTED_FIT_FOR_HETERO_GROUPS = True
WEIGHT_CLIP_RANGE = (0.05, 20.0)

# Matplotlib 中文字体设置。优先使用 Windows 常见中文字体，其次使用 Linux/macOS 常见中文字体。
CHINESE_FONT_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "SimSun", "KaiTi",
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC",
    "WenQuanYi Micro Hei", "Arial Unicode MS", "PingFang SC", "Heiti SC",
]


def configure_chinese_font() -> None:
    """尽量解决 Matplotlib 静态图片中文乱码和负号显示问题。"""
    matplotlib.rcParams["axes.unicode_minus"] = False

    available_font_names = set()
    for font_path in fm.findSystemFonts(fontext="ttf") + fm.findSystemFonts(fontext="ttc"):
        try:
            available_font_names.add(fm.FontProperties(fname=font_path).get_name())
        except Exception:
            continue

    for font_name in CHINESE_FONT_CANDIDATES:
        if font_name in available_font_names:
            matplotlib.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            matplotlib.rcParams["font.family"] = "sans-serif"
            return

    # 即使当前环境没有可识别中文字体，也给出候选族名。
    # 在 Windows 本机运行时通常能自动命中 Microsoft YaHei 或 SimHei。
    matplotlib.rcParams["font.sans-serif"] = CHINESE_FONT_CANDIDATES + ["DejaVu Sans"]
    matplotlib.rcParams["font.family"] = "sans-serif"
    print("警告：当前 Python 环境未明确识别到中文字体。若图片仍乱码，请安装 SimHei 或 Microsoft YaHei。")


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



def is_heteroscedastic_group(group_name: object) -> bool:
    return str(group_name) in HETEROSCEDASTIC_CONTROL_GROUPS


def estimate_score_center_scale(
    work: pd.DataFrame,
    residual: np.ndarray,
    inlier_mask: pd.Series,
    use_hetero_scale: bool,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict[str, float]]:
    """
    估计残差中心与残差尺度。

    0.8 组：返回全局中心和全局尺度，因此置信带等宽。
    1.4/1.65 组：沿 speed_sq 分箱估计局部尺度 sigma(x)，因此置信带随转速变化。
    """
    x_all = work["speed_sq"].to_numpy(dtype=float)
    r_all = np.asarray(residual, dtype=float)
    mask_all = np.asarray(inlier_mask, dtype=bool) & np.isfinite(x_all) & np.isfinite(r_all)

    if mask_all.sum() == 0:
        mask_all = np.isfinite(x_all) & np.isfinite(r_all)

    ref_resid = r_all[mask_all]
    global_center = float(np.median(ref_resid)) if ref_resid.size else 0.0
    global_sigma = float(robust_mad(ref_resid)) if ref_resid.size else 1.0

    diagnostics = {
        "global_residual_center": global_center,
        "global_residual_sigma": global_sigma,
        "local_scale_used": float(bool(use_hetero_scale)),
    }

    if (not use_hetero_scale) or mask_all.sum() < max(200, 4 * LOCAL_SCALE_MIN_SAMPLES_PER_BIN):
        center = np.full(len(work), global_center, dtype=float)
        scale = np.full(len(work), global_sigma, dtype=float)
        profile = pd.DataFrame({
            "speed_sq_center": [float(np.nanmedian(x_all))],
            "residual_center": [global_center],
            "residual_sigma": [global_sigma],
            "n_bin": [int(mask_all.sum())],
        })
        diagnostics.update({
            "score_scale_method": "global_constant",
            "local_scale_bin_count": 1,
            "local_scale_min": global_sigma,
            "local_scale_median": global_sigma,
            "local_scale_max": global_sigma,
        })
        return center, scale, profile, diagnostics

    ref = pd.DataFrame({
        "speed_sq": x_all[mask_all],
        "residual": r_all[mask_all],
    }).replace([np.inf, -np.inf], np.nan).dropna()

    if len(ref) < max(200, 4 * LOCAL_SCALE_MIN_SAMPLES_PER_BIN):
        center = np.full(len(work), global_center, dtype=float)
        scale = np.full(len(work), global_sigma, dtype=float)
        profile = pd.DataFrame({
            "speed_sq_center": [float(np.nanmedian(x_all))],
            "residual_center": [global_center],
            "residual_sigma": [global_sigma],
            "n_bin": [int(len(ref))],
        })
        diagnostics.update({
            "score_scale_method": "global_constant_fallback",
            "local_scale_bin_count": 1,
            "local_scale_min": global_sigma,
            "local_scale_median": global_sigma,
            "local_scale_max": global_sigma,
        })
        return center, scale, profile, diagnostics

    real_bins = min(LOCAL_SCALE_BINS, max(4, len(ref) // LOCAL_SCALE_MIN_SAMPLES_PER_BIN))
    ref["x_bin"] = qcut_with_fallback(ref["speed_sq"], q=real_bins)

    rows = []
    for _, g in ref.groupby("x_bin", observed=False):
        if len(g) < 10:
            continue
        local_center = float(np.median(g["residual"].to_numpy(dtype=float)))
        local_sigma = float(robust_mad(g["residual"].to_numpy(dtype=float)))
        rows.append({
            "speed_sq_center": float(np.median(g["speed_sq"].to_numpy(dtype=float))),
            "residual_center": local_center,
            "residual_sigma": local_sigma,
            "n_bin": int(len(g)),
        })

    profile = pd.DataFrame(rows).dropna().sort_values("speed_sq_center").reset_index(drop=True)
    if len(profile) < 3:
        center = np.full(len(work), global_center, dtype=float)
        scale = np.full(len(work), global_sigma, dtype=float)
        profile = pd.DataFrame({
            "speed_sq_center": [float(np.nanmedian(x_all))],
            "residual_center": [global_center],
            "residual_sigma": [global_sigma],
            "n_bin": [int(len(ref))],
        })
        diagnostics.update({
            "score_scale_method": "global_constant_fallback",
            "local_scale_bin_count": 1,
            "local_scale_min": global_sigma,
            "local_scale_median": global_sigma,
            "local_scale_max": global_sigma,
        })
        return center, scale, profile, diagnostics

    # 稳定局部尺度，避免某个分箱尺度过小导致误报暴增。对异方差组可选择强制单调变宽。
    floor = max(global_sigma * LOCAL_SCALE_FLOOR_RATIO, 1e-9)
    ceil = max(global_sigma * LOCAL_SCALE_CEIL_RATIO, floor)
    smoothed_sigma = (
        profile["residual_sigma"]
        .rolling(window=3, center=True, min_periods=1)
        .median()
        .clip(lower=floor, upper=ceil)
        .to_numpy(dtype=float)
    )
    if ENFORCE_MONOTONE_WIDENING:
        smoothed_sigma = np.maximum.accumulate(smoothed_sigma)

    profile["residual_sigma_raw"] = profile["residual_sigma"].to_numpy(dtype=float)
    profile["residual_sigma"] = smoothed_sigma

    x_centers = profile["speed_sq_center"].to_numpy(dtype=float)
    center_vals = profile["residual_center"].to_numpy(dtype=float)
    scale_vals = profile["residual_sigma"].to_numpy(dtype=float)

    center = np.interp(x_all, x_centers, center_vals, left=center_vals[0], right=center_vals[-1])
    scale = np.interp(x_all, x_centers, scale_vals, left=scale_vals[0], right=scale_vals[-1])
    scale = np.clip(scale, floor, ceil)

    diagnostics.update({
        "score_scale_method": "local_speed_dependent",
        "local_scale_bin_count": int(len(profile)),
        "local_scale_min": float(np.min(scale_vals)),
        "local_scale_median": float(np.median(scale_vals)),
        "local_scale_max": float(np.max(scale_vals)),
    })
    return center, scale, profile, diagnostics


def fit_huber_once(x: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> HuberRegressor:
    model = HuberRegressor(
        epsilon=HUBER_EPSILON,
        alpha=HUBER_ALPHA,
        fit_intercept=True,
        max_iter=500,
    )
    model.fit(x.reshape(-1, 1), y, sample_weight=sample_weight)
    return model


def iterative_huber_fit(work: pd.DataFrame) -> Tuple[HuberRegressor, pd.DataFrame, Dict[str, float]]:
    if len(work) < 20:
        raise ValueError("可评估样本过少，无法拟合。")

    group_name = str(work["control_group"].iloc[0])
    use_hetero_scale = is_heteroscedastic_group(group_name)
    use_weighted_fit = bool(use_hetero_scale and USE_WEIGHTED_FIT_FOR_HETERO_GROUPS)

    seed_mask = select_seed_by_local_band(work)
    if seed_mask.mean() < 0.2:
        seed_mask = pd.Series(np.ones(len(work), dtype=bool), index=work.index)

    current_inlier = seed_mask.copy().astype(bool)
    diagnostics: Dict[str, float] = {
        "seed_count": int(seed_mask.sum()),
        "seed_ratio": float(seed_mask.mean()),
        "heteroscedastic_enabled": float(use_hetero_scale),
        "weighted_fit_enabled": float(use_weighted_fit),
    }

    model: Optional[HuberRegressor] = None
    latest_profile = pd.DataFrame()
    latest_scale_diag: Dict[str, float] = {}

    for round_idx in range(MAX_HUBER_ROUNDS):
        fit_df = sample_fit_subset(work, current_inlier, FIT_MAX_ROWS_PER_GROUP_MODEL)
        x_fit = fit_df["speed_sq"].to_numpy(dtype=float)
        y_fit = fit_df["response_y"].to_numpy(dtype=float)

        sample_weight = None
        if use_weighted_fit and "fit_weight" in fit_df.columns:
            sample_weight = fit_df["fit_weight"].to_numpy(dtype=float)
            sample_weight = np.where(np.isfinite(sample_weight), sample_weight, 1.0)

        model = fit_huber_once(x_fit, y_fit, sample_weight=sample_weight)

        pred = model.predict(work["speed_sq"].to_numpy(dtype=float).reshape(-1, 1))
        resid = work["response_y"].to_numpy(dtype=float) - pred

        score_center, score_scale, latest_profile, latest_scale_diag = estimate_score_center_scale(
            work=work,
            residual=resid,
            inlier_mask=current_inlier,
            use_hetero_scale=use_hetero_scale,
        )
        score = np.abs(resid - score_center) / (score_scale + 1e-12)
        new_inlier = pd.Series(score <= INLIER_ZMAX, index=work.index)

        work["y_pred"] = pred.astype("float32")
        work["residual"] = resid.astype("float32")
        work["score_center"] = score_center.astype("float32")
        work["score_scale"] = score_scale.astype("float32")
        work["robust_score"] = score.astype("float32")
        work["is_inlier"] = new_inlier.to_numpy()

        if use_weighted_fit:
            weights = 1.0 / np.square(score_scale + 1e-12)
            med_w = np.nanmedian(weights[np.isfinite(weights)])
            if np.isfinite(med_w) and med_w > 0:
                weights = weights / med_w
            weights = np.clip(weights, WEIGHT_CLIP_RANGE[0], WEIGHT_CLIP_RANGE[1])
            work["fit_weight"] = weights.astype("float32")
        else:
            work["fit_weight"] = 1.0

        # 异方差组至少多跑一轮，让第二轮拟合真正使用上一步估计出的 WLS 权重。
        can_stop = (not use_weighted_fit) or (round_idx >= 1)
        if can_stop and new_inlier.equals(current_inlier):
            diagnostics["rounds"] = round_idx + 1
            diagnostics["fit_sample_count"] = len(fit_df)
            diagnostics.update(latest_scale_diag)
            diagnostics["final_center"] = float(np.median(work.loc[new_inlier, "score_center"].to_numpy(dtype=float)))
            diagnostics["final_sigma"] = float(np.median(work.loc[new_inlier, "score_scale"].to_numpy(dtype=float)))
            work.attrs["scale_profile"] = latest_profile
            return model, work, diagnostics

        current_inlier = new_inlier

    if model is None:
        raise RuntimeError("模型拟合失败。")

    diagnostics["rounds"] = MAX_HUBER_ROUNDS
    diagnostics["fit_sample_count"] = int(current_inlier.sum()) if FIT_MAX_ROWS_PER_GROUP_MODEL is None else min(int(current_inlier.sum()), FIT_MAX_ROWS_PER_GROUP_MODEL)
    diagnostics.update(latest_scale_diag)
    diagnostics["final_center"] = float(np.median(work.loc[current_inlier, "score_center"].to_numpy(dtype=float)))
    diagnostics["final_sigma"] = float(np.median(work.loc[current_inlier, "score_scale"].to_numpy(dtype=float)))
    work.attrs["scale_profile"] = latest_profile
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
        qlab = quantile_label(q)
        scored[f"is_anomaly_{qlab}"] = scored["robust_score"] > thr

        # 非等宽置信带的核心输出：
        # residual = y - y_pred，异常条件为 |residual - score_center| / score_scale > thr。
        if {"y_pred", "score_center", "score_scale"}.issubset(scored.columns):
            scored[f"band_lower_{qlab}"] = scored["y_pred"] + scored["score_center"] - thr * scored["score_scale"]
            scored[f"band_upper_{qlab}"] = scored["y_pred"] + scored["score_center"] + thr * scored["score_scale"]
            scored[f"residual_lower_{qlab}"] = scored["score_center"] - thr * scored["score_scale"]
            scored[f"residual_upper_{qlab}"] = scored["score_center"] + thr * scored["score_scale"]
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
        "control_group", "speed_raw", "speed_sq", "response_y", "y_pred", "residual",
        "score_center", "score_scale", "robust_score", f"band_lower_{primary_label}", f"band_upper_{primary_label}",
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
        "heteroscedastic_enabled": [diagnostics.get("heteroscedastic_enabled")],
        "weighted_fit_enabled": [diagnostics.get("weighted_fit_enabled")],
        "score_scale_method": [diagnostics.get("score_scale_method")],
        "local_scale_bin_count": [diagnostics.get("local_scale_bin_count")],
        "local_scale_min": [diagnostics.get("local_scale_min")],
        "local_scale_median": [diagnostics.get("local_scale_median")],
        "local_scale_max": [diagnostics.get("local_scale_max")],
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


def primary_count_summary(scored: pd.DataFrame) -> Dict[str, float]:
    primary_label = quantile_label(PRIMARY_THRESHOLD_QUANTILE)
    anomaly_col = f"is_anomaly_{primary_label}"
    total = int(len(scored))
    anomaly = int(scored[anomaly_col].sum()) if anomaly_col in scored.columns else 0
    normal = total - anomaly
    anomaly_rate = anomaly / total if total else np.nan
    normal_rate = normal / total if total else np.nan
    return {
        "total": total,
        "normal": normal,
        "anomaly": anomaly,
        "normal_rate": normal_rate,
        "anomaly_rate": anomaly_rate,
    }


def format_count_summary(scored: pd.DataFrame) -> str:
    s = primary_count_summary(scored)
    return (
        f"可评估总点数: {s['total']:,}\n"
        f"正常点: {s['normal']:,} ({s['normal_rate']:.2%})\n"
        f"异常点: {s['anomaly']:,} ({s['anomaly_rate']:.2%})\n"
        f"主阈值: {PRIMARY_THRESHOLD_QUANTILE:.4f}"
    )


def add_count_textbox(ax, scored: pd.DataFrame) -> None:
    ax.text(
        0.02, 0.98, format_count_summary(scored),
        transform=ax.transAxes,
        va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.82, edgecolor="gray"),
    )


def _band_lines_for_x(scored: pd.DataFrame, model: HuberRegressor, xs: np.ndarray, q: float = PRIMARY_THRESHOLD_QUANTILE) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    qlab = quantile_label(q)
    base_x = scored["speed_sq"].to_numpy(dtype=float)
    order = np.argsort(base_x)
    base_x = base_x[order]

    center_col = "score_center" if "score_center" in scored.columns else "residual"
    scale_col = "score_scale" if "score_scale" in scored.columns else "robust_score"
    center_vals = scored[center_col].to_numpy(dtype=float)[order]
    scale_vals = scored[scale_col].to_numpy(dtype=float)[order]

    valid = np.isfinite(base_x) & np.isfinite(center_vals) & np.isfinite(scale_vals)
    if valid.sum() < 2:
        y_pred = model.predict(xs.reshape(-1, 1))
        return y_pred, y_pred, y_pred, y_pred

    base_x = base_x[valid]
    center_vals = center_vals[valid]
    scale_vals = scale_vals[valid]

    # 合并重复 x，避免 interp 在重复点处抖动。
    profile = pd.DataFrame({"x": base_x, "center": center_vals, "scale": scale_vals}).groupby("x", as_index=False).median()
    base_x = profile["x"].to_numpy(dtype=float)
    center_vals = profile["center"].to_numpy(dtype=float)
    scale_vals = profile["scale"].to_numpy(dtype=float)

    center = np.interp(xs, base_x, center_vals, left=center_vals[0], right=center_vals[-1])
    scale = np.interp(xs, base_x, scale_vals, left=scale_vals[0], right=scale_vals[-1])
    y_pred = model.predict(xs.reshape(-1, 1))

    # 直接使用已算好的主阈值列时，逻辑与 enrich_threshold_flags 保持一致。
    if f"band_lower_{qlab}" in scored.columns and f"band_upper_{qlab}" in scored.columns:
        lower_vals = scored[f"band_lower_{qlab}"].to_numpy(dtype=float)[order][valid]
        upper_vals = scored[f"band_upper_{qlab}"].to_numpy(dtype=float)[order][valid]
        band_profile = pd.DataFrame({"x": base_x, "lower": lower_vals[:len(base_x)], "upper": upper_vals[:len(base_x)]}) if len(lower_vals) == len(base_x) else None
    thr = np.nanquantile(scored.loc[scored["is_inlier"], "robust_score"].to_numpy(dtype=float), q)
    lower = y_pred + center - thr * scale
    upper = y_pred + center + thr * scale
    return y_pred, lower, upper, scale


def save_missing_bar(missing_summary: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(missing_summary["raw_column"], missing_summary["missing_rate"])
    ax.set_xticklabels(missing_summary["raw_column"], rotation=30, ha="right")
    ax.set_ylabel("缺失率")
    ax.set_title(f"{missing_summary['model_name'].iloc[0]} | 控制组 {missing_summary['control_group'].iloc[0]} | 关键字段缺失率")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_density_fit_png(scored: pd.DataFrame, model: HuberRegressor, model_spec: ModelSpec, out_png: Path) -> None:
    primary_label = quantile_label(PRIMARY_THRESHOLD_QUANTILE)
    anomaly_col = f"is_anomaly_{primary_label}"

    fig, ax = plt.subplots(figsize=(11, 8))
    hb = ax.hexbin(scored["speed_sq"], scored["response_y"], gridsize=120, mincnt=1, bins="log")
    fig.colorbar(hb, ax=ax, label="样本密度(log)")

    x_min, x_max = float(scored["speed_sq"].min()), float(scored["speed_sq"].max())
    xs = np.linspace(x_min, x_max, 400)
    ys, lower, upper, _ = _band_lines_for_x(scored, model, xs)
    ax.plot(xs, ys, linewidth=2.0, label="拟合线")
    ax.plot(xs, lower, linestyle="--", linewidth=1.4, label="下置信带/控制线")
    ax.plot(xs, upper, linestyle="--", linewidth=1.4, label="上置信带/控制线")

    anomalies = scored.loc[scored[anomaly_col]].copy()
    if len(anomalies) > STATIC_ANOMALY_OVERLAY_MAX:
        anomalies = anomalies.sample(n=STATIC_ANOMALY_OVERLAY_MAX, random_state=PLOT_SAMPLE_RANDOM_STATE)
    ax.scatter(anomalies["speed_sq"], anomalies["response_y"], s=10, alpha=0.65, label="异常点")

    ax.set_xlabel("二次侧泵转速平方")
    ax.set_ylabel(model_spec.response_desc)
    ax.set_title(f"{model_spec.display_name} | 控制组 {scored['control_group'].iloc[0]} | 密度图 + 非等宽置信带")
    ax.legend(loc="best")
    add_count_textbox(ax, scored)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_residual_density_png(scored: pd.DataFrame, thresholds: Dict[float, float], model_spec: ModelSpec, out_png: Path) -> None:
    primary_label = quantile_label(PRIMARY_THRESHOLD_QUANTILE)
    anomaly_col = f"is_anomaly_{primary_label}"

    fig, ax = plt.subplots(figsize=(11, 8))
    hb = ax.hexbin(scored["speed_raw"], scored["residual"], gridsize=120, mincnt=1, bins="log")
    fig.colorbar(hb, ax=ax, label="样本密度(log)")

    speed_min, speed_max = float(scored["speed_raw"].min()), float(scored["speed_raw"].max())
    speed_grid = np.linspace(speed_min, speed_max, 400)
    speed_sq_grid = speed_grid ** 2

    # residual 控制线：center(x) ± threshold * sigma(x)
    sorted_df = scored.sort_values("speed_sq")
    profile = sorted_df.groupby("speed_sq", as_index=False)[["score_center", "score_scale"]].median()
    center = np.interp(speed_sq_grid, profile["speed_sq"], profile["score_center"])
    scale = np.interp(speed_sq_grid, profile["speed_sq"], profile["score_scale"])
    primary_thr = thresholds[PRIMARY_THRESHOLD_QUANTILE]
    ax.plot(speed_grid, center + primary_thr * scale, linestyle="--", linewidth=1.4, label="上残差控制线")
    ax.plot(speed_grid, center - primary_thr * scale, linestyle="--", linewidth=1.4, label="下残差控制线")
    ax.axhline(0, linewidth=1.0, alpha=0.6, label="零残差线")

    anomalies = scored.loc[scored[anomaly_col]].copy()
    if len(anomalies) > STATIC_ANOMALY_OVERLAY_MAX:
        anomalies = anomalies.sample(n=STATIC_ANOMALY_OVERLAY_MAX, random_state=PLOT_SAMPLE_RANDOM_STATE)
    ax.scatter(anomalies["speed_raw"], anomalies["residual"], s=10, alpha=0.65, label="异常点")

    ax.set_xlabel("二次侧泵转速")
    ax.set_ylabel("残差")
    ax.set_title(f"{model_spec.display_name} | 控制组 {scored['control_group'].iloc[0]} | 残差-转速图 + 非等宽控制线")
    ax.legend(loc="best")
    add_count_textbox(ax, scored)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_score_hist_png(scored: pd.DataFrame, thresholds: Dict[float, float], model_spec: ModelSpec, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(scored["robust_score"], bins=120)
    for q, thr in thresholds.items():
        ax.axvline(thr, linestyle="--", linewidth=1.4, label=f"阈值 {q:.4f}")
    ax.set_xlabel("标准化异常分数 |残差-局部中心|/局部尺度")
    ax.set_ylabel("样本数")
    ax.set_title(f"{model_spec.display_name} | 控制组 {scored['control_group'].iloc[0]} | 分数分布")
    ax.legend()
    add_count_textbox(ax, scored)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_device_rate_hist_png(device_summary: pd.DataFrame, out_png: Path) -> None:
    primary_label = quantile_label(PRIMARY_THRESHOLD_QUANTILE)
    fig, ax = plt.subplots(figsize=(9, 6))
    vals = device_summary[f"anomaly_rate_{primary_label}"].dropna().to_numpy()
    ax.hist(vals, bins=80)
    ax.set_xlabel("设备异常率")
    ax.set_ylabel("设备数")
    total_devices = len(device_summary)
    alarm_devices = int((device_summary[f"anomaly_count_{primary_label}"] > 0).sum())
    ax.set_title(
        f"{device_summary['model_name'].iloc[0]} | 控制组 {device_summary['control_group'].iloc[0]} | 设备异常率分布\n"
        f"设备总数: {total_devices:,}；存在异常点设备数: {alarm_devices:,} ({alarm_devices / total_devices:.2%})"
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


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
    title_stats = format_count_summary(scored).replace("\n", "；")

    hover_data = {
        "device_name": True,
        "raw_file_stem": True,
        "source_speed_bin": True,
        "row_in_file": True,
        "speed_raw": ":.4f",
        "response_y": ":.6f",
        "y_pred": ":.6f",
        "residual": ":.6f",
        "score_center": ":.6f",
        "score_scale": ":.6f",
        "robust_score": ":.4f",
        f"band_lower_{primary_label}": ":.6f",
        f"band_upper_{primary_label}": ":.6f",
        f"anomaly_rate_{primary_label}": ":.4%",
        f"anomaly_count_{primary_label}": True,
        "total_samples": True,
        "evaluable_samples": True,
        "max_score": ":.4f",
        "p95_score": ":.4f",
        "mean_abs_residual": ":.6f",
    }

    fig = px.scatter(
        sampled,
        x="speed_sq",
        y="response_y",
        color="status",
        hover_data=hover_data,
        title=f"{model_spec.display_name} | 控制组 {scored['control_group'].iloc[0]} | 交互散点图<br>{title_stats}",
        opacity=0.6,
    )
    x_min, x_max = float(sampled["speed_sq"].min()), float(sampled["speed_sq"].max())
    xs = np.linspace(x_min, x_max, 400)
    ys, lower, upper, _ = _band_lines_for_x(scored, model, xs)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="拟合线"))
    fig.add_trace(go.Scatter(x=xs, y=upper, mode="lines", name="上置信带", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=xs, y=lower, mode="lines", name="下置信带", line=dict(dash="dash")))
    fig.write_html(out_html)


def save_interactive_residual(scored: pd.DataFrame, device_summary: pd.DataFrame, model_spec: ModelSpec, out_html: Path) -> None:
    sampled = sample_for_plot(scored, {}, INTERACTIVE_MAX_POINTS)
    sampled = _merge_hover_device_stats(sampled, device_summary)
    primary_label = quantile_label(PRIMARY_THRESHOLD_QUANTILE)
    sampled["status"] = np.where(sampled[f"is_anomaly_{primary_label}"], "异常", "正常")
    title_stats = format_count_summary(scored).replace("\n", "；")

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
            "score_center": ":.6f",
            "score_scale": ":.6f",
            "robust_score": ":.4f",
            f"anomaly_rate_{primary_label}": ":.4%",
            f"anomaly_count_{primary_label}": True,
            "evaluable_samples": True,
            "max_score": ":.4f",
        },
        title=f"{model_spec.display_name} | 控制组 {scored['control_group'].iloc[0]} | 交互残差图<br>{title_stats}",
        opacity=0.6,
    )

    speed_min, speed_max = float(scored["speed_raw"].min()), float(scored["speed_raw"].max())
    speed_grid = np.linspace(speed_min, speed_max, 400)
    speed_sq_grid = speed_grid ** 2
    sorted_df = scored.sort_values("speed_sq")
    profile = sorted_df.groupby("speed_sq", as_index=False)[["score_center", "score_scale"]].median()
    center = np.interp(speed_sq_grid, profile["speed_sq"], profile["score_center"])
    scale = np.interp(speed_sq_grid, profile["speed_sq"], profile["score_scale"])
    thr = float(np.nanquantile(scored.loc[scored["is_inlier"], "robust_score"].to_numpy(dtype=float), PRIMARY_THRESHOLD_QUANTILE))
    fig.add_trace(go.Scatter(x=speed_grid, y=center + thr * scale, mode="lines", name="上残差控制线", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=speed_grid, y=center - thr * scale, mode="lines", name="下残差控制线", line=dict(dash="dash")))
    fig.add_hline(y=0, line_dash="dot")
    fig.write_html(out_html)


def save_interactive_device_scatter(device_summary: pd.DataFrame, out_html: Path) -> None:
    """
    保存设备级交互风险散点图。

    修复点：Plotly 的 marker.size 不能包含 NaN、inf 或负数。
    部分设备可能因为关键字段缺失、转速被排除等原因没有可评估样本，
    这些设备在 device_summary.csv 中仍应保留，但不适合参与设备风险散点图绘制。
    因此这里额外构造 plot_size，并只绘制 evaluable_samples > 0 的设备。
    """
    primary_label = quantile_label(PRIMARY_THRESHOLD_QUANTILE)
    rate_col = f"anomaly_rate_{primary_label}"
    count_col = f"anomaly_count_{primary_label}"

    total_devices = len(device_summary)
    if count_col in device_summary.columns:
        alarm_devices = int((pd.to_numeric(device_summary[count_col], errors="coerce").fillna(0) > 0).sum())
    else:
        alarm_devices = 0

    plot_df = device_summary.copy()

    # 把绘图需要的字段统一转为数值，避免字符串、inf、NaN 传入 Plotly。
    numeric_cols = [
        "total_samples",
        "evaluable_samples",
        count_col,
        rate_col,
        "max_score",
        "p95_score",
        "mean_abs_residual",
        "speed_min",
        "speed_max",
    ]
    for c in numeric_cols:
        if c in plot_df.columns:
            plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")

    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)

    # 只绘制有可评估样本的设备；没有可评估样本的设备仍保留在 CSV 汇总表中。
    if "evaluable_samples" in plot_df.columns:
        plot_df = plot_df.loc[plot_df["evaluable_samples"].fillna(0) > 0].copy()

    # 如果没有任何可绘制设备，输出一个提示 HTML，而不是让程序中断。
    if plot_df.empty:
        out_html.parent.mkdir(parents=True, exist_ok=True)
        out_html.write_text(
            "<html><head><meta charset='utf-8'></head><body>"
            "<p>没有可绘制的设备：所有设备的 evaluable_samples 都为 0，"
            "或设备级绘图字段全部缺失。请检查列名映射、关键字段缺失情况和转速排除区间。</p>"
            "</body></html>",
            encoding="utf-8",
        )
        return

    # y 轴异常率缺失时按 0 处理。
    if rate_col in plot_df.columns:
        plot_df[rate_col] = plot_df[rate_col].fillna(0.0).clip(lower=0.0)
    else:
        plot_df[rate_col] = 0.0

    # Plotly 的 size 不能包含 NaN/inf/负数，因此不要直接使用 max_score。
    if "max_score" in plot_df.columns:
        plot_df["plot_size"] = plot_df["max_score"].fillna(0.0).clip(lower=0.0)
    else:
        plot_df["plot_size"] = 1.0

    # 如果所有设备 max_score 都是 0，给固定点大小，避免图中点不可见或 size 校验失败。
    if (not np.isfinite(plot_df["plot_size"]).all()) or float(plot_df["plot_size"].max()) <= 0:
        plot_df["plot_size"] = 1.0

    # 先排序再取前 N 个，使图中展示风险较高、样本较多的设备。
    sort_cols = [c for c in [rate_col, "max_score", "evaluable_samples"] if c in plot_df.columns]
    if sort_cols:
        ascending = [False] * len(sort_cols)
        plot_df = plot_df.sort_values(sort_cols, ascending=ascending)
    plot_df = plot_df.head(min(DEVICE_SCATTER_TOPN, len(plot_df)))

    hover_data = {
        "device_name": True,
        "total_samples": True,
        "evaluable_samples": True,
        count_col: True,
        rate_col: ":.4%",
        "max_score": ":.4f",
        "p95_score": ":.4f",
        "mean_abs_residual": ":.6f",
        "n_raw_files": True,
        "n_speed_bins": True,
        "speed_min": ":.4f",
        "speed_max": ":.4f",
        "plot_size": False,
    }
    # 删除当前表中不存在的 hover 字段，避免某些模型字段缺失时绘图中断。
    hover_data = {k: v for k, v in hover_data.items() if k in plot_df.columns}

    fig = px.scatter(
        plot_df,
        x="evaluable_samples",
        y=rate_col,
        size="plot_size",
        hover_data=hover_data,
        title=(
            f"{plot_df['model_name'].iloc[0]} | 控制组 {plot_df['control_group'].iloc[0]} | 设备风险散点图"
            f"<br>设备总数: {total_devices:,}；存在异常点设备数: {alarm_devices:,} "
            f"({alarm_devices / total_devices:.2%})"
        ),
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
    scale_profile = scored.attrs.get("scale_profile")
    if isinstance(scale_profile, pd.DataFrame) and len(scale_profile) > 0:
        scale_profile = scale_profile.copy()
        scale_profile["control_group"] = group_name
        scale_profile["model_id"] = model_spec.model_id
        scale_profile["model_name"] = model_spec.display_name
        save_csv(scale_profile, model_out_dir / f"local_scale_profile_{group_slug}.csv")

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
    configure_chinese_font()
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
