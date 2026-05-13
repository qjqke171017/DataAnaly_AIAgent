# -*- coding: utf-8 -*-
"""
GMM 聚类 / 无监督异常检测完整脚本

数据结构假设：
1. 根目录下有 3 个控制压差文件夹，例如 0.8、1.4、1.65；
2. 每个控制压差文件夹下有 5 个按转速划分的子文件夹；
3. 每个子文件夹下有若干 CSV 文件；
4. CSV 文件名中包含“_二次侧”，其前面的字符串为设备名；
5. 本脚本对 0.8、1.4、1.65 三组分别建 GMM，不在聚类时区分设备；
6. 聚类/打分结束后，再按设备汇总异常率、最大异常分数等指标。

使用方法：
只需要修改“0. 用户配置区”中的 GROUP_DIRS、OUTPUT_DIR、GMM 参数和特征参数。
"""

from __future__ import annotations

import gc
import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ==========================================================
# 0. 用户配置区：路径、模型参数、输出参数都在这里改
# ==========================================================

@dataclass
class Config:
    # --------------------------
    # 1) 路径配置
    # --------------------------
    # 方式一：直接写三个控制压差组的绝对路径。推荐。
    GROUP_DIRS: Dict[str, str] = field(default_factory=lambda: {
        "0.8": r"/path/to/控制压差0.8",
        "1.4": r"/path/to/控制压差1.4",
        "1.65": r"/path/to/控制压差1.65",
    })

    OUTPUT_DIR: str = r"/path/to/gmm_output"

    # --------------------------
    # 2) CSV 读取配置
    # --------------------------
    CSV_ENCODINGS: Tuple[str, ...] = ("utf-8", "utf-8-sig", "gbk", "gb18030")
    DEVICE_SPLIT_TOKEN: str = "_二次侧"

    # 如果自动识别列名失败，可在这里强制指定。
    # 例如：{"speed_col": "二次侧泵转速", "valve_open_col": "二次侧阀开度"}
    COLUMN_NAME_OVERRIDES: Dict[str, str] = field(default_factory=dict)

    # --------------------------
    # 3) 特征配置
    # --------------------------
    # None 表示自动选择连续数值型特征；
    # 如果想手动指定，就写：FEATURE_COLUMNS=["二次侧泵转速", "二次侧泵压差", ...]
    FEATURE_COLUMNS: Optional[List[str]] = None

    # 是否加入脚本自动构造的物理派生特征。
    ADD_DERIVED_FEATURES: bool = True

    # 自动选特征时，以下关键词命中的列默认不参与 GMM。
    # 控制压差目标值已经分组，不应再作为聚类特征；结构码一般是类别码，不适合直接作为连续变量。
    AUTO_EXCLUDE_KEYWORDS: Tuple[str, ...] = (
        "时间", "timestamp", "目标值", "结构码", "编号", "代码", "状态", "标签", "异常",
    )

    # 自动选择特征时，缺失率太高的列不参与。
    MAX_FEATURE_MISSING_RATE: float = 0.35

    # 每行至少有多少个非缺失特征才进入 GMM。
    MIN_NON_MISSING_FEATURES_PER_ROW: int = 3

    # 可选：排除某个转速范围。默认不排除。
    # 如果仍想沿用以前逻辑，可设为 (95.0, 100.0)。
    EXCLUDE_SPEED_RANGE: Optional[Tuple[float, float]] = None

    # --------------------------
    # 4) GMM 模型配置
    # --------------------------
    # 通过 BIC/AIC 在候选 K 中选择最优成分数。
    N_COMPONENTS_CANDIDATES: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)

    # 协方差类型："full" 表达能力强，但大样本高维时更慢；"diag" 更稳更快。
    COVARIANCE_TYPE: str = "full"

    # 防止协方差矩阵奇异。
    REG_COVAR: float = 1e-6

    GMM_N_INIT: int = 3
    GMM_MAX_ITER: int = 300
    RANDOM_STATE: int = 42

    # GMM 在 PCA 后的空间中拟合，通常更稳定。
    USE_PCA_FOR_GMM: bool = True
    # None 表示自动确定：最多 8 维，且不超过原始特征数。
    GMM_PCA_N_COMPONENTS: Optional[int] = None
    GMM_PCA_MAX_COMPONENTS_AUTO: int = 8

    # 大数据时，不用全部样本选 K / 训练 GMM，避免太慢。
    # 但训练好后，会对全部可评估样本打分。
    BIC_SAMPLE_MAX_ROWS: int = 120_000
    GMM_TRAIN_MAX_ROWS: int = 300_000

    # 异常阈值：根据 anomaly_score = -log p(x) 的分位数确定。
    PRIMARY_QUANTILE: float = 0.9975
    EXTRA_QUANTILES: Tuple[float, ...] = (0.995, 0.999)

    # --------------------------
    # 5) 输出与绘图配置
    # --------------------------
    SAVE_FULL_SCORED_SAMPLES: bool = False  # 全量样本得分文件可能很大，默认不保存。
    TOP_ANOMALY_SAMPLES: int = 3000
    PLOT_SAMPLE_MAX_ROWS: int = 70_000
    DEVICE_RISK_TOPN: int = 800
    CLUSTER_PROFILE_TOP_FEATURES: int = 20

    # 生成“转速平方-关键变量”散点图时，最多展示几个关键 y 变量。
    MAX_KEY_SCATTER_FEATURES: int = 8


CFG = Config()


# ==========================================================
# 1. 中文字体与通用工具
# ==========================================================

CHINESE_FONT_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "SimSun", "KaiTi",
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC",
    "WenQuanYi Micro Hei", "Arial Unicode MS", "PingFang SC", "Heiti SC",
]


def configure_chinese_font() -> None:
    """尽量保证 Matplotlib 静态图中中文和负号正常显示。"""
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

    matplotlib.rcParams["font.sans-serif"] = CHINESE_FONT_CANDIDATES + ["DejaVu Sans"]
    matplotlib.rcParams["font.family"] = "sans-serif"
    print("警告：未明确识别到中文字体。若静态图中文乱码，请安装 SimHei 或 Microsoft YaHei。")


def plotly_chinese_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        font=dict(family="Microsoft YaHei, SimHei, Noto Sans CJK SC, Arial Unicode MS, sans-serif"),
        template="plotly_white",
    )
    return fig


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_read_csv(path: Path, encodings: Sequence[str]) -> pd.DataFrame:
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"读取 CSV 失败: {path}\n最后一个错误: {last_error}")


def find_csv_files(root_dir: str) -> List[Path]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"控制组路径不存在: {root}")
    files = sorted([p for p in root.rglob("*.csv") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"路径下未找到 CSV 文件: {root}")
    return files


def extract_device_name(file_stem: str, token: str) -> str:
    idx = file_stem.find(token)
    if idx >= 0:
        return file_stem[:idx]
    return file_stem


def sanitize_filename(text: object) -> str:
    return str(text).replace("/", "__").replace("\\", "__").replace(":", "_").replace(" ", "_")


def quantile_label(q: float) -> str:
    s = f"{q:.4f}".rstrip("0").rstrip(".")
    return "q" + s.replace(".", "")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    safe_mkdir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def sample_df(df: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=random_state).copy()


def numeric_series(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if col and col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def mean_existing(df: pd.DataFrame, cols: Sequence[Optional[str]]) -> pd.Series:
    real_cols = [c for c in cols if c and c in df.columns]
    if not real_cols:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    tmp = pd.concat([pd.to_numeric(df[c], errors="coerce") for c in real_cols], axis=1)
    return tmp.mean(axis=1)


# ==========================================================
# 2. 列名识别与派生特征
# ==========================================================

COLUMN_CANDIDATES: Dict[str, List[str]] = {
    # 一次侧
    "pri_inlet_temp_col": ["一次侧入口温度", "一次入口温度"],
    "pri_filter_inlet_pressure_col": ["一次侧过滤器入口压力", "一次过滤器入口压力"],
    "pri_filter_outlet_pressure_col": ["一次侧过滤器出口压力", "一次过滤器出口压力"],
    "pri_outlet_pressure_col": ["一次侧出口压力", "一次出口压力"],
    "pri_valve_open_col": ["一次侧阀开度", "一次阀开度"],
    "pri_sr_dp_col": ["一次侧供回水压差", "一次供回水压差"],
    "pri_filter_dp_col": ["一次侧过滤器压差", "一次过滤器压差"],
    "pri_hex_dp_col": ["一次侧板换压差", "一次板换压差"],
    "pri_pipe_ratio_col": ["一次侧管阻比", "一次管阻比"],
    "pri_flow_col": ["一次侧流量", "一次流量"],
    "pri_struct_code_col": ["一次侧结构码", "一次结构码"],

    # 二次侧
    "sec_inlet_pressure1_col": ["二次侧入口压力1", "二次入口压力1"],
    "sec_inlet_pressure2_col": ["二次侧入口压力2", "二次入口压力2"],
    "sec_inlet_temp_col": ["二次侧入口温度", "二次入口温度"],
    "sec_outlet_pressure1_col": ["二次侧出口压力1", "二次出口压力1"],
    "sec_outlet_pressure2_col": ["二次侧出口压力2", "二次出口压力2"],
    "sec_outlet_temp1_col": ["二次侧出口温度1", "二次出口温度1"],
    "sec_outlet_temp2_col": ["二次侧出口温度2", "二次出口温度2"],
    "sec_pump_inlet_pressure1_col": ["二次侧泵入口压力1", "二次泵入口压力1"],
    "sec_pump_inlet_pressure2_col": ["二次侧泵入口压力2", "二次泵入口压力2"],
    "sec_pump_outlet_pressure_col": ["二次侧泵出口压力", "二次泵出口压力"],
    "speed_col": ["二次侧泵转速", "泵转速", "二次侧转速", "二次泵转速"],
    "valve_open_col": ["二次侧阀开度", "二次阀开度", "阀开度"],
    "control_pressure_col": ["控制压差目标值", "控制压差目标", "控制压差"],
    "control_temperature_col": ["控制温度目标值", "控制温度目标", "控制温度"],
    "sec_sr_dp_col": ["二次侧供回水压差", "二次供回水压差"],
    "sec_hex_dp_col": ["二次侧板换压差", "二次板换压差"],
    "sec_pump_dp_col": ["二次侧泵压差", "二次泵压差"],
    "sec_filter_dp_col": ["二次侧过滤器压差", "二次过滤器压差", "二次侧过滤压差"],
    "sec_pipe_ratio_col": ["二次侧管阻比", "二次管阻比"],
    "sec_flow_col": ["二次侧流量", "二次流量"],
    "sec_struct_code_col": ["二次侧结构码", "二次结构码"],
    "timestamp_col": ["时间戳", "timestamp", "时间", "采样时间"],
}


def choose_existing_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    col_set = set(columns)
    for cand in candidates:
        if cand in col_set:
            return cand

    normalized = {str(c).strip().lower().replace(" ", ""): c for c in columns}
    for cand in candidates:
        key = str(cand).strip().lower().replace(" ", "")
        if key in normalized:
            return normalized[key]
    return None


def resolve_columns(columns: Sequence[str], overrides: Dict[str, str]) -> Dict[str, Optional[str]]:
    resolved: Dict[str, Optional[str]] = {}
    cols = list(columns)
    for role, candidates in COLUMN_CANDIDATES.items():
        if role in overrides and overrides[role] in cols:
            resolved[role] = overrides[role]
        else:
            resolved[role] = choose_existing_column(cols, candidates)
    return resolved


def add_derived_features(df: pd.DataFrame, resolved_cols: Dict[str, Optional[str]]) -> pd.DataFrame:
    """加入一些具有物理意义的派生特征。能构造则构造，不能构造则自动跳过。"""
    out = df.copy()

    speed = numeric_series(out, resolved_cols.get("speed_col"))
    if speed.notna().any():
        out["派生_二次侧泵转速平方"] = speed ** 2

    sec_pump_inlet_mean = mean_existing(out, [
        resolved_cols.get("sec_pump_inlet_pressure1_col"),
        resolved_cols.get("sec_pump_inlet_pressure2_col"),
    ])
    if sec_pump_inlet_mean.notna().any():
        out["派生_二次侧泵入口压力均值"] = sec_pump_inlet_mean

    sec_inlet_mean = mean_existing(out, [
        resolved_cols.get("sec_inlet_pressure1_col"),
        resolved_cols.get("sec_inlet_pressure2_col"),
    ])
    if sec_inlet_mean.notna().any():
        out["派生_二次侧入口压力均值"] = sec_inlet_mean

    sec_outlet_mean = mean_existing(out, [
        resolved_cols.get("sec_outlet_pressure1_col"),
        resolved_cols.get("sec_outlet_pressure2_col"),
    ])
    if sec_outlet_mean.notna().any():
        out["派生_二次侧出口压力均值"] = sec_outlet_mean

    sec_pump_outlet = numeric_series(out, resolved_cols.get("sec_pump_outlet_pressure_col"))
    if sec_pump_outlet.notna().any() and sec_pump_inlet_mean.notna().any():
        out["派生_二次侧泵压差_泵出口减泵入口均值"] = sec_pump_outlet - sec_pump_inlet_mean

    sec_hex_dp = numeric_series(out, resolved_cols.get("sec_hex_dp_col"))
    sec_filter_dp = numeric_series(out, resolved_cols.get("sec_filter_dp_col"))
    sec_sr_dp = numeric_series(out, resolved_cols.get("sec_sr_dp_col"))
    if sec_hex_dp.notna().any() and sec_filter_dp.notna().any():
        out["派生_二次侧板换压差加过滤器压差"] = sec_hex_dp + sec_filter_dp
    if sec_sr_dp.notna().any() and sec_hex_dp.notna().any():
        out["派生_二次侧供回水压差加板换压差"] = sec_sr_dp + sec_hex_dp

    sec_outlet_temp_mean = mean_existing(out, [
        resolved_cols.get("sec_outlet_temp1_col"),
        resolved_cols.get("sec_outlet_temp2_col"),
    ])
    sec_inlet_temp = numeric_series(out, resolved_cols.get("sec_inlet_temp_col"))
    if sec_outlet_temp_mean.notna().any() and sec_inlet_temp.notna().any():
        out["派生_二次侧出口入口温差"] = sec_outlet_temp_mean - sec_inlet_temp

    pri_hex_dp = numeric_series(out, resolved_cols.get("pri_hex_dp_col"))
    pri_filter_dp = numeric_series(out, resolved_cols.get("pri_filter_dp_col"))
    pri_sr_dp = numeric_series(out, resolved_cols.get("pri_sr_dp_col"))
    if pri_hex_dp.notna().any() and pri_filter_dp.notna().any():
        out["派生_一次侧板换压差加过滤器压差"] = pri_hex_dp + pri_filter_dp
    if pri_sr_dp.notna().any() and pri_hex_dp.notna().any():
        out["派生_一次侧供回水压差加板换压差"] = pri_sr_dp + pri_hex_dp

    return out


def load_group_table(group_name: str, group_dir: str, cfg: Config) -> pd.DataFrame:
    csv_files = find_csv_files(group_dir)
    root = Path(group_dir).resolve()
    frames: List[pd.DataFrame] = []

    for idx, fp in enumerate(csv_files, start=1):
        try:
            df = safe_read_csv(fp, cfg.CSV_ENCODINGS)
        except Exception as exc:
            print(f"警告：跳过读取失败文件: {fp}\n{exc}")
            continue

        df.columns = [str(c).strip() for c in df.columns]
        rel = fp.resolve().relative_to(root)
        speed_bin_folder = rel.parts[0] if len(rel.parts) >= 2 else "speed_bin_unknown"
        device_name = extract_device_name(fp.stem, cfg.DEVICE_SPLIT_TOKEN)

        df["device_name"] = device_name
        df["raw_file_stem"] = fp.stem
        df["source_speed_bin"] = speed_bin_folder
        df["source_relpath"] = rel.as_posix()
        df["row_in_file"] = np.arange(len(df), dtype=np.int32)
        df["control_group"] = str(group_name)
        frames.append(df)

        if idx % 100 == 0:
            print(f"  已读取 {idx}/{len(csv_files)} 个 CSV")

    if not frames:
        raise RuntimeError(f"控制组 {group_name} 未成功读取任何 CSV。")

    data = pd.concat(frames, ignore_index=True)
    for c in ["device_name", "raw_file_stem", "source_speed_bin", "source_relpath", "control_group"]:
        data[c] = data[c].astype("category")

    return data


def auto_select_feature_columns(df: pd.DataFrame, cfg: Config) -> Tuple[List[str], pd.DataFrame]:
    metadata_cols = {"device_name", "raw_file_stem", "source_speed_bin", "source_relpath", "row_in_file", "control_group"}
    rows = []
    selected = []

    for col in df.columns:
        if col in metadata_cols:
            continue
        col_str = str(col)
        if any(k.lower() in col_str.lower() for k in cfg.AUTO_EXCLUDE_KEYWORDS):
            rows.append({"feature": col, "selected": False, "reason": "命中排除关键词", "missing_rate": np.nan, "numeric_rate": np.nan, "n_unique": np.nan})
            continue

        s = pd.to_numeric(df[col], errors="coerce")
        numeric_rate = float(s.notna().mean()) if len(s) else 0.0
        missing_rate = float(s.isna().mean()) if len(s) else 1.0
        n_unique = int(s.dropna().nunique())

        reason = "保留"
        keep = True
        if numeric_rate <= 0:
            keep = False
            reason = "无法转为数值"
        elif missing_rate > cfg.MAX_FEATURE_MISSING_RATE:
            keep = False
            reason = f"缺失率过高>{cfg.MAX_FEATURE_MISSING_RATE:.0%}"
        elif n_unique <= 1:
            keep = False
            reason = "近似常量/唯一值过少"

        if keep:
            selected.append(col)

        rows.append({
            "feature": col,
            "selected": keep,
            "reason": reason,
            "missing_rate": missing_rate,
            "numeric_rate": numeric_rate,
            "n_unique": n_unique,
        })

    return selected, pd.DataFrame(rows)


def prepare_feature_table(group_df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, List[str], pd.DataFrame, Dict[str, Optional[str]]]:
    resolved_cols = resolve_columns(group_df.columns, cfg.COLUMN_NAME_OVERRIDES)
    data = add_derived_features(group_df, resolved_cols) if cfg.ADD_DERIVED_FEATURES else group_df.copy()

    if cfg.FEATURE_COLUMNS is None:
        feature_cols, feature_report = auto_select_feature_columns(data, cfg)
    else:
        feature_cols = [c for c in cfg.FEATURE_COLUMNS if c in data.columns]
        missing = [c for c in cfg.FEATURE_COLUMNS if c not in data.columns]
        feature_report = pd.DataFrame({
            "feature": list(cfg.FEATURE_COLUMNS),
            "selected": [c in feature_cols for c in cfg.FEATURE_COLUMNS],
            "reason": ["手动指定" if c in feature_cols else "手动指定但原表不存在" for c in cfg.FEATURE_COLUMNS],
        })
        if missing:
            print("警告：以下手动指定特征在数据中不存在，已跳过：", missing)

    if not feature_cols:
        raise ValueError("没有可用于 GMM 的特征。请检查列名、缺失率或手动设置 FEATURE_COLUMNS。")

    # 特征全部转数值。
    feature_num = data[feature_cols].apply(pd.to_numeric, errors="coerce")
    non_missing_count = feature_num.notna().sum(axis=1)
    valid_mask = non_missing_count >= cfg.MIN_NON_MISSING_FEATURES_PER_ROW

    # 可选排除转速范围。
    speed_col = resolved_cols.get("speed_col")
    if cfg.EXCLUDE_SPEED_RANGE is not None and speed_col and speed_col in data.columns:
        lo, hi = cfg.EXCLUDE_SPEED_RANGE
        speed = pd.to_numeric(data[speed_col], errors="coerce")
        valid_mask &= ~speed.between(lo, hi, inclusive="both").fillna(False)

    meta_cols = ["device_name", "raw_file_stem", "source_speed_bin", "source_relpath", "row_in_file", "control_group"]
    work = data.loc[valid_mask, meta_cols].copy().reset_index(drop=True)
    for c in feature_cols:
        work[c] = pd.to_numeric(data.loc[valid_mask, c], errors="coerce").to_numpy(dtype="float64")

    if len(work) < 50:
        raise ValueError(f"可用于 GMM 的样本太少：{len(work)}。请放宽缺失率或减少特征。")

    return work, feature_cols, feature_report, resolved_cols


# ==========================================================
# 3. GMM 训练、选 K、打分
# ==========================================================

@dataclass
class FittedGMMResult:
    preprocess: Pipeline
    pca_for_gmm: Optional[PCA]
    pca_for_plot: PCA
    gmm: GaussianMixture
    bic_table: pd.DataFrame
    selected_k: int
    gmm_feature_dim: int
    gmm_pca_explained_variance_sum: Optional[float]


def make_preprocess_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def decide_gmm_pca_components(n_features: int, n_samples: int, cfg: Config) -> int:
    if not cfg.USE_PCA_FOR_GMM:
        return n_features
    if cfg.GMM_PCA_N_COMPONENTS is not None:
        return max(1, min(cfg.GMM_PCA_N_COMPONENTS, n_features, n_samples - 1))
    return max(1, min(cfg.GMM_PCA_MAX_COMPONENTS_AUTO, n_features, n_samples - 1))


def fit_one_gmm_k(x: np.ndarray, k: int, cfg: Config, covariance_type: Optional[str] = None) -> GaussianMixture:
    return GaussianMixture(
        n_components=k,
        covariance_type=covariance_type or cfg.COVARIANCE_TYPE,
        reg_covar=cfg.REG_COVAR,
        n_init=cfg.GMM_N_INIT,
        max_iter=cfg.GMM_MAX_ITER,
        random_state=cfg.RANDOM_STATE,
    ).fit(x)


def fit_gmm_with_bic(work: pd.DataFrame, feature_cols: List[str], cfg: Config) -> FittedGMMResult:
    x_raw = work[feature_cols].to_numpy(dtype="float64")

    # 训练/选 K 用抽样，避免样本过多导致耗时过长。
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    n = len(work)
    bic_sample_n = min(n, cfg.BIC_SAMPLE_MAX_ROWS)
    train_sample_n = min(n, cfg.GMM_TRAIN_MAX_ROWS)

    bic_idx = rng.choice(n, size=bic_sample_n, replace=False) if bic_sample_n < n else np.arange(n)
    train_idx = rng.choice(n, size=train_sample_n, replace=False) if train_sample_n < n else np.arange(n)

    preprocess = make_preprocess_pipeline()
    preprocess.fit(x_raw[train_idx])

    x_bic_std = preprocess.transform(x_raw[bic_idx])
    x_train_std = preprocess.transform(x_raw[train_idx])

    # GMM 使用 PCA 空间；二维可视化另用 pca_for_plot。
    pca_for_gmm = None
    if cfg.USE_PCA_FOR_GMM:
        n_comp = decide_gmm_pca_components(x_train_std.shape[1], x_train_std.shape[0], cfg)
        pca_for_gmm = PCA(n_components=n_comp, random_state=cfg.RANDOM_STATE)
        pca_for_gmm.fit(x_train_std)
        x_bic_gmm = pca_for_gmm.transform(x_bic_std)
        x_train_gmm = pca_for_gmm.transform(x_train_std)
        ev_sum = float(np.sum(pca_for_gmm.explained_variance_ratio_))
    else:
        x_bic_gmm = x_bic_std
        x_train_gmm = x_train_std
        ev_sum = None

    rows = []
    fitted_by_k: Dict[int, GaussianMixture] = {}
    max_k_allowed = max(1, min(max(cfg.N_COMPONENTS_CANDIDATES), len(x_bic_gmm) - 1))
    candidates = [k for k in cfg.N_COMPONENTS_CANDIDATES if 1 <= k <= max_k_allowed]

    for k in candidates:
        try:
            gmm_k = fit_one_gmm_k(x_bic_gmm, k, cfg)
            bic = float(gmm_k.bic(x_bic_gmm))
            aic = float(gmm_k.aic(x_bic_gmm))
            converged = bool(gmm_k.converged_)
            fitted_by_k[k] = gmm_k
        except Exception as exc:
            print(f"  警告：K={k} 使用 covariance_type={cfg.COVARIANCE_TYPE} 失败，尝试 diag。错误：{exc}")
            try:
                gmm_k = fit_one_gmm_k(x_bic_gmm, k, cfg, covariance_type="diag")
                bic = float(gmm_k.bic(x_bic_gmm))
                aic = float(gmm_k.aic(x_bic_gmm))
                converged = bool(gmm_k.converged_)
                fitted_by_k[k] = gmm_k
            except Exception as exc2:
                print(f"  警告：K={k} 使用 diag 仍失败，跳过。错误：{exc2}")
                bic = np.nan
                aic = np.nan
                converged = False

        rows.append({"n_components": k, "bic": bic, "aic": aic, "converged": converged})

    bic_table = pd.DataFrame(rows)
    valid_bic = bic_table.dropna(subset=["bic"])
    if valid_bic.empty:
        raise RuntimeError("所有 GMM 候选 K 均拟合失败。请减少特征、改用 covariance_type='diag' 或增大 REG_COVAR。")

    selected_k = int(valid_bic.sort_values("bic").iloc[0]["n_components"])

    # 用选定 K 在训练抽样上重新拟合最终 GMM。
    try:
        final_gmm = fit_one_gmm_k(x_train_gmm, selected_k, cfg)
    except Exception:
        final_gmm = fit_one_gmm_k(x_train_gmm, selected_k, cfg, covariance_type="diag")

    # 二维 PCA 用于可视化。若只有 1 个特征，补一个零列。
    if x_train_std.shape[1] >= 2:
        pca_for_plot = PCA(n_components=2, random_state=cfg.RANDOM_STATE).fit(x_train_std)
    else:
        # 兼容 1 维特征的极端情况。
        pca_for_plot = PCA(n_components=1, random_state=cfg.RANDOM_STATE).fit(x_train_std)

    return FittedGMMResult(
        preprocess=preprocess,
        pca_for_gmm=pca_for_gmm,
        pca_for_plot=pca_for_plot,
        gmm=final_gmm,
        bic_table=bic_table,
        selected_k=selected_k,
        gmm_feature_dim=x_train_gmm.shape[1],
        gmm_pca_explained_variance_sum=ev_sum,
    )


def transform_for_gmm(x_raw: np.ndarray, fit: FittedGMMResult) -> Tuple[np.ndarray, np.ndarray]:
    x_std = fit.preprocess.transform(x_raw)
    if fit.pca_for_gmm is not None:
        x_gmm = fit.pca_for_gmm.transform(x_std)
    else:
        x_gmm = x_std
    return x_std, x_gmm


def score_all_samples(work: pd.DataFrame, feature_cols: List[str], fit: FittedGMMResult, cfg: Config) -> pd.DataFrame:
    x_raw = work[feature_cols].to_numpy(dtype="float64")
    x_std, x_gmm = transform_for_gmm(x_raw, fit)

    log_likelihood = fit.gmm.score_samples(x_gmm)
    anomaly_score = -log_likelihood
    cluster = fit.gmm.predict(x_gmm)
    proba = fit.gmm.predict_proba(x_gmm)
    max_proba = np.max(proba, axis=1)

    # 二维 PCA 坐标。
    pca_plot = fit.pca_for_plot.transform(x_std)
    if pca_plot.shape[1] == 1:
        pca1 = pca_plot[:, 0]
        pca2 = np.zeros_like(pca1)
    else:
        pca1 = pca_plot[:, 0]
        pca2 = pca_plot[:, 1]

    scored = work.copy()
    scored["gmm_cluster"] = cluster.astype(int)
    scored["gmm_log_likelihood"] = log_likelihood.astype("float64")
    scored["gmm_anomaly_score"] = anomaly_score.astype("float64")
    scored["gmm_max_posterior_prob"] = max_proba.astype("float64")
    scored["pca1"] = pca1.astype("float64")
    scored["pca2"] = pca2.astype("float64")

    all_q = sorted(set([cfg.PRIMARY_QUANTILE, *cfg.EXTRA_QUANTILES]))
    for q in all_q:
        thr = float(np.quantile(scored["gmm_anomaly_score"].to_numpy(dtype=float), q))
        qlab = quantile_label(q)
        scored[f"threshold_{qlab}"] = thr
        scored[f"is_anomaly_{qlab}"] = scored["gmm_anomaly_score"] > thr

    return scored


# ==========================================================
# 4. 汇总表
# ==========================================================

def cluster_entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return np.nan
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def build_device_summary(scored: pd.DataFrame, fit: FittedGMMResult, cfg: Config) -> pd.DataFrame:
    primary_label = quantile_label(cfg.PRIMARY_QUANTILE)
    anomaly_col = f"is_anomaly_{primary_label}"

    base = scored.groupby("device_name", observed=False).agg(
        evaluable_samples=("device_name", "size"),
        n_raw_files=("raw_file_stem", "nunique"),
        n_speed_bins=("source_speed_bin", "nunique"),
        mean_score=("gmm_anomaly_score", "mean"),
        median_score=("gmm_anomaly_score", "median"),
        max_score=("gmm_anomaly_score", "max"),
        p95_score=("gmm_anomaly_score", lambda s: float(np.quantile(s, 0.95)) if len(s) else np.nan),
        mean_max_posterior=("gmm_max_posterior_prob", "mean"),
        min_max_posterior=("gmm_max_posterior_prob", "min"),
        pca1_mean=("pca1", "mean"),
        pca2_mean=("pca2", "mean"),
    ).reset_index()

    for q in sorted(set([cfg.PRIMARY_QUANTILE, *cfg.EXTRA_QUANTILES])):
        qlab = quantile_label(q)
        col = f"is_anomaly_{qlab}"
        tmp = scored.groupby("device_name", observed=False).agg(
            **{f"anomaly_count_{qlab}": (col, "sum")}
        ).reset_index()
        base = base.merge(tmp, on="device_name", how="left")
        base[f"anomaly_count_{qlab}"] = base[f"anomaly_count_{qlab}"].fillna(0).astype(int)
        base[f"anomaly_rate_{qlab}"] = base[f"anomaly_count_{qlab}"] / base["evaluable_samples"].replace(0, np.nan)

    # 每个设备的主导簇、簇熵、各簇占比。
    cluster_counts = pd.crosstab(scored["device_name"], scored["gmm_cluster"])
    cluster_counts.columns = [f"cluster_{int(c)}_count" for c in cluster_counts.columns]
    cluster_counts = cluster_counts.reset_index()
    base = base.merge(cluster_counts, on="device_name", how="left")

    cluster_count_cols = [c for c in base.columns if c.startswith("cluster_") and c.endswith("_count")]
    for c in cluster_count_cols:
        base[c] = base[c].fillna(0).astype(int)
        rate_col = c.replace("_count", "_rate")
        base[rate_col] = base[c] / base["evaluable_samples"].replace(0, np.nan)

    if cluster_count_cols:
        count_matrix = base[cluster_count_cols].to_numpy(dtype=float)
        base["cluster_entropy"] = [cluster_entropy_from_counts(row) for row in count_matrix]
        dominant_idx = np.argmax(count_matrix, axis=1)
        dominant_cols = np.array(cluster_count_cols)[dominant_idx]
        base["dominant_cluster"] = [int(re.search(r"cluster_(\d+)_count", c).group(1)) for c in dominant_cols]
        base["dominant_cluster_rate"] = np.max(count_matrix, axis=1) / base["evaluable_samples"].replace(0, np.nan).to_numpy(dtype=float)
    else:
        base["cluster_entropy"] = np.nan
        base["dominant_cluster"] = np.nan
        base["dominant_cluster_rate"] = np.nan

    base["control_group"] = str(scored["control_group"].iloc[0])
    base["selected_k"] = fit.selected_k
    base = base.sort_values([f"anomaly_rate_{primary_label}", "max_score", "evaluable_samples"], ascending=[False, False, False]).reset_index(drop=True)
    return base


def build_cluster_summary(scored: pd.DataFrame, feature_cols: List[str], cfg: Config) -> pd.DataFrame:
    primary_label = quantile_label(cfg.PRIMARY_QUANTILE)
    anomaly_col = f"is_anomaly_{primary_label}"

    rows = []
    total = len(scored)
    for cl, g in scored.groupby("gmm_cluster", observed=False):
        row = {
            "control_group": str(scored["control_group"].iloc[0]),
            "gmm_cluster": int(cl),
            "sample_count": int(len(g)),
            "sample_rate": len(g) / total if total else np.nan,
            "anomaly_count": int(g[anomaly_col].sum()),
            "anomaly_rate": float(g[anomaly_col].mean()),
            "mean_score": float(g["gmm_anomaly_score"].mean()),
            "median_score": float(g["gmm_anomaly_score"].median()),
            "p95_score": float(np.quantile(g["gmm_anomaly_score"], 0.95)),
            "mean_max_posterior": float(g["gmm_max_posterior_prob"].mean()),
        }
        # 原始特征均值，方便解释簇。
        for feat in feature_cols:
            row[f"mean__{feat}"] = float(pd.to_numeric(g[feat], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("gmm_cluster").reset_index(drop=True)


def build_model_summary(scored: pd.DataFrame, fit: FittedGMMResult, feature_cols: List[str], cfg: Config) -> pd.DataFrame:
    primary_label = quantile_label(cfg.PRIMARY_QUANTILE)
    out = {
        "control_group": [str(scored["control_group"].iloc[0])],
        "n_samples": [len(scored)],
        "n_features": [len(feature_cols)],
        "feature_columns": [";".join(feature_cols)],
        "selected_k_by_bic": [fit.selected_k],
        "covariance_type": [fit.gmm.covariance_type],
        "gmm_converged": [bool(fit.gmm.converged_)],
        "gmm_n_iter": [int(fit.gmm.n_iter_)],
        "gmm_feature_dim": [fit.gmm_feature_dim],
        "gmm_pca_explained_variance_sum": [fit.gmm_pca_explained_variance_sum],
        "score_mean": [float(scored["gmm_anomaly_score"].mean())],
        "score_median": [float(scored["gmm_anomaly_score"].median())],
        "score_p95": [float(np.quantile(scored["gmm_anomaly_score"], 0.95))],
        "score_max": [float(scored["gmm_anomaly_score"].max())],
    }
    for q in sorted(set([cfg.PRIMARY_QUANTILE, *cfg.EXTRA_QUANTILES])):
        qlab = quantile_label(q)
        thr = float(scored[f"threshold_{qlab}"].iloc[0])
        out[f"threshold_{qlab}"] = [thr]
        out[f"anomaly_count_{qlab}"] = [int(scored[f"is_anomaly_{qlab}"].sum())]
        out[f"anomaly_rate_{qlab}"] = [float(scored[f"is_anomaly_{qlab}"].mean())]
    return pd.DataFrame(out)


def build_top_anomaly_samples(scored: pd.DataFrame, feature_cols: List[str], cfg: Config) -> pd.DataFrame:
    primary_label = quantile_label(cfg.PRIMARY_QUANTILE)
    keep_cols = [
        "control_group", "device_name", "raw_file_stem", "source_speed_bin", "source_relpath", "row_in_file",
        "gmm_cluster", "gmm_anomaly_score", "gmm_log_likelihood", "gmm_max_posterior_prob",
        "pca1", "pca2", f"is_anomaly_{primary_label}",
    ] + feature_cols
    return scored.sort_values("gmm_anomaly_score", ascending=False).head(cfg.TOP_ANOMALY_SAMPLES)[keep_cols].copy()


# ==========================================================
# 5. 可视化
# ==========================================================

def save_feature_missing_bar(feature_report: pd.DataFrame, out_png: Path, title: str) -> None:
    if "missing_rate" not in feature_report.columns:
        return
    plot_df = feature_report.dropna(subset=["missing_rate"]).copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values("missing_rate", ascending=False).head(40)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(plot_df["feature"].astype(str), plot_df["missing_rate"].astype(float))
    ax.set_ylabel("缺失率")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=60)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_bic_aic_plot(bic_table: pd.DataFrame, selected_k: int, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bic_table["n_components"], bic_table["bic"], marker="o", label="BIC")
    ax.plot(bic_table["n_components"], bic_table["aic"], marker="s", label="AIC")
    ax.axvline(selected_k, linestyle="--", linewidth=1.2, label=f"选择 K={selected_k}")
    ax.set_xlabel("GMM 成分数 K")
    ax.set_ylabel("信息准则值，越小越好")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_score_hist(scored: pd.DataFrame, cfg: Config, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(scored["gmm_anomaly_score"], bins=120)
    for q in sorted(set([cfg.PRIMARY_QUANTILE, *cfg.EXTRA_QUANTILES])):
        qlab = quantile_label(q)
        thr = float(scored[f"threshold_{qlab}"].iloc[0])
        ax.axvline(thr, linestyle="--", linewidth=1.3, label=f"阈值 {q:.4f}")
    ax.set_xlabel("GMM 异常分数 = -log p(x)")
    ax.set_ylabel("样本数")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_cluster_count_bar(scored: pd.DataFrame, cfg: Config, out_png: Path, title: str) -> None:
    primary_label = quantile_label(cfg.PRIMARY_QUANTILE)
    anomaly_col = f"is_anomaly_{primary_label}"
    tmp = scored.groupby("gmm_cluster", observed=False).agg(
        total=("gmm_cluster", "size"),
        anomaly=(anomaly_col, "sum"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(tmp["gmm_cluster"].astype(str), tmp["total"], label="全部样本")
    ax.bar(tmp["gmm_cluster"].astype(str), tmp["anomaly"], label="异常样本")
    ax.set_xlabel("GMM 簇编号")
    ax.set_ylabel("样本数")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_cluster_profile_heatmap(scored: pd.DataFrame, feature_cols: List[str], cfg: Config, out_png: Path, title: str) -> None:
    if not feature_cols:
        return
    # 用每簇的标准化均值解释簇。这里重新对绘图样本标准化即可。
    x = scored[feature_cols].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True))
    std = x.std(axis=0).replace(0, np.nan)
    z = (x - x.mean(axis=0)) / std
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0)

    prof = z.join(scored["gmm_cluster"]).groupby("gmm_cluster", observed=False).mean()
    if prof.empty:
        return

    # 只画簇间差异最大的若干特征，避免图太挤。
    feature_order = prof.var(axis=0).sort_values(ascending=False).head(cfg.CLUSTER_PROFILE_TOP_FEATURES).index.tolist()
    prof = prof[feature_order]

    fig_w = max(10, 0.45 * len(feature_order))
    fig_h = max(4, 0.55 * len(prof))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(prof.to_numpy(dtype=float), aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("簇内标准化均值")
    ax.set_xticks(np.arange(len(feature_order)))
    ax.set_xticklabels(feature_order, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(prof.index)))
    ax.set_yticklabels([f"簇 {i}" for i in prof.index])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def get_plot_sample(scored: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    primary_label = quantile_label(cfg.PRIMARY_QUANTILE)
    anomaly_col = f"is_anomaly_{primary_label}"
    anomalies = scored.loc[scored[anomaly_col]].copy()
    normals = scored.loc[~scored[anomaly_col]].copy()
    if len(anomalies) >= cfg.PLOT_SAMPLE_MAX_ROWS:
        return anomalies.sample(n=cfg.PLOT_SAMPLE_MAX_ROWS, random_state=cfg.RANDOM_STATE)
    normal_n = cfg.PLOT_SAMPLE_MAX_ROWS - len(anomalies)
    if len(normals) > normal_n:
        normals = normals.sample(n=normal_n, random_state=cfg.RANDOM_STATE)
    return pd.concat([anomalies, normals], ignore_index=True)


def save_interactive_pca_cluster(scored: pd.DataFrame, cfg: Config, out_html: Path, title: str) -> None:
    primary_label = quantile_label(cfg.PRIMARY_QUANTILE)
    anomaly_col = f"is_anomaly_{primary_label}"
    plot_df = get_plot_sample(scored, cfg)
    plot_df["异常状态"] = np.where(plot_df[anomaly_col], "异常", "正常")
    plot_df["GMM簇"] = plot_df["gmm_cluster"].astype(str)

    fig = px.scatter(
        plot_df,
        x="pca1",
        y="pca2",
        color="GMM簇",
        symbol="异常状态",
        hover_data={
            "device_name": True,
            "raw_file_stem": True,
            "source_speed_bin": True,
            "row_in_file": True,
            "gmm_anomaly_score": ":.4f",
            "gmm_max_posterior_prob": ":.4f",
            "pca1": ":.4f",
            "pca2": ":.4f",
        },
        title=title,
        opacity=0.65,
    )
    plotly_chinese_layout(fig)
    fig.write_html(out_html)


def save_interactive_pca_score(scored: pd.DataFrame, cfg: Config, out_html: Path, title: str) -> None:
    primary_label = quantile_label(cfg.PRIMARY_QUANTILE)
    anomaly_col = f"is_anomaly_{primary_label}"
    plot_df = get_plot_sample(scored, cfg)
    plot_df["异常状态"] = np.where(plot_df[anomaly_col], "异常", "正常")

    fig = px.scatter(
        plot_df,
        x="pca1",
        y="pca2",
        color="gmm_anomaly_score",
        symbol="异常状态",
        hover_data={
            "device_name": True,
            "raw_file_stem": True,
            "source_speed_bin": True,
            "row_in_file": True,
            "gmm_cluster": True,
            "gmm_anomaly_score": ":.4f",
            "gmm_max_posterior_prob": ":.4f",
        },
        color_continuous_scale="Viridis",
        title=title,
        opacity=0.7,
    )
    plotly_chinese_layout(fig)
    fig.write_html(out_html)


def choose_key_scatter_features(feature_cols: List[str], resolved_cols: Dict[str, Optional[str]], cfg: Config) -> Tuple[Optional[str], List[str]]:
    # 横轴优先用转速平方，其次用二次侧泵转速。
    x_candidates = [
        "派生_二次侧泵转速平方",
        resolved_cols.get("speed_col"),
    ]
    x_col = next((c for c in x_candidates if c and c in feature_cols), None)

    preferred_y = [
        "派生_二次侧板换压差加过滤器压差",
        "派生_二次侧供回水压差加板换压差",
        "派生_二次侧泵压差_泵出口减泵入口均值",
        resolved_cols.get("sec_pump_dp_col"),
        resolved_cols.get("sec_hex_dp_col"),
        resolved_cols.get("sec_filter_dp_col"),
        resolved_cols.get("sec_sr_dp_col"),
        resolved_cols.get("sec_pipe_ratio_col"),
        resolved_cols.get("valve_open_col"),
        resolved_cols.get("sec_flow_col"),
    ]
    y_cols = []
    for c in preferred_y:
        if c and c in feature_cols and c != x_col and c not in y_cols:
            y_cols.append(c)
    return x_col, y_cols[:cfg.MAX_KEY_SCATTER_FEATURES]


def save_interactive_key_scatter(scored: pd.DataFrame, cfg: Config, x_col: str, y_col: str, out_html: Path, title: str) -> None:
    primary_label = quantile_label(cfg.PRIMARY_QUANTILE)
    anomaly_col = f"is_anomaly_{primary_label}"
    plot_df = get_plot_sample(scored, cfg)
    plot_df["异常状态"] = np.where(plot_df[anomaly_col], "异常", "正常")
    plot_df["GMM簇"] = plot_df["gmm_cluster"].astype(str)

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color="GMM簇",
        symbol="异常状态",
        hover_data={
            "device_name": True,
            "raw_file_stem": True,
            "source_speed_bin": True,
            "row_in_file": True,
            "gmm_anomaly_score": ":.4f",
            "gmm_max_posterior_prob": ":.4f",
            x_col: ":.6f",
            y_col: ":.6f",
        },
        title=title,
        opacity=0.65,
    )
    plotly_chinese_layout(fig)
    fig.write_html(out_html)


def save_static_key_scatter(scored: pd.DataFrame, cfg: Config, x_col: str, y_col: str, out_png: Path, title: str) -> None:
    primary_label = quantile_label(cfg.PRIMARY_QUANTILE)
    anomaly_col = f"is_anomaly_{primary_label}"
    plot_df = get_plot_sample(scored, cfg)
    normal = plot_df.loc[~plot_df[anomaly_col]]
    abnormal = plot_df.loc[plot_df[anomaly_col]]

    fig, ax = plt.subplots(figsize=(10, 7))
    # 不指定具体颜色，交给 matplotlib 默认色系。
    ax.scatter(normal[x_col], normal[y_col], s=5, alpha=0.35, label="正常")
    ax.scatter(abnormal[x_col], abnormal[y_col], s=12, alpha=0.8, label="异常")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_device_anomaly_rate_hist(device_summary: pd.DataFrame, cfg: Config, out_png: Path, title: str) -> None:
    primary_label = quantile_label(cfg.PRIMARY_QUANTILE)
    rate_col = f"anomaly_rate_{primary_label}"
    vals = pd.to_numeric(device_summary[rate_col], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(vals, bins=80)
    ax.set_xlabel("设备异常率")
    ax.set_ylabel("设备数")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_interactive_device_risk(device_summary: pd.DataFrame, cfg: Config, out_html: Path, title: str) -> None:
    primary_label = quantile_label(cfg.PRIMARY_QUANTILE)
    rate_col = f"anomaly_rate_{primary_label}"
    count_col = f"anomaly_count_{primary_label}"

    plot_df = device_summary.copy()
    numeric_cols = ["evaluable_samples", rate_col, count_col, "max_score", "p95_score", "mean_score", "cluster_entropy", "dominant_cluster_rate"]
    for c in numeric_cols:
        if c in plot_df.columns:
            plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.loc[plot_df["evaluable_samples"].fillna(0) > 0].copy()
    if plot_df.empty:
        out_html.write_text("<html><body><p>没有可绘制设备。</p></body></html>", encoding="utf-8")
        return

    plot_df[rate_col] = plot_df[rate_col].fillna(0.0)
    plot_df["plot_size"] = plot_df["max_score"].fillna(0.0).clip(lower=0)
    if plot_df["plot_size"].max() <= 0:
        plot_df["plot_size"] = 1.0
    plot_df["dominant_cluster"] = plot_df["dominant_cluster"].astype(str)

    plot_df = plot_df.sort_values([rate_col, "max_score", "evaluable_samples"], ascending=[False, False, False]).head(
        min(cfg.DEVICE_RISK_TOPN, len(plot_df))
    )

    fig = px.scatter(
        plot_df,
        x="evaluable_samples",
        y=rate_col,
        size="plot_size",
        color="dominant_cluster",
        hover_data={
            "device_name": True,
            "evaluable_samples": True,
            count_col: True,
            rate_col: ":.4%",
            "max_score": ":.4f",
            "p95_score": ":.4f",
            "mean_score": ":.4f",
            "dominant_cluster": True,
            "dominant_cluster_rate": ":.4%",
            "cluster_entropy": ":.4f",
            "n_raw_files": True,
            "n_speed_bins": True,
            "plot_size": False,
        },
        title=title,
    )
    plotly_chinese_layout(fig)
    fig.write_html(out_html)


def save_all_visualizations(
    scored: pd.DataFrame,
    device_summary: pd.DataFrame,
    feature_report: pd.DataFrame,
    fit: FittedGMMResult,
    feature_cols: List[str],
    resolved_cols: Dict[str, Optional[str]],
    group_name: str,
    out_dir: Path,
    cfg: Config,
) -> None:
    safe_mkdir(out_dir)
    group_slug = sanitize_filename(group_name)

    save_feature_missing_bar(
        feature_report,
        out_dir / f"01_feature_missing_rate_{group_slug}.png",
        title=f"控制组 {group_name} | 特征缺失率 Top40",
    )

    save_bic_aic_plot(
        fit.bic_table,
        fit.selected_k,
        out_dir / f"02_bic_aic_curve_{group_slug}.png",
        title=f"控制组 {group_name} | GMM 成分数选择：BIC/AIC",
    )

    save_score_hist(
        scored,
        cfg,
        out_dir / f"03_gmm_score_hist_{group_slug}.png",
        title=f"控制组 {group_name} | GMM 异常分数分布",
    )

    save_cluster_count_bar(
        scored,
        cfg,
        out_dir / f"04_cluster_count_bar_{group_slug}.png",
        title=f"控制组 {group_name} | GMM 各簇样本数与异常数",
    )

    save_cluster_profile_heatmap(
        scored,
        feature_cols,
        cfg,
        out_dir / f"05_cluster_profile_heatmap_{group_slug}.png",
        title=f"控制组 {group_name} | GMM 簇画像：标准化特征均值",
    )

    save_interactive_pca_cluster(
        scored,
        cfg,
        out_dir / f"06_interactive_pca_cluster_{group_slug}.html",
        title=f"控制组 {group_name} | PCA 二维投影：颜色=GMM簇，形状=异常状态",
    )

    save_interactive_pca_score(
        scored,
        cfg,
        out_dir / f"07_interactive_pca_score_{group_slug}.html",
        title=f"控制组 {group_name} | PCA 二维投影：颜色=GMM异常分数",
    )

    x_col, y_cols = choose_key_scatter_features(feature_cols, resolved_cols, cfg)
    if x_col and y_cols:
        for i, y_col in enumerate(y_cols, start=1):
            y_slug = sanitize_filename(y_col)
            save_static_key_scatter(
                scored,
                cfg,
                x_col,
                y_col,
                out_dir / f"08_{i:02d}_static_{sanitize_filename(x_col)}_vs_{y_slug}_{group_slug}.png",
                title=f"控制组 {group_name} | {x_col} - {y_col}：异常点突出显示",
            )
            save_interactive_key_scatter(
                scored,
                cfg,
                x_col,
                y_col,
                out_dir / f"09_{i:02d}_interactive_{sanitize_filename(x_col)}_vs_{y_slug}_{group_slug}.html",
                title=f"控制组 {group_name} | {x_col} - {y_col}：颜色=GMM簇，形状=异常状态",
            )

    save_device_anomaly_rate_hist(
        device_summary,
        cfg,
        out_dir / f"10_device_anomaly_rate_hist_{group_slug}.png",
        title=f"控制组 {group_name} | 设备异常率分布",
    )

    save_interactive_device_risk(
        device_summary,
        cfg,
        out_dir / f"11_interactive_device_risk_{group_slug}.html",
        title=f"控制组 {group_name} | 设备风险散点图：横轴=样本数，纵轴=异常率，大小=最大分数",
    )


# ==========================================================
# 6. 单组处理与主流程
# ==========================================================

def process_one_group(group_name: str, group_dir: str, cfg: Config, out_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"\n==============================")
    print(f"开始处理控制组 {group_name}")
    print(f"路径: {group_dir}")
    print(f"==============================")

    group_out = out_root / f"control_group_{sanitize_filename(group_name)}"
    fig_out = group_out / "figures"
    safe_mkdir(group_out)
    safe_mkdir(fig_out)

    print("1) 读取 CSV ...")
    group_df = load_group_table(group_name, group_dir, cfg)
    print(f"   原始行数: {len(group_df):,}；设备数: {group_df['device_name'].nunique():,}")

    print("2) 构造与筛选 GMM 特征 ...")
    work, feature_cols, feature_report, resolved_cols = prepare_feature_table(group_df, cfg)
    print(f"   可评估行数: {len(work):,}；特征数: {len(feature_cols)}")
    print("   使用特征:")
    for c in feature_cols:
        print(f"     - {c}")

    # 保存列名映射与特征报告。
    save_csv(pd.DataFrame({"role": list(resolved_cols.keys()), "resolved_column": list(resolved_cols.values())}), group_out / "resolved_columns.csv")
    save_csv(feature_report, group_out / "feature_selection_report.csv")

    print("3) BIC/AIC 选择 GMM 成分数并训练模型 ...")
    fit = fit_gmm_with_bic(work, feature_cols, cfg)
    print(f"   BIC 选择 K={fit.selected_k}；GMM 输入维度={fit.gmm_feature_dim}")
    save_csv(fit.bic_table, group_out / "gmm_bic_aic_table.csv")

    print("4) 对全部可评估样本计算 GMM 簇与异常分数 ...")
    scored = score_all_samples(work, feature_cols, fit, cfg)
    primary_label = quantile_label(cfg.PRIMARY_QUANTILE)
    print(
        f"   主阈值 {cfg.PRIMARY_QUANTILE:.4f}: "
        f"异常样本 {int(scored[f'is_anomaly_{primary_label}'].sum()):,}/"
        f"{len(scored):,} ({scored[f'is_anomaly_{primary_label}'].mean():.3%})"
    )

    print("5) 按设备、簇和模型输出汇总表 ...")
    device_summary = build_device_summary(scored, fit, cfg)
    cluster_summary = build_cluster_summary(scored, feature_cols, cfg)
    model_summary = build_model_summary(scored, fit, feature_cols, cfg)
    top_samples = build_top_anomaly_samples(scored, feature_cols, cfg)

    save_csv(model_summary, group_out / "model_summary.csv")
    save_csv(device_summary, group_out / "device_summary.csv")
    save_csv(cluster_summary, group_out / "cluster_summary.csv")
    save_csv(top_samples, group_out / "top_anomaly_samples.csv")

    if cfg.SAVE_FULL_SCORED_SAMPLES:
        keep_cols = [
            "control_group", "device_name", "raw_file_stem", "source_speed_bin", "source_relpath", "row_in_file",
            "gmm_cluster", "gmm_log_likelihood", "gmm_anomaly_score", "gmm_max_posterior_prob", "pca1", "pca2",
        ]
        for q in sorted(set([cfg.PRIMARY_QUANTILE, *cfg.EXTRA_QUANTILES])):
            qlab = quantile_label(q)
            keep_cols.extend([f"threshold_{qlab}", f"is_anomaly_{qlab}"])
        save_csv(scored[keep_cols + feature_cols], group_out / "scored_samples_full.csv")
    else:
        sampled = get_plot_sample(scored, cfg)
        save_csv(sampled, group_out / "scored_samples_for_plot.csv")

    print("6) 生成可视化图表 ...")
    save_all_visualizations(
        scored=scored,
        device_summary=device_summary,
        feature_report=feature_report,
        fit=fit,
        feature_cols=feature_cols,
        resolved_cols=resolved_cols,
        group_name=group_name,
        out_dir=fig_out,
        cfg=cfg,
    )

    print(f"控制组 {group_name} 完成。输出目录: {group_out}")

    # 释放大对象。
    del group_df, work, scored
    gc.collect()

    return model_summary, device_summary, cluster_summary


def main(cfg: Config = CFG) -> None:
    configure_chinese_font()
    out_root = Path(cfg.OUTPUT_DIR)
    safe_mkdir(out_root)

    all_model_summaries: List[pd.DataFrame] = []
    all_device_summaries: List[pd.DataFrame] = []
    all_cluster_summaries: List[pd.DataFrame] = []

    for group_name, group_dir in cfg.GROUP_DIRS.items():
        model_summary, device_summary, cluster_summary = process_one_group(group_name, group_dir, cfg, out_root)
        all_model_summaries.append(model_summary)
        all_device_summaries.append(device_summary)
        all_cluster_summaries.append(cluster_summary)

    save_csv(pd.concat(all_model_summaries, ignore_index=True), out_root / "model_summary_all_groups.csv")
    save_csv(pd.concat(all_device_summaries, ignore_index=True), out_root / "device_summary_all_groups.csv")
    save_csv(pd.concat(all_cluster_summaries, ignore_index=True), out_root / "cluster_summary_all_groups.csv")

    print("\n全部控制组处理完成。")
    print("总输出目录:", out_root)
    print("建议优先查看：")
    print("  1) model_summary_all_groups.csv")
    print("  2) device_summary_all_groups.csv")
    print("  3) 每个 control_group_xxx/figures 下的 HTML 与 PNG 图")


if __name__ == "__main__":
    main()
