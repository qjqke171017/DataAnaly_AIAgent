# -*- coding: utf-8 -*-
"""
绘制正常样本（N 文件夹）散点图：
横轴：转速平方
纵轴：模型三因变量 = 二次侧板换压差 + 二次侧过滤器压差
颜色：控制压差目标值 0.8 / 1.4 / 1.65 / 其他
"""

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. 用户配置区
# ============================================================

# 数据根目录（其下应有 N、F03、F04、F06 等文件夹）
DATA_ROOT_DIR = r"D:\data\secondary_raw"

# 输出目录
OUTPUT_DIR = r"D:\plot_normal_m3_scatter_output"

# 是否递归读取 N 文件夹下所有 csv
CSV_RECURSIVE = True

# 如果数据量太大，可设置每个 CSV 最多随机抽样多少行用于绘图；
# None 表示全量读取
MAX_ROWS_PER_FILE_FOR_PLOT = 20000

# 总图最多绘制多少个点；None 表示不再额外抽样
MAX_TOTAL_POINTS_FOR_PLOT = 300000

# 控制压差保留小数位
CONTROL_ROUND_DIGITS = 3

# 图像参数
FIGSIZE = (11.5, 8)
FIG_DPI = 260
POINT_SIZE = 6
POINT_ALPHA = 0.16

# 是否限制转速范围；如果你想画所有正常样本，就设为 False
USE_SPEED_FILTER = False
MIN_SPEED = 50.0
MAX_SPEED = 95.0


# ============================================================
# 2. 精确列名
# ============================================================

COL_CONTROL = "控制压差目标值"
COL_SPEED = "二次侧泵转速"
COL_PLATE_DP = "二次侧板换压差"
COL_FILTER_DP = "二次侧过滤器压差"


# ============================================================
# 3. 工具函数
# ============================================================

def stable_seed(base: int, text: str) -> int:
    h = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
    return (int(h[:8], 16) + int(base)) % (2**32 - 1)


def to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = series.astype(str).str.strip()
    s = (
        s.str.replace("−", "-", regex=False)
         .str.replace("－", "-", regex=False)
         .str.replace("，", ".", regex=False)
    )
    out = pd.to_numeric(s, errors="coerce")

    need = out.isna() & s.notna() & ~s.str.lower().isin(["", "nan", "none", "null"])
    if need.any():
        extracted = s[need].str.extract(r"([-+]?\d+(?:[\.,]\d+)?)", expand=False)
        extracted = extracted.str.replace(",", ".", regex=False)
        out.loc[need] = pd.to_numeric(extracted, errors="coerce")

    return out


def normalize_speed(speed_raw: pd.Series) -> pd.Series:
    """
    若转速是 0~100，则除以 100；
    若本身已是 0~1，则保持不变。
    """
    s = pd.to_numeric(speed_raw, errors="coerce")
    q95 = s.dropna().quantile(0.95) if s.notna().any() else np.nan

    if pd.notna(q95) and q95 > 2:
        return s / 100.0
    return s


def speed_in_range(speed_raw: pd.Series) -> pd.Series:
    s = pd.to_numeric(speed_raw, errors="coerce")
    q95 = s.dropna().quantile(0.95) if s.notna().any() else np.nan

    if pd.notna(q95) and q95 > 2:
        return s.notna() & (s >= MIN_SPEED) & (s < MAX_SPEED)
    else:
        return s.notna() & (s >= MIN_SPEED / 100.0) & (s < MAX_SPEED / 100.0)


def read_csv_with_sampling(path: Path, usecols: list[str], max_rows: int | None, seed: int) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"]
    last_err = None

    for enc in encodings:
        try:
            header = list(pd.read_csv(path, encoding=enc, nrows=0).columns)
            missing = [c for c in usecols if c not in header]
            if missing:
                raise RuntimeError(f"缺少列：{missing}")

            if max_rows is None:
                return pd.read_csv(path, encoding=enc, usecols=usecols)

            rng = np.random.default_rng(seed)
            chunks = []

            for chunk in pd.read_csv(path, encoding=enc, usecols=usecols, chunksize=200000):
                chunk = chunk.copy()
                chunk["__rand__"] = rng.random(len(chunk))
                chunks.append(chunk)

            if not chunks:
                return pd.DataFrame(columns=usecols)

            df = pd.concat(chunks, ignore_index=True)
            if len(df) > max_rows:
                df = df.nsmallest(max_rows, "__rand__")

            df = df.drop(columns="__rand__").reset_index(drop=True)
            return df

        except Exception as e:
            last_err = e

    raise RuntimeError(f"读取失败：{path}；最后错误：{last_err}")


def classify_control_group(control_series: pd.Series) -> pd.Series:
    cp = to_numeric(control_series).round(CONTROL_ROUND_DIGITS)

    out = pd.Series("其他", index=cp.index, dtype="object")
    out[np.isclose(cp, 0.8)] = "0.8"
    out[np.isclose(cp, 1.4)] = "1.4"
    out[np.isclose(cp, 1.65)] = "1.65"
    return out


# ============================================================
# 4. 读取 N 文件夹数据
# ============================================================

def load_normal_data() -> pd.DataFrame:
    root = Path(DATA_ROOT_DIR)
    n_dir = root / "N"

    if not n_dir.exists():
        raise FileNotFoundError(f"未找到 N 文件夹：{n_dir}")

    usecols = [COL_CONTROL, COL_SPEED, COL_PLATE_DP, COL_FILTER_DP]

    if CSV_RECURSIVE:
        csv_files = sorted([p for p in n_dir.rglob("*.csv") if p.is_file()])
    else:
        csv_files = sorted([p for p in n_dir.glob("*.csv") if p.is_file()])

    if not csv_files:
        raise RuntimeError("N 文件夹下未找到任何 CSV。")

    all_parts = []

    for fp in csv_files:
        seed = stable_seed(42, str(fp))
        try:
            raw = read_csv_with_sampling(
                path=fp,
                usecols=usecols,
                max_rows=MAX_ROWS_PER_FILE_FOR_PLOT,
                seed=seed
            )
        except Exception as e:
            print(f"[WARN] 跳过文件 {fp.name}: {e}")
            continue

        if raw.empty:
            continue

        df = pd.DataFrame()
        df[COL_CONTROL] = to_numeric(raw[COL_CONTROL])
        df[COL_SPEED] = to_numeric(raw[COL_SPEED])
        df[COL_PLATE_DP] = to_numeric(raw[COL_PLATE_DP])
        df[COL_FILTER_DP] = to_numeric(raw[COL_FILTER_DP])

        if USE_SPEED_FILTER:
            keep = speed_in_range(df[COL_SPEED])
            df = df.loc[keep].copy()

        if df.empty:
            continue

        df["speed_norm"] = normalize_speed(df[COL_SPEED])
        df["x"] = df["speed_norm"] ** 2
        df["y"] = df[COL_PLATE_DP] + df[COL_FILTER_DP]
        df["control_group"] = classify_control_group(df[COL_CONTROL])
        df["device_name"] = fp.stem

        valid = (
            df["x"].notna() &
            df["y"].notna() &
            np.isfinite(df["x"]) &
            np.isfinite(df["y"])
        )
        df = df.loc[valid].copy()

        if not df.empty:
            all_parts.append(df)

    if not all_parts:
        raise RuntimeError("没有读取到有效正常样本。")

    data = pd.concat(all_parts, ignore_index=True)

    if (MAX_TOTAL_POINTS_FOR_PLOT is not None) and (len(data) > MAX_TOTAL_POINTS_FOR_PLOT):
        data = data.sample(n=MAX_TOTAL_POINTS_FOR_PLOT, random_state=42).reset_index(drop=True)

    return data


# ============================================================
# 5. 绘图
# ============================================================

def plot_scatter(data: pd.DataFrame):
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    color_map = {
        "0.8": "#1f77b4",   # 蓝
        "1.4": "#ff7f0e",   # 橙
        "1.65": "#2ca02c",  # 绿
        "其他": "#7f7f7f",   # 灰
    }

    fig, ax = plt.subplots(figsize=FIGSIZE)

    group_order = ["0.8", "1.4", "1.65", "其他"]
    summary_rows = []

    total_n = len(data)

    for g in group_order:
        sub = data[data["control_group"] == g]
        if sub.empty:
            continue

        ax.scatter(
            sub["x"],
            sub["y"],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            c=color_map[g],
            label=f"控制压差={g} (n={len(sub):,}, {len(sub)/total_n:.1%})",
            linewidths=0
        )

        summary_rows.append({
            "control_group": g,
            "count": len(sub),
            "proportion": len(sub) / total_n,
            "x_mean": sub["x"].mean(),
            "x_std": sub["x"].std(),
            "y_mean": sub["y"].mean(),
            "y_std": sub["y"].std(),
        })

    ax.set_title("正常样本散点图：模型三因变量 vs 转速平方", fontsize=15)
    ax.set_xlabel("(二次侧泵转速/100)^2", fontsize=12)
    ax.set_ylabel("二次侧板换压差 + 二次侧过滤器压差", fontsize=12)
    ax.grid(True, alpha=0.25)

    ax.text(
        0.01, 0.99,
        f"总绘制点数：{len(data):,}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85)
    )

    ax.legend(loc="best", fontsize=9, frameon=True)
    fig.tight_layout()

    suffix = "all_speed" if not USE_SPEED_FILTER else f"speed_{MIN_SPEED}_{MAX_SPEED}"
    fig_path = out_dir / f"normal_M3_scatter_by_control_{suffix}.png"
    csv_path = out_dir / f"normal_M3_scatter_by_control_{suffix}_summary.csv"

    fig.savefig(fig_path, dpi=FIG_DPI)
    plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"[DONE] 图片已保存：{fig_path}")
    print(f"[DONE] 统计表已保存：{csv_path}")


# ============================================================
# 6. 主程序
# ============================================================

def main():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    data = load_normal_data()
    print(f"[INFO] 读取完成，有效正常样本点数：{len(data):,}")
    print(data["control_group"].value_counts(dropna=False))

    plot_scatter(data)


if __name__ == "__main__":
    main()
