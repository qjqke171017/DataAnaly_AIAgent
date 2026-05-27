# -*- coding: utf-8 -*-
"""
模型3 M3_plate_plus_filter 高误报正常设备散点高亮图。

功能：
1. 只使用模型3：二次侧板换压差 + 二次侧过滤器压差 ~ 转速平方；
2. 只使用 q=0.9975 阈值；
3. 从旧实验输出表中自动筛选 N 文件夹中误报率 >= 40% 的设备；
4. 在带 WLS 拟合线和阈值带的散点图中高亮这些设备；
5. 背景点随机抽样 50000 个；每个高误报设备随机抽样 500 个点。
"""

from pathlib import Path
import hashlib
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. 用户配置区
# ============================================================

DATA_ROOT_DIR = r"D:\data\secondary_raw"

TABLES_DIR = r"D:\wls_secondary_multi_repeat_output\tables"

OUTPUT_DIR = r"D:\wls_secondary_multi_repeat_output\figures_highlight_high_false_alarm"

MODEL_ID = "M3_plate_plus_filter"
THRESHOLD_QUANTILE = 0.9975

# 一张散点图只能对应一个控制压差组，因为 WLS 是分控制压差拟合的
CONTROL_GROUP = "1.4"

# 自动筛选 N 文件夹中误报率 >= 40% 的设备
FALSE_ALARM_RATE_CUTOFF = 0.40

# 背景点抽样数量
BACKGROUND_SAMPLE_SIZE = 50000

# 每台高误报设备最多绘制多少点
POINTS_PER_HIGH_FALSE_DEVICE = 500

# 每个 CSV 读取时的最大抽样行数；None 表示全量读取
# 背景设备可抽样，高误报设备会优先读取更多数据
BACKGROUND_MAX_ROWS_PER_FILE = 8000
HIGHLIGHT_MAX_ROWS_PER_FILE = None

# 转速过滤：只保留 50 <= speed < 95
MIN_SPEED = 50.0
MAX_SPEED = 95.0

CONTROL_ROUND_DIGITS = 3

# WLS 拟合参数
TRAIN_NORMAL_DEVICE_FRAC = 0.70
RANDOM_STATE = 42
WLS_BINS = 10
MIN_ROWS_PER_BIN = 30
MIN_SIGMA = 1e-6

CSV_RECURSIVE = True

FIG_DPI = 260
FIG_SIZE = (12.5, 7.6)


# ============================================================
# 2. 列名配置
# ============================================================

COL_CONTROL = "控制压差目标值"
COL_SPEED = "二次侧泵转速"
COL_PLATE_DP = "二次侧板换压差"
COL_FILTER_DP = "二次侧过滤器压差"

NEEDED_CODES = {"N", "F03", "F04", "F06"}


# ============================================================
# 3. 基础函数
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


def control_key(values: pd.Series, digits: int) -> pd.Series:
    scale = 10 ** int(digits)
    x = pd.to_numeric(values, errors="coerce").astype("float64")
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    ok = x.notna()
    out.loc[ok] = np.rint(x.loc[ok] * scale)
    return out


def normalize_speed(speed_raw: pd.Series) -> pd.Series:
    q95 = speed_raw.dropna().quantile(0.95) if speed_raw.notna().any() else np.nan
    if pd.notna(q95) and q95 > 2:
        return speed_raw / 100.0
    return speed_raw


def in_speed_range(speed_raw: pd.Series) -> pd.Series:
    q95 = speed_raw.dropna().quantile(0.95) if speed_raw.notna().any() else np.nan
    x = pd.to_numeric(speed_raw, errors="coerce")

    if pd.notna(q95) and q95 > 2:
        return x.notna() & (x >= MIN_SPEED) & (x < MAX_SPEED)
    else:
        return x.notna() & (x >= MIN_SPEED / 100.0) & (x < MAX_SPEED / 100.0)


def find_code_from_path(fp: Path, root: Path) -> str | None:
    try:
        parts = fp.relative_to(root).parts
    except Exception:
        parts = fp.parts

    for p in parts:
        code = str(p).strip().upper()
        if code in NEEDED_CODES:
            return code
    return None


def read_csv_selected(path: Path, usecols: list[str], max_rows: int | None, seed: int) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"]
    last_err = None

    for enc in encodings:
        try:
            header = list(pd.read_csv(path, encoding=enc, nrows=0).columns)
            missing = [c for c in usecols if c not in header]
            if missing:
                raise RuntimeError(f"缺少列：{missing}")

            if max_rows is None:
                df = pd.read_csv(path, encoding=enc, usecols=usecols)
                df["row_id"] = np.arange(len(df), dtype=np.int64)
                return df

            rng = np.random.default_rng(seed)
            chunks = []
            offset = 0

            for chunk in pd.read_csv(path, encoding=enc, usecols=usecols, chunksize=200000):
                n = len(chunk)
                chunk = chunk.copy()
                chunk["row_id"] = np.arange(offset, offset + n, dtype=np.int64)
                chunk["__rand__"] = rng.random(n)
                chunks.append(chunk)
                offset += n

            if not chunks:
                return pd.DataFrame(columns=usecols + ["row_id"])

            df = pd.concat(chunks, ignore_index=True)
            if len(df) > max_rows:
                df = df.nsmallest(max_rows, "__rand__")

            df = df.drop(columns=["__rand__"]).sort_values("row_id").reset_index(drop=True)
            return df

        except Exception as e:
            last_err = e

    raise RuntimeError(f"读取失败：{path}；最后错误：{last_err}")


def robust_sigma(x: np.ndarray, min_sigma: float = 1e-6) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return min_sigma

    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sig = 1.4826 * mad

    if not np.isfinite(sig) or sig < min_sigma:
        sig = np.std(x)

    if not np.isfinite(sig) or sig < min_sigma:
        sig = min_sigma

    return float(sig)


def wls_solve(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    if w is None:
        return np.linalg.pinv(X.T @ X) @ X.T @ y

    sw = np.sqrt(np.asarray(w, dtype=float))
    Xw = X * sw[:, None]
    yw = y * sw
    return np.linalg.pinv(Xw.T @ Xw) @ Xw.T @ yw


def make_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.array([-np.inf, np.inf])

    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))

    if len(edges) < 3:
        return np.array([-np.inf, np.inf])

    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def assign_bins(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges[1:-1], x, side="right")


def fit_wls(train_df: pd.DataFrame) -> dict:
    x = train_df["x"].to_numpy(dtype=float)
    y = train_df["y"].to_numpy(dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    if len(x) < 80 or len(np.unique(x)) < 2:
        raise RuntimeError(f"WLS 训练样本不足：n={len(x)}, unique_x={len(np.unique(x))}")

    X = np.column_stack([np.ones_like(x), x])

    beta_ols = wls_solve(X, y)
    resid0 = y - X @ beta_ols

    global_sigma0 = robust_sigma(resid0, MIN_SIGMA)
    edges = make_bins(x, WLS_BINS)
    bid = assign_bins(x, edges)

    sigma0 = []
    for b in range(len(edges) - 1):
        rb = resid0[bid == b]
        if len(rb) >= MIN_ROWS_PER_BIN:
            sigma0.append(robust_sigma(rb, MIN_SIGMA))
        else:
            sigma0.append(global_sigma0)

    sigma0 = np.maximum(np.asarray(sigma0), MIN_SIGMA)
    weights = 1.0 / sigma0[np.clip(bid, 0, len(sigma0) - 1)] ** 2

    beta = wls_solve(X, y, weights)

    resid = y - X @ beta
    global_sigma = robust_sigma(resid, MIN_SIGMA)
    bid = assign_bins(x, edges)

    sigma_final = []
    for b in range(len(edges) - 1):
        rb = resid[bid == b]
        if len(rb) >= MIN_ROWS_PER_BIN:
            sigma_final.append(robust_sigma(rb, MIN_SIGMA))
        else:
            sigma_final.append(global_sigma)

    sigma_final = np.maximum(np.asarray(sigma_final), MIN_SIGMA)

    return {
        "beta0": float(beta[0]),
        "beta1": float(beta[1]),
        "edges": edges,
        "sigma": sigma_final,
        "global_sigma": float(global_sigma),
    }


def predict_wls(x: np.ndarray, model: dict) -> tuple[np.ndarray, np.ndarray]:
    yhat = model["beta0"] + model["beta1"] * x
    bid = assign_bins(x, model["edges"])
    bid = np.clip(bid, 0, len(model["sigma"]) - 1)
    sigma = model["sigma"][bid]
    return yhat, sigma


# ============================================================
# 4. 自动筛选高误报设备
# ============================================================

def get_high_false_alarm_devices() -> list[str]:
    table_path = Path(TABLES_DIR) / "wls_normal_device_false_alarm_by_device.csv"

    if not table_path.exists():
        raise FileNotFoundError(f"找不到正常设备误报汇总表：{table_path}")

    df = pd.read_csv(table_path, encoding="utf-8-sig")

    sub = df[
        (df["model_id"].astype(str) == MODEL_ID) &
        (np.isclose(df["threshold_quantile"].astype(float), THRESHOLD_QUANTILE)) &
        (df["control_group"].astype(str) == str(CONTROL_GROUP))
    ].copy()

    if sub.empty:
        raise RuntimeError("没有筛到对应模型、阈值和控制压差下的正常设备误报记录。")

    if "false_alarm_rate_mean" not in sub.columns:
        raise RuntimeError("表中缺少 false_alarm_rate_mean 列。")

    sub = sub[sub["false_alarm_rate_mean"] >= FALSE_ALARM_RATE_CUTOFF].copy()
    sub = sub.sort_values("false_alarm_rate_mean", ascending=False)

    devices = sub["device_name"].astype(str).drop_duplicates().tolist()

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    list_path = out_dir / f"high_false_alarm_devices_{MODEL_ID}_cp_{CONTROL_GROUP}_q{int(THRESHOLD_QUANTILE*10000):04d}.txt"
    csv_path = out_dir / f"high_false_alarm_devices_{MODEL_ID}_cp_{CONTROL_GROUP}_q{int(THRESHOLD_QUANTILE*10000):04d}.csv"

    sub.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open(list_path, "w", encoding="utf-8") as f:
        for d in devices:
            f.write(d + "\n")

    print(f"[INFO] 高误报设备数量：{len(devices)}")
    print(f"[INFO] 设备列表已保存：{list_path}")
    print(f"[INFO] 设备明细已保存：{csv_path}")

    for d in devices:
        print("  -", d)

    if not devices:
        raise RuntimeError(f"没有找到误报率 >= {FALSE_ALARM_RATE_CUTOFF:.0%} 的设备。")

    return devices


# ============================================================
# 5. 读取绘图数据
# ============================================================

def load_plot_data(high_devices: list[str]) -> pd.DataFrame:
    root = Path(DATA_ROOT_DIR)

    if not root.exists():
        raise FileNotFoundError(f"DATA_ROOT_DIR 不存在：{root}")

    usecols = [COL_CONTROL, COL_SPEED, COL_PLATE_DP, COL_FILTER_DP]
    target_key = int(round(float(CONTROL_GROUP) * (10 ** CONTROL_ROUND_DIGITS)))

    csv_files = sorted(root.rglob("*.csv") if CSV_RECURSIVE else root.glob("*/*.csv"))

    rows = []

    for fp in csv_files:
        if not fp.is_file():
            continue

        code = find_code_from_path(fp, root)

        if code not in NEEDED_CODES:
            continue

        device_name = fp.stem
        is_high_device = (code == "N") and (device_name in high_devices)

        max_rows = HIGHLIGHT_MAX_ROWS_PER_FILE if is_high_device else BACKGROUND_MAX_ROWS_PER_FILE
        seed = stable_seed(RANDOM_STATE, str(fp))

        try:
            raw = read_csv_selected(fp, usecols, max_rows=max_rows, seed=seed)
        except Exception as e:
            print(f"[WARN] 跳过 {fp}: {e}")
            continue

        if raw.empty:
            continue

        df = pd.DataFrame()
        df[COL_CONTROL] = to_numeric(raw[COL_CONTROL])
        df[COL_SPEED] = to_numeric(raw[COL_SPEED])
        df[COL_PLATE_DP] = to_numeric(raw[COL_PLATE_DP])
        df[COL_FILTER_DP] = to_numeric(raw[COL_FILTER_DP])
        df["row_id"] = raw["row_id"].astype(np.int64)

        cp_key = control_key(df[COL_CONTROL], CONTROL_ROUND_DIGITS)
        keep_cp = cp_key.eq(target_key)

        speed_ok = in_speed_range(df[COL_SPEED])

        df = df.loc[keep_cp & speed_ok].copy()

        if df.empty:
            continue

        df["source_code"] = code
        df["device_name"] = device_name
        df["device_key"] = f"{code}/{device_name}"
        df["binary_label"] = 0 if code == "N" else 1
        df["is_high_false_alarm_device"] = is_high_device

        df["speed_norm"] = normalize_speed(df[COL_SPEED])
        df["x"] = df["speed_norm"] ** 2
        df["y"] = df[COL_PLATE_DP] + df[COL_FILTER_DP]

        valid = (
            df["x"].notna() &
            df["y"].notna() &
            np.isfinite(df["x"]) &
            np.isfinite(df["y"])
        )

        df = df.loc[valid].copy()

        if not df.empty:
            rows.append(df)

    if not rows:
        raise RuntimeError("没有读取到可用于绘图的数据。")

    data = pd.concat(rows, ignore_index=True)

    print(f"[INFO] 绘图数据总量：{len(data):,}")
    print(data.groupby(["source_code", "is_high_false_alarm_device"]).size())

    return data


# ============================================================
# 6. 抽样与绘图
# ============================================================

def split_normal_train(data: pd.DataFrame) -> set[str]:
    normal_keys = np.array(
        sorted(data.loc[data["binary_label"].eq(0), "device_key"].astype(str).unique())
    )

    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(normal_keys)

    n_train = max(1, int(round(len(normal_keys) * TRAIN_NORMAL_DEVICE_FRAC)))
    n_train = min(n_train, len(normal_keys) - 1)

    return set(normal_keys[:n_train])


def sample_for_plot(data: pd.DataFrame, high_devices: list[str]) -> pd.DataFrame:
    background = data[~data["is_high_false_alarm_device"]].copy()
    high = data[data["is_high_false_alarm_device"]].copy()

    if len(background) > BACKGROUND_SAMPLE_SIZE:
        background = background.sample(
            n=BACKGROUND_SAMPLE_SIZE,
            random_state=RANDOM_STATE
        )

    high_parts = []
    for dev in high_devices:
        sub = high[high["device_name"].astype(str).eq(str(dev))]
        if sub.empty:
            continue

        if len(sub) > POINTS_PER_HIGH_FALSE_DEVICE:
            sub = sub.sample(
                n=POINTS_PER_HIGH_FALSE_DEVICE,
                random_state=stable_seed(RANDOM_STATE, dev)
            )

        high_parts.append(sub)

    high_sampled = pd.concat(high_parts, ignore_index=True) if high_parts else high.iloc[0:0]

    return pd.concat([background, high_sampled], ignore_index=True)


def plot_scatter(data: pd.DataFrame, high_devices: list[str]) -> None:
    train_keys = split_normal_train(data)

    train_df = data[
        (data["binary_label"].eq(0)) &
        (data["device_key"].astype(str).isin(train_keys))
    ].copy()

    model = fit_wls(train_df)

    yhat_train, sigma_train = predict_wls(train_df["x"].to_numpy(dtype=float), model)
    score_train = np.abs(train_df["y"].to_numpy(dtype=float) - yhat_train) / np.maximum(sigma_train, MIN_SIGMA)
    threshold = float(np.quantile(score_train[np.isfinite(score_train)], THRESHOLD_QUANTILE))

    plot_df = sample_for_plot(data, high_devices)

    normal_common = plot_df[
        (plot_df["binary_label"].eq(0)) &
        (~plot_df["is_high_false_alarm_device"])
    ]

    anomaly = plot_df[plot_df["binary_label"].eq(1)]

    high = plot_df[plot_df["is_high_false_alarm_device"]]

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.scatter(
        normal_common["x"],
        normal_common["y"],
        s=7,
        c="#ff7f0e",
        alpha=0.16,
        linewidths=0,
        label=f"普通正常样本 sampled={len(normal_common):,}"
    )

    ax.scatter(
        anomaly["x"],
        anomaly["y"],
        s=7,
        c="#1f77b4",
        alpha=0.20,
        linewidths=0,
        label=f"二次侧异常 sampled={len(anomaly):,}"
    )

    marker_list = [
        "^",      # 三角形
        "s",      # 正方形
        r"$\heartsuit$",  # 心形
        "*",      # 五角星
        "h",      # 六角形
        "x",      # ×号
        "D",      # 菱形
        (4, 1, 45),  # 尖四角星
        r"$\checkmark$",  # 对勾
        "P",      # 加粗十字
        "v",      # 倒三角
        "X",      # 加粗 X
        "p",      # 五边形
        "<",      # 左三角
        ">",      # 右三角
    ]

    color_list = [
        "red", "purple", "green", "brown", "magenta",
        "cyan", "black", "darkorange", "darkred", "navy",
        "teal", "deeppink", "olive", "slateblue", "gray"
    ]

    for i, dev in enumerate(high_devices):
        sub = high[high["device_name"].astype(str).eq(str(dev))]

        if sub.empty:
            continue

        ax.scatter(
            sub["x"],
            sub["y"],
            s=90,
            marker=marker_list[i % len(marker_list)],
            c=color_list[i % len(color_list)],
            edgecolors="black",
            linewidths=0.5,
            alpha=0.95,
            zorder=10,
            label=f"{dev} sampled={len(sub):,}"
        )

    x_min = float(data["x"].quantile(0.01))
    x_max = float(data["x"].quantile(0.99))
    x_grid = np.linspace(x_min, x_max, 300)

    y_grid, sigma_grid = predict_wls(x_grid, model)

    ax.plot(
        x_grid,
        y_grid,
        color="#00a6d6",
        linewidth=2.3,
        label="WLS拟合线"
    )

    ax.plot(
        x_grid,
        y_grid + threshold * sigma_grid,
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.5,
        label=f"阈值带上界 q={THRESHOLD_QUANTILE:.4f}"
    )

    ax.plot(
        x_grid,
        y_grid - threshold * sigma_grid,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.5,
        label=f"阈值带下界 q={THRESHOLD_QUANTILE:.4f}"
    )

    total = len(data)
    normal_total = int(data["binary_label"].eq(0).sum())
    anomaly_total = int(data["binary_label"].eq(1).sum())
    high_total = int(data["is_high_false_alarm_device"].sum())

    text = (
        f"全量有效点 {total:,}\n"
        f"正常 {normal_total:,}；二次侧异常 {anomaly_total:,}\n"
        f"高误报设备 {len(high_devices)} 台；其有效点 {high_total:,}\n"
        f"背景抽样 {len(normal_common) + len(anomaly):,} 点；每台高误报设备最多 {POINTS_PER_HIGH_FALSE_DEVICE} 点"
    )

    ax.text(
        0.01,
        0.99,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.82, edgecolor="gray")
    )

    ax.set_title(
        f"M3 | 控制压差={CONTROL_GROUP} | N中误报率≥{FALSE_ALARM_RATE_CUTOFF:.0%}设备高亮 | q={THRESHOLD_QUANTILE:.4f}",
        fontsize=14
    )

    ax.set_xlabel("(二次侧泵转速/100)^2", fontsize=12)
    ax.set_ylabel("二次侧板换压差 + 二次侧过滤器压差", fontsize=12)

    ax.grid(True, alpha=0.25)

    ax.legend(
        loc="best",
        fontsize=7.2,
        frameon=True,
        markerscale=1.1,
        ncol=1
    )

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"{MODEL_ID}_cp_{CONTROL_GROUP}_q{int(THRESHOLD_QUANTILE * 10000):04d}_false_ge_{int(FALSE_ALARM_RATE_CUTOFF * 100)}"

    fig_path = out_dir / f"{suffix}_highlight_scatter.png"
    csv_path = out_dir / f"{suffix}_highlight_sampled_points.csv"

    high.sort_values(["device_name", "row_id"]).to_csv(csv_path, index=False, encoding="utf-8-sig")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=FIG_DPI)
    plt.close(fig)

    print(f"[DONE] 图片已保存：{fig_path}")
    print(f"[DONE] 高误报设备抽样点已保存：{csv_path}")


# ============================================================
# 7. 主程序
# ============================================================

def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    high_devices = get_high_false_alarm_devices()
    data = load_plot_data(high_devices)
    plot_scatter(data, high_devices)


if __name__ == "__main__":
    main()
