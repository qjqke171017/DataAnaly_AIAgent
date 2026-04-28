import re
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# 1. 修改这里的输入输出路径
# =========================
INPUT_CSV = Path("device_summary_all.csv")

OUTPUT_CSV = Path("device_summary_model3_merged_by_device.csv")
CHECK_CSV = Path("device_summary_model3_device_condition_check.csv")


# =========================
# 2. 工具函数
# =========================

def normalize_col_name(col):
    """
    pandas 读取重复列名时，可能会把 anomaly_count 变成 anomaly_count.1。
    这个函数用于判断列名类型时去掉 .1/.2 这种后缀。
    """
    col = str(col).strip()
    col = re.sub(r"\.\d+$", "", col)
    return col.lower()


def clean_device_name(x):
    """
    去掉设备名末尾的一个或多个下划线。
    例如：
    ucb23a-kl_ -> ucb23a-kl
    """
    if pd.isna(x):
        return x
    x = str(x).strip()
    x = re.sub(r"_+$", "", x)
    return x


def fmt_control_group(x):
    """
    把工况值格式化成比较干净的字符串。
    例如 0.800000 -> 0.8
    """
    if pd.isna(x):
        return ""
    try:
        xf = float(x)
        return f"{xf:g}"
    except Exception:
        return str(x).strip()


def sort_control_key(x):
    """
    工况排序用。
    """
    try:
        return float(x)
    except Exception:
        return str(x)


def find_col(df, candidates, required=True):
    """
    根据候选列名寻找真实列名。
    支持精确匹配和前缀匹配。
    """
    norm_to_real = {}
    for c in df.columns:
        norm_to_real[normalize_col_name(c)] = c

    # 精确匹配
    for cand in candidates:
        cand_norm = cand.lower()
        if cand_norm in norm_to_real:
            return norm_to_real[cand_norm]

    # 前缀匹配
    for c in df.columns:
        c_norm = normalize_col_name(c)
        for cand in candidates:
            if c_norm.startswith(cand.lower()):
                return c

    if required:
        raise ValueError(f"没有找到这些列中的任何一个：{candidates}")
    return None


def safe_divide(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def safe_numeric_sum(series):
    return pd.to_numeric(series, errors="coerce").sum(min_count=1)


def safe_name(col):
    """
    用于生成派生列名。
    """
    col = str(col)
    col = col.replace(".", "_")
    col = re.sub(r"[^0-9a-zA-Z_\u4e00-\u9fa5]+", "_", col)
    return col.strip("_")


# =========================
# 3. 读取数据
# =========================

df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
df.columns = [str(c).strip() for c in df.columns]

device_col = find_col(df, ["device_name", "device"], required=True)
control_col = find_col(df, ["control_group", "control", "group"], required=True)

model_id_col = find_col(df, ["model_id"], required=False)
model_name_col = find_col(df, ["model_name"], required=False)

total_col = find_col(df, ["total_samples", "total_sample", "total"], required=False)
evaluable_col = find_col(
    df,
    ["evaluable_samples", "evaluable_sample", "evaluable_count", "evaluable"],
    required=True
)

# 清理设备名末尾的下划线
df[device_col] = df[device_col].apply(clean_device_name)


# =========================
# 4. 只保留 model3
# =========================

mask_model3 = pd.Series(False, index=df.index)

if model_id_col is not None:
    s = df[model_id_col].astype(str).str.lower().str.strip()

    # 匹配 m3、m3_xxx、model3、model_3、3 等情况
    mask_model3 |= s.str.contains(
        r"(^|[^a-z0-9])m3([^0-9a-z]|$)|"
        r"(^|[^a-z0-9])model_?3([^0-9a-z]|$)|"
        r"^3$",
        regex=True,
        na=False
    )

if model_name_col is not None:
    s = df[model_name_col].astype(str).str.lower().str.strip()

    # 匹配 模型3、模型 3、model3、model 3 等情况
    mask_model3 |= s.str.contains(
        r"模型\s*3|model\s*3|model3",
        regex=True,
        na=False
    )

df_m3 = df.loc[mask_model3].copy()

if df_m3.empty:
    print("没有筛选到 model3。请检查 model_id 或 model_name 的实际写法。")
    if model_id_col is not None:
        print("\nmodel_id 的取值示例：")
        print(df[model_id_col].drop_duplicates().head(20))
    if model_name_col is not None:
        print("\nmodel_name 的取值示例：")
        print(df[model_name_col].drop_duplicates().head(20))
    raise SystemExit


# =========================
# 5. 识别可合并的基本数量列
# =========================

sum_base_names = {
    "total_samples",
    "total_sample",
    "missing_records",
    "missing_record",
    "invalid_speed",
    "invalid_speeds",
    "invalid_resid",
    "invalid_residual",
    "invalid_residuals",
    "excluded_samples",
    "excluded_sample",
    "evaluable_samples",
    "evaluable_sample",
    "evaluable_count",
    "n_raw_files",
}

anomaly_count_cols = []
anomaly_rate_cols = []
severe_count_cols = []

for c in df_m3.columns:
    c_norm = normalize_col_name(c)

    if c_norm.startswith("anomaly_count") or c_norm.startswith("anomaly_c"):
        anomaly_count_cols.append(c)

    if c_norm.startswith("anomaly_rate") or c_norm.startswith("anomaly_r"):
        anomaly_rate_cols.append(c)

    if c_norm.startswith("severe_count") or c_norm.startswith("severe_c"):
        severe_count_cols.append(c)


# anomaly_count 和 anomaly_rate 按列出现顺序配对
anomaly_rate_pair = {}
for i, c in enumerate(anomaly_count_cols):
    if i < len(anomaly_rate_cols):
        anomaly_rate_pair[c] = anomaly_rate_cols[i]
    else:
        anomaly_rate_pair[c] = None


basic_sum_cols = []

for c in df_m3.columns:
    c_norm = normalize_col_name(c)

    if c_norm in sum_base_names:
        basic_sum_cols.append(c)
    elif c in anomaly_count_cols:
        basic_sum_cols.append(c)
    elif c in severe_count_cols:
        basic_sum_cols.append(c)

# 去重但保留顺序
basic_sum_cols = list(dict.fromkeys(basic_sum_cols))


# 为每个 anomaly_count 新增一个 normal_count
normal_count_cols = {
    c: f"normal_count_for_{safe_name(c)}"
    for c in anomaly_count_cols
}

# 为 severe_count 新增 severe_rate
severe_rate_cols = {
    c: f"severe_rate_for_{safe_name(c)}"
    for c in severe_count_cols
}


# =========================
# 6. 生成检查表：每个设备涉及几个工况
# =========================

check_rows = []

for dev, g in df_m3.groupby(device_col, dropna=False, sort=True):
    controls = sorted(
        [fmt_control_group(x) for x in g[control_col].dropna().unique()],
        key=sort_control_key
    )

    row = {
        device_col: dev,
        "source_row_count": len(g),
        "n_control_groups": len(controls),
        "control_groups": "|".join(controls),
        "is_multi_control_group": len(controls) > 1,
        "evaluable_samples_sum": safe_numeric_sum(g[evaluable_col]),
    }

    if total_col is not None:
        row["total_samples_sum"] = safe_numeric_sum(g[total_col])

    check_rows.append(row)

check_df = pd.DataFrame(check_rows)
check_df.to_csv(CHECK_CSV, index=False, encoding="utf-8-sig")


# =========================
# 7. 合并同一设备
# =========================

meta_cols = [
    device_col,
    "source_row_count",
    "n_control_groups",
    "control_groups",
    "merge_type",
]

derived_cols = list(normal_count_cols.values()) + list(severe_rate_cols.values())

# 输出列：元信息 + 原始列 + 派生列
output_cols = (
    meta_cols
    + [c for c in df_m3.columns if c != device_col]
    + derived_cols
)

# 去重但保留顺序
output_cols = list(dict.fromkeys(output_cols))

merged_rows = []

for dev, g in df_m3.groupby(device_col, dropna=False, sort=True):
    g = g.copy()

    controls = sorted(
        [fmt_control_group(x) for x in g[control_col].dropna().unique()],
        key=sort_control_key
    )

    n_controls = len(controls)
    source_row_count = len(g)

    row = {c: np.nan for c in output_cols}

    row[device_col] = dev
    row["source_row_count"] = source_row_count
    row["n_control_groups"] = n_controls
    row["control_groups"] = "|".join(controls)

    # 情况一：这个设备只出现了一行，也就是只涉及一个工况
    # 这种情况直接保留原 device_summary 的所有指标
    if source_row_count == 1:
        first = g.iloc[0]

        for c in df_m3.columns:
            if c == device_col:
                row[c] = dev
            else:
                row[c] = first[c]

        row["merge_type"] = "single_control_keep_original"

        denom = pd.to_numeric(pd.Series([first[evaluable_col]]), errors="coerce").iloc[0]

        for count_col in anomaly_count_cols:
            cnt = pd.to_numeric(pd.Series([first[count_col]]), errors="coerce").iloc[0]
            row[normal_count_cols[count_col]] = denom - cnt if pd.notna(denom) and pd.notna(cnt) else np.nan

        for severe_col in severe_count_cols:
            sev = pd.to_numeric(pd.Series([first[severe_col]]), errors="coerce").iloc[0]
            row[severe_rate_cols[severe_col]] = safe_divide(sev, denom)

    # 情况二：这个设备出现多行
    # 如果涉及多个工况，则只合并可计算的基本数量指标
    # 如果是同一工况重复出现，也按基本数量合并，并在 merge_type 中标注
    else:
        if n_controls > 1:
            row["merge_type"] = "multi_control_aggregate_basic_counts_only"
        else:
            row["merge_type"] = "duplicate_same_control_aggregate_basic_counts_only"

        # model 信息保留第一行即可
        if model_id_col is not None:
            row[model_id_col] = g.iloc[0][model_id_col]
        if model_name_col is not None:
            row[model_name_col] = g.iloc[0][model_name_col]

        # control_group 列写成 0.8|1.4|1.65
        row[control_col] = "|".join(controls)

        # 对可以相加的基本数量列求和
        for c in basic_sum_cols:
            row[c] = safe_numeric_sum(g[c])

        denom = row[evaluable_col]

        # anomaly_count 求和后，anomaly_rate 重新计算
        for count_col in anomaly_count_cols:
            cnt_sum = row[count_col]

            rate_col = anomaly_rate_pair.get(count_col)
            if rate_col is not None:
                row[rate_col] = safe_divide(cnt_sum, denom)

            row[normal_count_cols[count_col]] = (
                denom - cnt_sum
                if pd.notna(denom) and pd.notna(cnt_sum)
                else np.nan
            )

        # severe_count 求和后，severe_rate 重新计算
        for severe_col in severe_count_cols:
            sev_sum = row[severe_col]
            row[severe_rate_cols[severe_col]] = safe_divide(sev_sum, denom)

    merged_rows.append(row)

merged_df = pd.DataFrame(merged_rows, columns=output_cols)


# =========================
# 8. 保存结果
# =========================

merged_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("处理完成。")
print(f"model3 原始行数：{len(df_m3)}")
print(f"model3 合并后设备数：{len(merged_df)}")
print(f"涉及多个工况的设备数：{(check_df['n_control_groups'] > 1).sum()}")

print(f"\n已输出合并结果：{OUTPUT_CSV}")
print(f"已输出工况检查表：{CHECK_CSV}")

print("\n识别到的 anomaly_count 列：")
print(anomaly_count_cols)

print("\n识别到的 anomaly_rate 列：")
print(anomaly_rate_cols)

print("\n识别到的 severe_count 列：")
print(severe_count_cols)
