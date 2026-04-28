import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px


# ======================================
# 1. 路径：改成你的文件路径
# ======================================
INPUT_CSV = Path(r"C:\Users\86158\Desktop\device_summary_model3_merged_by_device.csv")
OUTPUT_HTML = INPUT_CSV.parent / "model3_all_devices_interactive_scatter.html"


# ======================================
# 2. 读取数据
# ======================================
df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")


# ======================================
# 3. 自动识别列名
# ======================================
def normalize_col_name(col):
    col = str(col).strip()
    col = re.sub(r"\.\d+$", "", col)
    return col.lower()


def find_col(df, candidates, required=True):
    norm_to_real = {}
    for c in df.columns:
        norm_to_real[normalize_col_name(c)] = c

    for cand in candidates:
        cand_norm = cand.lower()
        if cand_norm in norm_to_real:
            return norm_to_real[cand_norm]

    for c in df.columns:
        c_norm = normalize_col_name(c)
        for cand in candidates:
            if c_norm.startswith(cand.lower()):
                return c

    if required:
        raise ValueError(f"没有找到这些列中的任何一个：{candidates}")
    return None


device_col = find_col(df, ["device_name", "device"], required=True)
evaluable_col = find_col(df, ["evaluable_samples", "evaluable_sample", "evaluable_count"], required=True)
n_groups_col = find_col(df, ["n_control_groups"], required=True)
control_groups_col = find_col(df, ["control_groups"], required=True)
merge_type_col = find_col(df, ["merge_type"], required=True)

# 这里优先找最基础的一组 anomaly_count / anomaly_rate
anomaly_count_col = find_col(df, ["anomaly_count", "anomaly_c"], required=True)
anomaly_rate_col = find_col(df, ["anomaly_rate", "anomaly_r"], required=True)

# 有些列可能没有
total_col = find_col(df, ["total_samples", "total_sample", "total"], required=False)
severe_col = find_col(df, ["severe_count", "severe_c"], required=False)


# ======================================
# 4. 数据清理
# ======================================
# 转成数值
for c in [evaluable_col, anomaly_count_col, anomaly_rate_col, n_groups_col]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

if total_col is not None:
    df[total_col] = pd.to_numeric(df[total_col], errors="coerce")

if severe_col is not None:
    df[severe_col] = pd.to_numeric(df[severe_col], errors="coerce")

# 去掉关键字段缺失
df_plot = df.dropna(subset=[device_col, evaluable_col, anomaly_rate_col, anomaly_count_col]).copy()

# 异常率截断在 [0,1]
df_plot[anomaly_rate_col] = df_plot[anomaly_rate_col].clip(lower=0, upper=1)

# 工况数转成字符串用于上色
df_plot["工况数"] = df_plot[n_groups_col].astype("Int64").astype(str)

# 更友好的中文标签
merge_type_map = {
    "single_control_keep_original": "单工况-保留原始指标",
    "multi_control_aggregate_basic_counts_only": "多工况-仅合并基本数量",
    "duplicate_same_control_aggregate_basic_counts_only": "同工况重复-仅合并基本数量"
}
df_plot["合并类型"] = df_plot[merge_type_col].astype(str).map(merge_type_map).fillna(df_plot[merge_type_col].astype(str))

# 气泡大小：避免极端值过大，用平方根缩放
df_plot["bubble_size"] = np.sqrt(df_plot[anomaly_count_col].clip(lower=0))

# 为悬停显示构造文本
hover_cols = {
    device_col: True,
    evaluable_col: ":,.0f",
    anomaly_count_col: ":,.0f",
    anomaly_rate_col: ".2%",
    control_groups_col: True,
    n_groups_col: True,
    "合并类型": True,
}

if total_col is not None:
    hover_cols[total_col] = ":,.0f"

if severe_col is not None:
    hover_cols[severe_col] = ":,.0f"


# ======================================
# 5. 绘图
# ======================================
fig = px.scatter(
    df_plot,
    x=evaluable_col,
    y=anomaly_rate_col,
    size="bubble_size",
    color="工况数",
    hover_name=device_col,
    hover_data=hover_cols,
    title="模型3：合并所有控制组后的设备异常率总览图",
    labels={
        evaluable_col: "可评估样本数",
        anomaly_rate_col: "异常率",
        "工况数": "涉及工况数"
    }
)

fig.update_traces(
    marker=dict(sizemode="area", line=dict(width=1)),
    opacity=0.75
)

fig.update_layout(
    xaxis_title="evaluable_samples（可评估样本数）",
    yaxis_title="anomaly_rate（异常率）",
    yaxis=dict(range=[0, 1.05]),
    legend_title_text="涉及工况数",
    title=dict(x=0.5),
    template="plotly_white",
    hoverlabel=dict(font_size=13)
)

# 保存为 HTML
fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn")

print("绘图完成！")
print(f"输出文件：{OUTPUT_HTML}")
fig.show()
