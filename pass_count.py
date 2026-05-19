import pandas as pd

path = r"D:\你的输出目录\tables\wls_repeat_metrics_by_threshold_control.csv"
out_path = r"D:\你的输出目录\tables\pass_count_by_model_threshold.csv"

df = pd.read_csv(path)

# 只看正式评价口径：heldout + all
d = df[
    (df["eval_scope"] == "heldout") &
    (df["control_group"].astype(str) == "all")
].copy()

d["pass_flag"] = (d["fpr"] < 0.05) & (d["recall"] > 0.50)

summary = (
    d.groupby(["model_id", "model_name", "threshold_quantile"], as_index=False)
     .agg(
         repeat_count=("pass_flag", "size"),
         pass_count=("pass_flag", "sum"),
         pass_rate=("pass_flag", "mean"),
         fpr_mean=("fpr", "mean"),
         fpr_std=("fpr", "std"),
         recall_mean=("recall", "mean"),
         recall_std=("recall", "std"),
         precision_mean=("precision", "mean"),
         accuracy_mean=("accuracy", "mean"),
         f1_mean=("f1", "mean"),
         mcc_mean=("mcc", "mean"),
         balanced_accuracy_mean=("balanced_accuracy", "mean"),
     )
)

summary = summary.sort_values(
    ["pass_count", "recall_mean", "fpr_mean"],
    ascending=[False, False, True]
)

summary.to_csv(out_path, index=False, encoding="utf-8-sig")

print(summary)
print(f"已输出：{out_path}")
