import joblib
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# 加載第一組報告數據
path_ = r"mast_test\reports"
reports = []
reports.append(0)
for i in range(1, 101):
    print(os.path.exists(fr"mast_test/reports/report_{i}.joblib"))
    reports.append(joblib.load(fr"mast_test/reports/report_{i}.joblib"))

# 構建第一組準確率數據框
df_accuracy = pd.DataFrame({"seed": [0], "fully_connect": [0], "svm": [0], "pca_svm": [0], "lda_svm": [0]})
for i in range(1, 101):
    df_accuracy.loc[i, "seed"] = i
    df_accuracy.loc[i, "fully_connect"] = reports[i][1]["accuracy"]
    df_accuracy.loc[i, "svm"] = reports[i][2]["accuracy"]
    df_accuracy.loc[i, "pca_svm"] = reports[i][3]["accuracy"]
    df_accuracy.loc[i, "lda_svm"] = reports[i][4]["accuracy"]

# 加載第二組報告數據
path_ = r"mast_test\reports2"
reports2 = []
reports2.append(0)
for i in range(1, 6):
    print(os.path.exists(fr"mast_test/reports2/report_{i}.joblib"))
    reports2.append(joblib.load(fr"mast_test/reports2/report_{i}.joblib"))

# 構建第二組準確率數據框
df_accuracy2 = pd.DataFrame({"seed": [0], "fully_connect": [0], "svm": [0], "pca_svm": [0], "lda_svm": [0]})
for i in range(1, 6):
    df_accuracy2.loc[i, "seed"] = i
    df_accuracy2.loc[i, "fully_connect"] = reports2[i][1]["accuracy"]
    df_accuracy2.loc[i, "svm"] = reports2[i][2]["accuracy"]
    df_accuracy2.loc[i, "pca_svm"] = reports2[i][3]["accuracy"]
    df_accuracy2.loc[i, "lda_svm"] = reports2[i][4]["accuracy"]

# 填充 NaN 值
df_accuracy["seed"] = df_accuracy["seed"].fillna(-1)
df_accuracy2["seed"] = df_accuracy2["seed"].fillna(-1)

# 添加 'group' 列來區分數據來源
df_accuracy["group"] = "Reports"
df_accuracy2["group"] = "Reports2"

# 過濾掉 'seed' 為 0 的行
df_accuracy = df_accuracy[df_accuracy["seed"] != 0]
df_accuracy2 = df_accuracy2[df_accuracy2["seed"] != 0]

# 合併兩組數據
df_long_accuracy = df_accuracy.melt(id_vars=["seed", "group"], var_name="model", value_name="accuracy")
df_long_accuracy2 = df_accuracy2.melt(id_vars=["seed", "group"], var_name="model", value_name="accuracy")


for i, model in enumerate(df_long_accuracy2["model"].unique()):
    # 提取每個模型的準確率
    model_accuracy = df_long_accuracy2[df_long_accuracy2["model"] == model]["accuracy"]
    mean_accuracy = model_accuracy.mean()
    ci = stats.t.interval(0.95, len(model_accuracy)-1, loc=mean_accuracy, scale=stats.sem(model_accuracy))
    print(mean_accuracy,ci[0],ci[1])


for i, model in enumerate(df_long_accuracy["model"].unique()):
    # 提取每個模型的準確率
    model_accuracy = df_long_accuracy[df_long_accuracy["model"] == model]["accuracy"]
    mean_accuracy = model_accuracy.mean()
    ci = stats.t.interval(0.95, len(model_accuracy)-1, loc=mean_accuracy, scale=stats.sem(model_accuracy))
    print(mean_accuracy,ci[0],ci[1])


