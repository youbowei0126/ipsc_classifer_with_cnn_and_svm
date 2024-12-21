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

# 設置繪圖風格
sns.set_theme(style="whitegrid")

# 創建子圖來顯示兩個不同的結果
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 繪製第一組報告的結果
sns.violinplot(
    x="model",          # x 軸為模型名稱
    y="accuracy",       # y 軸為準確率
    hue="group",        # 顏色分組
    data=df_long_accuracy, # 數據
    split=True,         # 將小提琴圖分為兩半
    inner="point",      # 顯示分位數
    ax=axes[0]          # 指定為第一個子圖
)
axes[0].set_title("without data argumentation")
axes[0].set_xlabel("Model")
axes[0].set_ylabel("Accuracy")
axes[0].legend(title="Group")
axes[0].set_ylim(0.6,1)

for i, model in enumerate(df_long_accuracy2["model"].unique()):
    # 提取每個模型的準確率
    model_accuracy = df_long_accuracy2[df_long_accuracy2["model"] == model]["accuracy"]
    mean_accuracy = model_accuracy.mean()
    ci = stats.t.interval(0.95, len(model_accuracy)-1, loc=mean_accuracy, scale=stats.sem(model_accuracy))

    # 標註平均值和95%置信區間
    axes[1].text(i, mean_accuracy + 0.02, f'{mean_accuracy:.4f} \n(95% CI: {ci[0]:.4f}-{ci[1]:.4f})',
                 horizontalalignment='center', fontsize=10, color='black')

# 在第一個小提琴圖上添加垂直線
for i in range(4):  # 一組四個模型
    axes[0].axvline(x=i+0.4, color='gray')  # 添加垂直線

for i, model in enumerate(df_long_accuracy["model"].unique()):
    # 提取每個模型的準確率
    model_accuracy = df_long_accuracy[df_long_accuracy["model"] == model]["accuracy"]
    mean_accuracy = model_accuracy.mean()
    ci = stats.t.interval(0.95, len(model_accuracy)-1, loc=mean_accuracy, scale=stats.sem(model_accuracy))

    # 標註平均值和95%置信區間
    axes[0].text(i, mean_accuracy + 0.02, f'{mean_accuracy:.4f} \n(95% CI: {ci[0]:.4f}-{ci[1]:.4f})',
                 horizontalalignment='center', fontsize=10, color='black')



# 繪製第二組報告的結果
sns.violinplot(
    x="model",          # x 軸為模型名稱
    y="accuracy",       # y 軸為準確率
    hue="group",        # 顏色分組
    data=df_long_accuracy2, # 數據
    split=True,         # 將小提琴圖分為兩半
    inner="point",      # 顯示分位數
    ax=axes[1]          # 指定為第二個子圖
)
axes[1].set_title("with data argumentation")
axes[1].set_xlabel("Model")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0.6,1)
# 在第二個小提琴圖上添加垂直線
for i in range(4):  # 第二組四個模型
    axes[1].axvline(x=i+0.4, color='gray')  # 添加垂直線

# 顯示圖表
plt.tight_layout()
plt.savefig(r"mast_test/accuracy_8.svg")
plt.show()
