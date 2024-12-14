import joblib
import pandas
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
'''
[input_seed, report_fully_connect, report_svm, report_pca_svm, report_lda_svm, train_history.history]
'''


path_=r"mast_test\reports"
reports=[]
reports.append(0)
for i in range(1,101,1):
    reports.append(joblib.load(path_+fr"\report_{i}.joblib"))
print(reports[1][1])
    
df_accuracy=pd.DataFrame({"seed":[0],"fully_connect":[0],"svm":[0],"pca_svm":[0],"lda_svm":[0]})
for i in range(1,101,1):
    df_accuracy.loc[i,"seed"]=i
    
    df_accuracy.loc[i,"fully_connect"]=reports[i][1]["accuracy"]
    df_accuracy.loc[i,"svm"]=reports[i][2]["accuracy"]
    df_accuracy.loc[i,"pca_svm"]=reports[i][3]["accuracy"]
    df_accuracy.loc[i,"lda_svm"]=reports[i][4]["accuracy"]
    print(i)
    
print(df_accuracy)

mean_values = df_accuracy.mean(axis=0)[1:]  # 排除 'seed' 列
ci_95_lower = []
ci_95_upper = []

for col in df_accuracy.columns[1:]:
    data = df_accuracy[col].values
    ci_95 = 1.96 * np.std(data) / np.sqrt(len(data))  # 95% CI 計算
    ci_95_lower.append(np.mean(data) - ci_95)
    ci_95_upper.append(np.mean(data) + ci_95)

# 畫出長條圖和誤差範圍
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(mean_values.index, mean_values.values, yerr=[np.array(mean_values) - np.array(ci_95_lower), np.array(ci_95_upper) - np.array(mean_values)], capsize=5)

# 添加平均值標註
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, round(yval, 3), ha='center', va='bottom', color='black')

# 添加標籤和標題
ax.set_ylabel('Accuracy')
ax.set_title('Bar Plot with Mean and 95% CI')
ax.set_xticks(np.arange(len(mean_values)))
ax.set_xticklabels(mean_values.index)
plt.xticks(rotation=45)

# 顯示圖形
plt.show()