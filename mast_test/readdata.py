import joblib
import pandas
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
import os
import seaborn as sns
import scipy.stats as stats

'''
[input_seed, report_fully_connect, report_svm, report_pca_svm, report_lda_svm, train_history.history]
'''
# mast_test\reports\report_1.joblib

path_=r"mast_test\reports"
reports=[]
reports.append(0)
for i in range(1,101,1):
    print(os.path.exists(fr"mast_test/reports/report_{i}.joblib"))
    reports.append(joblib.load(fr"mast_test/reports/report_{i}.joblib"))
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

mean_values = df_accuracy.mean(axis=0)[1:]  
ci_95_lower = []
ci_95_upper = []

for col in df_accuracy.columns[1:]:
    data = df_accuracy[col].values
    ci_95 = 1.96 * np.std(data) / np.sqrt(len(data))  
    ci_95_lower.append(np.mean(data) - ci_95)
    ci_95_upper.append(np.mean(data) + ci_95)


fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(mean_values.index, mean_values.values, yerr=[np.array(mean_values) - np.array(ci_95_lower), np.array(ci_95_upper) - np.array(mean_values)], capsize=5)


for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, round(yval, 3), ha='center', va='bottom', color='black')


ax.set_ylabel('Accuracy')
ax.set_xticks(np.arange(len(mean_values)))
ax.set_xticklabels(mean_values.index)
plt.xticks(rotation=45)
plt.grid(clip_on=False)
plt.savefig(r"mast_test/accuracy_bar.svg",Transparent=True)


# plt.show()



df_long = pd.melt(df_accuracy, id_vars="seed", 
                  var_name="Model", 
                  value_name="Accuracy")

plt.figure(figsize=(10, 6))

sns.violinplot(x="Model", y="Accuracy", data=df_long, inner="point", palette="muted")
plt.xticks(fontsize=20)

plt.title("Model Accuracy Distribution", fontsize=20)
plt.xlabel("Model", fontsize=20)
plt.ylabel("Accuracy", fontsize=20)
plt.ylim(0.5,1)

p_value_2 = stats.ttest_ind(df_accuracy["fully_connect"], df_accuracy["svm"]).pvalue
p_value_3 = stats.ttest_ind(df_accuracy["fully_connect"], df_accuracy["pca_svm"]).pvalue
p_value_4 = stats.ttest_ind(df_accuracy["fully_connect"], df_accuracy["lda_svm"]).pvalue
print(p_value_2,p_value_3,p_value_4)
# plt.grid(clip_on=False)
plt.savefig(r"mast_test/accuracy.svg",Transparent=True)
plt.show()