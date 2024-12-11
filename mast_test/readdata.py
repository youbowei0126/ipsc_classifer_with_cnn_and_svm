import joblib
import pandas
import numpy as np
from sklearn.metrics import classification_report


reports=joblib.load(r"mast_test\data.joblib")
print(np.array(reports,dtype=object).shape)
print(reports[1][2])
print(reports[2][2])