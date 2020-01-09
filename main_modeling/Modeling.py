# 561 Project
# Kuiyu Zhu
# Group 3, other members: Tianxing Jiang, Zhiyu Lin, Ning Hu



# 1. country
# null. race
# 2. sex

# Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from sklearn.metrics import explained_variance_score, mean_absolute_error, median_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')
import time

# Training set
train = pd.read_csv('AdultTraining 1.csv')
train = train.replace("<=50K", 0, regex=True)
train = train.replace(">50K", 1, regex=True)
# transform label
train_array = train.values
X_train = train_array[:, 0:2]
y_train = train_array[:, 3]

# Testing set 1
test1 = pd.read_csv('AdultOriginalTesting 1.csv')
test1 = test1.replace("<=50K", 0, regex=True)
test1 = test1.replace(">50K", 1, regex=True)
test1_array = test1.values
X_test1 = test1_array[:, 0:2]
y_test1 = test1_array[:, 3]


# Testing set 2
test2 = pd.read_csv('AdultNewTesting 1.csv')
test2 = test2.replace("<=50K", 0, regex=True)
test2 = test2.replace(">50K", 1, regex=True)
test2_array = test2.values
X_test2 = test2_array[:, 0:2]
y_test2 = test2_array[:, 3]

# Compare testing sets with LinearSVC, LogisticRegression

# svc
svc = SVC().fit(X_train, y_train)
pred1 = svc.predict(X_test1)
pred2 = svc.predict(X_test2)
print('_______'*5)
print('____SVC____')
print('Precision Score:')
print(precision_score(y_test1, pred1, average='binary'), ' vs ',  precision_score(y_test2, pred2, average='binary'))
print('Accuracy Score:')
print(accuracy_score(y_test1, pred1), ' vs ', precision_score(y_test2, pred2))
print('f1-score:')
print(f1_score(y_test1, pred1, average='binary'), ' vs ',  f1_score(y_test2, pred2, average='binary'))
print('recall-score:')
print(recall_score(y_test1, pred1,), ' vs ',  recall_score(y_test2, pred2))
print('roc_auc_score:')
print(roc_auc_score(y_test1, pred1,), ' vs ',  roc_auc_score(y_test2, pred2))


# lg
time1 = time.time()
lg = LogisticRegression(solver='lbfgs').fit(X_train, y_train)
pred1 = lg.predict(X_test1)
pred2 = lg.predict(X_test2)
print('_______'*5)
print('____LogisticReg____')
print('Precision Score:')
print(precision_score(y_test1, pred1, average='binary'), ' vs ', precision_score(y_test2, pred2, average='binary'))
print('Accuracy Score:')
print(accuracy_score(y_test1, pred1), ' vs ', precision_score(y_test2, pred2))
print('f1-score:')
print(f1_score(y_test1, pred1, average='binary'), ' vs ', f1_score(y_test2, pred2, average='binary'))
print('recall-score:')
print(recall_score(y_test1, pred1,), ' vs ',  recall_score(y_test2, pred2))
print('roc_auc_score:')
print(roc_auc_score(y_test1, pred1,), ' vs ',  roc_auc_score(y_test2, pred2))
time2 = time.time()
print(str(time2-time1))
# lr
lr = LinearRegression().fit(X_train, y_train)
pred1 = lr.predict(X_test1)
pred2 = lr.predict(X_test2)
print('_______'*5)
print('____LinearReg____')
print('mean_absolute_error:')
print(mean_absolute_error(y_test1, pred1), ' vs ', mean_absolute_error(y_test2, pred2))
print('median_absolute_error:')
print(median_absolute_error(y_test1, pred1), ' vs ', median_absolute_error(y_test2, pred2))
print('r2-score:')
print(r2_score(y_test1, pred1), ' vs ', r2_score(y_test2, pred2))
print('explained_value_score:')
print(explained_variance_score(y_test1, pred1), ' vs ', explained_variance_score(y_test2, pred2))
print('_______'*5)
