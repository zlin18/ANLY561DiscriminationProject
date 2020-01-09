import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import *

df = pd.read_csv("adult.data", names=['age', 'workclass','fnlwgt', 'education', 
                                      'education-num','marital-status','occupation',
                                      'relationship','race','sex','capital-gain', 
                                      'capital-loss','hours-per-week','native-country',
                                      'income'])

## Drop columns that will not be used
df = df.drop('workclass', axis='columns')
df = df.drop('fnlwgt', axis='columns')
df = df.drop('education', axis='columns')
df = df.drop('marital-status', axis='columns')
df = df.drop('occupation', axis='columns')
df = df.drop('relationship', axis='columns')
df = df.drop('race', axis='columns')
df = df.drop('sex', axis='columns')
df = df.drop('capital-gain', axis='columns')
df = df.drop('capital-loss', axis='columns')
## Remove NA rows
df.dropna(inplace=True)

NativeCountryUS = []
for index, row in df.iterrows():
    if row['native-country'] == ' United-States': 
        NativeCountryUS.append(1)
    else: 
        NativeCountryUS.append(0)

## New column make US and Non-US binary. 
df['NativeCountryUS'] = NativeCountryUS
print(df)
df = df.drop('native-country', axis='columns')

## Pick training and testing sets
y = df['income']
df = df.drop('income', axis='columns')
X = df.drop('NativeCountryUS', axis='columns')

US = df[df.NativeCountryUS == 1]
NonUS = df[df.NativeCountryUS == 0]

p = NonUS.shape[0]/(df.shape[0])

USindex = list(US.index)
NonUSindex = list(NonUS.index)

seed(123)
TrainIndex = sample(USindex, 23336)+sample(NonUSindex, 2713)
TrainIndex.sort()
TestIndex = []
for i in range(len(TrainIndex)):
    if TrainIndex[i] != TrainIndex[26048]: 
        a = TrainIndex[i+1]
        while a-TrainIndex[i] != 1: 
            TestIndex.append(a-1)
            a = a-1

X_train = X.loc[TrainIndex , :]
y_train = [y[index] for index in TrainIndex]

df_test = df.loc[TestIndex , :]
US0 = df_test[df_test.NativeCountryUS == 1]
NonUS0 = df_test[df_test.NativeCountryUS == 0]

seed(123)
OriginalTestIndex = sample(list(US0.index), 3051)+sample(list(NonUS0.index), 339)
X_original_test = X.loc[OriginalTestIndex, :]
y_original_test = [y[index] for index in OriginalTestIndex]

seed(123)
NewTestIndex = sample(list(US0.index), 2712)+sample(list(NonUS0.index), 678)
X_new_test = X.loc[NewTestIndex, :]
y_new_test = [y[index] for index in NewTestIndex]

X_train['income'] = y_train
X_original_test['income'] = y_original_test
X_new_test['income'] = y_new_test

X_train.to_csv('AdultTraining.csv', index=False)
X_original_test.to_csv('AdultOriginalTesting.csv', index=False)
X_new_test.to_csv('AdultNewTesting.csv', index=False)