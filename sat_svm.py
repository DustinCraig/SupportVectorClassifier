import numpy as np
import pandas as pd
import pylab as pl
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import pylab as pl
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def warn(*args, **kwargs):
    pass
warnings.warn = warn

# Load sat dataset
sat_df = pd.read_csv('sat.trn', delim_whitespace=True)
sat_df_test = pd.read_csv('sat.tst', delim_whitespace=True)

####### Dataset Analysis #######
X = sat_df[sat_df.columns[:-1]]
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
Y = sat_df[sat_df.columns[-1]]
X2 = sat_df_test[sat_df_test.columns[:-1]]
Y2 = sat_df_test[sat_df_test.columns[-1]]

x_train, y_train = X, Y
x_test, y_test = X2, Y2
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, random_state=0)
# y_train_converted = y_train.values.ravel()

# # Course grid search parameters
tuned_parameters = [
    {
        'kernel': ['linear'], 
        'C': [1, 10]
    },
    {
        'kernel': ['rbf'], 
        'C': [1, 10]
    },
    {
        'kernel': ['poly'],
        'degree': [1,2,3],
        'gamma': [1e-3, 1e-4, 1e-5],
        'C': [1, 10]
    },
    {
        'kernel': ['sigmoid'],
        'C': [1, 10]
    }
]

hypertuned_parameters = [
    {
        'kernel': ['poly'],
        'degree': [1,2,3,4],
        'gamma':[1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
        'C': [1,2,3,4,5]
    }
]

# Best parameters: 
# {'C': 3, 'degree': 4, 'gamma': 1e-05, 'kernel': 'poly'}


print('# Performing Course Grid Search')
print()

clf = GridSearchCV(
    SVC(), tuned_parameters
)
clf.fit(x_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()