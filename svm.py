import numpy as np
import pandas as pd
import pylab as pl
import warnings
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC

def warn(*args, **kwargs):
    pass
warnings.warn = warn

# Load spectral dataset
sat_df = pd.read_csv('sat.trn')

# Load ionosphere dataset
ion_df = pd.read_csv('ionosphere.data')

####### Ion Dataset Analysis #######
X = ion_df[ion_df.columns[:-1]]
Y = ion_df[ion_df.columns[-1]]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, random_state=0)
y_train_converted = y_train.values.ravel()
svc     = svm.SVC(kernel='linear').fit(x_train, y_train)
predicted_linear = svc.predict(x_test)


# Course grid search parameters
tuned_parameters = [
    {
        'kernel': ['linear'], 
        'C': [1, 10, 100, 1000]
    },
    {
        'kernel': ['poly'], 
        'degree': [2, 3, 4],
        'C': [1, 10, 100, 1000]
    },
    {
        'kernel': ['rbf'], 
        'gamma': [1e-3, 1e-4],
        'C': [1, 10, 100, 1000]
    }
]

# Fine grid search parameters
hypertuned_parameters = [
    {
        'kernel': ['poly'],
        'C': [0.5, 1, 5, 10, 15, 20, 25],
        'degree': [1,2,3,4,5,6]
    }
]

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

print('# Performing fine grid search')
print()

clf = GridSearchCV(
    SVC(), hypertuned_parameters
)
clf.fit(x_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()

print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(x_test)
print(classification_report(y_true, y_pred))