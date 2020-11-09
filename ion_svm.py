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

def convertToNumber(row):
    if row[0] == 'g':
        return 1
    elif row[0] == 'b':
        return 0

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
        'kernel': ['rbf', 'sigmoid'], 
        'gamma': [1e-3, 1e-4],
        'C': [1, 10, 100, 1000]
    },
    {
        'kernel': ['poly'],
        'degree': [1,2,3,4,5,6],
        'gamma': [1e-3, 1e-4, 1e-5],
        'C': [1, 10, 100, 1000]
    }
]

# Fine grid search parameters
hypertuned_parameters = [
    {
        'kernel': ['linear'],
        'C': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
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

print('Creating plots...')
# Perform Principal component analysis to get our data in 3-dimensions to be able to plot it
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data=principalComponents, columns=['pca1', 'pca2'])
finalDf = pd.concat([principalDf, ion_df[ion_df.columns[-1]]], axis=1)
X = finalDf[finalDf.columns[:-1]]
Y = finalDf[finalDf.columns[-1]]
Y = Y.apply(convertToNumber)
print("How much of our variance is explained?")
print(pca.explained_variance_ratio_)
print()
print() 

print("Which features matter most?")
print(abs(pca.components_))

h = 0.2
clf = SVC(kernel='linear', C=5)
clf.fit(X, Y)
x_min, x_max = principalDf[principalDf.columns[0]].min() - 1, principalDf[principalDf.columns[0]].max() + 1
y_min, y_max = principalDf[principalDf.columns[1]].min() - 1, principalDf[principalDf.columns[1]].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
print('Z ', Z)
Z = Z.reshape(xx.shape)
pl.contourf(xx, yy, Z)
pl.axis('tight')
# pl.scatter(X[X.columns[0]], X[X.columns[1]], c=Y)
pl.title('2D PCA with best hyper-parameters (Linear, C=5)')
pl.show()