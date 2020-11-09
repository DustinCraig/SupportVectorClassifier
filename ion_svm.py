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
ion_df = pd.read_csv('ionosphere.data', names=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29', 'f30', 'f31', 'f32', 'f33', 'f34', 'class'])

####### Ion Dataset Analysis #######
X = ion_df[ion_df.columns[:-1]]
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
Y = ion_df[ion_df.columns[-1]]
Y = Y.apply(convertToNumber)

# Plot correlation heatmap
plt.title("Correlation Matrix")
sns.heatmap(ion_df.corr())
plt.show()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, random_state=0)

# Course grid search parameters
tuned_parameters = [
    {
        'kernel': ['linear'], 
        'C': [0.5, 1, 10, 100, 1000]
    },
    {
        'kernel': ['rbf'], 
        'C': [0.5, 1, 10, 100, 1000]
    },
    {
        'kernel': ['poly'],
        'degree': [1,2,3,4,5,6, 20, 50, 100],
        'gamma': [1e-3, 1e-4, 1e-5],
        'C': [0.5, 1, 10, 100, 1000]
    },
    {
        'kernel': ['sigmoid'],
        'C': [0.5, 1, 10, 100, 1000]
    }
]

# Fine grid search parameters
hypertuned_parameters = [
    {
        'kernel': ['rbf'],
        'C': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
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
plt.scatter(means, stds*2)
plt.title('Mean vs Variance')
plt.xlabel('Mean')
plt.ylabel('Variance')
plt.show()
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

# Perform Principal component analysis to get our data in 2-dimensions to be able to plot it
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data=principalComponents, columns=['pca1', 'pca2'])
finalDf = pd.concat([principalDf, ion_df[ion_df.columns[-1]]], axis=1)

X = finalDf[finalDf.columns[:-1]]
Y = finalDf[finalDf.columns[-1]]
Y = Y.apply(convertToNumber)
print("Explained Variance")
print(pca.explained_variance_ratio_)
print()
print() 

print("PCA Components")
print(abs(pca.components_))

h = 0.02
clf = SVC(kernel='rbf', C=3)
clf.fit(X, Y)
x_min, x_max = finalDf[finalDf.columns[0]].min() - 1, finalDf[finalDf.columns[0]].max() + 1
y_min, y_max = finalDf[finalDf.columns[1]].min() - 1, finalDf[finalDf.columns[1]].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
pl.set_cmap('jet')
pl.contourf(xx, yy, Z)
pl.axis('tight')
pl.scatter(X[X.columns[0]], X[X.columns[1]], c=Y)
pl.title('2D PCA with hyper-parameters (rbf, C=3)')
pl.show()