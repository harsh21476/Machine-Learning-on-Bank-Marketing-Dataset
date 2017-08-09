from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier

# Load libraries
import pandas
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
np.set_printoptions(suppress=True)

filename = 'bank_encoded.csv'
names = ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y"]
dataset = pandas.read_csv(filename,skipinitialspace=True,names=names)
#
# filename = 'testdata_encoded.csv'
# names = ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y"]
# dataset2 = pandas.read_csv(filename,skipinitialspace=True,names=names)



# shape
#print(dataset.shape)

# head
#print(dataset.head(20))

# descriptions
#print(dataset.describe())

# class distribution
#print(dataset.groupby('y').size())
#print(dataset2.groupby('y').size())
# Split-out validation dataset
array = dataset.values
X = array[:,0:16]
Y = array[:,16]
print X
print Y
# array2 = dataset2.values
# X2 = array2[:,0:16]
# Y2 = array2[:,16]

#print(array)
#print(X)
#print(Y)


'''
===============================================================================================================
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
================================================================================
'''
print X.shape, Y.shape
Y = np.reshape(Y,(45211,1))
print Y.shape

kf = KFold(n_splits=10)
kf.get_n_splits(X)
accucray_array =[]
for train_index, test_index in kf.split(X):
    print ("train and test")
    print train_index
    print test_index
    clf = LinearSVC()
    # clf = SVC()
    clf.fit(X[train_index], Y[train_index])
    predictions = clf.predict(X[test_index])
    print("SVC accuracy score:  ")
    a = accuracy_score(Y[test_index], predictions)
    print a
    accucray_array.append(a)
    print(confusion_matrix(Y[test_index], predictions))
    #print knn_accucray_array
    print ("/n/n/n/n")

print ("SVC mean accuray:  ")
print (np.mean(accucray_array))
#print (1-np.mean(knn_accucray_array))