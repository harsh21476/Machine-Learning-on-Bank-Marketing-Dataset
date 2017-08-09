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
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
np.set_printoptions(suppress=True)

filename = 'bank_encoded.csv'
names = ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y"]
dataset = pandas.read_csv(filename,skipinitialspace=True,names=names)

filename = 'testdata_encoded.csv'
names = ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y"]
dataset2 = pandas.read_csv(filename,skipinitialspace=True,names=names)



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
array2 = dataset2.values
X2 = array2[:,0:16]
Y2 = array2[:,16]

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

'''
#=====================================================================================
# Make predictions on validation dataset
#---------KNN-----------#
knn = KNeighborsClassifier()
knn.fit(X,Y)
a=6
#predictions = knn.predict([X[a]])
print ("Actual Output: ")
print Y[a]
#print("Knn output: ")
#print predictions
#print(accuracy_score(Y, predictions))


#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))

#----------------NAIVE BAYES------------#
nb = GaussianNB()

# Train the model using the training sets
nb.fit(X,Y)

#Predict Output
predicted= nb.predict([X[a]])
print ("Bayes_output")
print predicted

#print(accuracy_score(Y, predicted))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))

#==============================================================================================
'''

# Make predictions on validation dataset
#---------KNN-----------#
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,Y)
predictions = knn.predict(X)

print("KNN accuracy score:  ")
print(accuracy_score(Y, predictions))
'''print ("Confusion Matrix")
print(confusion_matrix(Y2, predictions))
print ("Classifcation Report")
print(classification_report(Y2, predictions))
'''
#----------------NAIVE BAYES------------#
nb = GaussianNB()

# Train the model using the training sets
nb.fit(X,Y)

#Predict Output
predicted= nb.predict(X)
print("Naive Bayes accuracy score:  ")
print(accuracy_score(Y, predicted))
'''print ("Confusion Matrix")
print(confusion_matrix(Y2, predicted))
print ("Classifcation Report")
print(classification_report(Y2, predicted))
'''