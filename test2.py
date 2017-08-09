# Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

filename = 'poker-hand-training-true.data.csv'
names = ['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5','class']
dataset = pandas.read_csv(filename,skipinitialspace=True,names=names)

#print(dataset)

# shape
#print(dataset.shape)

# head
#print(dataset.head(20))

# descriptions
#print(dataset.describe())

# class distribution
#print(dataset.groupby('class').size())

# Split-out validation dataset
array = dataset.values
X = array[:,0:10]
Y = array[:,10]
#print(array)
#print(X)
#print(Y)


'''validation_size = 0.20
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
'''

'''
# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X,Y)
# a=8523
predictions = knn.predict([X[5]])
print("This is knn Predicted: ")
print(predictions)
#print(accuracy_score(Y, predictions))


print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

nb = GaussianNB()

# Train the model using the training sets
nb.fit(X,Y)

#Predict Output
predicted= nb.predict(X)

#print predicted

print(accuracy_score(Y, predicted))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
'''
# Make predictions on validation dataset
#---------KNN-----------#
knn = KNeighborsClassifier()
knn.fit(X,Y)
predictions = knn.predict(X)
print(accuracy_score(Y, predictions))


#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))

#----------------NAIVE BAYES------------#
nb = GaussianNB()

# Train the model using the training sets
nb.fit(X,Y)

#Predict Output
predicted= nb.predict(X)

print(accuracy_score(Y, predicted))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))
