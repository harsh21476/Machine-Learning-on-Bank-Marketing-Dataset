# Recursive Feature Elimination
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random

# load the iris datasets
filename = 'bank_encoded.csv'
names = ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y"]
dataset = pandas.read_csv(filename,skipinitialspace=True,names=names)
print(dataset.describe())

array = dataset.values
X = array[:,0:16]
Y = array[:,16]
print Y[1:10]

#print (np.array(X).shape)

kf = KFold(n_splits=10)
kf.get_n_splits(X)
knn_accucray_array =[]
nb_accuracy_array=[]
for train_index, test_index in kf.split(X):
    print ("train and test")
    print train_index
    print test_index
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X[train_index], Y[train_index])
    predictions = knn.predict(X[test_index])
    print("KNN accuracy score below:  ")
    a = accuracy_score(Y[test_index], predictions)
    print a
    knn_accucray_array.append(a)
    print(confusion_matrix(Y[test_index], predictions))
    # print(precision_score(Y[test_index], predictions))
    # false_positive_rate, true_positive_rate, thresholds = roc_curve(Y[test_index], predictions)
    #

    # roc_auc = auc(false_positive_rate, true_positive_rate)
    # print "fpr tpr",false_positive_rate,true_positive_rate
    # plt.plot(false_positive_rate, true_positive_rate, 'b',
    #          label='AUC = %0.2f' % roc_auc)
    # plt.show()
    # #print knn_accucray_array
    fpr, tpr, thresholds = roc_curve(Y[test_index],
                                     predictions,
                                     pos_label=1)
    roc_auc = auc(fpr, tpr)
    # roc_auc = auc(Y[test_index],predictions)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.5f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # nb = GaussianNB()
    # nb.fit(X[train_index], Y[train_index])
    # predicted = nb.predict(X[test_index])
    # print("Naive Bayes accuracy score below:  ")
    # a = accuracy_score(Y[test_index], predicted)
    # print a
    # print(confusion_matrix(Y[test_index], predicted))
    # nb_accuracy_array.append(a)
    # #print nb_accuracy_array
    # print ("/n/n/n/n")
    # fpr, tpr, thresholds = roc_curve(Y[test_index],
    #                                  predicted,
    #                                  pos_label=1)
    # roc_auc = auc(fpr, tpr)
    # # roc_auc = auc(Y[test_index],predictions)
    # plt.plot(fpr, tpr, 'b', label='AUC = %0.5f' % roc_auc)
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()

print ("Knn mean accuray:  ")
print (np.mean(knn_accucray_array))
#print (1-np.mean(knn_accucray_array))
print ("nb mean accuray:  ")
print (np.mean(nb_accuracy_array))
#print (1-np.mean(nb_accuracy_array))

'''
# Make predictions on validation dataset
#---------KNN-----------#
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_new,Y)
predictions = knn.predict(X_new)

print("KNN accuracy score:  ")
print(accuracy_score(Y, predictions))
print ("Confusion Matrix")
print(confusion_matrix(Y2, predictions))
print ("Classifcation Report")
print(classification_report(Y2, predictions))

#----------------NAIVE BAYES------------#
nb = GaussianNB()

# Train the model using the training sets
nb.fit(X_new,Y)

#Predict Output
predicted= nb.predict(X_new)
print("Naive Bayes accuracy score:  ")
print(accuracy_score(Y, predicted))
print ("Confusion Matrix")
print(confusion_matrix(Y2, predicted))
print ("Classifcation Report")
print(classification_report(Y2, predicted))
'''