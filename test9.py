

# Recursive Feature Elimination
import numpy as np

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import my_ml_lib

X,Y = my_ml_lib.dataset_to_xy('bank-full.csv')
col = [1,2,3,4,6,7,8,9,10,15]
y_encoder= preprocessing.LabelEncoder()
Y=y_encoder.fit_transform(Y)
my_ml_lib.my_label_encoder(X,col)

X_new,mirror = my_ml_lib.my_OHEnc(X,col)
X_new=preprocessing.normalize(X_new)
full_dataset=np.append(X_new.T,Y.T)
print 'This is: ',full_dataset.shape
X_new=np.array(X_new)
Y=np.array(Y)
print 'This is first: ',X_new.shape, Y.shape

#load the iris datasets
# filename = 'bank_encoded.csv'
# names = ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y"]
# dataset = pandas.read_csv(filename,skipinitialspace=True,names=names)
#
# array = dataset.values
# X = array[:,0:16]
# Y = array[:,16]
# print "xy"
# print X.shape
# print Y.shape

kf = KFold(n_splits=10)
kf.get_n_splits(X)
knn_accucray_array =[]
nb_accuracy_array=[]
for train_index, test_index in kf.split(X):
    print ("train and test")
    print train_index
    print test_index
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_new[train_index], Y[train_index])
    predictions = knn.predict(X_new[test_index])
    print("KNN accuracy score:  ")
    a = accuracy_score(Y[test_index], predictions)
    #print a
    print confusion_matrix(Y[test_index], predictions)
    knn_accucray_array.append(a)
    print knn_accucray_array
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

    nb = GaussianNB()
    nb.fit(X_new[train_index], Y[train_index])
    predicted = nb.predict(X_new[test_index])
    print("Naive Bayes accuracy score:  ")
    a = accuracy_score(Y[test_index], predicted)
    #print a
    print confusion_matrix(Y[test_index], predicted)
    nb_accuracy_array.append(a)
    print nb_accuracy_array

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