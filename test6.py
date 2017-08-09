#Load libraries
import pandas
import numpy as np
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
from sklearn import preprocessing
from sklearn.preprocessing import normalize

filename = 'bank-additional.csv'
#names = ['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5','class']
dataset = pandas.read_csv(filename)#,skipinitialspace=True,names=names)
'''
print(dataset)

print ("Raw Training Data Shape")
# shape
print(dataset.shape)

# head
#print(dataset.head(20))
print ("Description")
# descriptions
print(dataset.describe())

'''
array = dataset.values
X = array[:,0:20]
Y = array[:,20]

#print(array)
#print(X)
#print(Y)

le = preprocessing.LabelEncoder()

#print le.fit_transform(X[:,1])
#print le.fit_transform(Y)
x0 = np.array(X[:,0])
x1 = np.array(le.fit_transform(X[:,1]))
x2 = np.array(le.fit_transform(X[:,2]))
x3 = np.array(le.fit_transform(X[:,3]))
x4 = np.array(le.fit_transform(X[:,4]))
x5 = np.array(X[:,5])
x6 = np.array(le.fit_transform(X[:,6]))
x7 = np.array(le.fit_transform(X[:,7]))
x8 = np.array(le.fit_transform(X[:,8]))
x9 = np.array(X[:,10])
x10 = np.array(le.fit_transform(X[:,10]))
x11 = np.array(X[:,11])
x12 = np.array(X[:,12])
x13 = np.array(X[:,13])
x14 = np.array(X[:,14])
x15 = np.array(le.fit_transform(X[:,15]))
y_out= le.fit_transform(Y)


matrix=[]
le.fit(X[:,1])
matrix.append([1,le.classes_])
le.fit(X[:,2])
matrix.append([2,le.classes_])
le.fit(X[:,3])
matrix.append([3,le.classes_])
le.fit(X[:,4])
matrix.append([4,le.classes_])
le.fit(X[:,6])
matrix.append([6,le.classes_])
le.fit(X[:,7])
matrix.append([7,le.classes_])
le.fit(X[:,8])
matrix.append([8,le.classes_])
le.fit(X[:,10])
matrix.append([10,le.classes_])
le.fit(X[:,15])
matrix.append([15,le.classes_])


np.savetxt('class_refernce.txt', matrix, delimiter=",", fmt="%s")

con = np.concatenate([[x0],[x1],[x2],[x3],[x4],[x5],[x6],[x7],[x8],[x9],[x10],[x11],[x12],[x13],[x14],[x15],[y_out]])

#print con

#min_max_scaler = preprocessing.MinMaxScaler()
#X_train_minmax = min_max_scaler.fit_transform(con.T)

#The same instance of the transformer can then be applied to some new test data unseen during the fit call: the same scaling and shifting operations will be applied to be consistent with the transformation performed on the train data
#X_test_minmax = min_max_scaler.transform(testarray_to_be_scaled.T)

#MaxAbsScaler works in a very similar fashion, but scales in a way that the training data lies within the range [-1, 1] by dividing through the largest maximum value in each feature. It is meant for data that is already centered at zero or sparse data.

np.savetxt("bank_encoded.csv",con.T, delimiter=",",fmt='%.4f')

#print ("Raw Testing Encoded Data Shape")
# shape
#con2 = pandas.DataFrame(con.transpose())
#print(con2.shape)

# head
#print(dataset.head(20))
#print ("Description")
# descriptions

#abc = con2.describe()
#print(abc.to_string())
