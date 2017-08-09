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
from collections import Counter
from random import shuffle

def probability_finder(value,array):
    count = 0
    for i in range(0,len(array)):
        if(value==array[i]):
            count = count+1
    return count.__float__()/len(array).__float__()

#def encoder_finder(value)


def array_encoder(inputarr):
    def referencer(value,reference_array,encoded_classes):

        for i in range(len(reference_array)):
            if value==encoded_classes[i]:
                return reference_array[i]

    le.fit_transform(inputarr)

    reference_array=[]
    encoded_array=[]
    for i in range(len(le.classes_)):
        reference_array.append(probability_finder(le.classes_[i],inputarr))

    for i in range(len(inputarr)):
        encoded_array.append(referencer(inputarr[i],reference_array,le.classes_))

    return encoded_array,reference_array


filename = 'bank-full.csv'
#names = ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y"]
dataset = pandas.read_csv(filename,delimiter=';')
#filename2 = 'bank_modified.csv'
#names = ["age","job","marital","education","default","housing","loan","contact","month","day_of_week","duration","campaign","pdays","previous","poutcome","emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed","y"]
#dataset2 = pandas.read_csv(filename2,skipinitialspace=True,names=names)


#print(dataset)

# shape
#print ("Raw Testing Data Shape")
#print(dataset.shape)
#print ("Training Data Shape")
#print (dataset2.shape)
# head
#print(dataset.head(20))
# descriptions
#print ("Testing Data Description")
#print(dataset.describe())

array = dataset.values
print array
np.random.shuffle(array)
print array
X = array[:,0:16]
Y = array[:,16]
#for i in range(17):
 #   print (Counter(array[:,i]))
#array2 = dataset2.values
#X2 = array2[:,0:20]
#Y2 = array2[:,20]

#array22 = np.array(array2)
#np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
matrix=[]
le = preprocessing.LabelEncoder()

x0 = np.array(X[:,0])
x1,ref_array = array_encoder(X[:,1])
print len(x1)
matrix.append([1,le.classes_,ref_array])
x2,ref_array = array_encoder(X[:,2])
matrix.append([2,le.classes_,ref_array])
x3,ref_array = array_encoder(X[:,3])
matrix.append([3,le.classes_,ref_array])
x4,ref_array = array_encoder(X[:,4])
matrix.append([4,le.classes_,ref_array])
print len(x4)
x5 = np.array(X[:,5])
print len(x5)
x6,ref_array = array_encoder(X[:,6])
matrix.append([6,le.classes_,ref_array])
x7,ref_array = array_encoder(X[:,7])
matrix.append([7,le.classes_,ref_array])
x8,ref_array = array_encoder(X[:,8])
matrix.append([8,le.classes_,ref_array])
x9 = np.array(X[:,9])
x10,ref_array = array_encoder(X[:,10])
matrix.append([10,le.classes_,ref_array])
x11 = np.array(X[:,11])
x12 = np.array(X[:,12])
x13 = np.array(X[:,13])
x14 = np.array(X[:,14])
x15,ref_array = array_encoder(X[:,15])
matrix.append([15,le.classes_,ref_array])
y_out= le.fit_transform(Y)
print len(y_out)

np.savetxt('class_refernce_full.txt', matrix, delimiter=",", fmt="%s")

con = np.concatenate([[x0],[x1],[x2],[x3],[x4],[x5],[x6],[x7],[x8],[x9],[x10],[x11],[x12],[x13],[x14],[x15],[y_out]])


max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(con.T)
#X_train_maxabs2 = max_abs_scaler.transform(array22)

np.savetxt("bank_encoded.csv",(con.T), delimiter=",",fmt='%.4f')
np.savetxt("bank_encoded_scaled.csv",X_train_maxabs, delimiter=",",fmt='%.4f')
#np.savetxt("bank_modified_scaled.csv",X_train_maxabs2, delimiter=",",fmt='%.4f')

#print ("Raw Training Encoded Data Shape")
# shape
#con2 = pandas.DataFrame(X_train_maxabs2)
#print(con2.shape)

# head
#print(dataset.head(20))
#print ("Description")
# descriptions

#abc = con2.describe()
#print(abc.to_string())
import qwertyuiop