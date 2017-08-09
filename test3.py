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
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

filename = 'bank_encoded.csv'
#names = ["age","job","marital","education","default","housing","loan","contact","month","day_of_week","duration","campaign","pdays","previous","poutcome","emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed","y"]
dataset = pandas.read_csv(filename)#,skipinitialspace=True,names=names)

#filename = 'bank_modified.csv'
#names = ["age","job","marital","education","default","housing","loan","contact","month","day_of_week","duration","campaign","pdays","previous","poutcome","emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed","y"]
#dataset2 = pandas.read_csv(filename,skipinitialspace=True,names=names)

train,test = train_test_split(dataset,test_size=0.2)

np.savetxt("traindata_encoded.csv",train, delimiter=",",fmt='%.4f')
np.savetxt("testdata_encoded.csv",test, delimiter=",",fmt='%.4f')
'''
con2 = pandas.DataFrame(train)
print(con2.shape)

# head
#print(dataset.head(20))
print ("Description")
# descriptions

abc = con2.describe()
print(abc.to_string())
'''