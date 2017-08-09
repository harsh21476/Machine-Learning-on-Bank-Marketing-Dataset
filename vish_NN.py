from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
import my_ml_lib

X, Y = my_ml_lib.dataset_to_xy('bank-full.csv')
col = [1, 2, 3, 4, 6, 7, 8, 9, 10, 15]
my_ml_lib.my_label_encoder(X, col)
# X_new = X
X_new,mirror = my_ml_lib.my_OHEnc(X,col)
X_new = StandardScaler().fit_transform(X_new)
# X_new=normalize(X_new)
full_dataset = np.append(X_new.T, Y.T)
X_new = np.array(X_new)
Y_new = np.array(Y)

kf = KFold(n_splits=10)
kf.get_n_splits(X_new)
NN_accucray_array = []
nb_accuracy_array = []
print('The Shape of new X: ', X_new.shape)
for train_index, test_index in kf.split(X_new):
    # knn = KNeighborsClassifier(n_neighbors=10)
    # knn.fit(X_new[train_index], Y[train_index])
    # predictions = knn.predict(X_new[test_index])
    nnclf = MLPClassifier(activation='logistic',solver='sgd',
                          hidden_layer_sizes=(40,),max_iter=150,
                          learning_rate='adaptive',learning_rate_init=0.25)

    nnclf.fit(X_new, Y)
    predictions = nnclf.predict(X_new[test_index])
    print("NN accuracy score:  ")
    print(nnclf.n_outputs_)
    a = accuracy_score(Y[test_index], predictions)
    # print(a)
    # print('This is Actual: ',Y[test_index],'\nThis is predicted: ',predictions)
    confus_mat_res = confusion_matrix(Y[test_index], predictions)
    print(confus_mat_res)
    NN_accucray_array.append(a)
    print(NN_accucray_array)

    # nb = GaussianNB()
    # nb.fit(X_new[train_index], Y[train_index])
    # predicted = nb.predict(X_new[test_index])
    # print("Naive Bayes accuracy score:  ")
    # a = accuracy_score(Y[test_index], predicted)
    # print(a)
    # confus_mat_res = confusion_matrix(Y[test_index], predictions)
    # print(confus_mat_res)
    # nb_accuracy_array.append(a)
    # print(nb_accuracy_array)

print("NN mean accuray:  ")
print (np.mean(NN_accucray_array))
# print (1-np.mean(knn_accucray_array))
# print("nb mean accuray:  ")
# print(np.mean(nb_accuracy_array))
# print (1-np.mean(nb_accuracy_array))

''' OUTPUTS:
* For solver='lbfgs',alpha=1, hidden_layer_sizes=(25,),
                          random_state=1 -> 0.922318677759
* For solver='lbfgs',alpha=1e-5, hidden_layer_sizes=(25,),
                          random_state=1 -> 0.922097448626
* For solver='lbfgs',alpha=0.5, hidden_layer_sizes=(25,),
                          random_state=1 -> 0.923402513661
* For activation='logistic',solver='sgd', hidden_layer_sizes=(25,),
                          learning_rate='adaptive',learning_rate_init=0.5 -> 0.92338035553
* For activation='logistic',solver='sgd', hidden_layer_sizes=(25,),
                          learning_rate='adaptive' -> 0.903252153374
* For activation='logistic',solver='sgd', hidden_layer_sizes=(25,) -> 0.903362758158
* For activation='logistic',solver='sgd', hidden_layer_sizes=(25,),
                            learning_rate_init=0.5 -> 0.923690021533
* For activation='logistic',solver='sgd', hidden_layer_sizes=(13,2),
                          learning_rate_init=0.5 -> 0.912033401353
* For activation='logistic',solver='sgd', hidden_layer_sizes=(40,),
                          learning_rate_init=0.5 -> 0.949458500265
* For activation='logistic',solver='sgd', hidden_layer_sizes=(40,),max_iter=300,
                          learning_rate_init=0.25 -> 0.949657561484
* For activation='logistic',solver='sgd', hidden_layer_sizes=(40,),max_iter=500,
                          learning_rate_init=0.25 -> 0.948794905802
* For activation='logistic',solver='sgd', hidden_layer_sizes=(40,),max_iter=100,
                          learning_rate_init=0.25 -> 0.95149346296
* For activation='logistic',solver='sgd', hidden_layer_sizes=(40,),max_iter=50,
                          learning_rate_init=0.25 -> 0.92654336767
* For activation='logistic',solver='sgd',
                          hidden_layer_sizes=(40,),max_iter=50,
                          learning_rate='invscaling',learning_rate_init=0.5 -> 0.945477085117
* For activation='logistic',solver='sgd',
                          hidden_layer_sizes=(40,),max_iter=100,
                          learning_rate='adaptive',learning_rate_init=0.5 -> 0.956514412422
* For activation='logistic',solver='sgd',
                          hidden_layer_sizes=(40,),max_iter=100,
                          learning_rate='adaptive',learning_rate_init=0.25 -> 0.952842697516
* For activation='logistic',solver='sgd',
                          hidden_layer_sizes=(40,),max_iter=150,
                          learning_rate='adaptive',learning_rate_init=0.25 -> 0.957531954912 ========== MAX TILL NOW
* For activation='logistic',solver='sgd',
                          hidden_layer_sizes=(40,),max_iter=200,
                          learning_rate='adaptive',learning_rate_init=0.25 -> 0.9569346881

Without One Hot key Encoding:

* For activation='logistic',solver='sgd', hidden_layer_sizes=(8,),max_iter=100,
                          learning_rate_init=0.5 -> 0.901416266573
* For activation='logistic',solver='sgd', hidden_layer_sizes=(8,),
                          learning_rate_init=0.5 -> 0.902013499145
* For activation='logistic',solver='sgd',
                          hidden_layer_sizes=(8,),max_iter=200,
                          learning_rate='adaptive',learning_rate_init=0.25 -> 0.901681753273
* For activation='logistic',solver='sgd',
                          hidden_layer_sizes=(8,),max_iter=150,
                          learning_rate='adaptive',learning_rate_init=0.25 -> 0.902101945797 ==== almost MAX
* For activation='logistic',solver='sgd',
                          hidden_layer_sizes=(8,),max_iter=200,
                          learning_rate='adaptive',learning_rate_init=0.5 -> 0.902610697476 ==== MAX
* For activation='logistic',solver='sgd',
                          hidden_layer_sizes=(8,),max_iter=100,
                          learning_rate='adaptive',learning_rate_init=0.5 -> 0.900973920809
LINEAR:

* For activation='identity',solver='sgd',
                          hidden_layer_sizes=(40,),max_iter=100,
                          learning_rate='adaptive',learning_rate_init=0.5 -> 0.902610702368
* For activation='identity',solver='sgd',
                          hidden_layer_sizes=(40,),max_iter=150,
                          learning_rate='adaptive',learning_rate_init=0.25 -> 0.90285401137
WITHOUT ONE-HOTKEY ENCODING:
* For activation='identity',solver='sgd',
                          hidden_layer_sizes=(8,),max_iter=150,
                          learning_rate='adaptive',learning_rate_init=0.25 -> 0.891064589144

                          p
'''
