import numpy as np
import pandas as pd
from sklearn import preprocessing


def dataset_to_xy(location):
    dataset = pd.read_csv(location,delimiter=';')
    dataset=dataset.iloc[np.random.permutation(len(dataset))]
    ncols=len(dataset.columns)
    index = [("x" + str(i)) for i in range(ncols-1)]
    index.append('y')
    dataset.rename(columns=dict(zip(dataset, index)), inplace=True)
    x = dataset.iloc[:, 0:(ncols-1)]
    y = dataset.iloc[:, (ncols-1)]
    return x,y

def my_label_encoder(dataset,columns):
    enc_mirrors=[preprocessing.LabelEncoder() for i in range(0,len(dataset.columns))]
    for i in range(0,len(columns)):
        enc_mirrors[i]=preprocessing.LabelEncoder()
    for i in range(0,len(columns)):
        dataset.iloc[:, columns[i]]=enc_mirrors[columns[i]].fit_transform(dataset.iloc[:,columns[i]])
    return enc_mirrors

def my_OHEnc(dataset,columns):
    new_dataset=dataset.T
    ONHEnc_mirrors=[preprocessing.OneHotEncoder() for i in range(0,len(dataset.columns))]
    for i in range(0,len(columns)):
        pd_array=dataset.iloc[:,columns[i]].as_matrix().reshape(1,len(dataset.index))
        np_array=np.reshape(np.array(pd_array),(len(dataset.index),1))
        temp=ONHEnc_mirrors[columns[i]].fit_transform(np_array)
        temp_df=pd.DataFrame(temp.toarray())
        new_dataset=new_dataset.append(temp_df.T)
    new_dataset=new_dataset.T
    new_dataset.drop(new_dataset.columns[columns],axis=1,inplace=True)
    return new_dataset,ONHEnc_mirrors