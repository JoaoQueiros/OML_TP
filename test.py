import pandas as pd
import numpy as np


np.random.seed(10)

def df_import(dataset):
    # import data 
    df = pd.read_csv(dataset, header = None)
    # use random search 0.8
    msk = np.random.rand(len(df)) < 0.8
    # train
    train = df[msk]
    X_train = train.iloc[:,0:2].values
    y_train = train.iloc[:,2:3].values
    # test
    test = df[~msk]
    X_test = test.iloc[:,0:2].values
    y_test = test.iloc[:,2:3].values

    return np.mat(X_train), np.mat(y_train), np.mat(X_test), np.mat(y_test)

def formula_14(x, i, j):

    res = 2 * (x[j] * x[i].T) - (x[i] * x[i].T) - (x[j] * x[j].T)
    #res = (x[i] * x[i].T)
    return res


X_train, y_train, X_test, y_test = df_import("ex2data1.csv")
m = len(X_train)
alpha = pd.DataFrame(0, index=np.arange(m), columns=['a']).values
b = 0
alpha = np.mat(alpha)
#res = formula_2(alpha, b, X_train, y_train)
i = 1
j = 2
res = formula_14(X_train, i, j)
print(res)