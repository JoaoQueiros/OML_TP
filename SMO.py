import random
import numpy as np
import pandas as pd

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

    return np.mat(X_train), np.mat(y_train)#, np.mat(X_test), np.mat(y_test)


def svm():
    x, y = df_import('ex2data1.csv')
    c = 1.2
    tol = 1 * 10^-4
    max_passes = 5

    SMO(c, tol, max_passes, x, y)

    return 0


#####################################################################
def formula_2(alfa, b, x, y, i):

    res = np.multiply(y, alfa).T * x * x[i].T + b

    return res

#####################################################################
def formula_10(alfaI, alfaJ, C):

    L = max(0, alfaI - alfaJ)
    H = min(C, C + alfaJ - alfaI)

    return L, H

#####################################################################
def formula_11(alfaI, alfaJ, C):

    L = max(0, alfaI + alfaJ - C)
    H = min(C, alfaI + alfaJ)

    return L, H

#####################################################################
def formula_12(aJ, Ei, Ej, N, yJ):

    aJ = aJ - ((yJ * ( Ei - Ej)) / N)
    
    return aJ

#####################################################################
def formula_14(x, i, j):

    res = 2 * (x[j].T * x[i].T) - (x[i].T * x[i].T) - (x[j].T * x[j].T)

    return res

#####################################################################
def formula_15(aJ, H, L):
    if aJ > H:
        res = H
    elif L <= aJ <= H:
        res = aJ
    elif aJ < L:
        res = L

    return res

#####################################################################
def formula_16():
    return res

#####################################################################
def formula_17():
    return res

#####################################################################
def formula_18():
    return res

#####################################################################
def formula_19(b1, b2, aI, aJ, c):

    if 0 < aI < C:
        b = b1
    elif 0 < aJ < C:
        b = b2
    else:
        b = (b1 / b2) / 2

    return b

def randomIndex(i, m):
    j = i
    while(j == i):
        j = int (random.uniform(0, m))
    return j


def SMO(c, tol, max_passes, x, y):
    m = len(x)
    alfa = pd.DataFrame(0, index=np.arange(m), columns=['a']).values

    b = 0
    passes = 0

    while(passes < max_passes): #Ciclo While exterior
        num_changed_alfas = 0
        for i in range(1, m):

            # CALCULAR Ei -> FORMULA 2
            fX = formula_2(alfa, b, x, y, i)
            Ei = fX - y[i]
            if 1: #AQUELE IF ENORME

                #SELECIONAR RANDOMLY
                j = randomIndex(i, m)

                #CALCULAR Ej -> FORMULA 2
                fX = formula_2(alfa, b, x, y, j)
                Ej = fX - y[j]

                #GUARDAR ALFAS ANTIGOS
                aI_old = alfa[i] 
                aJ_old = alfa[j]

                #CALCULAR L E H -> FORMULA 10 OU 11
                if y[i] != y[j]: 
                    L, H = formula_10(alfa[i], alfa[j], c) 
                else:
                    L, H = formula_11(alfa[i], alfa[j], c)
                
                #PASSAR PARA A PRÓXIMA ITERAÇÃO
                if(L == H): continue
                
                #CALCULAR N -> FORMULA 14
                N = formula_14(x, i, j)

                #PASSAR PARA A PRÓXIMA ITERAÇÃO
                if(N > 0): continue

                #CALCULAR O NOVO Aj -> FORMULA 12 E 15
                alfa[j] = formula_12(alfa[j], Ei, Ej, N, y[j])
                alfa[j] = formula_15(alfa[j], H, L)

                #PASSAR PARA A PRÓXIMA ITERAÇÃO
                if abs(alfa[j] - aJ_old) < 1 * 10^(-5):
                    continue

                #CALCULAR Ai -> FÓRMULA 16
                alfa[i] = formula_16()

                #CALCULAR B1 -> FÓRMULA 17
                b1 = formula_17()

                #CALCULAR B2 -> FÓRMULA 18
                b2 = formula_18()

                #CALCULAR B -> FÓRMULA 19
                b = formula_19(b1, b2, alfa[i], alfa[j], c)

                num_changed_alfas += 1

        if num_changed_alfas == 0:
            passes += 1
        else:
            passses = 0

    return alfa, b



svm()