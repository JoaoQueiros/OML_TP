import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot

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

def calc_b(idx, alfas_idx, y_idx, x_idx):

    b = 0
    z= 0
    l = len(idx)
    for i in range(l):
        aux = 0
        for n in range(l):
            aux += y_idx[n] * alfas_idx[n] * x_idx[n].T * x_idx[i]
        b += y_idx[i] - aux

    b = b / l
    return b

def calc_accuracy(y, predictions):

    right = 0
    samples = len(y)

    for i in range(samples):
        if y[i] == predictions[i]:
            right += 1
    
    return right / samples

def plot(x, y, xT, yT, b, alfas):

    x0 = []; x1 = []; y0 = []; y1 = []
    xS0 = []; xS1 = []; yS0 = []; yS1 = []



    for i in range(len(y)):
        if y[i] == 1:
            x1.append(x[i,0])
            y1.append(x[i,1])
        else:
            x0.append(x[i,0])
            y0.append(x[i,1])

    for i in range(len(yT)):
        if yT[i] == 1:
            xS1.append(xT[i,0])
            yS1.append(xT[i,1])
        else:
            xS0.append(xT[i,0])
            yS0.append(xT[i,1])

    

    plot = pyplot.figure()
    ax=plot.add_axes([0,0,1,1])
    ax=plot.add_subplot(1,1,1)
    ax.scatter(x0, y0, marker='o', s=20, c='orange')
    ax.scatter(x1, y1, marker='o', s=20, c='green')
    ax.scatter(xS0, yS0, marker='o',facecolors='none', edgecolors='blue', s=80)
    ax.scatter(xS1, yS1, marker='o',facecolors='none', edgecolors='blue', s=80)
    pyplot.xlabel('x1')
    pyplot.ylabel('x2')

    x_axis = np.linspace(-100.,100.)

    #fig,ax = plot.subplots()
    #ax.plot(x_axis,np.multiply(y, alfas).T * K(x,x[i]) + b)

    #ax.set_xlim((-100.,100.))
    #ax.set_ylim((-100.,100.))

    #y_axis = 2 * x + 1 #* x_axis
    
    #ax.plot(x_axis,y_axis)
    ax.axis([min(x[:,0])-0.2,max(x[:,0]) +0.2,min(x[:,1])-0.2,max(x[:,1]) +0.2])
    pyplot.show()



def svm():
    x, y, x_test, y_test = df_import('ex1data1.csv')
    #x, y, x_test, y_test = df_import('ex1data2.csv')
    c = 10
    tol = 1 * 10^-4
    max_passes = 2

    alfas = SMO(c, tol, max_passes, x, y).T

    #x_alfa
    #y_alfa

    idx = []
    alfas_idx = []
    x_idx = []
    y_idx = []


    for i in range(len(alfas[0])):
        if alfas[0][i] > 0:
            idx.append(i)
            alfas_idx.append(alfas[0][i])
            y_idx.append(int(y[i][0]))
            x_idx.append(x[i])

    #print('Índices:', idx)
    #print('Alfas:', alfas_idx)
    #print('y:', y_idx)
    #print('x:' , x_idx)

    print('Indice:               Alfa:               X:                    Y:')
    for i in range(len(alfas_idx)):
        print(idx[i],'                   ', alfas_idx[i], '                 ', x_idx[i][0,0], x_idx[i][0,1], '    ', y_idx[i])


    b = calc_b(idx, alfas_idx, y_idx, x_idx)

    size = len(x_test)
    reshape_size = len(x_idx)
    predictions = []
    for i in range(size):
        res = np.multiply(y_idx, alfas_idx).T * (x_idx * x_test[i].T).reshape(reshape_size, 1)
        if res > 0:
            predictions.append(1)
        else:
            predictions.append(-1)

    accuracy = calc_accuracy(y_test, predictions)

    print()
    print('Accuracy:', accuracy)

    plot(x,y,x_idx,y_idx, b, alfas_idx)

    return 0

#def K(X, x):
#    return 0


#####################################################################
def formula_2(alfa, b, x, y, i):

    res = np.multiply(y, alfa).T * (x * x[i].T) + b

    return res

#####################################################################
def formula_10(alfaI, alfaJ, C):

    L = max(0, alfaJ - alfaI)
    H = min(C, C + alfaJ - alfaI)

    return L, H

#####################################################################
def formula_11(alfaI, alfaJ, C):

    L = max(0, alfaI + alfaJ - C)
    H = min(C, alfaI + alfaJ)

    return L, H

#####################################################################
def formula_12(aJ, Ei, Ej, N, yJ):

    res = aJ - ((yJ * ( Ei - Ej)) / N)
    
    return res

#####################################################################
def formula_14(x, i, j):

    res = 2 * (x[j] * x[i].T) - (x[i] * x[i].T) - (x[j] * x[j].T)

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
def formula_16(alfaI, alfaJ, alfaJ_old, yI, yJ):

    res = alfaI + yI * yJ * (alfaJ_old - alfaJ)

    return res

#####################################################################
def formula_17(b, Ei, alfa, x, y, aI_old, aJ_old, i, j):

    res = b - Ei - y[i] * (alfa[i] - aI_old) * (x[i] * x[i].T) - y[j] * (alfa[j] - aJ_old) * (x[j] * x[i].T)

    return res

#####################################################################
def formula_18(b, Ej, alfa, x, y, aI_old, aJ_old, i, j):

    res = b - Ej - y[i] * (alfa[i] - aI_old) * (x[j] * x[i].T) - y[j] * (alfa[j] - aJ_old) * (x[j] * x[j].T)

    return res

#####################################################################
def formula_19(b1, b2, aI, aJ, C):

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
    max_it = 5
    it = 0

    while(passes < max_passes and it < max_it): #Ciclo While exterior
        num_changed_alfas = 0
        it += 1
        for i in range(1, m):

            # CALCULAR Ei -> FORMULA 2
            fX = formula_2(alfa, b, x, y, i)
            Ei = fX - y[i]
            if (y[i] * Ei < -tol and alfa[i] < c) or (y[i] * Ei > tol and alfa[i] > 0):

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
                alfa[i] = formula_16(alfa[i], alfa[j], aJ_old, y[i], y[j])

                #CALCULAR B1 -> FÓRMULA 17
                b1 = formula_17(b, Ei, alfa, x, y, aI_old, aJ_old, i, j)

                #CALCULAR B2 -> FÓRMULA 18
                b2 = formula_18(b, Ej, alfa, x, y, aI_old, aJ_old, i, j)

                #CALCULAR B -> FÓRMULA 19
                b = formula_19(b1, b2, alfa[i], alfa[j], c)

                num_changed_alfas += 1

        #print(num_changed_alfas)
        if num_changed_alfas == 0:
            passes += 1
        else:
            passses = 0

    return alfa

#sigma=1
gm = 1/2
svm()
