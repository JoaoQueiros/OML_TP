import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# l = linear
# g = gaussiana
# p = polinomial
k_function = 'g'

#sigma=1
gm = 1/2

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

#####################################################################
def calc_b(idx, alfas_idx, y_idx, x_idx):

    b = 0
    z= 0
    l = len(idx)
    for i in range(l):
        aux = 0
        for n in range(l):
            aux += y_idx[n] * alfas_idx[n] * K(x_idx[i], x_idx[n])
        b += y_idx[i] - aux

    b = b / l
    return b

#####################################################################
def calc_accuracy(y, predictions):

    right = 0
    samples = len(y)

    for i in range(samples):
        if y[i] == predictions[i]:
            right += 1
    
    return right / samples

#####################################################################
def calc_grid(b, alfas , yT, xT, grid):

    h = len(grid[0])
    w = len(grid[0][0])
    Yp = [[0 for x in range(w)] for y in range(h)]

    for i in range(h):
        for j in range(w):
            Yp[i][j] = int(np.sign((np.multiply(yT,alfas).T * K(xT, np.matrix([grid[0][i][j],grid[1][i][j]])) + b).item()))

    return Yp

#####################################################################
def plot(x, y, xT, yT, b, alfas, x_test, y_test):

    x0 = []; x1 = []; y0 = []; y1 = []
    xS0 = []; xS1 = []; yS0 = []; yS1 = []
    xT0 = []; xT1 = []; yT0 = []; yT1 = []

    #Scatter dos dados de treino
    for i in range(len(y)):
        if y[i] == 1:
            x1.append(x[i,0])
            y1.append(x[i,1])
        else:
            x0.append(x[i,0])
            y0.append(x[i,1])

    #Scatter dos dados de suporte
    for i in range(len(yT)):
        if yT[i] == 1:
            xS1.append(xT[i,0])
            yS1.append(xT[i,1])
        else:
            xS0.append(xT[i,0])
            yS0.append(xT[i,1])

    #Scatter dos dados de validação
    for i in range(len(y_test)):
        if y_test[i] == 1:
            xT1.append(x_test[i,0])
            yT1.append(x_test[i,1])
        else:
            xT0.append(x_test[i,0])
            yT0.append(x_test[i,1])

    minim = min(min(x[:,0]),min(x[:,1])) -0.2
    maxim = max(max(x[:,0]),max(x[:,1])) +0.2

    x_axis = np.arange(minim,maxim, 0.1)
    y_axis = np.arange(minim,maxim, 0.1)
    xx,yy = np.meshgrid(x_axis, y_axis)
    grid = [xx,yy]

    Yp= calc_grid(b, alfas , yT, xT, grid)
    plt.contour(xx,yy,Yp, levels = [0,10], colors='black')

    plt.scatter(x0, y0, marker='o', s=20, c='orange', label = 'Classe -1, Treino')
    plt.scatter(xT0, yT0, marker='o',facecolors='none', edgecolors='orange', s=20, label = 'Classe -1, Validação')
    plt.scatter(x1, y1, marker='o', s=20, c='green', label = 'Classe 1, Treino')
    plt.scatter(xT1, yT1, marker='o',facecolors='none', edgecolors='green', s=20, label = 'Classe 1, Validação')
    plt.scatter(xS0, yS0, marker='o',facecolors='none', edgecolors='blue', s=80, label='Alfa de Suporte')
    plt.scatter(xS1, yS1, marker='o',facecolors='none', edgecolors='blue', s=80)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()


    plt.axis([min(x[:,0])-0.2,max(x[:,0]) +0.2,min(x[:,1])-0.2,max(x[:,1]) +0.2])
    plt.show()



def svm():
    
    #x, y, x_test, y_test = df_import('ex1data1.csv') #Dataset Linear 1
    #x, y, x_test, y_test = df_import('ex1data2.csv') #Dataset Linear 2
    x, y, x_test, y_test = df_import('ex2data1.csv') #Dataset Gaussiano
    #x, y, x_test, y_test = df_import('ex2data2.csv') #Dataset Polinomial

    c = 10
    tol = 1 * 10^-4
    max_passes = 2

    alfas = SMO(c, tol, max_passes, x, y).T

    idx = []
    alfas_idx = []
    x_idx = []
    y_idx = []


    for i in range(len(alfas[0])):
        if alfas[0][i] > 0:
            idx.append(i)
            alfas_idx.append(alfas[0][i])
            y_idx.append(int(y[i][0]))
            x_idx.append([x[i, 0], x[i, 1]])

    x_idx = np.matrix(x_idx)

    print('Indice:               Alfa:               X:                    Y:')
    for i in range(len(alfas_idx)):
        print(idx[i],'                   ', alfas_idx[i], '                 ', x_idx[i, 0], x_idx[i, 1], '    ', y_idx[i])


    b = calc_b(idx, alfas_idx, y_idx, x_idx)

    size = len(x_test)
    predictions = []
    for i in range(size):
        res = np.multiply(y_idx, alfas_idx).T * K(x_idx, x_test[i])
        if res > 0:
            predictions.append(1)
        else:
            predictions.append(-1)

    accuracy = calc_accuracy(y_test, predictions)

    print()
    print('Accuracy:', accuracy)

    plot(x,y,x_idx,y_idx, b, alfas_idx, x_test, y_test)

#####################################################################
def K(X, x):
    if k_function == 'l':
        return X * x.T
    if k_function == 'g':
        res = []
        for i in range(len(X[:,0])):
            res.append(math.exp(-gm * np.linalg.norm(X[i] - x)))
        return np.matrix(res).T
    if k_function == 'p':
        return np.power(X * x.T, 3)


#####################################################################
def formula_2(alfa, b, x, y, i):

    res = np.multiply(y, alfa).T * K(x,x[i]) + b

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

    res = 2 * K(x[j], x[i]) - K(x[i], x[i]) - K(x[j], x[j])

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

    res = b - Ei - y[i] * (alfa[i] - aI_old) * K(x[i], x[i]) - y[j] * (alfa[j] - aJ_old) * K(x[j], x[i])

    return res

#####################################################################
def formula_18(b, Ej, alfa, x, y, aI_old, aJ_old, i, j):

    res = b - Ej - y[i] * (alfa[i] - aI_old) * K(x[j], x[i]) - y[j] * (alfa[j] - aJ_old) * K(x[j], x[j])

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
    max_it = 100
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
            print(passes)
        else:
            passses = 0

    return alfa


svm()