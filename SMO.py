import random


#####################################################################
def formula_2(alfa, b, x, y): #NOT DONE

    for i in range(x.size):
        res += alfa[i] *  + b

    return res

#####################################################################
def formula_10(alfaI, alfaJ, C, yI, yJ):   #APLICAR SÓ SE yI != yJ:

    L = max(0, aI - aJ)
    H = min(C, C + aJ - aI)

    return L, H

#####################################################################
def formula_11(alfaI, alfaJ, c, yI, yJ):    #APLICAR SÓ SE yI == yJ:

    L = max(0, aI + aJ - C)
    H = min(C, aI + aJ)

    return L, H

#####################################################################
def formula_12(aJ, Ei, Ej, N, yJ):

    aJ = aJ - ((yJ * ( Ei - Ej)) / N)
    
    return aJ

#####################################################################
def formula_14():
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
def formula_19():
    return res

def randomIndex(i, x):
    j = i
    while(j == i):
        j = int (random.uniform(0, x.size))
    return j


def SMO(c, tol, max_passes, x, y):
    alfa = []
    b = 0
    passes = 0

    while(passes < max_passes): #Ciclo While exterior
        num_changed_alfas = 0
        for i in range(1, x.size):
            Ei = formula_2(alfa, b, x, y) # CALCULAR Ei -> FORMULA 2
            if 1: #AQUELE IF ENORME

                #SELECIONAR RANDOMLY
                j = randomIndex(i, x)

                #CALCULAR Ej -> FORMULA 2
                Ej = formula_2(alfa, b, x, y)

                #GUARDAR ALFAS ANTIGOS
                aI_old = alfa[i] 
                aJ_old = alfa[j]

                #CALCULAR L E H -> FORMULA 10 OU 11
                if y[i] != y[j]: 
                    L, H = formula_10(alfa[i], alfa[j], C, y[i], y[j]) 
                else:
                    L, H = formula_11(alfa[i], alfa[j], c, y[i], y[j])
                
                #PASSAR PARA A PRÓXIMA ITERAÇÃO
                if(L == H): continue
                
                #CALCULAR N -> FORMULA 14
                N = formula_14(alfa[j], H, L)

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
                b = formula_19()

                num_changed_alfas += 1

        if num_changed_alfas == 0:
            passes += 1
        else:
            passses = 0

    return alfa, b