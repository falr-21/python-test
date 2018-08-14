#
# declaração de todas bibliotecas relativas ao programa
#
import numpy as np
import math
#
# cálculo da matriz admitância a partir de um dicionário contendo os dados das barras do sistema
#
def calc_admt(line_data):
    def admt(lin, col):
        adm = 0
        if (lin, col) in line_data.keys():
            imp = complex(line_data[(lin, col)]['r'], line_data[(lin, col)]['x'])
            adm = -1/imp
        if (col, lin) in line_data.keys():
            imp = complex(line_data[(col, lin)]['r'], line_data[(col, lin)]['x'])
            adm = -1/imp
        return adm

    def admt_diag(i, dim):
        adm = 0
        for num in range(dim+1):
            if num != i and (i, num) in line_data.keys():
                imp = complex(line_data[(i, num)]['r'], line_data[(i, num)]['x'])
                adm = adm + 1/imp + 0.5*complex(0, line_data[(i, num)]['bsh'])
            if num != i and (num, i) in line_data.keys():
                imp = complex(line_data[(num, i)]['r'], line_data[(num, i)]['x'])
                adm = adm + 1/imp + 0.5*complex(0, line_data[(num, i)]['bsh'])
        return adm

    Y = np.zeros((num_bar, num_bar), dtype = np.complex_)
    for i in range(num_bar):
        for j in range(num_bar):
            if (i != j):
                Y[i][j] = admt(i+1, j+1)
            else:
                Y[i][j] = admt_diag(i+1, num_bar)
    return Y, Y.real, Y.imag
#
# cálculo da matriz Jacobiana mais atualizada
#
def calc_H(k, m):
    if (k == m):
        res = -B[k][k]*(V[k]**2)-V[k]*sum([V[x]*(G[k][x]*math.sin(T[k]-T[x])-B[k][x]*math.cos(T[k]-T[x])) for x  in range(num_bar)])
    else:
        res = V[k]*V[m]*(G[k][m]*math.sin(T[k]-T[m])-B[k][m]*math.cos(T[k]-T[m]))
    return res

def calc_N(k, m):
    if (k == m):
        res = G[k][k]*V[k]+sum([V[x]*(G[k][x]*math.cos(T[k]-T[x])+B[k][x]*math.sin(T[k]-T[x])) for x  in range(num_bar)])
    else:
        res = V[k]*(G[k][m]*math.cos(T[k]-T[m])+B[k][m]*math.sin(T[k]-T[m]))
    return res

def calc_M(k, m):
    if (k == m):
        res = -G[k][k]*(V[k]**2)+V[k]*sum([V[x]*(G[k][x]*math.cos(T[k]-T[x])+B[k][x]*math.sin(T[k]-T[x])) for x  in range(num_bar)])
    else:
        res = -V[k]*V[m]*(G[k][m]*math.cos(T[k]-T[m])+B[k][m]*math.sin(T[k]-T[m]))
    return res

def calc_L(k, m):
    if (k == m):
        res = -B[k][k]*V[k]+sum([V[x]*(G[k][x]*math.sin(T[k]-T[x])-B[k][x]*math.cos(T[k]-T[x])) for x  in range(num_bar)])
    else:
        res = V[k]*(G[k][m]*math.sin(T[k]-T[m])-B[k][m]*math.cos(T[k]-T[m]))
    return res

def calc_jaco(size, thres):
    J_temp = np.zeros((size, size),dtype=np.float_)
    for i in range(size):
        for j in range(size):
            if i < thres and j < thres:
                J_temp[i][j] = calc_H(idx[i], idx[j])
            if i < thres and j >= thres:
                J_temp[i][j] = calc_N(idx[i], idx[j])
            if i >= thres and j < thres:
                J_temp[i][j] = calc_M(idx[i], idx[j])
            if i >= thres and j >= thres:
                J_temp[i][j] = calc_L(idx[i], idx[j])
    return J_temp
#
# cálculo dos mismatches de potência
#
def calc_mis(V, T, P, Q, T_idx, V_idx, idx):
    P_mis = [0]*len(T_idx)
    Q_mis = [0]*len(V_idx)
    for m, k in zip(V_idx, range(len(V_idx))):
        Q_mis[k] = Q[m] - V[m]*sum([V[x]*(G[m][x]*math.sin(T[m]-T[x])-B[m][x]*math.cos(T[m]-T[x])) for x  in range(num_bar)])

    for n, k in zip(T_idx, range(len(T_idx))):
        P_mis[k] = P[n] - V[n]*sum([V[x]*(G[n][x]*math.cos(T[n]-T[x])+B[n][x]*math.sin(T[n]-T[x])) for x  in range(num_bar)])
    
    mis = P_mis + Q_mis
    return mis
#
# atualizando os valores das incógnitas
#
def calc_inc(V, T, T_idx, V_idx, idx, mis):
    inc = [0]*len(idx)
    inc = np.linalg.inv(J) @ mis
    for i, j in zip(T_idx, range(len(T_idx))):
        T[i] += inc[j]
    for i, j in zip(V_idx, range(len(V_idx))):
        V[i] += inc[j + len(T_idx)]
    return V, T
#
# calculando a potência líquida nas barras
#
def calc_pot(V, T, P, Q, num_bar):
    for k in range(num_bar):
        P[k] = V[k]*sum([V[x]*(G[k][x]*math.cos(T[k]-T[x])+B[k][x]*math.sin(T[k]-T[x])) for x  in range(num_bar)])
    for k in range(num_bar):
        Q[k] = V[k]*sum([V[x]*(G[k][x]*math.sin(T[k]-T[x])-B[k][x]*math.cos(T[k]-T[x])) for x  in range(num_bar)])
    return P, Q
#
# dados
#
line_data = {(1, 2):{'r':0.1, 'x':1, 'bsh':0.01}, (1, 3):{'r':0.2, 'x':2, 'bsh':0.02}, (2, 3):{'r':0.1, 'x':1, 'bsh':0.01}}

V = [1, '-', 0.98]
Q = ['-', -0.02, '-']

T = [0, '-', '-']
P = ['-', -0.05, -0.15]

num_bar = len(V)
#
# adequação dos dados
#
V_idx = [i for i, x in enumerate(V) if x == '-']        
T_idx = [i for i, x in enumerate(T) if x == '-']
idx = T_idx + V_idx
#
# adequação dos dados
#
for i, j in zip(V_idx, range(len(V_idx))):
    if V[i] == '-':
        V[i] = 1

for i, j in zip(T_idx, range(len(T_idx))):
    if T[i] == '-':
        T[i] = 0
#
# definição do erro tolerado
#
E = float(input("Insira o erro tolerado: "))
#
# resolução do problema de fluxo de potência
#
Y, G, B = calc_admt(line_data)
print("-----------------------")
print("Matriz admitância:")
print("-----------------------")
print(Y)
flag = 0
count = 0
while (flag == 0):
    print("-----------------------")
    print("Iteração ", count)
    print("-----------------------")
    print("- Matriz dos módulos das tensões:\n", V)
    print("- Matriz das fases das tensões:\n", T)
    J = calc_jaco(len(idx), len(T_idx))
    print("- Matriz Jacobiana:\n", J)
    mis = calc_mis(V, T, P, Q, T_idx, V_idx, idx)
    print("- Matriz de mismatches:\n", mis)
    #
    # verificando o resultado
    #
    if ((len(list(filter(lambda x: abs(x) <  E, mis)))) == len(mis)):
        #
        # finalização do programa
        #
        print("-----------------------")
        print("Resultado final:")
        print("-----------------------")
        print("- Matriz dos módulos das tensões:\n", V)
        print("- Matriz das fases das tensões:\n", T)
        P, Q = calc_pot(V, T, P, Q, num_bar)
        print("- Matriz de potência ativa:\n", P)
        print("- Matriz de potência aparente:\n", Q)
        flag = 1
    #
    # correção das incógnitas
    #
    V, T = calc_inc(V, T, T_idx, V_idx, idx, mis)
    count += 1
#