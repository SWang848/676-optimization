import cvxpy
import numpy as np
import random
import math


def input_generator(pmin, pmax, length, infinitesimal=True, dim=1):
    w = []; v = [];
    while len(w) < length:
        if infinitesimal:
            rand_v = np.random.beta(.2,1,1).item()
            rand_w = np.random.uniform(10e-6, 10e-3, dim)
            if dim > 1:
                if np.all(rand_v/rand_w <= pmax) & np.all(rand_v/rand_w >= pmin):
                    w.append(rand_w)
                    v.append(rand_v)
                else:
                    pass
            else:
                if pmin <= rand_v/rand_w <= pmax:
                    w.append(rand_w)
                    v.append(rand_v)
                else:
                    pass
        else:
            rand_v = np.random.beta(.2,1,1).item()
            rand_w = random.uniform(0.01, 0.1)
            if pmin <= rand_v/rand_w <= pmax:
                w.append(rand_w)
                v.append(rand_v)
            else:
                pass
    return (v, w)

#adapted from https://towardsdatascience.com/integer-programming-in-python-1cbdfa240df2
#in: weights, values, capacity
#out: solution vector (1) if item i chosen, value of objective
def offlineSolve(value, weight, cap):
    selection = cvxpy.Variable((len(value)),boolean=True)

    c1 = sum(cvxpy.multiply(weight,selection)) <= cap

    kp = cvxpy.Problem(cvxpy.Maximize(sum(cvxpy.multiply(value, selection))), [c1])

    kp.solve(solver=cvxpy.GLPK_MI)

    sol = selection.value
    sol = np.asarray(sol)
    maxobj = sum(np.multiply(sol,value))

    # print(sol)
    # print(maxobj)

    return sol, maxobj

def psi(y, p_min, p_max, infinitesimal, dim):

    beta_op =  1/(1+math.log(p_max/p_min))

    if dim==1:
        if infinitesimal:
            if 0 <= y < beta_op:
                return p_min
            if beta_op <= y <= 1:
              return p_min * math.exp(y/beta_op-1)
        else:
            if 0 <= y < beta_op:
                return p_min
            if beta_op <= y:
                return p_min * math.exp(y/beta_op-1)
    else:
        if np.all(y < beta_op) & np.all(y >= 0):
            return p_min
        else:
            return p_min * math.exp(y/beta_op-1)

def main(p_min, p_max, length=50, runs=50, infinitesimal=True, dim=1):
    
    CRs = [];
    for _ in range(runs):
        input = input_generator(p_min, p_max, length, infinitesimal, dim)
        if dim==1:
            y = 0
        else:
            y = np.full((dim, 1), 0)
        obj=0; X = [];
        items = []

        for value, weight in zip(input[0], input[1]):
            if np.all((value/weight) < psi(y, p_min, p_max, infinitesimal, dim)):
                x = 0
            else:
                x = 1
            
            y = np.asarray(weight).reshape((dim, -1))*x + y
            obj += x*value
            X.append(x)
            items.append((value, weight))

        items = np.asarray(items)
        # offlineSol, offlineMax = offlineSolve(items[:, 0], items[:, 1], 1)
        # print(offlineMax)
        # print(obj)
        # CRs.append(offlineMax/obj)

    print("Average CR over {} runs is {}" .format(runs, np.mean(np.asarray(CRs))))

main(1, 10, infinitesimal=True, dim=2)