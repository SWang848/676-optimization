import cvxpy as cp
import numpy as np


### primal
import cvxpy as cp
import numpy as np

x = cp.Variable(shape=(2,1),name="x")

A = np.array([[1,2],[-1,1],[3,1]])
B = np.array([[4],[1],[9]])
constraints = [cp.matmul(A,x) <=B, x>=0]

r=np.array([-20,4])
objective = cp.Minimize(cp.matmul(r,x))

problem = cp.Problem(objective,constraints)

solution = problem.solve(solver = 'ECOS_BB')

print(solution)

print(x.value)


### dual
x = cp.Variable(shape=(3,1),name="x")

A = np.array([[1,-1,3],[2,1,1]])
B = np.array([[20],[-4]])
constraints = [cp.matmul(A,x) >=B, x>=0]

r=np.array([-4,-1,-9])
objective = cp.Maximize(cp.matmul(r,x))

problem = cp.Problem(objective,constraints)

solution = problem.solve(solver = 'ECOS_BB')

print(solution)

print(x.value)