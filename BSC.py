import numpy as np
from toqito.nonlocal_games.nonlocal_game import NonlocalGame

def bcs(M, b):
    m, n = M.shape
    constraints = []
    for i in range(m):
        rhs = (-1) ** b[i]  
        constraint = np.concatenate((M[i], np.array([rhs])))
        constraints.append(constraint)
    return constraints

def approximate(M, b, level):
    constraints = bcs(M, b)
    game = NonlocalGame.from_bcs_game(constraints, reps=1)
    
    val = game.commuting_measurement_value_upper_bound(k=level)
    return val

M = np.array([[1, 1, 1],[0, 1, 1]])
b = np.array([1, 0])
    
val = approximate(M, b, 3)
print("Commuting measurement value upper bound:", val)


