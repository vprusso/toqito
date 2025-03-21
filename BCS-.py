import numpy as np
from toqito.nonlocal_games.nonlocal_game import NonlocalGame

def bcs(M, b):
    """
    Constructs a NonlocalGame object for the given binary constraint system.
    """

    m, n = M.shape
    aq = list(range(m))  # List of Alice's possible questions
    bq = {}#Bob's questions
    for i in range(m):
        L = []
        for j in range(n):
            if M[i, j] == 1:
                L.append(j)
        bq[i] = L

    aa = {#Alice's possible answers with sublists of length 2**len(bq[s])
        s: [[int(bit) for bit in format(i, f'0{len(bq[s])}b')]
            for i in range(2 ** len(bq[s]))]
        for s in range(m)
    }
    
    ba = {s: [0, 1] for s in range(m)}#Bob's assignment always either 0 or 1 on one variable
    prob = np.zeros((m, n), dtype=float)
    for i in range(m):
        for j in range(n):
            if M[i, j] == 1:
                prob[i, j] = 1
    prob /= np.sum(prob) #Normalized uniformly distributed to assign

    pred = np.zeros((2 ** max(len(bq[s]) for s in range(m)), 2, m, n), dtype=float)#Alice's output, Bob's output, Alice's input, Bob's input 

    for i in range(m):
        for j in bq[i]:
            for k in range(len(aa[i])):
                G = aa[i][k]
                if sum(G) % 2 == b[i]:
                    for l in range(2):
                        if G[bq[i].index(j)] == l:
                            pred[k, l, i, j] = 1.0

    return NonlocalGame(prob, pred, reps=1)#Each round run once only

def approximate(M, b, level):
    game = bcs(M, b)
    return game.commuting_measurement_value_upper_bound(k=level)


M = np.array([[1, 1, 1], [0, 1, 1]])
b = np.array([1, 0])#here is x11+x12+x13=b1 mod 2; x22+x23=b2 

val = approximate(M, b, level=2)
print("Commuting measurement value upper bound:", val)

