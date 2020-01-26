from toqito.helper.constants import e0, e1, e00, e01, e10, e11, ep, em
from toqito.hedging.calculate_q import calculate_q
import numpy as np


def calculate_hedging_value(n, k, alpha, theta):
    v = np.cos(theta)*e00 + np.sin(theta)*e11
    sigma = v * v.conj().T

    P1 = sigma
    P0 = np.identity(4) - P1
    
    w = alpha * np.cos(theta)*e00 + np.sqrt(1-alpha**2) * np.sin(theta)*e11
    l1 = -alpha*np.sin(theta)*e00 + np.sqrt(1-alpha**2) * np.cos(theta)*e11
    l2 = alpha * np.sin(theta)*e10
    l3 = np.sqrt(1-alpha**2)*np.cos(theta)*e01
    
    Q1 = w * w.conj().T

    Q0 = l1 * l1.conj().T + l2 * l2.conj().T + l3 * l3.conj().T
    
    print(Q0)
    print(Q1)


    #Q0_nk, Q1_nk = calculate_q(Q0, Q1, n, k)



    #print(Q0_nk)
    #print(Q1_nk)
    
