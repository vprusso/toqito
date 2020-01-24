import numpy as np


def calculate_hedging_angles(alpha: int, n: int) -> float:
    theta_n = np.atan(np.sqrt(1/(alpha**2)-1)*(2**(1/n)-1))
    vtheta_n = np.atan(np.sqrt(1/(alpha**2)-1)*(1/(2**(1/n)-1)))
    return theta_n, vtheta_n
    



