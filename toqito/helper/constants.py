from toqito.matrix.operations.tensor import tensor
import numpy as np

e0 = np.array([[1], [0]])
e1 = np.array([[0], [1]])
ep = np.array([[1], [1]])/np.sqrt(2)
em = np.array([[1], [-1]])/np.sqrt(2)

e00 = tensor(e0, e0)
e01 = tensor(e0, e1)
e10 = tensor(e1, e0)
e11 = tensor(e1, e1)

