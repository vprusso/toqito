
def permute_systems(X, perm, dim=None, row_only: bool = False, inv_perm: bool = False):
    print("\n\n")
    print("PERMUTE SYSTEMS")
    print("\n\n")
    if len(X.shape) == 1:
        dX = (1, X.shape[0])
    else:
        dX = X.shape

    is_vec = X.ndim == 1
    num_sys = len(perm)
    print(f"dX: {dX}")
    print(f"is_vec: {is_vec}")
    print(f"num_sys: {num_sys}")
    if is_vec:
        # 1 if column vector
        if len(dX) == 2:
            vec_orien = 1
        # 2 if row vector
        else:
            vec_orien = 0
        print(f"vec_orien: {vec_orien}")

    if dim is None:
        x = dX[0]**(1/num_sys) * np.ones(num_sys)
        y = dX[1]**(1/num_sys) * np.ones(num_sys)
        dim = np.array([x, y])

    print(f"dim: {dim}")
    print(f"row_only: {row_only}")
    print(f"inv_perm: {inv_perm}")
    print(f"size(dim): {dim.shape}")

    if len(dim.shape) == 1:
        # Force dim to be a row vector.
        dim_tmp = dim[:].T
        print(f"dim_tmp: {dim_tmp}")
        print(f"is_vec: {is_vec}")
        if is_vec:
            dim = np.ones((2, len(dim)))
            print(f"is_vec_dim: {dim}")
            print(f"vec_orien: {vec_orien}")
            dim[vec_orien, :] = dim_tmp
            print(f"dim[vec_orien, :]: {dim[vec_orien, :]}")
        else:
            dim = np.array([[dim_tmp],
                            [dim_tmp]])

    prod_dimR = np.prod(dim[0, :])
    prod_dimC = np.prod(dim[1, :])

    print(f"prod_dimR: {prod_dimR}")
    print(f"prod_dimC: {prod_dimC}")


    if len(perm) != num_sys:
        raise ValueError("InvalidPerm: length(PERM) must equal length(DIM).")
    elif sorted(perm) != list(range(1, num_sys+1)):
        raise ValueError("InvalidPerm: PERM must be a permutation vector")
    elif dX[0] != prod_dimR or (not row_only and dX[1] != prod_dimC):
        raise ValueError("InvalidDim: The dimensions specified in DIM do not")

    if is_vec:
        if inv_perm:
            PX = None
        else:
            print(f"X: {X}")
            print(f"dim[vec_orien, ::-1]: {dim[vec_orien, ::-1]}")
            PX_1 = X.reshape(dim[vec_orien, ::-1].astype(int), order="F")
            print(f"PX_1: {PX_1}")
            print(f"perm: {perm}")
            print(f"num_sys - perm[::-1]: {num_sys - np.array(perm[::-1])}")
            PX = vec(np.transpose(PX_1, num_sys - np.array(perm[::-1]))).T
            print(f"PX is_vec not flat: {PX}")
            # We need to flatten out the array.
            PX = functools.reduce(operator.iconcat, PX, [])
            print(f"PX is_vec flat: {PX}")
            #PX = np.array([0, 4, 1, 5, 2, 6, 3, 7])
        return PX
    
    vec_arg = np.array(list(range(0, dX[0])))

    print(f"vec_arg: {vec_arg}")
    print(f"perm: {perm}")
    print(f"dim[0, :]: {dim[0, :]}")

    row_perm = permute_systems(vec_arg, perm, dim[0, :], False, inv_perm)
    print(f"row_perm: {row_perm}")
    PX = X[row_perm, :]
    print(PX)

    if not row_only:
        vec_arg = np.array(list(range(0, dX[1])))
        col_perm = permute_systems(vec_arg, perm, dim[1, :], False, inv_perm)
        print(f"Before PX: {PX}")
        PX = PX[:, col_perm]
        print(f"col_perm: {col_perm}")
        print(f"After PX: {PX}")

    print(PX)
    return PX

def main():

    X = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])

    X_perm = permute_systems(X, [2, 1])
    print("***")
    print(X_perm)
    print("***")

    X = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                  [9, 10, 11, 12, 13, 14, 15, 16],
                  [17, 18, 19, 20, 21, 22, 23, 24],
                  [25, 26, 27, 28, 29, 30, 31, 32],
                  [33, 34, 35, 36, 37, 38, 39, 40],
                  [41, 42, 43, 44, 45, 46, 47, 48],
                  [49, 50, 51, 52, 53, 54, 55, 56],
                  [57, 58, 59, 60, 61, 62, 63, 64]])

    #X_perm = permute_systems(X, [2, 3, 1])
    #print("***")
    #print(X_perm)
    #print("***")
