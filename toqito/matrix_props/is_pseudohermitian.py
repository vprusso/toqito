def is_pseudohermitian(mat: np.ndarray, signature_tup: tuple):
    """
    Checks if a matrix is pseudo-hermitian with respect to a given signature.

    Args:
        mat (np.ndarray): The matrix to check.
        signature_tup (tuple): A tuple (p, q) where p is the number of +1's and q is the number of -1's in the metric.

    Returns:
        bool: True if the matrix is pseudo-hermitian, False otherwise.
    """
    # Check if the matrix is square
    if mat.shape[0] != mat.shape[1]:
        return print("Error: The matrix must be square.")
        

    # Check if the signature tuple is valid
    if len(signature_tup) != 2:
        return print("Error: The signature tuple must have exactly two elements (p, q).")
        

    p, q = signature_tup
    if p + q != mat.shape[0]:
        return print("Error: The signature does not match the matrix dimension.")
        

    # Construct the metric matrix
    metric = np.diag([1] * p + [-1] * q)
    inverse_metric = np.linalg.inv(metric)


    # Use np.allclose for approximate equality
    is_pseudo = np.allclose(metric @ mat @ inverse_metric, mat.conj().T)
    return is_pseudo