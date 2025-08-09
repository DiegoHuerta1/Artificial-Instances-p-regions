import numpy as np

### Based on: The Uniqueness of a Good Optimum for K-Means


def get_k_means_optimality_status(X: np.ndarray, num_regions: int, c_labels: np.ndarray) -> str:
    """
    Get the optimality status of a partition of vectors.

    Args:
        X (np.ndarray): Shape (n, d). Describes n vectors of length d.
        num_regions (int). Number of regions of the partition.
        c_labels (np.ndarray): Shape (n,). Describes a partition. Indices (0, 1, ..., num_regions-1)

    Returns:
        str: Optimality status of the partition.
    """
    status: str = "Not computed"

    # Compute an upper bound of the error
    validity, max_ME = collorary_4_bound(X, num_regions, c_labels)
    max_errors = max_ME * X.shape[0] # max number of missclassified vectors

    # Global optimum is achieved
    if validity and max_errors < 1:
        status = "Global Optimum"

    # The bound works, but does not indicate global optimum
    elif validity and max_errors >= 1:
        status = f"Max errors = {max_errors}"

    # The bound is not useful
    else:
        status = "Unknown"

    return status


def collorary_4_bound(X: np.ndarray, num_regions: int, c_labels: np.ndarray) -> tuple[bool, float]:
    """
    Computes the upper bound of the Misclassification Error (ME)

    Args:
        X (np.ndarray): Shape (n, d). Describes n vectors of length d.
        num_regions (int): Number of regions of the partition.
        c_labels (np.ndarray):  Shape (n,). Describes a partition. Indices (0, 1, ..., num_regions-1)

    Returns:
        tuple[bool, float]: (validity, bound)
            validity: boolean. Indicates if the upper bound is valid or not
            bound: value of the bound, it is valid only if validity=True
    """

    # Compute helper variables
    Z = X - np.mean(X, axis = 0)
    p_max, p_min = compute_p_max_and_min(num_regions, c_labels)
    lambda_ = get_lambda_from_labels(Z, num_regions, c_labels)
    epsilon = compute_epsilon_from_labels(Z, num_regions, c_labels, c_labels)

    # Check if the result is valid
    if lambda_ <= (num_regions - 1)/2 and epsilon <= p_min:
        return (True, epsilon*p_max)
    else:
        return (False, float("inf"))
    

def compute_epsilon_from_labels(Z, K, labels, labels_prime):
  lambda_ = get_lambda_from_labels(Z, K, labels)
  lambda_prime = get_lambda_from_labels(Z, K, labels_prime)
  return 2 * np.sqrt(lambda_ * lambda_prime * (1 - lambda_/(K-1)) * (1 - lambda_prime/(K-1)) )


def get_lambda_from_labels(Z, K, labels):
    S = Z @ Z.T 
    small_sigmas, sigma_K, _, _, _ = principal_eigencomponents(S, K)
    D_star = S.trace() - sum(small_sigmas)
    lambda_ = (distorsion_from_labels(Z, K, labels) - D_star) / (small_sigmas[-1] - sigma_K)
    return lambda_


def compute_p_max_and_min(K, labels):
    n_k = np.array([ (labels == k).astype(float).sum() for k in range(K)])
    n_k_normalized = n_k / n_k.sum()
    max_value = np.max(n_k_normalized)
    min_value = np.min(n_k_normalized)
    return max_value, min_value


def distorsion_from_labels(Z, K, labels):
    dist = 0
    for k in range(K):
        points_cluster = Z[labels == k]
        mu_k = points_cluster.mean(axis = 0)
        squared_norms = ((points_cluster - mu_k)**2).sum(axis = 1)
        dist += squared_norms.sum()
    return dist


def principal_eigencomponents(matrix: np.ndarray, K: int):
    """
    Returns the top K-1 principal eigenvalues and corresponding eigenvectors
    of a given square matrix.

    Parameters:
    matrix (np.ndarray): An n x n square matrix (symmetric recommended).
    K (int): Number such that K - 1 principal components are returned.

    Returns:
    eigvals (np.ndarray): (K-1,) array of the largest eigenvalues.
    eigvecs (np.ndarray): (n, K-1) array of corresponding eigenvectors.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")
    if K < 2 or K > matrix.shape[0]:
        raise ValueError("K must be in the range [2, n] where n is matrix size.")

    # Compute all eigenvalues and eigenvectors
    eigvals_all, eigvecs_all = np.linalg.eigh(matrix)

    # Sort in descending order
    idx = np.argsort(eigvals_all)[::-1]
    eigvals_sorted = eigvals_all[idx]
    eigvecs_sorted = eigvecs_all[:, idx]

    # Return top K-1
    return eigvals_sorted[:K-1], eigvals_sorted[K], eigvecs_sorted[:, :K-1], eigvecs_sorted[K-1, :], eigvecs_sorted

