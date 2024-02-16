import numpy as np

def compute_chordal_distances(X0, U_array, require_grad=False):
    n = len(U_array)
    chordal_dist = np.zeros((n, n))
    chordal_gradients = np.empty((n, n), dtype=object) if require_grad else None

    # Precompute frequently used matrices
    X0_X0T = [X0[i] @ X0[i].T for i in range(n)]

    for i in range(n):
        for j in range(n):
            A = X0_X0T[i] @ U_array[j]
            # Efficient SVD

            if require_grad:
                U_A, s_A, VT_A = np.linalg.svd(A, full_matrices=False, compute_uv=True)
                chordal_gradients[i][j] = -2 * s_A[0] * np.outer(U_A[:, 0], VT_A[0, :])
            else:
                s_A = np.linalg.svd(A, full_matrices=False, compute_uv=False)

            chordal_dist[i, j] = 1 - s_A[0]**2

    return chordal_dist, chordal_gradients


def compute_weights(chordal_dist, weight_factor, chordal_gradients = None, require_grad=False):
    n = len(chordal_dist)
    # Compute weights
    w = np.exp(weight_factor * -0.5 * (chordal_dist + chordal_dist.T))

    # Initialize w_gradients only if gradients are required
    w_gradients = np.empty((n, n), dtype=object) if require_grad else None

    if require_grad:
        # Vectorized computation of gradients
        factor = weight_factor * -0.5
        w_gradients = factor * w * chordal_gradients

    return w, w_gradients



def compute_geodesic_distances(U_array, require_grad=False):
    n, m, r = len(U_array), U_array[0].shape[0], U_array[0].shape[1]
    U_jT_U_j = [U @ U.T for U in U_array]  # Precompute U_j^T U_j
    geodesic_distances = np.zeros((n, n))
    geodesic_gradients = np.zeros((n, n, m, r)) if require_grad else None

    for i in range(n):
        for j in range(n):
            A = U_jT_U_j[j] @ U_array[i]
            u_j, s_j, vt_j = np.linalg.svd(A)
            s_j = np.clip(s_j, a_min=None, a_max=1)
            geodesic_distances[i, j] = np.sum(np.arccos(s_j[:r])**2)

            if require_grad:
                geodesic_gradients[i][j] = np.zeros((m, r))
                for r_index in range(r): # if/ else to account for computational errors when s=1, equivalent forms in the limit
                    if s_j[r_index] < 1:
                        geodesic_gradients[i][j] += -2 * np.arccos(s_j[r_index]) / np.sqrt(1 - s_j[r_index]**2) * np.outer(u_j[:, r_index] , vt_j[r_index, :])
                    else:
                        geodesic_gradients[i][j] += -2 * np.outer(u_j[:, r_index] , vt_j[r_index, :]) # gradient w.r.t U_i

    return geodesic_distances, geodesic_gradients
