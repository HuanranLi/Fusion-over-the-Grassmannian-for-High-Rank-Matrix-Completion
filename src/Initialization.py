import numpy as np

#OLD CODE; init a low-rank subspace
def create_low_rank_matrix(_n,_m,_r):
    #random init
    _X_full = np.random.randn(_n,_m)

    #truncated SVD
    _U, _s, _VT = np.linalg.svd(_X_full)
    _Xr = np.round(_U[:,:_r],2)@np.round(np.diag(_s[:_r]),2)@np.round(_VT[:_r,:],2)
    return _Xr

#OLD CODE; combine n subspace into 1 cluster
def create_n_subspace_clusters(n_clusters = 2, shape = (100,100,5)):

    (m,n,r) = shape

    #init several low-rank subspace
    X_lowRank_array = [create_low_rank_matrix(m,n,r) for i in range(n_clusters)]

    #init the random masks for each subspace
    masks_order = [i for i in range(n)]
    np.random.shuffle(masks_order)
    mask_length = n // n_clusters
    masks = [masks_order[i*mask_length: (i+1) * mask_length] if i != n_clusters - 1 else masks_order[i*mask_length:] for i in range(n_clusters)]
    print('masks shape: ', [np.shape(i) for i in masks])

    #fill in the cluster
    Xm = np.zeros((m,n))
    noise = np.random.randn(m,n)
    for matrix_i in range(n_clusters):
        for col in masks[matrix_i]:
            Xm[:,col] = X_lowRank_array[matrix_i][:,col]

    labels = []
    for i in range(n):
        for mask_i in range(len(masks)):
            if i in masks[mask_i]:
                labels.append(mask_i)
                break


    return Xm,masks,X_lowRank_array,labels


def initialize_X_with_missing(m,n,r,K,missing_rate):

    shape = (m,n,r)
    X, masks, X_lowRank_array,labels = create_n_subspace_clusters(n_clusters=K, shape = shape)

    #observed index
    Omega = np.random.choice(m*n, size = int(m*n * (1-missing_rate) ), replace= False )

    #create observed matrix
    X_omega = np.zeros((m,n))
    for p in Omega:
        X_omega[p // n, p % n] = X[p // n, p % n]

    info = {'X':X, 'masks': masks, 'X_lowRank_array': X_lowRank_array,  }
    return X_omega, labels, Omega, info
