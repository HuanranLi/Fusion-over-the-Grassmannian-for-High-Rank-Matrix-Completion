import numpy as np
import matplotlib.pyplot as plt
import time
from helper_functions import dUU
import os
from distances import *

class GrassmannianFusion:
    X: np.ndarray
    Omega: np.ndarray
    r: int
    lamb: float
    step_size: float
    g_threshold: float
    bound_zero: float
    singular_value_bound: float
    g_column_norm_bound: float
    U_manifold_bound: float
    shape:list

    m:int
    n:int

    U_array: np.ndarray
    X0:list

    def save_model(self, path):
        np.savez_compressed(path, X=self.X,
                                Omega = self.Omega,
                                r = self.r,
                                lamb = self.lamb,
                                g_threshold = self.g_threshold,
                                bound_zero = self.bound_zero,
                                singular_value_bound = self.singular_value_bound,
                                g_column_norm_bound = self.g_column_norm_bound,
                                U_manifold_bound = self.U_manifold_bound,
                                U_array = self.U_array)

        print('Successfully save to: ' + path )
        return True

    def load_model(path):
        #rebuild the object
        data = np.load(path)
        GF = GrassmannianFusion(X = data['X'],
                Omega = data['Omega'],
                r = data['r'],
                lamb = data['lamb'],
                g_threshold = data['g_threshold'],
                bound_zero = data['bound_zero'],
                singular_value_bound = data['singular_value_bound'],
                g_column_norm_bound = data['g_column_norm_bound'],
                U_manifold_bound = data['U_manifold_bound'],
                U_array = data['U_array'])

        print('Successfully loaded the model!')
        return GF


    #U_array load version
    def __init__(self, X, Omega, r, lamb,
                g_threshold = 0.15,
                bound_zero = 1e-10,
                singular_value_bound = 1e-2,
                g_column_norm_bound = 1e-5,
                U_manifold_bound = 1e-2,
                **optional_params):

        print('\n########### GrassmannianFusion Initialization Start ###########')
        self.X = X
        self.Omega = Omega
        self.r = r
        self.lamb = lamb
        self.g_threshold = g_threshold
        self.bound_zero = bound_zero
        self.singular_value_bound = singular_value_bound
        self.g_column_norm_bound = g_column_norm_bound
        self.U_manifold_bound = U_manifold_bound

        self.m = X.shape[0]
        self.n = X.shape[1]

        self.shape = [self.m, self.n, self.r]

        if 'U_array' in optional_params:
            self.load_U_array(optional_params['U_array'])
            print('U_array Loaded successfully!')
        else:
            self.initialize_U_array()
            print('U_array initialized successfully!')

        self.construct_X0()

        print('########### GrassmannianFusion Initialization END ###########\n')

    def get_U_array(self):
        return self.U_array

    def load_U_array(self, U_array):
        # del self.U_array
        self.U_array = U_array

    def change_lambda(lamb: float):
        self.lamb = lamb

    def initialize_U_array(self):
        U_array = np.array([np.random.randn(self.m, self.r) for i in range(self.n)])
        for i in range(self.n):
            U_array[i,:,0] = self.X[:,i] / np.linalg.norm(self.X[:,i])
            q_i,r_i = np.linalg.qr(U_array[i])
            U_array[i,:,:] = q_i * r_i[0,0]

            #print(U_array[i].shape)
            #make sure the first col is x_i
            assert np.linalg.norm(U_array[i,:,0] - self.X[:,i] / np.linalg.norm(self.X[:,i])) < self.bound_zero
            #make sure its orthogonal
            assert np.linalg.norm(U_array[i].T @ U_array[i] - np.identity(self.r)) < self.bound_zero
            #make sure its normal
            assert  np.linalg.norm( np.linalg.norm(U_array[i], axis = 0) - np.ones(self.r) )  < self.bound_zero

        self.load_U_array(U_array)

    def construct_X0(self):
        #construct X^0_i
        Omega_i = [np.sort( self.Omega[ self.Omega % self.n == self.i]) // self.n for self.i in range(self.n)]
        #find the compliment of Omega_i
        Omega_i_compliment = [sorted(list(set([i for i in range(self.m)]) - set(list(o_i)))) for o_i in Omega_i]
        #calculate length of U
        len_Omega = [o.shape[0] for o in Omega_i]

        #init X^0
        self.X0 = [np.zeros((self.m, self.m - len_Omega[i] + 1)) for i in range(self.n)]
        for i in range(self.n):
            #fill in the first row with normalized column
            self.X0[i][:,0] = self.X[:,i] / np.linalg.norm(self.X[:,i])
            for col_index,row_index in enumerate(Omega_i_compliment[i]):
                #fill in the "identity matrix"
                self.X0[i][row_index, col_index+1] = 1


    def train(self, max_iter:int, step_size:float):
        obj_record = []
        gradient_record = []
        start_time = time.time()

        print('\n################ Training Process Begin ################')
        #main algo
        for iter in range(max_iter):

            new_U_array, end, gradient_norm = self.Armijo_step(alpha = step_size,
                                                        beta = 0.5,
                                                        sigma = 1e-5)

            new_np_U_array = np.empty((self.n, self.m, self.r))

            for i in range(self.n):
                u,s,vt = np.linalg.svd(new_U_array[i], full_matrices= False)
                new_np_U_array[i,:,:] = u@vt

            # assert np.linalg.norm(new_np_U_array[i].T @ new_np_U_array[i] - np.identity(new_np_U_array[i].shape[1])) < self.U_manifold_bound
            self.load_U_array(new_np_U_array)

            #record
            obj = self.cal_obj(self.U_array)
            obj_record.append(obj)
            gradient_record.append(gradient_norm)

            #print log
            if iter % 10 == 0:
                print('iter', iter)
                print('Obj value:', obj)
                print('gradient:', gradient_norm)
                print('Time Cost(min): ', (time.time() - start_time)/60 )
                print()

            if end:
                print('Final iter', iter)
                print('Final Obj value:', obj)
                break

        print('################ Training Process END ################\n')


    def cal_obj(self, U_array, require_grad = False):

        # Chordal Distance
        chordal_distances, chordal_gradients = compute_chordal_distances(self.X0, U_array, require_grad)

        # Geodesic Distance
        geodesic_distances, geodesic_gradients = compute_geodesic_distances(U_array, require_grad)

        # Compute objective function
        # obj = np.sum(chordal_distances.diagonal()) + self.lamb / 2 * np.sum(w * geodesic_distances)
        obj = np.sum(chordal_distances.diagonal()) + self.lamb / 2 * np.sum(geodesic_distances)

        if require_grad:
            return obj, {'chordal_dist':chordal_distances, 'geodesic_distances': geodesic_distances,
                        'chordal_gradients': chordal_gradients, 'geodesic_gradients':geodesic_gradients}
        else:
            return obj


    def compute_gradients_and_norm(self, chordal_gradients, geodesic_distances, geodesic_gradients):
        grad_f_array = []
        identity_m = np.identity(self.m)

        # Precompute terms that are used repeatedly
        U_proj = [identity_m - U @ U.T for U in self.U_array]

        for i in range(self.n):
            grad_f_i = chordal_gradients[i][i].copy()  # Copy to avoid modifying the original gradient

            # Accumulate gradients
            for j in range(self.n):
                grad_f_i += self.lamb / 2 * geodesic_gradients[i][j]

            # Apply projection
            grad_f_i = U_proj[i] @ grad_f_i
            grad_f_array.append(grad_f_i)

        # Compute gradient norm
        gradient_norm = np.sqrt(sum(np.trace(grad.T @ grad) for grad in grad_f_array))

        return grad_f_array, gradient_norm



    def Armijo_step(self, alpha = 1, beta = 0.5, sigma = 0.9):
        L = R = 0

        obj, info = self.cal_obj(self.U_array, require_grad = True)
        chordal_dist = info['chordal_dist']
        geodesic_distances = info['geodesic_distances']
        chordal_gradients = info['chordal_gradients']
        geodesic_gradients = info['geodesic_gradients']


        grad_f_array, gradient_norm = self.compute_gradients_and_norm(chordal_gradients, geodesic_distances, geodesic_gradients)

        arm_m = 0
        constant_term_L = np.sum(chordal_dist.diagonal()) + self.lamb / 2 * np.sum( geodesic_distances)
        grad_norm_squared = np.sum([np.trace(g.T @ g) for g in grad_f_array])

        while True:
            step_size = (beta ** arm_m) * alpha
            new_U_array = []

            for i in range(self.n):
                # Compute new U using the Armijo step
                Gamma_i, Del_i, ET_i = np.linalg.svd(-step_size * grad_f_array[i], full_matrices=False)
                first_term = np.concatenate((self.U_array[i] @ ET_i.T, Gamma_i), axis=1)
                second_term = np.concatenate((np.diag(np.cos(Del_i)), np.sin(np.diag(Del_i))), axis=0)
                new_U_array.append(first_term @ second_term @ ET_i)

            # Compute L for the new U_array
            L = constant_term_L - self.cal_obj(new_U_array)

            # Compute R
            R = -sigma * grad_norm_squared * step_size

            if L >= R:
                return new_U_array, False, gradient_norm
            else:
                if step_size < 1e-10:
                    return self.U_array, True, gradient_norm

            arm_m += 1


    def distance_matrix(self):
        # Assuming self.U_array is a 3D numpy array where each 'slice' self.U_array[i] is a matrix
        n = self.U_array.shape[0]

        d_matrix = np.zeros((n, n))  # Preallocate distance matrix
        for i in range(n):
            for j in range(i+1, n):  # Only compute upper triangle
                d_matrix[i, j] = dUU(self.U_array[i], self.U_array[j], self.r)

        # Mirror the upper triangle to the lower triangle since the matrix is symmetric
        d_matrix += d_matrix.T

        return d_matrix
