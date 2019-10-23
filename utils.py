from __future__ import division, print_function
import numpy as np 
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt 

def zeros_infDiag(shape):
    '''Matrix of zeros with np.inf on the diangonal'''
    M = np.zeros(shape)
    np.fill_diagonal(M, np.inf)
    return M

def compute_true_D(X):
    '''Compute true Euclidean distance'''
    return squareform(pdist(X))  

def d_oracle(D, p1, p2, sigma):
    '''Distance oracle. D is n x n true distance matrix,
        p1, p2 are indices. sigma is the standard deviation'''
    return D[p1,p2] + sigma * np.random.standard_normal(D[p1,p2].shape)

def get_true_nns(X):
    '''Returns each points true nearest neighbor'''
    nbrs = NearestNeighbors(n_neighbors=2).fit(X)
    _, true_nns = nbrs.kneighbors(X)
    return true_nns[:, 1]

def shuffle_zeros(a, order):
    '''Randomly permute zero indices in a list'''
    inds = np.where(a[order] == 0)[0]
    if len(inds) < 2: return order
    shuffled_inds = inds.copy()
    np.random.shuffle(shuffled_inds)
    order[inds] = order[shuffled_inds]
    return order

def plot_top_k(rand_instance, ann_instance, anntri_instance, Ks, k=1):
    '''Plot top K error'''
    K_dict = {key: val for val, key in enumerate(Ks)}
    rand_errs = rand_instance.average_errors()[:, K_dict[1]]
    ann_errs = ann_instance.average_errors()[:, K_dict[1]]
    anntri_errs = anntri_instance.average_errors()[:, K_dict[1]]

    plt.figure(1)
    plt.semilogx(rand_errs, 'b', label='Rand')
    plt.semilogx(anntri_errs, 'g', label='ANNTri')
    plt.semilogx(ann_errs, 'r', label='ANN')
    plt.xlabel('Samples per point')
    plt.ylabel('Error rate')
    plt.title('Top {} error rates for ANNTri, ANN, Rand'.format(k))
    plt.legend(loc='best')
    plt.show()
