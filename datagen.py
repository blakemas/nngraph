from __future__ import division, print_function
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets.samples_generator import make_blobs
from utils import compute_true_D

### Code to make several random Euclidean emebddings  

def point_in_unit_sphere():
    '''Generate random point in unit sphere in R^2'''
    r = np.random.rand()
    theta = 2*np.pi*np.random.rand()
    return [r*np.cos(theta), r*np.sin(theta)]

def point_in_unit_d_sphere(d):
    '''
    Code to generate point uniformly in unit d sphere.
    https://math.stackexchange.com/questions/87230/
        picking-random-points-in-the-volume-of-sphere-with
        -uniform-probability
    '''
    x = np.random.randn(d)
    radius = np.random.rand() ** (1/d)
    return x*radius/np.linalg.norm(x)

def k_points_in_unit_sphere(k):
    return np.array([point_in_unit_sphere() for _ in range(k)])

def k_points_in_unit_d_sphere(k, d):
    return np.array([point_in_unit_d_sphere(d) for _ in range(k)])

def generate_spherical_clusters(K, ppc, separation_factor=2):
    '''
    Generates K unit sphere clusters with ppc points
    per cluster. Each cluster is a unit sphere distributed 
    one a shell. Scale to be in unit square
    '''
    X = np.array([0, 0])     # first row is a placeholder
    r = (separation_factor + 1)*2/np.sqrt(2 - 2*np.cos(2*np.pi/K))    # radius of shell    # radius of shell
    for k in range(K):
        theta = 2*np.pi*k/K
        center = [r*np.cos(theta), r*np.sin(theta)]
        cluster = k_points_in_unit_sphere(ppc) + center
        X = np.vstack((X, cluster))
    X = X[1:, :]   # remove placeholder
    return(X)

def generate_high_dim_spherical_clusters(K, ppc, separation_factor=2):
    '''
    Generates K unit sphere clusters in K dimensions with ppc points
    per cluster. Each cluster is a unit sphere distributed 
    one a shell. Scale to be in unit square
    '''
    X = np.zeros(K)     # first row is a placeholder
    centers = 2*(1 + separation_factor)/np.sqrt(2) * np.eye(K)
    for k in range(K):
        cluster = k_points_in_unit_d_sphere(ppc, K) + centers[k]
        X = np.vstack((X, cluster))
    X = X[1:, :]   # remove placeholder
    return(X)

def generate_clustered_data(num_pts, dim, num_clusters, cluster_std=0.6):
    X, y = make_blobs(n_samples=num_pts, n_features=dim, centers=num_clusters, 
                                                         cluster_std=cluster_std)
    return X

def generate_1d_cluster(D_range, num_per_cluster, num_clusters):
    '''Clusters in 1 dimension'''
    X = np.array([])
    for i in range(num_clusters):
        new_cluster = generate_data(D_range, num_per_cluster, 1) + 2*i
        X = np.concatenate([X, np.squeeze(new_cluster)])
    X = X.reshape((X.shape[0], 1))
    return X  

def generate_uniform_data(D_range, num, dim, gaussian=False):
    '''Generate data uniformly in d cube'''
    if not gaussian: return D_range * np.random.random_sample((num, dim))    # Dataset
    else: return np.random.randn(num, dim)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # spheres 2D
    X = generate_spherical_clusters(10, 10, separation_factor=0.1)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()