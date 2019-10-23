from __future__ import division, print_function
import numpy as np 
from utils import *

class ANNTri:
    def __init__(self, dist_func, embedding, epsilon, sigma, delta, triangle, Ks, 
                                                                    maxPulls=10**5,
                                                                    random_sample=False):

        '''
        Implementation of ANNTri Algorithm from "Learning Nearest Neighbor Graphs
        with noise" by Mason, et al., 2019. The algorithm assumes access to a 
        callable distance oracle. Taking in the embedding is for convenience for 
        error computation purposes an is not used in learning the NN Graph directly. 
        By toggling the "triangle" and "random_sample" flags, this code may also be
        used to run the Random and ANN algorithms in the paper. 

        Parameters:
        dist_func: callable function that takes a distance matrix D, i,j indices
                    and a noise standard deviation and returns D[i,j] + noise
        embedding: nd array used to create a Euclidean distance matrix. For metrics 
                    other than Euclidean, code can be modified to take in a distance 
                    matrix and true_nns can be computed from that instead.
        epsilon: float, parameter for law of iterated log bound
        delta: failure probability, parameter for law of iterated log bound
        sigma: standard deviation of desired noise level
        triangle: boolean, whether or not to use the triangle inequality. 
                    If not randomly sampling, triangle=False produces ANN algorithm
        Ks: list[int] values of k for top K error, eg, is true NN in k closest points
            at a given time
        maxPulls: int - Maximum number of samples to give a single point, before
                        halting. 
        random_sample: bool, whether to sample randomly or actively. random_sample
                        = false produces RANDOM algorithm in paper
        '''

        # parameters
        self.d_oracle = dist_func 
        self.embedding = embedding
        self.true_D = compute_true_D(self.embedding)
        self.true_nns = get_true_nns(self.embedding)
        self.epsilon = epsilon
        self.sigma = sigma
        self.delta = delta
        self.triangle = triangle        # bool, whether to use triangle inequality
        self.Ks = Ks        # list of top-k errors to compute
        self.num_pts = self.true_D.shape[0]
        self.maxPulls = maxPulls
        self.rand_samp = random_sample

        # main experiment variables
        init_shape = (self.num_pts, self.num_pts)
        self.counts = np.zeros(init_shape)
        self.emp_means = zeros_infDiag(init_shape)
        self.ub = np.inf * np.ones(init_shape)
        self.lb = zeros_infDiag(init_shape)
        self.tri_ub = np.inf * np.ones(init_shape)
        self.tri_lb = -np.inf * np.ones(init_shape)
        self.errors = [[] for _ in range(self.num_pts)]
        self.pull_counts = np.zeros(self.num_pts, dtype=int)

    def finite_LIL_bd(self, pulls):
        '''Finite Law of the iterated logarithm bound
            from Lil'UCB, by Jameison et al.'''
        return ((1 + np.sqrt(self.epsilon)) * self.sigma * 
                    np.sqrt(2 * (1+self.epsilon) / pulls \
                                * np.log(np.log(2 + (1+self.epsilon) *\
                                 pulls) / (self.delta / self.num_pts))))

    def play_arms(self, ref, i):
        '''In round ref, pull arm i and update bounds ''' 
        prev_val_sum = self.emp_means[ref, i] * self.counts[ref, i]
        # query oracle for distance from p
        current_play = self.d_oracle(self.true_D, ref, i, self.sigma)
        self.counts[ref, i] += 1
        self.emp_means[ref, i] = (prev_val_sum + current_play) / self.counts[ref, i]
        lil_shift = self.finite_LIL_bd(self.counts[ref, i])
        self.ub[ref, i] = self.emp_means[ref, i] + lil_shift
        self.lb[ref, i] = self.emp_means[ref, i] - lil_shift
        self.pull_counts[ref] += 1
        self.errors[ref].append(self.errors_at_ks(ref))

    def errors_at_ks(self, ref):
        '''Compute top k error for different values of k in round ref'''
        closest_pts = np.argsort(self.emp_means[ref, :])
        closest_pts = shuffle_zeros(self.emp_means[ref, :], closest_pts)
        active = set(self.active_for_nn_of_ref(ref))
        valid = [i for i, pt_idx in enumerate(closest_pts) if pt_idx in active]
        closest_pts = closest_pts[valid]
        return np.array([self.true_nns[ref] not in closest_pts[:k] for k in self.Ks])

    def symmetrize(self, ref):
        '''Symmetrize matrices after round ref to propagate information'''
        self.counts[:, ref] = self.counts[ref, :]
        self.emp_means[:, ref] = self.emp_means[ref, :]
        self.ub[:, ref] = self.ub[ref, :]
        self.lb[:, ref] = self.lb[ref, :]
        self.tri_ub[:, ref] = self.tri_ub[ref, :]
        self.tri_lb[:, ref] = self.tri_lb[ref, :]

    def active_for_nn_of_ref(self, ref):
        ''' Finds all arms that are not eliminated from being NN(ref) yet.
        '''
        if self.triangle and not self.rand_samp: 
            ub_ref = np.minimum(self.ub[ref, :], self.tri_ub[ref, :])
            lb_ref = np.maximum(self.lb[ref, :], self.tri_lb[ref, :])
        else:
            ub_ref, lb_ref = self.ub[ref, :], self.lb[ref, :]
        if ref == 0: 
            min_ub = np.min(ub_ref[ref:])
        elif ref == self.num_pts - 1: 
            min_ub = np.min(ub_ref[:ref])
        else: 
            min_ub = min(np.min(ub_ref[:ref]), np.min(ub_ref[ref+1:]))  
        indexed_before = np.where(lb_ref[:ref] <= min_ub)[0]
        indexed_after = ref + 1 + np.where(lb_ref[ref+1:] <= min_ub)[0]
        return np.concatenate((indexed_before, indexed_after))

    def least_sampled_arms(self, ref, actives):
        ''' Pull arms that have been least sampled first.
            See line 3 of Pseudocode for SETri in paper.
            actives is the current active set. 
        '''
        min_samples = np.min(self.counts[ref, actives])
        return [a for a in actives if self.counts[ref, a] == min_samples]

    def tri_ub_update(self, ref, prev, arm):
        '''Update triangle upper bounds for D[ref, arm] via
            triangle with pt. prev '''
        candidate = min(self.ub[prev, arm], self.tri_ub[prev, arm]) \
                + min(self.ub[prev, ref], self.tri_ub[prev, ref])
        self.tri_ub[ref, arm] = min(candidate, self.tri_ub[ref, arm])

    def tri_lb_update(self, ref, prev, arm):
        '''Update triangle lower bounds for D[ref, arm] via
            triangle with pt. prev '''
        lb_ref = max(self.lb[prev, ref], self.tri_lb[prev, ref])
        lb_arm = max(self.lb[prev, arm], self.tri_lb[prev, arm])
        ub_ref = min(self.ub[prev, ref], self.tri_ub[prev, ref])
        ub_arm = min(self.ub[prev, arm], self.tri_ub[prev, arm])
        candidate = max(lb_ref, lb_arm) - min(ub_ref, ub_arm)
        self.tri_lb[ref, arm] = max(candidate, self.tri_lb[ref, arm])

    def full_update_triangles(self, ref):
        '''Update all triangle bounds'''
        for r in range(ref + 1):
            # update triangle for ref_pt r
            for prev in range(ref):    # previous refs
                for arm in range(self.num_pts):
                    if arm != prev and arm != r and prev != r:
                        self.tri_ub_update(r, prev, arm)
                        self.tri_lb_update(r, prev, arm)
            self.symmetrize(r)

    def SETri(self, ref):
        '''Implementation of SETri from paper. If not self.triangle,
            this implements ANN in the paper and ignores triangle bounds'''
        possible_nns = self.active_for_nn_of_ref(ref)   # initial active set
        while len(possible_nns) > 1 and self.pull_counts[ref] < self.maxPulls:    
            # pull arms with fewest samples
            possible_nns_min_samples = self.least_sampled_arms(ref, possible_nns)
            # pull all possible nns
            for arm in possible_nns_min_samples:
                if self.pull_counts[ref] >= self.maxPulls: break
                self.play_arms(ref, arm)
            # check if arms have been eliminated
            possible_nns = self.active_for_nn_of_ref(ref)
        if self.triangle: print('ANNTri: Found nn for arm {} in {} pulls'\
                                .format(ref+1, self.pull_counts[ref]))
        else: print('ANN: Found nn for arm {} in {} pulls'\
                        .format(ref+1, self.pull_counts[ref]))
        self.errors[ref].append(self.errors_at_ks(ref))

    def choose_random_arm(self, ref):
        '''Choose an arm uniformly at random'''
        valid_arms = [i for i in range(self.num_pts) if i != ref]
        return np.random.choice(valid_arms, size=1, replace=False)

    def passive(self, ref):
        '''Find a NN by passive sampling. Referred to
            as  'RANDOM' in paper.'''
        while len(self.active_for_nn_of_ref(ref)) > 1 \
                and self.pull_counts[ref] < self.maxPulls:
            self.play_arms(ref, self.choose_random_arm(ref))
        print('Random: Found nn for arm {}, in {} pulls'\
                    .format(ref+1, self.pull_counts[ref]))
        self.errors[ref].append(self.errors_at_ks(ref))

    def run(self):
        '''Single implementation of ANNTri, ANN, and RANDOM,
            based on set flags, self.random_sample and self.triangle'''
        for ref in range(self.num_pts):
            if self.triangle and not self.rand_samp: 
                self.full_update_triangles(ref)
            if self.rand_samp: self.passive(ref)
            else: self.SETri(ref)
            self.errors[ref] = np.array(self.errors[ref])   
            self.symmetrize(ref)

    def combine_errors(self):
        '''Put all error lists (of different lengths into single array
            of max length needed'''
        max_snaps = max([self.errors[i].shape[0] for i in range(self.num_pts)])
        errors_out = np.zeros((self.num_pts, max_snaps, len(self.Ks)))
        for i in range(self.num_pts):
            num_snaps = self.errors[i].shape[0]
            errors_out[i, :num_snaps, :] = self.errors[i]
        return errors_out

    def save(self, filename):
        ''' Save errors'''
        np.save(filename, self.combine_errors())

    def average_errors(self):
        '''Compute average error'''
        return np.mean(self.combine_errors(), axis=0)

    def pulls_to_reach_stopping(self):
        '''Number of pulls to reach stopping for each point'''
        return self.pull_counts

    def pulls_per_arm(self):
        '''Total number of pulls for each arm'''
        return np.sum(self.counts, axis=0)

if __name__ == '__main__':
    # simple test of the code
    from time import time
    from data_gen import generate_spherical_clusters as generate  
    dist_func = d_oracle
    clusters = 5
    points_per_cluster = 5
    separation_factor = 2
    np.random.seed(40)
    embedding = generate(clusters, points_per_cluster, 
                                    separation_factor=separation_factor)
    epsilon = 0.7
    sigma = 0.1
    delta = 0.1
    triangle = True
    Ks = [1, 3, 5, 10]
    # instantiate
    instance = ANNTri(dist_func, embedding, epsilon, sigma, delta, triangle, 
                                                                Ks, 
                                                                maxPulls=10**5,
                                                                random_sample=False)
    # learn all nns
    ts = time()
    instance.run()
    print('ran in {} seconds'.format(time() - ts))
    # average error
    avg_errs = instance.average_errors()
