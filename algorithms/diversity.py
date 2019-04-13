SEED = 42

import numpy as np
np.random.seed(SEED)

from sklearn.metrics.pairwise import rbf_kernel # MAODiversity, MAOLambda
from sklearn.cluster import KMeans # MAOCluster

from base.base_active_learner import BaseActiveLearner


class MAOBase():

    def __init__(self):
        super(MAOBase, self).__init__()


    def get_indices(self, X_unlab, indices, dist, n_points=1):
        # We cannot limit the number of points in 'indices' to query_points,
        # but we cannot use all points either, otherwise appart from the first
        # point, the rest will be selected using only the diverse criterion,
        # and not the uncertainty criterion

        # Lets limit the pool of possible samples as:
        indices = indices[0:n_points * 10]

        # print('\n', X_unlab.shape[0], indices.shape[0])

        # Measure distances using the kernel function
        K = rbf_kernel(X_unlab, gamma=self.estimator.get_params()['gamma'])

        s_indices = np.zeros(n_points, dtype=np.int64)
        for i in range(n_points):
            # Add the first point (and remove it from pool)
            s_indices[i] = indices[0]
            indices = indices[1:]

            # Compute distances (kernel matrix)
            # Distances between selected samples (Sidx) and the rest (idx)
            Kdist = np.abs(K[s_indices[0:i+1], :][:, indices])

            # Obtain the minimum distance for each column
            Kdist = Kdist.min(axis=0)

            # Choose method to obtain indices
            if self.method == 'diversity':
                # Re-order by distance
                indices = indices[Kdist.argsort(axis=0)]

            elif self.method == 'lambda':
                # Trade-off between MS and Diversitylambda
                lambd = 0.6
                heuristic = dist[indices, -1] * lambd + Kdist * (1 - lambd)
                indices = indices[heuristic.argsort()]

        return s_indices



class MAODiversity(MAOBase):

    def __init__(self, estimator):
        super(MAODiversity, self).__init__()

        self.estimator = estimator
        self.method = 'diversity'



class MAOLambda(MAOBase):

    def __init__(self, estimator):
        super(MAOLambda, self).__init__()

        self.estimator = estimator
        self.method = 'lambda'



class MAOCluster():

    def __init__(self, estimator):
        super(MAOCluster, self).__init__()

        # Override errors while using all three methods together
        # (estimator is not used in this method)
        self.estimator = estimator


    def get_indices(self, X_unlab, indices, dist, n_points=1):
        # Cluster the unlabeled set
        kmeans = KMeans(n_clusters=n_points)

        # Select X_unlab based on indices in order to avoid dimension errors.
        # E.g. get_indices() for SSC or nEQB does not return the whole array
        # of original points ordered, just a part of it. Therefore, cluster_IDs
        # is going to have a higher dimension than indices
        cluster_IDs = kmeans.fit_predict(X_unlab[indices, :])

        s_indices = np.zeros(n_points, dtype=np.int64)
        for i in range(n_points):
            # Select one sample per cluster. More specifically, select the
            # first point in 'indices', already sorted according the MS
            # uncertainty criterion
            s_indices[i] = indices[cluster_IDs == i][0]

        return s_indices
