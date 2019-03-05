SEED = 42

import numpy as np
np.random.seed(SEED)

import pandas as pd

from scipy.spatial.distance import pdist

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.utils import resample # nEQB
from sklearn.metrics.pairwise import rbf_kernel # MAODiversity
from sklearn.cluster import KMeans # MAOCluster

from collections import defaultdict

from copy import copy

from tqdm import tqdm

from base.base_active_learner import BaseActiveLearner


# More info: https://stackoverflow.com/questions/37012320


class ActiveLearner(object):
    ''' Superclass that allows comparing different Active Learning algorithms
    '''

    def __init__(self, **kwargs):
        self.test_size = self.lab_size = 0.33

        self.n_queries = 50
        self.n_update_points = 10
        self.algorithms = dict()

        # TODO: rearrange self.estimator in order to add gamma adjusted to data
        # TODO: add scoring function

        self.seed = SEED

        # Update class attributes with keyword arguments (hyperparams)
        self.__dict__.update(kwargs)

        # Create dataframe to store scores for every estimator
        self.scores = pd.DataFrame()


    def _load_data(self, X=None, y=None, data=None):
        ''' Auxiliary method for allowing either the use of whole raw datasets
        or of already splitted ones
        '''
        if X and y:
            # Create train/test in main
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.seed
            )

            # Create lab/unlab split with train
            X_unlab, X_lab, y_unlab, y_lab = train_test_split(
                X_train, y_train, test_size=self.lab_size,
                stratify=y_train, random_state=self.seed
            )

            return X_unlab, X_lab, y_unlab, y_lab, X_test, y_test

        elif data:
            X_lab, y_lab = data['labeled']
            X_unlab, y_unlab = data['unlabeled']
            X_test, y_test = data['test']

            return X_unlab, X_lab, y_unlab, y_lab, X_test, y_test


    def fit(self, **kwargs):
        X_unlab, X_lab, y_unlab, y_lab, X_test, y_test = \
            self._load_data(**kwargs)

        # TODO: rearrange in __init__
        # Update estimator with gamma adjusted to data
        sigma = np.mean(pdist(X_lab))
        gamma = 1 / (2 * sigma * sigma)
        base_estimator = SVC(C=100, gamma=gamma, decision_function_shape='ovr')

        for alg_name, alg_class in self.algorithms.items():
            # Initalize labels in all algorithms (they must start with the
            # same labels) and base estimator
            alg_class = alg_class(base_estimator)
            alg_class.init_labels(X_unlab, X_lab, y_unlab, y_lab)
            self.algorithms[alg_name] = alg_class

            print('[+] Algorithm: {}'.format(alg_name))

            alg_scores = []
            for _ in tqdm(range(self.n_queries)):
                # 1. Fit estimator
                alg_class.estimator.fit(alg_class.X_lab, alg_class.y_lab)

                # 2. Obtain score on X_test, y_test
                alg_scores.append(alg_class.estimator.score(X_test, y_test))

                # 3. Choose samples from X_unlab, y_unlab
                indices = alg_class.get_indices(n_points=self.n_update_points)

                # 4. Update labels, moving them from pool to train
                alg_class.update_labels(indices)

            self.scores[alg_name] = alg_scores



class RandomLearner(BaseActiveLearner):

    def __init__(self, estimator):
        super(RandomLearner, self).__init__()

        self.estimator = estimator


    def get_indices(self, n_points=1):
        indices = np.random.permutation(self.X_unlab.shape[0])
        indices = indices[:n_points]

        return indices



class MarginSampling(BaseActiveLearner):

    def __init__(self, estimator):
        super(MarginSampling, self).__init__()

        self.estimator = estimator


    def get_indices(self, n_points=1):
        dist = np.abs(self.estimator.decision_function(self.X_unlab))

        # Sort inside each sample from lowest to most probable classes
        dist = np.sort(dist, axis=1)

        # Sort along all samples to get the most uncertain samples, this is,
        # samples with lowest distance to hyperplane for its most probable class
        indices = np.argsort(dist[:, -1])

        return indices[:n_points]



class MulticlassUncertainty(BaseActiveLearner):

    def __init__(self, estimator):
        super(MulticlassUncertainty, self).__init__()

        self.estimator = estimator


    def get_indices(self, n_points=1):
        dist = np.abs(self.estimator.decision_function(self.X_unlab))

        # Sort inside each sample from lowest to most probable classes
        dist = np.sort(dist, axis=1)

        # Sort along all samples to get the most uncertain samples, this is,
        # samples with lowest difference between the most probable classes
        indices = np.argsort(dist[:, -1] - dist[:, -2])

        return indices[:n_points]



class SignificanceSpaceConstruction(BaseActiveLearner):

    def __init__(self, estimator):
        super(SignificanceSpaceConstruction, self).__init__()

        self.estimator = estimator

        # Create second estimator, which will classify support vectors against
        # the rest of training samples, yielding samples that are likely to
        # become support vectors
        self.sv_estimator = copy(estimator)


    def get_indices(self, n_points=1):
        X_train = self.X_lab
        y_train = np.zeros(self.y_lab.shape)

        # Update support vectors already found by the classifier
        y_train[self.estimator.support_] = 1

        # If all samples are support vectors, choose randomly
        if len(np.unique(y_train)) == 1:
            indices = np.random.permutation(self.X_unlab.shape[0])

        else:
            self.sv_estimator.fit(X_train, y_train)
            possible_SVs = self.sv_estimator.predict(self.X_unlab)
            indices = np.arange(self.X_unlab.shape[0])[possible_SVs == 1]
            indices = np.random.permutation(indices)

        return indices[:n_points]



class nEQB(BaseActiveLearner):

    def __init__(self, estimator):
        super(nEQB, self).__init__()

        self.estimator = estimator

        # Create second estimator, which will predict labels for the candidates
        self.n_models = 4
        self.cand_estimator = copy(estimator)


    def get_indices(self, n_points=1):
        classes = np.unique(self.y_lab)
        n_classes = len(classes)
        n_unlab = self.X_unlab.shape[0]
        pred_matrix = np.zeros((n_unlab, self.n_models))

        for k in range(self.n_models):
            # Bootstrap replica
            while True:
                X_bag, y_bag = resample(self.X_lab, self.y_lab, replace=True)
                # Ensure that we have all classes in the bootstrap replica
                if len(np.unique(y_bag)) >= n_classes:
                    break

            self.cand_estimator.fit(X_bag, y_bag)
            pred_matrix[:, k] = self.cand_estimator.predict(self.X_unlab)

        # Count number of votes per class
        ct = np.zeros((self.X_unlab.shape[0], n_classes))
        for i, w in enumerate(classes):
            ct[:, i] = np.sum(pred_matrix == w, axis=1)

        # Divide ct by the number of models to obtain estimated probabilities
        ct /= self.n_models

        Hbag = ct.copy()
        # Set to 1 where Hbag == 0 to avoid -Inf and NaNs (0 * -Inf = NaN)
        Hbag[Hbag == 0] = 1
        Hbag = -np.sum(ct * np.log(Hbag), axis=1)

        logNi = np.log(np.sum(ct > 0, axis=1))
        # Avoid division by zero
        logNi[logNi == 0] = 1

        # Normalize entropy by the logarithm of number of classes
        nEQB = Hbag / logNi

        # Select randomly one element among the ones with maximum entropy
        indices = np.where(nEQB == np.max(nEQB))[0]
        np.random.shuffle(indices)

        return indices[:n_points]



class MAOBase(BaseActiveLearner):

    def __init__(self):
        super(MAOBase, self).__init__()


    def get_indices(self, n_points=1):
        dist = np.abs(self.estimator.decision_function(self.X_unlab))

        # Sort inside each sample from lowest to most probable classes
        dist = np.sort(dist, axis=1)

        # Sort along all samples to get the most uncertain samples, this is,
        # samples with lowest distance to hyperplane for its most probable class
        indices = np.argsort(dist[:, -1])

        # We cannot limit the number of points in 'indices' to query_points,
        # but we cannot use all points either, otherwise appart from the first
        # point, the rest will be selected using only the diverse criterion,
        # and not the uncertainty criterion

        # Lets limit the pool of possible samples as:
        indices = indices[0:n_points * 10]


        # Measure distances using the kernel function
        K = rbf_kernel(self.X_unlab, gamma=self.estimator.get_params()['gamma'])


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



class MAOCluster(BaseActiveLearner):

    def __init__(self, estimator):
        super(MAOCluster, self).__init__()

        self.estimator = estimator


    def get_indices(self, n_points=1):
        dist = np.abs(self.estimator.decision_function(self.X_unlab))

        # Sort inside each sample from lowest to most probable classes
        dist = np.sort(dist, axis=1)

        # Sort along all samples to get the most uncertain samples, this is,
        # samples with lowest distance to hyperplane for its most probable class
        indices = np.argsort(dist[:, -1])
        # We can limit the pool of candidates when clustering to relax the
        # computational burden, but in principle using all unlabeled
        # samples should obtain a better partition
        # idx = idx[0:3*queried_points]

        # Cluster the unlabeled set
        kmeans = KMeans(n_clusters=n_points)
        cluster_IDs = kmeans.fit_predict(self.X_unlab)

        s_indices = np.zeros(n_points, dtype=np.int64)
        for i in range(n_points):
            # Select one sample per cluster. More specifically, select the
            # first point in 'indices', already sorted according the MS
            # uncertainty criterion
            s_indices[i] = indices[cluster_IDs == i][0]

        return s_indices
